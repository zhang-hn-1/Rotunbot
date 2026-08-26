"""Rotunbot point-to-point obstacle navigation with a forward depth camera."""

import math
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import torch_rand_float

from legged_gym.maps import (
    build_maze,
    cell_centers_to_world,
    reachable_free_cells,
    wall_cells,
)
from .rotunbot_target_obstacle import RotunbotTargetObstacle
from .rotunbot_target_repro import RotunbotTargetRepro
from .rotunbot_target_depth_config import RotunbotTargetDepthCfg


class RotunbotTargetDepth(RotunbotTargetRepro):
    """Target reproduction upgraded to sensor-driven obstacle avoidance.

    The old target task remains unchanged.  This task reuses its target,
    controller, curriculum, and evaluation logic, while adding maze collision
    geometry, randomized valid road-center goals, and a front-facing depth
    sensor.  On Isaac Gym runs without a
    graphics device it uses a deterministic ray/AABB depth model so headless
    training remains possible.
    """

    cfg: RotunbotTargetDepthCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self._camera_handles = []
        self._camera_depth_tensors = []
        self._camera_ready = False
        self._camera_error_reported = False
        # Optional diagnostic mode: compute the deterministic maze fallback
        # alongside the real Isaac Gym camera at the same simulator state.
        # It is disabled by default and does not change normal training.
        self.capture_depth_comparison = False
        # The task config selects the training sensor source.  The evaluator
        # can override this after construction for controlled comparisons.
        self.depth_policy_source = str(
            getattr(getattr(cfg, "camera", None), "policy_source", "fallback")
        )
        print(
            "[RotunbotTargetDepth] policy depth source: "
            f"{self.depth_policy_source}; "
            "headless camera sensors: "
            f"{bool(getattr(cfg, 'enable_camera_sensors_in_headless', False))}"
        )
        self._maze_enabled = bool(getattr(getattr(cfg, "maze", None), "enabled", False))
        self.maze_layout = None
        self._maze_wall_centers_cpu = None
        self._maze_wall_sizes_cpu = None
        self._maze_geodesic_distance_cpu = None
        self._maze_next_cells_cpu = None
        self._maze_geodesic_distance = None
        self._maze_start_position = np.zeros(2, dtype=np.float32)
        self._maze_goal_position = np.zeros(2, dtype=np.float32)
        self._maze_goal_cells = None
        self._maze_goal_positions = None
        self._maze_goal_segments_cpu = None
        self._maze_goal_segment_endpoints_cpu = None
        self._maze_goal_segment_bins_cpu = None
        self._maze_goal_segment_valid_intervals_cpu = None
        self._maze_goal_segment_valid_lengths_cpu = None
        self._maze_goal_segment_orders_cpu = None
        self._maze_goal_segment_pointers = None
        self._maze_goal_segment_bin_cursor = 0
        # ``train.py`` offsets cfg.seed per DDP rank.  Use that seed only for
        # target sampling so every rank keeps the same maze but explores a
        # different sequence of continuous goals.
        goal_seed = int(getattr(cfg, "seed", cfg.maze.seed))
        if goal_seed < 0:
            goal_seed = int(cfg.maze.seed)
        self._maze_goal_rng = np.random.default_rng(goal_seed + 1)

        if self._maze_enabled:
            self._initialize_maze_geometry(cfg)
            # Offset the first near/mid/far choice by seed.  This matters when
            # an evaluator creates a fresh simulator for every episode;
            # otherwise every fresh instance would always begin with the near
            # bin even though the persistent training environment is balanced.
            self._maze_goal_segment_bin_cursor = goal_seed % len(
                self._maze_goal_segment_bins_cpu
            )
            self._obstacle_centers_cpu = np.broadcast_to(
                self._maze_wall_centers_cpu,
                (int(cfg.env.num_envs),) + self._maze_wall_centers_cpu.shape,
            ).copy()
        else:
            self._obstacle_centers_cpu = self._sample_obstacle_layouts(cfg)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _initialize_maze_geometry(self, cfg):
        self.maze_layout = build_maze(
            grid_size=cfg.maze.grid_size,
            seed=cfg.maze.seed,
            center_clearance_radius=cfg.maze.center_clearance_radius,
        )

        start_cell = tuple(int(value) for value in cfg.maze.start_cell)
        reachable = reachable_free_cells(self.maze_layout, start=start_cell)
        self._maze_start_position = cell_centers_to_world(
            np.asarray([start_cell]),
            self.maze_layout.shape,
            cfg.maze.cell_size,
        )[0].astype(np.float32)
        minimum_distance = float(getattr(cfg.maze, "min_goal_distance", 8.0))

        # Every free cell is a road-center vertex.  Adjacent free-cell centers
        # form continuous centerline segments, so targets are not restricted
        # to integer/grid coordinates.
        self._maze_goal_cells = np.asarray(reachable, dtype=np.int64)
        self._maze_goal_positions = cell_centers_to_world(
            self._maze_goal_cells,
            self.maze_layout.shape,
            cfg.maze.cell_size,
        ).astype(np.float32)
        self._maze_goal_position = self._maze_goal_positions[0].copy()
        cell_lookup = np.full(self.maze_layout.shape, -1, dtype=np.int64)
        for index, cell in enumerate(self._maze_goal_cells.tolist()):
            cell_lookup[tuple(cell)] = index

        segments = []
        segment_endpoints = []
        segment_distance = []
        for index, cell in enumerate(self._maze_goal_cells.tolist()):
            x, y = int(cell[0]), int(cell[1])
            for dx, dy in ((1, 0), (0, 1)):
                neighbor = (x + dx, y + dy)
                if not (
                    0 <= neighbor[0] < self.maze_layout.shape[0]
                    and 0 <= neighbor[1] < self.maze_layout.shape[1]
                ):
                    continue
                neighbor_index = int(cell_lookup[neighbor])
                if neighbor_index < 0:
                    continue
                p0 = self._maze_goal_positions[index]
                p1 = self._maze_goal_positions[neighbor_index]
                max_distance = max(
                    float(np.linalg.norm(p0 - self._maze_start_position)),
                    float(np.linalg.norm(p1 - self._maze_start_position)),
                )
                if max_distance < minimum_distance:
                    continue
                segments.append((p0, p1))
                segment_endpoints.append((index, neighbor_index))
                segment_distance.append(max_distance)

        if not segments:
            raise ValueError(
                "maze has no reachable road-centerline segment with "
                f"max start distance >= {minimum_distance} m"
            )
        self._maze_goal_segments_cpu = np.asarray(segments, dtype=np.float32)
        self._maze_goal_segment_endpoints_cpu = np.asarray(
            segment_endpoints, dtype=np.int64
        )
        # For every segment, precompute the part whose Euclidean distance from
        # the fixed start is at least the configured minimum.  Sampling from
        # these intervals is exact; rejection sampling could over-sample the
        # far end of a segment and would make the coverage less uniform.
        valid_intervals = []
        valid_lengths = []
        start_position = self._maze_start_position.astype(np.float64)
        for segment in self._maze_goal_segments_cpu.astype(np.float64):
            p0, p1 = segment
            direction = p1 - p0
            offset = p0 - start_position
            quadratic_a = float(np.dot(direction, direction))
            quadratic_b = float(2.0 * np.dot(offset, direction))
            quadratic_c = float(
                np.dot(offset, offset) - minimum_distance * minimum_distance
            )
            intervals = []
            if quadratic_a <= 1.0e-12:
                if quadratic_c >= 0.0:
                    intervals.append((0.0, 1.0))
            else:
                discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
                if discriminant <= 0.0:
                    if quadratic_c >= 0.0:
                        intervals.append((0.0, 1.0))
                else:
                    root = math.sqrt(discriminant)
                    root_a = (-quadratic_b - root) / (2.0 * quadratic_a)
                    root_b = (-quadratic_b + root) / (2.0 * quadratic_a)
                    if root_a > 0.0:
                        intervals.append((0.0, min(1.0, root_a)))
                    if root_b < 1.0:
                        intervals.append((max(0.0, root_b), 1.0))
            intervals = [
                (max(0.0, float(left)), min(1.0, float(right)))
                for left, right in intervals
                if right - left > 1.0e-7
            ]
            while len(intervals) < 2:
                intervals.append((0.0, 0.0))
            valid_intervals.append(intervals[:2])
            valid_lengths.append(
                [max(0.0, right - left) for left, right in intervals[:2]]
            )
        self._maze_goal_segment_valid_intervals_cpu = np.asarray(
            valid_intervals, dtype=np.float32
        )
        self._maze_goal_segment_valid_lengths_cpu = np.asarray(
            valid_lengths, dtype=np.float32
        )
        segment_distance = np.asarray(segment_distance, dtype=np.float32)
        valid_segment_mask = (
            self._maze_goal_segment_valid_lengths_cpu.sum(axis=1) > 1.0e-7
        )
        if not np.all(valid_segment_mask):
            self._maze_goal_segments_cpu = self._maze_goal_segments_cpu[
                valid_segment_mask
            ]
            self._maze_goal_segment_endpoints_cpu = (
                self._maze_goal_segment_endpoints_cpu[valid_segment_mask]
            )
            self._maze_goal_segment_valid_intervals_cpu = (
                self._maze_goal_segment_valid_intervals_cpu[valid_segment_mask]
            )
            self._maze_goal_segment_valid_lengths_cpu = (
                self._maze_goal_segment_valid_lengths_cpu[valid_segment_mask]
            )
            segment_distance = segment_distance[valid_segment_mask]
        if len(self._maze_goal_segments_cpu) == 0:
            raise ValueError(
                "maze has no positive-length road-centerline interval with "
                f"start distance >= {minimum_distance} m"
            )
        order = np.argsort(segment_distance)
        self._maze_goal_segment_bins_cpu = [
            chunk for chunk in np.array_split(order, 3) if len(chunk) > 0
        ]
        # Shuffle within each distance bin and consume each segment before it
        # can be selected again.  The bin cursor cycles near/mid/far, so a
        # batch cannot accidentally contain only short or only long goals.
        self._maze_goal_segment_orders_cpu = []
        for segment_bin in self._maze_goal_segment_bins_cpu:
            shuffled = np.asarray(segment_bin, dtype=np.int64).copy()
            self._maze_goal_rng.shuffle(shuffled)
            self._maze_goal_segment_orders_cpu.append(shuffled)
        self._maze_goal_segment_pointers = np.zeros(
            len(self._maze_goal_segment_orders_cpu), dtype=np.int64
        )
        self._maze_goal_segment_bin_cursor = 0
        wall_indices = wall_cells(self.maze_layout)
        self._maze_wall_centers_cpu = cell_centers_to_world(
            wall_indices,
            self.maze_layout.shape,
            cfg.maze.cell_size,
        ).astype(np.float32)
        self._maze_wall_sizes_cpu = np.full(
            (len(wall_indices), 2),
            float(cfg.maze.cell_size),
            dtype=np.float32,
        )

        # Store one geodesic/next-cell map for every reachable road-center
        # vertex.  A continuous target on a segment uses the two endpoint maps.
        geodesic_maps = []
        next_cell_maps = []
        for goal_cell_values in self._maze_goal_cells.tolist():
            goal_distance = self._compute_geodesic_map(
                self.maze_layout, tuple(goal_cell_values)
            )
            geodesic_maps.append(goal_distance * float(cfg.maze.cell_size))
            goal_next_cells = np.full(
                self.maze_layout.shape + (2,), -1, dtype=np.int64
            )
            for x in range(self.maze_layout.shape[0]):
                for y in range(self.maze_layout.shape[1]):
                    if not np.isfinite(goal_distance[x, y]):
                        continue
                    next_cell = (x, y)
                    best_distance = goal_distance[x, y]
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        neighbor = (x + dx, y + dy)
                        if not (
                            0 <= neighbor[0] < self.maze_layout.shape[0]
                            and 0 <= neighbor[1] < self.maze_layout.shape[1]
                        ):
                            continue
                        if goal_distance[neighbor] < best_distance:
                            best_distance = goal_distance[neighbor]
                            next_cell = neighbor
                    goal_next_cells[x, y] = next_cell
            next_cell_maps.append(goal_next_cells)
        self._maze_geodesic_distance_cpu = np.stack(geodesic_maps, axis=0)
        self._maze_next_cells_cpu = np.stack(next_cell_maps, axis=0)

    @staticmethod
    def _compute_geodesic_map(maze, source):
        """Return four-neighbor cell distance from ``source`` in cells."""
        maze = np.asarray(maze)
        distance = np.full(maze.shape, np.inf, dtype=np.float32)
        source = (int(source[0]), int(source[1]))
        distance[source] = 0.0
        queue = deque([source])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (x + dx, y + dy)
                if not (
                    0 <= neighbor[0] < maze.shape[0]
                    and 0 <= neighbor[1] < maze.shape[1]
                ):
                    continue
                if maze[neighbor] != 0 or np.isfinite(distance[neighbor]):
                    continue
                distance[neighbor] = distance[x, y] + 1.0
                queue.append(neighbor)
        return distance

    @staticmethod
    def _sample_obstacle_layouts(cfg):
        """Create reproducible per-environment obstacle jitter."""
        nominal = np.asarray(cfg.obstacles.centers, dtype=np.float32)
        num_envs = int(cfg.env.num_envs)
        seed = int(getattr(cfg, "seed", 0)) + 17
        rng = np.random.default_rng(seed)
        layouts = np.broadcast_to(nominal, (num_envs,) + nominal.shape).copy()
        layouts += rng.uniform(-0.35, 0.35, size=layouts.shape).astype(np.float32)
        return layouts

    # ------------------------------------------------------------------ scene
    def _create_scene_assets(self):
        if self._maze_enabled:
            wall_options = gymapi.AssetOptions()
            wall_options.fix_base_link = True
            wall_asset = self.gym.create_box(
                self.sim,
                float(self.cfg.maze.cell_size),
                float(self.cfg.maze.cell_size),
                float(self.cfg.maze.wall_height),
                wall_options,
            )
            return {
                "wall_asset": wall_asset,
                "wall_color": gymapi.Vec3(*self.cfg.maze.wall_color),
            }

        obstacle_options = gymapi.AssetOptions()
        obstacle_options.fix_base_link = True
        obstacle_assets = []
        for width, depth, height in self.cfg.obstacles.sizes:
            obstacle_assets.append(
                self.gym.create_box(
                    self.sim,
                    float(width),
                    float(depth),
                    float(height),
                    obstacle_options,
                )
            )
        return {"obstacle_assets": obstacle_assets}

    def _create_scene_actors(self, env_handle, env_id, scene_assets):
        if self._maze_enabled:
            origin = self.env_origins[env_id]
            maze_shape = np.asarray(self.maze_layout.shape, dtype=np.float64)
            cell_size = float(self.cfg.maze.cell_size)
            wall_height = float(self.cfg.maze.wall_height)
            for x, y in wall_cells(self.maze_layout):
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(
                    float(origin[0].item() + (x - maze_shape[0] / 2.0 + 0.5) * cell_size),
                    float(origin[1].item() + (y - maze_shape[1] / 2.0 + 0.5) * cell_size),
                    float(origin[2].item() + wall_height / 2.0),
                )
                pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                actor = self.gym.create_actor(
                    env_handle,
                    scene_assets["wall_asset"],
                    pose,
                    f"maze_wall_{int(x)}_{int(y)}",
                    env_id,
                    0,
                    0,
                )
                self.gym.set_rigid_body_color(
                    env_handle,
                    actor,
                    0,
                    gymapi.MESH_VISUAL,
                    scene_assets["wall_color"],
                )
            return

        origin = self.env_origins[env_id]
        colors = self.cfg.obstacles.colors
        for obstacle_id, (center, asset) in enumerate(
            zip(self._obstacle_centers_cpu[env_id], scene_assets["obstacle_assets"])
        ):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(
                float(origin[0].item() + center[0]),
                float(origin[1].item() + center[1]),
                float(self.cfg.obstacles.sizes[obstacle_id][2]) / 2.0,
            )
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            actor = self.gym.create_actor(
                env_handle,
                asset,
                pose,
                f"nav_obstacle_{obstacle_id}",
                env_id,
                2,
                0,
            )
            if obstacle_id < len(colors):
                self.gym.set_rigid_body_color(
                    env_handle,
                    actor,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(*colors[obstacle_id]),
                )

    def _create_envs(self):
        # The obstacle task already has the correct robot + scene-actor setup.
        # Calling the method unbound lets this class keep the reproduction MRO
        # for all target and reward behavior while supplying its own scene hooks.
        RotunbotTargetObstacle._create_envs(self)
        self._create_camera_sensors()

    def _create_camera_sensors(self):
        camera_cfg = self.cfg.camera
        self._camera_handles = []
        if not bool(camera_cfg.enable):
            return
        if self.graphics_device_id < 0:
            return

        try:
            camera_props = gymapi.CameraProperties()
            camera_props.width = int(camera_cfg.width)
            camera_props.height = int(camera_cfg.height)
            camera_props.horizontal_fov = float(camera_cfg.horizontal_fov)
            camera_props.near_plane = float(camera_cfg.near_plane)
            camera_props.far_plane = float(camera_cfg.far_plane)
            camera_props.enable_tensors = True

            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(*[float(v) for v in camera_cfg.position])
            local_transform.r = gymapi.Quat(
                *[float(v) for v in camera_cfg.rotation]
            )

            base_body_name = "base_link"
            for env_handle, actor_handle in zip(self.envs, self.actor_handles):
                camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                body_handle = self.gym.find_actor_rigid_body_handle(
                    env_handle, actor_handle, base_body_name
                )
                self.gym.attach_camera_to_body(
                    camera_handle,
                    env_handle,
                    body_handle,
                    local_transform,
                    gymapi.FOLLOW_TRANSFORM,
                )
                self._camera_handles.append(camera_handle)
        except Exception as exc:  # pragma: no cover - depends on Isaac Gym build
            self._camera_handles = []
            if not self._camera_error_reported:
                print(
                    "[RotunbotTargetDepth] camera tensors unavailable; "
                    f"using headless depth fallback ({exc})"
                )
                self._camera_error_reported = True

    # --------------------------------------------------------------- state IO
    def _select_robot_root_states(self, all_root_states):
        """Use actor 0 (the robot) for robot state buffers."""
        if all_root_states.shape[0] % self.num_envs != 0:
            raise RuntimeError(
                "Actor root tensor is not evenly divisible by num_envs; "
                "cannot create robot state views for obstacle navigation."
            )
        actors_per_env = all_root_states.shape[0] // self.num_envs
        self._all_root_states = all_root_states
        self._actors_per_env = actors_per_env
        # Robot is deliberately created before each environment's obstacles.
        self.root_states = all_root_states.view(self.num_envs, actors_per_env, 13)[:, 0, :]
        return self.root_states

    def _rebind_robot_state_views(self):
        """Refresh derived views after the base buffers are initialized."""
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = self._quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = self._quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity = self._quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.base_pos = self.root_states[:, :3]

    @staticmethod
    def _quat_rotate_inverse(quat, vector):
        """Small local implementation to avoid rebuilding Isaac Gym views."""
        q_xyz = quat[:, :3]
        q_w = quat[:, 3:4]
        uv = torch.cross(q_xyz, vector, dim=1)
        uuv = torch.cross(q_xyz, uv, dim=1)
        return vector - 2.0 * (q_w * uv + uuv)

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = 0.0
        self.dof_vel[env_ids] = 0.0
        actor_indices = self.robot_actor_indices[env_ids].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices),
        )

    def _reset_root_states(self, env_ids):
        if len(env_ids) == 0:
            return
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.0

        if self._maze_enabled:
            self.root_states[env_ids, :2] = (
                self.env_origins[env_ids, :2]
                + torch.as_tensor(
                    self._maze_start_position,
                    dtype=self.root_states.dtype,
                    device=self.device,
                )
            )

        if bool(getattr(self.cfg.commands, "random_start_yaw", True)):
            yaw = torch_rand_float(
                -math.pi,
                math.pi,
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
            half_yaw = 0.5 * yaw
            quat = torch.zeros(len(env_ids), 4, device=self.device)
            quat[:, 2] = torch.sin(half_yaw)
            quat[:, 3] = torch.cos(half_yaw)
            self.root_states[env_ids, 3:7] = quat

        actor_indices = self.robot_actor_indices[env_ids].to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._all_root_states),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices),
        )

    # --------------------------------------------------------------- buffers
    def _init_buffers(self):
        super()._init_buffers()
        # RotunbotTargetObstacle initializes a convenient [env, actor, 13]
        # view, but target-depth reset must write the simulator's full actor
        # tensor using robot actor indices because every maze wall is an actor.
        all_root_states = gymtorch.wrap_tensor(
            self.gym.acquire_actor_root_state_tensor(self.sim)
        ).view(-1, 13)
        self._select_robot_root_states(all_root_states)
        self._rebind_robot_state_views()
        self.obstacle_centers = torch.as_tensor(
            self._obstacle_centers_cpu,
            dtype=torch.float32,
            device=self.device,
        )
        self.obstacle_sizes = torch.as_tensor(
            [size[:2] for size in self.cfg.obstacles.sizes],
            dtype=torch.float32,
            device=self.device,
        )
        if self._maze_enabled:
            self.obstacle_sizes = torch.as_tensor(
                self._maze_wall_sizes_cpu,
                dtype=torch.float32,
                device=self.device,
            )
        self.obstacle_collision_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.step_collision_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.obstacle_clearance = torch.full(
            (self.num_envs,), float(self.cfg.camera.far_plane), device=self.device
        )
        # Diagnostics only: preserve the wall clearance before reset so an
        # evaluation collision can be compared with the terminal speed.
        self.terminal_obstacle_clearance = torch.full(
            (self.num_envs,),
            float(self.cfg.camera.far_plane),
            device=self.device,
        )
        if self._maze_enabled:
            self._maze_geodesic_distance = torch.as_tensor(
                self._maze_geodesic_distance_cpu,
                dtype=torch.float32,
                device=self.device,
            )
            self._maze_goal_cells_tensor = torch.as_tensor(
                self._maze_goal_cells,
                dtype=torch.long,
                device=self.device,
            )
            self.maze_goal_endpoint_indices = torch.zeros(
                self.num_envs, 2, dtype=torch.long, device=self.device
            )
            self.maze_goal_alpha = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self.maze_goal_sampling_bin = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.maze_goal_distance = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self.last_maze_goal_distance = torch.zeros_like(self.maze_goal_distance)
            self.stall_no_progress_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._maze_next_cells = torch.as_tensor(
                self._maze_next_cells_cpu,
                dtype=torch.long,
                device=self.device,
            )
        self.depth_observation = torch.zeros(
            self.num_envs,
            int(self.cfg.env.depth_height),
            int(self.cfg.env.depth_width),
            dtype=torch.float32,
            device=self.device,
        )
        self.depth_fallback_observation = torch.zeros_like(self.depth_observation)
        self.depth_camera_observation = torch.zeros_like(self.depth_observation)
        self.depth_camera_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._init_camera_tensors()

    def _init_camera_tensors(self):
        if not self._camera_handles:
            return
        try:
            self._camera_depth_tensors = []
            for env_handle, camera_handle in zip(self.envs, self._camera_handles):
                raw_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_handle, camera_handle, gymapi.IMAGE_DEPTH
                )
                wrapped = gymtorch.wrap_tensor(raw_tensor)
                self._camera_depth_tensors.append(
                    wrapped.view(int(self.cfg.camera.height), int(self.cfg.camera.width))
                )
            self._camera_ready = len(self._camera_depth_tensors) == self.num_envs
        except Exception as exc:  # pragma: no cover - depends on Isaac Gym build
            self._camera_depth_tensors = []
            self._camera_ready = False
            if not self._camera_error_reported:
                print(
                    "[RotunbotTargetDepth] failed to access camera tensors; "
                    f"using depth fallback ({exc})"
                )
                self._camera_error_reported = True

    # --------------------------------------------------------------- sensing
    def _apply_depth_noise(self, depth):
        camera_cfg = self.cfg.camera
        if not bool(camera_cfg.add_noise):
            return depth.clamp(0.0, 1.0)
        depth = depth + torch.randn_like(depth) * float(camera_cfg.noise_std)
        dropout = torch.rand_like(depth) < float(camera_cfg.dropout_probability)
        # The observation is normalized distance (0 = near, 1 = far/open
        # space), so missing pixels are filled with the far-plane value.
        depth = torch.where(dropout, torch.ones_like(depth), depth)
        quantization = float(camera_cfg.quantization)
        if quantization > 0.0:
            depth = torch.round(depth / quantization) * quantization
        return depth.clamp(0.0, 1.0)

    def _resize_depth_to_observation(self, depth):
        """Downsample a camera frame to the policy observation resolution."""
        target_height = int(self.cfg.env.depth_height)
        target_width = int(self.cfg.env.depth_width)
        if depth.shape[-2:] == (target_height, target_width):
            return depth
        return F.interpolate(
            depth.unsqueeze(1),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    def _fallback_depth(self):
        """Vectorized pinhole-like horizontal ray/AABB depth observation."""
        height = int(self.cfg.env.depth_height)
        width = int(self.cfg.env.depth_width)
        camera_cfg = self.cfg.camera
        yaw = self.base_euler_tensor[:, 2]
        camera_offset_x = float(camera_cfg.position[0])
        camera_offset_y = float(camera_cfg.position[1])
        camera_x = self.root_states[:, 0] + (
            torch.cos(yaw) * camera_offset_x - torch.sin(yaw) * camera_offset_y
        )
        camera_y = self.root_states[:, 1] + (
            torch.sin(yaw) * camera_offset_x + torch.cos(yaw) * camera_offset_y
        )
        camera_x = camera_x - self.env_origins[:, 0]
        camera_y = camera_y - self.env_origins[:, 1]

        half_fov = math.radians(float(camera_cfg.horizontal_fov)) / 2.0
        ray_angles = torch.linspace(
            -half_fov, half_fov, width, device=self.device
        ).unsqueeze(0)
        ray_angles = ray_angles + yaw.unsqueeze(1)
        ray_x = torch.cos(ray_angles)
        ray_y = torch.sin(ray_angles)
        eps = torch.full_like(ray_x, 1.0e-6)
        ray_x = torch.where(torch.abs(ray_x) < 1.0e-6, eps, ray_x)
        ray_y = torch.where(torch.abs(ray_y) < 1.0e-6, eps, ray_y)

        nearest = torch.full(
            (self.num_envs, width), float(camera_cfg.far_plane), device=self.device
        )
        for obstacle_id, size in enumerate(self.obstacle_sizes):
            center = self.obstacle_centers[:, obstacle_id, :]
            min_corner = center - 0.5 * size
            max_corner = center + 0.5 * size
            tx_a = (min_corner[:, 0].unsqueeze(1) - camera_x.unsqueeze(1)) / ray_x
            tx_b = (max_corner[:, 0].unsqueeze(1) - camera_x.unsqueeze(1)) / ray_x
            ty_a = (min_corner[:, 1].unsqueeze(1) - camera_y.unsqueeze(1)) / ray_y
            ty_b = (max_corner[:, 1].unsqueeze(1) - camera_y.unsqueeze(1)) / ray_y
            t_near = torch.maximum(torch.minimum(tx_a, tx_b), torch.minimum(ty_a, ty_b))
            t_far = torch.minimum(torch.maximum(tx_a, tx_b), torch.maximum(ty_a, ty_b))
            hit = (t_far >= torch.maximum(t_near, torch.zeros_like(t_near))) & (t_far > 0.0)
            distance = torch.where(hit, torch.clamp(t_near, min=0.0), nearest)
            nearest = torch.minimum(nearest, distance)

        near = float(camera_cfg.near_plane)
        far = float(camera_cfg.far_plane)
        normalized_distance = (nearest - near) / max(far - near, 1.0e-6)
        return normalized_distance.unsqueeze(1).expand(-1, height, -1).clone()

    def _capture_depth(self):
        compare = bool(getattr(self, "capture_depth_comparison", False))
        policy_source = str(getattr(self, "depth_policy_source", "fallback"))
        if policy_source not in ("fallback", "camera"):
            policy_source = "fallback"
        fallback_depth = None
        if compare or policy_source == "fallback" or not self._camera_ready:
            # Keep this clean and un-noised so the comparison measures the
            # sensor/rendering difference rather than random observation noise.
            fallback_depth = self._resize_depth_to_observation(self._fallback_depth())
            if compare:
                self.depth_fallback_observation[:] = fallback_depth
                self.depth_camera_valid[:] = False

        if not self._camera_ready:
            if policy_source == "camera":
                raise RuntimeError(
                    "Isaac Gym GPU depth camera was requested as the policy "
                    "input, but camera tensors are unavailable. Check "
                    "enable_camera_sensors_in_headless, graphics_device_id, "
                    "and the GPU pipeline configuration."
                )
            self.depth_observation[:] = self._apply_depth_noise(fallback_depth)
            if compare:
                # This indicates that the requested real camera was not
                # available; keep the diagnostic tensors well-defined.
                self.depth_camera_observation[:] = fallback_depth
            return

        access_started = False
        try:
            # Attached camera transforms are updated by step_graphics after the
            # physics state changes and before the image batch is rendered.
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            access_started = True
            raw_depth = torch.stack(self._camera_depth_tensors, dim=0).to(self.device)
            raw_depth = torch.where(raw_depth < 0.0, -raw_depth, raw_depth)
            raw_depth = torch.where(
                torch.isfinite(raw_depth),
                raw_depth,
                torch.full_like(raw_depth, float(self.cfg.camera.far_plane)),
            )
            near = float(self.cfg.camera.near_plane)
            far = float(self.cfg.camera.far_plane)
            depth = (raw_depth.clamp(near, far) - near) / max(
                far - near, 1.0e-6
            )
            camera_depth = self._resize_depth_to_observation(depth)
            if compare:
                if fallback_depth is None:
                    fallback_depth = self._resize_depth_to_observation(
                        self._fallback_depth()
                    )
                    self.depth_fallback_observation[:] = fallback_depth
                self.depth_camera_observation[:] = camera_depth
                self.depth_camera_valid[:] = True
            selected_depth = camera_depth if policy_source == "camera" else fallback_depth
            if selected_depth is None:
                selected_depth = camera_depth
            self.depth_observation[:] = self._apply_depth_noise(selected_depth)
        except Exception as exc:  # pragma: no cover - depends on Isaac Gym build
            self._camera_ready = False
            if not self._camera_error_reported:
                print(
                    "[RotunbotTargetDepth] camera render failed; "
                    f"using depth fallback ({exc})"
                )
                self._camera_error_reported = True
            if policy_source == "camera":
                raise RuntimeError(
                    "Isaac Gym GPU depth rendering failed while the camera "
                    "was selected as the policy input."
                ) from exc
            if fallback_depth is None:
                fallback_depth = self._resize_depth_to_observation(self._fallback_depth())
            if compare:
                self.depth_fallback_observation[:] = fallback_depth
                self.depth_camera_observation[:] = fallback_depth
            self.depth_observation[:] = self._apply_depth_noise(fallback_depth)
        finally:
            if access_started:
                self.gym.end_access_image_tensors(self.sim)

    # -------------------------------------------------------------- navigation
    def _maze_geodesic_distance_at_robot(self):
        """Return the free-cell shortest-path distance for each robot."""
        local_position = self.root_states[:, :2] - self.env_origins[:, :2]
        maze_shape = self.maze_layout.shape
        cell_size = float(self.cfg.maze.cell_size)
        cell_x = torch.floor(
            local_position[:, 0] / cell_size + float(maze_shape[0]) / 2.0
        ).long().clamp(0, maze_shape[0] - 1)
        cell_y = torch.floor(
            local_position[:, 1] / cell_size + float(maze_shape[1]) / 2.0
        ).long().clamp(0, maze_shape[1] - 1)
        endpoint_indices = self.maze_goal_endpoint_indices
        alpha = self.maze_goal_alpha
        distance_to_first_endpoint = self._maze_geodesic_distance[
            endpoint_indices[:, 0], cell_x, cell_y
        ] + alpha * cell_size
        distance_to_second_endpoint = self._maze_geodesic_distance[
            endpoint_indices[:, 1], cell_x, cell_y
        ] + (1.0 - alpha) * cell_size
        choose_first_endpoint = distance_to_first_endpoint <= distance_to_second_endpoint
        distance = torch.minimum(
            distance_to_first_endpoint, distance_to_second_endpoint
        )
        next_first = self._maze_next_cells[
            endpoint_indices[:, 0], cell_x, cell_y
        ]
        next_second = self._maze_next_cells[
            endpoint_indices[:, 1], cell_x, cell_y
        ]
        next_cell = torch.where(
            choose_first_endpoint.unsqueeze(1), next_first, next_second
        )
        next_center = (
            next_cell.to(dtype=local_position.dtype)
            - torch.as_tensor(
                [float(maze_shape[0]) / 2.0 - 0.5, float(maze_shape[1]) / 2.0 - 0.5],
                dtype=local_position.dtype,
                device=self.device,
            )
        ) * cell_size
        goal_cell_indices = self._maze_goal_cells_tensor[endpoint_indices]
        current_cell = torch.stack((cell_x, cell_y), dim=1)
        in_goal_segment_cell = (
            (current_cell[:, None, :] == goal_cell_indices).all(dim=2).any(dim=1)
        )
        goal_position = self.commands[:, :2] - self.env_origins[:, :2]
        smooth_target = torch.where(
            in_goal_segment_cell.unsqueeze(1), goal_position, next_center
        )
        smooth_offset = torch.linalg.vector_norm(
            local_position - smooth_target, dim=1
        )
        # If the robot is already in one of the two endpoint cells, the
        # endpoint-map distance starts at the cell center and would count the
        # target's within-cell offset twice.  In that case use the direct
        # continuous distance to the target; elsewhere retain the usual
        # center-to-next-cell correction.
        distance = torch.where(
            in_goal_segment_cell, smooth_offset, distance + smooth_offset
        )
        # A contact can put the center briefly inside a wall cell before the
        # reset.  Use the continuous target distance for that terminal sample
        # instead of propagating infinity into the reward or critic targets.
        return torch.where(torch.isfinite(distance), distance, self.goal_dist)

    def step(self, actions):
        # ``extras`` is filled again by reset_idx only when an environment
        # terminates.  Clear the previous step's episode/time_out metadata so
        # PPO cannot reuse a stale timeout bootstrap flag on later transitions.
        self.extras = {}
        if self._maze_enabled and hasattr(self, "maze_goal_distance"):
            self.last_maze_goal_distance[:] = self.maze_goal_distance
        return super().step(actions)

    def post_physics_step(self):
        """Finish a physics step without leaking terminal action history.

        ``LeggedRobot.post_physics_step`` resets an environment, computes its
        initial observation, and then copies ``self.actions`` into
        ``self.last_actions``.  For an environment that just terminated, that
        copy is the terminal action from the previous episode and is therefore
        visible on the next observation update.  Keep reset environments at a
        clean zero-action state while preserving the original behavior for
        environments that are still running.
        """
        super().post_physics_step()
        reset_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_ids) == 0:
            return
        self.last_actions[reset_ids] = 0.0
        self.last_dof_vel[reset_ids] = 0.0
        self.last_root_vel[reset_ids] = 0.0

    def _obstacle_distance(self):
        local_position = self.root_states[:, :2] - self.env_origins[:, :2]
        delta = torch.abs(local_position[:, None, :] - self.obstacle_centers)
        outside = torch.clamp(delta - 0.5 * self.obstacle_sizes[None, :, :], min=0.0)
        return torch.linalg.vector_norm(outside, dim=-1).min(dim=1).values

    def _obstacle_collision_mask(self):
        collision_radius = (
            self.cfg.maze.robot_collision_radius
            if self._maze_enabled
            else self.cfg.obstacles.robot_collision_radius
        )
        return self._obstacle_distance() <= float(collision_radius)

    def _sample_maze_goal_segments(self, count):
        """Sample continuous points on balanced near/mid/far road centerlines."""
        segment_ids = np.empty(count, dtype=np.int64)
        selected_bin_ids = np.empty(count, dtype=np.int64)
        for index in range(count):
            bin_id = self._maze_goal_segment_bin_cursor % len(
                self._maze_goal_segment_orders_cpu
            )
            self._maze_goal_segment_bin_cursor += 1
            selected_bin_ids[index] = bin_id
            order = self._maze_goal_segment_orders_cpu[bin_id]
            pointer = int(self._maze_goal_segment_pointers[bin_id])
            if pointer >= len(order):
                self._maze_goal_rng.shuffle(order)
                pointer = 0
            segment_ids[index] = order[pointer]
            self._maze_goal_segment_pointers[bin_id] = pointer + 1

        endpoints = self._maze_goal_segment_endpoints_cpu[segment_ids].copy()
        segments = self._maze_goal_segments_cpu[segment_ids]
        interval_values = self._maze_goal_segment_valid_intervals_cpu[segment_ids]
        interval_lengths = self._maze_goal_segment_valid_lengths_cpu[segment_ids]
        length_sums = interval_lengths.sum(axis=1)
        interval_choice = (
            self._maze_goal_rng.random(count) * length_sums
        )
        choose_second_interval = interval_choice > interval_lengths[:, 0]
        selected_lengths = np.where(
            choose_second_interval,
            interval_lengths[:, 1],
            interval_lengths[:, 0],
        )
        selected_left = np.where(
            choose_second_interval,
            interval_values[:, 1, 0],
            interval_values[:, 0, 0],
        )
        alpha = (
            selected_left
            + self._maze_goal_rng.random(count) * selected_lengths
        ).astype(np.float32)
        local_goals = (
            segments[:, 0] * (1.0 - alpha[:, None])
            + segments[:, 1] * alpha[:, None]
        ).astype(np.float32)
        return endpoints, alpha, local_goals, selected_bin_ids

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        if self._maze_enabled:
            endpoints, alpha, local_goals, sampling_bins = (
                self._sample_maze_goal_segments(
                    len(env_ids)
                )
            )
            self.maze_goal_endpoint_indices[env_ids] = torch.as_tensor(
                endpoints, dtype=torch.long, device=self.device
            )
            self.maze_goal_alpha[env_ids] = torch.as_tensor(
                alpha, dtype=torch.float32, device=self.device
            )
            self.maze_goal_sampling_bin[env_ids] = torch.as_tensor(
                sampling_bins, dtype=torch.long, device=self.device
            )
            goal = torch.as_tensor(
                local_goals, dtype=self.commands.dtype, device=self.device
            )
            self.commands[env_ids, :2] = (
                goal + self.env_origins[env_ids, :2]
            )
            self.commands[env_ids, 2] = 0.0
            return

        RotunbotTargetRepro._resample_commands(self, env_ids)
        for _ in range(32):
            local_goal = self.commands[env_ids, :2] - self.env_origins[env_ids, :2]
            distance_to_origin = torch.linalg.vector_norm(local_goal, dim=1)
            delta = torch.abs(local_goal[:, None, :] - self.obstacle_centers[env_ids])
            outside = torch.clamp(
                delta - 0.5 * self.obstacle_sizes[None, :, :], min=0.0
            )
            goal_clearance = torch.linalg.vector_norm(outside, dim=-1).min(dim=1).values
            invalid = (distance_to_origin <= 0.5) | (
                goal_clearance < float(self.cfg.obstacles.goal_clearance)
            )
            if not bool(torch.any(invalid)):
                break
            invalid_ids = env_ids[invalid]
            self.commands[invalid_ids, 0] = torch_rand_float(
                self.command_ranges["pos_x"][0],
                self.command_ranges["pos_x"][1],
                (len(invalid_ids), 1),
                device=self.device,
            ).squeeze(1)
            self.commands[invalid_ids, 1] = torch_rand_float(
                self.command_ranges["pos_y"][0],
                self.command_ranges["pos_y"][1],
                (len(invalid_ids), 1),
                device=self.device,
            ).squeeze(1)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self._maze_enabled:
            self.maze_goal_distance[:] = self._maze_geodesic_distance_at_robot()
            path_progress = self.last_maze_goal_distance - self.maze_goal_distance
            has_progress = path_progress > 0.001
            self.stall_no_progress_steps[:] = torch.where(
                has_progress,
                torch.zeros_like(self.stall_no_progress_steps),
                self.stall_no_progress_steps + 1,
            )
        self._capture_depth()
        self.obstacle_clearance[:] = self._obstacle_distance()
        self._draw_maze_markers()

    def _draw_maze_markers(self):
        """Draw the fixed start and current per-environment goal markers."""
        if not self._maze_enabled or self.viewer is None:
            return
        if not bool(getattr(self.cfg.maze, "visualize_start_goal", True)):
            return

        self.gym.clear_lines(self.viewer)
        radius = float(getattr(self.cfg.maze, "marker_radius", 0.18))
        start_geometry = gymutil.WireframeSphereGeometry(
            radius, 12, 8, None, color=(0.1, 0.95, 0.1)
        )
        goal_geometry = gymutil.WireframeSphereGeometry(
            radius, 12, 8, None, color=(0.95, 0.1, 0.1)
        )

        # Positions are maze-local because each vectorized environment has a
        # different Isaac Gym origin.  Draw in every environment so the marker
        # remains correct when the viewer is moved to another instance.
        marker_z = float(self.cfg.init_state.pos[2]) + 0.05
        for env_id, env_handle in enumerate(self.envs):
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(
                float(self._maze_start_position[0]),
                float(self._maze_start_position[1]),
                marker_z,
            )
            goal_pose = gymapi.Transform()
            goal_position = (
                self.commands[env_id, :2] - self.env_origins[env_id, :2]
            ).detach().cpu()
            goal_pose.p = gymapi.Vec3(
                float(goal_position[0]),
                float(goal_position[1]),
                marker_z,
            )
            gymutil.draw_lines(
                start_geometry, self.gym, self.viewer, env_handle, start_pose
            )
            gymutil.draw_lines(
                goal_geometry, self.gym, self.viewer, env_handle, goal_pose
            )

    def check_termination(self):
        self.obstacle_collision_buf[:] = self._obstacle_collision_mask()
        # Keep the pre-reset value for rollout diagnostics.  reset_idx clears
        # obstacle_collision_buf before the runner reads the next observation.
        self.step_collision_buf[:] = self.obstacle_collision_buf
        super().check_termination()
        terminate_on_collision = (
            self.cfg.maze.terminate_on_collision
            if self._maze_enabled
            else self.cfg.obstacles.terminate_on_collision
        )
        if bool(terminate_on_collision):
            self.reset_buf |= self.obstacle_collision_buf

    def reset_idx(self, env_ids):
        terminal_success = None
        terminal_collision = None
        terminal_goal_distance = None
        terminal_speed = None
        terminal_obstacle_clearance = None
        terminal_timeout = None
        terminal_unstable = None
        terminal_out_of_bounds = None
        if len(env_ids) > 0:
            # Capture terminal flags before the parent reset replaces the
            # episode state.  These are training diagnostics only; they do
            # not alter rewards or reset behavior.
            terminal_success = self.success_buf[env_ids].float().detach()
            terminal_collision = self.obstacle_collision_buf[env_ids].float().detach()
            terminal_goal_distance = self.terminal_goal_dist[env_ids].detach()
            terminal_speed = self.terminal_speed[env_ids].detach()
            terminal_obstacle_clearance = self.obstacle_clearance[env_ids].detach()
            terminal_timeout = self.terminal_timeout[env_ids].float().detach()
            terminal_unstable = self.terminal_unstable[env_ids].float().detach()
            terminal_out_of_bounds = self.terminal_out_of_bounds[env_ids].float().detach()
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            # Do not leak a terminal collision or stale wall distance into the
            # first privileged observation of the next episode.
            self.obstacle_collision_buf[env_ids] = False
            self.obstacle_clearance[env_ids] = float(self.cfg.camera.far_plane)
            self.terminal_obstacle_clearance[env_ids] = terminal_obstacle_clearance
            # The frame captured just before a terminal reset belongs to the
            # previous episode; start the new history from open space.  The
            # current encoding is normalized distance: 1 means far/open.
            self.depth_observation[env_ids] = 1.0
            if self._maze_enabled and hasattr(self, "maze_goal_distance"):
                reset_distance = self._maze_geodesic_distance_at_robot()
                self.maze_goal_distance[env_ids] = reset_distance[env_ids]
                self.last_maze_goal_distance[env_ids] = reset_distance[env_ids]
                self.stall_no_progress_steps[env_ids] = 0
            if "episode" in self.extras:
                self.extras["episode"]["success"] = terminal_success.mean()
                self.extras["episode"]["collision"] = terminal_collision.mean()
                self.extras["episode"]["_episode_count"] = torch.tensor(
                    float(len(env_ids)), device=self.device
                )
                self.extras["episode"]["terminal_goal_distance"] = (
                    terminal_goal_distance.mean()
                )
                self.extras["episode"]["terminal_speed"] = terminal_speed.mean()
                self.extras["episode"]["terminal_obstacle_clearance"] = (
                    terminal_obstacle_clearance.mean()
                )
                self.extras["episode"]["timeout"] = terminal_timeout.mean()
                self.extras["episode"]["unstable"] = terminal_unstable.mean()
                self.extras["episode"]["out_of_bounds"] = terminal_out_of_bounds.mean()

    # ------------------------------------------------------------- observations
    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(cfg.env.num_single_obs, device=self.device)
        self.add_noise = bool(cfg.noise.add_noise)
        scales = cfg.noise.noise_scales
        level = float(cfg.noise.noise_level)
        noise_vec[2:5] = float(scales.lin_vel) * level
        noise_vec[5:8] = float(scales.ang_vel) * level
        noise_vec[8:11] = float(scales.gravity) * level
        noise_vec[11:13] = float(scales.dof_pos) * level
        noise_vec[13:15] = float(scales.dof_vel) * level
        # Goal and actions are intentionally kept clean; camera noise is added
        # in image space where its spatial structure is preserved.
        return noise_vec

    def compute_observations(self):
        self._update_base_euler()
        yaw = self.base_euler_tensor[:, 2]
        goal_delta = self.commands[:, :2] - self.root_states[:, :2]
        relative_goal = torch.stack(
            (
                torch.cos(yaw) * goal_delta[:, 0] + torch.sin(yaw) * goal_delta[:, 1],
                -torch.sin(yaw) * goal_delta[:, 0] + torch.cos(yaw) * goal_delta[:, 1],
            ),
            dim=1,
        )
        goal_range = max(
            abs(float(self.command_ranges["pos_x"][0])),
            abs(float(self.command_ranges["pos_x"][1])),
            abs(float(self.command_ranges["pos_y"][0])),
            abs(float(self.command_ranges["pos_y"][1])),
            1.0,
        )
        controller_target = torch.stack(
            (
                self.output_actions[:, 0]
                / max(float(self.cfg.control.first_vel_limits), 1.0e-6),
                self.output_actions[:, 1]
                / max(float(self.cfg.control.second_pos_limits), 1.0e-6),
            ),
            dim=1,
        ).clamp(-1.0, 1.0)
        clean_proprio = torch.cat(
            (
                (relative_goal / goal_range).clamp(-1.0, 1.0),
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.dof_pos * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.last_actions,
                controller_target,
            ),
            dim=1,
        )
        proprio = clean_proprio
        if self.add_noise:
            proprio = proprio + (
                2.0 * torch.rand_like(proprio) - 1.0
            ) * self.noise_scale_vec[: self.cfg.env.proprio_dim]
        obs_now = torch.cat((proprio, self.depth_observation.flatten(1)), dim=1)
        self.obs_history.append(obs_now)
        clearance_scale = max(
            float(
                self.cfg.maze.cell_size
                if self._maze_enabled
                else self.cfg.obstacles.safety_clearance
            ),
            1.0e-6,
        )
        critic_clearance = (
            self.obstacle_clearance / clearance_scale
        ).clamp(0.0, 1.0).unsqueeze(1)
        critic_collision = self.obstacle_collision_buf.float().unsqueeze(1)
        critic_frame = torch.cat(
            (clean_proprio.detach(), critic_clearance, critic_collision), dim=1
        )
        self.critic_history.append(critic_frame)
        self.obs_buf = torch.stack(tuple(self.obs_history), dim=1).flatten(1)
        self.privileged_obs_buf = torch.cat(tuple(self.critic_history), dim=1)

    # --------------------------------------------------------------- rewards
    def _reward_approaching_target(self):
        """Give continuous progress feedback instead of a binary sign.

        The inherited +1/-1 reward is brittle in a maze: a necessary turn or
        detour can briefly increase Euclidean goal distance.  Clipping the
        normalized distance change preserves directional shaping while making
        those transitions less discontinuous.
        """
        normalization = max(
            float(getattr(self.cfg.rewards, "progress_normalization", 0.05)),
            1.0e-6,
        )
        progress = self.last_goal_dist - self.goal_dist
        if self._maze_enabled and hasattr(self, "maze_goal_distance"):
            maze_progress = self.last_maze_goal_distance - self.maze_goal_distance
            euclidean_weight = float(
                getattr(self.cfg.rewards, "maze_euclidean_progress_weight", 0.25)
            )
            progress = maze_progress + euclidean_weight * progress
        return (progress / normalization).clamp(-1.0, 1.0)

    def _reward_collision(self):
        return self.obstacle_collision_buf.float()

    def _reward_obstacle_clearance(self):
        safety_clearance = (
            self.cfg.maze.safety_clearance
            if self._maze_enabled
            else self.cfg.obstacles.safety_clearance
        )
        violation = torch.relu(
            float(safety_clearance) - self.obstacle_clearance
        )
        return violation.square()

    def _reward_stall_far_from_goal(self):
        """Softly discourage stopping in open space before reaching the goal."""
        success_distance = float(self._current_success_distance())
        far_from_goal = self.goal_dist > success_distance
        speed_threshold = max(
            float(getattr(self.cfg.rewards, "stall_speed_threshold", 0.4)),
            1.0e-6,
        )
        speed = torch.linalg.vector_norm(self.base_lin_vel, dim=1)
        speed_deficit = torch.relu(speed_threshold - speed) / speed_threshold
        no_path_progress = torch.ones_like(far_from_goal)
        if self._maze_enabled and hasattr(self, "stall_no_progress_steps"):
            progress_window = int(
                getattr(self.cfg.rewards, "stall_progress_window", 8)
            )
            no_path_progress = self.stall_no_progress_steps >= progress_window
        open_space = self.obstacle_clearance > float(
            getattr(self.cfg.rewards, "stall_clearance_threshold", 0.55)
        )
        return (
            far_from_goal.float()
            * no_path_progress.float()
            * open_space.float()
            * speed_deficit
        )
