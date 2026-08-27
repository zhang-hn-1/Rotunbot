"""V0 depth-aware local navigation environment."""

import math

import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.navigation.local_goal import world_goal_to_robot_xy
from .rotunbot_maze import RotunbotMaze
from .rotunbot_maze_camera import DepthCameraMixin
from .rotunbot_maze_local_depth_config import RotunbotMazeLocalDepthCfg


def build_depth_local_observation(
    projected_gravity,
    base_lin_vel,
    base_ang_vel,
    actuated_joint_pos,
    dof_vel,
    local_goal,
    previous_actions,
    depth,
    max_goal_distance=8.0,
):
    """Build the exact V0 actor layout without world-frame leakage."""
    fields = (
        ("projected_gravity", projected_gravity, 3),
        ("base_lin_vel", base_lin_vel, 3),
        ("base_ang_vel", base_ang_vel, 3),
        ("actuated_joint_pos", actuated_joint_pos, 1),
        ("dof_vel", dof_vel, 2),
        ("local_goal", local_goal, 2),
        ("previous_actions", previous_actions, 2),
    )
    batch = projected_gravity.shape[0]
    for name, value, width in fields:
        if value.ndim != 2 or value.shape != (batch, width):
            raise ValueError(f"{name} must have shape [{batch}, {width}]")
    if depth.ndim != 3 or tuple(depth.shape[1:]) != (8, 32) or depth.shape[0] != batch:
        raise ValueError(f"depth must have shape [{batch}, 8, 32]")
    return torch.cat(
        (
            projected_gravity,
            base_lin_vel,
            base_ang_vel,
            actuated_joint_pos,
            dof_vel,
            local_goal / float(max_goal_distance),
            previous_actions,
            depth.reshape(batch, -1),
        ),
        dim=1,
    )


class RotunbotMazeLocalDepth(DepthCameraMixin, RotunbotMaze):
    """Local-goal executor that retains the existing planar action interface."""

    cfg: RotunbotMazeLocalDepthCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.num_single_obs = 272
        self.num_short_obs = 272
        self._init_depth_camera_state()
        self._depth_scene_centers = []
        self._depth_scene_half_extents = []
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _create_envs(self):
        super()._create_envs()
        self._create_camera_sensors()

    def _create_scene_assets(self):
        scene_mode = getattr(self.cfg.maze, "scene_mode", "none")
        if scene_mode == "corridor":
            self._depth_scene_centers = [(0.0, -1.4), (0.0, 1.4)]
            self._depth_scene_half_extents = [(8.0, 0.2), (8.0, 0.2)]
            wall_options = gymapi.AssetOptions()
            wall_options.fix_base_link = True
            wall_asset = self.gym.create_box(self.sim, 16.0, 0.4, 1.2, wall_options)
            return {
                "corridor_asset": wall_asset,
                "corridor_color": gymapi.Vec3(0.32, 0.38, 0.48),
            }
        if not bool(getattr(self.cfg.maze, "enabled", False)):
            return {}
        return super()._create_scene_assets()

    def _create_scene_actors(self, env_handle, env_id, scene_assets):
        if getattr(self.cfg.maze, "scene_mode", "none") == "corridor":
            origin = self.env_origins[env_id]
            for wall_index, (center_x, center_y) in enumerate(self._depth_scene_centers):
                wall_pose = gymapi.Transform()
                wall_pose.p = gymapi.Vec3(
                    origin[0].item() + center_x,
                    origin[1].item() + center_y,
                    origin[2].item() + 0.6,
                )
                wall_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                wall_actor = self.gym.create_actor(
                    env_handle,
                    scene_assets["corridor_asset"],
                    wall_pose,
                    f"corridor_wall_{wall_index}",
                    env_id,
                    0,
                    0,
                )
                self.gym.set_rigid_body_color(
                    env_handle,
                    wall_actor,
                    0,
                    gymapi.MESH_VISUAL,
                    scene_assets["corridor_color"],
                )
            return
        if bool(getattr(self.cfg.maze, "enabled", False)):
            return super()._create_scene_actors(env_handle, env_id, scene_assets)

    def _get_noise_scale_vec(self, cfg):
        self.add_noise = bool(getattr(cfg.noise, "add_noise", False))
        return torch.zeros(cfg.env.num_single_obs, device=self.device)

    def _init_buffers(self):
        if not bool(getattr(self.cfg.maze, "enabled", False)):
            self._maze_wall_centers = []
        super()._init_buffers()
        self.global_goal_xy_world = torch.zeros(self.num_envs, 2, device=self.device)
        self.active_local_goal_xy_world = torch.zeros_like(self.global_goal_xy_world)
        self.active_local_goal_xy_robot = torch.zeros_like(self.global_goal_xy_world)
        self.waypoint_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.needs_new_waypoint = torch.zeros_like(self.waypoint_reached)
        self.global_goal_reached = torch.zeros_like(self.waypoint_reached)
        self.waypoint_changed = torch.zeros_like(self.waypoint_reached)
        self.waypoint_reach_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.terminal_local_success = torch.zeros_like(self.waypoint_reached)
        self.terminal_global_success = torch.zeros_like(self.waypoint_reached)
        self.terminal_collision = torch.zeros_like(self.waypoint_reached)
        self.terminal_timeout = torch.zeros_like(self.waypoint_reached)
        self.terminal_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self.terminal_local_goal_distance = torch.zeros_like(self.terminal_goal_distance)
        self.terminal_position = torch.zeros(self.num_envs, 2, device=self.device)
        self.terminal_waypoint_reach_count = torch.zeros_like(self.waypoint_reach_count)
        self.prev_local_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.depth_observation = torch.ones(self.num_envs, 8, 32, device=self.device)
        self._init_camera_tensors()
        self.depth_backend = self.depth_backend_actual

    def _get_depth_fallback_aabbs(self):
        if getattr(self.cfg.maze, "scene_mode", "none") == "corridor":
            return (
                torch.as_tensor(self._depth_scene_centers, dtype=torch.float32, device=self.device),
                torch.as_tensor(self._depth_scene_half_extents, dtype=torch.float32, device=self.device),
            )
        if not bool(getattr(self.cfg.maze, "enabled", False)):
            return (
                torch.empty(0, 2, device=self.device),
                torch.empty(0, 2, device=self.device),
            )
        centers = getattr(self, "maze_wall_centers", None)
        if centers is None:
            return torch.empty(0, 2, device=self.device), torch.empty(0, 2, device=self.device)
        half = float(self.cfg.maze.cell_size) / 2.0
        return centers, torch.full_like(centers, half)

    def _yaw_from_quaternion(self):
        qx, qy, qz, qw = self.base_quat.unbind(dim=1)
        return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy.square() + qz.square()))

    def _update_base_euler(self):
        """Update roll/pitch/yaw only for termination and frame transforms."""
        qx, qy, qz, qw = self.base_quat.unbind(dim=1)
        roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx.square() + qy.square()))
        pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
        yaw = self._yaw_from_quaternion()
        self.base_euler_tensor[:] = torch.stack((roll, pitch, yaw), dim=1)

    def _sample_local_goals(self, env_ids):
        count = len(env_ids)
        distance_range = getattr(self.cfg.commands, "local_goal_distance", (0.4, 1.5))
        lateral_range = getattr(self.cfg.commands, "local_goal_lateral", (-0.6, 0.6))
        forward = torch_rand_float(distance_range[0], distance_range[1], (count, 1), device=self.device).squeeze(1)
        lateral = torch_rand_float(lateral_range[0], lateral_range[1], (count, 1), device=self.device).squeeze(1)
        local = torch.stack((forward, lateral), dim=1)
        yaw = self._yaw_from_quaternion()[env_ids]
        c, s = torch.cos(yaw), torch.sin(yaw)
        world_delta = torch.stack((c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1]), dim=1)
        self.global_goal_xy_world[env_ids] = self.root_states[env_ids, :2] + world_delta
        self.active_local_goal_xy_world[env_ids] = self.global_goal_xy_world[env_ids]
        self.commands[env_ids, :2] = self.global_goal_xy_world[env_ids]

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        if len(env_ids) == 0 or not bool(getattr(self.cfg.commands, "random_start_yaw", False)):
            return
        yaw = torch_rand_float(-math.pi, math.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 3] = 0.0
        self.root_states[env_ids, 4] = 0.0
        self.root_states[env_ids, 5] = torch.sin(0.5 * yaw)
        self.root_states[env_ids, 6] = torch.cos(0.5 * yaw)
        actor_indices = self.robot_actor_indices[env_ids].to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.actor_root_state),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices),
        )

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        stage = int(getattr(self.cfg.commands, "local_curriculum_stage", 0))
        if stage >= 4 and bool(getattr(self.cfg.maze, "enabled", False)):
            super()._resample_commands(env_ids)
            self.global_goal_xy_world[env_ids] = self.maze_global_goals[env_ids]
            self.active_local_goal_xy_world[env_ids] = self.global_goal_xy_world[env_ids]
        else:
            self._sample_local_goals(env_ids)

    def _update_local_goal(self):
        yaw = self._yaw_from_quaternion()
        self.active_local_goal_xy_robot[:] = world_goal_to_robot_xy(
            self.root_states[:, :2], yaw, self.active_local_goal_xy_world
        )

    def _wall_distance(self):
        centers, half_extents = self._get_depth_fallback_aabbs()
        if centers.numel() == 0:
            return torch.full((self.num_envs,), float(self.cfg.camera.far_plane), device=self.device)
        position = self.root_states[:, :2] - self.env_origins[:, :2]
        delta = (position[:, None, :] - centers[None, :, :]).abs() - half_extents[None, :, :]
        distance = torch.linalg.vector_norm(torch.clamp(delta, min=0.0), dim=-1)
        return distance.min(dim=1).values

    def _maze_collision_mask(self):
        return self._wall_distance() <= float(getattr(self.cfg.maze, "robot_collision_radius", 0.4))

    def _post_physics_step_callback(self):
        self._update_base_euler()
        self._update_local_goal()
        self.obstacle_clearance = self._wall_distance()
        self.maze_collision_buf = self._maze_collision_mask()
        self.step_collision_buf = self.maze_collision_buf.clone()

    def compute_observations(self):
        self._update_local_goal()
        self.depth_observation[:] = self.capture_depth()
        self.obs_buf[:] = build_depth_local_observation(
            self.projected_gravity,
            self.base_lin_vel,
            self.base_ang_vel,
            self.dof_pos[:, 1:2],
            self.dof_vel,
            self.active_local_goal_xy_robot,
            self.last_actions,
            self.depth_observation,
        )
        self.privileged_obs_buf[:, :16] = self.obs_buf[:, :16]
        clearance = self.obstacle_clearance / float(self.cfg.camera.far_plane)
        self.privileged_obs_buf[:, 16] = clearance.clamp(0.0, 1.0)
        self.privileged_obs_buf[:, 17] = self.maze_collision_buf.float()
        self.depth_backend = self.depth_backend_actual

    def _advance_active_waypoint(self, env_ids=None):
        """Mark reached waypoints pending planner replacement.

        The planner, rather than this environment callback, owns the next
        waypoint selection.  Keeping the current target here prevents one
        transition from exposing the final global goal to the actor.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.needs_new_waypoint[env_ids] = True
        if hasattr(self, "waypoint_reach_count"):
            self.waypoint_reach_count[env_ids] += 1

    def set_active_waypoint(self, waypoint_xy_world, env_ids=None):
        """Latch a planner-provided feasible waypoint for every environment."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        waypoint_xy_world = torch.as_tensor(
            waypoint_xy_world, dtype=torch.float32,
            device=self.device,
        )
        expected_shape = (len(env_ids), 2)
        if waypoint_xy_world.shape != expected_shape:
            raise ValueError(f"waypoint must have shape {expected_shape}")
        self.active_local_goal_xy_world[env_ids] = waypoint_xy_world
        self._update_local_goal()
        self.prev_local_goal_dist[env_ids] = torch.linalg.vector_norm(
            self.active_local_goal_xy_robot[env_ids], dim=1
        )
        self.needs_new_waypoint[env_ids] = False
        self.waypoint_reached[env_ids] = False
        self.waypoint_changed[env_ids] = True
        # The planner normally injects the replacement between environment
        # steps.  Refresh the goal slots immediately so the next actor call
        # cannot consume the pending waypoint's stale observation.
        if hasattr(self, "obs_buf") and self.obs_buf.shape[-1] >= 16:
            local_goal = self.active_local_goal_xy_robot[env_ids] / 8.0
            self.obs_buf[env_ids, 12:14] = local_goal
            if getattr(self, "privileged_obs_buf", None) is not None:
                self.privileged_obs_buf[env_ids, 12:14] = local_goal

    def filter_feasible_waypoints(self, local_waypoints):
        """Return candidates compatible with the measured planar action interface."""
        candidates = torch.as_tensor(local_waypoints, dtype=torch.float32, device=self.device)
        if candidates.ndim != 2 or candidates.shape[1] != 2:
            raise ValueError("local_waypoints must have shape [M, 2]")
        distance = torch.linalg.vector_norm(candidates, dim=1)
        bearing = torch.abs(torch.atan2(candidates[:, 1], candidates[:, 0]))
        distance_limit = getattr(self.cfg.commands, "distance_limit", (0.25, 2.0))
        lateral_limit = float(getattr(self.cfg.commands, "lateral_limit", 0.8))
        minimum_forward = float(getattr(self.cfg.commands, "minimum_forward_component", 0.15))
        bearing_limit = math.radians(float(getattr(self.cfg.commands, "bearing_limit_deg", 120.0)))
        return (
            (distance >= float(distance_limit[0]))
            & (distance <= float(distance_limit[1]))
            & (candidates[:, 1].abs() <= lateral_limit)
            & (candidates[:, 0] >= minimum_forward)
            & (bearing <= bearing_limit)
        )

    def check_termination(self):
        self._update_local_goal()
        local_dist = torch.linalg.vector_norm(self.active_local_goal_xy_robot, dim=1)
        global_dist = torch.linalg.vector_norm(self.global_goal_xy_world - self.root_states[:, :2], dim=1)
        stage = int(getattr(self.cfg.commands, "local_curriculum_stage", 0))
        self.waypoint_reached[:] = local_dist <= float(getattr(self.cfg.commands, "local_waypoint_radius", 0.25))
        self.global_goal_reached[:] = global_dist <= float(getattr(self.cfg.commands, "global_goal_radius", 0.35))
        if stage >= 4:
            advance = self.waypoint_reached & ~self.global_goal_reached & ~self.needs_new_waypoint
            if torch.any(advance):
                self._advance_active_waypoint(advance.nonzero(as_tuple=False).flatten())
            success = self.global_goal_reached
        else:
            success = self.waypoint_reached
        roll_cutoff = torch.abs(self.base_euler_tensor[:, 0]) > 1.2
        pitch_cutoff = torch.abs(self.base_euler_tensor[:, 1]) > 1.2
        local_position = self.root_states[:, :2] - self.env_origins[:, :2]
        out_of_bounds = (local_position.abs() > 15.0).any(dim=1)
        self.time_out_buf[:] = self.episode_length_buf >= self.max_episode_length
        self.success_buf[:] = success
        terminal_collision = self.maze_collision_buf & bool(getattr(self.cfg.maze, "terminate_on_collision", True))
        if hasattr(self, "terminal_local_success"):
            self.terminal_local_success[:] = self.waypoint_reached
            self.terminal_global_success[:] = self.global_goal_reached
            self.terminal_collision[:] = terminal_collision
            self.terminal_timeout[:] = self.time_out_buf
            self.terminal_goal_distance[:] = global_dist
            self.terminal_local_goal_distance[:] = local_dist
            self.terminal_position[:] = self.root_states[:, :2]
            self.terminal_waypoint_reach_count[:] = self.waypoint_reach_count
        self.reset_buf[:] = self.time_out_buf | success | roll_cutoff | pitch_cutoff | out_of_bounds | terminal_collision

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.waypoint_reached[env_ids] = False
        self.needs_new_waypoint[env_ids] = False
        self.global_goal_reached[env_ids] = False
        self.waypoint_changed[env_ids] = False
        if hasattr(self, "waypoint_reach_count"):
            self.waypoint_reach_count[env_ids] = 0
        self.active_local_goal_xy_world[env_ids] = self.global_goal_xy_world[env_ids]
        self._update_local_goal()
        self.prev_local_goal_dist[env_ids] = torch.linalg.vector_norm(self.active_local_goal_xy_robot[env_ids], dim=1)

    def _reward_local_progress(self):
        distance = torch.linalg.vector_norm(self.active_local_goal_xy_robot, dim=1)
        progress = self.prev_local_goal_dist - distance
        return torch.where(self.waypoint_changed, torch.zeros_like(progress), progress)

    def _reward_local_reach(self):
        return self.waypoint_reached.float()

    def _reward_wall_penalty(self):
        safety = float(getattr(self.cfg.maze, "safety_clearance", 0.8))
        return -torch.relu(safety - self.obstacle_clearance)

    def _reward_collision(self):
        return self.maze_collision_buf.float()

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_time(self):
        return torch.ones(self.num_envs, device=self.device)

    def post_physics_step(self):
        """Advance the progress baseline only after reward computation."""
        super().post_physics_step()
        self.prev_local_goal_dist[:] = torch.linalg.vector_norm(
            self.active_local_goal_xy_robot, dim=1
        )
        self.waypoint_changed[:] = False
