"""V0 depth-aware local navigation environment."""

import math

import torch
from isaacgym import gymtorch
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
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _create_envs(self):
        super()._create_envs()
        self._create_camera_sensors()

    def _create_scene_assets(self):
        if not bool(getattr(self.cfg.maze, "enabled", False)):
            return {}
        return super()._create_scene_assets()

    def _create_scene_actors(self, env_handle, env_id, scene_assets):
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
        self.global_goal_reached = torch.zeros_like(self.waypoint_reached)
        self.waypoint_changed = torch.zeros_like(self.waypoint_reached)
        self.prev_local_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.depth_observation = torch.ones(self.num_envs, 8, 32, device=self.device)
        self._init_camera_tensors()
        self.depth_backend = self.depth_backend_actual

    def _get_depth_fallback_aabbs(self):
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
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.active_local_goal_xy_world[env_ids] = self.global_goal_xy_world[env_ids]
        self._update_local_goal()
        self.prev_local_goal_dist[env_ids] = torch.linalg.vector_norm(
            self.active_local_goal_xy_robot[env_ids], dim=1
        )
        self.waypoint_changed[env_ids] = True

    def set_active_waypoint(self, waypoint_xy_world):
        """Latch a planner-provided feasible waypoint for every environment."""
        waypoint_xy_world = torch.as_tensor(
            waypoint_xy_world, dtype=torch.float32,
            device=self.device,
        )
        if waypoint_xy_world.shape != self.active_local_goal_xy_world.shape:
            raise ValueError(f"waypoint must have shape {tuple(self.active_local_goal_xy_world.shape)}")
        self.active_local_goal_xy_world[:] = waypoint_xy_world
        self._update_local_goal()
        self.prev_local_goal_dist[:] = torch.linalg.vector_norm(self.active_local_goal_xy_robot, dim=1)
        self.waypoint_changed[:] = True

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
            advance = self.waypoint_reached & ~self.global_goal_reached
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
        self.reset_buf[:] = self.time_out_buf | success | roll_cutoff | pitch_cutoff | out_of_bounds | terminal_collision

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.waypoint_reached[env_ids] = False
        self.global_goal_reached[env_ids] = False
        self.waypoint_changed[env_ids] = False
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
