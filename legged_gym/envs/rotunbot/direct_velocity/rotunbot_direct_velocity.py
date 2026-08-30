"""Depth-aware direct velocity navigation through the frozen V62 stack."""

import math

import torch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.navigation.direct_velocity import (
    goal_kinematic_recovery,
    goal_speed_alignment,
    goal_turn_alignment,
    normalized_action_to_velocity_command,
    update_goal_recovery_phase,
    velocity_command_rate_penalty,
)
from legged_gym.navigation.direct_velocity_observation import (
    build_direct_velocity_observation,
)
from legged_gym.navigation.v62_corridor_task import RotunbotVelCorridor
from ..maze.rotunbot_maze_camera import DepthCameraMixin
from .rotunbot_direct_velocity_config import RotunbotDirectVelocityCfg


class RotunbotDirectVelocity(RotunbotVelCorridor, DepthCameraMixin):
    """SRU chooses desired velocities; V62 remains the actuator executor."""

    cfg: RotunbotDirectVelocityCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self._init_depth_camera_state()
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _create_envs(self):
        super()._create_envs()
        self._create_camera_sensors()

    def _init_buffers(self):
        super()._init_buffers()
        self.num_single_obs = int(self.cfg.env.num_single_obs)
        self.num_short_obs = int(self.cfg.env.num_short_obs)
        self.global_goal_xy_world = torch.zeros(self.num_envs, 2, device=self.device)
        self.previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self.base_euler_tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self.previous_velocity_command = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self.last_velocity_command = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self.goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.terminal_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self.depth_observation = torch.ones(
            self.num_envs, self.cfg.env.depth_height, self.cfg.env.depth_width,
            device=self.device,
        )
        self.obstacle_clearance = torch.full(
            (self.num_envs,), float(self.cfg.camera.far_plane), device=self.device
        )
        self.step_collision_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.goal_reached_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.goal_recovery_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.goal_recovery_activation_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.terminal_applied_feasible_command = torch.zeros_like(
            self.applied_feasible_command
        )
        self.terminal_tracking_velocity = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self.terminal_position = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self.terminal_command_target = torch.zeros_like(self.command_targets)
        self.terminal_goal_xy_robot = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self.terminal_transition_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.terminal_transition_state = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.terminal_goal_recovery_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.terminal_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.terminal_collision = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.terminal_timeout = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.success_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._init_camera_tensors()

    def _get_depth_fallback_aabbs(self):
        obstacles = getattr(self.cfg, "direct_obstacle_aabbs", ())
        if not obstacles:
            return (
                torch.empty(0, 2, device=self.device),
                torch.empty(0, 2, device=self.device),
            )
        centers, half_extents = zip(*obstacles)
        return (
            torch.as_tensor(centers, dtype=torch.float32, device=self.device),
            torch.as_tensor(half_extents, dtype=torch.float32, device=self.device),
        )

    def _yaw_from_quaternion(self, quaternion=None):
        quaternion = self.base_quat if quaternion is None else quaternion
        qx, qy, qz, qw = quaternion.unbind(dim=1)
        return torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy.square() + qz.square()),
        )

    def _goal_xy_robot(self):
        delta = self.global_goal_xy_world - self.root_states[:, :2]
        yaw = self._yaw_from_quaternion()
        c, s = torch.cos(yaw), torch.sin(yaw)
        return torch.stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]), dim=1)

    def _proprioception(self):
        return torch.cat(
            (
                self.projected_gravity,
                self.tracking_lin_vel,
                self.tracking_ang_vel,
                self.dof_pos[:, 1:2],
                self.dof_vel,
            ),
            dim=1,
        )

    def _wall_distance(self):
        centers, half_extents = self._get_depth_fallback_aabbs()
        if centers.numel() == 0:
            return torch.full(
                (self.num_envs,), float(self.cfg.camera.far_plane), device=self.device
            )
        position = self.root_states[:, :2] - self.env_origins[:, :2]
        delta = (position[:, None, :] - centers[None, :, :]).abs() - half_extents[None, :, :]
        return torch.linalg.vector_norm(torch.clamp(delta, min=0.0), dim=-1).min(dim=1).values

    def _post_physics_step_callback(self):
        # Explicitly retain V62 command-governor and tracking updates.
        from ..vel_tracking.rotunbot_vel import RotunbotVel
        RotunbotVel._post_physics_step_callback(self)
        qx, qy, qz, qw = self.base_quat.unbind(dim=1)
        roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx.square() + qy.square()))
        pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
        self.base_euler_tensor[:, 0] = roll
        self.base_euler_tensor[:, 1] = pitch
        self.base_euler_tensor[:, 2] = self._yaw_from_quaternion()
        self.obstacle_clearance = self._wall_distance()
        self.step_collision_buf[:] = self.obstacle_clearance <= float(self.cfg.maze.robot_collision_radius)
        self.goal_dist[:] = torch.linalg.vector_norm(
            self.global_goal_xy_world - self.root_states[:, :2], dim=1
        )
        recovery = update_goal_recovery_phase(
            self.goal_recovery_active,
            self._goal_xy_robot(),
            minimum_turn_radius=self.cfg.commands.minimum_turn_radius,
            goal_radius=self.cfg.commands.goal_radius,
            enter_bearing=self.cfg.commands.recovery_enter_bearing,
            exit_bearing=self.cfg.commands.recovery_exit_bearing,
            exit_distance_margin=self.cfg.commands.recovery_exit_distance_margin,
        )
        self.goal_recovery_activation_count += (
            recovery & ~self.goal_recovery_active
        ).long()
        self.goal_recovery_active.copy_(recovery)

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        distances = torch_rand_float(
            self.cfg.commands.goal_distance[0],
            self.cfg.commands.goal_distance[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        bearings = torch_rand_float(
            self.cfg.commands.goal_bearing[0],
            self.cfg.commands.goal_bearing[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        replay_specs = getattr(self.cfg.commands, "replay_goal_specs", ())
        if replay_specs:
            selector = torch.rand(len(env_ids), device=self.device)
            lower = 0.0
            for probability, distance_range, bearing_range in replay_specs:
                upper = lower + float(probability)
                replay_mask = (selector >= lower) & (selector < upper)
                if torch.any(replay_mask):
                    count = int(replay_mask.sum().item())
                    distances[replay_mask] = torch_rand_float(
                        distance_range[0], distance_range[1], (count, 1), device=self.device
                    ).squeeze(1)
                    bearings[replay_mask] = torch_rand_float(
                        bearing_range[0], bearing_range[1], (count, 1), device=self.device
                    ).squeeze(1)
                lower = upper
        # During reset, ``base_quat`` still contains the previous physics
        # snapshot.  Read the freshly randomized root quaternion so a random
        # start yaw cannot desynchronize the sampled world goal.
        yaw = self._yaw_from_quaternion(self.root_states[env_ids, 3:7])
        world_bearing = yaw + bearings
        self.global_goal_xy_world[env_ids, 0] = self.root_states[env_ids, 0] + distances * torch.cos(world_bearing)
        self.global_goal_xy_world[env_ids, 1] = self.root_states[env_ids, 1] + distances * torch.sin(world_bearing)
        self.commands[env_ids, :2] = 0.0
        self.goal_dist[env_ids] = distances

    def _compute_torques(self, actions):
        # V62 maps the currently governed command to actuator targets. The
        # navigation action is consumed in step(), never as an actuator action.
        from ..vel_tracking.rotunbot_vel import RotunbotVel
        return RotunbotVel._compute_torques(self, torch.zeros_like(actions))

    def step(self, actions):
        if self.common_step_counter % self.upper_level_command_interval_steps == 0:
            command = normalized_action_to_velocity_command(
                actions,
                self.cfg.commands.max_forward_speed,
                self.cfg.commands.max_yaw_rate,
                self.cfg.commands.minimum_turn_radius,
                self.cfg.commands.feasible_envelope_fraction,
                preserve_curvature_when_saturating=bool(
                    getattr(
                        self.cfg.commands,
                        "preserve_curvature_when_saturating",
                        False,
                    )
                ),
                curvature_fraction_breakpoints=getattr(
                    self.cfg.commands,
                    "stable_curvature_fraction_breakpoints",
                    None,
                ),
                curvature_max_speed_values=getattr(
                    self.cfg.commands,
                    "stable_curvature_max_speed_values",
                    None,
                ),
            )
            self.last_velocity_command.copy_(self.previous_velocity_command)
            self.previous_velocity_command.copy_(command)
            self.set_command_targets(command)
        from ..vel_tracking.rotunbot_vel import RotunbotVel
        return RotunbotVel.step(self, torch.zeros_like(actions))

    def compute_observations(self):
        self.depth_observation[:] = self.capture_depth()
        self.obs_buf[:] = build_direct_velocity_observation(
            self._proprioception(),
            self._goal_xy_robot(),
            self.previous_velocity_command,
            self.depth_observation,
            self.cfg.commands.maximum_goal_distance,
        )
        self.privileged_obs_buf[:] = torch.cat(
            (
                self._proprioception(),
                self._goal_xy_robot(),
                self.previous_velocity_command,
                self.obstacle_clearance.unsqueeze(1),
                self.step_collision_buf.float().unsqueeze(1),
            ),
            dim=1,
        )

    def check_termination(self):
        goal_distance = torch.linalg.vector_norm(
            self.global_goal_xy_world - self.root_states[:, :2], dim=1
        )
        self.goal_dist.copy_(goal_distance)
        self.goal_reached_buf[:] = goal_distance <= float(self.cfg.commands.goal_radius)
        self.terminal_goal_distance.copy_(goal_distance)
        self.time_out_buf[:] = self.episode_length_buf >= self.max_episode_length
        roll = torch.abs(self.base_euler_tensor[:, 0]) > 1.2
        pitch = torch.abs(self.base_euler_tensor[:, 1]) > 1.2
        out_of_bounds = (self.root_states[:, :2] - self.env_origins[:, :2]).abs().max(dim=1).values > 15.0
        self.success_buf[:] = self.goal_reached_buf
        self.reset_buf[:] = self.goal_reached_buf | self.time_out_buf | roll | pitch | out_of_bounds | self.step_collision_buf

    def _reward_goal_progress(self):
        current = torch.linalg.vector_norm(self.global_goal_xy_world - self.root_states[:, :2], dim=1)
        if not hasattr(self, "previous_goal_distance"):
            self.previous_goal_distance = current.clone()
        progress = self.previous_goal_distance - current
        self.previous_goal_distance.copy_(current)
        return progress

    def _reward_goal_reach(self):
        return self.goal_reached_buf.float()

    def _reward_collision(self):
        return self.step_collision_buf.float()

    def _reward_action_rate(self):
        return velocity_command_rate_penalty(
            self.previous_velocity_command, self.last_velocity_command
        )

    def _reward_goal_turn_alignment(self):
        return goal_turn_alignment(
            self._goal_xy_robot(), self.previous_velocity_command
        )

    def _reward_goal_speed_alignment(self):
        return goal_speed_alignment(
            self._goal_xy_robot(),
            self.previous_velocity_command,
            self.cfg.commands.max_forward_speed,
            self.cfg.commands.goal_radius,
            minimum_turn_radius=self.cfg.commands.minimum_turn_radius,
            recovery_active=self.goal_recovery_active,
        )

    def _reward_goal_kinematic_recovery(self):
        return goal_kinematic_recovery(
            self._goal_xy_robot(),
            self.previous_velocity_command,
            self.cfg.commands.minimum_turn_radius,
            recovery_active=self.goal_recovery_active,
        )

    def reset_idx(self, env_ids):
        if len(env_ids):
            self.terminal_applied_feasible_command[env_ids] = (
                self.applied_feasible_command[env_ids]
            )
            self.terminal_tracking_velocity[env_ids, 0] = self.tracking_lin_vel[
                env_ids, 0
            ]
            self.terminal_tracking_velocity[env_ids, 1] = self.tracking_ang_vel[
                env_ids, 2
            ]
            self.terminal_position[env_ids] = self.root_states[env_ids, :2]
            self.terminal_command_target[env_ids] = self.command_targets[env_ids]
            self.terminal_goal_xy_robot[env_ids] = self._goal_xy_robot()[env_ids]
            self.terminal_transition_active[env_ids] = self.transition_active[env_ids]
            self.terminal_transition_state[env_ids] = self.transition_state[env_ids]
            self.terminal_goal_recovery_active[env_ids] = self.goal_recovery_active[
                env_ids
            ]
            self.terminal_success[env_ids] = self.success_buf[env_ids]
            self.terminal_collision[env_ids] = self.step_collision_buf[env_ids]
            self.terminal_timeout[env_ids] = self.time_out_buf[env_ids]
            terminal_success = self.success_buf[env_ids].detach().float().mean()
            terminal_collision = self.step_collision_buf[env_ids].detach().float().mean()
            terminal_timeout = self.time_out_buf[env_ids].detach().float().mean()
            terminal_goal_distance = self.terminal_goal_distance[env_ids].detach().mean()
            terminal_speed = torch.linalg.vector_norm(
                self.tracking_lin_vel[env_ids, :2].detach(), dim=1
            ).mean()
        else:
            terminal_success = terminal_collision = terminal_timeout = None
            terminal_goal_distance = terminal_speed = None
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.extras["episode"].update(
            {
                "success": terminal_success,
                "collision": terminal_collision,
                "timeout": terminal_timeout,
                "terminal_goal_distance": terminal_goal_distance,
                "terminal_speed": terminal_speed,
            }
        )
        self.previous_velocity_command[env_ids] = 0.0
        self.last_velocity_command[env_ids] = 0.0
        self.goal_recovery_active[env_ids] = False
        self.goal_recovery_activation_count[env_ids] = 0
        self.previous_goal_distance[env_ids] = torch.linalg.vector_norm(
            self.global_goal_xy_world[env_ids] - self.root_states[env_ids, :2], dim=1
        )
