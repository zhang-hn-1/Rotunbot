"""P2P fine-tuning task for robot-frame local waypoints on an empty plane."""

import math

import torch
from isaacgym.torch_utils import torch_rand_float

from ..target_point.rotunbot_target_repro import RotunbotTargetRepro
from .rotunbot_local_p2p_config import RotunbotLocalP2PCfg


class RotunbotLocalP2P(RotunbotTargetRepro):
    cfg: RotunbotLocalP2PCfg

    def _init_buffers(self):
        super()._init_buffers()
        self.local_waypoint_radius = float(self.cfg.commands.local_waypoint_radius)
        # The inherited LH task stores success as float; this task uses the
        # terminal boolean as the single reach definition.
        self.success_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _local_sampling_ranges(self):
        stage = int(self.cfg.commands.local_curriculum_stage)
        if stage <= 1:
            return (
                self.cfg.commands.local_distance_stage_a,
                self.cfg.commands.local_bearing_stage_a,
            )
        if stage == 2:
            return (
                self.cfg.commands.local_distance_stage_b,
                self.cfg.commands.local_bearing_stage_b,
            )
        return (
            self.cfg.commands.local_distance_stage_c,
            self.cfg.commands.local_bearing_stage_c,
        )

    def _resample_commands(self, env_ids):
        """Sample one robot-frame waypoint and latch its world position."""
        if len(env_ids) == 0:
            return

        distance_range, bearing_range = self._local_sampling_ranges()
        distances = torch_rand_float(
            distance_range[0], distance_range[1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        bearings = torch_rand_float(
            math.radians(bearing_range[0]),
            math.radians(bearing_range[1]),
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        qx, qy, qz, qw = self.root_states[env_ids, 3:7].unbind(dim=1)
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy.square() + qz.square()),
        )
        local_x = distances * torch.cos(bearings)
        local_y = distances * torch.sin(bearings)
        world_x = torch.cos(yaw) * local_x - torch.sin(yaw) * local_y
        world_y = torch.sin(yaw) * local_x + torch.cos(yaw) * local_y
        self.commands[env_ids, 0] = self.root_states[env_ids, 0] + world_x
        self.commands[env_ids, 1] = self.root_states[env_ids, 1] + world_y
        self.commands[env_ids, 2] = 0.0

    def check_termination(self):
        """Terminate on local waypoint reach, without a stop-speed condition."""
        self.time_out_buf[:] = self.episode_length_buf >= self.max_episode_length

        reached = self.goal_dist <= self.local_waypoint_radius
        self.arrived_target_buf = reached
        self.stop_buf = torch.linalg.norm(self.base_lin_vel, dim=1) <= 0.1
        self.success_buf[:] = reached

        roll_cutoff = torch.abs(self.base_euler_tensor[:, 0]) > 1.2
        pitch_cutoff = torch.abs(self.base_euler_tensor[:, 1]) > 1.2
        x_cutoff = torch.abs(self.base_pos[:, 0]) > 10.0
        y_cutoff = torch.abs(self.base_pos[:, 1]) > 10.0

        self.terminal_goal_dist[:] = self.goal_dist
        self.terminal_speed[:] = torch.linalg.norm(self.base_lin_vel, dim=1)
        self.terminal_position[:] = self.root_states[:, :2]
        self.terminal_timeout[:] = self.time_out_buf
        self.terminal_unstable[:] = roll_cutoff | pitch_cutoff
        self.terminal_out_of_bounds[:] = x_cutoff | y_cutoff

        terminal = self.time_out_buf.bool() | self.success_buf | roll_cutoff | pitch_cutoff | x_cutoff | y_cutoff
        # BaseTask initializes reset_buf as int64; assign the boolean terminal
        # result rather than mixing it in-place with float success buffers.
        self.reset_buf[:] = terminal

    def _reward_local_progress(self):
        return self.last_goal_dist - self.goal_dist

    def _reward_local_reach(self):
        return self.success_buf.float()

    def _reward_local_time(self):
        return torch.ones_like(self.goal_dist)

    def _reward_local_action_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)
