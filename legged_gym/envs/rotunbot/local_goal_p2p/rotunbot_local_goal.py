"""Explicit Robot-frame Local P2P environment."""

import math

import torch
from isaacgym.torch_utils import torch_rand_float

from ..target_point.rotunbot_target_repro import RotunbotTargetRepro
from .local_goal_utils import build_local_observation, world_to_robot_xy
from .rotunbot_local_goal_config import RotunbotLocalGoalCfg


class RotunbotLocalGoal(RotunbotTargetRepro):
    """Single-target, single-frame local-goal task on an empty plane."""

    cfg: RotunbotLocalGoalCfg

    def _init_buffers(self):
        super()._init_buffers()
        self.world_goal = torch.zeros(self.num_envs, 2, device=self.device)
        self.local_goal = torch.zeros(self.num_envs, 2, device=self.device)
        self.raw_actions = torch.zeros_like(self.actions)
        self.clip_count = torch.zeros(1, dtype=torch.long, device=self.device)
        self.action_count = torch.zeros(1, dtype=torch.long, device=self.device)
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.arrived_target_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _get_noise_scale_vec(self, cfg):
        self.add_noise = bool(cfg.noise.add_noise)
        return torch.zeros(cfg.env.num_single_obs, device=self.device)

    def _sampling_ranges(self):
        stage = str(self.cfg.commands.local_curriculum_stage).upper()
        if stage == "A":
            return self.cfg.commands.stage_a
        if stage == "B":
            return self.cfg.commands.stage_b
        if stage == "C":
            return self.cfg.commands.stage_c
        raise ValueError(f"unknown local curriculum stage: {stage}")

    def _yaw_from_quaternion(self):
        qx, qy, qz, qw = self.base_quat.unbind(dim=1)
        return torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy.square() + qz.square()),
        )

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        ranges = self._sampling_ranges()
        distances = torch_rand_float(
            ranges.distance[0],
            ranges.distance[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        bearings = torch_rand_float(
            math.radians(ranges.bearing_deg[0]),
            math.radians(ranges.bearing_deg[1]),
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        qx, qy, qz, qw = self.root_states[env_ids, 3:7].unbind(dim=1)
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy.square() + qz.square()),
        )
        local_delta = torch.stack(
            (distances * torch.cos(bearings), distances * torch.sin(bearings)), dim=1
        )
        c, s = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)
        world_delta = torch.cat(
            (c * local_delta[:, 0:1] - s * local_delta[:, 1:2],
             s * local_delta[:, 0:1] + c * local_delta[:, 1:2]),
            dim=1,
        )
        self.world_goal[env_ids] = self.root_states[env_ids, :2] + world_delta
        self.commands[env_ids, :2] = self.world_goal[env_ids]

    def _update_local_goal(self):
        world_delta = self.world_goal - self.root_states[:, :2]
        self.local_goal[:] = world_to_robot_xy(world_delta, self._yaw_from_quaternion())
        self.goal_dist[:] = torch.linalg.norm(world_delta, dim=1)

    def _post_physics_step_callback(self):
        self._update_base_euler()
        self._update_local_goal()

    def compute_observations(self):
        self._update_local_goal()
        obs_now = build_local_observation(
            self.local_goal,
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.dof_pos,
            self.dof_vel,
            self.last_actions,
            float(self.cfg.commands.local_goal_max_distance),
        )
        if self.add_noise:
            obs_now = obs_now + (2.0 * torch.rand_like(obs_now) - 1.0) * self.noise_scale_vec
        self.obs_buf[:] = obs_now
        self.privileged_obs_buf[:] = obs_now

    def reset_idx(self, env_ids):
        terminal_success = self.success_buf[env_ids].detach().clone() if len(env_ids) else None
        super().reset_idx(env_ids)
        if terminal_success is not None and "episode" in self.extras:
            self.extras["episode"]["local_success"] = terminal_success.float().mean()
        if len(env_ids):
            self._update_local_goal()
            self.last_goal_dist[env_ids] = self.goal_dist[env_ids]

    def step(self, actions):
        self.raw_actions[:] = actions
        self.clip_count += torch.count_nonzero(torch.abs(actions) > 1.0).to(torch.long)
        self.action_count += actions.numel()
        return super().step(actions)

    def check_termination(self):
        self.time_out_buf[:] = self.episode_length_buf >= self.max_episode_length
        self.arrived_target_buf[:] = self.goal_dist <= float(self.cfg.commands.local_goal_radius)
        self.success_buf[:] = self.arrived_target_buf

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

        terminal = (
            self.time_out_buf
            | self.success_buf
            | roll_cutoff
            | pitch_cutoff
            | x_cutoff
            | y_cutoff
        )
        self.reset_buf[:] = terminal

    def _reward_local_progress(self):
        return self.last_goal_dist - self.goal_dist

    def _reward_local_reach(self):
        return self.success_buf.float()

    def _reward_local_time(self):
        return torch.ones_like(self.goal_dist)

    def _reward_local_action_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

