"""V1 straight visual corridor with direct SRU velocity control."""

import math

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

from ..direct_velocity.rotunbot_direct_velocity import RotunbotDirectVelocity
from .rotunbot_visual_corridor_v1_config import RotunbotVisualCorridorV1Cfg


class RotunbotVisualCorridorV1(RotunbotDirectVelocity):
    """Train Depth to correct randomized lateral and yaw starts in a corridor."""

    cfg: RotunbotVisualCorridorV1Cfg

    def _path_remaining(self):
        local_position = self.root_states[:, :2] - self.env_origins[:, :2]
        return torch.clamp(
            float(self.cfg.corridor_length_m) - local_position[:, 0], min=0.0
        )

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        if len(env_ids) == 0:
            return
        lateral = torch_rand_float(
            -float(self.cfg.init_state.random_start_lateral),
            float(self.cfg.init_state.random_start_lateral),
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        yaw = torch_rand_float(
            -float(self.cfg.init_state.random_start_yaw),
            float(self.cfg.init_state.random_start_yaw),
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.root_states[env_ids, 0] = self.env_origins[env_ids, 0]
        self.root_states[env_ids, 1] = self.env_origins[env_ids, 1] + lateral
        self.root_states[env_ids, 3] = 0.0
        self.root_states[env_ids, 4] = 0.0
        self.root_states[env_ids, 5] = torch.sin(0.5 * yaw)
        self.root_states[env_ids, 6] = torch.cos(0.5 * yaw)
        actor_ids = self._robot_actor_ids(env_ids).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._all_root_states),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        goal = self.env_origins[env_ids, :2].clone()
        goal[:, 0] += float(self.cfg.corridor_length_m)
        self.global_goal_xy_world[env_ids] = goal
        self.goal_dist[env_ids] = torch.linalg.vector_norm(
            goal - self.root_states[env_ids, :2], dim=1
        )
        self.previous_goal_distance[env_ids] = self._path_remaining()[env_ids]
        self.commands[env_ids, :2] = 0.0

    def _reward_path_progress(self):
        current = self._path_remaining()
        progress = self.previous_goal_distance - current
        self.previous_goal_distance.copy_(current)
        return progress

    def _reward_wall_clearance(self):
        clearance = self.obstacle_clearance
        safety_clearance = float(self.cfg.maze.safety_clearance)
        return torch.relu(torch.full_like(clearance, safety_clearance) - clearance)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids):
            self.previous_goal_distance[env_ids] = self._path_remaining()[env_ids]
