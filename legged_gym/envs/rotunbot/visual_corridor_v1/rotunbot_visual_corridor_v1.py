"""V1 straight visual corridor with direct SRU velocity control."""

import math

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.navigation.v1_curriculum import V1PerformanceCurriculum
from legged_gym.navigation.visual_corridor_v1 import v1_curriculum_goal_distance
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

    def _init_buffers(self):
        super()._init_buffers()
        self.v1_curriculum = V1PerformanceCurriculum(
            seed=int(getattr(self.cfg.commands, "v1_curriculum_seed", 4))
        )
        self.v1_goal_sampling_kind = ["fixed"] * self.num_envs

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
        distance = float(self.cfg.corridor_length_m)
        performance_curriculum = bool(
            getattr(self.cfg.commands, "v1_performance_curriculum_enabled", False)
        )
        if performance_curriculum:
            distances, kinds = self.v1_curriculum.sample_distances(len(env_ids))
            distance_tensor = torch.as_tensor(
                distances, dtype=torch.float32, device=self.device
            )
            goal[:, 0] += distance_tensor
            env_id_values = (
                env_ids.detach().cpu().tolist()
                if hasattr(env_ids, "detach")
                else list(env_ids)
            )
            for env_id, kind in zip(env_id_values, kinds):
                self.v1_goal_sampling_kind[env_id] = kind
        elif bool(getattr(self.cfg.commands, "v1_goal_curriculum_enabled", False)):
            distance = v1_curriculum_goal_distance(
                self.common_step_counter,
                self.cfg.commands.v1_curriculum_start_distance,
                self.cfg.corridor_length_m,
                self.cfg.commands.v1_curriculum_horizon_steps,
            )
            goal[:, 0] += distance
        else:
            goal[:, 0] += distance
        self.global_goal_xy_world[env_ids] = goal
        self.goal_dist[env_ids] = torch.linalg.vector_norm(
            goal - self.root_states[env_ids, :2], dim=1
        )
        self.previous_goal_distance[env_ids] = self.goal_dist[env_ids]
        self.commands[env_ids, :2] = 0.0

    def _reward_goal_progress(self):
        """Reward Euclidean progress toward the currently sampled goal."""
        current = torch.linalg.vector_norm(
            self.global_goal_xy_world - self.root_states[:, :2], dim=1
        )
        progress = self.previous_goal_distance - current
        self.previous_goal_distance.copy_(current)
        return progress

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
            self.previous_goal_distance[env_ids] = self.goal_dist[env_ids]

    def get_checkpoint_state(self):
        """Persist only the curriculum state; network and V62 remain unchanged."""
        if not hasattr(self, "v1_curriculum"):
            return None
        return {
            "v1_performance_curriculum": self.v1_curriculum.to_dict(),
        }

    def set_checkpoint_state(self, state):
        """Restore V1 distance-curriculum position when resuming training."""
        if not state or "v1_performance_curriculum" not in state:
            return
        self.v1_curriculum = V1PerformanceCurriculum.from_dict(
            state["v1_performance_curriculum"]
        )
