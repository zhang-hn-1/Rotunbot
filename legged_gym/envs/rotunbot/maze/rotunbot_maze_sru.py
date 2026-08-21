"""Rotunbot maze with frame-stacked observations for the SRU policy.

The legacy maze/obstacle environment exposes a single 19-D frame with Euler
angles.  For the SRU memory encoder (which scans a window of frames) this
subclass stacks ``frame_stack`` frames exactly like the paper-reproduction
env, so the flattened observation becomes ``frame_stack * num_single_obs``
and the DWL/SRU runners can consume it unchanged.

Frame layout (19-D, Euler-based):
  0:2 command_xy, 2:5 position_xyz, 5:8 euler_xyz, 8:11 linear velocity,
  11:14 angular velocity, 14 dof_pos, 15:17 dof_vel, 17:19 previous actions.
"""

import collections

import torch

from .rotunbot_maze import RotunbotMaze
from .rotunbot_maze_sru_config import RotunbotMazeSRUCfg


class RotunbotMazeSRU(RotunbotMaze):
    """Maze task whose actor observation is a frame-stacked history."""

    cfg: RotunbotMazeSRUCfg

    def _init_buffers(self):
        super()._init_buffers()
        # Critic uses the same (stacked) observations; no privileged channel.
        self.num_privileged_obs = None
        self.privileged_obs_buf = None
        self.num_short_obs = int(
            self.cfg.env.num_single_obs * self.cfg.env.short_frame_stack
        )
        self.obs_history = collections.deque(
            [
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    device=self.device,
                    dtype=torch.float,
                )
                for _ in range(self.cfg.env.frame_stack)
            ],
            maxlen=self.cfg.env.frame_stack,
        )

    def compute_observations(self):
        # Parent sets the 19-D single frame (with noise when configured).
        super().compute_observations()
        self.obs_history.append(self.obs_buf.clone())
        self.obs_buf = torch.stack(list(self.obs_history), dim=1).reshape(
            self.num_envs, -1
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            for frame in self.obs_history:
                frame[env_ids] = 0.0
