"""Rotunbot maze with frame-stacked observations + wall-ray sensing (SRU).

The legacy maze/obstacle environment exposes a single 19-D frame with Euler
angles and no wall information, so a policy cannot navigate the maze.  This
subclass appends 16 body-frame wall-ray distances (grid ray-march, 8 m max)
to each frame, then stacks ``frame_stack`` frames exactly like the
paper-reproduction env.  The flattened observation becomes
``frame_stack * (19 + 16)`` and the SRU memory encoder scans it.

Frame layout (35-D, Euler-based):
  0:2 command_xy, 2:5 position_xyz, 5:8 euler_xyz, 8:11 linear velocity,
  11:14 angular velocity, 14 dof_pos, 15:17 dof_vel, 17:19 previous actions,
  19:35 wall-ray distances (16 rays, 22.5 deg spacing, body frame, 0-8 m).
"""

import collections
import math

import torch

from .rotunbot_maze import RotunbotMaze
from .rotunbot_maze_sru_config import RotunbotMazeSRUCfg


class RotunbotMazeSRU(RotunbotMaze):
    """Maze task whose actor observation is a frame-stacked history + rays."""

    cfg: RotunbotMazeSRUCfg

    def _init_buffers(self):
        super()._init_buffers()
        # Critic uses the same (stacked) observations; no privileged channel.
        self.num_privileged_obs = None
        self.privileged_obs_buf = None
        self.num_short_obs = int(
            self.cfg.env.num_single_obs * self.cfg.env.short_frame_stack
        )
        # Wall occupancy grid (1 = wall) in maze-local coordinates.
        self.maze_grid = torch.as_tensor(
            self.maze_layout, dtype=torch.long, device=self.device
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

    # -- wall-ray sensing -----------------------------------------------------

    def _wall_ray_distances(self, num_rays=16, max_range=8.0, step=0.1):
        """Body-frame wall-ray distances via grid ray-march (vectorized)."""
        num_envs = self.num_envs
        local = self.root_states[:, :2] - self.env_origins[:, :2]  # (N,2)
        yaw = self.base_euler_tensor[:, 2]
        angles = (
            torch.arange(num_rays, device=self.device, dtype=torch.float)
            * (2.0 * math.pi / num_rays)
        )
        theta = yaw.unsqueeze(1) + angles.unsqueeze(0)  # (N,K)
        directions = torch.stack(
            (torch.cos(theta), torch.sin(theta)), dim=-1
        )  # (N,K,2)
        position = local.unsqueeze(1).expand(num_envs, num_rays, 2).clone()
        grid = self.maze_grid  # (G,G)
        grid_size = int(grid.shape[0])
        half = grid_size / 2.0
        cell = float(self.cfg.maze.cell_size)
        distances = torch.full(
            (num_envs, num_rays), float(max_range), device=self.device
        )
        steps = int(max_range / step)
        for t in range(steps):
            position = position + directions * step
            ix = torch.floor(position[..., 0] / cell + half).long()
            iy = torch.floor(position[..., 1] / cell + half).long()
            inside = (
                (ix >= 0) & (ix < grid_size) & (iy >= 0) & (iy < grid_size)
            )
            hit = torch.zeros(
                (num_envs, num_rays), dtype=torch.bool, device=self.device
            )
            valid = inside.nonzero(as_tuple=False)
            if valid.numel() > 0:
                hit[valid[:, 0], valid[:, 1]] = (
                    grid[ix[inside], iy[inside]] == 1
                )
            new_hit = hit & (distances >= float(max_range))
            distances[new_hit] = (t + 1) * step
            if bool(new_hit.all()):
                break
        return distances

    # -- observation ----------------------------------------------------------

    def compute_observations(self):
        # Parent sets the 19-D single frame (with noise when configured).
        super().compute_observations()
        rays = self._wall_ray_distances()
        frame = torch.cat((self.obs_buf, rays), dim=-1)
        self.obs_history.append(frame.clone())
        self.obs_buf = torch.stack(list(self.obs_history), dim=1).reshape(
            self.num_envs, -1
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            for frame in self.obs_history:
                frame[env_ids] = 0.0

    # -- reward: silent version of the obstacle to_target (no per-step print) --

    def _reward_to_target(self):
        pos_error = torch.sum(
            torch.square(self.commands[:, :2] - self.root_states[:, :2]), dim=1
        )
        return torch.exp(-pos_error / self.cfg.rewards.tracking_sigma_main)
