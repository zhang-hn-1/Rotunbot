"""Rotunbot maze with frame-stacked repro-style obs + wall-ray sensing (SRU).

The legacy maze/obstacle environment exposes a single 19-D Euler-based frame
with no wall information.  This subclass builds the **paper-reproduction
frame layout** (quaternion, LH obs scales -- exactly what the accepted
uniform-4150 DWL policy was trained on), appends 16 body-frame wall-ray
distances (grid ray-march, 8 m max), then stacks ``frame_stack`` frames.

Frame layout (35-D per frame):
  0:2 command_xy, 2:4 position_xy, 4:8 quaternion, 8:11 linear velocity,
  11:14 angular velocity, 14 dof_pos, 15:17 dof_vel, 17:19 prev actions,
  19:35 wall-ray distances (16 rays, 22.5 deg spacing, body frame, 0-8 m).

The first 19 channels are byte-compatible with the flat-plane policy's
observation, so 方案 B (ActorCriticSRUModulate) can use uniform 4150 as its
frozen base on a 19-D slice while the SRU modulator consumes the full 35-D
frame including wall rays.
"""

import collections
import math

import numpy as np
import torch

from .rotunbot_maze import RotunbotMaze
from .rotunbot_maze_sru_config import RotunbotMazeSRUCfg


class RotunbotMazeSRU(RotunbotMaze):
    """Maze task whose actor observation is a frame-stacked history + rays."""

    cfg: RotunbotMazeSRUCfg

    def _init_buffers(self):
        super()._init_buffers()
        # Critic sees the same (stacked) observations as the actor.
        self.num_privileged_obs = int(self.cfg.env.num_observations)
        self.privileged_obs_buf = torch.zeros(
            self.num_envs, self.num_privileged_obs,
            device=self.device, dtype=torch.float,
        )
        self.num_single_obs = int(self.cfg.env.num_single_obs)
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
        self._init_goal_curriculum()

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
        """Build repro-style 19-D frame + 16 wall rays, then stack history."""
        # No noise here: the frame below is built from raw state like the
        # paper-reproduction env; the maze adds rays deterministically.
        frame = torch.cat(
            (
                self.commands[:, :2] * self.obs_scales.command,
                self.root_states[:, :2] * self.obs_scales.pos,
                self.base_quat * self.obs_scales.quat,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.dof_pos[:, 1].unsqueeze(1) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        rays = self._wall_ray_distances()
        frame = torch.cat((frame, rays), dim=-1)
        self.obs_history.append(frame.clone())
        self.obs_buf = torch.stack(list(self.obs_history), dim=1).reshape(
            self.num_envs, -1
        )
        self.privileged_obs_buf[:] = self.obs_buf

    def reset_idx(self, env_ids):
        terminal_success = None
        if len(env_ids) > 0 and hasattr(self, "success_buf"):
            terminal_success = self.success_buf[env_ids].detach().clone()
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            for frame in self.obs_history:
                frame[env_ids] = 0.0
        if terminal_success is not None:
            self._update_goal_curriculum(terminal_success)

    def check_termination(self):
        super().check_termination()
        # The obstacle env computes arrived/stop but never publishes
        # success_buf; the curriculum and evaluation need it.
        self.success_buf = self.arrived_target_buf & self.stop_buf

    def step(self, actions):
        # Clip to the frozen base policy's output range ([-3, 3] on the flat
        # task).  The observation carries the previous RAW action in channels
        # 17:18; without this clip an out-of-distribution output snowballs
        # through the feedback (observed up to +/-70) and the base policy
        # extrapolates badly in the maze.
        actions = torch.clip(actions, -3.0, 3.0)
        return super().step(actions)

    # -- target-distance curriculum (near -> far) -----------------------------

    def _init_goal_curriculum(self):
        self.curriculum_radius = float(
            getattr(self.cfg.commands, "curriculum_goal_radius_start", 4.0)
        )
        self._goal_curriculum_successes = 0
        self._goal_curriculum_attempts = 0

    def _update_goal_curriculum(self, terminal_success):
        if not bool(getattr(self.cfg.commands, "target_curriculum", False)):
            return
        window = int(getattr(self.cfg.commands, "target_curriculum_window", 2048))
        threshold = float(
            getattr(self.cfg.commands, "target_curriculum_success_rate", 0.70)
        )
        step_radius = float(getattr(self.cfg.commands, "curriculum_goal_radius_step", 2.0))
        max_radius = float(getattr(self.cfg.commands, "curriculum_goal_radius_max", 20.0))
        self._goal_curriculum_attempts += int(terminal_success.numel())
        self._goal_curriculum_successes += int(terminal_success.sum().item())
        if self._goal_curriculum_attempts < window:
            return
        rate = self._goal_curriculum_successes / max(self._goal_curriculum_attempts, 1)
        print(
            f"[maze curriculum] window success rate={rate:.2%} "
            f"(threshold {threshold:.0%}) radius={self.curriculum_radius:.1f}",
            flush=True,
        )
        if rate >= threshold and self.curriculum_radius < max_radius:
            self.curriculum_radius = min(max_radius, self.curriculum_radius + step_radius)
            print(
                f"[maze curriculum] success {rate:.2%} >= {threshold:.0%}; "
                f"goal radius -> {self.curriculum_radius:.1f} m",
                flush=True,
            )
        self._goal_curriculum_successes = 0
        self._goal_curriculum_attempts = 0

    def _resample_commands(self, env_ids):
        """Sample goals within the current curriculum radius (near -> far)."""
        super()._resample_commands(env_ids)
        if not bool(getattr(self.cfg.commands, "target_curriculum", False)):
            return
        if len(env_ids) == 0:
            return
        # Keep targets whose distance from the center spawn is within radius.
        goal_positions = self._maze_goal_positions  # numpy [N, 2]
        radii = np.linalg.norm(goal_positions, axis=1)
        allowed = np.flatnonzero(radii <= self.curriculum_radius)
        if allowed.size == 0:
            allowed = np.flatnonzero(radii <= self.curriculum_radius + 4.0)
        count = len(env_ids)
        indices = self._goal_rng.integers(0, allowed.size, size=count)
        chosen = goal_positions[allowed[indices]]
        self.commands[env_ids, :2] = (
            torch.as_tensor(chosen, dtype=self.commands.dtype, device=self.device)
            + self.env_origins[env_ids, :2]
        )

    # -- reward: silent version of the obstacle to_target (no per-step print) --

    def _reward_to_target(self):
        pos_error = torch.sum(
            torch.square(self.commands[:, :2] - self.root_states[:, :2]), dim=1
        )
        return torch.exp(-pos_error / self.cfg.rewards.tracking_sigma_main)

    def _reward_time(self):
        # Constant per-step penalty so wandering without reaching the target
        # costs (the obstacle env has no _reward_time).
        return 1.0

    def _reward_stop(self):
        # Reward the STRICT success condition (arrived AND stopped).  The
        # inherited obstacle version rewards proximity only (50 * (goal_dist
        # < stop_distance)), so the policy learned to hover near targets and
        # collect reward without ever stopping (window success rate 3% while
        # rew_stop reached 0.39).
        return 50.0 * self.success_buf.float()

    def _reward_near_goal_speed(self):
        # Brake shaping: inside the brake gate, penalize excess speed so the
        # policy stops within the 0.20 m success radius (evaluation showed the
        # robot passes through the target but never brakes).
        brake_distance = float(
            getattr(self.cfg.rewards, "near_goal_brake_distance", 0.50)
        )
        near_goal = (self.goal_dist <= brake_distance).float()
        speed = torch.linalg.norm(self.base_lin_vel, dim=1)
        stop_velocity = float(
            getattr(getattr(self.cfg, "evaluation", None), "stop_velocity_threshold", 0.1)
            if hasattr(self.cfg, "evaluation") else 0.1
        )
        excess_speed = torch.relu(speed - stop_velocity)
        return near_goal * torch.square(excess_speed)
