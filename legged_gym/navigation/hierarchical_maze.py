"""Maze scene composition for evaluating the unchanged frozen P2P skill."""

import numpy as np

from legged_gym.envs.rotunbot.maze.rotunbot_maze import RotunbotMaze
from legged_gym.envs.rotunbot.maze.rotunbot_maze_config import RotunbotMazeCfg
from legged_gym.envs.rotunbot.target_point.rotunbot_target_obstacle import (
    RotunbotTargetObstacle,
)
from legged_gym.envs.rotunbot.target_point.rotunbot_target_repro import (
    RotunbotTargetRepro,
)
from legged_gym.envs.rotunbot.target_point.rotunbot_target_repro_config import (
    RotunbotTargetReproCfg,
)


class HierarchicalMazeCfg(RotunbotTargetReproCfg):
    """Reproduction observation/control config plus the existing maze scene."""

    class env(RotunbotTargetReproCfg.env):
        num_envs = 1
        episode_length_s = 120.0

    class maze:
        grid_size = tuple(RotunbotMazeCfg.maze.grid_size)
        cell_size = float(RotunbotMazeCfg.maze.cell_size)
        wall_height = float(RotunbotMazeCfg.maze.wall_height)
        center_clearance_radius = int(RotunbotMazeCfg.maze.center_clearance_radius)
        min_goal_distance = float(RotunbotMazeCfg.maze.min_goal_distance)
        seed = int(RotunbotMazeCfg.maze.seed)
        wall_color = tuple(RotunbotMazeCfg.maze.wall_color)
        robot_collision_radius = float(RotunbotMazeCfg.maze.robot_collision_radius)
        terminate_on_collision = True

    class commands(RotunbotTargetReproCfg.commands):
        class ranges(RotunbotTargetReproCfg.commands.ranges):
            pos_x = [-12.0, 12.0]
            pos_y = [-12.0, 12.0]


class HierarchicalMazeP2P(RotunbotTargetRepro):
    """Reuse the existing maze actors while retaining Repro P2P observations."""

    cfg: HierarchicalMazeCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.maze_layout = None
        self.mazes = []
        self._goal_rng = np.random.default_rng(int(cfg.maze.seed) + 1)
        self._maze_goal_positions = None
        self._maze_wall_centers = None
        self._intermediate_goal = False
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _create_scene_assets(self):
        # The map and wall construction remain the existing RotunbotMaze
        # implementation; only the P2P-compatible class owns the lifecycle.
        return RotunbotMaze._create_scene_assets(self)

    def _create_envs(self):
        # The existing Maze scene hooks are wired into the obstacle task's
        # environment lifecycle.  Reuse that lifecycle here while retaining
        # this class's Repro P2P observation/control inheritance.
        return RotunbotTargetObstacle._create_envs(self)

    def _create_scene_actors(self, env_handle, env_id, scene_assets):
        return RotunbotMaze._create_scene_actors(self, env_handle, env_id, scene_assets)

    def _init_buffers(self):
        # Keep the exact Repro/LH buffers and observation history, then append
        # only maze collision state required by the Oracle evaluator.
        from collections import deque
        import torch

        # Walls add actors to the simulator state tensor.  The existing
        # obstacle lifecycle correctly exposes actor_root_state[:, 0] as the
        # robot state; reuse it instead of duplicating that tensor plumbing.
        RotunbotTargetObstacle._init_buffers(self)

        # These are the additions made by the frozen Repro task after the
        # shared LH/obstacle buffers have been initialized.
        self.output_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device
        )
        self.last_output_actions = torch.zeros_like(self.output_actions)
        self.success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.arrived_target_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs,
                dtype=torch.float, device=self.device
            ))
        for _ in range(self.cfg.env.c_frame_stack):
            privileged_dim = self.cfg.env.single_num_privileged_obs
            if self.cfg.terrain.measure_heights:
                privileged_dim += self.cfg.terrain.num_height
            self.critic_history.append(torch.zeros(
                self.num_envs, privileged_dim, dtype=torch.float, device=self.device
            ))

        latency_cfg = getattr(self.cfg, "latency", None)
        self.direct_gains = torch.full(
            (self.num_envs,),
            float(getattr(self.cfg.control, "direct_velocity_gain", 35.0)),
            device=self.device,
        )
        self.latency_enabled = bool(
            latency_cfg is not None and getattr(latency_cfg, "enabled", False)
        )
        self.max_observation_delay_steps = int(
            getattr(latency_cfg, "max_observation_steps", 0)
        ) if latency_cfg is not None else 0
        self.max_action_delay_steps = int(
            getattr(latency_cfg, "max_action_steps", 0)
        ) if latency_cfg is not None else 0
        self.observation_delay_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.action_delay_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.observation_delay_buffer = torch.zeros(
            self.max_observation_delay_steps + 1,
            self.num_envs,
            self.cfg.env.num_single_obs,
            device=self.device,
        )
        self.action_delay_buffer = torch.zeros(
            self.max_action_delay_steps + 1,
            self.num_envs,
            self.num_actions,
            device=self.device,
        )
        self.observation_delay_write_index = 0
        self.action_delay_write_index = 0
        self.observation_delay_needs_init = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.terminal_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.terminal_speed = torch.zeros(self.num_envs, device=self.device)
        self.terminal_balance_reward = torch.zeros(self.num_envs, device=self.device)
        self.terminal_position = torch.zeros(self.num_envs, 2, device=self.device)
        self.terminal_yaw = torch.zeros(self.num_envs, device=self.device)
        self.terminal_timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terminal_unstable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terminal_out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.training_success_distance = float(
            getattr(
                self.cfg.commands,
                "curriculum_success_distance_start",
                self.cfg.evaluation.target_error_threshold,
            )
        )
        self.target_curriculum_successes = 0
        self.target_curriculum_attempts = 0
        self.target_curriculum_last_success_rate = 0.0

        self.maze_wall_centers = torch.as_tensor(
            self._maze_wall_centers, dtype=torch.float32, device=self.device
        )
        self.maze_collision_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _reset_root_states(self, env_ids):
        """Reset only robot actors while preserving Repro's initial yaw rule."""
        import math
        import torch
        from isaacgym import gymtorch
        from isaacgym.torch_utils import torch_rand_float

        # The obstacle implementation handles the multi-actor root tensor and
        # indexed simulator update.  Apply the Repro yaw convention to the
        # robot slice afterwards, then update that same indexed tensor.
        RotunbotTargetObstacle._reset_root_states(self, env_ids)
        if len(env_ids) == 0 or not self.cfg.commands.random_start_yaw:
            return
        yaw = torch_rand_float(
            -math.pi, math.pi, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        half_yaw = 0.5 * yaw
        quat = torch.zeros(len(env_ids), 4, device=self.device)
        quat[:, 2] = torch.sin(half_yaw)
        quat[:, 3] = torch.cos(half_yaw)
        self.actor_root_state[env_ids, 0, 3:7] = quat
        robot_actor_indices = self.robot_actor_indices[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.actor_root_state),
            gymtorch.unwrap_tensor(robot_actor_indices),
            len(robot_actor_indices),
        )

    def _resample_commands(self, env_ids):
        return RotunbotMaze._resample_commands(self, env_ids)

    def _maze_collision_mask(self):
        return RotunbotMaze._maze_collision_mask(self)

    def set_intermediate_goal(self, enabled):
        self._intermediate_goal = bool(enabled)

    def check_termination(self):
        import torch

        super().check_termination()
        self.terminal_yaw[:] = torch.atan2(
            2.0 * (self.base_quat[:, 3] * self.base_quat[:, 2] + self.base_quat[:, 0] * self.base_quat[:, 1]),
            1.0 - 2.0 * (self.base_quat[:, 1] * self.base_quat[:, 1] + self.base_quat[:, 2] * self.base_quat[:, 2]),
        )
        self.maze_collision_buf = self._maze_collision_mask()
        if self.cfg.maze.terminate_on_collision:
            self.reset_buf |= self.maze_collision_buf
        if self._intermediate_goal:
            self.reset_buf &= ~self.success_buf
