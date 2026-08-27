import types
import unittest

import numpy as np

if not hasattr(np, "float"):
    np.float = float
import isaacgym
import torch

from legged_gym.envs.rotunbot.maze.rotunbot_maze_local_depth import RotunbotMazeLocalDepth
from legged_gym.envs.rotunbot.maze.rotunbot_maze_local_depth_config import RotunbotMazeLocalDepthCfg


def fake_env(stage=0):
    env = object.__new__(RotunbotMazeLocalDepth)
    env.num_envs = 2
    env.device = torch.device("cpu")
    env.cfg = types.SimpleNamespace(
        commands=types.SimpleNamespace(
            local_waypoint_radius=0.25,
            global_goal_radius=0.35,
            distance_limit=(0.25, 2.0),
            lateral_limit=0.8,
            minimum_forward_component=0.15,
            bearing_limit_deg=120.0,
            local_curriculum_stage=stage,
        ),
        maze=types.SimpleNamespace(robot_collision_radius=0.4, terminate_on_collision=True, safety_clearance=0.8),
        camera=types.SimpleNamespace(far_plane=8.0),
    )
    env.root_states = torch.zeros(2, 13)
    env.env_origins = torch.zeros(2, 3)
    env.base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    env.base_euler_tensor = torch.zeros(2, 3)
    env.global_goal_xy_world = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    env.active_local_goal_xy_world = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    env.active_local_goal_xy_robot = env.active_local_goal_xy_world.clone()
    env.prev_local_goal_dist = torch.tensor([1.2, 0.8])
    env.waypoint_changed = torch.zeros(2, dtype=torch.bool)
    env.waypoint_reached = torch.zeros(2, dtype=torch.bool)
    env.global_goal_reached = torch.zeros(2, dtype=torch.bool)
    env.maze_collision_buf = torch.zeros(2, dtype=torch.bool)
    env.obstacle_clearance = torch.tensor([0.5, 2.0])
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.max_episode_length = 100
    env.success_buf = torch.zeros(2, dtype=torch.bool)
    env.reset_buf = torch.zeros(2, dtype=torch.long)
    env.time_out_buf = torch.zeros(2, dtype=torch.bool)
    env.base_lin_vel = torch.zeros(2, 3)
    env.actions = torch.zeros(2, 2)
    env.last_actions = torch.zeros(2, 2)
    return env


class DepthLocalRewardTests(unittest.TestCase):
    def test_progress_sign_and_switch_zero(self):
        env = fake_env()
        env.active_local_goal_xy_robot = torch.tensor([[0.5, 0.0], [1.5, 0.0]])
        self.assertTrue(torch.allclose(env._reward_local_progress(), torch.tensor([0.7, -0.7])))
        env.waypoint_changed[:] = True
        self.assertTrue(torch.equal(env._reward_local_progress(), torch.zeros(2)))

    def test_wall_penalty_is_finite_and_has_safety_cutoff(self):
        env = fake_env()
        penalty = env._reward_wall_penalty()
        self.assertTrue(torch.isfinite(penalty).all())
        self.assertLess(float(penalty[0]), 0.0)
        self.assertEqual(float(penalty[1]), 0.0)

    def test_negative_total_is_not_clipped_by_configuration(self):
        cfg = RotunbotMazeLocalDepthCfg()
        total = -0.25 * 0.1
        self.assertFalse(cfg.rewards.only_positive_rewards)
        self.assertLess(total, 0.0)


if __name__ == "__main__":
    unittest.main()
