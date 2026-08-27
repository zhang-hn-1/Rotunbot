import unittest

import numpy as np

if not hasattr(np, "float"):
    np.float = float
import isaacgym
import torch

from legged_gym.envs.rotunbot.maze.rotunbot_maze_local_depth import (
    RotunbotMazeLocalDepth,
    build_depth_local_observation,
)
from legged_gym.envs.rotunbot.maze.rotunbot_maze_local_depth_config import RotunbotMazeLocalDepthCfg


class DepthLocalObservationTests(unittest.TestCase):
    def test_registry_contract_and_layout(self):
        cfg = RotunbotMazeLocalDepthCfg()
        self.assertEqual(cfg.env.num_observations, 272)
        self.assertEqual(cfg.env.num_single_obs, 272)
        self.assertEqual(cfg.env.num_short_obs, 272)
        self.assertEqual(cfg.env.frame_stack, 1)
        self.assertEqual(cfg.env.num_privileged_obs, 18)
        n = 2
        obs = build_depth_local_observation(
            torch.zeros(n, 3), torch.ones(n, 3), torch.ones(n, 3) * 2,
            torch.ones(n, 1) * 3, torch.ones(n, 2) * 4,
            torch.ones(n, 2) * 5, torch.ones(n, 2) * 6, torch.ones(n, 8, 32) * 7,
        )
        self.assertEqual(tuple(obs.shape), (2, 272))
        self.assertTrue(torch.allclose(obs[:, 0:3], torch.zeros(n, 3)))
        self.assertTrue(torch.allclose(obs[:, 9:10], torch.ones(n, 1) * 3))
        self.assertTrue(torch.allclose(obs[:, 12:14], torch.ones(n, 2) * 5 / 8.0))
        self.assertTrue(torch.allclose(obs[:, 14:16], torch.ones(n, 2) * 6))
        self.assertTrue(torch.allclose(obs[:, 16:], torch.ones(n, 8, 32).reshape(n, -1) * 7))

    def test_actor_observation_has_no_world_xy_or_yaw_slots(self):
        self.assertNotIn("root_states[:, :2]", RotunbotMazeLocalDepth.compute_observations.__code__.co_names)
        self.assertEqual(RotunbotMazeLocalDepthCfg.env.state_dim, 16)


if __name__ == "__main__":
    unittest.main()
