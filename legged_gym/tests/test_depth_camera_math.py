import unittest

import numpy as np

if not hasattr(np, "float"):
    np.float = float
import isaacgym
import torch

from legged_gym.envs.rotunbot.maze.rotunbot_maze_camera import (
    DepthCameraMixin,
    normalize_depth_image,
)


class _CameraCfg:
    near_plane = 0.05
    far_plane = 8.0
    horizontal_fov = 105.0
    depth_height = 8
    depth_width = 32
    position = (0.0, 0.0, 0.0)
    add_noise = False
    noise_std = 0.0
    dropout_probability = 0.0
    quantization = 0.0


class _EnvCfg:
    camera = _CameraCfg()
    class env:
        depth_height = 8
        depth_width = 32


class _FallbackEnv(DepthCameraMixin):
    def __init__(self):
        self.cfg = _EnvCfg()
        self.device = torch.device("cpu")
        self.num_envs = 1
        self.root_states = torch.zeros(1, 13)
        self.env_origins = torch.zeros(1, 3)
        self.base_euler_tensor = torch.zeros(1, 3)
        self.depth_backend_requested = "fallback"
        self.depth_backend_actual = "fallback"

    def _get_depth_fallback_aabbs(self):
        centers = torch.tensor([[3.0, -1.0], [3.0, 1.0]])
        half_extents = torch.tensor([[0.2, 0.2], [0.2, 0.2]])
        return centers, half_extents


class DepthCameraMathTests(unittest.TestCase):
    def test_normalization_clamps_and_fills_invalid_values(self):
        raw = torch.tensor([[-1.0, 0.05, 4.025, 9.0, float("nan")]])
        normalized = normalize_depth_image(raw, near=0.05, far=8.0)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)
        self.assertEqual(float(normalized[0, 0]), 1.0)
        self.assertEqual(float(normalized[0, 4]), 1.0)

    def test_symmetric_corridor_is_symmetric(self):
        env = _FallbackEnv()
        centers = torch.tensor([[3.0, -1.0], [3.0, 1.0]])
        half_extents = torch.tensor([[0.05, 0.05], [0.05, 0.05]])
        env._get_depth_fallback_aabbs = lambda: (centers, half_extents)
        depth = env._fallback_depth()
        self.assertEqual(tuple(depth.shape), (1, 8, 32))
        self.assertTrue(torch.allclose(depth[0, 0], depth[0, 0].flip(0), atol=1e-5))

    def test_geometry_provider_and_backend_labels_are_explicit(self):
        env = _FallbackEnv()
        provided = env._get_depth_fallback_aabbs()
        self.assertEqual(tuple(provided[0].shape), (2, 2))
        self.assertEqual(tuple(provided[1].shape), (2, 2))
        self.assertEqual(env.depth_backend_requested, "fallback")
        self.assertEqual(env.depth_backend_actual, "fallback")


if __name__ == "__main__":
    unittest.main()
