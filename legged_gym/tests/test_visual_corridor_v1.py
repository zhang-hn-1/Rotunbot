import math
import unittest

from legged_gym.navigation.visual_corridor_v1 import (
    V1_CORRIDOR_LENGTH_M,
    V1_CORRIDOR_WIDTH_M,
    build_v1_straight_geometry,
)


class VisualCorridorV1Tests(unittest.TestCase):
    def test_v1_geometry_has_two_local_side_walls_and_expected_extent(self):
        segments, obstacles = build_v1_straight_geometry(
            width_m=V1_CORRIDOR_WIDTH_M,
            length_m=V1_CORRIDOR_LENGTH_M,
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(len(obstacles), 2)
        self.assertAlmostEqual(segments[0][0][0], 0.0)
        self.assertAlmostEqual(segments[0][1][0], V1_CORRIDOR_LENGTH_M)
        centers = sorted(float(item[0][1]) for item in obstacles)
        self.assertAlmostEqual(centers[0], -V1_CORRIDOR_WIDTH_M / 2.0)
        self.assertAlmostEqual(centers[1], V1_CORRIDOR_WIDTH_M / 2.0)

    def test_v1_task_contract_is_direct_velocity_and_randomizes_only_start_pose(self):
        from legged_gym.envs import task_registry

        cfg = task_registry.get_cfgs("rotunbot_sru_visual_corridor_v1")[0]

        self.assertEqual(cfg.env.num_actions, 2)
        self.assertEqual(cfg.env.num_observations, 273)
        self.assertEqual(cfg.env.num_privileged_obs, 19)
        self.assertEqual(cfg.visual_stage, "V1")
        self.assertAlmostEqual(cfg.corridor_width_m, 2.0)
        self.assertAlmostEqual(cfg.corridor_length_m, 6.0)
        self.assertAlmostEqual(cfg.init_state.random_start_lateral, 0.30)
        self.assertAlmostEqual(cfg.init_state.random_start_yaw, math.radians(10.0))
        self.assertFalse(cfg.maze.enabled)
        self.assertEqual(cfg.camera.depth_backend, "fallback")

    def test_v1_path_progress_is_available_without_oracle_observation(self):
        from legged_gym.envs.rotunbot.visual_corridor_v1.rotunbot_visual_corridor_v1 import (
            RotunbotVisualCorridorV1,
        )

        self.assertFalse(hasattr(RotunbotVisualCorridorV1, "oracle_waypoint"))
        self.assertTrue(hasattr(RotunbotVisualCorridorV1, "_reward_path_progress"))


if __name__ == "__main__":
    unittest.main()
