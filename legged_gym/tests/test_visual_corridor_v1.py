import math
import unittest

from legged_gym.navigation.visual_corridor_v1 import (
    V1_CORRIDOR_LENGTH_M,
    V1_CORRIDOR_WIDTH_M,
    build_v1_straight_geometry,
    v1_curriculum_goal_distance,
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
        self.assertEqual(cfg.env.num_observations, 275)
        self.assertEqual(cfg.env.num_privileged_obs, 21)
        self.assertEqual(cfg.visual_stage, "V1")
        self.assertAlmostEqual(cfg.corridor_width_m, 2.0)
        self.assertAlmostEqual(cfg.corridor_length_m, 6.0)
        self.assertAlmostEqual(cfg.init_state.random_start_lateral, 0.30)
        self.assertAlmostEqual(cfg.init_state.random_start_yaw, math.radians(10.0))
        self.assertFalse(cfg.maze.enabled)
        self.assertEqual(cfg.camera.depth_backend, "fallback")
        self.assertFalse(cfg.commands.v1_performance_curriculum_enabled)

    def test_v1_uses_current_goal_progress_not_path_progress(self):
        from legged_gym.envs import task_registry

        cfg = task_registry.get_cfgs("rotunbot_sru_visual_corridor_v1")[0]
        self.assertAlmostEqual(cfg.rewards.scales.goal_progress, 20.0)
        self.assertAlmostEqual(cfg.rewards.scales.path_progress, 0.0)

    def test_v1_goal_progress_rewards_forward_and_lateral_correction(self):
        from legged_gym.envs.rotunbot.visual_corridor_v1.rotunbot_visual_corridor_v1 import (
            RotunbotVisualCorridorV1,
        )
        import torch

        env = object.__new__(RotunbotVisualCorridorV1)
        env.root_states = torch.tensor([[0.0, 0.0, 0.0]])
        env.global_goal_xy_world = torch.tensor([[2.0, 1.0]])
        env.previous_goal_distance = torch.linalg.vector_norm(
            env.global_goal_xy_world[:, :2] - env.root_states[:, :2], dim=1
        )
        env.root_states[0, 0] = 0.5
        forward_progress = env._reward_goal_progress()
        env.root_states[0, 1] = 0.5
        lateral_progress = env._reward_goal_progress()
        self.assertGreater(float(forward_progress[0]), 0.0)
        self.assertGreater(float(lateral_progress[0]), 0.0)

    def test_v1_path_progress_is_available_without_oracle_observation(self):
        from legged_gym.envs.rotunbot.visual_corridor_v1.rotunbot_visual_corridor_v1 import (
            RotunbotVisualCorridorV1,
        )

        self.assertFalse(hasattr(RotunbotVisualCorridorV1, "oracle_waypoint"))
        self.assertTrue(hasattr(RotunbotVisualCorridorV1, "_reward_path_progress"))

    def test_v1_curriculum_reaches_formal_corridor_length(self):
        self.assertAlmostEqual(
            v1_curriculum_goal_distance(0, 2.0, V1_CORRIDOR_LENGTH_M, 12000),
            2.0,
        )
        self.assertAlmostEqual(
            v1_curriculum_goal_distance(6000, 2.0, V1_CORRIDOR_LENGTH_M, 12000),
            4.0,
        )
        self.assertAlmostEqual(
            v1_curriculum_goal_distance(12000, 2.0, V1_CORRIDOR_LENGTH_M, 12000),
            V1_CORRIDOR_LENGTH_M,
        )
        self.assertAlmostEqual(
            v1_curriculum_goal_distance(20000, 2.0, V1_CORRIDOR_LENGTH_M, 12000),
            V1_CORRIDOR_LENGTH_M,
        )


if __name__ == "__main__":
    unittest.main()
