import unittest

from legged_gym.navigation.straight_exit_gate import (
    build_straight_exit_gate,
    summarize_reverse_diagnostics,
)


class StraightExitGateTests(unittest.TestCase):
    def test_reverse_diagnostics_reports_runs_magnitude_and_goal_regions(self):
        rows = [
            {"episode_id": 0, "macro_step": 0, "v_cmd": 0.10, "goal_distance": 2.0},
            {"episode_id": 0, "macro_step": 1, "v_cmd": -0.02, "goal_distance": 1.8},
            {"episode_id": 0, "macro_step": 2, "v_cmd": -0.04, "goal_distance": 1.9},
            {"episode_id": 0, "macro_step": 3, "v_cmd": 0.10, "goal_distance": 1.7},
            {"episode_id": 1, "macro_step": 0, "v_cmd": -0.01, "goal_distance": 0.3},
        ]
        result = summarize_reverse_diagnostics(
            rows,
            initial_goal_distance_by_episode={0: 2.0, 1: 1.0},
            collision_episodes=set(),
            timeout_episodes={1},
            dt=0.2,
        )
        self.assertEqual(result["reverse_step_count"], 3)
        self.assertEqual(result["reverse_episode_count"], 2)
        self.assertEqual(result["max_consecutive_reverse_steps"], 2)
        self.assertAlmostEqual(result["negative_v_p95"], -0.01)
        self.assertAlmostEqual(result["max_reverse_duration_sec"], 0.4)
        self.assertEqual(result["reverse_with_timeout_count"], 1)
        self.assertGreater(result["reverse_near_goal_ratio"], 0.0)

    def test_reverse_diagnostics_uses_policy_command_when_applied_command_was_projected(self):
        rows = [
            {"episode_id": 0, "macro_step": 0, "raw_v_cmd": -0.04, "v_cmd": 0.0, "goal_distance": 1.0},
        ]
        result = summarize_reverse_diagnostics(rows, {0: 1.0})
        self.assertEqual(result["reverse_step_count"], 1)

    def test_straight_exit_requires_overall_success_and_rejects_systematic_reverse(self):
        passing = {
            "1.0": {"episodes": 20, "success_count": 20, "collision_count": 0, "timeout_count": 0, "depth_backend_actual": "isaacgym",
                     "reverse_diagnostics": {"sustained_high_speed_reverse": False}},
            "1.5": {"episodes": 20, "success_count": 18, "collision_count": 0, "timeout_count": 1, "depth_backend_actual": "isaacgym",
                     "reverse_diagnostics": {"sustained_high_speed_reverse": False}},
            "2.0": {"episodes": 20, "success_count": 17, "collision_count": 0, "timeout_count": 1, "depth_backend_actual": "isaacgym",
                     "reverse_diagnostics": {"sustained_high_speed_reverse": False}},
            "2.5": {"episodes": 20, "success_count": 17, "collision_count": 0, "timeout_count": 1, "depth_backend_actual": "isaacgym",
                     "reverse_diagnostics": {"sustained_high_speed_reverse": False}},
        }
        result = build_straight_exit_gate(passing, "abc")
        self.assertEqual(result["status"], "PASS")
        self.assertAlmostEqual(result["overall_success_rate"], 0.9)
        passing["2.5"]["reverse_diagnostics"]["sustained_high_speed_reverse"] = True
        self.assertEqual(build_straight_exit_gate(passing, "abc")["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
