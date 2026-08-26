"""Tests for pure Local P2P evaluation aggregation."""

import unittest

from legged_gym.local_goal_metrics import aggregate_local_goal_records
from legged_gym.scripts.evaluate_local_goal_p2p import evaluation_grid


class LocalGoalEvaluationTests(unittest.TestCase):
    def test_stage_evaluation_grids_match_curriculum_ranges(self):
        self.assertEqual(
            evaluation_grid("A"), ((0.5, 1.0, 1.5, 2.0), (0.0, 45.0, -45.0))
        )
        self.assertEqual(
            evaluation_grid("B"),
            ((0.5, 1.0, 1.5, 2.0, 2.5), (0.0, 45.0, 90.0, -45.0, -90.0)),
        )
        self.assertEqual(
            evaluation_grid("C"),
            ((0.5, 1.0, 1.5, 2.0, 3.0), (0.0, 45.0, 90.0, -45.0, -90.0, 180.0)),
        )

    def test_aggregate_reports_rates_and_yaw_gap(self):
        records = [
            {"success": True, "divergent": False, "timeout": False, "near_miss": False, "min_distance": 0.2, "final_distance": 0.2, "steps": 10, "clip_ratio": 0.1, "yaw_deg": 0.0},
            {"success": False, "divergent": False, "timeout": True, "near_miss": True, "min_distance": 0.4, "final_distance": 0.5, "steps": 20, "clip_ratio": 0.3, "yaw_deg": 0.0},
            {"success": True, "divergent": False, "timeout": False, "near_miss": False, "min_distance": 0.2, "final_distance": 0.2, "steps": 12, "clip_ratio": 0.2, "yaw_deg": 90.0},
            {"success": False, "divergent": True, "timeout": True, "near_miss": False, "min_distance": 1.2, "final_distance": 1.3, "steps": 20, "clip_ratio": 0.4, "yaw_deg": 90.0},
        ]

        summary = aggregate_local_goal_records(records)

        self.assertEqual(summary["episodes"], 4)
        self.assertEqual(summary["success"], 2)
        self.assertEqual(summary["divergence"], 1)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertAlmostEqual(summary["divergence_rate"], 0.25)
        self.assertAlmostEqual(summary["mean_steps"], 15.5)
        self.assertAlmostEqual(summary["yaw_success_gap"], 0.0)
        self.assertAlmostEqual(summary["yaw_groups"]["0"]["success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
