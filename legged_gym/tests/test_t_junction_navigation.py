import math
import unittest

import numpy as np

from legged_gym.navigation.v1_t_junction import (
    build_t_junction_geometry,
    classify_t_branch,
)
from legged_gym.navigation.v1_t_junction_metrics import aggregate_t_gate


class TJunctionNavigationTests(unittest.TestCase):
    def test_left_and_right_share_walls_but_mirror_goals(self):
        left = build_t_junction_geometry("T_LEFT")
        right = build_t_junction_geometry("T_RIGHT")

        self.assertEqual(left.scenario.start_xy.tolist(), right.scenario.start_xy.tolist())
        np.testing.assert_allclose(left.wall_segments, right.wall_segments)
        np.testing.assert_allclose(left.obstacle_aabbs, right.obstacle_aabbs)
        np.testing.assert_allclose(left.waypoints[:, 0], right.waypoints[:, 0])
        np.testing.assert_allclose(left.waypoints[:, 1], -right.waypoints[:, 1])
        self.assertEqual(left.branch_direction, 1)
        self.assertEqual(right.branch_direction, -1)

    def test_geometry_has_fixed_dimensions_and_finite_values(self):
        geometry = build_t_junction_geometry("left")
        self.assertEqual(geometry.scenario.width_m, 3.0)
        self.assertEqual(geometry.waypoints.shape, (3, 2))
        self.assertEqual(geometry.reach_radius_m, 0.35)
        self.assertTrue(np.isfinite(geometry.waypoints).all())
        self.assertTrue(np.isfinite(np.asarray(geometry.wall_segments)).all())
        self.assertTrue(np.isfinite(np.asarray(geometry.obstacle_aabbs)).all())

    def test_branch_classifier_has_deadband(self):
        self.assertEqual(classify_t_branch((2.5, 1.0)), "LEFT")
        self.assertEqual(classify_t_branch((2.5, -1.0)), "RIGHT")
        self.assertEqual(classify_t_branch((2.5, 0.1), deadband_m=0.35), "UNDECIDED")

    def test_geometry_rejects_nonfinite_or_nonpositive_inputs(self):
        for kwargs in (
            {"width_m": math.nan},
            {"stem_length_m": math.inf},
            {"branch_length_m": 0.0},
            {"reach_radius_m": -0.1},
        ):
            with self.assertRaises(ValueError):
                build_t_junction_geometry("left", **kwargs)

    def test_gate_aggregates_metrics_and_paired_consistency(self):
        records = [
            {"episode_id": "a", "success": 1, "collision": 0, "timeout": 0,
             "wrong_turn": 0, "turn_completion": 1, "exit": 1,
             "branch_prediction": "LEFT", "expected_branch": "LEFT",
             "depth_backend_actual": "isaacgym"},
            {"episode_id": "b", "success": 1, "collision": 0, "timeout": 0,
             "wrong_turn": 0, "turn_completion": 1, "exit": 1,
             "branch_prediction": "RIGHT", "expected_branch": "RIGHT",
             "depth_backend_actual": "isaacgym"},
        ]
        result = aggregate_t_gate(records, pairs=[("a", "b")], ablations={})
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["success_rate"], 1.0)
        self.assertEqual(result["collision_rate"], 0.0)
        self.assertEqual(result["timeout_rate"], 0.0)
        self.assertEqual(result["wrong_turn_rate"], 0.0)
        self.assertEqual(result["turn_completion_rate"], 1.0)
        self.assertEqual(result["exit_rate"], 1.0)
        self.assertEqual(result["branch_accuracy"], 1.0)
        self.assertEqual(result["goal_consistency_rate"], 1.0)
        self.assertTrue(result["checks"]["real_depth_backend"])
        self.assertTrue(result["pass"])
        self.assertTrue(result["finite"])

    def test_gate_fails_threshold_and_backend_assertions(self):
        record = {"episode_id": "bad", "success": 0, "collision": 1,
                  "timeout": 0, "wrong_turn": 1, "turn_completion": 0,
                  "exit": 0, "branch_prediction": "UNDECIDED",
                  "expected_branch": "LEFT", "depth_backend_actual": "fallback"}
        result = aggregate_t_gate([record], pairs=[], ablations={})
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["real_depth_backend"])
        self.assertFalse(result["checks"]["success_rate_ge_0.80"])


if __name__ == "__main__":
    unittest.main()
