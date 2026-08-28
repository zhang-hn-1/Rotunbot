import unittest

from legged_gym.scripts.depth_local_diagnostics import (
    action_mapping_decision,
    policy_gy_metrics,
)


class DepthLocalDiagnosticMetricTests(unittest.TestCase):
    def test_policy_metrics_detects_lateral_response(self):
        records = [
            {"gx": 1.0, "gy": -0.6, "actor_mean_a0": 0.2, "actor_mean_a1": -0.3},
            {"gx": 1.0, "gy": -0.2, "actor_mean_a0": 0.2, "actor_mean_a1": -0.1},
            {"gx": 1.0, "gy": 0.2, "actor_mean_a0": 0.2, "actor_mean_a1": 0.1},
            {"gx": 1.0, "gy": 0.6, "actor_mean_a0": 0.2, "actor_mean_a1": 0.3},
        ]
        metrics = policy_gy_metrics(records)
        self.assertGreater(metrics["a1_response_span"], 0.5)
        self.assertEqual(metrics["sign_agreement_rate"], 1.0)
        self.assertGreater(metrics["pearson_gy_a1"], 0.99)
        self.assertGreater(metrics["spearman_gy_a1"], 0.99)
        self.assertLess(metrics["symmetry_error"], 1e-6)

    def test_policy_metrics_reports_forward_only_shortcut(self):
        records = [
            {"gx": 1.0, "gy": -0.6, "actor_mean_a0": 0.4, "actor_mean_a1": 0.02},
            {"gx": 1.0, "gy": 0.0, "actor_mean_a0": 0.4, "actor_mean_a1": 0.02},
            {"gx": 1.0, "gy": 0.6, "actor_mean_a0": 0.4, "actor_mean_a1": 0.02},
        ]
        metrics = policy_gy_metrics(records)
        self.assertAlmostEqual(metrics["a1_response_span"], 0.0)
        self.assertAlmostEqual(metrics["a0_response_span"], 0.0)
        self.assertEqual(metrics["sign_agreement_rate"], 0.5)

    def test_action_mapping_decision_uses_five_second_lateral_displacement(self):
        cases = [
            {"action1": -1.0, "duration_s": 5.0, "delta_body_y": -0.5, "final_body_vy": -0.1},
            {"action1": 1.0, "duration_s": 5.0, "delta_body_y": 0.5, "final_body_vy": 0.1},
        ]
        braking = [{"reversal": True, "stopped": True}]
        self.assertEqual(action_mapping_decision(cases, braking), "A-GOOD")

    def test_action_mapping_decision_rejects_tiny_lateral_motion(self):
        cases = [
            {"action1": -1.0, "duration_s": 5.0, "delta_body_y": -0.05, "final_body_vy": 0.0},
            {"action1": 1.0, "duration_s": 5.0, "delta_body_y": 0.05, "final_body_vy": 0.0},
        ]
        self.assertEqual(action_mapping_decision(cases, []), "A-FAIL")


if __name__ == "__main__":
    unittest.main()
