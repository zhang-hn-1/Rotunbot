import unittest

from legged_gym.navigation.phase_d_diagnostics import (
    classify_constant_command_gate,
    command_loss_breakdown,
)


class PhaseDDiagnosticsTests(unittest.TestCase):
    def test_command_loss_breakdown_reports_three_ratios(self):
        rows = [
            {
                "desired_v_raw_mps": 0.30,
                "desired_v_projected_mps": 0.25,
                "command_target_v_mps": 0.25,
                "applied_v_mps": 0.20,
                "actual_v_mps": 0.15,
                "desired_w_raw_rps": 0.01,
                "actual_w_rps": 0.005,
                "transition_state": 0,
                "global_goal_distance_m": 2.0,
            },
            {
                "desired_v_raw_mps": 0.30,
                "desired_v_projected_mps": 0.25,
                "command_target_v_mps": 0.25,
                "applied_v_mps": 0.20,
                "actual_v_mps": 0.15,
                "desired_w_raw_rps": 0.01,
                "actual_w_rps": 0.005,
                "transition_state": 0,
                "global_goal_distance_m": 1.0,
            },
        ]
        result = command_loss_breakdown(rows)
        self.assertAlmostEqual(result["r_projection"], 0.25 / 0.30)
        self.assertAlmostEqual(result["r_transition"], 0.20 / 0.25)
        self.assertAlmostEqual(result["r_tracking"], 0.15 / 0.20)
        self.assertEqual(result["transition_state_counts"], {"0": 2})
        self.assertAlmostEqual(result["mean_goal_progress_m"], 1.0)

    def test_empty_breakdown_is_explicit(self):
        result = command_loss_breakdown([])
        self.assertIsNone(result["r_projection"])
        self.assertEqual(result["sample_count"], 0)

    def test_command_loss_separates_second_projection_from_transition(self):
        result = command_loss_breakdown([
            {
                "desired_v_raw_mps": 0.24,
                "desired_v_projected_mps": 0.24,
                "command_target_v_mps": 0.12,
                "applied_v_mps": 0.12,
                "actual_v_mps": 0.12,
                "transition_state": 0,
            }
        ])
        self.assertAlmostEqual(result["r_target_projection"], 0.5)
        self.assertAlmostEqual(result["r_transition"], 1.0)
        self.assertAlmostEqual(result["r_tracking"], 1.0)

    def test_constant_gate_requires_tracking_evidence(self):
        self.assertEqual(
            classify_constant_command_gate(
                {"actual_over_applied": 0.8, "p90_tracking_error_v_mps": 0.04}
            ),
            "PASS",
        )
        self.assertEqual(
            classify_constant_command_gate(
                {"actual_over_applied": 0.2, "p90_tracking_error_v_mps": 0.04}
            ),
            "FAIL",
        )


if __name__ == "__main__":
    unittest.main()
