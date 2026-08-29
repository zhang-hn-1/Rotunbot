import unittest

from legged_gym.navigation.v49_stage1_2_diagnostics import (
    RAW_TRACE_FIELDS,
    REACHABILITY_GRID_FIELDS,
    compare_snapshot_sequences,
    direction_agreement,
    dynamic_response_ratio,
    high_level_alignment,
    reset_audit_rows,
    response_reachable,
    summarize_reachability_samples,
    velocity_bin,
)


class Stage12DiagnosticsTests(unittest.TestCase):
    def test_50hz_alignment_uses_ten_policy_steps_per_high_level_tick(self):
        self.assertEqual(high_level_alignment(0, 10), (0, 0, 0))
        self.assertEqual(high_level_alignment(9, 10), (0, 9, 0.18))
        self.assertEqual(high_level_alignment(10, 10), (1, 0, 0.20))

    def test_prefix_comparator_reports_first_variable_and_step(self):
        reference = [
            {"root": [0.0, 0.0], "observation": [1.0, 2.0]},
            {"root": [0.1, 0.0], "observation": [1.1, 2.0]},
        ]
        candidate = [
            {"root": [0.0, 0.0], "observation": [1.0, 2.0]},
            {"root": [0.1, 0.0], "observation": [1.1, 2.1]},
        ]
        result = compare_snapshot_sequences(reference, candidate)
        self.assertFalse(result["equivalent"])
        self.assertEqual(result["first_divergence_policy_step"], 1)
        self.assertEqual(result["first_divergence_variable"], "observation")
        self.assertGreater(result["absolute_difference"], 0.0)

    def test_prefix_comparator_accepts_float_noise(self):
        reference = [{"v": [0.1, -0.2]}]
        candidate = [{"v": [0.1 + 1e-8, -0.2 - 1e-8]}]
        self.assertTrue(compare_snapshot_sequences(reference, candidate)["equivalent"])

    def test_reset_audit_reports_pass_fail_and_unavailable(self):
        specs = [
            {"name": "command_rates", "expected": [0.0, 0.0]},
            {"name": "missing_history", "expected": 0.0},
        ]
        rows = reset_audit_rows(
            specs,
            before={"command_rates": [0.3, -0.2]},
            after={"command_rates": [0.0, 0.0]},
        )
        self.assertEqual(rows[0]["status"], "PASS")
        self.assertEqual(rows[1]["availability"], "not_available")

    def test_response_ratio_handles_zero_target_delta(self):
        self.assertEqual(dynamic_response_ratio(0.1, 0.1, 0.1), None)
        self.assertAlmostEqual(dynamic_response_ratio(0.0, 0.1, 0.04), 0.4)

    def test_direction_agreement_uses_targeted_delta_and_zero_is_neutral(self):
        self.assertTrue(direction_agreement(0.02, 0.01))
        self.assertFalse(direction_agreement(-0.02, 0.01))
        self.assertTrue(direction_agreement(0.0, 0.0))

    def test_response_reachable_requires_direction_and_twenty_percent_response(self):
        self.assertTrue(response_reachable(0.10, 0.03))
        self.assertFalse(response_reachable(0.10, -0.03))
        self.assertFalse(response_reachable(0.10, 0.01))
        self.assertTrue(response_reachable(0.0, 0.0))

    def test_velocity_bins_match_stage1_low_speed_boundaries(self):
        self.assertEqual(velocity_bin(0.06), "lt_0.08")
        self.assertEqual(velocity_bin(0.09), "0.08_to_0.10")
        self.assertEqual(velocity_bin(0.12), "ge_0.10")

    def test_reachability_summary_reports_20_100_200ms_and_schema(self):
        samples = [
            {
                "initial_v": 0.08,
                "initial_w": 0.0,
                "target_v": 0.10,
                "target_w": 0.02,
                "projected_v": 0.10,
                "projected_w": 0.02,
                "actual_v_20ms": 0.081,
                "actual_w_20ms": 0.002,
                "actual_v_100ms": 0.09,
                "actual_w_100ms": 0.012,
                "actual_v_200ms": 0.099,
                "actual_w_200ms": 0.019,
                "v_tracking_tolerance": 0.02,
                "w_tracking_tolerance": 0.01,
                "forward_sign_correct": True,
                "yaw_sign_correct": True,
            }
        ]
        summary = summarize_reachability_samples(samples)
        self.assertEqual(summary["repeat_count"], 1)
        self.assertAlmostEqual(summary["mean_actual_v_200ms"], 0.099)
        self.assertAlmostEqual(summary["mean_actual_w_200ms"], 0.019)
        self.assertEqual(summary["response_reachable_rate"], 1.0)
        self.assertEqual(summary["tracking_reachable_rate"], 1.0)

    def test_trace_and_grid_schemas_include_required_fields(self):
        for name in (
            "episode_id", "trial_id", "policy_step", "high_level_tick",
            "step_within_high_level_tick", "desired_v", "desired_w",
            "projected_v", "projected_w", "measured_v", "measured_w",
            "delta_command_v", "delta_command_w", "root_linear_velocity_body_0",
            "root_angular_velocity_body_2", "nominal_action_0", "final_action_1",
            "low_speed_lt_010", "low_speed_lt_008",
        ):
            self.assertIn(name, RAW_TRACE_FIELDS)
        for name in (
            "initial_v_bin", "target_v", "target_w", "repeat_count",
            "response_reachable_rate", "tracking_reachable_rate",
            "forward_sign_failure_rate", "yaw_sign_failure_rate",
        ):
            self.assertIn(name, REACHABILITY_GRID_FIELDS)


if __name__ == "__main__":
    unittest.main()
