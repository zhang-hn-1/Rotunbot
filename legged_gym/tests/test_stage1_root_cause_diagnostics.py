"""Pure contracts for the Stage 1.1 root-cause diagnostics."""

import unittest

import torch

from legged_gym.navigation.v49_waypoint_diagnostics import (
    DiagnosticMode,
    apply_diagnostic_projection,
    detect_low_speed_yaw_collapse,
    dynamic_transition_severity,
    rate_feedforward_active_ratio,
    summarize_command_transitions,
    summarize_terminal_results,
    yaw_sign_reversal_count,
)


class Stage1RootCauseDiagnosticsTests(unittest.TestCase):
    def test_baseline_mode_preserves_stage1_defaults(self):
        mode = DiagnosticMode()
        self.assertFalse(mode.smooth_reference)
        self.assertIsNone(mode.minimum_rolling_speed)

    def test_smooth_mode_only_changes_reference_flag(self):
        raw = torch.tensor([[0.12, 0.02]])
        baseline = apply_diagnostic_projection(raw, DiagnosticMode())
        smooth = apply_diagnostic_projection(
            raw, DiagnosticMode(smooth_reference=True)
        )
        self.assertTrue(torch.equal(baseline, smooth))

    def test_disabled_rolling_floor_preserves_command(self):
        raw = torch.tensor([[0.04, 0.0]])
        result = apply_diagnostic_projection(raw, DiagnosticMode())
        self.assertTrue(torch.equal(result, raw))

    def test_rolling_floor_only_raises_low_forward_speed(self):
        raw = torch.tensor([[0.04, 0.01], [-0.04, -0.01], [0.12, 0.01]])
        result = apply_diagnostic_projection(
            raw, DiagnosticMode(minimum_rolling_speed=0.10)
        )
        self.assertTrue(torch.allclose(result[:, 0], torch.tensor([0.10, -0.10, 0.12])))
        self.assertTrue(torch.equal(result[:, 1], raw[:, 1]))
        self.assertLessEqual(float(result[0, 1]), 0.27 * float(result[0, 0]) + 1e-6)

    def test_all_diagnostic_commands_remain_in_static_v49_domain(self):
        result = apply_diagnostic_projection(
            torch.tensor([[0.04, 0.10]]),
            DiagnosticMode(minimum_rolling_speed=0.10),
        )
        self.assertLessEqual(float(result[0, 0]), 0.13)
        self.assertLessEqual(float(result[0, 1]), 0.27 * float(result[0, 0]) + 1e-6)

    def test_transition_statistics_and_yaw_reversals(self):
        stats = summarize_command_transitions(
            torch.tensor([0.0, 0.007, 0.024, 0.031]),
            torch.tensor([0.0, 0.003, -0.004, 0.008]),
        )
        self.assertAlmostEqual(stats["max_abs_delta_v"], 0.017, places=6)
        self.assertAlmostEqual(stats["fraction_abs_delta_v_gt_0.008"], 1.0 / 3.0)
        self.assertAlmostEqual(stats["fraction_abs_delta_w_gt_0.004"], 2.0 / 3.0)
        self.assertEqual(
            yaw_sign_reversal_count(
                torch.tensor([0.02, 0.02, -0.02, 0.0]),
                torch.tensor([-0.02, 0.03, 0.02, -0.02]),
            ),
            2,
        )

    def test_low_speed_collapse_requires_duration_and_bearing(self):
        ticks = [
            {"measured_v": 0.05, "bearing_error": 0.30},
            {"measured_v": 0.05, "bearing_error": 0.29},
            {"measured_v": 0.05, "bearing_error": 0.28},
        ]
        self.assertTrue(detect_low_speed_yaw_collapse(ticks, tick_period_s=0.2))
        ticks[-1]["bearing_error"] = 0.20
        self.assertFalse(detect_low_speed_yaw_collapse(ticks, tick_period_s=0.2))

    def test_rate_feedforward_activity_uses_epsilon(self):
        self.assertAlmostEqual(
            rate_feedforward_active_ratio(
                torch.tensor([0.0, 1.0e-5, 0.0]),
                torch.tensor([0.0, 0.0, -2.0e-5]),
            ),
            2.0 / 3.0,
        )

    def test_dynamic_transition_severity_uses_training_increments(self):
        self.assertAlmostEqual(dynamic_transition_severity(0.016, 0.002), 2.0)

    def test_terminal_statistics_exclude_incomplete_routes(self):
        summary = summarize_terminal_results([
            {"route_complete": False, "terminal_speed_safe": False, "stop_distance": None},
            {"route_complete": True, "terminal_speed_safe": True, "stop_distance": 0.10},
            {"route_complete": True, "terminal_speed_safe": False, "stop_distance": 0.20},
        ])
        self.assertEqual(summary["route_complete_count"], 2)
        self.assertEqual(summary["route_incomplete_count"], 1)
        self.assertEqual(summary["completed_terminal_speed_safe_count"], 1)
        self.assertEqual(summary["completed_terminal_speed_failure_count"], 1)
        self.assertAlmostEqual(summary["mean_stop_distance"], 0.15)


if __name__ == "__main__":
    unittest.main()
