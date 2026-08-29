import unittest

from legged_gym.navigation.v49_dynamic_reachability import (
    DynamicReachabilityTable,
    ReachabilityState,
)
from legged_gym.navigation.v49_stage1_3_diagnostics import (
    STAGE13_SUMMARY_FIELDS,
    STAGE13_TRACE_FIELDS,
    aggregate_stage13_trials,
    detect_direction_reversal,
    detect_command_direction_reversal,
    horizon_policy_step,
    symmetric_yaw_grid,
)


def _rows():
    rows = []
    for current_v in (0.06, 0.10):
        for command_v in (0.08, 0.12):
            for command_w in (-0.02, 0.02):
                rows.append({
                    "current_v": current_v,
                    "projected_v": command_v,
                    "projected_w": command_w,
                    "predicted_forward_velocity_50ms": current_v + command_v + command_w,
                    "predicted_yaw_rate_50ms": current_v - command_v + command_w,
                    "predicted_forward_velocity_100ms": current_v + command_v + command_w,
                    "predicted_yaw_rate_100ms": current_v - command_v + command_w,
                    "predicted_forward_velocity_150ms": current_v + command_v + command_w,
                    "predicted_yaw_rate_150ms": current_v - command_v + command_w,
                    "predicted_forward_velocity_200ms": current_v + command_v + command_w,
                    "predicted_yaw_rate_200ms": current_v - command_v + command_w,
                })
    return rows


class DynamicReachabilityTableTests(unittest.TestCase):
    def test_state_and_command_use_physical_v_w_units(self):
        state = ReachabilityState(
            current_forward_velocity=0.08,
            current_yaw_rate=-0.01,
        )
        self.assertEqual(state.current_v, 0.08)
        self.assertEqual(state.current_w, -0.01)
        self.assertEqual(state.command_units, ("m/s", "rad/s"))

    def test_exact_lookup_returns_measured_horizon_values(self):
        table = DynamicReachabilityTable.from_rows(_rows())
        prediction = table.predict_reachable_response(
            ReachabilityState(0.06), (0.08, 0.02)
        )
        self.assertFalse(prediction.out_of_coverage)
        self.assertEqual(prediction.projected_command, (0.08, 0.02))
        self.assertAlmostEqual(prediction.predicted_forward_velocity_200ms, 0.16)
        self.assertAlmostEqual(prediction.predicted_yaw_rate_200ms, 0.0)

    def test_trilinear_interpolation_is_deterministic(self):
        table = DynamicReachabilityTable.from_rows(_rows())
        prediction = table.predict_reachable_response(
            ReachabilityState(0.08), (0.10, 0.0)
        )
        self.assertAlmostEqual(prediction.predicted_forward_velocity_200ms, 0.18)
        self.assertAlmostEqual(prediction.predicted_yaw_rate_200ms, -0.02)
        self.assertEqual(prediction.coverage, "interpolated")

    def test_out_of_coverage_is_clamped_and_marked(self):
        table = DynamicReachabilityTable.from_rows(_rows())
        prediction = table.predict_reachable_response(
            ReachabilityState(0.20), (0.20, 0.10)
        )
        self.assertTrue(prediction.out_of_coverage)
        self.assertEqual(prediction.coverage, "clamped")
        self.assertEqual(prediction.projected_command, (0.12, 0.02))

    def test_symmetric_yaw_commands_preserve_signed_response(self):
        table = DynamicReachabilityTable.from_rows(_rows())
        positive = table.predict_reachable_response(ReachabilityState(0.06), (0.08, 0.02))
        negative = table.predict_reachable_response(ReachabilityState(0.06), (0.08, -0.02))
        self.assertAlmostEqual(positive.predicted_yaw_rate_200ms, 0.0)
        self.assertAlmostEqual(negative.predicted_yaw_rate_200ms, -0.04)

    def test_stage13_horizon_alignment_matches_fifty_hz_policy(self):
        self.assertEqual(horizon_policy_step(50, 0.02), 3)
        self.assertEqual(horizon_policy_step(100, 0.02), 5)
        self.assertEqual(horizon_policy_step(150, 0.02), 8)
        self.assertEqual(horizon_policy_step(200, 0.02), 10)

    def test_stage13_yaw_grid_is_symmetric_and_includes_zero(self):
        self.assertEqual(symmetric_yaw_grid(0.04, 0.02), (-0.04, -0.02, 0.0, 0.02, 0.04))

    def test_stage13_direction_reversal_is_detected(self):
        self.assertTrue(detect_direction_reversal(0.10, (0.09, -0.01, 0.07)))
        self.assertFalse(detect_direction_reversal(0.10, (0.10, 0.11, 0.12)))
        self.assertTrue(detect_command_direction_reversal(-0.10, (0.0, 0.04, 0.12)))
        self.assertFalse(detect_command_direction_reversal(0.10, (0.0, 0.04, 0.12)))

    def test_stage13_aggregation_has_five_horizon_summary(self):
        rows = []
        for repeat in range(3):
            rows.append({
                "initial_forward_velocity": 0.08, "initial_yaw_rate": 0.0,
                "forward_velocity_command": 0.10, "yaw_rate_command": 0.02,
                "projected_forward_velocity": 0.10, "projected_yaw_rate": 0.02,
                "actual_v_50ms": 0.08, "actual_w_50ms": 0.002,
                "actual_v_100ms": 0.085, "actual_w_100ms": 0.006,
                "actual_v_150ms": 0.09, "actual_w_150ms": 0.01,
                "actual_v_200ms": 0.095, "actual_w_200ms": 0.014,
                "cumulative_yaw_change_200ms": 0.001,
                "body_displacement_x_200ms": 0.018,
                "body_displacement_y_200ms": 0.001,
                "forward_direction_reversed": False,
                "yaw_direction_reversed": False,
                "simulation_unstable": False,
            })
        result = aggregate_stage13_trials(rows)
        self.assertEqual(result["repeat_count"], 3)
        self.assertAlmostEqual(result["mean_actual_w_200ms"], 0.014)
        self.assertEqual(result["simulation_instability_count"], 0)

    def test_stage13_trace_schema_contains_required_fields(self):
        for name in (
            "initial_forward_velocity", "initial_yaw_rate",
            "forward_velocity_command", "yaw_rate_command", "simulation_dt",
            "control_dt", "seed", "trial_id",
        ):
            self.assertIn(name, STAGE13_TRACE_FIELDS)
        for name in (
            "actual_v_50ms", "actual_w_50ms", "actual_v_100ms",
            "actual_w_150ms", "actual_v_200ms", "cumulative_yaw_change_200ms",
            "body_displacement_x_200ms",
        ):
            self.assertIn(name, STAGE13_SUMMARY_FIELDS)


if __name__ == "__main__":
    unittest.main()
