import unittest

from legged_gym.navigation.v49_dynamic_governor_diagnostics import (
    STAGE14_SCENARIOS,
    aggregate_governor_rows,
    count_command_oscillations,
)


class DynamicGovernorDiagnosticsTests(unittest.TestCase):
    def test_ten_scenarios_have_physical_command_sequences(self):
        self.assertEqual(len(STAGE14_SCENARIOS), 10)
        for scenario in STAGE14_SCENARIOS:
            self.assertTrue(scenario.name)
            self.assertIn(scenario.group, ("low_speed", "high_speed", "mixed"))
            self.assertGreaterEqual(len(scenario.commands), 2)
            for command in scenario.commands:
                self.assertEqual(len(command), 2)
                self.assertTrue(all(isinstance(value, float) for value in command))

    def test_metrics_include_percentiles_and_governor_counters(self):
        rows = [
            {"mode": "Baseline", "scenario": "x", "trial": 0,
             "v_error": 0.1, "w_error": 0.2, "command_modified": False,
             "static_saturated": True, "fallback": False, "selected_yaw": 0.1},
            {"mode": "Baseline", "scenario": "x", "trial": 0,
             "v_error": 0.3, "w_error": 0.4, "command_modified": False,
             "static_saturated": False, "fallback": False, "selected_yaw": -0.1},
        ]
        result = aggregate_governor_rows(rows)
        metrics = result["Baseline"]
        self.assertAlmostEqual(metrics["v_error_mean"], 0.2)
        self.assertAlmostEqual(metrics["w_error_median"], 0.3)
        self.assertIn("w_error_p90", metrics)
        self.assertEqual(metrics["static_saturation_count"], 1)
        self.assertEqual(metrics["command_modification_count"], 0)
        self.assertEqual(metrics["fallback_count"], 0)

    def test_oscillation_counts_alternating_signed_commands(self):
        self.assertEqual(count_command_oscillations((0.0, 0.1, -0.1, 0.1)), 2)
        self.assertEqual(count_command_oscillations((0.0, 0.1, 0.2, 0.0)), 0)


if __name__ == "__main__":
    unittest.main()
