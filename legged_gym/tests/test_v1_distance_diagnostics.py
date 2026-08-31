import unittest


class V1DistanceDiagnosticTests(unittest.TestCase):
    def test_distance_grid_is_deterministic_and_inclusive(self):
        from legged_gym.navigation.v1_distance_diagnostics import distance_grid

        self.assertEqual(
            distance_grid(0.50, 6.00, 0.25),
            tuple(round(0.50 + 0.25 * i, 2) for i in range(23)),
        )

    def test_scan_row_contains_raw_mapped_and_projected_command_fields(self):
        from legged_gym.navigation.v1_distance_diagnostics import scan_row

        row = scan_row(
            distance_m=6.0,
            normalized_goal_x=0.75,
            raw_action=(0.2, -0.1),
            mapped_command=(0.07, -0.01),
            projected_command=(0.07, -0.01),
        )
        self.assertEqual(
            tuple(row),
            (
                "distance_m",
                "normalized_goal_x",
                "raw_policy_mean_a_v",
                "raw_policy_mean_a_w",
                "mapped_v_cmd",
                "mapped_w_cmd",
                "projected_v_cmd",
                "projected_w_cmd",
            ),
        )

    def test_first_zero_crossing_is_interpolated(self):
        from legged_gym.navigation.v1_distance_diagnostics import first_zero_crossing

        self.assertAlmostEqual(
            first_zero_crossing(((2.0, 0.2), (2.5, -0.1))), 2.3333333333
        )
        self.assertIsNone(first_zero_crossing(((2.0, 0.2), (2.5, 0.1))))

    def test_causal_pair_preserves_physical_goal_and_labels_visible_goal(self):
        from legged_gym.navigation.v1_distance_diagnostics import causal_pair

        row = causal_pair(
            physical_goal_distance=6.0,
            visible_goal_distance=2.0,
            raw_action=(0.3, -0.1),
            mapped_command=(0.075, -0.01),
        )
        self.assertEqual(row["physical_goal_distance_m"], 6.0)
        self.assertEqual(row["visible_goal_distance_m"], 2.0)
        self.assertEqual(row["raw_policy_mean_a_v"], 0.3)
        self.assertEqual(row["mapped_v_cmd"], 0.075)


if __name__ == "__main__":
    unittest.main()
