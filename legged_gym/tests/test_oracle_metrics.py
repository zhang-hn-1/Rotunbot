import unittest

from legged_gym.navigation.oracle_metrics import maze_spl, summarize_oracle_results
from legged_gym.scripts.compare_oracle_variants import build_comparison_table


class OracleMetricsTests(unittest.TestCase):
    def test_spl_is_zero_for_failed_episode(self):
        self.assertEqual(maze_spl(False, 10.0, 20.0), 0.0)
        self.assertAlmostEqual(maze_spl(True, 10.0, 20.0), 0.5)

    def test_summary_separates_physical_failures_and_software_errors(self):
        summary = summarize_oracle_results([
            {
                "reason": "global_success",
                "waypoint_count": 2,
                "local_waypoint_reached_count": 2,
                "actual_path_length_m": 3.0,
                "bfs_shortest_path_length_m": 2.0,
                "maze_spl": 2.0 / 3.0,
                "completion_time_s": 4.0,
            },
            {
                "reason": "collision",
                "waypoint_count": 1,
                "local_waypoint_reached_count": 0,
                "actual_path_length_m": 1.0,
                "bfs_shortest_path_length_m": 2.0,
                "maze_spl": 0.0,
                "completion_time_s": 1.0,
                "coordinate_error_count": 1,
            },
        ])
        self.assertEqual(summary["global_success_rate"], 0.5)
        self.assertEqual(summary["collision_rate"], 0.5)
        self.assertEqual(summary["local_waypoint_reach_rate"], 2.0 / 3.0)
        self.assertEqual(summary["coordinate_error_count"], 1)
        self.assertEqual(summary["planner_error_count"], 0)

    def test_summary_counts_final_approach_outcomes(self):
        summary = summarize_oracle_results([
            {
                "reason": "global_success",
                "waypoint_count": 2,
                "local_waypoint_reached_count": 2,
                "final_approach_entered": True,
                "final_approach_success": True,
                "final_approach_timeout": False,
                "final_approach_escape": False,
            },
            {
                "reason": "timeout",
                "waypoint_count": 3,
                "local_waypoint_reached_count": 3,
                "final_approach_entered": True,
                "final_approach_success": False,
                "final_approach_timeout": True,
                "final_approach_escape": False,
            },
            {
                "reason": "final_approach_escape",
                "waypoint_count": 4,
                "local_waypoint_reached_count": 4,
                "final_approach_entered": True,
                "final_approach_success": False,
                "final_approach_timeout": False,
                "final_approach_escape": True,
            },
        ])
        self.assertEqual(summary["final_approach_entry_count"], 3)
        self.assertEqual(summary["final_approach_success_count"], 1)
        self.assertEqual(summary["final_approach_timeout_count"], 1)
        self.assertEqual(summary["final_approach_escape_count"], 1)

    def test_comparison_table_uses_stable_variant_order(self):
        table = build_comparison_table({
            "D": {"global_sr": 0.4},
            "A": {"global_sr": 0.1},
            "C": {"global_sr": 0.3},
            "B": {"global_sr": 0.2},
        })
        self.assertEqual([row["variant"] for row in table], ["A", "B", "C", "D"])
        self.assertIn("waypoint_failure_rate", table[0])
        self.assertIn("planner_error_count", table[0])


if __name__ == "__main__":
    unittest.main()
