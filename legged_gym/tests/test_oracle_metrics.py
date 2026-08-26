import unittest

from legged_gym.navigation.oracle_metrics import maze_spl, summarize_oracle_results


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


if __name__ == "__main__":
    unittest.main()
