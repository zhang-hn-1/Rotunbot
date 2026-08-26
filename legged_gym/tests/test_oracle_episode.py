import unittest

import numpy as np

from legged_gym.navigation.oracle_episode import (
    FINAL_APPROACH,
    NAVIGATE,
    OracleEpisodePlanner,
    waypoint_reached,
)


class OracleEpisodeTests(unittest.TestCase):
    def test_goal_cell_enters_final_approach_without_repeating_local_waypoint(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        planner = OracleEpisodePlanner(grid, maze_shape=(5, 5), cell_size=2.0)

        waypoint = planner.next_local_waypoint([4.0, 0.0], 0.0, [4.0, 0.0])

        self.assertEqual(planner.phase, FINAL_APPROACH)
        self.assertTrue(waypoint.is_final_approach)
        self.assertEqual(waypoint.world_goal_xy, (4.0, 0.0))
        self.assertEqual(waypoint.temporary_world_goal_xy, (4.0, 0.0))
        planner.next_local_waypoint([4.0, 0.0], 0.0, [4.0, 0.0])
        self.assertEqual(planner.final_approach_entry_count, 1)

        escaped = planner.next_local_waypoint([2.0, 0.0], 0.0, [4.0, 0.0])
        self.assertEqual(planner.phase, FINAL_APPROACH)
        self.assertTrue(escaped.is_final_approach)
        self.assertEqual(planner.final_approach_escape_count, 1)

    def test_turn_aware_switch_requires_speed_only_for_large_turns(self):
        self.assertTrue(waypoint_reached(0.35, 1.0, 30.0, turn_aware=True))
        self.assertFalse(waypoint_reached(0.35, 0.31, 45.0, turn_aware=True))
        self.assertTrue(waypoint_reached(0.35, 0.30, 90.0, turn_aware=True))

    def test_planner_returns_a_local_waypoint_from_current_actual_pose(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        planner = OracleEpisodePlanner(grid, maze_shape=(5, 5), cell_size=2.0)
        waypoint = planner.next_local_waypoint(
            robot_xy=np.array([0.0, 0.0]),
            robot_yaw=np.pi / 2.0,
            global_goal_xy=np.array([4.0, 0.0]),
        )
        self.assertEqual(waypoint.cell, (3, 2))
        self.assertEqual(waypoint.world_goal_xy, (2.0, 0.0))
        np.testing.assert_allclose(waypoint.local_goal_xy, [0.0, -2.0], atol=1e-12)

    def test_filter_is_applied_before_world_adapter(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        planner = OracleEpisodePlanner(
            grid,
            maze_shape=(5, 5),
            cell_size=2.0,
            reachability=__import__(
                "legged_gym.navigation.reachability",
                fromlist=["ReachabilityEnvelope"],
            ).ReachabilityEnvelope((0.0,), (1.0,)),
        )
        waypoint = planner.next_local_waypoint([0.0, 0.0], 0.0, [4.0, 0.0])
        np.testing.assert_allclose(waypoint.filtered_local_goal_xy, [1.0, 0.0])
        np.testing.assert_allclose(waypoint.temporary_world_goal_xy, [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
