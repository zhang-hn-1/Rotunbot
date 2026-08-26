"""Unit tests for the map-only Oracle local-subgoal planner."""

import unittest

import numpy as np

from legged_gym.planners import OracleLocalSubgoalPlanner


class OracleLocalSubgoalPlannerTest(unittest.TestCase):
    def test_shortest_path_avoids_wall(self):
        maze = np.zeros((5, 5), dtype=np.uint8)
        maze[2, 1:4] = 1
        planner = OracleLocalSubgoalPlanner(maze, cell_size=1.0)
        waypoint, path = planner.plan((-1.5, -1.5), (1.5, 1.5))
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))
        self.assertEqual(len(path), 9)
        np.testing.assert_allclose(waypoint, planner.cell_to_world(path[1]))
        self.assertTrue(all(maze[cell] == 0 for cell in path))

    def test_lookahead_is_latched_to_path_cell(self):
        maze = np.zeros((5, 5), dtype=np.uint8)
        planner = OracleLocalSubgoalPlanner(maze, cell_size=2.0, lookahead_cells=2)
        waypoint, path = planner.plan((-4.0, -4.0), (4.0, 4.0))
        np.testing.assert_allclose(waypoint, planner.cell_to_world(path[2]))


if __name__ == "__main__":
    unittest.main()
