import unittest

import numpy as np

from legged_gym.navigation.bfs_planner import (
    cell_center,
    plan_cells,
    select_next_waypoint,
    world_to_cell,
)


class BfsPlannerTests(unittest.TestCase):
    def test_finds_shortest_four_neighbor_path(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 1:4] = 1
        path = plan_cells(grid, (0, 0), (4, 4))
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))
        self.assertEqual(len(path), 9)
        for left, right in zip(path, path[1:]):
            self.assertEqual(abs(left[0] - right[0]) + abs(left[1] - right[1]), 1)

    def test_rejects_wall_and_unreachable_goal(self):
        grid = np.zeros((3, 3), dtype=np.uint8)
        grid[1, 1] = 1
        with self.assertRaises(ValueError):
            plan_cells(grid, (1, 1), (0, 0))
        sealed = np.zeros((5, 5), dtype=np.uint8)
        sealed[1:4, 2] = 1
        sealed[0, 2] = 1
        sealed[4, 2] = 1
        with self.assertRaises(ValueError):
            plan_cells(sealed, (2, 1), (2, 3))

    def test_cell_conversion_and_waypoint_selection(self):
        self.assertEqual(world_to_cell([0.9, -1.1], (5, 5), 2.0), (2, 1))
        np.testing.assert_allclose(cell_center((2, 3), (5, 5), 2.0), [0.0, 2.0])
        self.assertEqual(select_next_waypoint(((1, 1), (1, 2)), 0), (1, 2))
        self.assertEqual(select_next_waypoint(((1, 1), (1, 2)), 1), (1, 2))
        with self.assertRaises(IndexError):
            select_next_waypoint(((1, 1),), 1)


if __name__ == "__main__":
    unittest.main()
