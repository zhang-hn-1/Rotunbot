import unittest

import numpy as np

from legged_gym.navigation.control_diagnostics import (
    C2_INITIAL_SPEEDS_MPS,
    select_corner_case,
    select_detour_case,
    select_straight_case,
    select_real_corner_case,
    select_real_straight_case,
)


class ControlDiagnosticsTests(unittest.TestCase):
    def test_c2_speed_sweep_is_fixed(self):
        self.assertEqual(C2_INITIAL_SPEEDS_MPS, (0.0, 0.2, 0.4, 0.6))

    def test_real_straight_case_requires_continuous_walls_and_excludes_center(self):
        layout = np.ones((9, 9), dtype=np.int8)
        cells = ((1, 1), (1, 2), (1, 3), (1, 4))
        layout[tuple(np.asarray(cells).T)] = 0
        for cell in cells:
            layout[cell[0] - 1, cell[1]] = 1
            layout[cell[0] + 1, cell[1]] = 1
        case = select_real_straight_case(
            layout, center_cell=(4, 4), center_clearance_radius=2, minimum_edges=3
        )
        self.assertEqual(case["cells"], cells)
        self.assertEqual(len(case["wall_cells"]), 8)
        self.assertTrue(case["topology_validated"])

    def test_real_straight_case_rejects_center_cleared_area(self):
        layout = np.ones((9, 9), dtype=np.int8)
        cells = ((4, 1), (4, 2), (4, 3), (4, 4))
        layout[tuple(np.asarray(cells).T)] = 0
        for cell in cells:
            layout[cell[0] - 1, cell[1]] = 1
            layout[cell[0] + 1, cell[1]] = 1
        with self.assertRaises(ValueError):
            select_real_straight_case(
                layout, center_cell=(4, 4), center_clearance_radius=2, minimum_edges=3
            )

    def test_straight_case_can_be_bounded_for_a_control_gate(self):
        layout = np.zeros((7, 7), dtype=np.int8)
        case = select_straight_case(
            layout, start_cell=(3, 3), minimum_edges=2, maximum_edges=2
        )
        self.assertEqual(case["cells"], ((3, 3), (3, 4), (3, 5)))

    def test_corner_case_is_one_90_degree_turn(self):
        layout = np.zeros((5, 5), dtype=np.int8)
        case = select_corner_case(layout, start_cell=(2, 2))
        cells = case["cells"]
        first = np.asarray(cells[1]) - cells[0]
        second = np.asarray(cells[2]) - cells[1]
        self.assertEqual(abs(int(first.dot(second))), 0)
        self.assertEqual(len(cells), 3)

    def test_real_corner_requires_inner_and_outer_wall_topology(self):
        layout = np.ones((9, 9), dtype=np.int8)
        cells = ((1, 1), (1, 2), (2, 2))
        layout[tuple(np.asarray(cells).T)] = 0
        # Inner corner plus the outer wall continuation on both legs.
        for cell in ((2, 1), (0, 1), (0, 2), (1, 3), (2, 3)):
            layout[cell] = 1
        case = select_real_corner_case(
            layout, center_cell=(5, 5), center_clearance_radius=2
        )
        self.assertEqual(case["cells"], cells)
        self.assertIn((2, 1), case["inner_wall_cells"])
        self.assertGreaterEqual(len(case["outer_wall_cells"]), 2)
        self.assertTrue(case["topology_validated"])

    def test_detour_case_is_longer_than_manhattan_path(self):
        layout = np.zeros((5, 5), dtype=np.int8)
        layout[2, 1:4] = 1
        layout[1, 2] = 0
        case = select_detour_case(layout, start_cell=(2, 0))
        start, goal = case["cells"][0], case["cells"][-1]
        manhattan = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        self.assertGreater(len(case["cells"]) - 1, manhattan)


if __name__ == "__main__":
    unittest.main()
