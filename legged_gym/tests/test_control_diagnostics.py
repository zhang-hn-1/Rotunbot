import unittest

import numpy as np

from legged_gym.navigation.control_diagnostics import (
    C2_INITIAL_SPEEDS_MPS,
    select_corner_case,
    select_detour_case,
    select_straight_case,
)


class ControlDiagnosticsTests(unittest.TestCase):
    def test_c2_speed_sweep_is_fixed(self):
        self.assertEqual(C2_INITIAL_SPEEDS_MPS, (0.0, 0.2, 0.4, 0.6))

    def test_straight_case_contains_only_collinear_free_cells(self):
        layout = np.zeros((7, 7), dtype=np.int8)
        case = select_straight_case(layout, start_cell=(3, 3), minimum_edges=3)
        self.assertEqual(case["cells"], ((3, 3), (3, 4), (3, 5), (3, 6)))

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
