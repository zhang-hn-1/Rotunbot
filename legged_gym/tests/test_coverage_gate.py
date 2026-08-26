import unittest

from legged_gym.scripts.evaluate_single_local_goal_coverage import (
    BEARINGS_DEG,
    DISTANCES_M,
    build_success_rate_matrix,
)


class CoverageGateTests(unittest.TestCase):
    def test_success_matrix_is_distance_by_bearing(self):
        rows = [
            {
                "distance_m": distance,
                "bearing_deg": bearing,
                "success_rate": distance / 10.0 + (bearing + 180.0) / 10000.0,
            }
            for distance in DISTANCES_M
            for bearing in BEARINGS_DEG
        ]
        matrix = build_success_rate_matrix(rows)
        self.assertEqual(len(matrix), len(DISTANCES_M))
        self.assertEqual(len(matrix[0]), len(BEARINGS_DEG))
        self.assertAlmostEqual(matrix[2][4], 0.15 + 360.0 / 10000.0)


if __name__ == "__main__":
    unittest.main()
