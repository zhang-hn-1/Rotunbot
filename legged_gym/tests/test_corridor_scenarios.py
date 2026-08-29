import unittest

import isaacgym  # noqa: F401 - package imports existing Isaac Gym modules
import numpy as np

from legged_gym.navigation.corridor_scenarios import (
    make_double_turn_scenario,
    make_l_scenario,
    make_straight_scenario,
)


class CorridorScenarioTests(unittest.TestCase):
    def test_straight_scenario_is_deterministic_and_has_expected_geometry(self):
        first = make_straight_scenario(width_m=2.0, length_m=5.0, seed=17)
        second = make_straight_scenario(width_m=2.0, length_m=5.0, seed=17)

        self.assertEqual(first.family, "straight")
        self.assertEqual(first.width_m, 2.0)
        self.assertEqual(first.seed, 17)
        np.testing.assert_allclose(first.centerline, second.centerline)
        self.assertAlmostEqual(np.linalg.norm(first.goal_xy - first.start_xy), 5.0)
        self.assertEqual(first.centerline.shape[1], 2)

    def test_l_scenario_rejects_invalid_geometry_and_has_one_turn(self):
        with self.assertRaises(ValueError):
            make_l_scenario(width_m=0.0, straight_m=3.0, turn_radius_m=2.0, seed=1)
        with self.assertRaises(ValueError):
            make_l_scenario(width_m=2.0, straight_m=3.0, turn_radius_m=0.0, seed=1)

        scenario = make_l_scenario(
            width_m=2.0, straight_m=3.0, turn_radius_m=2.0, seed=1
        )
        self.assertEqual(scenario.family, "l")
        self.assertEqual(len(scenario.turns), 1)
        self.assertGreater(scenario.path_length_m, 6.0)

    def test_double_turn_contains_both_turns_and_seed_replay(self):
        left_right = make_double_turn_scenario(
            width_m=2.0, turn_radius_m=2.0, handedness="left_right", seed=9
        )
        replay = make_double_turn_scenario(
            width_m=2.0, turn_radius_m=2.0, handedness="left_right", seed=9
        )
        right_left = make_double_turn_scenario(
            width_m=2.0, turn_radius_m=2.0, handedness="right_left", seed=9
        )

        self.assertEqual(left_right.family, "double_turn")
        self.assertEqual([turn.direction for turn in left_right.turns], [1, -1])
        self.assertEqual([turn.direction for turn in right_left.turns], [-1, 1])
        np.testing.assert_allclose(left_right.centerline, replay.centerline)


if __name__ == "__main__":
    unittest.main()
