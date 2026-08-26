import unittest

import numpy as np

from legged_gym.navigation.local_goal_adapter import local_to_world, world_to_local


class LocalGoalAdapterTests(unittest.TestCase):
    def test_requested_yaws_and_directions_round_trip(self):
        yaws = (0, 45, 90, 135, 180, -45, -90, -135)
        goals = (
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (1.0, -1.0),
        )
        robot_xy = np.array([2.5, -3.25])
        for yaw_deg in yaws:
            for local_goal in goals:
                world_goal = local_to_world(
                    robot_xy, np.deg2rad(yaw_deg), np.array(local_goal)
                )
                recovered = world_to_local(robot_xy, np.deg2rad(yaw_deg), world_goal)
                np.testing.assert_allclose(recovered, local_goal, atol=1e-10)

    def test_zero_yaw_is_translation(self):
        np.testing.assert_allclose(
            local_to_world([2.0, 3.0], 0.0, [0.5, -1.5]), [2.5, 1.5]
        )

    def test_ninety_degree_yaw_rotates_forward_to_world_left(self):
        np.testing.assert_allclose(
            local_to_world([0.0, 0.0], np.pi / 2.0, [1.0, 0.0]), [0.0, 1.0], atol=1e-12
        )

    def test_rejects_bad_or_nonfinite_vectors(self):
        with self.assertRaises(ValueError):
            local_to_world([0.0, 0.0, 0.0], 0.0, [1.0, 0.0])
        with self.assertRaises(ValueError):
            world_to_local([0.0, 0.0], np.nan, [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
