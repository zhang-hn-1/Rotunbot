import math
import unittest

import numpy as np
import torch

from legged_gym.navigation.v1_l_turn import build_l_turn_geometry
from legged_gym.navigation.v1_velocity_teacher import (
    V1VelocityTeacherConfig,
    teacher_velocity_diagnostics,
)
from legged_gym.navigation.v1_waypoint_manager import V1WaypointManager


class LTurnNavigationTests(unittest.TestCase):
    def test_left_and_right_geometry_are_mirrors(self):
        left = build_l_turn_geometry("left")
        right = build_l_turn_geometry("right")
        np.testing.assert_allclose(
            left.scenario.centerline[:, 0], right.scenario.centerline[:, 0], atol=1.0e-8
        )
        np.testing.assert_allclose(
            left.scenario.centerline[:, 1], -right.scenario.centerline[:, 1], atol=1.0e-8
        )
        np.testing.assert_allclose(left.waypoints[:, 0], right.waypoints[:, 0])
        np.testing.assert_allclose(left.waypoints[:, 1], -right.waypoints[:, 1])
        self.assertEqual(left.turn_direction, 1)
        self.assertEqual(right.turn_direction, -1)

    def test_l_turn_geometry_has_safe_ordered_waypoints_and_valid_goal(self):
        geometry = build_l_turn_geometry("left")
        self.assertEqual(geometry.waypoints.shape, (4, 2))
        self.assertTrue(np.isfinite(geometry.waypoints).all())
        np.testing.assert_allclose(geometry.waypoints[0], geometry.scenario.start_xy)
        np.testing.assert_allclose(geometry.waypoints[-1], geometry.scenario.goal_xy)
        self.assertGreater(geometry.waypoints[1, 0], geometry.waypoints[0, 0])
        self.assertGreater(geometry.waypoints[2, 1], geometry.waypoints[1, 1])

    def test_waypoint_manager_switches_and_transforms_to_robot_frame(self):
        manager = V1WaypointManager(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))), reach_radius=0.2
        )
        manager.reset()
        np.testing.assert_allclose(manager.get_current_waypoint(), (0.0, 0.0))
        self.assertTrue(manager.update((0.05, 0.0, 0.0)))
        self.assertEqual(manager.current_index, 1)
        np.testing.assert_allclose(
            manager.get_current_waypoint_robot((0.0, 0.0, math.pi / 2.0)),
            (0.0, -1.0),
            atol=1.0e-8,
        )
        self.assertFalse(manager.is_final_goal_reached((1.0, 0.0)))
        manager.update((1.0, 0.0, 0.0))
        manager.update((1.0, 1.0, 0.0))
        self.assertTrue(manager.is_final_goal_reached((1.0, 1.0)))

    def test_teacher_turn_yaw_sign_matches_robot_frame(self):
        config = V1VelocityTeacherConfig()
        actual = torch.zeros(2, 2)
        goals = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        result = teacher_velocity_diagnostics(
            goals, actual, torch.full((2,), 8.0), config
        )
        self.assertGreater(float(result["applied_command"][0, 1]), 0.0)
        self.assertLess(float(result["applied_command"][1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
