"""TDD contracts for the empty-map V49 waypoint controller."""

import math
import unittest

import torch

from legged_gym.navigation.v49_waypoint_controller import (
    V49WaypointConfig,
    V49WaypointController,
    WaypointSequenceController,
)


class V49WaypointControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = V49WaypointController(V49WaypointConfig())

    def test_front_waypoint_has_zero_bearing_and_forward_speed(self):
        result = self.controller.command(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([0.0]),
            torch.tensor([[1.0, 0.0]]),
        )
        self.assertAlmostEqual(float(result.bearing_error[0]), 0.0, places=6)
        self.assertGreater(float(result.raw_command[0, 0]), 0.0)
        self.assertAlmostEqual(float(result.raw_command[0, 1]), 0.0, places=6)

    def test_left_and_right_waypoints_have_opposite_yaw_signs(self):
        left = self.controller.command(
            torch.zeros(1, 2), torch.zeros(1), torch.tensor([[1.0, 1.0]])
        )
        right = self.controller.command(
            torch.zeros(1, 2), torch.zeros(1), torch.tensor([[1.0, -1.0]])
        )
        self.assertGreater(float(left.raw_command[0, 1]), 0.0)
        self.assertLess(float(right.raw_command[0, 1]), 0.0)

    def test_angle_wrap_is_continuous_at_pi(self):
        robot_yaw = torch.tensor([math.pi - 0.01])
        waypoint_heading = torch.tensor([-math.pi + 0.01])
        result = self.controller.command(
            torch.zeros(1, 2), robot_yaw,
            torch.stack((torch.cos(waypoint_heading), torch.sin(waypoint_heading)), dim=1),
        )
        self.assertAlmostEqual(float(result.bearing_error[0]), 0.02, places=4)

    def test_projection_is_delegated_to_v49_feasible_domain(self):
        result = self.controller.command(
            torch.zeros(1, 2), torch.zeros(1), torch.tensor([[1.0, 10.0]])
        )
        self.assertLessEqual(float(result.projected_command[0, 0]), 0.13)
        self.assertLessEqual(
            abs(float(result.projected_command[0, 1])),
            0.27 * abs(float(result.projected_command[0, 0])) + 1e-6,
        )

    def test_sequence_switches_one_waypoint_per_tick_without_reset(self):
        sequence = WaypointSequenceController(
            torch.tensor([[0.1, 0.0], [0.2, 0.0], [1.0, 0.0]])
        )
        first = sequence.tick(torch.tensor([[0.0, 0.0]]), torch.tensor([0.0]))
        self.assertEqual(first.active_waypoint_index, 1)
        self.assertTrue(first.waypoint_switched)
        self.assertFalse(first.done)
        self.assertEqual(sequence.switch_count, 1)
        second = sequence.tick(torch.tensor([[0.0, 0.0]]), torch.tensor([0.0]))
        self.assertEqual(second.active_waypoint_index, 2)
        self.assertTrue(second.waypoint_switched)
        self.assertEqual(sequence.switch_count, 2)

    def test_final_waypoint_emits_zero_command(self):
        sequence = WaypointSequenceController(torch.tensor([[0.1, 0.0]]))
        result = sequence.tick(torch.tensor([[0.1, 0.0]]), torch.tensor([0.0]))
        self.assertTrue(result.sequence_complete)
        self.assertTrue(torch.equal(result.projected_command, torch.zeros(1, 2)))

    def test_command_is_held_for_ten_policy_steps(self):
        sequence = WaypointSequenceController(torch.tensor([[1.0, 0.0]]))
        commands = []
        for policy_step in range(20):
            robot_xy = torch.tensor([[0.0, 0.0]])
            if policy_step >= 10:
                robot_xy = torch.tensor([[0.4, 0.0]])
            commands.append(sequence.command_for_policy_step(
                policy_step,
                robot_xy,
                torch.tensor([0.0]),
            ).clone())
        for policy_step in range(1, 10):
            self.assertTrue(torch.equal(commands[0], commands[policy_step]))
        self.assertFalse(torch.equal(commands[0], commands[10]))


if __name__ == "__main__":
    unittest.main()
