import unittest

import torch

from legged_gym.navigation.local_goal import world_goal_to_robot_xy


class LocalGoalTests(unittest.TestCase):
    def test_forward_at_zero_yaw(self):
        out = world_goal_to_robot_xy(
            torch.tensor([[0.0, 0.0]]), torch.tensor([0.0]), torch.tensor([[2.0, 0.0]])
        )
        self.assertTrue(torch.allclose(out, torch.tensor([[2.0, 0.0]]), atol=1e-6))

    def test_world_y_goal_is_robot_forward_at_ninety_degrees(self):
        out = world_goal_to_robot_xy(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([torch.pi / 2]),
            torch.tensor([[0.0, 2.0]]),
        )
        self.assertTrue(torch.allclose(out, torch.tensor([[2.0, 0.0]]), atol=1e-5))

    def test_inverse_cardinal_yaws(self):
        robot = torch.zeros(4, 2)
        yaw = torch.tensor([0.0, torch.pi / 2, torch.pi, -torch.pi / 2])
        goals = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        out = world_goal_to_robot_xy(robot, yaw, goals)
        self.assertTrue(torch.allclose(out, torch.ones(4, 2) * torch.tensor([1.0, 0.0]), atol=1e-5))

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            world_goal_to_robot_xy(torch.zeros(2, 2), torch.zeros(1), torch.zeros(2, 2))


if __name__ == "__main__":
    unittest.main()
