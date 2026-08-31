import unittest
from types import SimpleNamespace

import torch

from legged_gym.navigation.v1_velocity_teacher import (
    V1VelocityTeacherConfig,
    teacher_velocity_command,
    teacher_velocity_diagnostics,
)


class V1VelocityTeacherTests(unittest.TestCase):
    def setUp(self):
        self.cfg = V1VelocityTeacherConfig()

    def test_forward_and_side_goals_choose_speed_and_turn_sign(self):
        goals = torch.tensor([[2.0, 0.0], [2.0, 1.0], [2.0, -1.0]])
        actual = torch.zeros(3, 2)
        distance = torch.full((3,), 2.0)
        command = teacher_velocity_command(goals, actual, distance, self.cfg)
        self.assertGreater(float(command[0, 0]), 0.0)
        self.assertGreater(float(command[1, 1]), 0.0)
        self.assertLess(float(command[2, 1]), 0.0)
        self.assertGreater(float(command[0, 0]), float(command[1, 0]))

    def test_near_obstacle_and_large_bearing_reduce_forward_speed(self):
        goals = torch.tensor([[2.0, 0.0], [0.2, 2.0]])
        actual = torch.zeros(2, 2)
        obstacle_distance = torch.tensor([2.0, 0.5])
        command = teacher_velocity_command(goals, actual, obstacle_distance, self.cfg)
        self.assertGreater(float(command[0, 0]), float(command[1, 0]))
        self.assertLessEqual(float(command[1, 0]), self.cfg.max_forward_speed)

    def test_diagnostics_are_finite_and_projected_into_v62_domain(self):
        goals = torch.tensor([[2.0, 1.0], [1.0, -0.5], [0.35, 0.0]])
        actual = torch.tensor([[0.10, 0.01], [0.0, -0.01], [0.2, 0.0]])
        obstacle_distance = torch.tensor([float("inf"), 1.0, 0.4])
        diagnostics = teacher_velocity_diagnostics(
            goals, actual, obstacle_distance, self.cfg
        )
        for value in diagnostics.values():
            self.assertTrue(torch.isfinite(value).all())
        applied = diagnostics["applied_command"]
        self.assertTrue(torch.all(applied[:, 0].abs() <= self.cfg.max_forward_speed + 1e-6))
        self.assertTrue(torch.all(applied[:, 1].abs() <= self.cfg.max_yaw_rate + 1e-6))
        self.assertTrue(torch.all(diagnostics["projection_correction_norm"] >= 0.0))

    def test_config_like_objects_are_supported(self):
        cfg = SimpleNamespace(
            max_forward_speed=0.25,
            max_yaw_rate=0.10,
            minimum_turn_radius=2.0,
            feasible_envelope_fraction=1.0,
            goal_radius=0.35,
        )
        command = teacher_velocity_command(
            torch.tensor([[1.0, 0.0]]),
            torch.zeros(1, 2),
            torch.tensor([2.0]),
            cfg,
        )
        self.assertTrue(torch.isfinite(command).all())


if __name__ == "__main__":
    unittest.main()
