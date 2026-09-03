import unittest
from types import SimpleNamespace

import torch

from legged_gym.navigation.v1_velocity_teacher import (
    V1VelocityTeacherConfig,
    evaluate_teacher_gate,
    summarize_teacher_episodes,
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

    def test_open_space_large_bearing_retains_turn_authority(self):
        goal = torch.tensor([[0.2, 2.0]])
        diagnostics = teacher_velocity_diagnostics(
            goal, torch.zeros(1, 2), torch.tensor([2.0]), self.cfg
        )
        command = diagnostics["applied_command"][0]
        # With |w| <= v/R, reducing v to cos(bearing) would also remove the
        # yaw authority needed to recover.  Open-space recovery keeps enough
        # forward speed for the measured feasible turn envelope.
        self.assertGreaterEqual(float(command[0]), 0.15)
        self.assertGreaterEqual(float(command[1]), 0.075)

    def test_goal_approach_does_not_settle_outside_success_disk(self):
        diagnostics = teacher_velocity_diagnostics(
            torch.tensor([[0.38, 0.0]]),
            torch.zeros(1, 2),
            torch.tensor([2.0]),
            self.cfg,
        )
        # The plant has command/velocity lag.  At 0.38 m, the teacher must
        # still issue a measurable approach command rather than asymptotically
        # settling outside V1's 0.35 m success disk.
        self.assertGreater(float(diagnostics["applied_command"][0, 0]), 0.05)

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

    def test_nan_obstacle_distance_is_rejected_but_positive_inf_is_open_space(self):
        with self.assertRaises(ValueError):
            teacher_velocity_diagnostics(
                torch.tensor([[1.0, 0.0]]),
                torch.zeros(1, 2),
                torch.tensor([float("nan")]),
                self.cfg,
            )
        result = teacher_velocity_diagnostics(
            torch.tensor([[1.0, 0.0]]),
            torch.zeros(1, 2),
            torch.tensor([float("inf")]),
            self.cfg,
        )
        self.assertTrue(torch.isfinite(result["applied_command"]).all())

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

    def test_teacher_summary_contains_auditable_velocity_and_tracking_metrics(self):
        records = [
            {
                "success": True,
                "collision": False,
                "timeout": False,
                "initial_goal_distance_m": 1.0,
                "terminal_goal_distance_m": 0.2,
                "path_length_m": 1.1,
                "steps": 10,
                "spl": 1.0 / 1.1,
                "teacher_command_count": 10,
                "reverse_command_count": 0,
                "projection_activation_count": 2,
                "governor_modification_count": 3,
                "tracking_v_abs_error_sum": 0.4,
                "tracking_w_abs_error_sum": 0.2,
                "tracking_sample_count": 10,
                "teacher_v_sum": 1.0,
                "teacher_v_sq_sum": 0.12,
                "teacher_v_min": 0.05,
                "teacher_v_max": 0.15,
                "teacher_w_sum": 0.2,
                "teacher_w_sq_sum": 0.01,
                "teacher_w_min": -0.01,
                "teacher_w_max": 0.04,
                "projection_correction_sum": 0.03,
                "projection_correction_max": 0.02,
            }
        ]
        summary = summarize_teacher_episodes(records)
        for key in (
            "success_rate", "collision_rate", "timeout_rate", "spl",
            "path_efficiency", "mean_teacher_v_mps", "std_teacher_v_mps",
            "mean_teacher_w_rps", "std_teacher_w_rps",
            "reverse_command_ratio", "projection_activation_ratio",
            "governor_modification_ratio", "tracking_v_mae_mps",
            "tracking_w_mae_rps", "finite",
        ):
            self.assertIn(key, summary)
        self.assertAlmostEqual(summary["tracking_v_mae_mps"], 0.04)
        self.assertAlmostEqual(summary["tracking_w_mae_rps"], 0.02)
        self.assertTrue(summary["finite"])

    def test_teacher_gate_enforces_distance_specific_success_and_safety(self):
        summary = {
            "episodes": 100,
            "success_rate": 0.98,
            "collision_rate": 0.0,
            "timeout_rate": 0.02,
            "reverse_command_ratio": 0.0,
            "mean_projection_correction_norm": 0.01,
            "finite": True,
        }
        self.assertTrue(evaluate_teacher_gate(summary, 0.98)["pass"])
        self.assertFalse(evaluate_teacher_gate(summary, 0.99)["pass"])
        failed = dict(summary, collision_rate=0.01)
        self.assertFalse(evaluate_teacher_gate(failed, 0.98)["pass"])


if __name__ == "__main__":
    unittest.main()
