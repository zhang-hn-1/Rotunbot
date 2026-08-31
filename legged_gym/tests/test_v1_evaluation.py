import unittest

import torch


class V1EvaluationTests(unittest.TestCase):
    def test_evaluator_explicitly_destroys_sim_when_environment_has_no_close(self):
        from types import SimpleNamespace
        from legged_gym.scripts.eval_sru_visual_corridor_v1 import close_environment

        calls = []
        env = SimpleNamespace(
            viewer="viewer",
            sim="sim",
            gym=SimpleNamespace(
                destroy_viewer=lambda viewer: calls.append(("viewer", viewer)),
                destroy_sim=lambda sim: calls.append(("sim", sim)),
            ),
        )
        close_environment(env)
        self.assertEqual(calls, [("viewer", "viewer"), ("sim", "sim")])

    def test_curriculum_history_row_contains_gate_and_command_diagnostics(self):
        from legged_gym.scripts.train_sru_visual_corridor_v1 import curriculum_history_row

        row = curriculum_history_row(
            iteration=50,
            level=1,
            current_distance=3.0,
            next_distance=4.0,
            current_summary={
                "success_rate": 0.9,
                "collision_rate": 0.0,
                "timeout_rate": 0.1,
                "reverse_motion_ratio": 0.02,
            },
            next_summary={
                "success_rate": 0.8,
                "collision_rate": 0.0,
                "timeout_rate": 0.2,
                "reverse_motion_ratio": 0.03,
            },
            gate={"pass": True},
        )
        self.assertEqual(row["iteration"], 50)
        self.assertEqual(row["current_level"], 1)
        self.assertEqual(row["current_eval_success_rate"], 0.9)
        self.assertEqual(row["next_eval_success_rate"], 0.8)
        self.assertTrue(row["gate_pass"])

    def test_episode_summary_aggregates_command_path_safety_counters(self):
        from legged_gym.navigation.v1_evaluation import summarize_v1_episodes

        record = {
            "success": False,
            "collision": False,
            "timeout": True,
            "initial_goal_distance_m": 4.0,
            "terminal_goal_distance_m": 3.0,
            "steps": 2,
            "path_length_m": 0.1,
            "rate_violation_count": 1,
            "feasible_domain_violation_count": 2,
            "hidden_projection_jump_count": 3,
            "mean_command_correction": 0.04,
        }
        summary = summarize_v1_episodes([record])
        self.assertEqual(summary["rate_violation_count"], 1)
        self.assertEqual(summary["feasible_domain_violation_count"], 2)
        self.assertEqual(summary["hidden_projection_jump_count"], 3)
        self.assertAlmostEqual(summary["mean_command_correction"], 0.04)

    def test_episode_state_uses_the_matching_parallel_environment_position(self):
        from types import SimpleNamespace
        from legged_gym.scripts.eval_sru_visual_corridor_v1 import _episode_state

        env = SimpleNamespace(
            root_states=torch.tensor([[0.0, 0.0], [10.0, 2.0]]),
            dt=0.02,
            cfg=SimpleNamespace(
                commands=SimpleNamespace(
                    maximum_linear_acceleration=1.0,
                    maximum_yaw_acceleration=1.0,
                )
            ),
        )
        state = _episode_state(env, {"episode_id": 0, "seed": 1, "distance_m": 6.0}, 1)
        self.assertEqual(tuple(state["previous_position"]), (10.0, 2.0))

    def test_failure_plot_trajectory_row_matches_corridor_plot_contract(self):
        from legged_gym.scripts.eval_sru_visual_corridor_v1 import _trajectory_row

        row = _trajectory_row(
            episode_id=1,
            step=2,
            distance_m=6.0,
            position=(0.5, -0.1),
            goal_distance=5.5,
            raw_v=0.1,
            raw_w=0.02,
            requested_v=0.08,
            requested_w=0.01,
            applied_v=0.07,
            applied_w=0.01,
            actual_v=0.06,
            actual_w=0.01,
            dt=0.02,
        )
        for key in (
            "x", "y", "time_s", "v_cmd", "w_cmd", "v_actual", "w_actual",
            "goal_distance",
        ):
            self.assertIn(key, row)

    def test_evaluation_targets_supports_current_next_and_fixed_formal_modes(self):
        from legged_gym.scripts.eval_sru_visual_corridor_v1 import evaluation_targets

        self.assertEqual(
            evaluation_targets(4.0, 5.0, episodes=30),
            [("current", 4.0, 30), ("next", 5.0, 30)],
        )
        self.assertEqual(
            evaluation_targets(6.0, None, episodes=100),
            [("fixed_6m", 6.0, 100)],
        )

    def test_fixed_distance_specs_are_reproducible_and_auditable(self):
        from legged_gym.navigation.v1_evaluation import build_fixed_distance_specs

        first = build_fixed_distance_specs(6.0, episodes=5, seed=2026)
        second = build_fixed_distance_specs(6.0, episodes=5, seed=2026)
        self.assertEqual(first, second)
        self.assertEqual([row["distance_m"] for row in first], [6.0] * 5)
        self.assertEqual([row["episode_id"] for row in first], list(range(5)))
        self.assertTrue(all("seed" in row and "bearing_rad" in row for row in first))

    def test_episode_summary_reports_required_metrics_without_inventing_spl(self):
        from legged_gym.navigation.v1_evaluation import summarize_v1_episodes

        records = [
            {
                "success": True,
                "collision": False,
                "timeout": False,
                "initial_goal_distance_m": 6.0,
                "terminal_goal_distance_m": 0.3,
                "episode_length": 100,
                "path_length_m": 6.5,
                "mean_forward_velocity": 0.08,
                "reverse_steps": 2,
                "steps": 100,
            },
            {
                "success": False,
                "collision": False,
                "timeout": True,
                "initial_goal_distance_m": 6.0,
                "terminal_goal_distance_m": 1.0,
                "episode_length": 200,
                "path_length_m": 7.0,
                "mean_forward_velocity": 0.04,
                "reverse_steps": 10,
                "steps": 200,
            },
        ]
        summary = summarize_v1_episodes(records)
        self.assertEqual(summary["episodes"], 2)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertAlmostEqual(summary["collision_rate"], 0.0)
        self.assertAlmostEqual(summary["timeout_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_initial_goal_distance_m"], 6.0)
        self.assertAlmostEqual(summary["reverse_motion_ratio"], 12.0 / 300.0)
        self.assertIn("spl", summary)
        self.assertAlmostEqual(summary["spl"], (6.0 / 6.5) / 2.0)
        self.assertIn("path_efficiency", summary)

    def test_curriculum_gate_compares_current_and_next_distance(self):
        from legged_gym.navigation.v1_evaluation import curriculum_gate

        result = curriculum_gate(
            {"success_rate": 0.90, "collision_rate": 0.10},
            {"success_rate": 0.80, "collision_rate": 0.0},
        )
        self.assertTrue(result["pass"])
        self.assertFalse(
            curriculum_gate(
                {"success_rate": 0.89, "collision_rate": 0.0},
                {"success_rate": 0.90, "collision_rate": 0.0},
            )["pass"]
        )


if __name__ == "__main__":
    unittest.main()
