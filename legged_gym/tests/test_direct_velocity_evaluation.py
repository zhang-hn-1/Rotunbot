import math
import csv
import json
import tempfile
import unittest
from pathlib import Path

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch
import torch

from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import project_velocity_commands
from legged_gym.envs.rotunbot.vel_tracking.feasible_transition_manager import (
    FeasibleVelocityTransitionManager,
    TransitionState,
)
from legged_gym.navigation.direct_velocity import normalized_action_to_velocity_command
import legged_gym.navigation.direct_velocity_evaluation as direct_velocity_evaluation
from legged_gym.navigation.direct_velocity_evaluation import (
    CommandDiagnostics,
    build_fixed_goal_specs,
    evaluate_b_gate_chain,
    evaluate_stage_gate,
    load_checkpoint_identity,
    summarize_evaluation,
    write_failure_artifacts,
)


class DirectVelocityEvaluationTests(unittest.TestCase):
    def test_s2b_formal_set_is_deterministic_exact_mixture_from_fixed_seeds(self):
        first = build_fixed_goal_specs("S2B", episodes=100)
        second = build_fixed_goal_specs("S2B", episodes=100)

        self.assertEqual(first, second)
        self.assertEqual({row["seed"] for row in first}, {0, 1, 2})
        self.assertEqual(
            {stage: sum(row["component"] == stage for row in first) for stage in ("S2B", "S2", "S1")},
            {"S2B": 70, "S2": 20, "S1": 10},
        )
        bounds = {
            "S2B": (0.5, 2.0, 45.0),
            "S2": (0.5, 1.5, 30.0),
            "S1": (0.5, 1.0, 10.0),
        }
        for row in first:
            distance_min, distance_max, bearing_max_deg = bounds[row["component"]]
            self.assertGreaterEqual(row["distance_m"], distance_min)
            self.assertLessEqual(row["distance_m"], distance_max)
            self.assertLessEqual(abs(row["bearing_rad"]), math.radians(bearing_max_deg))

    def test_gate_chain_is_strict_and_missing_safety_counters_fail(self):
        safe = {
            "episodes": 100,
            "success_rate": 0.95,
            "collision_count": 0,
            "timeout_rate": 0.05,
            "divergence_rate": 0.02,
            "rate_violation_count": 0,
            "feasible_domain_violation_count": 0,
            "hidden_projection_jump_count": 0,
        }
        self.assertTrue(evaluate_stage_gate(dict(safe, success_rate=0.90), "S2B")["pass"])

        incomplete = dict(safe)
        incomplete.pop("hidden_projection_jump_count")
        self.assertFalse(evaluate_stage_gate(incomplete, "S2B")["pass"])

        passed = evaluate_b_gate_chain(
            dict(safe, success_rate=0.90),
            dict(safe, success_rate=0.90),
            dict(safe, success_rate=0.93),
        )
        self.assertTrue(passed["pass"])
        failed = evaluate_b_gate_chain(
            dict(safe, success_rate=0.90),
            dict(safe, success_rate=0.90),
            dict(safe, success_rate=0.929),
        )
        self.assertFalse(failed["pass"])
        self.assertTrue(any("B1" in reason for reason in failed["failures"]))

    def test_reverse_requests_and_transition_application_are_measured_separately(self):
        diagnostics = CommandDiagnostics(
            policy_dt=0.02,
            maximum_linear_acceleration=0.4,
            maximum_yaw_acceleration=0.2,
            projection_jump_threshold=(0.05, 0.05),
        )
        first = diagnostics.record(
            raw_command=(-0.25, -0.10),
            requested_command=(-0.20, -0.08),
            applied_command=(0.0, 0.0),
            projected_applied_command=(0.0, 0.0),
            transition_active=True,
        )
        second = diagnostics.record(
            raw_command=(-0.25, -0.10),
            requested_command=(-0.20, -0.08),
            applied_command=(-0.005, -0.002),
            projected_applied_command=(-0.005, -0.002),
            transition_active=True,
        )
        third = diagnostics.record(
            raw_command=(-0.25, -0.10),
            requested_command=(-0.20, -0.08),
            applied_command=(-0.020, -0.002),
            projected_applied_command=(-0.019, -0.002),
            transition_active=False,
        )

        self.assertEqual(first["transition_activation_event"], 1)
        self.assertEqual(second["transition_activation_event"], 0)
        self.assertEqual(third["rate_violation"], 1)
        self.assertEqual(third["feasible_domain_violation"], 1)
        summary = diagnostics.summary()
        self.assertEqual(summary["raw_reverse_command_count"], 3)
        self.assertEqual(summary["requested_reverse_command_count"], 3)
        self.assertEqual(summary["applied_reverse_command_count"], 2)
        self.assertEqual(summary["transition_activation_count"], 1)
        self.assertEqual(summary["reverse_transition_activation_count"], 1)
        self.assertEqual(summary["rate_violation_count"], 1)
        self.assertEqual(summary["hidden_projection_jump_count"], 0)
        self.assertEqual(summary["feasible_domain_violation_count"], 1)

    def test_projection_jump_and_rate_violation_can_fail_independently(self):
        rate_only = CommandDiagnostics(
            policy_dt=0.02,
            maximum_linear_acceleration=0.4,
            maximum_yaw_acceleration=0.2,
            projection_jump_threshold=(0.05, 0.05),
        )
        rate_only.record((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), False)
        rate_row = rate_only.record(
            (0.02, 0.0), (0.02, 0.0), (0.02, 0.0), (0.02, 0.0), False
        )

        jump_only = CommandDiagnostics(
            policy_dt=0.02,
            maximum_linear_acceleration=5.0,
            maximum_yaw_acceleration=5.0,
            projection_jump_threshold=(0.05, 0.05),
        )
        jump_only.record((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), False)
        jump_row = jump_only.record(
            (0.06, 0.0), (0.06, 0.0), (0.06, 0.0), (0.06, 0.0), False
        )

        self.assertEqual((rate_row["rate_violation"], rate_row["hidden_projection_jump"]), (1, 0))
        self.assertEqual((jump_row["rate_violation"], jump_row["hidden_projection_jump"]), (0, 1))

    def test_terminal_step_selector_keeps_exposed_post_step_telemetry(self):
        post_reset = {
            "applied_command": (0.0, 0.0),
            "actual_velocity": (0.0, 0.0),
            "position": (9.0, 9.0),
            "transition_active": False,
        }
        terminal_post_step = {
            "applied_command": (-0.04, 0.01),
            "actual_velocity": (-0.03, 0.008),
            "position": (1.2, -0.3),
            "transition_active": True,
        }

        selected = direct_velocity_evaluation.select_step_telemetry(
            auto_done=True,
            post_step=post_reset,
            terminal_post_step=terminal_post_step,
        )

        self.assertEqual(selected, terminal_post_step)
        self.assertIsNot(selected, terminal_post_step)

    def test_negative_action_crosses_real_projection_boundary_with_distinct_telemetry(self):
        action = torch.tensor([[-0.2, 1.0]])
        raw = action * torch.tensor([[0.25, 0.10]])
        requested = normalized_action_to_velocity_command(
            action,
            maximum_forward_speed=0.25,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=2.0,
            envelope_fraction=1.0,
        )
        applied = project_velocity_commands(
            requested,
            maximum_forward_speed=0.25,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=2.0,
            envelope_fraction=1.0,
        )
        diagnostics = CommandDiagnostics(
            policy_dt=0.02,
            maximum_linear_acceleration=5.0,
            maximum_yaw_acceleration=5.0,
            projection_jump_threshold=(0.20, 0.20),
        )

        row = diagnostics.record(
            raw[0].tolist(),
            requested[0].tolist(),
            applied[0].tolist(),
            project_velocity_commands(
                applied,
                maximum_forward_speed=0.25,
                maximum_yaw_rate=0.10,
                minimum_turn_radius=2.0,
                envelope_fraction=1.0,
            )[0].tolist(),
            transition_active=False,
        )

        self.assertTrue(torch.isfinite(applied).all())
        self.assertLessEqual(float(applied[:, 0].abs().max()), 0.25)
        self.assertLessEqual(float(applied[:, 1].abs().max()), 0.10)
        self.assertLessEqual(float(applied[:, 1].abs().max()), float(applied[:, 0].abs().max()) / 2.0)
        self.assertAlmostEqual(float(raw[0, 1]), 0.10, places=6)
        self.assertAlmostEqual(float(requested[0, 1]), 0.025, places=6)
        self.assertEqual(
            (row["raw_reverse_command"], row["requested_reverse_command"], row["applied_reverse_command"]),
            (1, 1, 1),
        )
        self.assertEqual((row["projection_active"], row["governor_active"]), (1, 0))

    def test_negative_action_crosses_real_v62_transition_manager_boundary(self):
        manager = FeasibleVelocityTransitionManager(
            num_envs=1,
            device="cpu",
            dtype=torch.float32,
            dt=0.02,
            maximum_linear_acceleration=0.10,
            maximum_yaw_acceleration=0.007,
            maximum_forward_speed=0.25,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=2.0,
            envelope_fraction=1.0,
            stationary_threshold=0.0,
            reversal_detection_v=0.05,
            reversal_detection_w=0.015,
            reversal_minimum_request_jump_v=0.10,
            reversal_minimum_request_jump_w=0.03,
            settle_v_threshold=0.01,
            settle_w_threshold=0.005,
            settle_time=0.10,
            curvature_fraction_breakpoints=(0.0, 0.25, 0.50, 1.0),
            curvature_max_speed_values=(0.25, 0.20, 0.15, 0.10),
        )
        current = torch.tensor([[0.14, 0.035]])
        manager.update_target(
            current, current, current[:, 0], current[:, 1]
        )
        diagnostics = CommandDiagnostics(
            policy_dt=0.02,
            maximum_linear_acceleration=0.10,
            maximum_yaw_acceleration=0.007,
            projection_jump_threshold=(0.20, 0.20),
        )
        diagnostics.record(
            current[0].tolist(),
            current[0].tolist(),
            current[0].tolist(),
            current[0].tolist(),
            transition_active=False,
        )

        action = torch.tensor([[-0.2, 1.0]])
        raw = action * torch.tensor([[0.25, 0.10]])
        requested = normalized_action_to_velocity_command(
            action,
            maximum_forward_speed=0.25,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=2.0,
            envelope_fraction=1.0,
        )
        manager.update_target(
            requested, current, current[:, 0], current[:, 1]
        )
        applied, state, active = manager.advance(
            current, current[:, 0], current[:, 1]
        )
        transition_row = diagnostics.record(
            raw[0].tolist(),
            requested[0].tolist(),
            applied[0].tolist(),
            project_velocity_commands(
                applied,
                maximum_forward_speed=0.25,
                maximum_yaw_rate=0.10,
                minimum_turn_radius=2.0,
                envelope_fraction=1.0,
            )[0].tolist(),
            transition_active=active[0],
        )

        self.assertTrue(torch.allclose(raw, torch.tensor([[-0.05, 0.10]])))
        self.assertTrue(torch.allclose(requested, torch.tensor([[-0.05, 0.025]])))
        self.assertEqual(int(state.item()), TransitionState.BRAKE_TO_ORIGIN)
        self.assertTrue(bool(active.item()))
        self.assertGreater(float(applied[0, 0]), 0.0)
        self.assertLess(float(applied[0, 0]), float(current[0, 0]))
        self.assertEqual(
            (
                transition_row["raw_reverse_command"],
                transition_row["requested_reverse_command"],
                transition_row["applied_reverse_command"],
            ),
            (1, 1, 0),
        )
        self.assertEqual(transition_row["transition_activation_event"], 1)
        self.assertEqual(
            (
                transition_row["transition_active"],
                transition_row["projection_active"],
                transition_row["governor_active"],
            ),
            (1, 1, 1),
        )

        application_row = None
        current = applied
        for _ in range(400):
            current, state, active = manager.advance(
                current, current[:, 0], current[:, 1]
            )
            if float(current[0, 0]) < -1.0e-6:
                application_row = diagnostics.record(
                    raw[0].tolist(),
                    requested[0].tolist(),
                    current[0].tolist(),
                    project_velocity_commands(
                        current,
                        maximum_forward_speed=0.25,
                        maximum_yaw_rate=0.10,
                        minimum_turn_radius=2.0,
                        envelope_fraction=1.0,
                    )[0].tolist(),
                    transition_active=active[0],
                )
                break

        self.assertIsNotNone(application_row)
        self.assertEqual(application_row["applied_reverse_command"], 1)
        self.assertEqual(application_row["transition_active"], 1)
        self.assertTrue(torch.isfinite(current).all())
        self.assertLessEqual(float(current[:, 0].abs().max()), 0.25)
        self.assertLessEqual(float(current[:, 1].abs().max()), 0.10)
        summary = diagnostics.summary()
        self.assertEqual(summary["transition_activation_count"], 1)
        self.assertEqual(summary["reverse_transition_activation_count"], 1)

    def test_checkpoint_identity_hashes_evaluated_checkpoint_and_declared_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent.pt"
            checkpoint = root / "model_800.pt"
            parent.write_bytes(b"parent")
            checkpoint.write_bytes(b"child")
            (root / "checkpoint_metadata.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "parent_checkpoint": str(parent),
                        "sha256": "ddc9e669194254cef019a29d3619a2c16592e5d52e1a81e98b01bd52319149a3",
                    }
                )
            )

            identity = load_checkpoint_identity(checkpoint)

            self.assertEqual(identity["checkpoint"], str(checkpoint.resolve()))
            self.assertEqual(
                identity["sha256"],
                "ddc9e669194254cef019a29d3619a2c16592e5d52e1a81e98b01bd52319149a3",
            )
            self.assertEqual(identity["parent_checkpoint"], str(parent.resolve()))
            self.assertEqual(
                identity["parent_sha256"],
                "e47125968b3b71049fbc4802d1e40a71ea1359decfabacf70b34588037d4ff0c",
            )

    def test_failed_episode_artifacts_keep_full_command_path_and_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {
                    "episode_id": 7,
                    "time_s": 0.02,
                    "x": 0.0,
                    "y": 0.0,
                    "goal_distance": 1.0,
                    "raw_v_cmd": -0.25,
                    "raw_w_cmd": 0.10,
                    "requested_v_cmd": -0.20,
                    "requested_w_cmd": 0.08,
                    "v_cmd": -0.01,
                    "w_cmd": 0.002,
                    "v_actual": 0.04,
                    "w_actual": 0.001,
                    "transition_activation_event": 1,
                },
                {
                    "episode_id": 7,
                    "time_s": 0.04,
                    "x": 0.001,
                    "y": 0.0,
                    "goal_distance": 0.999,
                    "raw_v_cmd": -0.25,
                    "raw_w_cmd": 0.10,
                    "requested_v_cmd": -0.20,
                    "requested_w_cmd": 0.08,
                    "v_cmd": -0.02,
                    "w_cmd": 0.004,
                    "v_actual": 0.03,
                    "w_actual": 0.002,
                    "transition_activation_event": 0,
                },
            ]
            paths = write_failure_artifacts(
                directory,
                {"episode_id": 7, "seed": 1, "success": False, "timeout": True},
                rows,
            )

            self.assertEqual(Path(paths["root"]).name, "episode_007")
            with Path(paths["trajectory_csv"]).open(newline="") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(written[0]["raw_v_cmd"], "-0.25")
            self.assertEqual(written[0]["requested_v_cmd"], "-0.2")
            self.assertEqual(written[0]["v_cmd"], "-0.01")
            self.assertEqual(written[0]["transition_activation_event"], "1")
            self.assertEqual(
                {Path(path).name for path in paths["plots"]},
                {"xy_trajectory.png", "velocity_tracking.png", "goal_distance.png"},
            )
            self.assertTrue(all(Path(path).is_file() for path in paths["plots"]))

    def test_summary_contract_aggregates_episode_evidence_and_gate(self):
        records = [
            {
                "component": "S2B",
                "success": True,
                "collision": False,
                "timeout": False,
                "divergent": False,
                "path_length_m": 1.2,
                "terminal_goal_distance_m": 0.34,
                "rate_violation_count": 0,
                "feasible_domain_violation_count": 0,
                "hidden_projection_jump_count": 0,
                "transition_activation_count": 1,
                "reverse_transition_activation_count": 0,
                "raw_reverse_command_count": 0,
                "requested_reverse_command_count": 0,
                "applied_reverse_command_count": 0,
            },
            {
                "component": "S2",
                "success": False,
                "collision": False,
                "timeout": True,
                "divergent": False,
                "path_length_m": 2.0,
                "terminal_goal_distance_m": 0.8,
                "rate_violation_count": 0,
                "feasible_domain_violation_count": 0,
                "hidden_projection_jump_count": 0,
                "transition_activation_count": 2,
                "reverse_transition_activation_count": 1,
                "raw_reverse_command_count": 4,
                "requested_reverse_command_count": 4,
                "applied_reverse_command_count": 3,
            },
        ]
        identity = {
            "checkpoint": "/tmp/model.pt",
            "sha256": "child-sha",
            "parent_checkpoint": "/tmp/parent.pt",
            "parent_sha256": "parent-sha",
            "metadata_path": "/tmp/checkpoint_metadata.json",
        }

        summary = summarize_evaluation(
            records,
            stage="S2B",
            seed_list=(0, 1, 2),
            checkpoint_identity=identity,
            wall_clock_seconds=12.5,
        )

        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["timeout_count"], 1)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["fixed_seeds"], [0, 1, 2])
        self.assertEqual(summary["checkpoint_sha256"], "child-sha")
        self.assertEqual(summary["parent_checkpoint_sha256"], "parent-sha")
        self.assertEqual(summary["applied_reverse_command_count"], 3)
        self.assertEqual(summary["transition_activation_count"], 3)
        self.assertFalse(summary["gate"]["pass"])


if __name__ == "__main__":
    unittest.main()
