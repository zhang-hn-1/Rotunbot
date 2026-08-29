"""Regression contracts for the isolated frozen V49 integration task."""

import hashlib
import os
from types import SimpleNamespace
import unittest

import isaacgym  # noqa: F401

from legged_gym.envs import task_registry
from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import (
    RotunbotVel,
    command_update_interval_steps,
    project_velocity_commands,
)
from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel_config import (
    RotunbotVelSRU50V49IntegrationCfg,
)


class V49LowLevelIntegrationContractTests(unittest.TestCase):
    def test_integration_task_and_timing_contract(self):
        env_cfg, train_cfg = task_registry.get_cfgs(
            name="rotunbot_vel_sru50_v49_integration"
        )
        self.assertEqual(
            env_cfg.asset.file.rsplit("/", 1)[-1], "Rotunbot_test2.urdf"
        )
        self.assertAlmostEqual(1.0 / env_cfg.sim.dt, 200.0)
        policy_dt = env_cfg.sim.dt * env_cfg.control.decimation
        self.assertAlmostEqual(1.0 / policy_dt, 50.0)
        self.assertEqual(
            command_update_interval_steps(
                policy_dt, env_cfg.commands.upper_level_command_frequency_hz
            ),
            10,
        )
        self.assertEqual(
            train_cfg.runner.experiment_name,
            "rotunbot_vel_sru50_v49_integration",
        )

    def test_projection_is_the_frozen_cone(self):
        import torch

        commands = torch.tensor([[0.50, 0.50], [-0.13, -0.20]])
        projected = project_velocity_commands(
            commands,
            maximum_forward_speed=0.13,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=1.0 / 0.27,
            envelope_fraction=1.0,
            stationary_threshold=0.0,
            turn_authority_start_speed=0.08,
            turn_authority_full_speed=0.10,
        )
        self.assertAlmostEqual(float(projected[0, 0]), 0.13)
        self.assertAlmostEqual(float(projected[0, 1]), 0.0351, places=4)
        self.assertAlmostEqual(float(projected[1, 1]), -0.0351, places=4)

    def test_dynamic_governor_is_opt_in_and_runs_before_final_projection(self):
        import torch
        from legged_gym.navigation.v49_dynamic_governor import (
            DynamicGovernorConfig,
            StateDependentReachabilityGovernor,
        )
        from legged_gym.navigation.v49_dynamic_reachability import (
            DynamicReachabilityTable,
        )

        rows = []
        for current_v in (0.0, 0.10):
            for command_v in (0.0, 0.04, 0.08):
                for command_w in (-0.02, 0.0, 0.02):
                    row = {
                        "current_v": current_v,
                        "projected_v": command_v,
                        "projected_w": command_w,
                    }
                    for horizon in (50, 100, 150, 200):
                        row["predicted_forward_velocity_%dms" % horizon] = command_v
                        row["predicted_yaw_rate_%dms" % horizon] = command_w
                    rows.append(row)
        governor = StateDependentReachabilityGovernor(
            DynamicReachabilityTable.from_rows(rows),
            DynamicGovernorConfig(
                maximum_forward_speed=0.10,
                maximum_yaw_rate=0.04,
                minimum_turn_radius=0.50,
                envelope_fraction=1.0,
            ),
        )

        env = RotunbotVel.__new__(RotunbotVel)
        env.num_envs = 1
        env.device = torch.device("cpu")
        env.command_targets = torch.zeros((1, 2))
        env.tracking_lin_vel = torch.zeros((1, 3))
        env.tracking_ang_vel = torch.zeros((1, 3))
        env.cfg = SimpleNamespace(
            commands=SimpleNamespace(
                dynamic_governor_enabled=False,
                max_forward_speed=0.10,
                max_yaw_rate=0.04,
                minimum_turn_radius=0.50,
                feasible_envelope_fraction=1.0,
            ),
            rewards=SimpleNamespace(stationary_command_threshold=0.0),
        )
        calls = []

        def capture(targets, env_ids=None):
            calls.append(targets.detach().clone())

        env.set_command_targets = capture
        env.set_governed_command_targets(torch.tensor([[0.08, 0.02]]), governor)
        self.assertTrue(torch.equal(calls[-1], torch.tensor([[0.08, 0.02]])))

        env.cfg.commands.dynamic_governor_enabled = True
        decisions = env.set_governed_command_targets(
            torch.tensor([[0.04, 0.02]]), governor
        )
        self.assertEqual(len(decisions), 1)
        self.assertAlmostEqual(float(calls[-1][0, 0]), 0.04, places=6)
        self.assertAlmostEqual(float(calls[-1][0, 1]), 0.02, places=6)

    def test_existing_depth_task_registration_and_camera_are_unchanged(self):
        env_cfg, train_cfg = task_registry.get_cfgs(
            name="rotunbot_maze_local_depth"
        )
        self.assertEqual(train_cfg.runner.experiment_name, "rotunbot_maze_local_depth")
        self.assertTrue(env_cfg.enable_camera_sensors_in_headless)
        self.assertTrue(env_cfg.camera.enable)
        self.assertEqual(env_cfg.camera.depth_backend, "fallback")
        self.assertEqual((env_cfg.camera.width, env_cfg.camera.height), (32, 8))
        self.assertEqual(env_cfg.asset.file.rsplit("/", 1)[-1], "Rotunbot.urdf")

    def test_frozen_checkpoint_hash(self):
        checkpoint = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "V49_FINAL_MODEL_TRAINING_CODE_TESTS_20260827",
                "01_frozen_v49_release",
                "model_frozen.pt",
            )
        )
        if not os.path.exists(checkpoint):
            self.skipTest("frozen release package is outside the project checkout")
        digest = hashlib.sha256()
        with open(checkpoint, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.assertEqual(
            digest.hexdigest(),
            "5cd24ff2b8b4b0a16e7a96f0bdd707c25d27c08c9ed7ae8931c37103bba1769a",
        )


if __name__ == "__main__":
    unittest.main()
