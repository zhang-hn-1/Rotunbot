"""Source and numerical contracts for the Rotunbot velocity tracker."""

import inspect
import os
import tempfile
import unittest

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch
import torch

from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import (
    RotunbotVel,
    advance_correlated_velocity_commands,
    command_update_interval_steps,
    command_target_gap_mask,
    command_request_jump_mask,
    error_aligned_residual_actions,
    feasible_yaw_rate_limit,
    independent_feasible_velocity_profile,
    lead_compensated_velocity_commands,
    nominal_actuator_actions,
    planar_velocity_in_heading_frame,
    project_velocity_commands,
    rate_limit_velocity_commands,
    reversal_brake_mask,
    smooth_feasible_velocity_profile,
    speed_scheduled_value,
    tracking_integral_reset_mask,
    update_rate_gated_error_integral,
    velocity_error_feedback_actions,
    velocity_error_derivative_actions,
    velocity_error_integral_actions,
    velocity_rate_feedforward_actions,
    yaw_reversal_brake_mask,
)
from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel_config import (
    RotunbotVelCfg,
    RotunbotVelDynamicCfg,
    RotunbotVelDynamicRateCfg,
    RotunbotVelDynamicPreviewCfg,
    RotunbotVelDynamicPICfg,
    RotunbotVelDynamicSteadyPICfg,
    RotunbotVelDynamicAnticipatoryCfg,
    RotunbotVelDynamicAnticipatoryV20Cfg,
    RotunbotVelDynamicSmoothFeedforwardV21Cfg,
    RotunbotVelDynamicSmoothBrakeV22Cfg,
    RotunbotVelDynamicAccelerationV23Cfg,
    RotunbotVelDynamicDualLeadV24Cfg,
    RotunbotVelDynamicDualPhaseV25Cfg,
    RotunbotVelDynamicExplicitPhaseV26Cfg,
    RotunbotVelIndependentV28Cfg,
    RotunbotVelIndependentRobustV29Cfg,
    RotunbotVelIndependentFeasibleV30Cfg,
    RotunbotVelIndependentDynamicEnvelopeV31Cfg,
    RotunbotVelIndependentBrakeGovernorV32Cfg,
    RotunbotVelIndependentTargetedBrakeV33Cfg,
    RotunbotVelIndependentTargetedBrakeV34Cfg,
    RotunbotVelIndependentScheduledFeedbackV35Cfg,
    RotunbotVelIndependentWaveRobustV36Cfg,
    RotunbotVelIndependentSafeEnvelopeV37Cfg,
    RotunbotVelIndependentYawBrakeV38Cfg,
    RotunbotVelIndependentYawTransitionV39Cfg,
    RotunbotVelIndependentTransitionEnvelopeV40Cfg,
    RotunbotVelSRU50V41Cfg,
    RotunbotVelSRU50V41CfgPPO,
    RotunbotVelSRU50LearnableV42Cfg,
    RotunbotVelSRU50LearnableV42CfgPPO,
    RotunbotVelSRU50AngularResidualV43Cfg,
    RotunbotVelSRU50AngularResidualV43CfgPPO,
    RotunbotVelSRU50TransientV44Cfg,
    RotunbotVelSRU50TransientV44CfgPPO,
    RotunbotVelSRU50ReachableV45Cfg,
    RotunbotVelSRU50ReachableV45CfgPPO,
    RotunbotVelSRU50LinearResidualV46Cfg,
    RotunbotVelSRU50LinearResidualV46CfgPPO,
    RotunbotVelSRU50DirectV47Cfg,
    RotunbotVelSRU50DirectV47CfgPPO,
    RotunbotVelSRU50DirectDynamicV48Cfg,
    RotunbotVelSRU50DirectDynamicV48CfgPPO,
    RotunbotVelSRU50ReleaseV49Cfg,
    RotunbotVelSRU50ReleaseV49CfgPPO,
)
from legged_gym.utils.helpers import get_load_path


class VelocityCommandProjectionTests(unittest.TestCase):
    def test_sru_frequency_contract_is_200_50_5_hz(self):
        physics_dt = float(RotunbotVelSRU50V41Cfg.sim.dt)
        policy_dt = physics_dt * int(RotunbotVelSRU50V41Cfg.control.decimation)
        self.assertAlmostEqual(1.0 / physics_dt, 200.0)
        self.assertAlmostEqual(1.0 / policy_dt, 50.0)
        self.assertEqual(
            command_update_interval_steps(
                policy_dt,
                RotunbotVelSRU50V41Cfg.commands.upper_level_command_frequency_hz,
            ),
            10,
        )
        self.assertEqual(RotunbotVelSRU50V41CfgPPO.runner.num_steps_per_env, 64)

    def test_command_frequency_must_have_integral_low_level_hold(self):
        self.assertEqual(command_update_interval_steps(0.02, 5.0), 10)
        with self.assertRaises(ValueError):
            command_update_interval_steps(0.02, 7.0)

    def test_sru_v42_keeps_timing_and_restores_bounded_residual_authority(self):
        policy_dt = float(RotunbotVelSRU50LearnableV42Cfg.sim.dt) * int(
            RotunbotVelSRU50LearnableV42Cfg.control.decimation
        )
        self.assertAlmostEqual(policy_dt, 0.02)
        self.assertEqual(
            command_update_interval_steps(
                policy_dt,
                RotunbotVelSRU50LearnableV42Cfg.commands.upper_level_command_frequency_hz,
            ),
            10,
        )
        self.assertAlmostEqual(
            RotunbotVelSRU50LearnableV42Cfg.control.nominal_yaw_gain_intercept,
            0.145,
        )
        self.assertEqual(
            RotunbotVelSRU50LearnableV42Cfg.control.residual_action_scale,
            [0.04, 0.20],
        )
        self.assertFalse(
            RotunbotVelSRU50LearnableV42Cfg.control.disable_residual_during_braking
        )
        self.assertEqual(
            RotunbotVelSRU50LearnableV42CfgPPO.runner.num_steps_per_env, 64
        )

    def test_sru_v43_preserves_linear_controller_and_learns_angular_residual(self):
        self.assertEqual(
            RotunbotVelSRU50AngularResidualV43Cfg.control.residual_action_scale,
            [0.0, 0.35],
        )
        self.assertTrue(
            RotunbotVelSRU50AngularResidualV43Cfg.control.residual_error_alignment_filter
        )
        self.assertFalse(
            RotunbotVelSRU50AngularResidualV43Cfg.control.disable_residual_during_braking
        )
        self.assertEqual(
            RotunbotVelSRU50AngularResidualV43CfgPPO.runner.num_steps_per_env, 64
        )

    def test_sru_v44_matches_navigation_transients_and_observes_momentum(self):
        self.assertEqual(RotunbotVelSRU50TransientV44Cfg.env.num_observations, 24)
        self.assertAlmostEqual(
            RotunbotVelSRU50TransientV44Cfg.commands.resampling_time, 2.0
        )
        self.assertTrue(
            RotunbotVelSRU50TransientV44Cfg.commands.observe_tracking_error_derivatives
        )
        self.assertEqual(
            RotunbotVelSRU50TransientV44Cfg.control.residual_action_scale,
            [0.0, 0.35],
        )
        self.assertEqual(
            RotunbotVelSRU50TransientV44CfgPPO.runner.num_steps_per_env, 64
        )

    def test_sru_v45_uses_measured_dynamic_reachable_set(self):
        self.assertAlmostEqual(
            RotunbotVelSRU50ReachableV45Cfg.control.angular_feedback_gain, 0.15
        )
        self.assertEqual(
            RotunbotVelSRU50ReachableV45Cfg.control.residual_action_scale,
            [0.0, 0.15],
        )
        self.assertAlmostEqual(
            RotunbotVelSRU50ReachableV45Cfg.commands.maximum_linear_acceleration,
            0.08,
        )
        self.assertAlmostEqual(
            RotunbotVelSRU50ReachableV45Cfg.commands.maximum_yaw_acceleration,
            0.010,
        )
        self.assertEqual(
            RotunbotVelSRU50ReachableV45CfgPPO.runner.num_steps_per_env, 64
        )

    def test_sru_v46_can_correct_linear_sign_transitions(self):
        self.assertEqual(
            RotunbotVelSRU50LinearResidualV46Cfg.control.residual_action_scale,
            [0.04, 0.15],
        )
        self.assertAlmostEqual(
            RotunbotVelSRU50LinearResidualV46Cfg.commands.maximum_linear_acceleration,
            0.08,
        )
        self.assertAlmostEqual(
            RotunbotVelSRU50LinearResidualV46Cfg.commands.maximum_yaw_acceleration,
            0.010,
        )
        self.assertAlmostEqual(
            RotunbotVelSRU50LinearResidualV46Cfg.rewards.scales.linear_wrong_direction,
            -8.0,
        )
        self.assertEqual(
            RotunbotVelSRU50LinearResidualV46CfgPPO.runner.num_steps_per_env, 64
        )

    def test_sru_v47_uses_exact_reachable_command_contract(self):
        self.assertTrue(
            RotunbotVelSRU50DirectV47Cfg.commands.direct_command_tracking
        )
        self.assertAlmostEqual(
            RotunbotVelSRU50DirectV47Cfg.commands.max_forward_speed, 0.13
        )
        self.assertAlmostEqual(
            RotunbotVelSRU50DirectV47Cfg.commands.minimum_turn_radius,
            2.8333333333333335,
        )
        self.assertEqual(
            RotunbotVelSRU50DirectV47Cfg.control.residual_action_scale,
            [0.08, 0.30],
        )
        self.assertGreater(
            RotunbotVelSRU50DirectV47Cfg.rewards.scales.curvature_tracking, 0.0
        )
        self.assertEqual(
            RotunbotVelSRU50DirectV47CfgPPO.runner.num_steps_per_env, 64
        )

    def test_sru_v48_uses_held_5hz_rate_and_primary_velocity_rewards(self):
        self.assertTrue(
            RotunbotVelSRU50DirectDynamicV48Cfg.commands.direct_command_tracking
        )
        self.assertTrue(
            RotunbotVelSRU50DirectDynamicV48Cfg.commands.hold_upper_command_rate
        )
        self.assertGreater(
            RotunbotVelSRU50DirectDynamicV48Cfg.commands.random_walk_profile_fraction,
            0.0,
        )
        scales = RotunbotVelSRU50DirectDynamicV48Cfg.rewards.scales
        self.assertGreater(scales.tracking_lin_vel, scales.curvature_tracking)
        self.assertGreater(scales.tracking_ang_vel, scales.curvature_tracking)
        self.assertEqual(
            RotunbotVelSRU50DirectDynamicV48CfgPPO.runner.num_steps_per_env, 64
        )

    def test_correlated_command_walk_stays_in_domain_and_keeps_drive_sign(self):
        torch.manual_seed(11)
        commands = torch.tensor([[0.10, 0.02], [-0.10, -0.02]])
        updated = advance_correlated_velocity_commands(
            commands,
            linear_step=0.008,
            yaw_step=0.004,
            minimum_speed=0.08,
            maximum_forward_speed=0.13,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=2.8333333333333335,
            envelope_fraction=0.85,
            stationary_threshold=0.02,
            turn_authority_start_speed=0.08,
            turn_authority_full_speed=0.10,
        )
        self.assertTrue(torch.equal(torch.sign(updated[:, 0]), torch.tensor([1.0, -1.0])))
        self.assertTrue(torch.all(torch.abs(updated[:, 0]) >= 0.08))
        self.assertTrue(torch.all(torch.abs(updated[:, 0]) <= 0.13))
        yaw_limit = feasible_yaw_rate_limit(
            updated[:, 0], 0.10, 2.8333333333333335, 0.85, 0.08, 0.10
        )
        self.assertTrue(torch.all(torch.abs(updated[:, 1]) <= yaw_limit + 1.0e-7))

    def test_sru_v49_freezes_calibrated_release_controller(self):
        control = RotunbotVelSRU50ReleaseV49Cfg.control
        self.assertAlmostEqual(control.angular_feedback_gain, 0.40)
        self.assertAlmostEqual(control.angular_rate_feedforward_time, 0.65)
        self.assertAlmostEqual(control.angular_feedback_action_limit, 0.30)
        self.assertAlmostEqual(control.angular_rate_feedforward_action_limit, 0.12)
        self.assertAlmostEqual(
            RotunbotVelSRU50ReleaseV49Cfg.commands.minimum_turn_radius,
            3.148148148148148,
        )
        self.assertTrue(
            RotunbotVelSRU50ReleaseV49Cfg.commands.direct_command_tracking
        )
        self.assertEqual(
            RotunbotVelSRU50ReleaseV49CfgPPO.runner.num_steps_per_env, 64
        )

    def test_measured_speed_blocks_premature_yaw_authority(self):
        commands = torch.tensor([[0.15, 0.10], [0.15, 0.10], [0.15, 0.10]])
        projected = project_velocity_commands(
            commands,
            maximum_forward_speed=0.15,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=0.90,
            envelope_fraction=0.85,
            turn_authority_start_speed=0.08,
            turn_authority_full_speed=0.10,
            authority_forward_velocity=torch.tensor([0.0, 0.09, 0.11]),
        )
        self.assertEqual(float(projected[0, 1]), 0.0)
        self.assertGreater(float(projected[1, 1]), 0.0)
        self.assertLess(float(projected[1, 1]), float(projected[2, 1]))
        self.assertAlmostEqual(float(projected[2, 1]), 0.085, places=6)

    def test_absolute_checkpoint_run_does_not_require_experiment_root(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            run_dir = os.path.join(temporary_root, "foreign_run")
            os.makedirs(run_dir)
            checkpoint_path = os.path.join(run_dir, "model_7.pt")
            with open(checkpoint_path, "wb"):
                pass
            missing_experiment_root = os.path.join(temporary_root, "empty_task")
            self.assertEqual(
                get_load_path(missing_experiment_root, run_dir, checkpoint=7),
                checkpoint_path,
            )

    def test_zero_speed_forbids_yaw(self):
        limit = feasible_yaw_rate_limit(
            torch.tensor([0.0]),
            maximum_yaw_rate=0.1,
            minimum_turn_radius=0.9,
            envelope_fraction=0.85,
        )
        self.assertEqual(float(limit.item()), 0.0)

    def test_yaw_limit_is_speed_dependent_and_capped(self):
        limit = feasible_yaw_rate_limit(
            torch.tensor([0.045, 0.35, -0.35]),
            maximum_yaw_rate=0.1,
            minimum_turn_radius=0.9,
            envelope_fraction=0.85,
        )
        expected = torch.tensor([0.0425, 0.085, 0.085])
        self.assertTrue(torch.allclose(limit, expected, atol=1.0e-6))

    def test_dynamic_turn_authority_fades_smoothly_at_low_speed(self):
        speeds = torch.tensor([0.03, 0.04, 0.06, 0.08, 0.15])
        limit = feasible_yaw_rate_limit(
            speeds,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=2.125,
            envelope_fraction=0.85,
            turn_authority_start_speed=0.04,
            turn_authority_full_speed=0.08,
        )
        self.assertTrue(
            torch.allclose(
                limit,
                torch.tensor([0.0, 0.0, 0.012, 0.032, 0.060]),
                atol=1.0e-6,
            )
        )

    def test_speed_scheduled_value_preserves_low_and_full_speed_calibration(self):
        values = speed_scheduled_value(
            torch.tensor([0.03, 0.04, 0.06, 0.08, 0.15]),
            low_speed_value=0.75,
            full_speed_value=1.0,
            transition_start_speed=0.04,
            transition_full_speed=0.08,
        )
        self.assertTrue(
            torch.allclose(
                values,
                torch.tensor([0.75, 0.75, 0.875, 1.0, 1.0]),
                atol=1.0e-6,
            )
        )

    def test_dynamic_envelope_projection_rejects_unreachable_low_speed_yaw(self):
        projected = project_velocity_commands(
            torch.tensor([[0.03, 0.10], [0.05, 0.10], [0.08, 0.10]]),
            maximum_forward_speed=0.15,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=2.125,
            envelope_fraction=0.85,
            stationary_threshold=0.02,
            turn_authority_start_speed=0.04,
            turn_authority_full_speed=0.08,
        )
        self.assertTrue(
            torch.allclose(
                projected[:, 1], torch.tensor([0.0, 0.003125, 0.032]), atol=1.0e-6
            )
        )

    def test_projection_clamps_speed_and_unreachable_yaw(self):
        commands = torch.tensor(
            [
                [0.0, 0.1],
                [0.05, 0.1],
                [0.5, -0.2],
                [-0.5, 0.2],
            ]
        )
        projected = project_velocity_commands(
            commands,
            maximum_forward_speed=0.35,
            maximum_yaw_rate=0.1,
            minimum_turn_radius=0.9,
            envelope_fraction=0.85,
            stationary_threshold=0.02,
        )
        expected = torch.tensor(
            [
                [0.0, 0.0],
                [0.05, 0.047222222],
                [0.35, -0.085],
                [-0.35, 0.085],
            ]
        )
        self.assertTrue(torch.allclose(projected, expected, atol=1.0e-6))

    def test_command_mixture_is_complete(self):
        cfg = RotunbotVelCfg.commands
        total = cfg.stop_fraction + cfg.straight_fraction + cfg.turn_fraction
        self.assertAlmostEqual(total, 1.0)
        self.assertEqual(cfg.straight_only_policy_steps, 0)
        self.assertEqual(cfg.mixed_policy_steps, 0)
        self.assertLess(cfg.mixed_turn_fraction, cfg.turn_fraction)
        self.assertGreaterEqual(cfg.turn_fraction, 0.60)
        self.assertGreater(cfg.minimum_turn_command_fraction, 0.0)
        self.assertLess(cfg.minimum_turn_command_fraction, 1.0)
        self.assertGreater(cfg.extreme_turn_fraction, 0.0)
        self.assertLessEqual(cfg.extreme_turn_fraction, 1.0)
        self.assertGreater(cfg.opposite_transition_fraction, 0.0)
        self.assertLessEqual(cfg.opposite_transition_fraction, 1.0)
        self.assertGreater(cfg.minimum_turn_speed, 0.0)
        self.assertGreater(cfg.minimum_turn_radius, 0.0)
        self.assertEqual(cfg.smooth_profile_fraction, 0.0)
        self.assertGreater(RotunbotVelDynamicCfg.commands.smooth_profile_fraction, 0.0)

    def test_policy_action_is_normalized(self):
        self.assertEqual(RotunbotVelCfg.normalization.clip_actions, 1.0)
        self.assertEqual(RotunbotVelCfg.env.num_actions, 2)
        self.assertEqual(RotunbotVelCfg.env.num_observations, 16)
        self.assertFalse(RotunbotVelCfg.init_state.randomize_initial_velocity)
        self.assertEqual(RotunbotVelDynamicRateCfg.env.num_observations, 18)
        self.assertTrue(RotunbotVelDynamicRateCfg.commands.observe_command_rates)
        self.assertEqual(RotunbotVelDynamicPreviewCfg.env.num_observations, 20)
        self.assertTrue(
            RotunbotVelDynamicPreviewCfg.commands.observe_preview_tracking_errors
        )
        self.assertEqual(RotunbotVelDynamicPICfg.env.num_observations, 22)
        self.assertTrue(
            RotunbotVelDynamicPICfg.commands.observe_tracking_error_integrals
        )
        self.assertGreater(RotunbotVelDynamicPICfg.control.angular_integral_gain, 0.0)
        self.assertEqual(
            RotunbotVelDynamicSteadyPICfg.control.integral_command_rate_threshold,
            [1.0e-4, 1.0e-4],
        )
        self.assertGreater(
            RotunbotVelDynamicAnticipatoryCfg.rewards.smooth_angular_reward_preview_time,
            0.0,
        )
        self.assertGreater(
            RotunbotVelDynamicAnticipatoryCfg.commands.smooth_profile_fraction,
            RotunbotVelDynamicSteadyPICfg.commands.smooth_profile_fraction,
        )
        self.assertGreater(
            RotunbotVelDynamicAnticipatoryV20Cfg.rewards.smooth_angular_reward_preview_time,
            RotunbotVelDynamicAnticipatoryCfg.rewards.smooth_angular_reward_preview_time,
        )
        self.assertGreater(
            RotunbotVelDynamicSmoothFeedforwardV21Cfg.control.angular_rate_feedforward_time,
            0.0,
        )
        self.assertGreater(
            RotunbotVelDynamicSmoothBrakeV22Cfg.control.smooth_angular_feedback_gain,
            0.0,
        )
        self.assertEqual(RotunbotVelDynamicAccelerationV23Cfg.env.num_observations, 24)
        self.assertTrue(
            RotunbotVelDynamicAccelerationV23Cfg.commands.observe_tracking_error_derivatives
        )
        self.assertLess(
            RotunbotVelDynamicAccelerationV23Cfg.rewards.scales.angular_acceleration_error,
            0.0,
        )
        self.assertGreater(
            RotunbotVelDynamicDualLeadV24Cfg.control.smooth_linear_command_lead_time,
            RotunbotVelDynamicDualLeadV24Cfg.control.linear_command_lead_time,
        )
        self.assertLess(
            RotunbotVelDynamicDualPhaseV25Cfg.control.smooth_angular_command_lead_time,
            RotunbotVelDynamicDualPhaseV25Cfg.control.angular_command_lead_time,
        )
        self.assertGreater(
            RotunbotVelDynamicDualPhaseV25Cfg.control.smooth_angular_command_lead_time,
            0.0,
        )
        self.assertLess(
            RotunbotVelDynamicDualPhaseV25Cfg.control.smooth_angular_command_gain,
            1.0,
        )
        self.assertGreater(
            RotunbotVelDynamicDualPhaseV25Cfg.control.angular_rate_feedforward_time,
            0.0,
        )
        self.assertGreater(
            RotunbotVelDynamicDualPhaseV25Cfg.control.angular_rate_feedforward_action_limit,
            0.0,
        )
        self.assertFalse(
            RotunbotVelDynamicDualPhaseV25Cfg.control.require_explicit_smooth_profile_for_phase_lead
        )
        self.assertTrue(
            RotunbotVelDynamicExplicitPhaseV26Cfg.control.require_explicit_smooth_profile_for_phase_lead
        )
        self.assertGreater(
            RotunbotVelDynamicExplicitPhaseV26Cfg.control.angular_rate_feedforward_time,
            RotunbotVelDynamicDualPhaseV25Cfg.control.angular_rate_feedforward_time,
        )
        self.assertLess(
            RotunbotVelDynamicExplicitPhaseV26Cfg.control.smooth_angular_command_gain,
            RotunbotVelDynamicDualPhaseV25Cfg.control.smooth_angular_command_gain,
        )
        self.assertEqual(RotunbotVelIndependentV28Cfg.commands.max_forward_speed, 0.15)
        self.assertEqual(
            RotunbotVelIndependentV28Cfg.control.nominal_yaw_gain_speed_slope,
            0.0,
        )
        self.assertGreater(
            RotunbotVelIndependentV28Cfg.commands.independent_smooth_profile_fraction,
            0.5,
        )
        self.assertLess(
            RotunbotVelIndependentV28Cfg.commands.smooth_profile_period_max_s,
            RotunbotVelIndependentV28Cfg.commands.resampling_time + 1.0e-6,
        )
        self.assertEqual(
            RotunbotVelIndependentRobustV29Cfg.commands.max_forward_speed,
            0.15,
        )
        self.assertEqual(
            RotunbotVelIndependentRobustV29Cfg.control.lead_projection_max_forward_speed,
            0.20,
        )
        self.assertTrue(
            RotunbotVelIndependentRobustV29Cfg.control.disable_integral_for_explicit_smooth_profiles
        )
        self.assertEqual(
            RotunbotVelIndependentRobustV29Cfg.commands.governor_projection_max_forward_speed,
            0.20,
        )
        self.assertGreater(
            RotunbotVelIndependentRobustV29Cfg.commands.minimum_turn_radius,
            RotunbotVelIndependentV28Cfg.commands.minimum_turn_radius,
        )
        low_speed_limit = feasible_yaw_rate_limit(
            torch.tensor([0.05]),
            RotunbotVelIndependentRobustV29Cfg.commands.max_yaw_rate,
            RotunbotVelIndependentRobustV29Cfg.commands.minimum_turn_radius,
            RotunbotVelIndependentRobustV29Cfg.commands.feasible_envelope_fraction,
        )
        self.assertLessEqual(float(low_speed_limit.item()), 0.030)
        v30_speeds = torch.tensor([0.05, 0.10, 0.15])
        v30_limits = feasible_yaw_rate_limit(
            v30_speeds,
            RotunbotVelIndependentFeasibleV30Cfg.commands.max_yaw_rate,
            RotunbotVelIndependentFeasibleV30Cfg.commands.minimum_turn_radius,
            RotunbotVelIndependentFeasibleV30Cfg.commands.feasible_envelope_fraction,
        )
        self.assertTrue(
            torch.allclose(v30_limits, torch.tensor([0.02, 0.04, 0.06]))
        )
        self.assertEqual(
            RotunbotVelIndependentDynamicEnvelopeV31Cfg.commands.max_forward_speed,
            0.15,
        )
        self.assertEqual(
            RotunbotVelIndependentDynamicEnvelopeV31Cfg.commands.turn_authority_start_speed,
            0.04,
        )
        self.assertEqual(
            RotunbotVelIndependentDynamicEnvelopeV31Cfg.commands.turn_authority_full_speed,
            0.08,
        )
        self.assertLess(
            RotunbotVelIndependentDynamicEnvelopeV31Cfg.commands.smooth_profile_fraction,
            RotunbotVelIndependentFeasibleV30Cfg.commands.smooth_profile_fraction,
        )
        self.assertEqual(
            RotunbotVelIndependentBrakeGovernorV32Cfg.commands.linear_deceleration_brake_ratio,
            0.50,
        )
        self.assertLess(
            RotunbotVelIndependentBrakeGovernorV32Cfg.commands.reversal_minimum_request_jump_v,
            RotunbotVelIndependentDynamicEnvelopeV31Cfg.commands.reversal_minimum_request_jump_v,
        )
        self.assertEqual(
            RotunbotVelIndependentTargetedBrakeV33Cfg.commands.linear_deceleration_target_speed_max,
            0.04,
        )
        self.assertEqual(
            RotunbotVelIndependentTargetedBrakeV34Cfg.commands.linear_deceleration_target_speed_max,
            0.05,
        )
        self.assertEqual(
            RotunbotVelIndependentScheduledFeedbackV35Cfg.control.low_speed_linear_feedback_gain,
            0.75,
        )
        self.assertEqual(
            RotunbotVelIndependentScheduledFeedbackV35Cfg.control.linear_feedback_transition_full_speed,
            RotunbotVelIndependentDynamicEnvelopeV31Cfg.commands.turn_authority_full_speed,
        )
        self.assertEqual(
            RotunbotVelIndependentWaveRobustV36Cfg.control.angular_rate_feedforward_time,
            0.0,
        )
        self.assertEqual(
            RotunbotVelIndependentWaveRobustV36Cfg.commands.max_forward_speed,
            0.15,
        )
        self.assertEqual(
            RotunbotVelIndependentSafeEnvelopeV37Cfg.commands.max_forward_speed,
            0.13,
        )
        self.assertEqual(
            RotunbotVelIndependentSafeEnvelopeV37Cfg.commands.ranges.lin_vel_x,
            [-0.13, 0.13],
        )
        self.assertEqual(
            RotunbotVelIndependentSafeEnvelopeV37Cfg.commands.minimum_turn_radius,
            2.125,
        )
        self.assertTrue(
            RotunbotVelIndependentYawBrakeV38Cfg.commands.yaw_only_braking
        )
        self.assertEqual(
            RotunbotVelIndependentYawTransitionV39Cfg.commands.max_forward_speed,
            0.13,
        )
        self.assertEqual(
            RotunbotVelIndependentYawTransitionV39Cfg.commands.reversal_detection_w,
            0.010,
        )
        self.assertEqual(
            RotunbotVelIndependentYawTransitionV39Cfg.commands.reversal_minimum_request_jump_w,
            0.015,
        )
        self.assertEqual(
            RotunbotVelIndependentYawTransitionV39Cfg.commands.reversal_release_measured_w,
            0.005,
        )
        self.assertGreater(
            RotunbotVelIndependentYawTransitionV39Cfg.commands.yaw_only_transition_fraction,
            0.0,
        )

    def test_v39_sampler_contains_disjoint_yaw_only_transition_curriculum(self):
        source = inspect.getsource(RotunbotVel._resample_commands)
        self.assertIn("yaw_only_targets[:, 1] *= -1.0", source)
        self.assertIn("opposite_mask = previous_moving & ~yaw_only_mask", source)

    def test_v40_uses_measured_dynamic_turning_envelope(self):
        commands = RotunbotVelIndependentTransitionEnvelopeV40Cfg.commands
        self.assertEqual(commands.max_forward_speed, 0.13)
        self.assertEqual(commands.turn_authority_start_speed, 0.08)
        self.assertEqual(commands.turn_authority_full_speed, 0.10)
        self.assertEqual(commands.minimum_turn_speed, 0.08)
        limits = feasible_yaw_rate_limit(
            torch.tensor([0.06, 0.08, 0.09, 0.10, 0.13]),
            commands.max_yaw_rate,
            commands.minimum_turn_radius,
            commands.feasible_envelope_fraction,
            commands.turn_authority_start_speed,
            commands.turn_authority_full_speed,
        )
        self.assertTrue(
            torch.allclose(
                limits,
                torch.tensor([0.0, 0.0, 0.018, 0.04, 0.052]),
                atol=1.0e-6,
            )
        )

    def test_abrupt_sampler_keeps_low_speed_straight_commands(self):
        source = inspect.getsource(RotunbotVel._resample_commands)
        self.assertIn("small_turn = turn_mask", source)
        self.assertNotIn("small_moving = moving_mask", source)

    def test_target_gap_mask_separates_smooth_update_from_step(self):
        governed = torch.tensor([[0.20, 0.030], [0.20, 0.030]])
        targets = torch.tensor([[0.203, 0.0308], [-0.20, -0.030]])
        mask = command_target_gap_mask(targets, governed, 0.02, 0.005)
        self.assertEqual(mask.tolist(), [True, False])

    def test_smooth_controller_does_not_switch_off_at_zero_rate_plateau(self):
        source = inspect.getsource(RotunbotVel._compute_torques)
        self.assertIn("smooth_lead_enabled = smooth_tracking_enabled", source)
        self.assertNotIn(
            "smooth_lead_enabled |=",
            source,
        )
        callback_source = inspect.getsource(
            RotunbotVel._post_physics_step_callback
        )
        self.assertIn(
            "self.tracking_error_integral[self.command_reference_is_smooth] = 0.0",
            callback_source,
        )

    def test_command_governor_limits_extreme_reversal(self):
        current = torch.tensor([[0.35, 0.085]])
        target = torch.tensor([[-0.35, -0.085]])
        governed = rate_limit_velocity_commands(
            current,
            target,
            maximum_linear_acceleration=0.30,
            maximum_yaw_acceleration=0.04,
            dt=0.04,
            maximum_forward_speed=0.35,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=0.90,
            envelope_fraction=0.85,
            stationary_threshold=0.02,
        )
        self.assertAlmostEqual(float(governed[0, 0]), 0.338, places=6)
        self.assertAlmostEqual(float(governed[0, 1]), 0.0834, places=6)

    def test_command_governor_forbids_yaw_near_zero_speed(self):
        current = torch.tensor([[0.01, 0.02]])
        target = torch.tensor([[-0.35, -0.085]])
        governed = rate_limit_velocity_commands(
            current,
            target,
            maximum_linear_acceleration=0.30,
            maximum_yaw_acceleration=0.04,
            dt=0.04,
            maximum_forward_speed=0.35,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=0.90,
            envelope_fraction=0.85,
            stationary_threshold=0.02,
        )
        self.assertEqual(float(governed[0, 1]), 0.0)

    def test_rate_feedforward_respects_drive_direction_and_limit(self):
        feedforward = velocity_rate_feedforward_actions(
            commands=torch.tensor([[0.30, 0.04], [-0.30, -0.04]]),
            command_rates=torch.tensor([[0.0, 0.02], [0.0, -0.02]]),
            measured_forward_velocity=torch.tensor([0.30, -0.30]),
            angular_preview_time=0.10,
            angular_action_limit=0.01,
        )
        self.assertTrue(torch.all(feedforward[:, 1] < 0.0))
        self.assertTrue(torch.all(torch.abs(feedforward[:, 1]) <= 0.01))
        self.assertTrue(torch.all(feedforward[:, 0] == 0.0))

    def test_error_derivative_feedback_respects_drive_direction(self):
        derivative = velocity_error_derivative_actions(
            commands=torch.tensor([[0.30, 0.04], [-0.30, -0.04]]),
            measured_forward_velocity=torch.tensor([0.30, -0.30]),
            error_derivative=torch.tensor([[0.0, 0.02], [0.0, -0.02]]),
            angular_derivative_gain=0.25,
            angular_action_limit=0.02,
        )
        self.assertTrue(torch.all(derivative[:, 1] < 0.0))
        self.assertTrue(torch.all(torch.abs(derivative[:, 1]) <= 0.02))
        self.assertTrue(torch.all(derivative[:, 0] == 0.0))

    def test_opposite_linear_or_yaw_direction_requires_braking(self):
        current = torch.tensor([[0.30, 0.07], [0.30, 0.07], [0.30, 0.07]])
        target = torch.tensor([[-0.30, -0.07], [0.30, -0.07], [0.20, 0.04]])
        mask = reversal_brake_mask(
            current,
            target,
            measured_forward_speed=torch.tensor([0.30, 0.30, 0.30]),
            measured_yaw_rate=torch.tensor([0.07, 0.07, 0.07]),
            linear_threshold=0.05,
            yaw_threshold=0.015,
        )
        self.assertTrue(torch.equal(mask, torch.tensor([True, True, False])))

    def test_linear_brake_can_exclude_yaw_only_reversal(self):
        mask = reversal_brake_mask(
            current_commands=torch.tensor([[0.10, 0.03], [0.10, 0.03]]),
            target_commands=torch.tensor([[-0.10, -0.03], [0.10, -0.03]]),
            measured_forward_speed=torch.tensor([0.10, 0.10]),
            measured_yaw_rate=torch.tensor([0.03, 0.03]),
            linear_threshold=0.05,
            yaw_threshold=0.015,
            include_yaw=False,
        )
        self.assertTrue(torch.equal(mask, torch.tensor([True, False])))

    def test_yaw_only_brake_detects_reversal_and_large_reduction(self):
        mask = yaw_reversal_brake_mask(
            current_commands=torch.tensor(
                [[0.10, 0.03], [0.10, 0.04], [0.10, 0.03]]
            ),
            target_commands=torch.tensor(
                [[0.10, -0.03], [0.10, 0.01], [0.10, 0.02]]
            ),
            measured_yaw_rate=torch.tensor([0.035, 0.05, 0.03]),
            yaw_threshold=0.015,
            yaw_deceleration_ratio=0.50,
            yaw_deceleration_delta=0.025,
        )
        self.assertTrue(torch.equal(mask, torch.tensor([True, True, False])))

    def test_yaw_only_brake_keeps_linear_target_in_governor(self):
        source = inspect.getsource(RotunbotVel._post_physics_step_callback)
        self.assertIn(
            "effective_targets[self.command_yaw_brake_pending, 1] = 0.0",
            source,
        )

    def test_measured_momentum_triggers_braking_near_zero_command(self):
        mask = reversal_brake_mask(
            current_commands=torch.tensor([[0.0, 0.0]]),
            target_commands=torch.tensor([[-0.30, -0.07]]),
            measured_forward_speed=torch.tensor([0.20]),
            measured_yaw_rate=torch.tensor([0.05]),
            linear_threshold=0.05,
            yaw_threshold=0.015,
        )
        self.assertTrue(bool(mask.item()))

    def test_large_same_direction_yaw_reduction_requires_braking(self):
        mask = reversal_brake_mask(
            current_commands=torch.tensor([[0.30, 0.085], [0.30, 0.085]]),
            target_commands=torch.tensor([[0.30, 0.010], [0.30, 0.060]]),
            measured_forward_speed=torch.tensor([0.30, 0.30]),
            measured_yaw_rate=torch.tensor([0.11, 0.09]),
            linear_threshold=0.05,
            yaw_threshold=0.015,
            yaw_deceleration_ratio=0.50,
            yaw_deceleration_delta=0.025,
        )
        self.assertTrue(torch.equal(mask, torch.tensor([True, False])))

    def test_large_same_direction_linear_reduction_requires_braking(self):
        mask = reversal_brake_mask(
            current_commands=torch.tensor(
                [[0.14, 0.0], [-0.10, 0.0], [0.07, 0.0], [0.14, 0.0]]
            ),
            target_commands=torch.tensor(
                [[0.03, 0.0], [-0.03, 0.0], [0.05, 0.0], [0.09, 0.0]]
            ),
            measured_forward_speed=torch.tensor([0.14, -0.10, 0.07, 0.14]),
            measured_yaw_rate=torch.zeros(4),
            linear_threshold=0.05,
            yaw_threshold=0.015,
            linear_deceleration_ratio=0.50,
            linear_deceleration_delta=0.04,
        )
        self.assertTrue(
            torch.equal(mask, torch.tensor([True, True, False, False]))
        )

    def test_targeted_linear_brake_ignores_moderate_speed_targets(self):
        mask = reversal_brake_mask(
            current_commands=torch.tensor([[0.14, 0.0], [0.14, 0.0]]),
            target_commands=torch.tensor([[0.03, 0.0], [0.06, 0.0]]),
            measured_forward_speed=torch.tensor([0.14, 0.14]),
            measured_yaw_rate=torch.zeros(2),
            linear_threshold=0.05,
            yaw_threshold=0.015,
            linear_deceleration_ratio=0.50,
            linear_deceleration_delta=0.04,
            linear_deceleration_target_speed_max=0.04,
        )
        self.assertTrue(torch.equal(mask, torch.tensor([True, False])))

    def test_only_discontinuous_requests_enable_full_stop_braking(self):
        previous = torch.tensor([[0.30, 0.07], [0.004, 0.001]])
        new = torch.tensor([[-0.30, -0.07], [-0.004, -0.001]])
        mask = command_request_jump_mask(
            previous,
            new,
            minimum_linear_jump=0.10,
            minimum_yaw_jump=0.03,
        )
        self.assertTrue(torch.equal(mask, torch.tensor([True, False])))

    def test_integral_memory_survives_continuous_target_updates(self):
        request_jump = torch.tensor([False, True, True])
        internal_smooth_profile = torch.tensor([False, False, True])
        reset = tracking_integral_reset_mask(
            request_jump,
            internal_smooth_profile,
        )
        self.assertTrue(torch.equal(reset, torch.tensor([False, True, False])))

    def test_rate_gated_integral_accumulates_only_at_steady_command(self):
        updated = update_rate_gated_error_integral(
            error_integral=torch.tensor([[0.10, -0.10], [0.10, -0.10]]),
            tracking_error=torch.tensor([[0.20, -0.20], [0.20, -0.20]]),
            command_rates=torch.tensor([[0.0, 0.0], [0.01, -0.01]]),
            dt=0.04,
            leak_rate=0.5,
            integral_limits=[0.25, 0.25],
            command_rate_thresholds=[1.0e-4, 1.0e-4],
        )
        self.assertTrue(torch.all(torch.abs(updated[0]) > 0.0))
        self.assertTrue(torch.equal(updated[1], torch.zeros(2)))

    def test_nominal_inverse_matches_identified_actuator_map(self):
        commands = torch.tensor(
            [[0.20, 0.08], [0.20, -0.08], [-0.20, 0.08], [0.0, 0.0]]
        )
        actions = nominal_actuator_actions(commands)
        expected = torch.tensor(
            [
                [0.50, -0.08 / (0.0915 + 0.175 * 0.20)],
                [0.50, 0.08 / (0.0915 + 0.175 * 0.20)],
                [-0.50, 0.08 / (0.0915 + 0.175 * 0.20)],
                [0.0, 0.0],
            ]
        )
        self.assertTrue(torch.allclose(actions, expected, atol=1.0e-6))

    def test_command_lead_is_projected_back_into_feasible_set(self):
        led = lead_compensated_velocity_commands(
            commands=torch.tensor([[0.30, 0.06], [-0.30, -0.06]]),
            command_rates=torch.tensor([[0.30, 0.04], [-0.30, -0.04]]),
            linear_lead_time=1.0,
            angular_lead_time=1.0,
            maximum_forward_speed=0.35,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=0.90,
            envelope_fraction=0.85,
            stationary_threshold=0.02,
        )
        expected = torch.tensor([[0.35, 0.085], [-0.35, -0.085]])
        self.assertTrue(torch.allclose(led, expected, atol=1.0e-6))

    def test_smooth_profile_is_feasible_and_crosses_zero_together(self):
        profile = smooth_feasible_velocity_profile(
            phase=torch.tensor([0.0, torch.pi / 2.0, torch.pi]),
            speed_amplitude=torch.tensor([0.30, 0.30, 0.30]),
            signed_curvature=torch.tensor([0.20, 0.20, 0.20]),
            maximum_forward_speed=0.35,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=0.90,
            envelope_fraction=0.85,
            stationary_threshold=0.02,
        )
        self.assertTrue(torch.allclose(profile[0], torch.zeros(2), atol=1.0e-6))
        self.assertTrue(torch.allclose(profile[1], torch.tensor([0.30, 0.06])))
        self.assertTrue(torch.allclose(profile[2], torch.zeros(2), atol=1.0e-6))

    def test_independent_profile_contains_fixed_v_alternating_w_and_is_feasible(self):
        profile = independent_feasible_velocity_profile(
            phase=torch.tensor([0.0, torch.pi / 2.0, torch.pi, 3.0 * torch.pi / 2.0]),
            velocity_offset=torch.full((4,), 0.10),
            velocity_amplitude=torch.zeros(4),
            yaw_amplitude=torch.full((4,), 0.07),
            yaw_phase_offset=torch.zeros(4),
            yaw_frequency_ratio=torch.ones(4),
            maximum_forward_speed=0.15,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=0.90,
            envelope_fraction=0.85,
            stationary_threshold=0.02,
        )
        self.assertTrue(torch.allclose(profile[:, 0], torch.full((4,), 0.10)))
        self.assertTrue(torch.allclose(profile[:, 1], torch.tensor([0.0, 0.07, 0.0, -0.07]), atol=1.0e-6))
        yaw_limit = feasible_yaw_rate_limit(
            profile[:, 0], 0.10, 0.90, envelope_fraction=0.85
        )
        self.assertTrue(torch.all(torch.abs(profile[:, 1]) <= yaw_limit + 1.0e-6))

    def test_residual_controller_is_bounded_around_nominal_map(self):
        self.assertEqual(RotunbotVelCfg.control.residual_action_scale, [0.05, 0.10])
        self.assertGreater(RotunbotVelCfg.control.nominal_forward_speed_per_action, 0.0)
        self.assertGreater(RotunbotVelCfg.control.nominal_yaw_gain_intercept, 0.0)
        self.assertGreater(RotunbotVelCfg.control.nominal_yaw_gain_speed_slope, 0.0)

    def test_velocity_feedback_corrects_tracking_error_and_is_bounded(self):
        commands = torch.tensor([[0.30, 0.07], [-0.30, -0.07]])
        feedback = velocity_error_feedback_actions(
            commands,
            measured_forward_velocity=torch.tensor([0.10, -0.10]),
            measured_yaw_rate=torch.tensor([0.01, -0.01]),
            linear_feedback_gain=0.20,
            linear_action_limit=0.10,
        )
        self.assertTrue(torch.all(feedback[:, 0] * commands[:, 0] > 0.0))
        nominal = nominal_actuator_actions(commands)
        self.assertTrue(torch.all(feedback[:, 1] * nominal[:, 1] > 0.0))
        self.assertTrue(torch.all(torch.abs(feedback[:, 0]) <= 0.10))
        self.assertTrue(torch.all(torch.abs(feedback[:, 1]) <= 0.15))

    def test_velocity_feedback_brakes_yaw_during_governed_stop(self):
        feedback = velocity_error_feedback_actions(
            commands=torch.zeros(2, 2),
            measured_forward_velocity=torch.tensor([0.20, -0.20]),
            measured_yaw_rate=torch.tensor([0.06, -0.06]),
        )
        # For either travel direction, applying this steering correction
        # produces yaw opposite to the measured residual yaw momentum.
        induced_yaw_sign = -torch.sign(torch.tensor([0.20, -0.20])) * torch.sign(
            feedback[:, 1]
        )
        self.assertTrue(
            torch.equal(induced_yaw_sign, -torch.sign(torch.tensor([0.06, -0.06])))
        )

    def test_integral_feedback_is_bounded_and_corrects_yaw_bias(self):
        correction = velocity_error_integral_actions(
            commands=torch.tensor([[0.30, 0.07], [-0.30, -0.07]]),
            measured_forward_velocity=torch.tensor([0.30, -0.30]),
            error_integral=torch.tensor([[0.0, 0.10], [0.0, -0.10]]),
            angular_integral_gain=0.20,
            angular_action_limit=0.08,
        )
        nominal = nominal_actuator_actions(
            torch.tensor([[0.30, 0.07], [-0.30, -0.07]])
        )
        self.assertTrue(torch.all(correction[:, 1] * nominal[:, 1] > 0.0))
        self.assertTrue(torch.all(torch.abs(correction[:, 1]) <= 0.08))

    def test_residual_safety_projection_keeps_only_error_reducing_actions(self):
        filtered = error_aligned_residual_actions(
            actions=torch.tensor([[0.5, -0.5], [-0.5, -0.5]]),
            commands=torch.tensor([[0.30, 0.07], [-0.30, -0.07]]),
            measured_forward_velocity=torch.tensor([0.10, -0.10]),
            measured_yaw_rate=torch.tensor([0.01, -0.01]),
        )
        self.assertTrue(
            torch.allclose(filtered, torch.tensor([[0.5, -0.5], [-0.5, -0.5]]))
        )
        rejected = error_aligned_residual_actions(
            actions=-filtered,
            commands=torch.tensor([[0.30, 0.07], [-0.30, -0.07]]),
            measured_forward_velocity=torch.tensor([0.10, -0.10]),
            measured_yaw_rate=torch.tensor([0.01, -0.01]),
        )
        self.assertTrue(torch.equal(rejected, torch.zeros_like(rejected)))

    def test_preview_alignment_allows_anticipatory_yaw_correction(self):
        commands = torch.tensor([[0.30, 0.04]])
        rates = torch.tensor([[0.0, -0.04]])
        preview = lead_compensated_velocity_commands(
            commands,
            rates,
            linear_lead_time=0.0,
            angular_lead_time=1.0,
            maximum_forward_speed=0.35,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=0.90,
            envelope_fraction=0.85,
            stationary_threshold=0.02,
        )
        action = torch.tensor([[0.0, 0.5]])
        instantaneous = error_aligned_residual_actions(
            action,
            commands,
            measured_forward_velocity=torch.tensor([0.30]),
            measured_yaw_rate=torch.tensor([0.04]),
        )
        anticipatory = error_aligned_residual_actions(
            action,
            preview,
            measured_forward_velocity=torch.tensor([0.30]),
            measured_yaw_rate=torch.tensor([0.04]),
        )
        self.assertEqual(float(instantaneous[0, 1]), 0.0)
        self.assertEqual(float(anticipatory[0, 1]), 0.5)

    def test_custom_torque_controller_uses_effort_dof_mode(self):
        source = inspect.getsource(RotunbotVel._process_dof_props)
        self.assertIn("DOF_MODE_EFFORT", source)
        self.assertIn('props["stiffness"].fill(0.0)', source)
        self.assertIn('props["damping"].fill(0.0)', source)

    def test_straight_yaw_has_a_nonzero_normalized_penalty(self):
        self.assertLess(RotunbotVelCfg.rewards.scales.straight_yaw, 0.0)
        self.assertLess(RotunbotVelCfg.rewards.scales.angular_tracking_error, 0.0)
        source = inspect.getsource(RotunbotVel._reward_straight_yaw)
        self.assertIn("angular_tracking_sigma", source)

    def test_forward_velocity_uses_integrated_heading(self):
        heading = torch.tensor([0.0, 0.0])
        world_velocity = torch.tensor([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]])
        measured = planar_velocity_in_heading_frame(heading, world_velocity)
        self.assertTrue(torch.allclose(measured[:, 0], torch.tensor([0.2, 0.2])))

    def test_heading_frame_rotates_with_yaw(self):
        heading = torch.tensor([torch.pi / 2.0])
        world_velocity = torch.tensor([[0.0, 0.3, 0.0]])
        measured = planar_velocity_in_heading_frame(heading, world_velocity)
        self.assertTrue(torch.allclose(measured[:, 0], torch.tensor([0.3]), atol=1e-6))
        self.assertTrue(torch.allclose(measured[:, 1], torch.tensor([0.0]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
