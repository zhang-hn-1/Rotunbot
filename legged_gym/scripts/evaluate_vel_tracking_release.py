"""Release evaluation for the direct Rotunbot velocity tracker.

The suite deliberately avoids mechanically unreasonable maximum reversals.  It
measures representative constant-state transitions, smooth sine tracking, and
a correlated 5 Hz random walk inside the empirically reachable domain.
"""

import argparse
from collections import deque
import csv
import distutils.version  # noqa: F401 - torch 1.10 tensorboard compatibility
import json
import os
import sys

import isaacgym  # noqa: F401 - must precede torch/task imports
from isaacgym import gymtorch
import numpy as np
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
import torch

from legged_gym.envs import *  # noqa: F401,F403 - task registration
from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import (
    command_update_interval_steps,
    project_velocity_commands,
)
from legged_gym.utils import get_args, task_registry


STEP_CASES = (
    ("forward_accel_turn", 0.08, 0.000, 0.11, 0.020),
    ("forward_change_turn", 0.11, 0.025, 0.09, -0.015),
    ("reverse_accel_turn", -0.08, 0.000, -0.11, -0.020),
    ("reverse_change_turn", -0.11, 0.025, -0.09, -0.015),
)


def _float_pair(text):
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 2 or values[0] > values[1]:
        raise argparse.ArgumentTypeError(
            "expected two ordered comma-separated values, for example 0.7,1.1"
        )
    return values


def _profile_list(text):
    profiles = [item.strip().lower() for item in text.split(",") if item.strip()]
    allowed = {"step", "sine", "random"}
    if not profiles or any(profile not in allowed for profile in profiles):
        raise argparse.ArgumentTypeError(
            "profiles must be a comma-separated subset of step,sine,random"
        )
    return tuple(dict.fromkeys(profiles))


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--release_envs", type=int, default=16)
    parser.add_argument("--release_seed", type=int, default=20260827)
    parser.add_argument("--release_output_dir", type=str, default=None)
    parser.add_argument(
        "--release_profiles", type=_profile_list,
        default=_profile_list("step,sine,random"),
    )
    parser.add_argument("--release_skip_traces", action="store_true")
    parser.add_argument(
        "--release_zero_residual_policy",
        action="store_true",
        help="Evaluate the deterministic nominal+feedback controller with PPO residuals set to zero.",
    )
    parser.add_argument("--step_precondition_s", type=float, default=5.0)
    parser.add_argument("--step_duration_s", type=float, default=5.0)
    parser.add_argument("--step_exclude_s", type=float, default=0.4)
    parser.add_argument("--sine_period_s", type=float, default=16.0)
    parser.add_argument("--sine_cycles", type=int, default=3)
    parser.add_argument("--sine_v_center", type=float, default=0.105)
    parser.add_argument("--sine_v_amplitude", type=float, default=0.020)
    parser.add_argument("--sine_w_amplitude", type=float, default=0.025)
    parser.add_argument("--sine_w_phase_offset", type=float, default=0.35)
    parser.add_argument("--random_duration_s", type=float, default=40.0)
    parser.add_argument("--random_warmup_s", type=float, default=5.0)
    parser.add_argument(
        "--random_increment_correlation", type=float, default=0.80
    )
    parser.add_argument("--random_initial_speed", type=float, default=0.10)
    parser.add_argument("--random_speed_min", type=float, default=0.08)
    parser.add_argument("--random_speed_max", type=float, default=0.13)
    parser.add_argument("--random_v_increment", type=float, default=0.008)
    parser.add_argument("--random_w_increment", type=float, default=0.004)
    parser.add_argument(
        "--random_sampling_mode", choices=("walk", "stratified"), default="walk",
        help=(
            "walk keeps the correlated 5 Hz process; stratified samples speed "
            "and normalized curvature uniformly inside the feasible R bound"
        ),
    )
    parser.add_argument(
        "--random_command_update_probability", type=float, default=1.0,
        help="Probability that one environment receives a new target at each 5 Hz tick.",
    )
    parser.add_argument(
        "--random_direction_flip_probability", type=float, default=0.0,
        help="Conditional probability of reversing v when a new 5 Hz target is drawn.",
    )
    parser.add_argument(
        "--release_allow_transient_projection_gap",
        action="store_true",
        help=(
            "Report, but do not fail on, requested-versus-applied curvature gaps. "
            "Use for held, stratified or reversal commands whose instantaneous "
            "jumps must be smoothed by the feasible-command governor."
        ),
    )
    parser.add_argument("--release_window_s", type=float, default=5.0)
    parser.add_argument("--release_window_stride_s", type=float, default=1.0)
    parser.add_argument("--release_window_v_mae_max", type=float, default=0.010)
    parser.add_argument("--release_window_w_mae_max", type=float, default=0.005)
    parser.add_argument("--release_window_v_p95_max", type=float, default=0.020)
    parser.add_argument("--release_window_w_p95_max", type=float, default=0.010)
    parser.add_argument("--release_noise_level", type=float, default=0.0)
    parser.add_argument("--release_sensor_v_white_std", type=float, default=0.0)
    parser.add_argument("--release_sensor_w_white_std", type=float, default=0.0)
    parser.add_argument("--release_sensor_v_bias_max", type=float, default=0.0)
    parser.add_argument("--release_sensor_w_bias_max", type=float, default=0.0)
    parser.add_argument(
        "--release_sensor_v_drift_std_per_sqrt_s", type=float, default=0.0
    )
    parser.add_argument(
        "--release_sensor_w_drift_std_per_sqrt_s", type=float, default=0.0
    )
    parser.add_argument("--release_sensor_delay_s", type=float, default=0.0)
    parser.add_argument(
        "--release_sensor_dropout_probability", type=float, default=0.0
    )
    parser.add_argument(
        "--release_friction_range", type=_float_pair, default=None,
        help="Enable per-environment friction randomization, e.g. 0.7,1.1.",
    )
    parser.add_argument(
        "--release_added_mass_range", type=_float_pair, default=None,
        help="Enable base-link added-mass randomization in kg, e.g. -2.5,2.5.",
    )
    parser.add_argument("--release_push_interval_s", type=float, default=0.0)
    parser.add_argument("--release_max_push_velocity", type=float, default=0.0)
    parser.add_argument("--release_additive_kick_interval_s", type=float, default=0.0)
    parser.add_argument("--release_additive_kick_velocity", type=float, default=0.0)
    parser.add_argument("--release_angular_feedback_gain", type=float, default=None)
    parser.add_argument(
        "--release_maximum_linear_acceleration", type=float, default=None
    )
    parser.add_argument(
        "--release_maximum_yaw_acceleration", type=float, default=None
    )
    parser.add_argument(
        "--release_nominal_yaw_gain_intercept", type=float, default=None
    )
    parser.add_argument(
        "--release_nominal_yaw_gain_speed_slope", type=float, default=None
    )
    parser.add_argument(
        "--release_wrong_direction_angular_feedback_gain",
        type=float,
        default=None,
    )
    parser.add_argument("--release_angular_feedback_limit", type=float, default=None)
    parser.add_argument(
        "--release_angular_derivative_gain", type=float, default=None
    )
    parser.add_argument(
        "--release_angular_derivative_limit", type=float, default=None
    )
    parser.add_argument(
        "--release_angular_integral_gain", type=float, default=None
    )
    parser.add_argument(
        "--release_angular_integral_limit", type=float, default=None
    )
    parser.add_argument(
        "--release_angular_rate_feedforward_time", type=float, default=None
    )
    parser.add_argument(
        "--release_angular_rate_feedforward_limit", type=float, default=None
    )
    parser.add_argument(
        "--release_residual_yaw_scale",
        type=float,
        default=None,
        help=(
            "Override only the yaw/steering entry of control.residual_action_scale "
            "for checkpoint ablations; the trained network weights are unchanged."
        ),
    )
    parser.add_argument("--release_residual_gate_activation_error", type=float)
    parser.add_argument("--release_residual_gate_release_error", type=float)
    parser.add_argument("--release_residual_gate_full_scale_error", type=float)
    parser.add_argument("--release_residual_gate_activation_time", type=float)
    parser.add_argument("--release_residual_gate_sign_flip_cooldown", type=float)
    parser.add_argument("--release_residual_gate_rate_bypass_start", type=float)
    parser.add_argument("--release_residual_gate_rate_bypass_full", type=float)
    parser.add_argument(
        "--release_residual_gate_force_error_alignment",
        type=int,
        choices=(0, 1),
    )
    original_argv = list(sys.argv)
    diagnostic, remaining = parser.parse_known_args()
    sys.argv = [original_argv[0]] + remaining
    try:
        args = get_args()
    finally:
        sys.argv = original_argv
    supported_tasks = (
        "rotunbot_vel_sru50_v47",
        "rotunbot_vel_sru50_v48",
        "rotunbot_vel_sru50_v49",
        "rotunbot_vel_sru50_v50_30deg_train",
        "rotunbot_vel_sru50_v50_30deg",
        "rotunbot_vel_sru50_v51_30deg_calibrated_train",
        "rotunbot_vel_sru50_v51_30deg_calibrated",
        "rotunbot_vel_sru50_v52_reachable_curvature",
        "rotunbot_vel_sru50_v53_symmetric_bounded",
        "rotunbot_vel_sru50_v54_curvature_governor",
        "rotunbot_vel_sru50_v55_phase_preview_040",
        "rotunbot_vel_sru50_v56_phase_preview_065",
        "rotunbot_vel_sru50_v57_rate_aligned_010",
        "rotunbot_vel_sru50_v58_rate_aligned_015",
        "rotunbot_vel_sru50_v59_calibrated_map",
        "rotunbot_vel_sru50_v60_hybrid_residual",
        "rotunbot_vel_sru50_v61_radius_priority",
        "rotunbot_vel_sru50_v62_safe_yaw_residual",
        "rotunbot_vel_sru50_v62_feasible_transition_manager",
    )
    if args.task not in supported_tasks:
        raise ValueError(
            "Release evaluation requires one of: %s"
            % ", ".join(supported_tasks)
        )
    if not args.load_run or args.checkpoint is None or int(args.checkpoint) < 0:
        raise ValueError("--load_run and an explicit --checkpoint are required")
    if diagnostic.release_envs < len(STEP_CASES):
        raise ValueError("--release_envs must be at least %d" % len(STEP_CASES))
    if diagnostic.sine_period_s <= 0.0 or diagnostic.sine_cycles <= 0:
        raise ValueError("sine period and cycles must be positive")
    if not 0.0 <= diagnostic.random_increment_correlation <= 1.0:
        raise ValueError("--random_increment_correlation must be in [0, 1]")
    if not 0.0 <= diagnostic.random_speed_min <= diagnostic.random_speed_max:
        raise ValueError("random speed bounds must be ordered and non-negative")
    if diagnostic.random_v_increment < 0.0 or diagnostic.random_w_increment < 0.0:
        raise ValueError("random command increments must be non-negative")
    probability_fields = (
        "random_command_update_probability",
        "random_direction_flip_probability",
    )
    for name in probability_fields:
        if not 0.0 <= getattr(diagnostic, name) <= 1.0:
            raise ValueError("--%s must be in [0, 1]" % name)
    if diagnostic.release_window_s <= 0.0:
        raise ValueError("--release_window_s must be positive")
    if diagnostic.release_window_stride_s <= 0.0:
        raise ValueError("--release_window_stride_s must be positive")
    window_limits = (
        "release_window_v_mae_max",
        "release_window_w_mae_max",
        "release_window_v_p95_max",
        "release_window_w_p95_max",
    )
    for name in window_limits:
        if getattr(diagnostic, name) <= 0.0:
            raise ValueError("--%s must be positive" % name)
    if diagnostic.release_noise_level < 0.0:
        raise ValueError("--release_noise_level must be non-negative")
    nonnegative_sensor_fields = (
        "release_sensor_v_white_std",
        "release_sensor_w_white_std",
        "release_sensor_v_bias_max",
        "release_sensor_w_bias_max",
        "release_sensor_v_drift_std_per_sqrt_s",
        "release_sensor_w_drift_std_per_sqrt_s",
        "release_sensor_delay_s",
    )
    for name in nonnegative_sensor_fields:
        if getattr(diagnostic, name) < 0.0:
            raise ValueError("--%s must be non-negative" % name)
    if not 0.0 <= diagnostic.release_sensor_dropout_probability <= 1.0:
        raise ValueError(
            "--release_sensor_dropout_probability must be in [0, 1]"
        )
    if diagnostic.release_push_interval_s < 0.0:
        raise ValueError("--release_push_interval_s must be non-negative")
    if diagnostic.release_max_push_velocity < 0.0:
        raise ValueError("--release_max_push_velocity must be non-negative")
    if diagnostic.release_additive_kick_interval_s < 0.0:
        raise ValueError("--release_additive_kick_interval_s must be non-negative")
    if diagnostic.release_additive_kick_velocity < 0.0:
        raise ValueError("--release_additive_kick_velocity must be non-negative")
    if (
        diagnostic.release_residual_yaw_scale is not None
        and diagnostic.release_residual_yaw_scale < 0.0
    ):
        raise ValueError("--release_residual_yaw_scale must be non-negative")
    gate_nonnegative = (
        "release_residual_gate_activation_error",
        "release_residual_gate_release_error",
        "release_residual_gate_full_scale_error",
        "release_residual_gate_activation_time",
        "release_residual_gate_sign_flip_cooldown",
        "release_residual_gate_rate_bypass_start",
        "release_residual_gate_rate_bypass_full",
    )
    for name in gate_nonnegative:
        value = getattr(diagnostic, name)
        if value is not None and value < 0.0:
            raise ValueError("--%s must be non-negative" % name)
    args.load_run = os.path.abspath(args.load_run)
    args.checkpoint = int(args.checkpoint)
    args.num_envs = int(diagnostic.release_envs)
    for name, value in vars(diagnostic).items():
        setattr(args, name, value)
    return args


def _configure(env_cfg, args):
    env_cfg.seed = int(args.release_seed)
    env_cfg.env.num_envs = args.release_envs
    env_cfg.env.episode_length_s = 180.0
    env_cfg.commands.resampling_time = 10000.0
    env_cfg.commands.smooth_profile_fraction = 0.0
    # Legacy release tasks promise exact request identity and are evaluated in
    # direct mode.  A governed task explicitly opts into evaluating the visible
    # mechanically reachable reference; overriding it here would silently turn
    # the governor off and make requested/applied traces identical.
    if not bool(
        getattr(env_cfg.commands, "release_evaluate_applied_commands", False)
    ):
        env_cfg.commands.direct_command_tracking = True
    env_cfg.noise.add_noise = args.release_noise_level > 0.0
    env_cfg.noise.noise_level = float(args.release_noise_level)
    env_cfg.domain_rand.randomize_friction = args.release_friction_range is not None
    if args.release_friction_range is not None:
        env_cfg.domain_rand.friction_range = list(args.release_friction_range)
    env_cfg.domain_rand.randomize_base_mass = (
        args.release_added_mass_range is not None
    )
    if args.release_added_mass_range is not None:
        env_cfg.domain_rand.added_mass_range = list(args.release_added_mass_range)
    env_cfg.domain_rand.push_robots = (
        args.release_push_interval_s > 0.0
        and args.release_max_push_velocity > 0.0
    )
    if env_cfg.domain_rand.push_robots:
        env_cfg.domain_rand.push_interval_s = float(args.release_push_interval_s)
        env_cfg.domain_rand.max_push_vel_xy = float(
            args.release_max_push_velocity
        )
    overrides = {
        "nominal_yaw_gain_intercept": (
            args.release_nominal_yaw_gain_intercept
        ),
        "nominal_yaw_gain_speed_slope": (
            args.release_nominal_yaw_gain_speed_slope
        ),
        "angular_feedback_gain": args.release_angular_feedback_gain,
        "wrong_direction_angular_feedback_gain": (
            args.release_wrong_direction_angular_feedback_gain
        ),
        "angular_feedback_action_limit": args.release_angular_feedback_limit,
        "angular_derivative_gain": args.release_angular_derivative_gain,
        "angular_derivative_action_limit": args.release_angular_derivative_limit,
        "angular_integral_gain": args.release_angular_integral_gain,
        "angular_integral_action_limit": args.release_angular_integral_limit,
        "angular_rate_feedforward_time": (
            args.release_angular_rate_feedforward_time
        ),
        "angular_rate_feedforward_action_limit": (
            args.release_angular_rate_feedforward_limit
        ),
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(env_cfg.control, name, float(value))
    command_overrides = {
        "maximum_linear_acceleration": (
            args.release_maximum_linear_acceleration
        ),
        "maximum_yaw_acceleration": args.release_maximum_yaw_acceleration,
    }
    for name, value in command_overrides.items():
        if value is not None:
            setattr(env_cfg.commands, name, float(value))
    if args.release_residual_yaw_scale is not None:
        residual_scale = list(env_cfg.control.residual_action_scale)
        if len(residual_scale) != 2:
            raise ValueError("control.residual_action_scale must contain two entries")
        residual_scale[1] = float(args.release_residual_yaw_scale)
        env_cfg.control.residual_action_scale = residual_scale
    gate_overrides = {
        "release_residual_gate_activation_error": (
            "residual_yaw_gate_activation_error"
        ),
        "release_residual_gate_release_error": "residual_yaw_gate_release_error",
        "release_residual_gate_full_scale_error": (
            "residual_yaw_gate_full_scale_error"
        ),
        "release_residual_gate_activation_time": "residual_yaw_gate_activation_time",
        "release_residual_gate_sign_flip_cooldown": (
            "residual_yaw_gate_sign_flip_cooldown"
        ),
        "release_residual_gate_rate_bypass_start": (
            "residual_yaw_gate_rate_bypass_start"
        ),
        "release_residual_gate_rate_bypass_full": (
            "residual_yaw_gate_rate_bypass_full"
        ),
    }
    for argument_name, config_name in gate_overrides.items():
        value = getattr(args, argument_name)
        if value is not None:
            setattr(env_cfg.control, config_name, float(value))
    if args.release_residual_gate_force_error_alignment is not None:
        env_cfg.control.residual_yaw_gate_force_error_alignment = bool(
            args.release_residual_gate_force_error_alignment
        )
    gate_release = float(env_cfg.control.residual_yaw_gate_release_error)
    gate_activation = float(env_cfg.control.residual_yaw_gate_activation_error)
    gate_full = float(env_cfg.control.residual_yaw_gate_full_scale_error)
    if not 0.0 <= gate_release < gate_activation < gate_full:
        raise ValueError(
            "residual yaw gate requires release < activation < full-scale error"
        )
    bypass_start = float(env_cfg.control.residual_yaw_gate_rate_bypass_start)
    bypass_full = float(env_cfg.control.residual_yaw_gate_rate_bypass_full)
    if bypass_full < bypass_start:
        raise ValueError("residual yaw rate bypass requires start <= full")


def _close_env(env):
    try:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
    finally:
        if env.sim is not None:
            env.gym.destroy_sim(env.sim)


def _set_commands(env, commands, smooth=False):
    env.command_reference_is_smooth.fill_(bool(smooth))
    env.set_command_targets(commands)
    env.compute_observations()


def _new_additive_kick_state(env, args, seed_offset):
    enabled = (
        args.release_additive_kick_interval_s > 0.0
        and args.release_additive_kick_velocity > 0.0
    )
    if not enabled:
        return None
    generator = torch.Generator(device=env.device)
    generator.manual_seed(int(args.release_seed) + int(seed_offset))
    return {
        "step": 0,
        "interval_steps": max(
            1,
            int(round(args.release_additive_kick_interval_s / float(env.dt))),
        ),
        "maximum_delta_velocity": float(args.release_additive_kick_velocity),
        "generator": generator,
    }


def _apply_additive_kick(env, state):
    if state is None:
        return
    step = state["step"]
    state["step"] += 1
    if step <= 0 or step % state["interval_steps"] != 0:
        return
    delta = (
        2.0 * torch.rand(
            env.num_envs, 2, generator=state["generator"], device=env.device
        ) - 1.0
    ) * state["maximum_delta_velocity"]
    env.root_states[:, 7:9].add_(delta)
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.root_states)
    )


def _new_velocity_sensor_state(env, args, seed_offset):
    values = (
        args.release_sensor_v_white_std,
        args.release_sensor_w_white_std,
        args.release_sensor_v_bias_max,
        args.release_sensor_w_bias_max,
        args.release_sensor_v_drift_std_per_sqrt_s,
        args.release_sensor_w_drift_std_per_sqrt_s,
        args.release_sensor_delay_s,
        args.release_sensor_dropout_probability,
    )
    if not any(float(value) > 0.0 for value in values):
        return None
    generator = torch.Generator(device=env.device)
    generator.manual_seed(int(args.release_seed) + int(seed_offset))
    bias_limits = torch.as_tensor(
        [args.release_sensor_v_bias_max, args.release_sensor_w_bias_max],
        device=env.device,
    )
    bias = (
        2.0 * torch.rand(
            env.num_envs, 2, generator=generator, device=env.device
        ) - 1.0
    ) * bias_limits
    return {
        "generator": generator,
        "white_std": torch.as_tensor(
            [args.release_sensor_v_white_std, args.release_sensor_w_white_std],
            device=env.device,
        ),
        "bias": bias,
        "drift": torch.zeros(env.num_envs, 2, device=env.device),
        "drift_std_per_sqrt_s": torch.as_tensor(
            [
                args.release_sensor_v_drift_std_per_sqrt_s,
                args.release_sensor_w_drift_std_per_sqrt_s,
            ],
            device=env.device,
        ),
        "delay_steps": int(round(args.release_sensor_delay_s / float(env.dt))),
        "history": None,
        "dropout_probability": float(args.release_sensor_dropout_probability),
        "previous": None,
        "last": None,
    }


def _apply_velocity_sensor(env, state):
    true_velocity = torch.stack(
        (env.tracking_lin_vel[:, 0], env.tracking_ang_vel[:, 2]), dim=1
    ).clone()
    if state is None:
        return true_velocity
    if state["history"] is None:
        state["history"] = deque(
            [true_velocity.clone() for _ in range(state["delay_steps"] + 1)],
            maxlen=state["delay_steps"] + 1,
        )
    else:
        state["history"].append(true_velocity)
    sensed = state["history"][0].clone()
    drift_increment = torch.randn(
        env.num_envs, 2, generator=state["generator"], device=env.device
    )
    state["drift"].add_(
        drift_increment
        * state["drift_std_per_sqrt_s"]
        * np.sqrt(float(env.dt))
    )
    white = torch.randn(
        env.num_envs, 2, generator=state["generator"], device=env.device
    ) * state["white_std"]
    sensed.add_(state["bias"]).add_(state["drift"]).add_(white)
    if state["previous"] is not None and state["dropout_probability"] > 0.0:
        dropout = torch.rand(
            env.num_envs, generator=state["generator"], device=env.device
        ) < state["dropout_probability"]
        sensed[dropout] = state["previous"][dropout]
    state["previous"] = sensed.clone()
    state["last"] = sensed.clone()
    env.tracking_lin_vel[:, 0].copy_(sensed[:, 0])
    env.tracking_ang_vel[:, 2].copy_(sensed[:, 1])
    return sensed


def _step_once(
    env,
    policy,
    commands,
    smooth=False,
    additive_kick_state=None,
    velocity_sensor_state=None,
):
    previous_applied = getattr(env, "applied_feasible_command", env.commands[:, :2]).clone()
    _set_commands(env, commands, smooth=smooth)
    _apply_additive_kick(env, additive_kick_state)
    sensed_velocity = _apply_velocity_sensor(env, velocity_sensor_state)
    if velocity_sensor_state is not None:
        # Rebuild policy observations from the same sensed v/w used by the
        # classical feedback terms in _compute_torques().  post_physics_step()
        # refreshes tracking_* from simulator truth after the control step.
        env.compute_observations()
    with torch.no_grad():
        actions = policy(env.get_observations())
        _, _, _, dones, _ = env.step(actions)
    if torch.any(dones):
        raise RuntimeError(
            "Release evaluation terminated environments %s"
            % torch.nonzero(dones, as_tuple=False).flatten().tolist()
        )
    applied_command = getattr(env, "applied_feasible_command", env.commands[:, :2])
    projected_applied = project_velocity_commands(
        applied_command,
        getattr(
            env.cfg.commands,
            "governor_projection_max_forward_speed",
            env.cfg.commands.max_forward_speed,
        ),
        env.cfg.commands.max_yaw_rate,
        env.cfg.commands.minimum_turn_radius,
        env.cfg.commands.feasible_envelope_fraction,
        stationary_threshold=env.cfg.rewards.stationary_command_threshold,
        preserve_curvature_when_saturating=bool(
            getattr(env.cfg.commands, "preserve_curvature_when_saturating", False)
        ),
        curvature_fraction_breakpoints=getattr(
            env.cfg.commands, "stable_curvature_fraction_breakpoints", None
        ),
        curvature_max_speed_values=getattr(
            env.cfg.commands, "stable_curvature_max_speed_values", None
        ),
    )
    return {
        "requested_v": commands[:, 0].detach().cpu().numpy(),
        "requested_w": commands[:, 1].detach().cpu().numpy(),
        "projected_target_v": env.command_targets[:, 0].detach().cpu().numpy(),
        "projected_target_w": env.command_targets[:, 1].detach().cpu().numpy(),
        "applied_v": applied_command[:, 0].detach().cpu().numpy(),
        "applied_w": applied_command[:, 1].detach().cpu().numpy(),
        "post_projection_delta_v": (
            applied_command[:, 0] - previous_applied[:, 0]
        ).detach().cpu().numpy(),
        "post_projection_delta_w": (
            applied_command[:, 1] - previous_applied[:, 1]
        ).detach().cpu().numpy(),
        "feasible_domain_violation": (
            torch.abs(projected_applied - applied_command).amax(dim=1) > 3.0e-6
        ).detach().cpu().numpy(),
        "measured_v": env.tracking_lin_vel[:, 0].detach().cpu().numpy(),
        "measured_w": env.tracking_ang_vel[:, 2].detach().cpu().numpy(),
        "sensor_v": sensed_velocity[:, 0].detach().cpu().numpy(),
        "sensor_w": sensed_velocity[:, 1].detach().cpu().numpy(),
        "command_rate_v": env.command_rates[:, 0].detach().cpu().numpy(),
        "command_rate_w": env.command_rates[:, 1].detach().cpu().numpy(),
        "residual_action0": env.applied_residual_actions[:, 0].detach().cpu().numpy(),
        "residual_action1": env.applied_residual_actions[:, 1].detach().cpu().numpy(),
        "nominal_action0": env.nominal_policy_actions[:, 0].detach().cpu().numpy(),
        "nominal_action1": env.nominal_policy_actions[:, 1].detach().cpu().numpy(),
        "feedback_action0": env.feedback_policy_actions[:, 0].detach().cpu().numpy(),
        "feedback_action1": env.feedback_policy_actions[:, 1].detach().cpu().numpy(),
        "rate_feedforward_action0": (
            env.rate_feedforward_policy_actions[:, 0].detach().cpu().numpy()
        ),
        "rate_feedforward_action1": (
            env.rate_feedforward_policy_actions[:, 1].detach().cpu().numpy()
        ),
        "action0": env.combined_policy_actions[:, 0].detach().cpu().numpy(),
        "action1": env.combined_policy_actions[:, 1].detach().cpu().numpy(),
        "residual_yaw_gate": env.residual_yaw_error_gate.detach().cpu().numpy(),
        "residual_yaw_gate_active": (
            env.residual_yaw_error_gate_active.detach().cpu().numpy()
        ),
        "transition_state": env.transition_state.detach().cpu().numpy(),
        "transition_active": env.transition_active.detach().cpu().numpy(),
        "transition_settle_counter": env.transition_settle_counter.detach().cpu().numpy(),
    }


def _stack(records):
    return {key: np.asarray([record[key] for record in records]) for key in records[0]}


def _tracking_metrics(samples, cfg, start_step=0):
    requested_v = samples["requested_v"][start_step:]
    requested_w = samples["requested_w"][start_step:]
    applied_v = samples["applied_v"][start_step:]
    applied_w = samples["applied_w"][start_step:]
    measured_v = samples["measured_v"][start_step:]
    measured_w = samples["measured_w"][start_step:]
    use_applied_reference = bool(
        getattr(cfg.commands, "release_evaluate_applied_commands", False)
    )
    reference_v = applied_v if use_applied_reference else requested_v
    reference_w = applied_w if use_applied_reference else requested_w
    v_error = measured_v - reference_v
    w_error = measured_w - reference_w
    v_scale = float(cfg.commands.max_forward_speed)
    w_scale = float(cfg.commands.max_yaw_rate)
    desired_v = reference_v / v_scale
    desired_w = reference_w / w_scale
    actual_v = measured_v / v_scale
    actual_w = measured_w / w_scale
    desired_norm = np.maximum(np.sqrt(desired_v ** 2 + desired_w ** 2), 1.0e-8)
    perpendicular = np.abs(actual_v * desired_w - actual_w * desired_v) / desired_norm
    moving_v = np.abs(reference_v) >= 0.02
    # Near zero yaw rate, a sign comparison is dominated by sensor/numerical
    # noise and by the unavoidable zero crossing of a continuous command.  The
    # controller's curvature reward uses the same 0.01 rad/s meaningful-turn
    # threshold, so release direction accuracy is evaluated on that subset.
    moving_w = np.abs(reference_w) >= 0.01
    v_sign = np.sign(measured_v[moving_v]) == np.sign(reference_v[moving_v])
    w_sign = np.sign(measured_w[moving_w]) == np.sign(reference_w[moving_w])
    requested_v_normalized = requested_v / v_scale
    requested_w_normalized = requested_w / w_scale
    applied_v_normalized = applied_v / v_scale
    applied_w_normalized = applied_w / w_scale
    requested_norm = np.maximum(
        np.sqrt(requested_v_normalized ** 2 + requested_w_normalized ** 2),
        1.0e-8,
    )
    projection_cross = np.abs(
        applied_v_normalized * requested_w_normalized
        - applied_w_normalized * requested_v_normalized
    ) / requested_norm
    speed_edges = np.asarray(
        [0.0, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, v_scale + 1.0e-6]
    )
    speed_hist, _ = np.histogram(np.abs(reference_v), bins=speed_edges)
    radius = float(cfg.commands.minimum_turn_radius)
    curvature_fraction = np.abs(reference_w) * radius / np.maximum(
        np.abs(reference_v), 1.0e-8
    )
    curvature_edges = np.asarray([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.01, np.inf])
    curvature_hist, _ = np.histogram(curvature_fraction, bins=curvature_edges)
    quadrant_counts = {}
    for v_name, v_condition in (
        ("forward", reference_v >= 0.02),
        ("reverse", reference_v <= -0.02),
    ):
        for w_name, w_condition in (
            ("left_positive_w", reference_w >= 0.005),
            ("right_negative_w", reference_w <= -0.005),
            ("straight", np.abs(reference_w) < 0.005),
        ):
            quadrant_counts[v_name + "_" + w_name] = int(
                np.count_nonzero(v_condition & w_condition)
            )
    metrics = {
        "tracking_reference": (
            "applied_feasible_command" if use_applied_reference else "requested_command"
        ),
        "v_mae_mps": float(np.mean(np.abs(v_error))),
        "w_mae_radps": float(np.mean(np.abs(w_error))),
        "v_rmse_mps": float(np.sqrt(np.mean(v_error ** 2))),
        "w_rmse_radps": float(np.sqrt(np.mean(w_error ** 2))),
        "v_p95_abs_error_mps": float(np.percentile(np.abs(v_error), 95)),
        "w_p95_abs_error_radps": float(np.percentile(np.abs(w_error), 95)),
        "v_p99_abs_error_mps": float(np.percentile(np.abs(v_error), 99)),
        "w_p99_abs_error_radps": float(np.percentile(np.abs(w_error), 99)),
        "v_max_abs_error_mps": float(np.max(np.abs(v_error))),
        "w_max_abs_error_radps": float(np.max(np.abs(w_error))),
        "v_sign_correct_ratio": float(np.mean(v_sign)) if v_sign.size else 1.0,
        "w_sign_correct_ratio": float(np.mean(w_sign)) if w_sign.size else 1.0,
        "v_sign_evaluated_samples": int(v_sign.size),
        "w_sign_evaluated_samples": int(w_sign.size),
        "w_sign_evaluation_threshold_radps": 0.01,
        "normalized_curvature_cross_mae": float(np.mean(perpendicular[moving_w]))
        if np.any(moving_w) else 0.0,
        "maximum_command_v_gap_mps": float(np.max(np.abs(applied_v - requested_v))),
        "maximum_command_w_gap_radps": float(np.max(np.abs(applied_w - requested_w))),
        "normalized_request_projection_cross_mae": float(
            np.mean(projection_cross)
        ),
        "combined_action_saturation_ratio": float(
            np.mean(
                (np.abs(samples["action0"][start_step:]) >= 0.999)
                | (np.abs(samples["action1"][start_step:]) >= 0.999)
            )
        ),
        "command_coverage": {
            "speed_abs_mps_edges": [float(value) for value in speed_edges],
            "speed_bin_samples": [int(value) for value in speed_hist],
            "curvature_fraction_edges": [
                "inf" if not np.isfinite(value) else float(value)
                for value in curvature_edges
            ],
            "curvature_bin_samples": [int(value) for value in curvature_hist],
            "quadrant_samples": quadrant_counts,
            "requested_projection_fraction": float(
                np.mean(
                    (np.abs(applied_v - requested_v) > 1.0e-6)
                    | (np.abs(applied_w - requested_w) > 1.0e-6)
                )
            ),
        },
    }
    if "residual_yaw_gate" in samples:
        gate = samples["residual_yaw_gate"][start_step:]
        active = samples["residual_yaw_gate_active"][start_step:]
        metrics["residual_yaw_gate_mean"] = float(np.mean(gate))
        metrics["residual_yaw_gate_active_ratio"] = float(np.mean(active))
        metrics["residual_yaw_gate_p95"] = float(np.percentile(gate, 95))
        abs_command_rate = np.abs(samples["command_rate_w"][start_step:])
        metrics["absolute_yaw_command_rate_p50_radps2"] = float(
            np.percentile(abs_command_rate, 50)
        )
        metrics["absolute_yaw_command_rate_p95_radps2"] = float(
            np.percentile(abs_command_rate, 95)
        )
        metrics["absolute_yaw_command_rate_max_radps2"] = float(
            np.max(abs_command_rate)
        )
    return metrics


def _window_stability_metrics(samples, cfg, args, start_step=0):
    """Expose local failures that aggregate MAE and P95 can hide."""
    use_applied_reference = bool(
        getattr(cfg.commands, "release_evaluate_applied_commands", False)
    )
    reference_prefix = "applied" if use_applied_reference else "requested"
    reference_v = samples[reference_prefix + "_v"]
    reference_w = samples[reference_prefix + "_w"]
    measured_v = samples["measured_v"]
    measured_w = samples["measured_w"]
    dt = float(cfg.sim.dt) * int(cfg.control.decimation)
    window_steps = max(1, int(round(args.release_window_s / dt)))
    stride_steps = max(1, int(round(args.release_window_stride_s / dt)))
    last_start = reference_v.shape[0] - window_steps
    rows = []
    if last_start < start_step:
        return {
            "window_count": 0,
            "all_windows_pass": False,
            "pass_ratio": 0.0,
            "reason": "profile shorter than one requested stability window",
        }
    for begin in range(start_step, last_start + 1, stride_steps):
        end = begin + window_steps
        v_error = measured_v[begin:end] - reference_v[begin:end]
        w_error = measured_w[begin:end] - reference_w[begin:end]
        abs_v = np.abs(v_error)
        abs_w = np.abs(w_error)
        for env_id in range(reference_v.shape[1]):
            meaningful_w = np.abs(reference_w[begin:end, env_id]) >= 0.01
            direction = 1.0
            if np.any(meaningful_w):
                direction = float(np.mean(
                    np.sign(measured_w[begin:end, env_id][meaningful_w])
                    == np.sign(reference_w[begin:end, env_id][meaningful_w])
                ))
            error_sign = np.sign(w_error[:, env_id])
            nonzero = np.abs(w_error[:, env_id]) >= 0.001
            signs = error_sign[nonzero]
            zero_crossings = int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0
            row = {
                "env_id": int(env_id),
                "start_s": float(begin * dt),
                "end_s": float(end * dt),
                "v_mae_mps": float(np.mean(abs_v[:, env_id])),
                "w_mae_radps": float(np.mean(abs_w[:, env_id])),
                "v_p95_abs_error_mps": float(np.percentile(abs_v[:, env_id], 95)),
                "w_p95_abs_error_radps": float(np.percentile(abs_w[:, env_id], 95)),
                "w_max_abs_error_radps": float(np.max(abs_w[:, env_id])),
                "w_direction_correct_ratio": direction,
                "w_error_zero_crossings": zero_crossings,
                "mean_abs_reference_v_mps": float(np.mean(np.abs(reference_v[begin:end, env_id]))),
                "mean_abs_reference_w_radps": float(np.mean(np.abs(reference_w[begin:end, env_id]))),
            }
            row["pass"] = bool(
                row["v_mae_mps"] <= args.release_window_v_mae_max
                and row["w_mae_radps"] <= args.release_window_w_mae_max
                and row["v_p95_abs_error_mps"] <= args.release_window_v_p95_max
                and row["w_p95_abs_error_radps"] <= args.release_window_w_p95_max
            )
            rows.append(row)
    ordered = sorted(
        rows,
        key=lambda item: (
            item["w_p95_abs_error_radps"], item["w_mae_radps"],
            item["v_p95_abs_error_mps"],
        ),
        reverse=True,
    )
    pass_count = sum(1 for row in rows if row["pass"])
    return {
        "window_s": float(args.release_window_s),
        "stride_s": float(args.release_window_stride_s),
        "window_count": int(len(rows)),
        "pass_count": int(pass_count),
        "pass_ratio": float(pass_count / max(len(rows), 1)),
        "all_windows_pass": bool(pass_count == len(rows)),
        "criteria": {
            "v_mae_max_mps": float(args.release_window_v_mae_max),
            "w_mae_max_radps": float(args.release_window_w_mae_max),
            "v_p95_max_mps": float(args.release_window_v_p95_max),
            "w_p95_max_radps": float(args.release_window_w_p95_max),
        },
        "worst_v_mae_mps": float(max(row["v_mae_mps"] for row in rows)),
        "worst_w_mae_radps": float(max(row["w_mae_radps"] for row in rows)),
        "worst_v_p95_abs_error_mps": float(max(
            row["v_p95_abs_error_mps"] for row in rows
        )),
        "worst_w_p95_abs_error_radps": float(max(
            row["w_p95_abs_error_radps"] for row in rows
        )),
        "worst_w_max_abs_error_radps": float(max(
            row["w_max_abs_error_radps"] for row in rows
        )),
        "worst_windows": ordered[:10],
    }


def _phase_lag(reference, measured, dt, maximum_lag_s):
    reference = np.asarray(reference) - np.mean(reference)
    measured = np.asarray(measured) - np.mean(measured)
    correlation = np.correlate(measured, reference, mode="full")
    lags = np.arange(-reference.size + 1, measured.size)
    maximum_steps = int(round(maximum_lag_s / dt))
    valid = np.abs(lags) <= maximum_steps
    return float(lags[valid][np.argmax(correlation[valid])] * dt)


def _sine_response_metrics(reference, measured, dt, maximum_lag_s):
    """Return per-environment sine lag and amplitude statistics.

    Environments deliberately use opposite command signs.  Averaging them
    before estimating lag cancels the reference waveform and produces an
    undefined amplitude ratio.  Estimate each environment independently and
    aggregate only the resulting scalar metrics.
    """
    reference = np.asarray(reference)
    measured = np.asarray(measured)
    lags = []
    amplitudes = []
    for env_id in range(reference.shape[1]):
        ref = reference[:, env_id]
        response = measured[:, env_id]
        lags.append(abs(_phase_lag(ref, response, dt, maximum_lag_s)))
        amplitudes.append(
            float(np.std(response) / max(float(np.std(ref)), 1.0e-8))
        )
    lags = np.asarray(lags, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    return {
        "phase_lag_abs_mean_s": float(np.mean(lags)),
        "phase_lag_abs_p95_s": float(np.percentile(lags, 95)),
        "amplitude_ratio_mean": float(np.mean(amplitudes)),
        "amplitude_ratio_p05": float(np.percentile(amplitudes, 5)),
        "amplitude_ratio_p95": float(np.percentile(amplitudes, 95)),
    }


def _run_steps(env, policy, args):
    initial = torch.tensor(
        [[case[1], case[2]] for case in STEP_CASES],
        dtype=torch.float32, device=env.device,
    )
    final = torch.tensor(
        [[case[3], case[4]] for case in STEP_CASES],
        dtype=torch.float32, device=env.device,
    )
    repeats = int(np.ceil(args.release_envs / len(STEP_CASES)))
    initial = initial.repeat(repeats, 1)[:args.release_envs]
    final = final.repeat(repeats, 1)[:args.release_envs]
    sensor = _new_velocity_sensor_state(env, args, seed_offset=1101)
    for _ in range(int(round(args.step_precondition_s / env.dt))):
        _step_once(env, policy, initial, velocity_sensor_state=sensor)
    additive_kick = _new_additive_kick_state(env, args, seed_offset=101)
    records = []
    for _ in range(int(round(args.step_duration_s / env.dt))):
        records.append(_step_once(
            env, policy, final, additive_kick_state=additive_kick,
            velocity_sensor_state=sensor,
        ))
    samples = _stack(records)
    excluded = int(round(args.step_exclude_s / env.dt))
    metrics = _tracking_metrics(samples, env.cfg, start_step=excluded)
    settled_start = max(excluded, len(records) - int(round(2.0 / env.dt)))
    metrics["settled"] = _tracking_metrics(samples, env.cfg, start_step=settled_start)
    metrics["excluded_initial_transient_s"] = args.step_exclude_s
    return samples, metrics


def _run_sine(env, policy, args):
    hold = command_update_interval_steps(env.dt, 5.0)
    total_steps = int(round(args.sine_cycles * args.sine_period_s / env.dt))
    signs = torch.where(
        torch.arange(args.release_envs, device=env.device) % 2 == 0,
        torch.ones(args.release_envs, device=env.device),
        -torch.ones(args.release_envs, device=env.device),
    )
    command = torch.zeros(args.release_envs, 2, device=env.device)
    records = []
    additive_kick = _new_additive_kick_state(env, args, seed_offset=211)
    sensor = _new_velocity_sensor_state(env, args, seed_offset=1211)
    for step in range(total_steps):
        if step % hold == 0:
            phase = 2.0 * np.pi * step * float(env.dt) / args.sine_period_s
            command[:, 0] = signs * (
                args.sine_v_center + args.sine_v_amplitude * np.sin(phase)
            )
            command[:, 1] = signs * (
                args.sine_w_amplitude
                * np.sin(phase + args.sine_w_phase_offset)
            )
            command.copy_(project_velocity_commands(
                command,
                env.cfg.commands.max_forward_speed,
                env.cfg.commands.max_yaw_rate,
                env.cfg.commands.minimum_turn_radius,
                env.cfg.commands.feasible_envelope_fraction,
                stationary_threshold=env.cfg.rewards.stationary_command_threshold,
                turn_authority_start_speed=env.cfg.commands.turn_authority_start_speed,
                turn_authority_full_speed=env.cfg.commands.turn_authority_full_speed,
            ))
        records.append(_step_once(
            env, policy, command, smooth=True,
            additive_kick_state=additive_kick,
            velocity_sensor_state=sensor,
        ))
    samples = _stack(records)
    warmup = int(round(args.sine_period_s / env.dt))
    metrics = _tracking_metrics(samples, env.cfg, start_step=warmup)
    use_applied_reference = bool(
        getattr(env.cfg.commands, "release_evaluate_applied_commands", False)
    )
    reference_prefix = "applied" if use_applied_reference else "requested"
    v_response = _sine_response_metrics(
        samples[reference_prefix + "_v"][warmup:],
        samples["measured_v"][warmup:],
        env.dt,
        2.0,
    )
    w_response = _sine_response_metrics(
        samples[reference_prefix + "_w"][warmup:],
        samples["measured_w"][warmup:],
        env.dt,
        2.0,
    )
    requested_v_response = _sine_response_metrics(
        samples["requested_v"][warmup:], samples["measured_v"][warmup:],
        env.dt, 2.0,
    )
    requested_w_response = _sine_response_metrics(
        samples["requested_w"][warmup:], samples["measured_w"][warmup:],
        env.dt, 2.0,
    )
    metrics.update({"v_" + key: value for key, value in v_response.items()})
    metrics.update({"w_" + key: value for key, value in w_response.items()})
    metrics.update({
        "requested_to_actual_v_" + key: value
        for key, value in requested_v_response.items()
    })
    metrics.update({
        "requested_to_actual_w_" + key: value
        for key, value in requested_w_response.items()
    })
    return samples, metrics


def _run_random_walk(env, policy, args):
    generator = torch.Generator(device=env.device)
    generator.manual_seed(args.release_seed)
    hold = command_update_interval_steps(env.dt, 5.0)
    total_steps = int(round(args.random_duration_s / env.dt))
    signs = torch.where(
        torch.arange(args.release_envs, device=env.device) % 2 == 0,
        torch.ones(args.release_envs, device=env.device),
        -torch.ones(args.release_envs, device=env.device),
    )
    command = torch.zeros(args.release_envs, 2, device=env.device)
    command[:, 0] = signs * args.random_initial_speed
    increment = torch.zeros_like(command)
    records = []
    additive_kick = _new_additive_kick_state(env, args, seed_offset=307)
    sensor = _new_velocity_sensor_state(env, args, seed_offset=1307)
    for step in range(total_steps):
        if step % hold == 0:
            update_mask = torch.rand(
                args.release_envs, generator=generator, device=env.device
            ) < float(args.random_command_update_probability)
            flip_mask = update_mask & (
                torch.rand(
                    args.release_envs, generator=generator, device=env.device
                ) < float(args.random_direction_flip_probability)
            )
            signs[flip_mask] *= -1.0
            proposed = command.clone()
            if args.random_sampling_mode == "stratified":
                speed = args.random_speed_min + (
                    args.random_speed_max - args.random_speed_min
                ) * torch.rand(
                    args.release_envs, generator=generator, device=env.device
                )
                curvature_fraction = 2.0 * torch.rand(
                    args.release_envs, generator=generator, device=env.device
                ) - 1.0
                proposed[:, 0] = signs * speed
                proposed[:, 1] = curvature_fraction * speed / float(
                    env.cfg.commands.minimum_turn_radius
                )
            else:
                noise = torch.stack(
                    (
                        (
                            2.0 * torch.rand(
                                args.release_envs,
                                generator=generator,
                                device=env.device,
                            ) - 1.0
                        ) * args.random_v_increment,
                        (
                            2.0 * torch.rand(
                                args.release_envs,
                                generator=generator,
                                device=env.device,
                            ) - 1.0
                        ) * args.random_w_increment,
                    ),
                    dim=1,
                )
                correlation = float(args.random_increment_correlation)
                increment.mul_(correlation).add_(noise, alpha=1.0 - correlation)
                speed = torch.clamp(
                    torch.abs(command[:, 0]) + increment[:, 0],
                    args.random_speed_min,
                    args.random_speed_max,
                )
                proposed[:, 0] = signs * speed
                proposed[:, 1] = command[:, 1] + increment[:, 1]
            command[update_mask] = proposed[update_mask]
            command.copy_(project_velocity_commands(
                command,
                env.cfg.commands.max_forward_speed,
                env.cfg.commands.max_yaw_rate,
                env.cfg.commands.minimum_turn_radius,
                env.cfg.commands.feasible_envelope_fraction,
                stationary_threshold=env.cfg.rewards.stationary_command_threshold,
                turn_authority_start_speed=env.cfg.commands.turn_authority_start_speed,
                turn_authority_full_speed=env.cfg.commands.turn_authority_full_speed,
            ))
        records.append(_step_once(
            env, policy, command, smooth=True,
            additive_kick_state=additive_kick,
            velocity_sensor_state=sensor,
        ))
    samples = _stack(records)
    warmup = int(round(args.random_warmup_s / env.dt))
    metrics = _tracking_metrics(samples, env.cfg, start_step=warmup)
    metrics["window_stability"] = _window_stability_metrics(
        samples, env.cfg, args, start_step=warmup
    )
    return samples, metrics


def _checks(
    step=None,
    sine=None,
    random_walk=None,
    allow_transient_projection_gap=False,
):
    checks = {}
    metrics = []
    if step is not None:
        metrics.append(step)
        checks.update({
            "step_dynamic_v_mae": step["v_mae_mps"] <= 0.020,
            "step_dynamic_w_mae": step["w_mae_radps"] <= 0.012,
            "step_settled_v_mae": step["settled"]["v_mae_mps"] <= 0.010,
            "step_settled_w_mae": step["settled"]["w_mae_radps"] <= 0.006,
        })
    if sine is not None:
        metrics.append(sine)
        checks.update({
            "sine_v_mae": sine["v_mae_mps"] <= 0.012,
            "sine_w_mae": sine["w_mae_radps"] <= 0.006,
            "sine_v_lag": sine["v_phase_lag_abs_p95_s"] <= 0.50,
            "sine_w_lag": sine["w_phase_lag_abs_p95_s"] <= 0.60,
            "sine_v_amplitude": (
                sine["v_amplitude_ratio_p05"] >= 0.85
                and sine["v_amplitude_ratio_p95"] <= 1.15
            ),
            "sine_w_amplitude": (
                sine["w_amplitude_ratio_p05"] >= 0.85
                and sine["w_amplitude_ratio_p95"] <= 1.15
            ),
        })
    if random_walk is not None:
        metrics.append(random_walk)
        checks.update({
            "random_v_mae": random_walk["v_mae_mps"] <= 0.015,
            "random_w_mae": random_walk["w_mae_radps"] <= 0.008,
            "random_v_p95": random_walk["v_p95_abs_error_mps"] <= 0.030,
            "random_w_p95": random_walk["w_p95_abs_error_radps"] <= 0.015,
            "random_v_direction": random_walk["v_sign_correct_ratio"] >= 0.99,
            "random_w_direction": random_walk["w_sign_correct_ratio"] >= 0.98,
            "random_all_stability_windows": bool(
                random_walk.get("window_stability", {}).get(
                    "all_windows_pass", False
                )
            ),
        })
    checks["curvature"] = max(
        metric["normalized_curvature_cross_mae"] for metric in metrics
    ) <= 0.10
    governed_reference = any(
        metric.get("tracking_reference") == "applied_feasible_command"
        for metric in metrics
    )
    if governed_reference:
        checks["governed_reference_is_explicit"] = True
        # A discontinuous step cannot retain the old velocity, adopt the new
        # velocity and preserve the new v/w ratio in the same instant while an
        # acceleration limit is active.  Treat that unavoidable transition as
        # a reported governor diagnostic.  The release requirement applies to
        # continuous SRU-like references (sine and correlated random commands),
        # where preserving the requested curvature is physically meaningful.
        continuous_metrics = [
            metric for metric in (sine, random_walk) if metric is not None
        ]
        if continuous_metrics:
            projection_cross_mae = max(
                metric["normalized_request_projection_cross_mae"]
                for metric in continuous_metrics
            )
            if allow_transient_projection_gap:
                checks["governor_transient_projection_gap_is_diagnostic_only"] = True
            else:
                checks["governor_continuous_request_curvature_deviation"] = (
                    projection_cross_mae <= 0.05
                )
    else:
        checks["command_identity"] = max(
            max(
                metric["maximum_command_v_gap_mps"],
                metric["maximum_command_w_gap_radps"],
            )
            for metric in metrics
        ) <= 2.0e-4
    return checks


def _write_trace(path, samples, dt):
    keys = list(samples.keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "env_id"] + keys)
        for step in range(samples[keys[0]].shape[0]):
            for env_id in range(samples[keys[0]].shape[1]):
                writer.writerow(
                    [step * dt, env_id] + [samples[key][step, env_id] for key in keys]
                )


def main():
    args = _parse_args()
    model_path = os.path.join(args.load_run, "model_%d.pt" % args.checkpoint)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _configure(env_cfg, args)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    try:
        runner, _ = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg
        )
        if args.release_zero_residual_policy:
            def policy(observations):
                return torch.zeros(
                    (observations.shape[0], env.num_actions),
                    dtype=observations.dtype,
                    device=observations.device,
                )
        else:
            policy = runner.get_inference_policy(device=env.device)
        samples_by_profile = {}
        metrics_by_profile = {}
        if "step" in args.release_profiles:
            env.reset()
            samples_by_profile["step"], metrics_by_profile["step"] = _run_steps(
                env, policy, args
            )
        if "sine" in args.release_profiles:
            env.reset()
            samples_by_profile["sine"], metrics_by_profile["sine"] = _run_sine(
                env, policy, args
            )
        if "random" in args.release_profiles:
            env.reset()
            samples_by_profile["random"], metrics_by_profile["random"] = (
                _run_random_walk(env, policy, args)
            )
        checks = _checks(
            step=metrics_by_profile.get("step"),
            sine=metrics_by_profile.get("sine"),
            random_walk=metrics_by_profile.get("random"),
            allow_transient_projection_gap=bool(
                args.release_allow_transient_projection_gap
            ),
        )
        summary = {
            "task": args.task,
            "checkpoint": args.checkpoint,
            "checkpoint_path": model_path,
            "policy_dt_s": float(env.dt),
            "physics_hz": 1.0 / float(env.sim_params.dt),
            "low_level_hz": 1.0 / float(env.dt),
            "upper_command_hz": 5.0,
            "evaluation_profile": "representative_non_extreme",
            "zero_residual_policy": bool(args.release_zero_residual_policy),
            "evaluated_profiles": list(args.release_profiles),
            "release_seed": int(args.release_seed),
            "profile_parameters": {
                "step_precondition_s": float(args.step_precondition_s),
                "step_duration_s": float(args.step_duration_s),
                "step_exclude_s": float(args.step_exclude_s),
                "sine_period_s": float(args.sine_period_s),
                "sine_cycles": int(args.sine_cycles),
                "sine_v_center_mps": float(args.sine_v_center),
                "sine_v_amplitude_mps": float(args.sine_v_amplitude),
                "sine_w_amplitude_radps": float(args.sine_w_amplitude),
                "sine_w_phase_offset_rad": float(args.sine_w_phase_offset),
                "random_duration_s": float(args.random_duration_s),
                "random_warmup_s": float(args.random_warmup_s),
                "random_initial_speed_mps": float(args.random_initial_speed),
                "random_speed_range_mps": [
                    float(args.random_speed_min), float(args.random_speed_max)
                ],
                "random_v_increment_mps": float(args.random_v_increment),
                "random_w_increment_radps": float(args.random_w_increment),
                "random_sampling_mode": str(args.random_sampling_mode),
                "random_command_update_probability": float(
                    args.random_command_update_probability
                ),
                "random_direction_flip_probability": float(
                    args.random_direction_flip_probability
                ),
                "stability_window_s": float(args.release_window_s),
                "stability_window_stride_s": float(
                    args.release_window_stride_s
                ),
            },
            "random_increment_correlation": float(
                args.random_increment_correlation
            ),
            "robustness": {
                "observation_noise_level": float(args.release_noise_level),
                "velocity_sensor_white_std": {
                    "v_mps": float(args.release_sensor_v_white_std),
                    "w_radps": float(args.release_sensor_w_white_std),
                },
                "velocity_sensor_uniform_bias_limit": {
                    "v_mps": float(args.release_sensor_v_bias_max),
                    "w_radps": float(args.release_sensor_w_bias_max),
                },
                "velocity_sensor_random_walk_std_per_sqrt_s": {
                    "v_mps": float(
                        args.release_sensor_v_drift_std_per_sqrt_s
                    ),
                    "w_radps": float(
                        args.release_sensor_w_drift_std_per_sqrt_s
                    ),
                },
                "velocity_sensor_delay_s": float(args.release_sensor_delay_s),
                "velocity_sensor_dropout_probability_per_50hz_sample": float(
                    args.release_sensor_dropout_probability
                ),
                "friction_range": (
                    None if args.release_friction_range is None
                    else list(args.release_friction_range)
                ),
                "base_added_mass_range_kg": (
                    None if args.release_added_mass_range is None
                    else list(args.release_added_mass_range)
                ),
                "push_interval_s": (
                    float(args.release_push_interval_s)
                    if env.cfg.domain_rand.push_robots else None
                ),
                "max_push_velocity_mps": (
                    float(args.release_max_push_velocity)
                    if env.cfg.domain_rand.push_robots else None
                ),
                "push_model": (
                    "instantaneous_random_xy_base_velocity"
                    if env.cfg.domain_rand.push_robots else None
                ),
                "additive_kick_interval_s": (
                    float(args.release_additive_kick_interval_s)
                    if args.release_additive_kick_interval_s > 0.0
                    and args.release_additive_kick_velocity > 0.0 else None
                ),
                "maximum_additive_delta_velocity_mps": (
                    float(args.release_additive_kick_velocity)
                    if args.release_additive_kick_interval_s > 0.0
                    and args.release_additive_kick_velocity > 0.0 else None
                ),
                "additive_kick_model": (
                    "current_xy_base_velocity_plus_uniform_delta"
                    if args.release_additive_kick_interval_s > 0.0
                    and args.release_additive_kick_velocity > 0.0 else None
                ),
            },
            "controller": {
                "residual_action_scale": [
                    float(value)
                    for value in env.cfg.control.residual_action_scale
                ],
                "persistent_yaw_residual_gate": bool(
                    getattr(
                        env.cfg.control,
                        "residual_persistent_yaw_error_gate",
                        False,
                    )
                ),
                "persistent_yaw_residual_gate_parameters": {
                    "activation_error_radps": float(
                        getattr(
                            env.cfg.control,
                            "residual_yaw_gate_activation_error",
                            0.010,
                        )
                    ),
                    "release_error_radps": float(
                        getattr(
                            env.cfg.control,
                            "residual_yaw_gate_release_error",
                            0.004,
                        )
                    ),
                    "full_scale_error_radps": float(
                        getattr(
                            env.cfg.control,
                            "residual_yaw_gate_full_scale_error",
                            0.025,
                        )
                    ),
                    "activation_time_s": float(
                        getattr(
                            env.cfg.control,
                            "residual_yaw_gate_activation_time",
                            0.20,
                        )
                    ),
                    "sign_flip_cooldown_s": float(
                        getattr(
                            env.cfg.control,
                            "residual_yaw_gate_sign_flip_cooldown",
                            0.40,
                        )
                    ),
                    "force_error_alignment": bool(
                        getattr(
                            env.cfg.control,
                            "residual_yaw_gate_force_error_alignment",
                            True,
                        )
                    ),
                    "rate_bypass_start_radps2": float(
                        getattr(
                            env.cfg.control,
                            "residual_yaw_gate_rate_bypass_start",
                            float("inf"),
                        )
                    ),
                    "rate_bypass_full_radps2": float(
                        getattr(
                            env.cfg.control,
                            "residual_yaw_gate_rate_bypass_full",
                            float("inf"),
                        )
                    ),
                },
                "angular_feedback_gain": float(
                    env.cfg.control.angular_feedback_gain
                ),
                "angular_feedback_action_limit": float(
                    env.cfg.control.angular_feedback_action_limit
                ),
                "wrong_direction_angular_feedback_gain": (
                    None
                    if getattr(
                        env.cfg.control,
                        "wrong_direction_angular_feedback_gain",
                        None,
                    ) is None
                    else float(
                        env.cfg.control.wrong_direction_angular_feedback_gain
                    )
                ),
                "angular_derivative_gain": float(
                    env.cfg.control.angular_derivative_gain
                ),
                "angular_derivative_action_limit": float(
                    env.cfg.control.angular_derivative_action_limit
                ),
                "angular_integral_gain": float(
                    env.cfg.control.angular_integral_gain
                ),
                "angular_integral_action_limit": float(
                    env.cfg.control.angular_integral_action_limit
                ),
                "angular_rate_feedforward_time": float(
                    env.cfg.control.angular_rate_feedforward_time
                ),
                "angular_rate_feedforward_action_limit": float(
                    env.cfg.control.angular_rate_feedforward_action_limit
                ),
            },
            "command_interface": {
                "direct_command_tracking": bool(
                    env.cfg.commands.direct_command_tracking
                ),
                "tracking_reference": (
                    "applied_feasible_command"
                    if bool(getattr(
                        env.cfg.commands,
                        "release_evaluate_applied_commands",
                        False,
                    ))
                    else "requested_command"
                ),
                "maximum_linear_acceleration_mps2": float(
                    env.cfg.commands.maximum_linear_acceleration
                ),
                "maximum_yaw_acceleration_radps2": float(
                    env.cfg.commands.maximum_yaw_acceleration
                ),
            },
            "governor_curvature_diagnostics": {
                "step_transition_normalized_cross_mae": (
                    None if metrics_by_profile.get("step") is None else float(
                        metrics_by_profile["step"][
                            "normalized_request_projection_cross_mae"
                        ]
                    )
                ),
                "continuous_profile_max_normalized_cross_mae": (
                    None if not any(
                        metrics_by_profile.get(name) is not None
                        for name in ("sine", "random")
                    ) else max(
                        float(metrics_by_profile[name][
                            "normalized_request_projection_cross_mae"
                        ])
                        for name in ("sine", "random")
                        if metrics_by_profile.get(name) is not None
                    )
                ),
                "step_transition_is_release_gate": False,
                "continuous_profiles_are_release_gate": not bool(
                    args.release_allow_transient_projection_gap
                ),
                "transient_projection_gap_allowed": bool(
                    args.release_allow_transient_projection_gap
                ),
            },
            "step": metrics_by_profile.get("step"),
            "sine": metrics_by_profile.get("sine"),
            "random_continuous": metrics_by_profile.get("random"),
            "checks": checks,
            "verdict": "PASS" if all(checks.values()) else "FAIL",
        }
        output_dir = args.release_output_dir or os.path.join(
            args.load_run, "velocity_release", "checkpoint_%d" % args.checkpoint
        )
        os.makedirs(output_dir, exist_ok=True)
        if not args.release_skip_traces:
            trace_names = {
                "step": "step_trace.csv",
                "sine": "sine_trace.csv",
                "random": "random_continuous_trace.csv",
            }
            for profile, samples in samples_by_profile.items():
                _write_trace(
                    os.path.join(output_dir, trace_names[profile]), samples, env.dt
                )
        json_path = os.path.join(output_dir, "release_summary.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("JSON: %s" % json_path)
    finally:
        _close_env(env)


if __name__ == "__main__":
    main()
