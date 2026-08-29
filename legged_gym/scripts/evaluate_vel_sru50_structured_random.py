"""Structured-random stress evaluation for the frozen SRU50 velocity controller.

This evaluator deliberately covers command transitions that ordinary random walks
can miss.  Every run evaluates the same transition families, while magnitudes,
signs and environment assignments are randomized from a recorded seed.  Raw SRU
requests, governed feasible commands and measured velocities are scored
separately so physically impossible requests are not mistaken for controller
tracking failures.
"""

import argparse
import csv
import distutils.version  # noqa: F401 - torch 1.10 tensorboard compatibility
import json
import os
import sys

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch
import numpy as np
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
import torch

from legged_gym.envs import *  # noqa: F401,F403 - task registration
from legged_gym.utils import get_args, task_registry

# Reuse the release evaluator's sensor/kick implementation and exact trace fields.
from evaluate_vel_tracking_release import (  # noqa: E402
    _close_env,
    _new_additive_kick_state,
    _new_velocity_sensor_state,
    _stack,
    _step_once,
)


FAMILY_ORDER = (
    "straight_v_reversal",
    "fixed_w_v_reversal",
    "constant_curvature_reversal",
    "fixed_v_w_reversal",
    "fixed_w_speed_change",
    "fixed_v_yaw_magnitude_change",
    "straight_stop_or_restart",
    "turn_stop_or_restart",
    "infeasible_low_speed_high_yaw",
    "boundary_curvature_jump",
    "all_quadrant_jump",
    "independent_feasible_jump",
)


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--structured_envs", type=int, default=64)
    parser.add_argument("--structured_seed", type=int, default=20260829)
    parser.add_argument(
        "--structured_noise_profile",
        choices=("nominal", "standard"),
        default="nominal",
    )
    parser.add_argument("--structured_precondition_s", type=float, default=5.0)
    parser.add_argument("--structured_precondition_max_s", type=float, default=30.0)
    parser.add_argument("--structured_precondition_window_s", type=float, default=2.0)
    parser.add_argument("--structured_transition_s", type=float, default=8.0)
    parser.add_argument("--structured_settled_s", type=float, default=2.0)
    parser.add_argument("--structured_output_dir", type=str, required=True)
    parser.add_argument(
        "--structured_skip_traces",
        action="store_true",
        help="Write summaries only; full compressed traces are normally retained.",
    )
    original_argv = list(sys.argv)
    diagnostic, remaining = parser.parse_known_args()
    sys.argv = [original_argv[0]] + remaining
    try:
        args = get_args()
    finally:
        sys.argv = original_argv
    if args.task == "anymal_c_flat":
        args.task = "rotunbot_vel_sru50_v62_safe_yaw_residual"
    if not args.load_run:
        raise ValueError("--load_run is required")
    if diagnostic.structured_envs < 12:
        raise ValueError("--structured_envs must be at least 12")
    if diagnostic.structured_precondition_s <= 0.0:
        raise ValueError("--structured_precondition_s must be positive")
    if diagnostic.structured_precondition_max_s < diagnostic.structured_precondition_s:
        raise ValueError("maximum precondition duration must exceed the minimum")
    if diagnostic.structured_precondition_window_s <= 0.0:
        raise ValueError("precondition window duration must be positive")
    if diagnostic.structured_transition_s <= 0.0:
        raise ValueError("--structured_transition_s must be positive")
    if diagnostic.structured_settled_s <= 0.0:
        raise ValueError("--structured_settled_s must be positive")
    if diagnostic.structured_settled_s > diagnostic.structured_transition_s:
        raise ValueError("settled duration cannot exceed transition duration")
    for name, value in vars(diagnostic).items():
        setattr(args, name, value)
    args.load_run = os.path.abspath(args.load_run)
    args.checkpoint = int(args.checkpoint)
    args.num_envs = int(args.structured_envs)
    return args


def _stratified(rng, count, low, high):
    if high <= low:
        return np.full(count, low, dtype=np.float32)
    phase = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    rng.shuffle(phase)
    return (low + (high - low) * phase).astype(np.float32)


def _alternating_sign(count, offset=0):
    return np.where((np.arange(count) + offset) % 2 == 0, 1.0, -1.0).astype(
        np.float32
    )


def _feasible_w(v, fraction, yaw_sign, radius, max_yaw):
    limit = np.minimum(max_yaw, np.abs(v) / radius)
    return (yaw_sign * fraction * limit).astype(np.float32)


def _clip_feasible(commands, radius, max_v, max_w):
    result = np.asarray(commands, dtype=np.float32).copy()
    result[:, 0] = np.clip(result[:, 0], -max_v, max_v)
    yaw_limit = np.minimum(max_w, np.abs(result[:, 0]) / radius)
    result[:, 1] = np.clip(result[:, 1], -yaw_limit, yaw_limit)
    result[np.abs(result[:, 0]) < 1.0e-7, 1] = 0.0
    return result


def generate_transition_family(name, count, seed, radius, max_v, max_w):
    """Return randomized but coverage-constrained [from, to] command arrays."""
    rng = np.random.default_rng(int(seed))
    s_v = _alternating_sign(count, 0)
    s_w = _alternating_sign(count, 1)
    selector = np.arange(count) % 2 == 0
    from_cmd = np.zeros((count, 2), dtype=np.float32)
    to_cmd = np.zeros_like(from_cmd)

    if name == "straight_v_reversal":
        speed_a = _stratified(rng, count, 0.04, max_v)
        speed_b = _stratified(rng, count, 0.04, max_v)
        from_cmd[:, 0] = s_v * speed_a
        to_cmd[:, 0] = -s_v * speed_b

    elif name == "fixed_w_v_reversal":
        yaw_mag = _stratified(rng, count, 0.012, max_w)
        minimum_speed = np.minimum(max_v, radius * yaw_mag + 0.005)
        room = np.maximum(0.0, max_v - minimum_speed)
        speed_a = minimum_speed + room * rng.random(count)
        speed_b = minimum_speed + room * rng.random(count)
        fixed_w = s_w * yaw_mag
        from_cmd[:, 0] = s_v * speed_a
        from_cmd[:, 1] = fixed_w
        to_cmd[:, 0] = -s_v * speed_b
        to_cmd[:, 1] = fixed_w

    elif name == "constant_curvature_reversal":
        speed_a = _stratified(rng, count, 0.04, max_v)
        speed_b = _stratified(rng, count, 0.04, max_v)
        curvature_fraction = _stratified(rng, count, 0.10, 1.0)
        # Use one curvature that is feasible at both endpoint speeds.  Otherwise
        # the maximum-yaw clamp can change w/v at only one endpoint.
        shared_curvature_limit = np.minimum(
            1.0 / radius,
            max_w / np.maximum(np.maximum(speed_a, speed_b), 1.0e-8),
        )
        signed_curvature = s_w * curvature_fraction * shared_curvature_limit
        from_cmd[:, 0] = s_v * speed_a
        from_cmd[:, 1] = signed_curvature * from_cmd[:, 0]
        to_cmd[:, 0] = -s_v * speed_b
        to_cmd[:, 1] = signed_curvature * to_cmd[:, 0]

    elif name == "fixed_v_w_reversal":
        speed = _stratified(rng, count, 0.04, max_v)
        fixed_v = s_v * speed
        fraction = _stratified(rng, count, 0.10, 1.0)
        yaw = _feasible_w(fixed_v, fraction, s_w, radius, max_w)
        from_cmd[:, 0] = fixed_v
        from_cmd[:, 1] = yaw
        to_cmd[:, 0] = fixed_v
        to_cmd[:, 1] = -yaw

    elif name == "fixed_w_speed_change":
        low_speed = _stratified(rng, count, 0.04, min(0.13, max_v))
        high_speed = np.maximum(
            low_speed, _stratified(rng, count, min(0.13, max_v), max_v)
        )
        yaw_fraction = _stratified(rng, count, 0.05, 0.95)
        fixed_w = _feasible_w(low_speed, yaw_fraction, s_w, radius, max_w)
        from_speed = np.where(selector, low_speed, high_speed)
        to_speed = np.where(selector, high_speed, low_speed)
        from_cmd[:, 0] = s_v * from_speed
        from_cmd[:, 1] = fixed_w
        to_cmd[:, 0] = s_v * to_speed
        to_cmd[:, 1] = fixed_w

    elif name == "fixed_v_yaw_magnitude_change":
        speed = _stratified(rng, count, 0.04, max_v)
        fixed_v = s_v * speed
        low_fraction = _stratified(rng, count, 0.0, 0.35)
        high_fraction = _stratified(rng, count, 0.65, 1.0)
        low_w = _feasible_w(fixed_v, low_fraction, s_w, radius, max_w)
        high_w = _feasible_w(fixed_v, high_fraction, s_w, radius, max_w)
        from_cmd[:, 0] = fixed_v
        from_cmd[:, 1] = np.where(selector, low_w, high_w)
        to_cmd[:, 0] = fixed_v
        to_cmd[:, 1] = np.where(selector, high_w, low_w)

    elif name == "straight_stop_or_restart":
        speed = s_v * _stratified(rng, count, 0.04, max_v)
        from_cmd[:, 0] = np.where(selector, speed, 0.0)
        to_cmd[:, 0] = np.where(selector, 0.0, speed)

    elif name == "turn_stop_or_restart":
        speed = s_v * _stratified(rng, count, 0.04, max_v)
        fraction = _stratified(rng, count, 0.15, 1.0)
        yaw = _feasible_w(speed, fraction, s_w, radius, max_w)
        moving = np.stack((speed, yaw), axis=1)
        # The stop-side raw request intentionally retains yaw.  It is physically
        # impossible and must be reported as projection gap, not tracking error.
        turning_stop = np.stack((np.zeros(count, dtype=np.float32), yaw), axis=1)
        from_cmd[:] = np.where(selector[:, None], moving, turning_stop)
        to_cmd[:] = np.where(selector[:, None], turning_stop, moving)

    elif name == "infeasible_low_speed_high_yaw":
        start_speed = s_v * _stratified(rng, count, 0.10, max_v)
        start_fraction = _stratified(rng, count, 0.05, 0.50)
        from_cmd[:, 0] = start_speed
        from_cmd[:, 1] = _feasible_w(
            start_speed, start_fraction, s_w, radius, max_w
        )
        to_cmd[:, 0] = s_v * _stratified(rng, count, 0.0, 0.09)
        to_cmd[:, 1] = s_w * _stratified(rng, count, 0.05, max_w)

    elif name == "boundary_curvature_jump":
        speed_a = _stratified(rng, count, 0.04, max_v)
        speed_b = _stratified(rng, count, 0.04, max_v)
        from_cmd[:, 0] = s_v * speed_a
        from_cmd[:, 1] = s_w * np.minimum(max_w, speed_a / radius)
        # Half preserve the curvature sign while reversing both channels; half
        # keep v sign and reverse the boundary curvature.
        to_cmd[:, 0] = np.where(selector, -s_v * speed_b, s_v * speed_b)
        to_cmd[:, 1] = np.where(
            selector,
            -s_w * np.minimum(max_w, speed_b / radius),
            -s_w * np.minimum(max_w, speed_b / radius),
        )

    elif name == "all_quadrant_jump":
        speed_a = _stratified(rng, count, 0.04, max_v)
        speed_b = _stratified(rng, count, 0.04, max_v)
        fraction_a = _stratified(rng, count, 0.10, 1.0)
        fraction_b = _stratified(rng, count, 0.10, 1.0)
        pattern = np.arange(count) % 4
        to_v_sign = np.where((pattern == 0) | (pattern == 1), -s_v, s_v)
        to_w_sign = np.where((pattern == 0) | (pattern == 2), -s_w, s_w)
        from_cmd[:, 0] = s_v * speed_a
        from_cmd[:, 1] = _feasible_w(
            from_cmd[:, 0], fraction_a, s_w, radius, max_w
        )
        to_cmd[:, 0] = to_v_sign * speed_b
        to_cmd[:, 1] = _feasible_w(
            to_cmd[:, 0], fraction_b, to_w_sign, radius, max_w
        )

    elif name == "independent_feasible_jump":
        from_cmd[:, 0] = s_v * _stratified(rng, count, 0.0, max_v)
        to_cmd[:, 0] = _alternating_sign(count, 1) * _stratified(
            rng, count, 0.0, max_v
        )
        fraction_a = rng.uniform(-1.0, 1.0, size=count).astype(np.float32)
        fraction_b = rng.uniform(-1.0, 1.0, size=count).astype(np.float32)
        from_cmd[:, 1] = fraction_a * np.minimum(
            max_w, np.abs(from_cmd[:, 0]) / radius
        )
        to_cmd[:, 1] = fraction_b * np.minimum(
            max_w, np.abs(to_cmd[:, 0]) / radius
        )

    else:
        raise ValueError("Unknown structured transition family: %s" % name)

    # Every precondition command must be physically feasible.  Selected target
    # families deliberately retain impossible raw requests for projection tests.
    from_cmd = _clip_feasible(from_cmd, radius, max_v, max_w)
    if name not in ("turn_stop_or_restart", "infeasible_low_speed_high_yaw"):
        to_cmd = _clip_feasible(to_cmd, radius, max_v, max_w)
    return from_cmd.astype(np.float32), to_cmd.astype(np.float32)


def _configure_noise_args(args):
    standard = args.structured_noise_profile == "standard"
    args.release_seed = int(args.structured_seed)
    args.release_sensor_v_white_std = 0.003 if standard else 0.0
    args.release_sensor_w_white_std = 0.002 if standard else 0.0
    args.release_sensor_v_bias_max = 0.002 if standard else 0.0
    args.release_sensor_w_bias_max = 0.001 if standard else 0.0
    args.release_sensor_v_drift_std_per_sqrt_s = 0.0002 if standard else 0.0
    args.release_sensor_w_drift_std_per_sqrt_s = 0.0001 if standard else 0.0
    args.release_sensor_delay_s = 0.04 if standard else 0.0
    args.release_sensor_dropout_probability = 0.02 if standard else 0.0
    args.release_additive_kick_interval_s = 8.0 if standard else 0.0
    args.release_additive_kick_velocity = 0.02 if standard else 0.0


def _configure_environment(env_cfg, args):
    standard = args.structured_noise_profile == "standard"
    env_cfg.seed = int(args.structured_seed)
    env_cfg.env.num_envs = int(args.structured_envs)
    env_cfg.env.episode_length_s = max(
        30.0,
        args.structured_precondition_s + args.structured_transition_s + 10.0,
    )
    env_cfg.commands.resampling_time = 10000.0
    env_cfg.commands.smooth_profile_fraction = 0.0
    env_cfg.noise.add_noise = standard
    env_cfg.noise.noise_level = 0.25 if standard else 0.0
    env_cfg.domain_rand.randomize_friction = standard
    if standard:
        env_cfg.domain_rand.friction_range = [0.8, 1.2]
    env_cfg.domain_rand.randomize_base_mass = standard
    if standard:
        env_cfg.domain_rand.added_mass_range = [-2.5, 2.5]
    env_cfg.domain_rand.push_robots = False


def _meaningful_sign_ratio(measured, reference, threshold):
    mask = np.abs(reference) >= threshold
    if not np.any(mask):
        return 1.0
    return float(np.mean(np.sign(measured[mask]) == np.sign(reference[mask])))


def _metric_block(samples, dt, settled_steps):
    applied_v = samples["applied_v"]
    applied_w = samples["applied_w"]
    requested_v = samples["requested_v"]
    requested_w = samples["requested_w"]
    measured_v = samples["measured_v"]
    measured_w = samples["measured_w"]
    v_error = measured_v - applied_v
    w_error = measured_w - applied_w
    request_v_gap = applied_v - requested_v
    request_w_gap = applied_w - requested_w
    settled = slice(max(0, v_error.shape[0] - settled_steps), None)

    def stats(error):
        absolute = np.abs(error)
        return {
            "mae": float(np.mean(absolute)),
            "p95": float(np.percentile(absolute, 95)),
            "p99": float(np.percentile(absolute, 99)),
            "max": float(np.max(absolute)),
        }

    delta_v = np.abs(samples["post_projection_delta_v"])
    delta_w = np.abs(samples["post_projection_delta_w"])
    rate_v_violation = delta_v > 0.002 + 1.0e-6
    rate_w_violation = delta_w > 0.00014 + 1.0e-7
    rate_violation = rate_v_violation | rate_w_violation
    domain_violation = samples["feasible_domain_violation"].astype(bool)
    active = samples["transition_active"].astype(bool)
    state = samples["transition_state"]
    completion = active[:-1] & ~active[1:] if active.shape[0] > 1 else np.zeros_like(active)
    reversal_durations = []
    for env_id in range(active.shape[1]):
        active_indices = np.flatnonzero(active[:, env_id])
        if active_indices.size:
            reversal_durations.append(
                (active_indices[-1] - active_indices[0] + 1) * float(dt)
            )

    yaw_error = measured_w - applied_w
    yaw_sign_error_count = int(
        np.count_nonzero(
            (np.abs(applied_w) >= 0.01)
            & (np.abs(measured_w) >= 0.01)
            & (np.sign(applied_w) != np.sign(measured_w))
        )
    )
    forward_sign_error_count = int(
        np.count_nonzero(
            (np.abs(applied_v) >= 0.02)
            & (np.abs(measured_v) >= 0.02)
            & (np.sign(applied_v) != np.sign(measured_v))
        )
    )

    result = {
        "applied_tracking_v": stats(v_error),
        "applied_tracking_w": stats(w_error),
        "settled_applied_tracking_v": stats(v_error[settled]),
        "settled_applied_tracking_w": stats(w_error[settled]),
        "requested_tracking_v": stats(measured_v - requested_v),
        "requested_tracking_w": stats(measured_w - requested_w),
        "request_projection_v": stats(request_v_gap),
        "request_projection_w": stats(request_w_gap),
        "projection_fraction": float(
            np.mean(
                (np.abs(request_v_gap) > 1.0e-6)
                | (np.abs(request_w_gap) > 1.0e-6)
            )
        ),
        "applied_v_sign_correct_ratio": _meaningful_sign_ratio(
            measured_v, applied_v, 0.02
        ),
        "applied_w_sign_correct_ratio": _meaningful_sign_ratio(
            measured_w, applied_w, 0.01
        ),
        "mean_abs_requested_yaw_integral_gap_rad": float(
            np.mean(np.abs(np.sum(request_w_gap, axis=0) * dt))
        ),
        "maximum_abs_requested_yaw_integral_gap_rad": float(
            np.max(np.abs(np.sum(request_w_gap, axis=0) * dt))
        ),
        "mean_abs_applied_yaw_tracking_integral_rad": float(
            np.mean(np.abs(np.sum(w_error, axis=0) * dt))
        ),
        "maximum_abs_applied_yaw_tracking_integral_rad": float(
            np.max(np.abs(np.sum(w_error, axis=0) * dt))
        ),
        "residual_yaw_gate_mean": float(np.mean(samples["residual_yaw_gate"])),
        "residual_yaw_gate_active_ratio": float(
            np.mean(samples["residual_yaw_gate_active"])
        ),
        "combined_action_saturation_ratio": float(
            np.mean(
                (np.abs(samples["action0"]) >= 0.999)
                | (np.abs(samples["action1"]) >= 0.999)
            )
        ),
        "yaw_sign_error_count": yaw_sign_error_count,
        "forward_direction_error_count": forward_sign_error_count,
        "hidden_projection_jump_count": int(np.count_nonzero(rate_violation)),
        "rate_bound_violation_count": int(np.count_nonzero(rate_violation)),
        "linear_rate_bound_violation_count": int(np.count_nonzero(rate_v_violation)),
        "yaw_rate_bound_violation_count": int(np.count_nonzero(rate_w_violation)),
        "feasible_domain_violation_count": int(np.count_nonzero(domain_violation)),
        "transition_completion_count": int(np.count_nonzero(completion)),
        "transition_timeout_count": 0,
        "transition_active_sample_count": int(np.count_nonzero(active)),
        "settle_sample_count": int(
            np.count_nonzero(state == 2)
        ),
        "mean_reversal_completion_time_s": float(np.mean(reversal_durations))
        if reversal_durations
        else 0.0,
        "p95_reversal_completion_time_s": float(np.percentile(reversal_durations, 95))
        if reversal_durations
        else 0.0,
        "post_settle_overshoot_radps": float(
            np.max(np.abs(yaw_error[state == 0]))
        )
        if np.any(state == 0)
        else 0.0,
        "post_settle_yaw_rebound_count": int(
            np.count_nonzero(
                (state == 0)
                & (np.abs(applied_w) >= 0.01)
                & (np.sign(yaw_error) != np.sign(applied_w))
            )
        ),
    }
    return result


def _per_environment_rows(
    family, seed, noise_profile, from_commands, to_commands, samples, dt, settled_steps
):
    rows = []
    settled = slice(max(0, samples["measured_v"].shape[0] - settled_steps), None)
    for env_id in range(samples["measured_v"].shape[1]):
        applied_v = samples["applied_v"][:, env_id]
        applied_w = samples["applied_w"][:, env_id]
        requested_v = samples["requested_v"][:, env_id]
        requested_w = samples["requested_w"][:, env_id]
        measured_v = samples["measured_v"][:, env_id]
        measured_w = samples["measured_w"][:, env_id]
        row = {
            "family": family,
            "seed": int(seed),
            "noise_profile": noise_profile,
            "environment": env_id,
            "from_v_mps": float(from_commands[env_id, 0]),
            "from_w_radps": float(from_commands[env_id, 1]),
            "to_requested_v_mps": float(to_commands[env_id, 0]),
            "to_requested_w_radps": float(to_commands[env_id, 1]),
            "applied_v_mae_mps": float(np.mean(np.abs(measured_v - applied_v))),
            "applied_w_mae_radps": float(np.mean(np.abs(measured_w - applied_w))),
            "applied_v_p95_mps": float(
                np.percentile(np.abs(measured_v - applied_v), 95)
            ),
            "applied_w_p95_radps": float(
                np.percentile(np.abs(measured_w - applied_w), 95)
            ),
            "settled_applied_v_mae_mps": float(
                np.mean(np.abs(measured_v[settled] - applied_v[settled]))
            ),
            "settled_applied_w_mae_radps": float(
                np.mean(np.abs(measured_w[settled] - applied_w[settled]))
            ),
            "request_projection_v_mae_mps": float(
                np.mean(np.abs(applied_v - requested_v))
            ),
            "request_projection_w_mae_radps": float(
                np.mean(np.abs(applied_w - requested_w))
            ),
            "requested_yaw_integral_gap_rad": float(
                np.sum(applied_w - requested_w) * dt
            ),
            "applied_yaw_tracking_integral_rad": float(
                np.sum(measured_w - applied_w) * dt
            ),
            "applied_v_sign_correct_ratio": _meaningful_sign_ratio(
                measured_v, applied_v, 0.02
            ),
            "applied_w_sign_correct_ratio": _meaningful_sign_ratio(
                measured_w, applied_w, 0.01
            ),
            "maximum_abs_applied_w_error_radps": float(
                np.max(np.abs(measured_w - applied_w))
            ),
            "hidden_projection_jump_count": int(
                np.count_nonzero(
                    (np.abs(samples["post_projection_delta_v"][:, env_id]) > 0.002 + 1.0e-6)
                    | (np.abs(samples["post_projection_delta_w"][:, env_id]) > 0.00014 + 1.0e-7)
                )
            ),
            "rate_bound_violation_count": int(
                np.count_nonzero(
                    (np.abs(samples["post_projection_delta_v"][:, env_id]) > 0.002 + 1.0e-6)
                    | (np.abs(samples["post_projection_delta_w"][:, env_id]) > 0.00014 + 1.0e-7)
                )
            ),
            "feasible_domain_violation_count": int(
                np.count_nonzero(samples["feasible_domain_violation"][:, env_id])
            ),
        }
        rows.append(row)
    return rows


def _write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _trace_subset(samples):
    keys = (
        "requested_v",
        "requested_w",
        "projected_target_v",
        "projected_target_w",
        "applied_v",
        "applied_w",
        "post_projection_delta_v",
        "post_projection_delta_w",
        "feasible_domain_violation",
        "measured_v",
        "measured_w",
        "sensor_v",
        "sensor_w",
        "command_rate_v",
        "command_rate_w",
        "residual_yaw_gate",
        "residual_yaw_gate_active",
        "transition_state",
        "transition_active",
        "transition_settle_counter",
        "action0",
        "action1",
    )
    return {key: samples[key] for key in keys}


def _run_adaptive_precondition(
    env,
    policy,
    commands,
    sensor,
    minimum_steps,
    maximum_steps,
    window_steps,
):
    """Wait until the governed reference and physical response are both stable."""
    recent = []
    check_interval = max(1, int(round(0.5 / float(env.dt))))
    stable = False
    checks = {}
    for step in range(maximum_steps):
        recent.append(
            _step_once(
                env,
                policy,
                commands,
                additive_kick_state=None,
                velocity_sensor_state=sensor,
            )
        )
        if len(recent) > window_steps:
            recent.pop(0)
        if (
            step + 1 < minimum_steps
            or len(recent) < window_steps
            or (step + 1) % check_interval != 0
        ):
            continue
        window = _stack(recent)
        applied_v = window["applied_v"]
        applied_w = window["applied_w"]
        measured_v = window["measured_v"]
        measured_w = window["measured_w"]
        per_env_v_mae = np.mean(np.abs(measured_v - applied_v), axis=0)
        per_env_w_mae = np.mean(np.abs(measured_w - applied_w), axis=0)
        applied_v_drift = np.abs(applied_v[-1] - applied_v[0])
        applied_w_drift = np.abs(applied_w[-1] - applied_w[0])
        checks = {
            "p95_environment_v_mae_mps": float(
                np.percentile(per_env_v_mae, 95)
            ),
            "p95_environment_w_mae_radps": float(
                np.percentile(per_env_w_mae, 95)
            ),
            "maximum_applied_v_window_drift_mps": float(
                np.max(applied_v_drift)
            ),
            "maximum_applied_w_window_drift_radps": float(
                np.max(applied_w_drift)
            ),
        }
        stable = bool(
            checks["p95_environment_v_mae_mps"] <= 0.010
            and checks["p95_environment_w_mae_radps"] <= 0.015
            and checks["maximum_applied_v_window_drift_mps"] <= 5.0e-4
            and checks["maximum_applied_w_window_drift_radps"] <= 5.0e-4
        )
        if stable:
            break
    return {
        "stable": stable,
        "steps": step + 1,
        "duration_s": float((step + 1) * env.dt),
        "window_steps": int(window_steps),
        "final_checks": checks,
    }


def main():
    args = _parse_args()
    _configure_noise_args(args)
    model_path = os.path.join(args.load_run, "model_%d.pt" % args.checkpoint)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _configure_environment(env_cfg, args)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    os.makedirs(args.structured_output_dir, exist_ok=True)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    try:
        runner, _ = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg
        )
        policy = runner.get_inference_policy(device=env.device)
        dt = float(env.dt)
        precondition_steps = int(round(args.structured_precondition_s / dt))
        precondition_max_steps = int(
            round(args.structured_precondition_max_s / dt)
        )
        precondition_window_steps = int(
            round(args.structured_precondition_window_s / dt)
        )
        transition_steps = int(round(args.structured_transition_s / dt))
        settled_steps = int(round(args.structured_settled_s / dt))
        radius = float(env.cfg.commands.minimum_turn_radius)
        max_v = float(env.cfg.commands.max_forward_speed)
        max_w = float(env.cfg.commands.max_yaw_rate)
        family_metrics = {}
        all_rows = []
        precondition_summaries = {}

        for family_index, family in enumerate(FAMILY_ORDER):
            family_seed = int(args.structured_seed) + 1009 * family_index
            from_np, to_np = generate_transition_family(
                family,
                args.structured_envs,
                family_seed,
                radius,
                max_v,
                max_w,
            )
            from_commands = torch.as_tensor(from_np, device=env.device)
            to_commands = torch.as_tensor(to_np, device=env.device)
            env.reset()
            sensor = _new_velocity_sensor_state(
                env, args, seed_offset=3001 + 37 * family_index
            )
            precondition_summary = _run_adaptive_precondition(
                env,
                policy,
                from_commands,
                sensor,
                precondition_steps,
                precondition_max_steps,
                precondition_window_steps,
            )
            precondition_summaries[family] = precondition_summary
            kick = _new_additive_kick_state(
                env, args, seed_offset=4001 + 37 * family_index
            )
            records = []
            for _ in range(transition_steps):
                records.append(
                    _step_once(
                        env,
                        policy,
                        to_commands,
                        additive_kick_state=kick,
                        velocity_sensor_state=sensor,
                    )
                )
            samples = _stack(records)
            metrics = _metric_block(samples, dt, settled_steps)
            per_env = _per_environment_rows(
                family,
                args.structured_seed,
                args.structured_noise_profile,
                from_np,
                to_np,
                samples,
                dt,
                settled_steps,
            )
            worst_env = int(
                max(per_env, key=lambda row: row["applied_w_p95_radps"])[
                    "environment"
                ]
            )
            projection_env = int(
                max(
                    per_env,
                    key=lambda row: row["request_projection_w_mae_radps"],
                )["environment"]
            )
            metrics["worst_applied_w_environment"] = worst_env
            metrics["largest_projection_gap_environment"] = projection_env
            metrics["from_command_ranges"] = {
                "v_mps": [float(np.min(from_np[:, 0])), float(np.max(from_np[:, 0]))],
                "w_radps": [float(np.min(from_np[:, 1])), float(np.max(from_np[:, 1]))],
            }
            metrics["to_requested_command_ranges"] = {
                "v_mps": [float(np.min(to_np[:, 0])), float(np.max(to_np[:, 0]))],
                "w_radps": [float(np.min(to_np[:, 1])), float(np.max(to_np[:, 1]))],
            }
            family_metrics[family] = metrics
            all_rows.extend(per_env)
            if not args.structured_skip_traces:
                trace_path = os.path.join(
                    args.structured_output_dir, family + "_trace.npz"
                )
                np.savez_compressed(
                    trace_path,
                    dt_s=np.asarray(dt),
                    from_commands=from_np,
                    to_requested_commands=to_np,
                    **_trace_subset(samples),
                )
            print(
                "%s: precondition=%.1fs stable=%s v_mae=%.6f w_mae=%.6f w_p95=%.6f "
                "projection=%.3f yaw_request_gap=%.4f rad"
                % (
                    family,
                    precondition_summary["duration_s"],
                    precondition_summary["stable"],
                    metrics["applied_tracking_v"]["mae"],
                    metrics["applied_tracking_w"]["mae"],
                    metrics["applied_tracking_w"]["p95"],
                    metrics["projection_fraction"],
                    metrics["mean_abs_requested_yaw_integral_gap_rad"],
                )
            )

        row_path = os.path.join(
            args.structured_output_dir, "structured_transition_environment_metrics.csv"
        )
        _write_rows(row_path, all_rows)
        completion_times = [
            family_metrics[family]["mean_reversal_completion_time_s"]
            for family in FAMILY_ORDER
            if family_metrics[family]["mean_reversal_completion_time_s"] > 0.0
        ]
        completion_p95_times = [
            family_metrics[family]["p95_reversal_completion_time_s"]
            for family in FAMILY_ORDER
            if family_metrics[family]["p95_reversal_completion_time_s"] > 0.0
        ]
        overall = {
            "applied_v_mae_mps": float(
                np.mean([row["applied_v_mae_mps"] for row in all_rows])
            ),
            "applied_w_mae_radps": float(
                np.mean([row["applied_w_mae_radps"] for row in all_rows])
            ),
            "p95_environment_applied_v_mae_mps": float(
                np.percentile([row["applied_v_mae_mps"] for row in all_rows], 95)
            ),
            "p95_environment_applied_w_mae_radps": float(
                np.percentile([row["applied_w_mae_radps"] for row in all_rows], 95)
            ),
            "maximum_environment_applied_w_p95_radps": float(
                max(row["applied_w_p95_radps"] for row in all_rows)
            ),
            "mean_request_projection_v_mae_mps": float(
                np.mean([row["request_projection_v_mae_mps"] for row in all_rows])
            ),
            "mean_request_projection_w_mae_radps": float(
                np.mean([row["request_projection_w_mae_radps"] for row in all_rows])
            ),
            "mean_abs_requested_yaw_integral_gap_rad": float(
                np.mean(
                    [abs(row["requested_yaw_integral_gap_rad"]) for row in all_rows]
                )
            ),
            "minimum_applied_v_sign_correct_ratio": float(
                min(row["applied_v_sign_correct_ratio"] for row in all_rows)
            ),
            "minimum_applied_w_sign_correct_ratio": float(
                min(row["applied_w_sign_correct_ratio"] for row in all_rows)
            ),
            "transitions": len(all_rows),
            "hidden_projection_jump_count": int(
                sum(row["hidden_projection_jump_count"] for row in all_rows)
            ),
            "rate_bound_violation_count": int(
                sum(row["rate_bound_violation_count"] for row in all_rows)
            ),
            "feasible_domain_violation_count": int(
                sum(row["feasible_domain_violation_count"] for row in all_rows)
            ),
            "transition_completion_count": int(
                sum(
                    family_metrics[family]["transition_completion_count"]
                    for family in FAMILY_ORDER
                )
            ),
            "transition_timeout_count": int(
                sum(
                    family_metrics[family]["transition_timeout_count"]
                    for family in FAMILY_ORDER
                )
            ),
            "mean_reversal_completion_time_s": float(
                np.mean(completion_times) if completion_times else 0.0
            ),
            "p95_reversal_completion_time_s": float(
                np.percentile(completion_p95_times, 95)
                if completion_p95_times
                else 0.0
            ),
        }
        summary = {
            "task": args.task,
            "checkpoint": int(args.checkpoint),
            "checkpoint_path": model_path,
            "structured_seed": int(args.structured_seed),
            "noise_profile": args.structured_noise_profile,
            "environments": int(args.structured_envs),
            "families": list(FAMILY_ORDER),
            "transitions": len(all_rows),
            "policy_dt_s": dt,
            "physics_hz": 1.0 / float(env.sim_params.dt),
            "low_level_hz": 1.0 / dt,
            "upper_command_hz": 5.0,
            "precondition_s": float(args.structured_precondition_s),
            "precondition_max_s": float(args.structured_precondition_max_s),
            "precondition_window_s": float(args.structured_precondition_window_s),
            "transition_s": float(args.structured_transition_s),
            "settled_s": float(args.structured_settled_s),
            "command_domain": {
                "max_abs_v_mps": max_v,
                "max_abs_w_radps": max_w,
                "minimum_turn_radius_m": radius,
                "no_in_place_yaw": True,
            },
            "noise": {
                "observation_noise_level": 0.25
                if args.structured_noise_profile == "standard"
                else 0.0,
                "sensor_v_white_std_mps": args.release_sensor_v_white_std,
                "sensor_w_white_std_radps": args.release_sensor_w_white_std,
                "sensor_v_bias_max_mps": args.release_sensor_v_bias_max,
                "sensor_w_bias_max_radps": args.release_sensor_w_bias_max,
                "sensor_delay_s": args.release_sensor_delay_s,
                "sensor_dropout_probability": args.release_sensor_dropout_probability,
                "friction_range": [0.8, 1.2]
                if args.structured_noise_profile == "standard"
                else None,
                "added_mass_range_kg": [-2.5, 2.5]
                if args.structured_noise_profile == "standard"
                else None,
                "kick_interval_s": args.release_additive_kick_interval_s,
                "kick_max_velocity_mps": args.release_additive_kick_velocity,
            },
            "tracking_reference": "applied_feasible_command",
            "overall": overall,
            "precondition_summaries": precondition_summaries,
            "family_metrics": family_metrics,
        }
        summary_path = os.path.join(
            args.structured_output_dir, "structured_random_summary.json"
        )
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(json.dumps(overall, indent=2, sort_keys=True))
        print("CSV:", row_path)
        print("JSON:", summary_path)
    finally:
        _close_env(env)


if __name__ == "__main__":
    main()
