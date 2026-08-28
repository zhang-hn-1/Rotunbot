"""Release evaluation for the direct Rotunbot velocity tracker.

The suite deliberately avoids mechanically unreasonable maximum reversals.  It
measures representative constant-state transitions, smooth sine tracking, and
a correlated 5 Hz random walk inside the empirically reachable domain.
"""

import argparse
import csv
import json
import os
import sys

import isaacgym  # noqa: F401 - must precede torch/task imports
import numpy as np
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


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--release_envs", type=int, default=16)
    parser.add_argument("--release_seed", type=int, default=20260827)
    parser.add_argument("--release_output_dir", type=str, default=None)
    parser.add_argument(
        "--release_asset_name",
        type=str,
        default=None,
        help="Override only the Rotunbot URDF basename for transfer tests.",
    )
    parser.add_argument("--step_precondition_s", type=float, default=5.0)
    parser.add_argument("--step_duration_s", type=float, default=5.0)
    parser.add_argument("--step_exclude_s", type=float, default=0.4)
    parser.add_argument("--sine_period_s", type=float, default=16.0)
    parser.add_argument("--sine_cycles", type=int, default=3)
    parser.add_argument("--random_duration_s", type=float, default=40.0)
    parser.add_argument("--random_warmup_s", type=float, default=5.0)
    parser.add_argument(
        "--random_increment_correlation", type=float, default=0.80
    )
    parser.add_argument("--release_angular_feedback_gain", type=float, default=None)
    parser.add_argument("--release_angular_feedback_limit", type=float, default=None)
    parser.add_argument(
        "--release_angular_rate_feedforward_time", type=float, default=None
    )
    parser.add_argument(
        "--release_angular_rate_feedforward_limit", type=float, default=None
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
        "rotunbot_vel_sru50_v49_integration",
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
    args.load_run = os.path.abspath(args.load_run)
    args.checkpoint = int(args.checkpoint)
    args.num_envs = int(diagnostic.release_envs)
    for name, value in vars(diagnostic).items():
        setattr(args, name, value)
    return args


def _configure(env_cfg, args):
    env_cfg.env.num_envs = args.release_envs
    env_cfg.env.episode_length_s = 180.0
    env_cfg.commands.resampling_time = 10000.0
    env_cfg.commands.smooth_profile_fraction = 0.0
    env_cfg.commands.direct_command_tracking = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    if args.release_asset_name is not None:
        allowed_assets = {"Rotunbot_test2.urdf", "Rotunbot.urdf"}
        if args.release_asset_name not in allowed_assets:
            raise ValueError(
                "--release_asset_name must be Rotunbot_test2.urdf or Rotunbot.urdf"
            )
        env_cfg.asset.file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/Rotunbot/urdf/"
            + args.release_asset_name
        )
    overrides = {
        "angular_feedback_gain": args.release_angular_feedback_gain,
        "angular_feedback_action_limit": args.release_angular_feedback_limit,
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


def _step_once(env, policy, commands, smooth=False):
    _set_commands(env, commands, smooth=smooth)
    with torch.no_grad():
        actions = policy(env.get_observations())
        _, _, _, dones, _ = env.step(actions)
    if torch.any(dones):
        raise RuntimeError(
            "Release evaluation terminated environments %s"
            % torch.nonzero(dones, as_tuple=False).flatten().tolist()
        )
    return {
        "requested_v": commands[:, 0].detach().cpu().numpy(),
        "requested_w": commands[:, 1].detach().cpu().numpy(),
        "applied_v": env.commands[:, 0].detach().cpu().numpy(),
        "applied_w": env.commands[:, 1].detach().cpu().numpy(),
        "measured_v": env.tracking_lin_vel[:, 0].detach().cpu().numpy(),
        "measured_w": env.tracking_ang_vel[:, 2].detach().cpu().numpy(),
        "action0": env.combined_policy_actions[:, 0].detach().cpu().numpy(),
        "action1": env.combined_policy_actions[:, 1].detach().cpu().numpy(),
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
    v_error = measured_v - requested_v
    w_error = measured_w - requested_w
    v_scale = float(cfg.commands.max_forward_speed)
    w_scale = float(cfg.commands.max_yaw_rate)
    desired_v = requested_v / v_scale
    desired_w = requested_w / w_scale
    actual_v = measured_v / v_scale
    actual_w = measured_w / w_scale
    desired_norm = np.maximum(np.sqrt(desired_v ** 2 + desired_w ** 2), 1.0e-8)
    perpendicular = np.abs(actual_v * desired_w - actual_w * desired_v) / desired_norm
    moving_v = np.abs(requested_v) >= 0.02
    # Near zero yaw rate, a sign comparison is dominated by sensor/numerical
    # noise and by the unavoidable zero crossing of a continuous command.  The
    # controller's curvature reward uses the same 0.01 rad/s meaningful-turn
    # threshold, so release direction accuracy is evaluated on that subset.
    moving_w = np.abs(requested_w) >= 0.01
    v_sign = np.sign(measured_v[moving_v]) == np.sign(requested_v[moving_v])
    w_sign = np.sign(measured_w[moving_w]) == np.sign(requested_w[moving_w])
    return {
        "v_mae_mps": float(np.mean(np.abs(v_error))),
        "w_mae_radps": float(np.mean(np.abs(w_error))),
        "v_rmse_mps": float(np.sqrt(np.mean(v_error ** 2))),
        "w_rmse_radps": float(np.sqrt(np.mean(w_error ** 2))),
        "v_p95_abs_error_mps": float(np.percentile(np.abs(v_error), 95)),
        "w_p95_abs_error_radps": float(np.percentile(np.abs(w_error), 95)),
        "v_sign_correct_ratio": float(np.mean(v_sign)) if v_sign.size else 1.0,
        "w_sign_correct_ratio": float(np.mean(w_sign)) if w_sign.size else 1.0,
        "v_sign_evaluated_samples": int(v_sign.size),
        "w_sign_evaluated_samples": int(w_sign.size),
        "w_sign_evaluation_threshold_radps": 0.01,
        "normalized_curvature_cross_mae": float(np.mean(perpendicular[moving_w]))
        if np.any(moving_w) else 0.0,
        "maximum_command_v_gap_mps": float(np.max(np.abs(applied_v - requested_v))),
        "maximum_command_w_gap_radps": float(np.max(np.abs(applied_w - requested_w))),
        "combined_action_saturation_ratio": float(
            np.mean(
                (np.abs(samples["action0"][start_step:]) >= 0.999)
                | (np.abs(samples["action1"][start_step:]) >= 0.999)
            )
        ),
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
    for _ in range(int(round(args.step_precondition_s / env.dt))):
        _step_once(env, policy, initial)
    records = []
    for _ in range(int(round(args.step_duration_s / env.dt))):
        records.append(_step_once(env, policy, final))
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
    for step in range(total_steps):
        if step % hold == 0:
            phase = 2.0 * np.pi * step * float(env.dt) / args.sine_period_s
            command[:, 0] = signs * (0.105 + 0.020 * np.sin(phase))
            command[:, 1] = signs * (0.025 * np.sin(phase + 0.35))
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
        records.append(_step_once(env, policy, command, smooth=True))
    samples = _stack(records)
    warmup = int(round(args.sine_period_s / env.dt))
    metrics = _tracking_metrics(samples, env.cfg, start_step=warmup)
    v_response = _sine_response_metrics(
        samples["requested_v"][warmup:], samples["measured_v"][warmup:],
        env.dt, 2.0,
    )
    w_response = _sine_response_metrics(
        samples["requested_w"][warmup:], samples["measured_w"][warmup:],
        env.dt, 2.0,
    )
    metrics.update({"v_" + key: value for key, value in v_response.items()})
    metrics.update({"w_" + key: value for key, value in w_response.items()})
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
    command[:, 0] = signs * 0.10
    increment = torch.zeros_like(command)
    records = []
    for step in range(total_steps):
        if step % hold == 0:
            noise = torch.stack(
                (
                    (
                        2.0 * torch.rand(
                            args.release_envs,
                            generator=generator,
                            device=env.device,
                        ) - 1.0
                    ) * 0.008,
                    (
                        2.0 * torch.rand(
                            args.release_envs,
                            generator=generator,
                            device=env.device,
                        ) - 1.0
                    ) * 0.004,
                ),
                dim=1,
            )
            correlation = float(args.random_increment_correlation)
            increment.mul_(correlation).add_(noise, alpha=1.0 - correlation)
            speed = torch.clamp(
                torch.abs(command[:, 0]) + increment[:, 0], 0.08, 0.13
            )
            command[:, 0] = signs * speed
            command[:, 1] += increment[:, 1]
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
        records.append(_step_once(env, policy, command, smooth=True))
    samples = _stack(records)
    warmup = int(round(args.random_warmup_s / env.dt))
    return samples, _tracking_metrics(samples, env.cfg, start_step=warmup)


def _checks(step, sine, random_walk):
    checks = {
        "step_dynamic_v_mae": step["v_mae_mps"] <= 0.020,
        "step_dynamic_w_mae": step["w_mae_radps"] <= 0.012,
        "step_settled_v_mae": step["settled"]["v_mae_mps"] <= 0.010,
        "step_settled_w_mae": step["settled"]["w_mae_radps"] <= 0.006,
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
        "random_v_mae": random_walk["v_mae_mps"] <= 0.015,
        "random_w_mae": random_walk["w_mae_radps"] <= 0.008,
        "random_v_p95": random_walk["v_p95_abs_error_mps"] <= 0.030,
        "random_w_p95": random_walk["w_p95_abs_error_radps"] <= 0.015,
        "random_v_direction": random_walk["v_sign_correct_ratio"] >= 0.99,
        "random_w_direction": random_walk["w_sign_correct_ratio"] >= 0.98,
        "curvature": max(
            step["normalized_curvature_cross_mae"],
            sine["normalized_curvature_cross_mae"],
            random_walk["normalized_curvature_cross_mae"],
        ) <= 0.10,
        "command_identity": max(
            step["maximum_command_v_gap_mps"], step["maximum_command_w_gap_radps"],
            sine["maximum_command_v_gap_mps"], sine["maximum_command_w_gap_radps"],
            random_walk["maximum_command_v_gap_mps"], random_walk["maximum_command_w_gap_radps"],
        ) <= 2.0e-4,
    }
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
        policy = runner.get_inference_policy(device=env.device)
        env.reset()
        step_samples, step_metrics = _run_steps(env, policy, args)
        env.reset()
        sine_samples, sine_metrics = _run_sine(env, policy, args)
        env.reset()
        random_samples, random_metrics = _run_random_walk(env, policy, args)
        checks = _checks(step_metrics, sine_metrics, random_metrics)
        summary = {
            "task": args.task,
            "checkpoint": args.checkpoint,
            "checkpoint_path": model_path,
            "policy_dt_s": float(env.dt),
            "physics_hz": 1.0 / float(env.sim_params.dt),
            "low_level_hz": 1.0 / float(env.dt),
            "upper_command_hz": 5.0,
            "evaluation_profile": "representative_non_extreme",
            "random_increment_correlation": float(
                args.random_increment_correlation
            ),
            "controller": {
                "angular_feedback_gain": float(
                    env.cfg.control.angular_feedback_gain
                ),
                "angular_feedback_action_limit": float(
                    env.cfg.control.angular_feedback_action_limit
                ),
                "angular_rate_feedforward_time": float(
                    env.cfg.control.angular_rate_feedforward_time
                ),
                "angular_rate_feedforward_action_limit": float(
                    env.cfg.control.angular_rate_feedforward_action_limit
                ),
            },
            "step": step_metrics,
            "sine": sine_metrics,
            "random_continuous": random_metrics,
            "checks": checks,
            "verdict": "PASS" if all(checks.values()) else "FAIL",
        }
        output_dir = args.release_output_dir or os.path.join(
            args.load_run, "velocity_release", "checkpoint_%d" % args.checkpoint
        )
        os.makedirs(output_dir, exist_ok=True)
        _write_trace(os.path.join(output_dir, "step_trace.csv"), step_samples, env.dt)
        _write_trace(os.path.join(output_dir, "sine_trace.csv"), sine_samples, env.dt)
        _write_trace(os.path.join(output_dir, "random_continuous_trace.csv"), random_samples, env.dt)
        json_path = os.path.join(output_dir, "release_summary.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("JSON: %s" % json_path)
    finally:
        _close_env(env)


if __name__ == "__main__":
    main()
