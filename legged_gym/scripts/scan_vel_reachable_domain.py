"""Measure the stable reachable ``(v, yaw-rate)`` domain of Rotunbot.

The scan starts every command from the same nominal rest state, waits for the
closed-loop controller to settle, and then measures tracking accuracy and
oscillation.  A point is accepted only when the command actually seen by the
controller equals the requested command during the measurement window.  This
prevents a command governor from making an unreachable request look successful.
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
    feasible_yaw_rate_limit,
    nominal_actuator_actions,
)
from legged_gym.utils import get_args, task_registry


def _float_list(text):
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--scan_v_values",
        type=_float_list,
        default=_float_list("-0.13,-0.10,-0.08,-0.06,0,0.06,0.08,0.10,0.13"),
    )
    parser.add_argument(
        "--scan_yaw_fractions",
        type=_float_list,
        default=_float_list("-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1"),
    )
    parser.add_argument("--scan_settle_steps", type=int, default=250)
    parser.add_argument("--scan_measure_steps", type=int, default=250)
    parser.add_argument("--scan_output_dir", type=str, default=None)
    parser.add_argument(
        "--scan_controller", choices=("policy", "feedforward"), default="policy"
    )
    parser.add_argument(
        "--scan_direct_command_contract",
        action="store_true",
        help="Bypass the command governor when the selected task supports it.",
    )
    parser.add_argument("--scan_v_mae_max", type=float, default=0.010)
    parser.add_argument("--scan_w_mae_max", type=float, default=0.006)
    parser.add_argument("--scan_v_std_max", type=float, default=0.008)
    parser.add_argument("--scan_w_std_max", type=float, default=0.006)
    parser.add_argument("--scan_command_gap_max", type=float, default=2.0e-4)
    parser.add_argument("--scan_action_saturation_max", type=float, default=0.10)
    parser.add_argument("--scan_angular_feedback_gain", type=float, default=None)
    parser.add_argument("--scan_minimum_turn_radius", type=float, default=None)
    original_argv = list(sys.argv)
    diagnostic, remaining = parser.parse_known_args()
    sys.argv = [original_argv[0]] + remaining
    try:
        args = get_args()
    finally:
        sys.argv = original_argv

    if not args.task.startswith("rotunbot_vel"):
        raise ValueError("scan_vel_reachable_domain.py supports Rotunbot velocity tasks")
    if diagnostic.scan_controller == "policy" and not args.load_run:
        raise ValueError("--load_run is required for policy scans")
    if diagnostic.scan_controller == "policy" and (
        args.checkpoint is None or int(args.checkpoint) < 0
    ):
        raise ValueError("--checkpoint must be an explicit non-negative integer")
    if diagnostic.scan_settle_steps <= 0 or diagnostic.scan_measure_steps <= 0:
        raise ValueError("scan settle/measure steps must be positive")

    args.load_run = os.path.abspath(args.load_run) if args.load_run else None
    args.checkpoint = int(args.checkpoint) if args.checkpoint is not None else -1
    for name, value in vars(diagnostic).items():
        setattr(args, name, value)
    return args


def _build_cases(env_cfg, v_values, yaw_fractions):
    cases = []
    seen = set()
    for requested_v in v_values:
        v = float(np.clip(requested_v, -env_cfg.commands.max_forward_speed,
                          env_cfg.commands.max_forward_speed))
        speed_tensor = torch.tensor([v], dtype=torch.float32)
        yaw_limit = float(
            feasible_yaw_rate_limit(
                speed_tensor,
                env_cfg.commands.max_yaw_rate,
                env_cfg.commands.minimum_turn_radius,
                env_cfg.commands.feasible_envelope_fraction,
                getattr(env_cfg.commands, "turn_authority_start_speed", 0.0),
                getattr(env_cfg.commands, "turn_authority_full_speed", 0.0),
            )[0]
        )
        fractions = [0.0] if yaw_limit < 1.0e-8 else yaw_fractions
        for fraction in fractions:
            w = float(np.clip(fraction, -1.0, 1.0) * yaw_limit)
            key = (round(v, 7), round(w, 7))
            if key in seen:
                continue
            seen.add(key)
            cases.append(("v_%+.3f_w_%+.4f" % (v, w), v, w, float(fraction)))
    return cases


def _configure_environment(env_cfg, case_count, total_steps, direct_contract):
    env_cfg.env.num_envs = case_count
    env_cfg.env.episode_length_s = max(60.0, total_steps * 0.02 + 10.0)
    env_cfg.commands.resampling_time = 10000.0
    env_cfg.commands.smooth_profile_fraction = 0.0
    if direct_contract:
        env_cfg.commands.direct_command_tracking = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False


def _close_env(env):
    try:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
    finally:
        if env.sim is not None:
            env.gym.destroy_sim(env.sim)


def _set_commands(env, commands):
    if hasattr(env, "set_command_targets"):
        env.set_command_targets(commands)
    else:
        env.commands[:, :2].copy_(commands)
    env.compute_observations()


def _feedforward_actions(env, commands):
    if hasattr(env.cfg.control, "residual_action_scale"):
        return torch.zeros_like(commands)
    return nominal_actuator_actions(commands)


def _metrics(case, samples, cfg, args):
    _, command_v, command_w, yaw_fraction = case
    measured_v = samples["v"]
    measured_w = samples["w"]
    applied = samples["applied"]
    combined = samples["combined"]
    requested_output = samples["requested_output"]
    executed_output = samples["executed_output"]
    command_gap_v = np.abs(applied[:, 0] - command_v)
    command_gap_w = np.abs(applied[:, 1] - command_w)
    v_error = measured_v - command_v
    w_error = measured_w - command_w

    v_scale = max(float(cfg.commands.max_forward_speed), 1.0e-8)
    w_scale = max(float(cfg.commands.max_yaw_rate), 1.0e-8)
    desired = np.asarray([command_v / v_scale, command_w / w_scale])
    actual = np.stack((measured_v / v_scale, measured_w / w_scale), axis=1)
    desired_norm = max(float(np.linalg.norm(desired)), 1.0e-8)
    perpendicular = (
        actual[:, 0] * desired[1] - actual[:, 1] * desired[0]
    ) / desired_norm

    moving_v = abs(command_v) >= 0.02
    moving_w = abs(command_w) >= 0.005
    v_sign_ratio = (
        float(np.mean(np.sign(measured_v) == np.sign(command_v)))
        if moving_v else 1.0
    )
    w_sign_ratio = (
        float(np.mean(np.sign(measured_w) == np.sign(command_w)))
        if moving_w else 1.0
    )
    mean_v = float(np.mean(measured_v))
    mean_w = float(np.mean(measured_w))
    desired_radius = command_v / command_w if moving_w else None
    actual_radius = mean_v / mean_w if moving_w and abs(mean_w) > 1.0e-5 else None
    radius_relative_error = (
        abs(actual_radius - desired_radius) / max(abs(desired_radius), 1.0e-8)
        if desired_radius is not None and actual_radius is not None else None
    )

    checks = {
        "v_mae": float(np.mean(np.abs(v_error))) <= args.scan_v_mae_max,
        "w_mae": float(np.mean(np.abs(w_error))) <= args.scan_w_mae_max,
        "v_stability": float(np.std(measured_v)) <= args.scan_v_std_max,
        "w_stability": float(np.std(measured_w)) <= args.scan_w_std_max,
        "v_direction": v_sign_ratio >= 0.99,
        "w_direction": w_sign_ratio >= 0.99,
        "command_identity": (
            float(np.max(command_gap_v)) <= args.scan_command_gap_max
            and float(np.max(command_gap_w)) <= args.scan_command_gap_max
        ),
        "action_saturation": float(np.mean(np.abs(combined) >= 0.999))
        <= args.scan_action_saturation_max,
    }
    return {
        "case": case[0],
        "command_v_mps": command_v,
        "command_w_radps": command_w,
        "yaw_limit_fraction": yaw_fraction,
        "mean_v_mps": mean_v,
        "mean_w_radps": mean_w,
        "v_mae_mps": float(np.mean(np.abs(v_error))),
        "w_mae_radps": float(np.mean(np.abs(w_error))),
        "v_p95_abs_error_mps": float(np.percentile(np.abs(v_error), 95)),
        "w_p95_abs_error_radps": float(np.percentile(np.abs(w_error), 95)),
        "v_std_mps": float(np.std(measured_v)),
        "w_std_radps": float(np.std(measured_w)),
        "v_sign_correct_ratio": v_sign_ratio,
        "w_sign_correct_ratio": w_sign_ratio,
        "mean_abs_normalized_curvature_cross_error": float(
            np.mean(np.abs(perpendicular))
        ),
        "desired_turn_radius_m": desired_radius,
        "measured_turn_radius_m": actual_radius,
        "turn_radius_relative_error": radius_relative_error,
        "maximum_command_v_gap_mps": float(np.max(command_gap_v)),
        "maximum_command_w_gap_radps": float(np.max(command_gap_w)),
        "combined_action_saturation_ratio": float(
            np.mean(np.abs(combined) >= 0.999)
        ),
        "controller_target_rate_limited_ratio": float(
            np.mean(np.abs(requested_output - executed_output) > 1.0e-5)
        ),
        "checks": checks,
        "stable_reachable": bool(all(checks.values())),
    }


def _summary(rows, args, task, checkpoint, policy_dt):
    passed = [row for row in rows if row["stable_reachable"]]
    turning = [row for row in passed if abs(row["command_w_radps"]) >= 0.005]
    boundaries = []
    for speed in sorted(set(abs(row["command_v_mps"]) for row in rows)):
        speed_rows = [
            row for row in passed if abs(abs(row["command_v_mps"]) - speed) < 1.0e-7
        ]
        boundaries.append({
            "abs_v_mps": speed,
            "maximum_passing_abs_w_radps": (
                max(abs(row["command_w_radps"]) for row in speed_rows)
                if speed_rows else None
            ),
            "passing_points": len(speed_rows),
        })
    return {
        "task": task,
        "checkpoint": checkpoint,
        "policy_dt_s": policy_dt,
        "criteria": {
            "v_mae_max_mps": args.scan_v_mae_max,
            "w_mae_max_radps": args.scan_w_mae_max,
            "v_std_max_mps": args.scan_v_std_max,
            "w_std_max_radps": args.scan_w_std_max,
            "command_gap_max": args.scan_command_gap_max,
            "action_saturation_max": args.scan_action_saturation_max,
        },
        "points": len(rows),
        "stable_reachable_points": len(passed),
        "stable_reachable_fraction": len(passed) / max(len(rows), 1),
        "turning_points_passing": len(turning),
        "maximum_passing_v_mae_mps": max(
            (row["v_mae_mps"] for row in passed), default=None
        ),
        "maximum_passing_w_mae_radps": max(
            (row["w_mae_radps"] for row in passed), default=None
        ),
        "maximum_passing_turn_radius_relative_error": max(
            (row["turn_radius_relative_error"] for row in turning
             if row["turn_radius_relative_error"] is not None),
            default=None,
        ),
        "empirical_boundaries": boundaries,
        "verdict": "PASS" if len(passed) == len(rows) else "PARTIAL",
    }


def _write_reports(output_dir, rows, summary):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "reachable_domain_points.csv")
    json_path = os.path.join(output_dir, "reachable_domain_summary.json")
    flattened = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "checks"}
        for name, value in row["checks"].items():
            flat["check_" + name] = value
        flattened.append(flat)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0].keys()))
        writer.writeheader()
        writer.writerows(flattened)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return csv_path, json_path


def main():
    args = _parse_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.scan_angular_feedback_gain is not None:
        env_cfg.control.angular_feedback_gain = float(
            args.scan_angular_feedback_gain
        )
    if args.scan_minimum_turn_radius is not None:
        env_cfg.commands.minimum_turn_radius = float(
            args.scan_minimum_turn_radius
        )
    cases = _build_cases(env_cfg, args.scan_v_values, args.scan_yaw_fractions)
    total_steps = args.scan_settle_steps + args.scan_measure_steps
    _configure_environment(
        env_cfg, len(cases), total_steps, args.scan_direct_command_contract
    )
    args.num_envs = len(cases)
    model_path = None
    if args.scan_controller == "policy":
        model_path = os.path.join(args.load_run, "model_%d.pt" % args.checkpoint)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        train_cfg.runner.resume = True
        train_cfg.runner.load_run = args.load_run
        train_cfg.runner.checkpoint = args.checkpoint

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    try:
        policy = None
        if args.scan_controller == "policy":
            runner, _ = task_registry.make_alg_runner(
                env=env, name=args.task, args=args, train_cfg=train_cfg
            )
            policy = runner.get_inference_policy(device=env.device)
        commands = torch.tensor(
            [[case[1], case[2]] for case in cases],
            dtype=torch.float32,
            device=env.device,
        )
        env.reset()
        _set_commands(env, commands)
        records = {key: [] for key in (
            "v", "w", "applied", "combined", "requested_output", "executed_output"
        )}
        with torch.no_grad():
            for step in range(total_steps):
                _set_commands(env, commands)
                observations = env.get_observations()
                actions = policy(observations) if policy is not None else _feedforward_actions(env, commands)
                _, _, _, dones, _ = env.step(actions)
                if torch.any(dones):
                    raise RuntimeError(
                        "Reachability scan terminated environments %s"
                        % torch.nonzero(dones, as_tuple=False).flatten().tolist()
                    )
                if step < args.scan_settle_steps:
                    continue
                records["v"].append(env.tracking_lin_vel[:, 0].cpu().numpy())
                records["w"].append(env.tracking_ang_vel[:, 2].cpu().numpy())
                records["applied"].append(env.commands[:, :2].cpu().numpy())
                records["combined"].append(env.combined_policy_actions.cpu().numpy())
                records["requested_output"].append(env.requested_output_actions.cpu().numpy())
                records["executed_output"].append(env.output_actions.cpu().numpy())
        records = {key: np.asarray(value) for key, value in records.items()}
        rows = []
        for index, case in enumerate(cases):
            samples = {key: value[:, index] for key, value in records.items()}
            rows.append(_metrics(case, samples, env.cfg, args))
        summary = _summary(rows, args, args.task, args.checkpoint, float(env.dt))
        summary["controller"] = args.scan_controller
        summary["checkpoint_path"] = model_path
        summary["settle_steps"] = args.scan_settle_steps
        summary["measure_steps"] = args.scan_measure_steps
        summary["direct_command_contract"] = bool(
            getattr(env.cfg.commands, "direct_command_tracking", False)
        )
        summary["angular_feedback_gain"] = float(
            env.cfg.control.angular_feedback_gain
        )
        summary["minimum_turn_radius_m"] = float(
            env.cfg.commands.minimum_turn_radius
        )
        output_dir = args.scan_output_dir or os.path.join(
            args.load_run if args.load_run else os.getcwd(),
            "reachable_domain",
            "checkpoint_%d" % args.checkpoint,
        )
        csv_path, json_path = _write_reports(output_dir, rows, summary)
        for row in rows:
            print(
                "%s request=(%+.3f,%+.4f) measured=(%+.3f,%+.4f) "
                "mae=(%.4f,%.4f) %s"
                % (
                    row["case"], row["command_v_mps"], row["command_w_radps"],
                    row["mean_v_mps"], row["mean_w_radps"], row["v_mae_mps"],
                    row["w_mae_radps"], "PASS" if row["stable_reachable"] else "FAIL",
                )
            )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("CSV:  %s" % csv_path)
        print("JSON: %s" % json_path)
    finally:
        _close_env(env)


if __name__ == "__main__":
    main()
