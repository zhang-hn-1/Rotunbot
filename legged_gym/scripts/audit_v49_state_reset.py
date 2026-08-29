"""Stage 1.2 Part A: audit V49 reset state and A/B prefix determinism."""

import argparse
import csv
import json
import os
import subprocess
import sys

import isaacgym  # noqa: F401
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import command_update_interval_steps
from legged_gym.navigation.v49_stage1_2_diagnostics import (
    compare_snapshot_sequences,
    direction_agreement,
    high_level_alignment,
    reset_audit_rows,
)
from legged_gym.navigation.v49_waypoint_controller import (
    V49WaypointConfig,
    WaypointSequenceController,
)
from legged_gym.scripts.evaluate_v49_waypoint_sequence import (
    PROFILE_ASSETS,
    TRAJECTORIES,
    _configure,
    _pose,
    _set_command,
    _set_initial_pose,
)
from legged_gym.utils import get_args, task_registry


PREFIX_FIELDS = (
    "desired_command", "projected_command", "observation", "root_states",
    "dof_pos", "dof_vel", "command_targets", "commands", "command_rates",
    "held_upper_command_rates", "tracking_error_integral", "last_tracking_error",
    "tracking_error_derivative", "command_brake_pending",
    "command_yaw_brake_pending", "actions", "last_actions",
    "nominal_policy_actions", "feedback_policy_actions",
    "derivative_feedback_policy_actions", "rate_feedforward_policy_actions",
    "applied_residual_actions", "combined_policy_actions", "output_actions",
    "policy_action", "tracking_lin_vel", "tracking_ang_vel",
)


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--audit_mode", choices=("prefix", "reset", "fresh"), default="prefix")
    parser.add_argument("--single_case", type=str, default=None)
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--fresh_repeats", type=int, default=10)
    parser.add_argument("--reuse_fresh_json", action="store_true")
    parser.add_argument("--audit_output_dir", type=str, default="logs/stage1_2_state_audit")
    parser.add_argument("--audit_max_policy_steps", type=int, default=1200)
    original_argv = list(sys.argv)
    custom, remaining = parser.parse_known_args()
    sys.argv = [original_argv[0]] + remaining
    try:
        args = get_args()
    finally:
        sys.argv = original_argv
    if args.task != "rotunbot_vel_sru50_v49_integration":
        raise ValueError("Stage 1.2 requires the V49 integration task")
    if not args.load_run or args.checkpoint is None:
        raise ValueError("--load_run and --checkpoint are required")
    if custom.fresh_repeats < 1:
        raise ValueError("--fresh_repeats must be positive")
    args.load_run = os.path.abspath(args.load_run)
    args.checkpoint = int(args.checkpoint)
    args.num_envs = 1
    args.waypoint_asset_profile = "v49_reference"
    for name, value in vars(custom).items():
        setattr(args, name, value)
    return args


def _make_runtime(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _configure(env_cfg, args)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    return env, runner.get_inference_policy(device=env.device)


def _close_env(env):
    try:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
    finally:
        if env.sim is not None:
            env.gym.destroy_sim(env.sim)


def _value(env, name):
    value = getattr(env, name, None)
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _snapshot(env, observation, policy_action, desired, projected):
    names = (
        "root_states", "dof_pos", "dof_vel", "command_targets", "commands",
        "command_rates", "held_upper_command_rates", "tracking_error_integral",
        "last_tracking_error", "tracking_error_derivative",
        "command_brake_pending", "command_yaw_brake_pending", "actions",
        "last_actions", "nominal_policy_actions", "feedback_policy_actions",
        "derivative_feedback_policy_actions", "rate_feedforward_policy_actions",
        "applied_residual_actions", "combined_policy_actions", "output_actions",
        "tracking_lin_vel", "tracking_ang_vel", "command_reference_is_smooth",
        "obs_buf", "privileged_obs_buf", "episode_length_buf", "reset_buf",
        "time_out_buf",
    )
    result = {
        "desired_command": desired.detach().cpu().tolist(),
        "projected_command": projected.detach().cpu().tolist(),
        "observation": observation.detach().cpu().tolist(),
        "policy_action": policy_action.detach().cpu().tolist(),
    }
    for name in names:
        value = _value(env, name)
        if value is not None:
            result[name] = value
    # Capture likely history/recurrent/latency buffers when a task exposes one;
    # absent variables remain absent and are reported by the reset inventory.
    for name in (
        "obs_history", "observation_history", "critic_history", "history",
        "hidden_state", "recurrent_hidden_state", "action_delay_buffer",
        "torque_delay_buffer", "latency_buffer",
    ):
        value = _value(env, name)
        if value is not None:
            result[name] = value
    return result


def _run_prefix(env, policy, trajectory_name, episode_id, max_steps):
    env.reset()
    # Use the exact Stage1 pose generator without importing private random state.
    from legged_gym.scripts.evaluate_v49_waypoint_sequence import initial_pose_for_episode
    pose = initial_pose_for_episode(20260828, episode_id)
    _set_initial_pose(env, pose)
    controller = WaypointSequenceController(
        torch.as_tensor(TRAJECTORIES[trajectory_name]),
        config=V49WaypointConfig(),
        policy_steps_per_tick=command_update_interval_steps(env.dt, 5.0),
    )
    snapshots = []
    rows = []
    desired = torch.zeros(1, 2, device=env.device)
    projected = desired
    for policy_step in range(max_steps):
        if policy_step % controller.policy_steps_per_tick == 0:
            xy, yaw = _pose(env)
            tick = controller.tick(xy, yaw)
            if tick.waypoint_switched:
                break
            target = controller.waypoints[tick.active_waypoint_index].to(
                device=env.device
            ).unsqueeze(0)
            desired = tick.raw_command.detach().clone()
            projected = tick.projected_command.detach().clone()
            _set_command(env, projected, smooth_reference=False)
        observation = env.get_observations().detach().clone()
        with torch.no_grad():
            policy_action = policy(observation)
        _, _, _, dones, _ = env.step(policy_action)
        high_tick, within, time_s = high_level_alignment(
            policy_step, controller.policy_steps_per_tick, float(env.dt)
        )
        snap = _snapshot(env, observation, policy_action, desired, projected)
        snapshots.append(snap)
        rows.append({
            "trajectory": trajectory_name,
            "episode_id": episode_id,
            "policy_step": policy_step,
            "high_level_tick": high_tick,
            "step_within_high_level_tick": within,
            "time_s": time_s,
            "sequence_complete": bool(torch.any(dones)),
        })
        if bool(torch.any(dones)):
            break
    return snapshots, rows, pose


def _write_prefix(args, env, policy):
    snapshots_a, rows_a, pose_a = _run_prefix(env, policy, "A", 4, args.audit_max_policy_steps)
    snapshots_b, rows_b, pose_b = _run_prefix(env, policy, "B", 4, args.audit_max_policy_steps)
    strict_comparison = compare_snapshot_sequences(
        snapshots_a, snapshots_b, fields=PREFIX_FIELDS, abs_tol=1.0e-6, rel_tol=1.0e-5
    )
    comparison = compare_snapshot_sequences(
        snapshots_a, snapshots_b, fields=PREFIX_FIELDS, abs_tol=1.0e-4, rel_tol=1.0e-3
    )
    output_dir = os.path.abspath(args.audit_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "A_B_prefix_equivalence.csv"), "w", newline="", encoding="utf-8") as handle:
        fields = ("policy_step", "variable", "equivalent", "absolute_difference", "relative_difference")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step, (left, right) in enumerate(zip(snapshots_a, snapshots_b)):
            for name in PREFIX_FIELDS:
                step_result = compare_snapshot_sequences(
                    [left], [right], fields=(name,), abs_tol=1.0e-4, rel_tol=1.0e-3
                )
                writer.writerow({
                    "policy_step": step,
                    "variable": name,
                    "equivalent": step_result["equivalent"],
                    "absolute_difference": step_result["absolute_difference"],
                    "relative_difference": step_result["relative_difference"],
                })
    summary = {
        "trajectory_A_initial_pose": pose_a,
        "trajectory_B_initial_pose": pose_b,
        "same_initial_pose": pose_a == pose_b,
        "first_waypoint": [1.0, 0.0],
        "strict_comparison": strict_comparison,
        "control_tolerance": {"absolute": 1.0e-4, "relative": 1.0e-3},
        "comparison": comparison,
        "prefix_gate": "PASS" if comparison["equivalent"] else "FAIL",
        "compared_fields": PREFIX_FIELDS,
    }
    with open(os.path.join(output_dir, "A_B_prefix_equivalence_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _reset_specs():
    zero_states = (
        "commands", "command_rates", "held_upper_command_rates",
        "tracking_error_integral", "last_tracking_error", "tracking_error_derivative",
        "requested_output_actions", "output_actions", "last_output_actions",
        "nominal_policy_actions", "feedback_policy_actions",
        "derivative_feedback_policy_actions", "rate_feedforward_policy_actions",
        "combined_policy_actions", "applied_residual_actions", "actions",
        "last_actions", "tracking_lin_vel", "tracking_ang_vel",
        "command_profile_phase", "command_profile_speed_amplitude",
        "command_profile_signed_curvature", "command_profile_velocity_offset",
        "command_profile_velocity_amplitude", "command_profile_yaw_amplitude",
        "command_profile_yaw_phase_offset", "command_profile_yaw_frequency_ratio",
        "dof_vel", "root_states_velocity", "episode_length_buf", "reset_buf",
        "time_out_buf",
    )
    specs = []
    for name in zero_states:
        location = "RotunbotVel.reset_idx / LeggedRobot.reset_idx"
        if name == "reset_buf":
            expected = True
        elif name == "command_profile_yaw_frequency_ratio":
            expected = 1.0
        elif name == "command_profile_period":
            expected = 1.0
        else:
            expected = 0.0
        specs.append({"name": name, "location": location, "expected": expected})
    specs.append({
        "name": "command_targets",
        "location": "LeggedRobot.reset_idx._resample_commands",
        "expected": "base reset intentionally resamples command target; evaluator overwrites it",
    })
    for name in (
        "command_brake_pending", "command_yaw_brake_pending",
        "command_profile_is_smooth", "command_profile_is_random_walk",
        "command_reference_is_smooth", "command_profile_is_independent",
    ):
        specs.append({
            "name": name,
            "location": "RotunbotVel.reset_idx",
            "expected": False,
        })
    for name in (
        "obs_history", "observation_history", "critic_history", "history",
        "hidden_state", "recurrent_hidden_state", "action_delay_buffer",
        "torque_delay_buffer", "latency_buffer",
    ):
        specs.append({"name": name, "location": "not found in current task", "expected": 0.0})
    return specs


def _capture_reset_value(env, name):
    if name == "root_states_velocity":
        return _value(env, "root_states")[0][7:13]
    return _value(env, name)


def _write_reset_audit(args, env, policy):
    from legged_gym.scripts.evaluate_v49_waypoint_sequence import initial_pose_for_episode
    env.reset()
    _set_initial_pose(env, initial_pose_for_episode(20260828, 4))
    _set_command(env, torch.tensor([[0.10, 0.02]], device=env.device))
    with torch.no_grad():
        for _ in range(20):
            env.step(policy(env.get_observations()))
    # Force the external reference flag high so reset behavior is observable.
    if hasattr(env, "command_reference_is_smooth"):
        env.command_reference_is_smooth[:] = True
    before = {spec["name"]: _capture_reset_value(env, spec["name"]) for spec in _reset_specs()}
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    after = {spec["name"]: _capture_reset_value(env, spec["name"]) for spec in _reset_specs()}
    rows = reset_audit_rows(_reset_specs(), before, after)
    output_dir = os.path.abspath(args.audit_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "reset_state_audit.csv"), "w", newline="", encoding="utf-8") as handle:
        fields = tuple(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "fail_count": sum(row["status"] == "FAIL" for row in rows),
        "not_available_count": sum(row["status"] == "NOT_AVAILABLE" for row in rows),
        "rows": rows,
    }
    with open(os.path.join(output_dir, "reset_state_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(output_dir, "reset_state_audit.md"), "w", encoding="utf-8") as handle:
        handle.write("# V49 reset-state audit\n\n")
        handle.write("| Variable | Availability | Status | Location | Expected | After reset |\n")
        handle.write("|---|---|---|---|---|---|\n")
        for row in rows:
            handle.write(
                "| {variable_name} | {availability} | {status} | {location} | {expected_reset_value} | {after_reset} |\n".format(**row)
            )
    return summary


def _case_summary(trace, trajectory, episode_id):
    measured_v = [float(row["tracking_lin_vel"][0][0]) for row in trace]
    measured_w = [float(row["tracking_ang_vel"][0][2]) for row in trace]
    desired_v = [float(row["projected_command"][0][0]) for row in trace]
    desired_w = [float(row["projected_command"][0][1]) for row in trace]
    return {
        "trajectory": trajectory,
        "episode_id": episode_id,
        "trace_policy_steps": len(trace),
        "mean_abs_v_tracking_error": float(np.mean(np.abs(np.asarray(desired_v) - measured_v))),
        "mean_abs_w_tracking_error": float(np.mean(np.abs(np.asarray(desired_w) - measured_w))),
        "direction_agreement_v": float(np.mean([
            direction_agreement(v, actual) for v, actual in zip(desired_v, measured_v)
        ])),
        "negative_measured_v_fraction": float(np.mean(np.asarray(measured_v) < 0.0)),
        "minimum_measured_v": min(measured_v, default=None),
        "maximum_measured_v": max(measured_v, default=None),
    }


def _run_single_case(args, case):
    trajectory = case[0].upper()
    episode_id = int(case[1:])
    env, policy = _make_runtime(args)
    try:
        trace, _, _ = _run_prefix(env, policy, trajectory, episode_id, 100)
        return _case_summary(trace, trajectory, episode_id)
    finally:
        _close_env(env)


def _run_fresh(args):
    output_dir = os.path.abspath(args.audit_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "fresh_process_reproducibility.json")
    if args.reuse_fresh_json:
        with open(json_path, encoding="utf-8") as handle:
            summary = json.load(handle)
        repeats = summary["fresh_process_repeats"]
        with open(os.path.join(output_dir, "fresh_process_reproducibility.csv"), "w", newline="", encoding="utf-8") as handle:
            fields = (
                "trajectory", "episode_id", "repeat", "trace_policy_steps",
                "direction_agreement_v", "negative_measured_v_fraction",
                "mean_abs_v_tracking_error", "mean_abs_w_tracking_error",
                "minimum_measured_v", "maximum_measured_v",
            )
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(repeats)
        return summary
    cases = ("A4", "B5", "B6", "B7")
    repeats = []
    if args.single_case:
        return _run_single_case(args, args.single_case)
    if args.sequence:
        env, policy = _make_runtime(args)
        try:
            summaries = []
            for case in args.sequence.split(","):
                trajectory = case[0].upper()
                episode_id = int(case[1:])
                trace, _, _ = _run_prefix(env, policy, trajectory, episode_id, 100)
                summaries.append(_case_summary(trace, trajectory, episode_id))
            return {"sequence": args.sequence, "episodes": summaries}
        finally:
            _close_env(env)
    for case in cases:
        for repeat in range(args.fresh_repeats):
            command = [
                sys.executable, os.path.abspath(__file__),
                "--single_case", case,
                "--task", args.task, "--load_run", args.load_run,
                "--checkpoint", str(args.checkpoint), "--headless",
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            marker = "RESULT_JSON:"
            payload = [line[len(marker):].strip() for line in completed.stdout.splitlines() if line.startswith(marker)]
            if completed.returncode != 0 or not payload:
                raise RuntimeError("fresh case failed: %s repeat %d\n%s" % (case, repeat, completed.stderr[-2000:]))
            result = json.loads(payload[-1])
            result["repeat"] = repeat
            repeats.append(result)
    sequences = ("A4,B4", "B4,A4", "A1,A4", "A4,A4", "A4,A4,A4")
    run_orders = []
    for sequence in sequences:
        command = [
            sys.executable, os.path.abspath(__file__), "--sequence", sequence,
            "--task", args.task, "--load_run", args.load_run,
            "--checkpoint", str(args.checkpoint), "--headless",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        marker = "RESULT_JSON:"
        payload = [line[len(marker):].strip() for line in completed.stdout.splitlines() if line.startswith(marker)]
        if completed.returncode != 0 or not payload:
            raise RuntimeError("run order failed: %s\n%s" % (sequence, completed.stderr[-2000:]))
        run_orders.append(json.loads(payload[-1]))
    summary = {"fresh_process_repeats": repeats, "run_order_sequences": run_orders}
    with open(os.path.join(output_dir, "fresh_process_reproducibility.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(output_dir, "fresh_process_reproducibility.csv"), "w", newline="", encoding="utf-8") as handle:
        fields = (
            "trajectory", "episode_id", "repeat", "trace_policy_steps",
            "direction_agreement_v", "negative_measured_v_fraction",
            "mean_abs_v_tracking_error", "mean_abs_w_tracking_error",
            "minimum_measured_v", "maximum_measured_v",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(repeats)
    return summary


def main():
    args = _parse_args()
    if args.single_case or args.sequence or args.audit_mode == "fresh":
        result = _run_fresh(args)
        print("RESULT_JSON: %s" % json.dumps(result, separators=(",", ":")))
        return
    env, policy = _make_runtime(args)
    try:
        if args.audit_mode == "prefix":
            result = _write_prefix(args, env, policy)
        else:
            result = _write_reset_audit(args, env, policy)
        print(json.dumps(result, indent=2))
    finally:
        _close_env(env)


if __name__ == "__main__":
    main()
