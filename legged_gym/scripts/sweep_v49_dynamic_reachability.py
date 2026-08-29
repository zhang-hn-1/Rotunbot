"""Stage 1.2 Part B: characterize frozen V49 response over one 0.2 s hold."""

import argparse
import csv
import json
import os
import sys

import isaacgym  # noqa: F401
from isaacgym.torch_utils import quat_rotate_inverse
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import (
    project_velocity_commands,
    yaw_from_quaternion,
)
from legged_gym.navigation.v49_stage1_2_diagnostics import (
    RAW_TRACE_FIELDS,
    REACHABILITY_GRID_FIELDS,
    direction_agreement,
    dynamic_response_ratio,
    high_level_alignment,
    summarize_reachability_samples,
    velocity_bin,
    response_reachable,
)
from legged_gym.scripts.audit_v49_state_reset import (
    _close_env,
    _make_runtime,
)
from legged_gym.scripts.evaluate_v49_waypoint_sequence import (
    PROFILE_ASSETS,
    _configure,
    _set_command,
    _set_initial_pose,
    initial_pose_for_episode,
)
from legged_gym.utils import get_args


INITIAL_V = (0.06, 0.08, 0.10, 0.12)
INITIAL_W = (-0.02, 0.0, 0.02)
TARGET_V = (0.06, 0.08, 0.10, 0.12)
TARGET_W = (-0.02, -0.01, 0.0, 0.01, 0.02)


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reach_output_dir", type=str, default="logs/stage1_2_reachability")
    parser.add_argument("--reach_repeats", type=int, default=5)
    parser.add_argument("--reach_stabilize_steps", type=int, default=180)
    parser.add_argument("--reach_stable_window", type=int, default=10)
    parser.add_argument("--reach_transition_steps", type=int, default=10)
    parser.add_argument("--reach_seed", type=int, default=20260829)
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
    if custom.reach_repeats < 1 or custom.reach_transition_steps != 10:
        raise ValueError("use positive repeats and exactly ten transition steps")
    if custom.reach_stabilize_steps < custom.reach_stable_window:
        raise ValueError("stabilization steps must cover the stable window")
    args.load_run = os.path.abspath(args.load_run)
    args.checkpoint = int(args.checkpoint)
    args.num_envs = 1
    args.waypoint_asset_profile = "v49_reference"
    for name, value in vars(custom).items():
        setattr(args, name, value)
    return args


def _projection(env, command):
    cfg = env.cfg.commands
    return project_velocity_commands(
        command,
        cfg.max_forward_speed,
        cfg.max_yaw_rate,
        cfg.minimum_turn_radius,
        cfg.feasible_envelope_fraction,
        stationary_threshold=env.cfg.rewards.stationary_command_threshold,
        turn_authority_start_speed=getattr(cfg, "turn_authority_start_speed", 0.0),
        turn_authority_full_speed=getattr(cfg, "turn_authority_full_speed", 0.0),
    )


def _measure(env):
    return float(env.tracking_lin_vel[0, 0]), float(env.tracking_ang_vel[0, 2])


def _establish_initial_state(env, policy, v0, w0, seed, trial_id, args, reset=True):
    if reset:
        env.reset()
    _set_initial_pose(env, initial_pose_for_episode(seed, trial_id))
    start_raw = torch.tensor([[v0, w0]], device=env.device)
    start_projected = _projection(env, start_raw)
    _set_command(env, start_projected, smooth_reference=False)
    stable = 0
    last_v, last_w = _measure(env)
    for step in range(args.reach_stabilize_steps):
        with torch.no_grad():
            env.step(policy(env.get_observations()))
        last_v, last_w = _measure(env)
        if abs(last_v - float(start_projected[0, 0])) <= 0.02 and abs(
            last_w - float(start_projected[0, 1])
        ) <= 0.01:
            stable += 1
            if stable >= args.reach_stable_window:
                return {
                    "status": "stabilized",
                    "initial_v": last_v,
                    "initial_w": last_w,
                    "requested_v": v0,
                    "requested_w": w0,
                    "projected_v": float(start_projected[0, 0]),
                    "projected_w": float(start_projected[0, 1]),
                    "stabilization_steps": step + 1,
                }
        else:
            stable = 0
    return {
        "status": "initial_state_not_stabilized",
        "initial_v": last_v,
        "initial_w": last_w,
        "requested_v": v0,
        "requested_w": w0,
        "projected_v": float(start_projected[0, 0]),
        "projected_w": float(start_projected[0, 1]),
        "stabilization_steps": args.reach_stabilize_steps,
    }


def _pair(value, first=0.0, second=0.0):
    if value is None:
        return first, second
    return float(value[0, 0]), float(value[0, 1])


def _row_value(env, name):
    value = getattr(env, name, None)
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    return value


def _action_fields(env):
    result = {}
    mapping = {
        "nominal_action": "nominal_policy_actions",
        "feedback_action": "feedback_policy_actions",
        "derivative_action": "derivative_feedback_policy_actions",
        "rate_feedforward_action": "rate_feedforward_policy_actions",
        "residual_action": "applied_residual_actions",
        "combined_action": "combined_policy_actions",
        "final_action": "output_actions",
    }
    for prefix, name in mapping.items():
        first, second = _pair(getattr(env, name, None))
        result[prefix + "_0"] = first
        result[prefix + "_1"] = second
    # The release environment has rate FF but no separate generic FF buffer.
    result["feedforward_action_0"] = None
    result["feedforward_action_1"] = None
    result["rate_feedforward_active"] = bool(
        abs(result["rate_feedforward_action_0"]) > 1.0e-6
        or abs(result["rate_feedforward_action_1"]) > 1.0e-6
    )
    return result


def _trace_row(env, episode_id, trial_id, policy_step, initial, target_raw,
               target_projected, previous_command, action, policy_dt):
    measured_v, measured_w = _measure(env)
    world_linear = env.root_states[0, 7:10].detach().cpu().tolist()
    world_angular = env.root_states[0, 10:13].detach().cpu().tolist()
    body_linear = env.base_lin_vel[0].detach().cpu().tolist()
    body_angular = env.base_ang_vel[0].detach().cpu().tolist()
    yaw = float(yaw_from_quaternion(env.root_states[:, 3:7])[0])
    yaw_rate = float(env.root_states[0, 12])
    high_tick, within, time_s = high_level_alignment(policy_step, 10, policy_dt)
    result = {
        "episode_id": episode_id,
        "trial_id": trial_id,
        "time_s": time_s,
        "physics_step": int(env.common_step_counter),
        "policy_step": policy_step,
        "high_level_tick": high_tick,
        "step_within_high_level_tick": within,
        "initial_v": initial["initial_v"],
        "initial_w": initial["initial_w"],
        "desired_v": float(target_raw[0, 0]),
        "desired_w": float(target_raw[0, 1]),
        "projected_v": float(target_projected[0, 0]),
        "projected_w": float(target_projected[0, 1]),
        "previous_command_v": float(previous_command[0, 0]),
        "previous_command_w": float(previous_command[0, 1]),
        "delta_command_v": float(target_projected[0, 0] - previous_command[0, 0]),
        "delta_command_w": float(target_projected[0, 1] - previous_command[0, 1]),
        "measured_v": measured_v,
        "measured_w": measured_w,
        "v_tracking_error": float(target_projected[0, 0]) - measured_v,
        "w_tracking_error": float(target_projected[0, 1]) - measured_w,
        "yaw": yaw,
        "yaw_rate": yaw_rate,
        "joint1_position": float(env.dof_pos[0, 0]),
        "joint1_velocity": float(env.dof_vel[0, 0]),
        "joint2_position": float(env.dof_pos[0, 1]),
        "joint2_velocity": float(env.dof_vel[0, 1]),
        "smooth_reference_flag": bool(getattr(env, "command_reference_is_smooth", torch.zeros(1))[0]),
        "contact_yaw_damping_factor": float(getattr(env, "contact_yaw_damping_speed_factor", torch.zeros(1))[0]),
        "low_speed_lt_010": abs(measured_v) < 0.10,
        "low_speed_lt_008": abs(measured_v) < 0.08,
        "direction_agreement_v": direction_agreement(
            float(target_projected[0, 0] - previous_command[0, 0]),
            measured_v - initial["initial_v"],
        ),
        "direction_agreement_w": direction_agreement(
            float(target_projected[0, 1] - previous_command[0, 1]),
            measured_w - initial["initial_w"],
        ),
    }
    for index, value in enumerate(world_linear):
        result["root_linear_velocity_world_%d" % index] = value
    for index, value in enumerate(body_linear):
        result["root_linear_velocity_body_%d" % index] = value
    for index, value in enumerate(world_angular):
        result["root_angular_velocity_world_%d" % index] = value
    for index, value in enumerate(body_angular):
        result["root_angular_velocity_body_%d" % index] = value
    result.update(_action_fields(env))
    return result


def _run_transition(env, policy, args, v0, w0, target_v, target_w, trial_id):
    initial = _establish_initial_state(env, policy, v0, w0, args.reach_seed, trial_id, args)
    target_raw = torch.tensor([[target_v, target_w]], device=env.device)
    target_projected = _projection(env, target_raw)
    previous_command = torch.tensor(
        [[initial["projected_v"], initial["projected_w"]]], device=env.device
    )
    if initial["status"] != "stabilized":
        return initial, [], {
            "status": initial["status"], "initial_v_bin": velocity_bin(v0),
            "initial_w_bin": w0, "target_v": target_v, "target_w": target_w,
            "projected_v": float(target_projected[0, 0]),
            "projected_w": float(target_projected[0, 1]), "repeat_count": 0,
        }
    _set_command(env, target_projected, smooth_reference=False)
    trace = []
    for policy_step in range(args.reach_transition_steps):
        with torch.no_grad():
            action = policy(env.get_observations())
            _, _, _, dones, _ = env.step(action)
        trace.append(_trace_row(
            env, 0, trial_id, policy_step, initial, target_raw,
            target_projected, previous_command, action, float(env.dt),
        ))
        if bool(torch.any(dones)):
            break
    outcome = {
        "status": "complete" if len(trace) == args.reach_transition_steps else "terminated",
        "initial_v_bin": velocity_bin(v0),
        "initial_w_bin": w0,
        "target_v": target_v,
        "target_w": target_w,
        "projected_v": float(target_projected[0, 0]),
        "projected_w": float(target_projected[0, 1]),
        "repeat_count": 1,
        "initial_v": initial["initial_v"],
        "initial_w": initial["initial_w"],
        "initial_state_status": initial["status"],
        "initial_state_stabilization_steps": initial["stabilization_steps"],
        "actual_v_20ms": trace[0]["measured_v"],
        "actual_w_20ms": trace[0]["measured_w"],
        "actual_v_100ms": trace[min(4, len(trace) - 1)]["measured_v"],
        "actual_w_100ms": trace[min(4, len(trace) - 1)]["measured_w"],
        "actual_v_200ms": trace[-1]["measured_v"],
        "actual_w_200ms": trace[-1]["measured_w"],
        "v_tracking_tolerance": 0.12,
        "w_tracking_tolerance": 0.025,
        "response_reachable": False,
        "tracking_reachable": False,
        "forward_sign_correct": trace[-1]["direction_agreement_v"],
        "yaw_sign_correct": trace[-1]["direction_agreement_w"],
    }
    outcome["response_reachable"] = bool(
        response_reachable(
            outcome["projected_v"] - outcome["initial_v"],
            outcome["actual_v_200ms"] - outcome["initial_v"],
        ) and response_reachable(
            outcome["projected_w"] - outcome["initial_w"],
            outcome["actual_w_200ms"] - outcome["initial_w"],
        )
    )
    outcome["tracking_reachable"] = bool(
        abs(outcome["actual_v_200ms"] - outcome["projected_v"]) <= 0.12
        and abs(outcome["actual_w_200ms"] - outcome["projected_w"]) <= 0.025
    )
    outcome["response_ratio_v"] = dynamic_response_ratio(
        outcome["initial_v"], outcome["projected_v"], outcome["actual_v_200ms"]
    )
    outcome["response_ratio_w"] = dynamic_response_ratio(
        outcome["initial_w"], outcome["projected_w"], outcome["actual_w_200ms"]
    )
    return initial, trace, outcome


def _write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _parse_args()
    env, policy = _make_runtime(args)
    try:
        output_dir = os.path.abspath(args.reach_output_dir)
        os.makedirs(output_dir, exist_ok=True)
        trace_rows = []
        transition_rows = []
        trial_id = 0
        for v0 in INITIAL_V:
            for w0 in INITIAL_W:
                for target_v in TARGET_V:
                    for target_w in TARGET_W:
                        for repeat in range(args.reach_repeats):
                            initial, trace, outcome = _run_transition(
                                env, policy, args, v0, w0, target_v, target_w, trial_id
                            )
                            for row in trace:
                                row["episode_id"] = trial_id
                                row["trial_id"] = trial_id
                            trace_rows.extend(trace)
                            outcome["trial_id"] = trial_id
                            outcome["repeat"] = repeat
                            outcome["requested_initial_v"] = v0
                            outcome["requested_initial_w"] = w0
                            transition_rows.append(outcome)
                            trial_id += 1
        grid_rows = []
        grouped = {}
        for row in transition_rows:
            if row.get("status") != "complete":
                continue
            key = (
                row["initial_v_bin"], row["initial_w_bin"],
                row["target_v"], row["target_w"],
                row["projected_v"], row["projected_w"],
            )
            grouped.setdefault(key, []).append(row)
        for key, samples in sorted(grouped.items(), key=lambda item: str(item[0])):
            initial_v_bin, initial_w_bin, target_v, target_w, projected_v, projected_w = key
            summary = summarize_reachability_samples(samples)
            summary.update({
                "initial_v_bin": initial_v_bin,
                "initial_w_bin": initial_w_bin,
                "target_v": target_v,
                "target_w": target_w,
                "projected_v": projected_v,
                "projected_w": projected_w,
            })
            grid_rows.append(summary)
        _write_csv(os.path.join(output_dir, "raw_50hz_trace.csv"), RAW_TRACE_FIELDS, trace_rows)
        transition_fields = tuple(sorted(set().union(*(set(row) for row in transition_rows))))
        _write_csv(os.path.join(output_dir, "transition_summary.csv"), transition_fields, transition_rows)
        _write_csv(os.path.join(output_dir, "reachability_grid.csv"), REACHABILITY_GRID_FIELDS, grid_rows)
        summary = {
            "task": args.task,
            "asset": PROFILE_ASSETS["v49_reference"],
            "checkpoint": args.checkpoint,
            "checkpoint_path": os.path.join(args.load_run, "model_%d.pt" % args.checkpoint),
            "seed": args.reach_seed,
            "repeat_count_per_transition": args.reach_repeats,
            "policy_hz": 1.0 / float(env.dt),
            "high_level_hz": 5.0,
            "policy_steps_per_hold": 10,
            "initial_v_values": INITIAL_V,
            "initial_w_values": INITIAL_W,
            "target_v_values": TARGET_V,
            "target_w_values": TARGET_W,
            "transition_count": len(transition_rows),
            "completed_transition_count": sum(row.get("status") == "complete" for row in transition_rows),
            "initial_state_not_stabilized_count": sum(
                row.get("status") == "initial_state_not_stabilized" for row in transition_rows
            ),
            "availability": {
                "generic_feedforward_action": "not_available",
                "recurrent_hidden_state": "not_available",
                "observation_history": "not_available",
                "contact_yaw_damping_factor": "available",
            },
            "analysis_thresholds": {
                "v_tracking_tolerance_mps": 0.12,
                "w_tracking_tolerance_radps": 0.025,
                "source": "RotunbotVelCfg.rewards.linear_tracking_sigma/angular_tracking_sigma",
            },
            "grid_row_count": len(grid_rows),
        }
        with open(os.path.join(output_dir, "reachability_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))
    finally:
        _close_env(env)


if __name__ == "__main__":
    main()
