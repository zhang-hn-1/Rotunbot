"""Stage1.3: measure the frozen V49 response on a 200 ms horizon."""

import argparse
import csv
import json
import math
import os
import sys

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import (
    project_velocity_commands,
    yaw_from_quaternion,
)
from legged_gym.navigation.v49_stage1_3_diagnostics import (
    STAGE13_HORIZONS_MS,
    STAGE13_TRACE_FIELDS,
    aggregate_stage13_trials,
    summarize_trace,
    symmetric_yaw_grid,
)
from legged_gym.scripts.audit_v49_state_reset import (
    _close_env,
    _make_runtime,
)
from legged_gym.scripts.sweep_v49_dynamic_reachability import (
    _establish_initial_state,
    _projection,
)
from legged_gym.scripts.evaluate_v49_waypoint_sequence import (
    PROFILE_ASSETS,
    _set_command,
    _set_initial_pose,
    initial_pose_for_episode,
)
from legged_gym.utils import get_args


INITIAL_FORWARD_VALUES = (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14)
INITIAL_YAW_VALUES = (0.0,)


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage13_output_dir", default="logs/stage1_3_dynamic_reachability")
    parser.add_argument("--stage13_repeats", type=int, default=3)
    parser.add_argument("--stage13_stabilize_steps", type=int, default=180)
    parser.add_argument("--stage13_stable_window", type=int, default=10)
    parser.add_argument("--stage13_seed", type=int, default=20260829)
    parser.add_argument("--stage13_max_trials", type=int, default=0)
    original_argv = list(sys.argv)
    custom, remaining = parser.parse_known_args()
    sys.argv = [original_argv[0]] + remaining
    try:
        args = get_args()
    finally:
        sys.argv = original_argv
    if args.task != "rotunbot_vel_sru50_v49_integration":
        raise ValueError("Stage1.3 requires the V49 integration task")
    if not args.load_run or args.checkpoint is None:
        raise ValueError("--load_run and --checkpoint are required")
    if custom.stage13_repeats < 3:
        raise ValueError("Stage1.3 requires at least three repeats")
    if custom.stage13_stabilize_steps < custom.stage13_stable_window:
        raise ValueError("stabilization steps must cover the stable window")
    args.load_run = os.path.abspath(args.load_run)
    args.checkpoint = int(args.checkpoint)
    args.num_envs = 1
    args.waypoint_asset_profile = "v49_reference"
    # Reuse the Stage1.2 stabilization helper without duplicating its logic.
    args.reach_stabilize_steps = custom.stage13_stabilize_steps
    args.reach_stable_window = custom.stage13_stable_window
    args.reach_seed = custom.stage13_seed
    for name, value in vars(custom).items():
        setattr(args, name, value)
    return args


def _forward_grid(maximum):
    maximum = float(maximum)
    return tuple(round(maximum * fraction, 8) for fraction in (
        -1.0, -0.75, -0.40, -0.20, 0.0, 0.20, 0.40, 0.75, 1.0
    ))


def _pair(value):
    if value is None:
        return 0.0, 0.0
    flat = value.detach().cpu().reshape(-1).tolist()
    return float(flat[0]), float(flat[1])


def _action_pair(env, name):
    return _pair(getattr(env, name, None))


def _trace_row(env, trial_id, seed, policy_step, initial, raw_target,
               projected_target, action, unstable):
    world_position = env.root_states[0, :3].detach().cpu().tolist()
    body_linear = env.base_lin_vel[0].detach().cpu().tolist()
    body_angular = env.base_ang_vel[0].detach().cpu().tolist()
    measured_v = float(env.tracking_lin_vel[0, 0])
    measured_w = float(env.tracking_ang_vel[0, 2])
    nominal = _action_pair(env, "nominal_policy_actions")
    feedback = _action_pair(env, "feedback_policy_actions")
    final = _action_pair(env, "output_actions")
    yaw = float(yaw_from_quaternion(env.root_states[:, 3:7])[0])
    return {
        "seed": seed,
        "env_id": 0,
        "trial_id": trial_id,
        "simulation_dt": float(env.sim_params.dt),
        "control_dt": float(env.dt),
        "policy_step": policy_step,
        "time_s": float(policy_step) * float(env.dt),
        "initial_forward_velocity": initial["initial_forward_velocity"],
        "initial_yaw_rate": initial["initial_yaw_rate"],
        "forward_velocity_command": float(raw_target[0, 0]),
        "yaw_rate_command": float(raw_target[0, 1]),
        "projected_forward_velocity": float(projected_target[0, 0]),
        "projected_yaw_rate": float(projected_target[0, 1]),
        "actual_v": measured_v,
        "actual_w": measured_w,
        "yaw": yaw,
        "yaw_rate": float(env.root_states[0, 12]),
        "root_position_x": world_position[0],
        "root_position_y": world_position[1],
        "root_position_z": world_position[2],
        "body_linear_velocity_x": body_linear[0],
        "body_linear_velocity_y": body_linear[1],
        "body_linear_velocity_z": body_linear[2],
        "body_angular_velocity_x": body_angular[0],
        "body_angular_velocity_y": body_angular[1],
        "body_angular_velocity_z": body_angular[2],
        "joint1_position": float(env.dof_pos[0, 0]),
        "joint1_velocity": float(env.dof_vel[0, 0]),
        "joint2_position": float(env.dof_pos[0, 1]),
        "joint2_velocity": float(env.dof_vel[0, 1]),
        "nominal_action_0": nominal[0],
        "nominal_action_1": nominal[1],
        "feedback_action_0": feedback[0],
        "feedback_action_1": feedback[1],
        "final_action_0": final[0],
        "final_action_1": final[1],
        "simulation_unstable": bool(unstable),
    }


def _run_transition(env, policy, args, v0, w0, target_v, target_w, trial_id):
    env.reset()
    _set_initial_pose(env, initial_pose_for_episode(args.stage13_seed, trial_id))
    initial = _establish_initial_state(
        env, policy, v0, w0, args.stage13_seed, trial_id, args, reset=False
    )
    raw_target = torch.tensor([[target_v, target_w]], device=env.device)
    projected_target = _projection(env, raw_target)
    if initial["status"] != "stabilized":
        return initial, [], {
            "status": "initial_state_not_stabilized",
            "trial_id": trial_id,
            "requested_initial_forward_velocity": v0,
            "requested_initial_yaw_rate": w0,
            "forward_velocity_command": target_v,
            "yaw_rate_command": target_w,
        }
    stabilization_steps = initial.get("stabilization_steps")
    initial = {
        "initial_forward_velocity": initial["initial_v"],
        "initial_yaw_rate": initial["initial_w"],
        "forward_velocity_command": target_v,
        "yaw_rate_command": target_w,
    }
    start_position = env.root_states[0, :3].detach().clone()
    start_yaw = float(yaw_from_quaternion(env.root_states[:, 3:7])[0])
    initial["initial_yaw"] = start_yaw
    initial["initial_root_position_x"] = float(start_position[0])
    initial["initial_root_position_y"] = float(start_position[1])
    _set_command(env, projected_target, smooth_reference=False)
    trace = []
    unstable = False
    for policy_step in range(1, 11):
        with torch.no_grad():
            action = policy(env.get_observations())
            _, _, _, dones, _ = env.step(action)
        unstable = unstable or bool(torch.any(dones))
        unstable = unstable or not bool(torch.isfinite(env.root_states).all())
        trace.append(_trace_row(
            env, trial_id, args.stage13_seed, policy_step, initial,
            raw_target, projected_target, action, unstable,
        ))
    summary = summarize_trace(
        trace, initial, (float(projected_target[0, 0]), float(projected_target[0, 1])),
        float(env.dt),
    )
    final_position = env.root_states[0, :3].detach().clone()
    dx = float(final_position[0] - start_position[0])
    dy = float(final_position[1] - start_position[1])
    cos_yaw = math.cos(start_yaw)
    sin_yaw = math.sin(start_yaw)
    summary["body_displacement_x_200ms"] = cos_yaw * dx + sin_yaw * dy
    summary["body_displacement_y_200ms"] = -sin_yaw * dx + cos_yaw * dy
    summary["joint1_position_min"] = min(float(row["joint1_position"]) for row in trace)
    summary["joint1_position_max"] = max(float(row["joint1_position"]) for row in trace)
    summary["joint1_velocity_min"] = min(float(row["joint1_velocity"]) for row in trace)
    summary["joint1_velocity_max"] = max(float(row["joint1_velocity"]) for row in trace)
    summary["joint2_position_min"] = min(float(row["joint2_position"]) for row in trace)
    summary["joint2_position_max"] = max(float(row["joint2_position"]) for row in trace)
    summary["joint2_velocity_min"] = min(float(row["joint2_velocity"]) for row in trace)
    summary["joint2_velocity_max"] = max(float(row["joint2_velocity"]) for row in trace)
    summary.update({
        "status": "simulation_unstable" if unstable else "complete",
        "trial_id": trial_id,
        "seed": args.stage13_seed,
        "requested_initial_forward_velocity": v0,
        "requested_initial_yaw_rate": w0,
        "initial_state_stabilization_steps": stabilization_steps,
    })
    return initial, trace, summary


def _write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _parse_args()
    env, policy = _make_runtime(args)
    output_dir = os.path.abspath(args.stage13_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    raw_rows = []
    summary_rows = []
    try:
        max_forward = float(env.cfg.commands.max_forward_speed)
        max_yaw = float(env.cfg.commands.max_yaw_rate)
        forward_commands = _forward_grid(max_forward)
        yaw_commands = symmetric_yaw_grid(max_yaw, max_yaw / 5.0)
        trial_id = 0
        for v0 in INITIAL_FORWARD_VALUES:
            if args.stage13_max_trials and trial_id >= args.stage13_max_trials:
                break
            for w0 in INITIAL_YAW_VALUES:
                if args.stage13_max_trials and trial_id >= args.stage13_max_trials:
                    break
                for target_v in forward_commands:
                    if args.stage13_max_trials and trial_id >= args.stage13_max_trials:
                        break
                    for target_w in yaw_commands:
                        if args.stage13_max_trials and trial_id >= args.stage13_max_trials:
                            break
                        for repeat in range(args.stage13_repeats):
                            if args.stage13_max_trials and trial_id >= args.stage13_max_trials:
                                break
                            _, trace, summary = _run_transition(
                                env, policy, args, v0, w0, target_v, target_w, trial_id
                            )
                            for row in trace:
                                raw_rows.append(row)
                            summary["repeat"] = repeat
                            summary_rows.append(summary)
                            trial_id += 1
        grouped = {}
        for row in summary_rows:
            if row.get("status") != "complete":
                continue
            key = (
                # The requested initial state is the experimental design
                # axis.  Using the post-stabilization measurement here would
                # split repeats into separate sparse knots and defeat table
                # interpolation even though the transitions are matched.
                row["requested_initial_forward_velocity"],
                row["requested_initial_yaw_rate"],
                row["projected_forward_velocity"],
                row["projected_yaw_rate"],
            )
            grouped.setdefault(key, []).append(row)
        aggregate_rows = []
        table_rows = []
        for key, samples in sorted(grouped.items(), key=lambda item: str(item[0])):
            aggregate = aggregate_stage13_trials(samples)
            aggregate.update({
                "requested_initial_forward_velocity": key[0],
                "requested_initial_yaw_rate": key[1],
                "projected_forward_velocity": key[2],
                "projected_yaw_rate": key[3],
            })
            aggregate_rows.append(aggregate)
            table = {
                "current_v": key[0],
                "projected_v": aggregate["projected_forward_velocity"],
                "projected_w": aggregate["projected_yaw_rate"],
            }
            for horizon in STAGE13_HORIZONS_MS:
                table["mean_actual_v_%dms" % horizon] = aggregate["mean_actual_v_%dms" % horizon]
                table["mean_actual_w_%dms" % horizon] = aggregate["mean_actual_w_%dms" % horizon]
            table_rows.append(table)
        _write_csv(os.path.join(output_dir, "dynamic_response_raw.csv"), STAGE13_TRACE_FIELDS, raw_rows)
        aggregate_fields = tuple(sorted(set().union(*(set(row) for row in summary_rows)))) if summary_rows else ()
        _write_csv(os.path.join(output_dir, "dynamic_response_trials.csv"), aggregate_fields, summary_rows)
        grouped_fields = tuple(sorted(set().union(*(set(row) for row in aggregate_rows)))) if aggregate_rows else ()
        _write_csv(os.path.join(output_dir, "dynamic_response_aggregated.csv"), grouped_fields, aggregate_rows)
        _write_csv(os.path.join(output_dir, "dynamic_reachability_table.csv"), tuple(sorted(set().union(*(set(row) for row in table_rows)))) if table_rows else (), table_rows)
        result = {
            "task": args.task,
            "checkpoint": args.checkpoint,
            "seed": args.stage13_seed,
            "simulation_dt": float(env.sim_params.dt),
            "control_dt": float(env.dt),
            "physics_hz": 1.0 / float(env.sim_params.dt),
            "policy_hz": 1.0 / float(env.dt),
            "decimation": int(env.cfg.control.decimation),
            "policy_steps_per_200ms": 10,
            "initial_forward_values": INITIAL_FORWARD_VALUES,
            "initial_yaw_values": INITIAL_YAW_VALUES,
            "forward_command_grid": forward_commands,
            "yaw_command_grid": yaw_commands,
            "repeat_count": args.stage13_repeats,
            "trial_count": len(summary_rows),
            "completed_trial_count": sum(row.get("status") == "complete" for row in summary_rows),
            "initial_state_not_stabilized_count": sum(row.get("status") == "initial_state_not_stabilized" for row in summary_rows),
            "simulation_instability_count": sum(row.get("status") == "simulation_unstable" for row in summary_rows),
            "table_row_count": len(table_rows),
        }
        with open(os.path.join(output_dir, "stage1_3_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result, indent=2))
    finally:
        _close_env(env)


if __name__ == "__main__":
    main()
