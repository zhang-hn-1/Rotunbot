"""Matched Baseline/Static/Dynamic GPU evaluation for Stage1.4."""

import argparse
import csv
import json
import os
import sys

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.navigation.v49_dynamic_governor import (
    DynamicGovernorConfig,
    StateDependentReachabilityGovernor,
)
from legged_gym.navigation.v49_dynamic_governor_diagnostics import (
    STAGE14_SCENARIOS,
    aggregate_governor_rows,
)
from legged_gym.navigation.v49_dynamic_reachability import (
    DynamicReachabilityTable,
    ReachabilityState,
)
from legged_gym.scripts.audit_v49_state_reset import _close_env, _make_runtime
from legged_gym.scripts.evaluate_v49_waypoint_sequence import (
    _set_command,
    _set_initial_pose,
    initial_pose_for_episode,
)
from legged_gym.scripts.sweep_v49_dynamic_reachability import (
    _establish_initial_state,
    _projection,
)
from legged_gym.utils import get_args


MODES = ("Baseline", "Static", "Dynamic")
FIELDS = (
    "mode", "scenario", "group", "trial", "command_index", "policy_step",
    "time_s", "requested_v", "requested_w", "selected_v", "selected_w",
    "static_v", "static_w", "measured_v", "measured_w", "v_error", "w_error",
    "requested_v_error", "requested_w_error",
    "command_modified", "forward_modified", "yaw_modified", "static_saturated",
    "fallback", "coverage", "yaw_sign_error", "oscillation", "unstable",
)


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage14_output_dir", default="logs/stage1_4_dynamic_governor")
    parser.add_argument("--stage14_table", default="logs/stage1_3_dynamic_reachability/dynamic_reachability_table.csv")
    parser.add_argument("--stage14_repeats", type=int, default=3)
    parser.add_argument("--stage14_stabilize_steps", type=int, default=180)
    parser.add_argument("--stage14_stable_window", type=int, default=10)
    parser.add_argument("--stage14_seed", type=int, default=20260829)
    parser.add_argument("--stage14_max_scenarios", type=int, default=0)
    original_argv = list(sys.argv)
    custom, remaining = parser.parse_known_args()
    sys.argv = [original_argv[0]] + remaining
    try:
        args = get_args()
    finally:
        sys.argv = original_argv
    if args.task != "rotunbot_vel_sru50_v49_integration":
        raise ValueError("Stage1.4 requires the V49 integration task")
    if not args.load_run or args.checkpoint is None:
        raise ValueError("--load_run and --checkpoint are required")
    if custom.stage14_repeats < 3:
        raise ValueError("Stage1.4 requires at least three repeats")
    args.load_run = os.path.abspath(args.load_run)
    args.checkpoint = int(args.checkpoint)
    args.num_envs = 1
    args.waypoint_asset_profile = "v49_reference"
    args.reach_stabilize_steps = custom.stage14_stabilize_steps
    args.reach_stable_window = custom.stage14_stable_window
    args.reach_seed = custom.stage14_seed
    for name, value in vars(custom).items():
        setattr(args, name, value)
    return args


def _governor_for_env(env, table):
    cfg = env.cfg.commands
    return StateDependentReachabilityGovernor(
        table,
        DynamicGovernorConfig(
            enable_dynamic_governor=True,
            maximum_forward_speed=float(cfg.max_forward_speed),
            maximum_yaw_rate=float(cfg.max_yaw_rate),
            minimum_turn_radius=float(cfg.minimum_turn_radius),
            envelope_fraction=float(cfg.feasible_envelope_fraction),
            stationary_threshold=float(env.cfg.rewards.stationary_command_threshold),
            turn_authority_start_speed=float(getattr(cfg, "turn_authority_start_speed", 0.0)),
            turn_authority_full_speed=float(getattr(cfg, "turn_authority_full_speed", 0.0)),
            maximum_forward_command_step=0.08,
            maximum_yaw_command_step=0.03,
            weight_forward_error=1.0,
            weight_yaw_error=8.0,
            weight_command_delta=0.10,
        ),
    )


def _set_mode(env, enabled):
    env.cfg.commands.dynamic_governor_enabled = bool(enabled)


def _record_row(mode, scenario, trial, command_index, policy_step, raw, selected,
                static, measured_v, measured_w, decision, static_saturated,
                oscillation, unstable, policy_dt):
    return {
        "mode": mode,
        "scenario": scenario.name,
        "group": scenario.group,
        "trial": trial,
        "command_index": command_index,
        "policy_step": policy_step,
        "time_s": (command_index * 10 + policy_step) * policy_dt,
        "requested_v": raw[0],
        "requested_w": raw[1],
        "selected_v": selected[0],
        "selected_w": selected[1],
        "static_v": static[0],
        "static_w": static[1],
        "measured_v": measured_v,
        "measured_w": measured_w,
        "v_error": selected[0] - measured_v,
        "w_error": selected[1] - measured_w,
        "requested_v_error": raw[0] - measured_v,
        "requested_w_error": raw[1] - measured_w,
        "command_modified": bool(decision and decision.modified),
        "forward_modified": bool(decision and decision.forward_modified),
        "yaw_modified": bool(decision and decision.yaw_modified),
        "static_saturated": bool(static_saturated),
        "fallback": bool(decision and decision.fallback),
        "coverage": decision.coverage if decision else "static",
        "yaw_sign_error": bool(abs(selected[1]) > 1.0e-6 and selected[1] * measured_w < -1.0e-8),
        "oscillation": bool(oscillation),
        "unstable": bool(unstable),
    }


def _run_mode(env, policy, governor, scenario, trial, args, mode):
    initial_v = 0.06 if scenario.group == "low_speed" else 0.12 if scenario.group == "high_speed" else 0.08
    env.reset()
    _set_initial_pose(env, initial_pose_for_episode(args.stage14_seed, trial))
    initial = _establish_initial_state(
        env, policy, initial_v, 0.0, args.stage14_seed, trial, args, reset=False
    )
    if initial["status"] != "stabilized":
        return [], False
    _set_mode(env, mode == "Dynamic")
    previous = (float(initial["projected_v"]), float(initial["projected_w"])) if "projected_v" in initial else (0.0, 0.0)
    rows = []
    unstable = False
    for command_index, raw in enumerate(scenario.commands):
        raw_tensor = torch.tensor([raw], device=env.device)
        static_tensor = _projection(env, raw_tensor)
        static = (float(static_tensor[0, 0]), float(static_tensor[0, 1]))
        decision = None
        if mode == "Baseline":
            selected_tensor = raw_tensor
            selected = raw
        elif mode == "Static":
            selected_tensor = static_tensor
            selected = static
        else:
            decision = env.set_governed_command_targets(raw_tensor, governor)[0]
            selected = decision.command
            selected_tensor = torch.tensor([selected], device=env.device)
            env.compute_observations()
        if mode != "Dynamic":
            _set_command(env, selected_tensor, smooth_reference=False)
        static_saturated = abs(raw[0] - static[0]) > 1.0e-6 or abs(raw[1] - static[1]) > 1.0e-6
        for policy_step in range(10):
            with torch.no_grad():
                action = policy(env.get_observations())
                _, _, _, dones, _ = env.step(action)
            measured_v = float(env.tracking_lin_vel[0, 0])
            measured_w = float(env.tracking_ang_vel[0, 2])
            unstable = unstable or bool(torch.any(dones)) or not bool(torch.isfinite(env.root_states).all())
            sign_flip = (
                policy_step == 0 and abs(previous[1]) > 1.0e-6 and
                abs(selected[1]) > 1.0e-6 and previous[1] * selected[1] < 0.0
            )
            rows.append(_record_row(
                mode, scenario, trial, command_index, policy_step, raw, selected,
                static, measured_v, measured_w, decision, static_saturated,
                sign_flip, unstable, float(env.dt),
            ))
            if bool(torch.any(dones)):
                break
        previous = selected
        if unstable:
            break
    return rows, True


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _parse_args()
    table = DynamicReachabilityTable.from_csv(args.stage14_table)
    env, policy = _make_runtime(args)
    output_dir = os.path.abspath(args.stage14_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    completed = 0
    try:
        governor = _governor_for_env(env, table)
        scenarios = STAGE14_SCENARIOS[:args.stage14_max_scenarios or None]
        for scenario_index, scenario in enumerate(scenarios):
            for trial in range(args.stage14_repeats):
                trial_id = scenario_index * args.stage14_repeats + trial
                for mode in MODES:
                    mode_rows, ok = _run_mode(env, policy, governor, scenario, trial_id, args, mode)
                    rows.extend(mode_rows)
                    completed += int(ok)
        _write_csv(os.path.join(output_dir, "stage1_4_trials.csv"), rows)
        aggregate = aggregate_governor_rows(rows)
        with open(os.path.join(output_dir, "stage1_4_aggregate.json"), "w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, indent=2)
        summary = {
            "task": args.task,
            "checkpoint": args.checkpoint,
            "seed": args.stage14_seed,
            "modes": MODES,
            "scenario_count": len(scenarios),
            "repeat_count": args.stage14_repeats,
            "completed_mode_trials": completed,
            "expected_mode_trials": len(scenarios) * args.stage14_repeats * len(MODES),
            "policy_hz": 1.0 / float(env.dt),
            "physics_hz": 1.0 / float(env.sim_params.dt),
            "policy_steps_per_command": 10,
            "metrics": aggregate,
        }
        with open(os.path.join(output_dir, "stage1_4_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))
    finally:
        _close_env(env)


if __name__ == "__main__":
    main()
