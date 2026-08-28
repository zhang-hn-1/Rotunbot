"""Evaluate the frozen V49 policy on empty-map multi-waypoint sequences.

The evaluator owns only the world-frame waypoint layer.  It sends a projected
``[v, w]`` command to the existing V49 task, holds that command for ten 50 Hz
policy steps, and never resets the environment between waypoints.
"""

import argparse
import csv
import json
import math
import os
import sys

import isaacgym  # noqa: F401 - must precede torch/task imports
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403 - task registration
from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import (
    command_update_interval_steps,
    yaw_from_quaternion,
)
from legged_gym.navigation.v49_waypoint_controller import (
    V49WaypointConfig,
    WaypointSequenceController,
)
from legged_gym.utils import get_args, task_registry


TRAJECTORIES = {
    "A": ((1.0, 0.0), (2.0, 0.0), (3.0, 0.0)),
    "B": ((1.0, 0.0), (2.0, 0.25), (3.0, 0.0)),
}
INITIAL_YAWS_DEG = (-15, -10, -5, 0, 5, 10, 15)
PROFILE_ASSETS = {
    "maze": "Rotunbot.urdf",
    "v49_reference": "Rotunbot_test2.urdf",
}
LOG_FIELDS = (
    "time_s", "episode_id", "policy_step", "active_waypoint_index",
    "pose_x", "pose_y", "pose_yaw", "target_x", "target_y", "distance",
    "bearing_error", "raw_v", "raw_w", "projected_v", "projected_w",
    "measured_v", "measured_w", "waypoint_reached", "waypoint_switched",
    "sequence_complete", "reset_count", "timeout", "nan_inf",
)


def trajectory_waypoints(name):
    """Return the immutable Stage 1 waypoint tuple for trajectory A or B."""
    key = str(name).upper()
    if key not in TRAJECTORIES:
        raise ValueError("trajectory must be A or B")
    return TRAJECTORIES[key]


def initial_pose_for_episode(seed, episode_id):
    """Return deterministic ``(x, y, yaw)`` initialization for one episode."""
    generator = np.random.RandomState(int(seed) + int(episode_id))
    x, y = generator.uniform(-0.05, 0.05, size=2)
    yaw_deg = int(generator.choice(np.asarray(INITIAL_YAWS_DEG)))
    return float(x), float(y), float(yaw_deg) * math.pi / 180.0


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--waypoint_episodes", type=int, default=30)
    parser.add_argument("--waypoint_seed", type=int, default=20260828)
    parser.add_argument("--waypoint_output_dir", type=str, default=None)
    parser.add_argument(
        "--waypoint_asset_profile",
        choices=tuple(PROFILE_ASSETS),
        default="maze",
    )
    parser.add_argument(
        "--waypoint_trajectory", choices=("A", "B", "both"), default="both"
    )
    parser.add_argument("--waypoint_max_route_steps", type=int, default=2500)
    parser.add_argument("--waypoint_settle_s", type=float, default=2.0)
    original_argv = list(sys.argv)
    diagnostic, remaining = parser.parse_known_args()
    sys.argv = [original_argv[0]] + remaining
    try:
        args = get_args()
    finally:
        sys.argv = original_argv
    if args.task != "rotunbot_vel_sru50_v49_integration":
        raise ValueError(
            "Stage 1 requires --task rotunbot_vel_sru50_v49_integration"
        )
    if not args.load_run or args.checkpoint is None or int(args.checkpoint) < 0:
        raise ValueError("--load_run and an explicit --checkpoint are required")
    if diagnostic.waypoint_episodes < 1:
        raise ValueError("--waypoint_episodes must be positive")
    if diagnostic.waypoint_max_route_steps < 1:
        raise ValueError("--waypoint_max_route_steps must be positive")
    args.load_run = os.path.abspath(args.load_run)
    args.checkpoint = int(args.checkpoint)
    args.num_envs = 1
    for name, value in vars(diagnostic).items():
        setattr(args, name, value)
    return args


def _configure(env_cfg, args):
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 90.0
    env_cfg.commands.resampling_time = 10000.0
    env_cfg.commands.direct_command_tracking = True
    env_cfg.commands.hold_upper_command_rate = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.asset.file = (
        "{LEGGED_GYM_ROOT_DIR}/resources/robots/Rotunbot/urdf/"
        + PROFILE_ASSETS[args.waypoint_asset_profile]
    )


def _close_env(env):
    try:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
    finally:
        if env.sim is not None:
            env.gym.destroy_sim(env.sim)


def _set_command(env, command):
    env.command_reference_is_smooth.fill_(False)
    env.set_command_targets(command)
    env.compute_observations()


def _set_initial_pose(env, pose):
    x, y, yaw = pose
    env.root_states[0, :3] = torch.as_tensor(
        [x, y, float(env.cfg.init_state.pos[2])], device=env.device
    ) + env.env_origins[0]
    env.root_states[0, 3:7] = torch.as_tensor(
        [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
        device=env.device,
    )
    env.root_states[0, 7:13] = 0.0
    env.dof_pos[0] = env.default_dof_pos
    env.dof_vel[0] = 0.0
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(torch.zeros(1, dtype=torch.int32, device=env.device)),
        1,
    )
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.root_states)
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel.zero_()
    env.base_ang_vel.zero_()
    env.tracking_heading[:] = yaw
    env.projected_gravity[:] = quat_rotate_inverse(
        env.base_quat, env.gravity_vec
    )
    env.compute_observations()


def _pose(env):
    xy = env.root_states[:, :2]
    yaw = yaw_from_quaternion(env.root_states[:, 3:7])
    return xy, yaw


def _finite_tensor(*tensors):
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


def _tick_row(env, episode_id, policy_step, tick, target, reset_count, timeout):
    xy, yaw = _pose(env)
    measured_v = env.tracking_lin_vel[:, 0]
    measured_w = env.tracking_ang_vel[:, 2]
    finite = _finite_tensor(
        xy, yaw, target, tick.raw_command, tick.projected_command,
        tick.distance, tick.bearing_error, measured_v, measured_w,
    )
    return {
        "time_s": float(policy_step * env.dt),
        "episode_id": int(episode_id),
        "policy_step": int(policy_step),
        "active_waypoint_index": int(tick.active_waypoint_index),
        "pose_x": float(xy[0, 0].item()),
        "pose_y": float(xy[0, 1].item()),
        "pose_yaw": float(yaw[0].item()),
        "target_x": float(target[0, 0].item()),
        "target_y": float(target[0, 1].item()),
        "distance": float(tick.distance[0].item()),
        "bearing_error": float(tick.bearing_error[0].item()),
        "raw_v": float(tick.raw_command[0, 0].item()),
        "raw_w": float(tick.raw_command[0, 1].item()),
        "projected_v": float(tick.projected_command[0, 0].item()),
        "projected_w": float(tick.projected_command[0, 1].item()),
        "measured_v": float(measured_v[0].item()),
        "measured_w": float(measured_w[0].item()),
        "waypoint_reached": bool(tick.waypoint_reached),
        "waypoint_switched": bool(tick.waypoint_switched),
        "sequence_complete": bool(tick.sequence_complete),
        "reset_count": int(reset_count),
        "timeout": bool(timeout),
        "nan_inf": not finite,
    }


def _step_policy(env, policy):
    with torch.no_grad():
        actions = policy(env.get_observations())
        return env.step(actions)


def _run_episode(env, policy, args, trajectory_name, episode_id):
    env.reset()
    _set_initial_pose(env, initial_pose_for_episode(args.waypoint_seed, episode_id))
    controller = WaypointSequenceController(
        torch.as_tensor(trajectory_waypoints(trajectory_name)),
        config=V49WaypointConfig(),
        policy_steps_per_tick=command_update_interval_steps(env.dt, 5.0),
    )
    rows = []
    reset_count = 0
    timeout = False
    hold_violations = 0
    projection_violations = 0
    previous_command = None
    previous_index = 0
    reached_count = 0
    last_tick = None
    policy_step = 0

    while policy_step < args.waypoint_max_route_steps:
        if policy_step % controller.policy_steps_per_tick == 0:
            xy, yaw = _pose(env)
            last_tick = controller.tick(xy, yaw)
            target = controller.waypoints[last_tick.active_waypoint_index].to(
                device=env.device
            ).unsqueeze(0)
            _set_command(env, last_tick.projected_command)
            if last_tick.waypoint_reached:
                reached_count += 1
            if last_tick.active_waypoint_index - previous_index > 1:
                projection_violations += 1
            previous_index = last_tick.active_waypoint_index
            previous_command = last_tick.projected_command.detach().clone()
            rows.append(_tick_row(
                env, episode_id, policy_step, last_tick, target,
                reset_count, timeout,
            ))
            if last_tick.sequence_complete:
                break

        _, _, _, dones, _ = _step_policy(env, policy)
        # The upper-layer request is assigned only in the tick branch above;
        # this explicit schedule check documents and audits the 10-step hold
        # without confusing V49's internal measured-state buffers with the
        # external command port.
        if policy_step % controller.policy_steps_per_tick:
            if previous_command is None:
                hold_violations += 1
        if bool(torch.any(dones)):
            reset_count += int(torch.sum(dones).item())
            timeout = bool(torch.any(env.time_out_buf))
            break
        policy_step += 1

    route_complete = bool(last_tick is not None and last_tick.sequence_complete)
    final_xy, _ = _pose(env)
    final_target = torch.as_tensor(
        trajectory_waypoints(trajectory_name)[-1],
        dtype=final_xy.dtype, device=final_xy.device,
    ).unsqueeze(0)
    arrival_error = float(torch.linalg.vector_norm(final_xy - final_target).item())
    settled_error = arrival_error
    settle_steps = int(round(args.waypoint_settle_s / env.dt))
    terminal_v = float(env.tracking_lin_vel[0, 0].item())
    if route_complete and reset_count == 0:
        zero_command = torch.zeros(1, 2, device=env.device)
        _set_command(env, zero_command)
        for _ in range(settle_steps):
            _, _, _, dones, _ = _step_policy(env, policy)
            if bool(torch.any(dones)):
                reset_count += int(torch.sum(dones).item())
                timeout = bool(torch.any(env.time_out_buf))
                break
        final_xy, _ = _pose(env)
        settled_error = float(torch.linalg.vector_norm(final_xy - final_target).item())
        terminal_v = float(env.tracking_lin_vel[0, 0].item())

    if any(row["nan_inf"] for row in rows):
        failure_classes = ["E_nan_inf_or_sim"]
    else:
        failure_classes = []
        if hold_violations or projection_violations:
            failure_classes.append("A_command_timing_or_projection")
        if reached_count < len(trajectory_waypoints(trajectory_name)):
            failure_classes.append("B_waypoint_geometry")
        if reset_count or timeout or (
            route_complete
            and controller.switch_count
            != len(trajectory_waypoints(trajectory_name)) - 1
        ):
            failure_classes.append("C_reset_done_or_skip")
        if not route_complete or arrival_error > 0.25 or abs(terminal_v) > 0.10:
            failure_classes.append("D_low_level_or_terminal")
    return {
        "trajectory": trajectory_name,
        "episode_id": int(episode_id),
        "initial_pose": initial_pose_for_episode(args.waypoint_seed, episode_id),
        "route_complete": route_complete,
        "waypoints_reached": int(reached_count),
        "waypoints_total": len(trajectory_waypoints(trajectory_name)),
        "switch_count": int(controller.switch_count),
        "reset_count": int(reset_count),
        "timeout": bool(timeout),
        "hold_violations": int(hold_violations),
        "projection_or_skip_violations": int(projection_violations),
        "final_position_error_m": arrival_error,
        "settled_position_error_m": settled_error,
        "terminal_measured_v_mps": terminal_v,
        "nan_inf": bool(any(row["nan_inf"] for row in rows)),
        "failure_classes": failure_classes,
        "rows": rows,
    }


def _write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(results, args, model_path):
    episodes = len(results)
    complete = sum(result["route_complete"] for result in results)
    reached = sum(result["waypoints_reached"] for result in results)
    total_waypoints = sum(result["waypoints_total"] for result in results)
    intermediate_resets = sum(result["reset_count"] for result in results)
    skips = sum(result["projection_or_skip_violations"] for result in results)
    nan_inf = sum(result["nan_inf"] for result in results)
    terminal_safe = sum(
        abs(result["terminal_measured_v_mps"]) <= 0.10 for result in results
    )
    success_ratio = float(complete) / max(episodes, 1)
    waypoint_ratio = float(reached) / max(total_waypoints, 1)
    final_error_ok = all(result["final_position_error_m"] <= 0.25 for result in results)
    checks = {
        "sequence_success_ge_90pct": success_ratio >= 0.90,
        "waypoint_reach_ge_95pct": waypoint_ratio >= 0.95,
        "final_error_le_0.25m": final_error_ok,
        "intermediate_resets_zero": intermediate_resets == 0,
        "skip_zero": skips == 0,
        "nan_inf_zero": nan_inf == 0,
        "terminal_speed_le_0.10mps": terminal_safe == episodes,
    }
    failure_counts = {}
    for result in results:
        for failure_class in result["failure_classes"]:
            failure_counts[failure_class] = failure_counts.get(failure_class, 0) + 1
    return {
        "task": args.task,
        "asset_profile": args.waypoint_asset_profile,
        "asset": PROFILE_ASSETS[args.waypoint_asset_profile],
        "checkpoint": args.checkpoint,
        "checkpoint_path": model_path,
        "physics_hz": 1.0 / float(args._sim_dt),
        "low_level_hz": 1.0 / float(args._policy_dt),
        "upper_command_hz": 5.0,
        "policy_steps_per_upper_tick": 10,
        "episodes": episodes,
        "sequence_success_count": int(complete),
        "sequence_success_ratio": success_ratio,
        "waypoints_reached": int(reached),
        "waypoints_total": int(total_waypoints),
        "waypoint_reach_ratio": waypoint_ratio,
        "intermediate_reset_count": int(intermediate_resets),
        "skip_count": int(skips),
        "nan_inf_count": int(nan_inf),
        "terminal_speed_safe_count": int(terminal_safe),
        "failure_class_counts": failure_counts,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "episodes_detail": results,
    }


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
    args._sim_dt = float(env.sim_params.dt)
    args._policy_dt = float(env.dt)
    try:
        runner, _ = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
        )
        policy = runner.get_inference_policy(device=env.device)
        trajectory_names = (
            ("A", "B") if args.waypoint_trajectory == "both"
            else (args.waypoint_trajectory,)
        )
        results = []
        output_dir = args.waypoint_output_dir or os.path.join(
            args.load_run, "v49_waypoint_sequence", args.waypoint_asset_profile
        )
        os.makedirs(output_dir, exist_ok=True)
        for trajectory_name in trajectory_names:
            all_rows = []
            for episode_id in range(args.waypoint_episodes):
                result = _run_episode(
                    env, policy, args, trajectory_name, episode_id
                )
                results.append(result)
                all_rows.extend(result.pop("rows"))
            _write_rows(
                os.path.join(
                    output_dir,
                    "trajectory_%s_%s.csv"
                    % (args.waypoint_asset_profile, trajectory_name),
                ),
                all_rows,
            )
        summary = _summarize(results, args, model_path)
        json_path = os.path.join(
            output_dir,
            "stage1_v49_waypoint_summary_%s.json" % args.waypoint_asset_profile,
        )
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("JSON: %s" % json_path)
        if summary["verdict"] != "PASS":
            raise SystemExit(1)
    finally:
        _close_env(env)


if __name__ == "__main__":
    main()
