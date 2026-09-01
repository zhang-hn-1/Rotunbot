"""Evaluate a recurrent visual V1 policy on mirrored fixed L-turn scenes."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.navigation.direct_velocity import normalized_action_to_velocity_command
from legged_gym.navigation.v1_l_turn import build_l_turn_geometry
from legged_gym.navigation.v1_waypoint_manager import V1WaypointManager
from legged_gym.scripts.evaluate_sru_direct_velocity import _parse_framework_args, _raw_velocity_command
from legged_gym.scripts.eval_sru_visual_corridor_v1 import reset_recurrent_hidden
from legged_gym.utils import task_registry


def _commit_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _close_environment(env):
    if env is None:
        return
    if getattr(env, "viewer", None) is not None:
        env.gym.destroy_viewer(env.viewer)
    if getattr(env, "sim", None) is not None:
        env.gym.destroy_sim(env.sim)


def _parse_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=2250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", required=True)
    parsed, remaining = parser.parse_known_args(argv)
    return parsed, remaining


def _pose(env):
    xy = env.root_states[0, :2] - env.env_origins[0, :2]
    yaw = env._yaw_from_quaternion(env.root_states[0:1, 3:7])[0]
    return float(xy[0]), float(xy[1]), float(yaw)


def _nearest_centerline_distance(position, centerline):
    points = torch.as_tensor(centerline, dtype=torch.float32)
    point = torch.as_tensor(position, dtype=torch.float32)
    return float(torch.linalg.vector_norm(points - point, dim=1).min().item())


def _assign_l_goal(env, geometry):
    goal = torch.as_tensor(
        geometry.scenario.goal_xy, dtype=env.root_states.dtype, device=env.device
    ) + env.env_origins[0, :2]
    env.global_goal_xy_world[0] = goal
    distance = torch.linalg.vector_norm(goal - env.root_states[0, :2])
    env.goal_dist[0] = distance
    env.terminal_goal_distance[0] = distance
    env.previous_goal_distance[0] = distance
    env.goal_reached_buf[0] = False
    env.success_buf[0] = False


def _failure_class(record):
    if record["success"]:
        return "SUCCESS"
    if record["collision"]:
        return "WALL_COLLISION"
    if record["timeout"] and not record["turn_completion"]:
        return "NO_TURN"
    if record["wrong_turn"]:
        return "WRONG_TURN"
    if record["turn_start_error_m"] > 1.0:
        return "TURN_TOO_LATE"
    if record["reverse_divergence"]:
        return "REVERSE_DIVERGENCE"
    if record["timeout"] and record["turn_completion"]:
        return "POST_TURN_STALL"
    if record["timeout"]:
        return "TIMEOUT"
    return "OTHER"


def _evaluate_scene(env, policy, actor_critic, scene, geometry, episodes, max_steps):
    records = []
    trajectories = []
    env.reset()
    _assign_l_goal(env, geometry)
    reset_recurrent_hidden(actor_critic, torch.ones(1, dtype=torch.bool, device=env.device))
    manager = V1WaypointManager(geometry.waypoints, reach_radius=0.35)
    held_action = torch.zeros(1, 2, device=env.device)
    held_raw = torch.zeros(1, 2, device=env.device)
    held_requested = torch.zeros(1, 2, device=env.device)
    for episode_id in range(int(episodes)):
        if episode_id:
            env.reset_idx(torch.tensor([0], dtype=torch.long, device=env.device))
            _assign_l_goal(env, geometry)
            manager.reset()
            reset_recurrent_hidden(actor_critic, torch.ones(1, dtype=torch.bool, device=env.device))
        state = {
            "steps": 0,
            "path": 0.0,
            "previous_position": env.root_states[0, :2].detach().cpu().clone(),
            "min_wall": float("inf"),
            "v_sum": 0.0,
            "abs_w_sum": 0.0,
            "reverse_steps": 0,
            "reverse_run": 0,
            "max_reverse_run": 0,
            "wrong_turn": False,
            "wrong_turn_steps": 0,
            "turn_start_error_m": 0.0,
            "turn_started": False,
            "post_turn_positive": 0,
            "post_turn_samples": 0,
            "corner_deviation": 0.0,
            "trajectory": [],
        }
        done = False
        with torch.inference_mode():
            while not done and state["steps"] < int(max_steps):
                if env.common_step_counter % env.upper_level_command_interval_steps == 0:
                    pose = _pose(env)
                    manager.update(pose)
                    waypoint_world = torch.as_tensor(
                        geometry.waypoints[manager.current_index],
                        dtype=env.root_states.dtype,
                        device=env.device,
                    ) + env.env_origins[0, :2]
                    env.set_observation_goal_world(waypoint_world.reshape(1, 2))
                    env.compute_observations()
                    action = policy(env.get_observations()).clamp(-1.0, 1.0)
                    held_action.copy_(action)
                    held_raw.copy_(_raw_velocity_command(
                        action, env.cfg.commands.max_forward_speed, env.cfg.commands.max_yaw_rate
                    ))
                    held_requested.copy_(normalized_action_to_velocity_command(
                        action,
                        env.cfg.commands.max_forward_speed,
                        env.cfg.commands.max_yaw_rate,
                        env.cfg.commands.minimum_turn_radius,
                        env.cfg.commands.feasible_envelope_fraction,
                        preserve_curvature_when_saturating=bool(
                            getattr(env.cfg.commands, "preserve_curvature_when_saturating", False)
                        ),
                    ))
                previous = env.root_states[0, :2].detach().cpu().clone()
                _, _, _, dones, _ = env.step(held_action)
                state["steps"] += 1
                done_now = bool(dones.flatten()[0].item())
                current = (
                    env.terminal_position[0, :2].detach().cpu().clone()
                    if done_now
                    else env.root_states[0, :2].detach().cpu().clone()
                )
                state["path"] += float(torch.linalg.vector_norm(current - previous).item())
                state["min_wall"] = min(state["min_wall"], float(env.obstacle_clearance[0].item()))
                actual_v = float(
                    env.terminal_tracking_velocity[0, 0].item()
                    if done_now else env.tracking_lin_vel[0, 0].item()
                )
                actual_w = float(
                    env.terminal_tracking_velocity[0, 1].item()
                    if done_now else env.tracking_ang_vel[0, 2].item()
                )
                state["v_sum"] += actual_v
                state["abs_w_sum"] += abs(actual_w)
                reverse = actual_v < -3.0e-6
                state["reverse_steps"] += int(reverse)
                state["reverse_run"] = state["reverse_run"] + 1 if reverse else 0
                state["max_reverse_run"] = max(state["max_reverse_run"], state["reverse_run"])
                state["corner_deviation"] = max(
                    state["corner_deviation"],
                    _nearest_centerline_distance(current, geometry.scenario.centerline),
                )
                if manager.current_index in (1, 2):
                    if abs(float(held_requested[0, 1])) > 0.005:
                        if not state["turn_started"]:
                            state["turn_started"] = True
                            state["turn_start_error_m"] = abs(float(current[0]) - geometry.waypoints[1, 0])
                        wrong = geometry.turn_direction * float(held_requested[0, 1]) < -0.005
                        state["wrong_turn_steps"] += int(wrong)
                        # Ignore isolated near-zero transients; require two
                        # seconds of opposing turn commands before labeling a
                        # real wrong-turn failure.
                        state["wrong_turn"] |= state["wrong_turn_steps"] >= 20
                if manager.current_index >= 3:
                    state["post_turn_samples"] += 1
                    state["post_turn_positive"] += int(actual_v > 0.02)
                state["trajectory"].append({
                    "episode_id": episode_id,
                    "step": state["steps"],
                    "macro_step": (state["steps"] - 1) // 10,
                    "scene_type": scene,
                    "x": float(current[0]),
                    "y": float(current[1]),
                    "goal_distance": float(
                        env.terminal_goal_distance[0].item()
                        if done_now else env.goal_dist[0].item()
                    ),
                    "waypoint_index": manager.current_index,
                    "raw_v_cmd": float(held_raw[0, 0]),
                    "raw_w_cmd": float(held_raw[0, 1]),
                    "requested_v_cmd": float(held_requested[0, 0]),
                    "requested_w_cmd": float(held_requested[0, 1]),
                    "v_cmd": float(env.applied_feasible_command[0, 0]),
                    "w_cmd": float(env.applied_feasible_command[0, 1]),
                    "v_actual": actual_v,
                    "w_actual": actual_w,
                })
                done = done_now
            pose = _pose(env)
            manager.update(pose)
        forced_timeout = not done
        success = bool(env.terminal_success[0].item()) if done else False
        collision = bool(env.terminal_collision[0].item()) if done else False
        timeout = forced_timeout or (bool(env.terminal_timeout[0].item()) if done else False)
        record = {
            "scene_type": scene,
            "episode_id": episode_id,
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "steps": state["steps"],
            "completion_time_s": state["steps"] * float(env.dt),
            "path_length_m": state["path"],
            "min_wall_distance_m": state["min_wall"],
            "mean_v_mps": state["v_sum"] / max(state["steps"], 1),
            "mean_abs_w_rps": state["abs_w_sum"] / max(state["steps"], 1),
            "reverse_step_ratio": state["reverse_steps"] / max(state["steps"], 1),
            "max_consecutive_reverse_steps": state["max_reverse_run"],
            "turn_completion": manager.current_index >= 3,
            "wrong_turn": bool(state["wrong_turn"]),
            "wrong_turn_steps": state["wrong_turn_steps"],
            "turn_start_error_m": state["turn_start_error_m"],
            "max_corner_deviation_m": state["corner_deviation"],
            "post_turn_positive_v_ratio": state["post_turn_positive"] / max(state["post_turn_samples"], 1),
            "reverse_divergence": state["max_reverse_run"] >= 30 and state["reverse_steps"] / max(state["steps"], 1) > 0.25,
            "trajectory_rows": len(state["trajectory"]),
        }
        record["failure_class"] = _failure_class(record)
        records.append(record)
        trajectories.extend(state["trajectory"])
    return records, trajectories


def main(argv=None):
    stage_args, remaining = _parse_args(sys.argv[1:] if argv is None else argv)
    args = _parse_framework_args(remaining)
    args.task = "rotunbot_sru_visual_corridor_v1"
    args.seed = int(stage_args.seed)
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.enable_camera_sensors_in_headless = True
    env_cfg.camera.depth_backend = "isaacgym"
    env_cfg.camera.add_noise = False
    env_cfg.init_state.random_start_lateral = 0.0
    env_cfg.init_state.random_start_yaw = 0.0
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.v1_goal_curriculum_enabled = False
    env_cfg.commands.v1_performance_curriculum_enabled = False
    output = Path(stage_args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_scene_results = {}
    env = None
    for scene in ("L_LEFT", "L_RIGHT"):
        geometry = build_l_turn_geometry("left" if scene == "L_LEFT" else "right")
        env_cfg.corridor_width_m = geometry.scenario.width_m
        env_cfg.corridor_wall_width_m = geometry.scenario.width_m
        env_cfg.corridor_wall_segments = geometry.wall_segments
        env_cfg.direct_obstacle_aabbs = geometry.obstacle_aabbs
        try:
            env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
            train_cfg.runner.resume = False
            runner, _ = task_registry.make_alg_runner(
                env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
            )
            runner.load(str(stage_args.checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=env.device)
            records, trajectories = _evaluate_scene(
                env, policy, runner.alg.actor_critic, scene, geometry,
                stage_args.episodes, stage_args.max_steps,
            )
            scene_root = output / scene
            scene_root.mkdir(parents=True, exist_ok=True)
            with (scene_root / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            with (scene_root / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(trajectories[0]))
                writer.writeheader()
                writer.writerows(trajectories)
            counts = {}
            for item in records:
                counts[item["failure_class"]] = counts.get(item["failure_class"], 0) + 1
            all_scene_results[scene] = {
                "episodes": len(records),
                "success_count": sum(bool(item["success"]) for item in records),
                "success_rate": sum(bool(item["success"]) for item in records) / max(len(records), 1),
                "collision_count": sum(bool(item["collision"]) for item in records),
                "timeout_count": sum(bool(item["timeout"]) for item in records),
                "turn_completion_rate": sum(bool(item["turn_completion"]) for item in records) / max(len(records), 1),
                "wrong_turn_count": sum(bool(item["wrong_turn"]) for item in records),
                "mean_turn_start_error_m": sum(item["turn_start_error_m"] for item in records) / max(len(records), 1),
                "mean_max_corner_deviation_m": sum(item["max_corner_deviation_m"] for item in records) / max(len(records), 1),
                "mean_post_turn_positive_v_ratio": sum(item["post_turn_positive_v_ratio"] for item in records) / max(len(records), 1),
                "mean_min_wall_distance_m": sum(item["min_wall_distance_m"] for item in records) / max(len(records), 1),
                "mean_path_length_m": sum(item["path_length_m"] for item in records) / max(len(records), 1),
                "failure_histogram": counts,
            }
        finally:
            _close_environment(env)
            env = None
    gate = all(
        result["success_rate"] >= 0.90
        and result["collision_count"] == 0
        and result["timeout_count"] == 0
        and result["wrong_turn_count"] == 0
        for result in all_scene_results.values()
    )
    payload = {
        "stage": "L_TURN_STUDENT",
        "status": "PASS" if gate else "FAIL",
        "commit": _commit_sha(),
        "checkpoint": str(Path(stage_args.checkpoint).resolve()),
        "seed": int(stage_args.seed),
        "episodes_per_scene": int(stage_args.episodes),
        "max_steps": int(stage_args.max_steps),
        "depth_backend_requested": "isaacgym",
        "depth_backend_actual": "isaacgym",
        "geometry": {
            "width_m": 3.0,
            "first_segment_length_m": 1.5,
            "second_segment_length_m": 1.5,
            "turn_radius_m": 2.0,
            "corner_clearance_m": 0.60,
        },
        "scenes": all_scene_results,
    }
    (output / "l_turn_student_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
