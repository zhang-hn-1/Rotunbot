"""Run deterministic L_LEFT/L_RIGHT teacher smoke through frozen V62."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import isaacgym  # noqa: F401 - must precede torch
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.navigation.v1_l_turn import build_l_turn_geometry
from legged_gym.navigation.v1_velocity_teacher import (
    V1VelocityTeacherConfig,
    teacher_velocity_diagnostics,
)
from legged_gym.navigation.v1_waypoint_manager import V1WaypointManager
from legged_gym.navigation.v1_teacher_dataset import TeacherSequenceWriter
from legged_gym.utils import get_args, task_registry


def _parse_framework_args(remaining):
    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + list(remaining)
    try:
        return get_args()
    finally:
        os.sys.argv = original


def _close_environment(env):
    if env is None:
        return
    viewer = getattr(env, "viewer", None)
    if viewer is not None:
        env.gym.destroy_viewer(viewer)
    if getattr(env, "sim", None) is not None:
        env.gym.destroy_sim(env.sim)


def _commit_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _assign_final_goal(env, geometry):
    goal = torch.as_tensor(
        geometry.scenario.goal_xy, dtype=env.root_states.dtype, device=env.device
    )
    env.global_goal_xy_world[0] = env.env_origins[0, :2] + goal
    distance = torch.linalg.vector_norm(
        env.global_goal_xy_world[0] - env.root_states[0, :2]
    )
    env.goal_dist[0] = distance
    env.terminal_goal_distance[0] = distance
    env.previous_goal_distance[0] = distance
    env.goal_reached_buf[0] = False
    env.success_buf[0] = False


def _local_pose(env):
    position = env.root_states[0, :2] - env.env_origins[0, :2]
    yaw = env._yaw_from_quaternion(env.root_states[0:1, 3:7])[0]
    return position, yaw


def _new_state(scene, episode_id, env):
    return {
        "scene_type": scene,
        "episode_id": int(episode_id),
        "steps": 0,
        "path_length_m": 0.0,
        "previous_position": env.root_states[0, :2].detach().cpu().clone(),
        "min_wall_distance": math.inf,
        "v_sum": 0.0,
        "abs_w_sum": 0.0,
        "wrong_yaw_sign_steps": 0,
        "turn_waypoint_reached": False,
        "trajectory": [],
    }


def _trajectory_row(state, env, teacher, waypoint_index):
    position = env.root_states[0, :2].detach().cpu()
    actual = torch.stack((env.tracking_lin_vel[0, 0], env.tracking_ang_vel[0, 2]))
    applied = env.applied_feasible_command[0].detach().cpu()
    return {
        "scene_type": state["scene_type"],
        "episode_id": state["episode_id"],
        "step": state["steps"],
        "time_s": state["steps"] * float(env.dt),
        "x": float(position[0]),
        "y": float(position[1]),
        "goal_distance": float(env.goal_dist[0].item()),
        "waypoint_index": int(waypoint_index),
        "teacher_v": float(teacher["applied_command"][0, 0].item()),
        "teacher_w": float(teacher["applied_command"][0, 1].item()),
        "v_cmd": float(applied[0]),
        "w_cmd": float(applied[1]),
        "v_actual": float(actual[0].item()),
        "w_actual": float(actual[1].item()),
    }


def _finish_state(state, env, manager, success, collision, timeout):
    return {
        "scene_type": state["scene_type"],
        "episode_id": state["episode_id"],
        "success": bool(success),
        "collision": bool(collision),
        "timeout": bool(timeout),
        "steps": int(state["steps"]),
        "completion_time_s": float(state["steps"] * env.dt),
        "path_length_m": float(state["path_length_m"]),
        "min_wall_distance_m": float(state["min_wall_distance"]),
        "mean_v_mps": float(state["v_sum"] / max(state["steps"], 1)),
        "mean_abs_w_rps": float(state["abs_w_sum"] / max(state["steps"], 1)),
        "turn_completion": bool(manager.current_index >= 3),
        "wrong_yaw_sign_steps": int(state["wrong_yaw_sign_steps"]),
        "trajectory_rows": len(state["trajectory"]),
    }


def evaluate_scene(
    env,
    scene,
    episodes,
    seed,
    max_steps,
    teacher_cfg,
    geometry,
    dataset_writer=None,
    episode_offset=0,
):
    with torch.inference_mode():
        env.reset()
    if env.depth_backend_actual != "isaacgym":
        raise RuntimeError("L-turn teacher requires real Isaac Gym IMAGE_DEPTH")
    _assign_final_goal(env, geometry)
    manager = V1WaypointManager(geometry.waypoints, reach_radius=0.35)
    held_actions = torch.zeros(1, 2, device=env.device)
    records = []
    trajectories = []
    state = _new_state(scene, 0, env)
    episode_id = 0
    primitive_steps = 0
    next_teacher_step = int(env.common_step_counter)
    pending_dataset_row = None
    with torch.inference_mode():
        while len(records) < int(episodes):
            if env.common_step_counter >= next_teacher_step:
                if pending_dataset_row is not None and dataset_writer is not None:
                    dataset_writer.append(pending_dataset_row)
                position, yaw = _local_pose(env)
                pose = (
                    float(position[0].item()),
                    float(position[1].item()),
                    float(yaw.item()),
                )
                manager.update(pose)
                waypoint_robot = manager.get_current_waypoint_robot(pose)
                goal_xy = torch.as_tensor(
                    waypoint_robot, dtype=torch.float32, device=env.device
                ).reshape(1, 2)
                actual = torch.stack(
                    (env.tracking_lin_vel[:, 0], env.tracking_ang_vel[:, 2]), dim=1
                )
                teacher = teacher_velocity_diagnostics(
                    goal_xy,
                    actual,
                    env.obstacle_clearance,
                    teacher_cfg,
                )
                held_actions[:, 0] = (
                    teacher["applied_command"][:, 0] / teacher_cfg.max_forward_speed
                )
                held_actions[:, 1] = (
                    teacher["applied_command"][:, 1] / teacher_cfg.max_yaw_rate
                )
                held_actions.clamp_(-1.0, 1.0)
                expected_sign = float(geometry.turn_direction)
                if manager.current_index in (1, 2):
                    state["wrong_yaw_sign_steps"] += int(
                        expected_sign * float(teacher["applied_command"][0, 1]) < -1.0e-4
                    )
                state["turn_waypoint_reached"] |= manager.current_index >= 2
                state["min_wall_distance"] = min(
                    state["min_wall_distance"], float(env.obstacle_clearance[0].item())
                )
                current_teacher = teacher
                current_waypoint_index = manager.current_index
                next_teacher_step = (
                    int(env.common_step_counter)
                    + int(env.upper_level_command_interval_steps)
                )
                if dataset_writer is not None:
                    pending_dataset_row = {
                        "episode_id": int(episode_offset + episode_id),
                        "step_id": int(state["steps"] // 10),
                        "depth": env.depth_observation[0].detach().clone(),
                        "goal_xy_robot": goal_xy[0].detach().clone(),
                        "proprioception": env._proprioception()[0].detach().clone(),
                        "previous_command": env.previous_velocity_command[0].detach().clone(),
                        "previous_actual_velocity": env.previous_actual_velocity[0].detach().clone(),
                        "teacher_command": teacher["applied_command"][0].detach().clone(),
                        "actual_velocity": actual[0].detach().clone(),
                        "governor_command": env.applied_feasible_command[0].detach().clone(),
                        "projection_command": teacher["applied_command"][0].detach().clone(),
                        "done": False,
                        "success": False,
                        "collision": False,
                        "goal_distance": torch.linalg.vector_norm(goal_xy[0]).detach().cpu(),
                    }
            previous_position = env.root_states[0, :2].detach().cpu().clone()
            _, _, _, dones, _ = env.step(held_actions)
            primitive_steps += 1
            state["steps"] += 1
            position = env.root_states[0, :2].detach().cpu()
            state["path_length_m"] += float(
                torch.linalg.vector_norm(position - previous_position).item()
            )
            state["previous_position"] = position
            actual = torch.stack((env.tracking_lin_vel[0, 0], env.tracking_ang_vel[0, 2]))
            state["v_sum"] += float(actual[0].item())
            state["abs_w_sum"] += abs(float(actual[1].item()))
            state["trajectory"].append(
                _trajectory_row(state, env, current_teacher, current_waypoint_index)
            )
            done = bool(dones.flatten()[0].item())
            timeout = primitive_steps >= int(max_steps) and not done
            if (done or timeout) and pending_dataset_row is not None and dataset_writer is not None:
                pending_dataset_row.update(
                    {
                        "done": True,
                        "success": bool(env.terminal_success[0].item()) if done else False,
                        "collision": bool(env.terminal_collision[0].item()) if done else False,
                        "goal_distance": torch.tensor(
                            float(env.terminal_goal_distance[0].item())
                            if done else float(env.goal_dist[0].item())
                        ),
                    }
                )
                dataset_writer.append(pending_dataset_row)
                pending_dataset_row = None
            if not done and not timeout:
                continue
            success = bool(env.terminal_success[0].item()) if done else False
            collision = bool(env.terminal_collision[0].item()) if done else False
            terminal_timeout = bool(env.terminal_timeout[0].item()) if done else False
            record = _finish_state(
                state, env, manager, success, collision, timeout or terminal_timeout
            )
            records.append(record)
            trajectories.extend(state["trajectory"])
            if len(records) >= int(episodes):
                break
            env.reset_idx(torch.as_tensor([0], dtype=torch.long, device=env.device))
            episode_id += 1
            _assign_final_goal(env, geometry)
            manager.reset()
            state = _new_state(scene, episode_id, env)
            primitive_steps = 0
            next_teacher_step = int(env.common_step_counter)
    return records, trajectories


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-output", default=None)
    stage_args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    args = _parse_framework_args(remaining)
    args.task = "rotunbot_sru_visual_corridor_v1"
    args.seed = int(stage_args.seed)
    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg.seed = int(stage_args.seed)
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
    teacher_cfg = V1VelocityTeacherConfig(
        max_forward_speed=float(env_cfg.commands.max_forward_speed),
        max_yaw_rate=float(env_cfg.commands.max_yaw_rate),
        minimum_turn_radius=float(env_cfg.commands.minimum_turn_radius),
        feasible_envelope_fraction=float(env_cfg.commands.feasible_envelope_fraction),
        goal_radius=float(env_cfg.commands.goal_radius),
    )
    output = Path(stage_args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_results = {}
    all_records = []
    all_trajectories = []
    dataset_writer = (
        TeacherSequenceWriter(sequence_length=16)
        if stage_args.dataset_output
        else None
    )
    for scene in ("L_LEFT", "L_RIGHT"):
        geometry = build_l_turn_geometry("left" if scene == "L_LEFT" else "right")
        env_cfg.corridor_width_m = geometry.scenario.width_m
        env_cfg.corridor_wall_width_m = geometry.scenario.width_m
        env_cfg.corridor_wall_segments = geometry.wall_segments
        env_cfg.direct_obstacle_aabbs = geometry.obstacle_aabbs
        env = None
        try:
            env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
            records, trajectories = evaluate_scene(
                env,
                scene,
                stage_args.episodes,
                stage_args.seed,
                stage_args.max_steps,
                teacher_cfg,
                geometry,
                dataset_writer=dataset_writer,
                episode_offset=sum(
                    int(item.get("episodes", 0)) for item in scene_results.values()
                ),
            )
        finally:
            _close_environment(env)
        scene_results[scene] = {
            "episodes": len(records),
            "success_count": sum(bool(item["success"]) for item in records),
            "success_rate": sum(bool(item["success"]) for item in records) / max(len(records), 1),
            "collision_count": sum(bool(item["collision"]) for item in records),
            "timeout_count": sum(bool(item["timeout"]) for item in records),
            "mean_min_wall_distance_m": sum(item["min_wall_distance_m"] for item in records) / max(len(records), 1),
            "mean_path_length_m": sum(item["path_length_m"] for item in records) / max(len(records), 1),
            "mean_completion_time_s": sum(item["completion_time_s"] for item in records) / max(len(records), 1),
            "mean_v_mps": sum(item["mean_v_mps"] for item in records) / max(len(records), 1),
            "mean_abs_w_rps": sum(item["mean_abs_w_rps"] for item in records) / max(len(records), 1),
            "turn_completion_rate": sum(bool(item["turn_completion"]) for item in records) / max(len(records), 1),
            "wrong_yaw_sign_steps": sum(item["wrong_yaw_sign_steps"] for item in records),
        }
        all_records.extend(records)
        all_trajectories.extend(trajectories)

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

    total = len(all_records)
    gate = all(
        scene_results[scene]["success_rate"] >= 0.90
        and scene_results[scene]["collision_count"] / max(scene_results[scene]["episodes"], 1) <= 0.10
        and scene_results[scene]["timeout_count"] / max(scene_results[scene]["episodes"], 1) <= 0.10
        for scene in scene_results
    )
    payload = {
        "stage": "L_TURN_TEACHER",
        "status": "PASS" if gate else "FAIL",
        "commit": _commit_sha(),
        "seed": int(stage_args.seed),
        "episodes_per_scene": int(stage_args.episodes),
        "max_steps": int(stage_args.max_steps),
        "depth_backend_actual": "isaacgym",
        "geometry": {
            "width_m": 3.0,
            "first_segment_length_m": 1.5,
            "second_segment_length_m": 1.5,
            "turn_radius_m": 2.0,
            "corner_clearance_m": 0.60,
        },
        "scenes": scene_results,
        "overall_success_rate": sum(bool(item["success"]) for item in all_records) / max(total, 1),
    }
    if dataset_writer is not None:
        dataset_path = Path(stage_args.dataset_output).resolve()
        dataset_writer.save(
            dataset_path,
            metadata={
                "schema_name": "V1-compatible L-turn teacher dataset",
                "depth_backend_requested": "isaacgym",
                "depth_backend_actual": "isaacgym",
                "depth_representation": "normalized IMAGE_DEPTH, far-is-open",
                "scene_types": ["L_LEFT", "L_RIGHT"],
                "episodes_per_scene": int(stage_args.episodes),
                "sequence_length": 16,
                "seed": int(stage_args.seed),
                "geometry": payload["geometry"],
            },
        )
        payload["dataset"] = str(dataset_path)
    (output / "l_turn_teacher_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
