"""Evaluate the explainable V1 velocity teacher through the frozen V62 stack."""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import isaacgym  # noqa: F401 - must precede torch
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.navigation.v1_velocity_teacher import (
    V1VelocityTeacherConfig,
    evaluate_teacher_gate,
    summarize_teacher_episodes,
    teacher_velocity_diagnostics,
)
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


def _assign_fixed_goal(env, env_index, distance):
    """Override V1's corridor-length reset goal for one teacher distance."""
    index = int(env_index)
    yaw = float(
        env._yaw_from_quaternion(env.root_states[index:index + 1, 3:7])[0].item()
    )
    env.global_goal_xy_world[index, 0] = env.root_states[index, 0] + float(distance) * math.cos(yaw)
    env.global_goal_xy_world[index, 1] = env.root_states[index, 1] + float(distance) * math.sin(yaw)
    env.goal_dist[index] = float(distance)
    env.terminal_goal_distance[index] = float(distance)
    env.previous_goal_distance[index] = float(distance)
    env.goal_reached_buf[index] = False
    env.success_buf[index] = False


def _record_episode(state, env, env_index, status, forced_timeout=False):
    path_length = float(state["path_length_m"])
    distance = float(state["distance_m"])
    return {
        "episode_id": int(state["episode_id"]),
        "distance_m": distance,
        "initial_goal_distance_m": distance,
        "success": bool(status["success"]),
        "collision": bool(status["collision"]),
        "timeout": bool(forced_timeout or status["timeout"]),
        "steps": int(state["steps"]),
        "path_length_m": path_length,
        "spl": (distance / max(path_length, distance)) if status["success"] else 0.0,
        "terminal_goal_distance_m": float(status["goal_distance"]),
        "teacher_command_count": int(state["teacher_command_count"]),
        "reverse_command_count": int(state["reverse_command_count"]),
        "projection_activation_count": int(state["projection_activation_count"]),
        "projection_correction_sum": float(state["projection_correction_sum"]),
        "projection_correction_max": float(state["projection_correction_max"]),
        "governor_modification_count": int(state["governor_modification_count"]),
        "tracking_sample_count": int(state["tracking_sample_count"]),
        "tracking_v_abs_error_sum": float(state["tracking_v_abs_error_sum"]),
        "tracking_w_abs_error_sum": float(state["tracking_w_abs_error_sum"]),
        "teacher_v_sum": float(state["teacher_v_sum"]),
        "teacher_v_sq_sum": float(state["teacher_v_sq_sum"]),
        "teacher_v_min": float(state["teacher_v_min"]),
        "teacher_v_max": float(state["teacher_v_max"]),
        "teacher_w_sum": float(state["teacher_w_sum"]),
        "teacher_w_sq_sum": float(state["teacher_w_sq_sum"]),
        "teacher_w_min": float(state["teacher_w_min"]),
        "teacher_w_max": float(state["teacher_w_max"]),
        # Retain the legacy per-episode fields for compatibility with quick
        # inspection tools that predate the formal teacher summary.
        "mean_projection_correction_norm": float(
            state["projection_correction_sum"] / max(state["teacher_command_count"], 1)
        ),
        "max_projection_correction_norm": float(
            state["projection_correction_max"]
        ),
        "mean_actual_speed_mps": float(
            state["actual_speed_sum"] / max(state["steps"], 1)
        ),
    }


def _new_episode_state(env, index, episode_id, distance):
    return {
        "episode_id": episode_id,
        "distance_m": distance,
        "steps": 0,
        "path_length_m": 0.0,
        "previous_position": env.root_states[index, :2].detach().clone(),
        "projection_correction_sum": 0.0,
        "projection_correction_max": 0.0,
        "teacher_command_count": 0,
        "reverse_command_count": 0,
        "projection_activation_count": 0,
        "governor_modification_count": 0,
        "tracking_sample_count": 0,
        "tracking_v_abs_error_sum": 0.0,
        "tracking_w_abs_error_sum": 0.0,
        "teacher_v_sum": 0.0,
        "teacher_v_sq_sum": 0.0,
        "teacher_v_min": math.inf,
        "teacher_v_max": -math.inf,
        "teacher_w_sum": 0.0,
        "teacher_w_sq_sum": 0.0,
        "teacher_w_min": math.inf,
        "teacher_w_max": -math.inf,
        "actual_speed_sum": 0.0,
    }


def _record_teacher_decisions(active, teacher):
    """Accumulate one 5 Hz teacher decision per active environment."""
    applied = teacher["applied_command"].detach()
    correction = teacher["projection_correction_norm"].detach()
    projected = correction > 1.0e-5
    for index, state in active.items():
        v = float(applied[index, 0].item())
        w = float(applied[index, 1].item())
        state["teacher_command_count"] += 1
        state["reverse_command_count"] += int(v < -1.0e-5)
        state["projection_activation_count"] += int(projected[index].item())
        state["projection_correction_sum"] += float(correction[index].item())
        state["projection_correction_max"] = max(
            state["projection_correction_max"], float(correction[index].item())
        )
        state["teacher_v_sum"] += v
        state["teacher_v_sq_sum"] += v * v
        state["teacher_v_min"] = min(state["teacher_v_min"], v)
        state["teacher_v_max"] = max(state["teacher_v_max"], v)
        state["teacher_w_sum"] += w
        state["teacher_w_sq_sum"] += w * w
        state["teacher_w_min"] = min(state["teacher_w_min"], w)
        state["teacher_w_max"] = max(state["teacher_w_max"], w)


def evaluate_distance(
    distance, episodes, seed, output_dir, num_envs=16, max_steps=None, framework_args=()
):
    started = time.monotonic()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = _parse_framework_args(framework_args)
    args.task = "rotunbot_sru_visual_corridor_v1"
    args.seed = int(seed)
    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg.seed = int(seed)
    env_cfg.env.num_envs = min(max(1, int(num_envs)), int(episodes))
    env_cfg.camera.depth_backend = "fallback"
    env_cfg.camera.add_noise = False
    env_cfg.commands.v1_goal_curriculum_enabled = False
    env_cfg.commands.v1_performance_curriculum_enabled = False
    env_cfg.commands.goal_distance = (float(distance), float(distance))
    env_cfg.commands.goal_bearing = (0.0, 0.0)
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    formal_steps = int(
        round(45.0 / (env_cfg.sim.dt * env_cfg.control.decimation))
    )
    max_steps = formal_steps if max_steps is None else int(max_steps)
    teacher_cfg = V1VelocityTeacherConfig(
        max_forward_speed=float(env_cfg.commands.max_forward_speed),
        max_yaw_rate=float(env_cfg.commands.max_yaw_rate),
        minimum_turn_radius=float(env_cfg.commands.minimum_turn_radius),
        feasible_envelope_fraction=float(env_cfg.commands.feasible_envelope_fraction),
        goal_radius=float(env_cfg.commands.goal_radius),
    )
    env = None
    records = []
    episode_counter = 0
    try:
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
        env.reset()
        active = {}
        for index in range(env.num_envs):
            _assign_fixed_goal(env, index, distance)
            active[index] = _new_episode_state(env, index, episode_counter, distance)
            episode_counter += 1
        held_actions = torch.zeros(env.num_envs, 2, device=env.device)
        held_teacher = torch.zeros(env.num_envs, 2, device=env.device)
        with torch.inference_mode():
            while len(records) < int(episodes):
                if env.common_step_counter % env.upper_level_command_interval_steps == 0:
                    actual = torch.stack(
                        (env.tracking_lin_vel[:, 0], env.tracking_ang_vel[:, 2]), dim=1
                    )
                    teacher = teacher_velocity_diagnostics(
                        env._goal_xy_robot(), actual, env.obstacle_clearance, teacher_cfg
                    )
                    held_teacher.copy_(teacher["applied_command"])
                    held_actions[:, 0] = held_teacher[:, 0] / teacher_cfg.max_forward_speed
                    held_actions[:, 1] = held_teacher[:, 1] / teacher_cfg.max_yaw_rate
                    held_actions.clamp_(-1.0, 1.0)
                    _record_teacher_decisions(active, teacher)
                previous_positions = env.root_states[:, :2].detach().clone()
                _, _, _, dones, _ = env.step(held_actions)
                done_mask = dones.flatten().bool()
                for index, state in list(active.items()):
                    state["steps"] += 1
                    position = env.root_states[index, :2].detach().clone()
                    if bool(done_mask[index].item()):
                        position = env.terminal_position[index].detach().clone()
                    state["path_length_m"] += float(
                        torch.linalg.vector_norm(position - state["previous_position"]).item()
                    )
                    state["previous_position"] = position
                    is_done = bool(done_mask[index].item())
                    actual_velocity = (
                        env.terminal_tracking_velocity[index]
                        if is_done
                        else torch.stack(
                            (env.tracking_lin_vel[index, 0], env.tracking_ang_vel[index, 2])
                        )
                    )
                    applied_v62 = (
                        env.terminal_applied_feasible_command[index]
                        if is_done
                        else env.applied_feasible_command[index]
                    )
                    state["tracking_sample_count"] += 1
                    state["tracking_v_abs_error_sum"] += abs(
                        float(actual_velocity[0].item())
                        - float(held_teacher[index, 0].item())
                    )
                    state["tracking_w_abs_error_sum"] += abs(
                        float(actual_velocity[1].item())
                        - float(held_teacher[index, 1].item())
                    )
                    state["governor_modification_count"] += int(
                        torch.linalg.vector_norm(
                            applied_v62 - held_teacher[index]
                        ).item()
                        > 1.0e-5
                    )
                    actual_speed = actual_velocity[0]
                    state["actual_speed_sum"] += float(actual_speed.item())
                    forced_timeout = state["steps"] >= max_steps and not is_done
                    if not is_done and not forced_timeout:
                        continue
                    if is_done:
                        status = {
                            "success": bool(env.terminal_success[index].item()),
                            "collision": bool(env.terminal_collision[index].item()),
                            "timeout": bool(env.terminal_timeout[index].item()),
                            "goal_distance": float(env.terminal_goal_distance[index].item()),
                        }
                    else:
                        status = {
                            "success": False,
                            "collision": False,
                            "timeout": True,
                            "goal_distance": float(env.goal_dist[index].item()),
                        }
                    records.append(_record_episode(state, env, index, status, forced_timeout))
                    if len(records) >= int(episodes):
                        break
                    if forced_timeout:
                        env.reset_idx(torch.as_tensor([index], device=env.device, dtype=torch.long))
                    _assign_fixed_goal(env, index, distance)
                    active[index] = _new_episode_state(env, index, episode_counter, distance)
                    episode_counter += 1
        fields = list(records[0]) if records else ["distance_m", "success"]
        with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        summary = summarize_teacher_episodes(records)
        summary.update(
            {
                "distance_m": float(distance),
                "successes": summary["success_count"],
                "collisions": summary["collision_count"],
                "timeouts": summary["timeout_count"],
                "teacher_config": teacher_cfg.__dict__,
                "max_steps": max_steps,
                "seed": int(seed),
                "wall_clock_seconds": time.monotonic() - started,
                "depth_backend": env.depth_backend_actual,
            }
        )
        threshold = 0.98 if float(distance) <= 1.5 else 0.95
        summary["gate"] = evaluate_teacher_gate(summary, threshold)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary
    finally:
        _close_environment(env)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", nargs="+", type=float, default=[1.0, 1.5, 2.0, 2.5])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", required=True)
    stage_args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    result = {}
    for distance in stage_args.distances:
        result[str(distance)] = evaluate_distance(
            distance,
            stage_args.episodes,
            stage_args.seed,
            Path(stage_args.output_dir) / ("distance_%0.1fm" % distance),
            num_envs=stage_args.num_envs,
            max_steps=stage_args.max_steps,
            framework_args=remaining,
        )
    root = Path(stage_args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    formal_pass = all(summary.get("gate", {}).get("pass", False) for summary in result.values())
    payload = {
        "formal_gate_pass": formal_pass,
        "distances": result,
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
