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
    return {
        "episode_id": int(state["episode_id"]),
        "distance_m": float(state["distance_m"]),
        "success": bool(status["success"]),
        "collision": bool(status["collision"]),
        "timeout": bool(forced_timeout or status["timeout"]),
        "steps": int(state["steps"]),
        "path_length_m": float(state["path_length_m"]),
        "terminal_goal_distance_m": float(status["goal_distance"]),
        "mean_projection_correction_norm": float(
            state["projection_correction_sum"] / max(state["steps"], 1)
        ),
        "max_projection_correction_norm": float(
            state["projection_correction_max"]
        ),
        "mean_actual_speed_mps": float(
            state["actual_speed_sum"] / max(state["steps"], 1)
        ),
    }


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
            active[index] = {
                "episode_id": episode_counter,
                "distance_m": distance,
                "steps": 0,
                "path_length_m": 0.0,
                "previous_position": env.root_states[index, :2].detach().clone(),
                "projection_correction_sum": 0.0,
                "projection_correction_max": 0.0,
                "actual_speed_sum": 0.0,
            }
            episode_counter += 1
        held_actions = torch.zeros(env.num_envs, 2, device=env.device)
        held_projection = torch.zeros(env.num_envs, device=env.device)
        with torch.inference_mode():
            while len(records) < int(episodes):
                if env.common_step_counter % env.upper_level_command_interval_steps == 0:
                    actual = torch.stack(
                        (env.tracking_lin_vel[:, 0], env.tracking_ang_vel[:, 2]), dim=1
                    )
                    teacher = teacher_velocity_diagnostics(
                        env._goal_xy_robot(), actual, env.obstacle_clearance, teacher_cfg
                    )
                    held_actions[:, 0] = teacher["applied_command"][:, 0] / teacher_cfg.max_forward_speed
                    held_actions[:, 1] = teacher["applied_command"][:, 1] / teacher_cfg.max_yaw_rate
                    held_actions.clamp_(-1.0, 1.0)
                    held_projection = teacher["projection_correction_norm"].detach().clone()
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
                    state["projection_correction_sum"] += float(held_projection[index].item())
                    state["projection_correction_max"] = max(
                        state["projection_correction_max"], float(held_projection[index].item())
                    )
                    actual_speed = env.terminal_tracking_velocity[index, 0] if bool(done_mask[index].item()) else env.tracking_lin_vel[index, 0]
                    state["actual_speed_sum"] += float(actual_speed.item())
                    forced_timeout = state["steps"] >= max_steps and not bool(done_mask[index].item())
                    if not bool(done_mask[index].item()) and not forced_timeout:
                        continue
                    if bool(done_mask[index].item()):
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
                    active[index] = {
                        "episode_id": episode_counter,
                        "distance_m": distance,
                        "steps": 0,
                        "path_length_m": 0.0,
                        "previous_position": env.root_states[index, :2].detach().clone(),
                        "projection_correction_sum": 0.0,
                        "projection_correction_max": 0.0,
                        "actual_speed_sum": 0.0,
                    }
                    episode_counter += 1
        fields = list(records[0]) if records else ["distance_m", "success"]
        with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        summary = {
            "distance_m": float(distance),
            "episodes": len(records),
            "successes": sum(int(row["success"]) for row in records),
            "collisions": sum(int(row["collision"]) for row in records),
            "timeouts": sum(int(row["timeout"]) for row in records),
            "success_rate": sum(int(row["success"]) for row in records) / max(len(records), 1),
            "mean_projection_correction_norm": sum(row["mean_projection_correction_norm"] for row in records) / max(len(records), 1),
            "teacher_config": teacher_cfg.__dict__,
            "max_steps": max_steps,
            "seed": int(seed),
            "wall_clock_seconds": time.monotonic() - started,
            "depth_backend": env.depth_backend_actual,
        }
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
    (root / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
