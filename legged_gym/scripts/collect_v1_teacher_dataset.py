"""Collect ordered V1 teacher labels from real Isaac Gym IMAGE_DEPTH."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import isaacgym  # noqa: F401 - must precede torch
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.navigation.v1_teacher_dataset import TeacherSequenceWriter
from legged_gym.navigation.phase_d_contracts import require_isaacgym_depth
from legged_gym.navigation.v1_velocity_teacher import (
    V1VelocityTeacherConfig,
    teacher_velocity_diagnostics,
)
from legged_gym.scripts.eval_v1_velocity_teacher import (
    _assign_fixed_goal,
    _close_environment,
    _teacher_obstacle_distance,
)
from legged_gym.utils import get_args, task_registry


def _parse_framework_args(remaining):
    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + list(remaining)
    try:
        return get_args()
    finally:
        os.sys.argv = original


def _row(
    env,
    index,
    episode_id,
    step_id,
    teacher,
    actual,
    done=False,
    success=False,
    collision=False,
    goal_distance=None,
):
    """Snapshot one macro decision before the next V62 transition."""
    return {
        "episode_id": episode_id,
        "step_id": step_id,
        "depth": env.depth_observation[index].detach().clone(),
        "depth_raw": (
            env._last_depth_raw[index].detach().clone()
            if getattr(env, "_last_depth_raw", None) is not None
            else env.depth_observation[index].detach().clone()
        ),
        "goal_xy_robot": env._goal_xy_robot()[index].detach().clone(),
        "proprioception": env._proprioception()[index].detach().clone(),
        "previous_command": env.previous_velocity_command[index].detach().clone(),
        "previous_actual_velocity": env.previous_actual_velocity[index].detach().clone(),
        "teacher_command": teacher["applied_command"][index].detach().clone(),
        "actual_velocity": actual[index].detach().clone(),
        "governor_command": env.applied_feasible_command[index].detach().clone(),
        "projection_command": teacher["applied_command"][index].detach().clone(),
        "timestamp_s": torch.tensor(float(env.common_step_counter) * float(env.dt)),
        "robot_pose": env.root_states[index, :7].detach().clone(),
        "global_goal_distance": torch.tensor(float(env.goal_dist[index].item() if goal_distance is None else goal_distance)),
        "waypoint": env._goal_xy_robot()[index].detach().clone(),
        "remaining_path": torch.tensor(0.0),
        "teacher_raw_command": teacher["raw_command"][index].detach().clone(),
        "teacher_projected_command": teacher["applied_command"][index].detach().clone(),
        "applied_feasible_command": env.applied_feasible_command[index].detach().clone(),
        "transition_state": env.transition_state[index].detach().clone(),
        "transition_active": bool(env.transition_active[index].item()),
        "failure_reason": "SUCCESS" if success else ("COLLISION" if collision else ("TIMEOUT" if done else "UNKNOWN")),
        "done": bool(done),
        "success": bool(success),
        "collision": bool(collision),
        "goal_distance": torch.tensor(
            float(env.goal_dist[index].item() if goal_distance is None else goal_distance)
        ),
    }


def collect_distance(
    env,
    writer,
    distance,
    episodes,
    episode_counter,
    teacher_cfg,
    max_steps,
):
    # The environment buffers are updated by the real-camera step under
    # inference mode.  Keep reset in the same context when reusing the env
    # between distance buckets; otherwise torch rejects the in-place buffer
    # update as an inference-tensor write from normal mode.
    with torch.inference_mode():
        env.reset()
    _assign_fixed_goal(env, 0, distance)
    with torch.inference_mode():
        env.compute_observations()
    require_isaacgym_depth(
        getattr(env, "depth_backend_requested", None),
        getattr(env, "depth_backend_actual", None),
    )
    current_episode = int(episode_counter)
    step_id = 0
    primitive_steps = 0
    pending = None
    completed = 0
    held_actions = torch.zeros(1, 2, device=env.device)
    with torch.inference_mode():
        while completed < int(episodes):
            if env.common_step_counter % env.upper_level_command_interval_steps == 0:
                if pending is not None:
                    writer.append(pending)
                actual = torch.stack(
                    (env.tracking_lin_vel[:, 0], env.tracking_ang_vel[:, 2]), dim=1
                )
                clearance = _teacher_obstacle_distance(env)
                teacher = teacher_velocity_diagnostics(
                    env._goal_xy_robot(), actual, clearance, teacher_cfg
                )
                held_actions[:, 0] = (
                    teacher["applied_command"][:, 0] / teacher_cfg.max_forward_speed
                )
                held_actions[:, 1] = (
                    teacher["applied_command"][:, 1] / teacher_cfg.max_yaw_rate
                )
                held_actions.clamp_(-1.0, 1.0)
                pending = _row(
                    env,
                    0,
                    current_episode,
                    step_id,
                    teacher,
                    actual,
                )
                step_id += 1
            _, _, _, dones, _ = env.step(held_actions)
            primitive_steps += 1
            is_done = bool(dones.flatten()[0].item())
            forced_timeout = primitive_steps >= max_steps and not is_done
            if not is_done and not forced_timeout:
                continue
            if pending is None:
                raise RuntimeError("episode terminated before a macro sample was captured")
            if is_done:
                terminal_success = bool(env.terminal_success[0].item())
                terminal_collision = bool(env.terminal_collision[0].item())
                pending.update(
                    {
                        "done": True,
                        "success": terminal_success,
                        "collision": terminal_collision,
                        "failure_reason": "SUCCESS" if terminal_success else ("COLLISION" if terminal_collision else "TIMEOUT"),
                        "goal_distance": torch.tensor(
                            float(env.terminal_goal_distance[0].item())
                        ),
                    }
                )
            else:
                pending.update(
                    {
                        "done": True,
                        "success": False,
                        "collision": False,
                        "failure_reason": "TIMEOUT",
                        "goal_distance": torch.tensor(float(env.goal_dist[0].item())),
                    }
                )
            writer.append(pending)
            pending = None
            completed += 1
            if completed >= int(episodes):
                break
            if forced_timeout:
                env.reset_idx(torch.as_tensor([0], device=env.device, dtype=torch.long))
            _assign_fixed_goal(env, 0, distance)
            current_episode += 1
            step_id = 0
            primitive_steps = 0
    return current_episode + 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", nargs="+", type=float, default=[1.0, 1.5, 2.0, 2.5])
    parser.add_argument("--episodes-per-distance", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
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
    env_cfg.commands.v1_goal_curriculum_enabled = False
    env_cfg.commands.v1_performance_curriculum_enabled = False
    env_cfg.commands.goal_bearing = (0.0, 0.0)
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    formal_steps = int(round(45.0 / (env_cfg.sim.dt * env_cfg.control.decimation)))
    teacher_cfg = V1VelocityTeacherConfig(
        max_forward_speed=float(env_cfg.commands.max_forward_speed),
        max_yaw_rate=float(env_cfg.commands.max_yaw_rate),
        minimum_turn_radius=float(env_cfg.commands.minimum_turn_radius),
        feasible_envelope_fraction=float(env_cfg.commands.feasible_envelope_fraction),
        goal_radius=float(env_cfg.commands.goal_radius),
    )
    writer = TeacherSequenceWriter(sequence_length=stage_args.sequence_length)
    env = None
    episode_counter = 0
    try:
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
        for distance in stage_args.distances:
            episode_counter = collect_distance(
                env,
                writer,
                float(distance),
                stage_args.episodes_per_distance,
                episode_counter,
                teacher_cfg,
                formal_steps,
            )
    finally:
        _close_environment(env)
    output = Path(stage_args.output).resolve()
    metadata = {
        "depth_backend_requested": "isaacgym",
        "depth_backend_actual": "isaacgym",
        "depth_representation": "normalized IMAGE_DEPTH, far-is-open",
        "distances_m": [float(value) for value in stage_args.distances],
        "episodes_per_distance": int(stage_args.episodes_per_distance),
        "seed": int(stage_args.seed),
        "sequence_length": int(stage_args.sequence_length),
        "teacher_config": teacher_cfg.__dict__,
    }
    writer.save(output, metadata=metadata)
    print(json.dumps({"output": str(output), "metadata": metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
