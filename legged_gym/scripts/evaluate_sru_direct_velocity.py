"""Evaluate a direct SRU velocity checkpoint on point-to-point goals."""

import argparse
import json
import os
import sys

import isaacgym  # noqa: F401 - must precede torch
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.navigation.direct_velocity_curriculum import configure_direct_velocity_stage
from legged_gym.navigation.direct_velocity import normalized_action_to_velocity_command
from legged_gym.utils import get_args, task_registry


def _bool(value):
    return bool(value.item()) if hasattr(value, "item") else bool(value)


def evaluate(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", choices=("S1", "S2", "S2B"), default="S1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output", default=None)
    stage_args, remaining = parser.parse_known_args(argv)
    if stage_args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + remaining
    try:
        args = get_args()
    finally:
        os.sys.argv = original

    args.task = "rotunbot_sru_direct_velocity"
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    configure_direct_velocity_stage(env_cfg, stage_args.stage)
    env_cfg.env.num_envs = max(1, int(stage_args.num_envs))
    env_cfg.noise.add_noise = False
    env_cfg.camera.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.commands.random_start_yaw = False

    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    runner.load(stage_args.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.reset()
    max_steps = stage_args.max_steps or int(env.max_episode_length)
    counts = {"success": 0, "collision": 0, "timeout": 0, "other_failure": 0}
    lengths = []
    terminal_distances = []
    initial_goal_distances = []
    initial_goal_bearings = []
    success_flags = []
    episode_alignment_sum = torch.zeros(env.num_envs, device=env.device)
    episode_forward_sum = torch.zeros(env.num_envs, device=env.device)
    episode_yaw_sum = torch.zeros(env.num_envs, device=env.device)
    episode_command_count = torch.zeros(env.num_envs, device=env.device)
    failed_alignment = []
    failed_forward_command = []
    failed_abs_yaw_command = []
    episode_bearing_sum = torch.zeros(env.num_envs, device=env.device)
    episode_bearing_count = torch.zeros(env.num_envs, device=env.device)
    failed_mean_abs_bearing = []
    failed_final_abs_bearing = []
    completed = 0
    episode_steps = [0 for _ in range(env.num_envs)]
    episode_goal_distances = env.goal_dist.detach().clone()
    episode_goal_bearings = torch.atan2(
        env._goal_xy_robot()[:, 1], env._goal_xy_robot()[:, 0]
    ).detach().clone()

    print(
        "Evaluation: stage={} episodes={} checkpoint={}".format(
            stage_args.stage, stage_args.episodes, stage_args.checkpoint
        ),
        flush=True,
    )
    with torch.no_grad():
        while completed < stage_args.episodes:
            actions = policy(obs)
            commands = normalized_action_to_velocity_command(
                actions,
                env.cfg.commands.max_forward_speed,
                env.cfg.commands.max_yaw_rate,
                env.cfg.commands.minimum_turn_radius,
                env.cfg.commands.feasible_envelope_fraction,
                preserve_curvature_when_saturating=bool(
                    getattr(
                        env.cfg.commands,
                        "preserve_curvature_when_saturating",
                        False,
                    )
                ),
                curvature_fraction_breakpoints=getattr(
                    env.cfg.commands,
                    "stable_curvature_fraction_breakpoints",
                    None,
                ),
                curvature_max_speed_values=getattr(
                    env.cfg.commands,
                    "stable_curvature_max_speed_values",
                    None,
                ),
            )
            goal_xy = env._goal_xy_robot()
            goal_bearing = torch.atan2(goal_xy[:, 1], goal_xy[:, 0])
            bearing_sign = torch.sign(goal_bearing)
            turning = torch.abs(goal_bearing) >= 0.05
            aligned = (~turning) | (bearing_sign * commands[:, 1] >= 0.0)
            episode_alignment_sum += aligned.float()
            episode_forward_sum += commands[:, 0]
            episode_yaw_sum += torch.abs(commands[:, 1])
            episode_command_count += 1.0
            episode_bearing_sum += torch.abs(goal_bearing)
            episode_bearing_count += 1.0
            obs, _, _, dones, _ = env.step(actions)
            for index in range(env.num_envs):
                episode_steps[index] += 1
            done_mask = dones.flatten().bool()
            forced_mask = torch.as_tensor(
                [steps >= max_steps for steps in episode_steps],
                device=env.device, dtype=torch.bool,
            ) & ~done_mask
            terminal_mask = done_mask | forced_mask
            manual_reset_ids = []
            for index in terminal_mask.nonzero(as_tuple=False).flatten().tolist():
                if completed >= stage_args.episodes:
                    break
                forced_timeout = bool(forced_mask[index].item())
                success = _bool(env.success_buf[index]) and not forced_timeout
                collision = _bool(env.step_collision_buf[index]) and not success
                timeout = (_bool(env.time_out_buf[index]) or forced_timeout) and not collision
                key = "success" if success else "collision" if collision else "timeout" if timeout else "other_failure"
                counts[key] += 1
                success_flags.append(bool(success))
                lengths.append(episode_steps[index])
                terminal_distances.append(float(env.terminal_goal_distance[index].item()))
                initial_goal_distances.append(float(episode_goal_distances[index].item()))
                initial_goal_bearings.append(float(episode_goal_bearings[index].item()))
                command_count = max(float(episode_command_count[index].item()), 1.0)
                if not success:
                    failed_alignment.append(
                        float(episode_alignment_sum[index].item()) / command_count
                    )
                    failed_forward_command.append(
                        float(episode_forward_sum[index].item()) / command_count
                    )
                    failed_abs_yaw_command.append(
                        float(episode_yaw_sum[index].item()) / command_count
                    )
                    failed_mean_abs_bearing.append(
                        float(episode_bearing_sum[index].item())
                        / max(float(episode_bearing_count[index].item()), 1.0)
                        * 180.0 / 3.141592653589793
                    )
                    failed_final_abs_bearing.append(
                        abs(float(goal_bearing[index].item()))
                        * 180.0 / 3.141592653589793
                    )
                completed += 1
                print(
                    "episode {:>3}: success={} collision={} timeout={} final_dist={:.3f} steps={}".format(
                        completed, int(success), int(collision), int(timeout),
                        terminal_distances[-1], episode_steps[index]
                    ),
                    flush=True,
                )
                episode_steps[index] = 0
                episode_alignment_sum[index] = 0.0
                episode_forward_sum[index] = 0.0
                episode_yaw_sum[index] = 0.0
                episode_command_count[index] = 0.0
                episode_bearing_sum[index] = 0.0
                episode_bearing_count[index] = 0.0
                if forced_timeout:
                    manual_reset_ids.append(index)
                if not forced_timeout:
                    episode_goal_distances[index] = env.goal_dist[index]
                    episode_goal_bearings[index] = torch.atan2(
                        env._goal_xy_robot()[index, 1], env._goal_xy_robot()[index, 0]
                    )
            if manual_reset_ids:
                env.reset_idx(torch.as_tensor(manual_reset_ids, device=env.device, dtype=torch.long))
                obs = env.get_observations()
                new_bearings = torch.atan2(
                    env._goal_xy_robot()[:, 1], env._goal_xy_robot()[:, 0]
                )
                for index in manual_reset_ids:
                    episode_goal_distances[index] = env.goal_dist[index]
                    episode_goal_bearings[index] = new_bearings[index]

    summary = {
        "stage": stage_args.stage,
        "checkpoint": os.path.abspath(stage_args.checkpoint),
        "episodes": stage_args.episodes,
        **counts,
        "success_rate": counts["success"] / stage_args.episodes,
        "collision_rate": counts["collision"] / stage_args.episodes,
        "timeout_rate": counts["timeout"] / stage_args.episodes,
        "other_failure_rate": counts["other_failure"] / stage_args.episodes,
        "mean_steps": sum(lengths) / len(lengths),
        "mean_terminal_distance": sum(terminal_distances) / len(terminal_distances),
        "mean_initial_goal_distance": sum(initial_goal_distances) / len(initial_goal_distances),
        "mean_initial_abs_bearing_deg": sum(abs(value) for value in initial_goal_bearings) / len(initial_goal_bearings) * 180.0 / 3.141592653589793,
        "failed_initial_goal_distances": [
            round(distance, 4) for distance, success in zip(initial_goal_distances, success_flags) if not success
        ],
        "failed_initial_abs_bearings_deg": [
            round(abs(bearing) * 180.0 / 3.141592653589793, 3)
            for bearing, success in zip(initial_goal_bearings, success_flags) if not success
        ],
        "failed_mean_command_alignment": [round(value, 4) for value in failed_alignment],
        "failed_mean_forward_command": [round(value, 4) for value in failed_forward_command],
        "failed_mean_abs_yaw_command": [round(value, 4) for value in failed_abs_yaw_command],
        "failed_mean_abs_bearing_deg": [round(value, 3) for value in failed_mean_abs_bearing],
        "failed_final_abs_bearing_deg": [round(value, 3) for value in failed_final_abs_bearing],
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if stage_args.output:
        with open(stage_args.output, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


if __name__ == "__main__":
    evaluate()
