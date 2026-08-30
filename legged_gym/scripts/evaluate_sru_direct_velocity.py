"""Evaluate a direct SRU velocity checkpoint on point-to-point goals."""

import argparse
import json
import os
import sys

import isaacgym  # noqa: F401 - must precede torch
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.navigation.direct_velocity_curriculum import configure_direct_velocity_stage
from legged_gym.utils import get_args, task_registry


def _bool(value):
    return bool(value.item()) if hasattr(value, "item") else bool(value)


def evaluate(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", choices=("S1", "S2", "S2B"), default="S1")
    parser.add_argument("--episodes", type=int, default=20)
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
    env_cfg.env.num_envs = 1
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
    completed = 0
    episode_steps = 0

    print(
        "Evaluation: stage={} episodes={} checkpoint={}".format(
            stage_args.stage, stage_args.episodes, stage_args.checkpoint
        ),
        flush=True,
    )
    with torch.no_grad():
        while completed < stage_args.episodes:
            actions = policy(obs)
            obs, _, _, dones, _ = env.step(actions)
            episode_steps += 1
            done = _bool(dones[0])
            forced_timeout = episode_steps >= max_steps
            if done or forced_timeout:
                success = _bool(env.success_buf[0]) and not forced_timeout
                collision = _bool(env.step_collision_buf[0]) and not success
                timeout = (_bool(env.time_out_buf[0]) or forced_timeout) and not collision
                key = "success" if success else "collision" if collision else "timeout" if timeout else "other_failure"
                counts[key] += 1
                lengths.append(episode_steps)
                terminal_distances.append(float(env.terminal_goal_distance[0].item()))
                completed += 1
                print(
                    "episode {:>3}: success={} collision={} timeout={} final_dist={:.3f} steps={}".format(
                        completed, int(success), int(collision), int(timeout),
                        terminal_distances[-1], episode_steps
                    ),
                    flush=True,
                )
                episode_steps = 0
                if forced_timeout and not done:
                    env.reset_idx(torch.tensor([0], device=env.device, dtype=torch.long))
                    obs = env.get_observations()

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
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if stage_args.output:
        with open(stage_args.output, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


if __name__ == "__main__":
    evaluate()
