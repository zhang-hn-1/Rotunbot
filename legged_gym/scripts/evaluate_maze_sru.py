#!/usr/bin/env python
"""Evaluate an SRU maze policy on the procedural maze (point-to-point).

Usage (GPU machine):

    python legged_gym/scripts/evaluate_maze_sru.py \
        --run-dir logs/rotunbot_maze_sru/<run> \
        --checkpoint -1 --episodes 40 --seed 0

Metrics per episode: success (reached target inside stop radius and stopped),
collision, timeout, episode time, path length, straight-line distance,
time-to-goal.  Aggregate: success rate, mean path length, mean time.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

np.float = float

import distutils
import distutils.version
distutils.version = distutils.version

import isaacgym
import torch
import isaacgym.torch_utils as torch_utils

# RTX-40 eager shims (same as train.py).
def _quat_rotate_inverse(q, v):
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(
        q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)
    ).squeeze(-1) * 2.0
    return a - b + c

def _quat_apply(a, b):
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2.0
    return b + a[:, 3:] * t + xyz.cross(t, dim=-1)

def _normalize(x, eps=1.0e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

def _torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower

torch_utils.quat_rotate_inverse = _quat_rotate_inverse
torch_utils.quat_apply = _quat_apply
torch_utils.normalize = _normalize
torch_utils.torch_rand_float = _torch_rand_float

from legged_gym.envs import *  # noqa
from legged_gym.utils import get_args, task_registry

TASK = "rotunbot_maze_sru"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.run_dir = str(Path(args.run_dir).resolve())

    sys.argv = [
        sys.argv[0], "--headless", "--sim_device=cuda:0", "--rl_device=cuda:0",
    ]
    gym_args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK)
    env_cfg.seed = int(args.seed)
    train_cfg.seed = int(args.seed)
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env, _ = task_registry.make_env(name=TASK, args=gym_args, env_cfg=env_cfg)

    gym_args.task = TASK
    gym_args.load_run = str(args.run_dir)
    gym_args.checkpoint = int(args.checkpoint)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = str(args.run_dir)
    train_cfg.runner.checkpoint = int(args.checkpoint)
    runner, _ = task_registry.make_alg_runner(
        env=env, args=gym_args, train_cfg=train_cfg,
        log_root=str(Path(args.run_dir).parent),
    )
    policy = runner.get_inference_policy(device=env.device)
    print(f"policy loaded from {args.run_dir} checkpoint {args.checkpoint}")

    results = []
    with torch.no_grad():
        for ep in range(int(args.episodes)):
            env.reset()
            obs = env.get_observations()
            target = env.commands[0, :2].cpu().numpy()
            start = env.root_states[0, :2].cpu().numpy()
            straight = float(np.linalg.norm(target - start))
            path_len = 0.0
            prev_xy = start.copy()
            success = False
            collision = False
            timeout = False
            time_to_goal = None
            steps = 0
            while True:
                action = policy(obs)
                obs, _, _, dones, _ = env.step(action)
                steps += 1
                xy = env.root_states[0, :2].cpu().numpy()
                path_len += float(np.linalg.norm(xy - prev_xy))
                prev_xy = xy
                goal_dist = float(
                    torch.norm(env.commands[0, :2] - env.root_states[0, :2]).item()
                )
                speed = float(torch.norm(env.base_lin_vel[0]).item())
                if env.maze_collision_buf[0].item():
                    collision = True  # record only; episode continues (matches training)
                stop_radius = float(env.cfg.commands.stop_distance)
                if goal_dist <= stop_radius and speed <= 0.1:
                    success = True
                    time_to_goal = steps * 0.02
                    break
                if dones[0].item():
                    timeout = True
                    break
                if steps > 6000:  # 120 s hard cap (matches training horizon)
                    timeout = True
                    break
            results.append({
                "episode": ep,
                "success": success,
                "collision": collision,
                "timeout": timeout,
                "steps": steps,
                "time_s": round(steps * 0.02, 2),
                "path_length": round(path_len, 3),
                "straight_line": round(straight, 3),
                "time_to_goal_s": round(time_to_goal, 2) if time_to_goal else None,
            })
            print(
                f"ep {ep:02d} success={int(success)} collision={int(collision)} "
                f"timeout={int(timeout)} steps={steps} path={path_len:.1f}m "
                f"straight={straight:.1f}m"
            )

    n = len(results)
    sr = sum(r["success"] for r in results) / n
    mean_path = np.mean([r["path_length"] for r in results if r["success"]]) if sr else 0.0
    mean_time = np.mean([r["time_to_goal_s"] for r in results if r["time_to_goal_s"] is not None]) if sr else 0.0
    collisions = sum(r["collision"] for r in results)
    summary = {
        "episodes": n,
        "success_rate": round(sr, 4),
        "successes": int(sr * n),
        "collisions": int(collisions),
        "timeouts": int(sum(r["timeout"] for r in results)),
        "mean_path_length_success_m": round(float(mean_path), 3),
        "mean_time_to_goal_s": round(float(mean_time), 2),
    }
    print(json.dumps(summary, indent=2))
    out = Path(args.run_dir) / "maze_eval_summary.json"
    out.write_text(json.dumps({"summary": summary, "episodes": results}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
