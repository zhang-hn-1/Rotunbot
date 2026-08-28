"""Probe a trained depth-local actor for sensitivity to the lateral goal."""

import argparse
import json
from pathlib import Path
import sys

import isaacgym  # noqa: F401 - must precede torch in Isaac Gym Preview 4
import numpy as np

if not hasattr(np, "float"):
    np.float = float

import torch

import legged_gym.envs  # noqa: F401 - registration side effects
from legged_gym.envs.rotunbot.maze.rotunbot_maze_local_depth import build_depth_local_observation
from legged_gym.scripts.depth_local_diagnostics import group_policy_metrics, policy_gy_metrics
from legged_gym.utils import get_args, task_registry


TASK = "rotunbot_maze_local_depth"
GY_VALUES = tuple(round(-0.6 + 0.1 * index, 1) for index in range(13))


def _make_runner(checkpoint):
    old_argv = sys.argv
    sys.argv = [old_argv[0], "--headless"]
    try:
        args = get_args()
    finally:
        sys.argv = old_argv
    args.task = TASK
    env_cfg, train_cfg = task_registry.get_cfgs(TASK)
    env_cfg.env.num_envs = 1
    env_cfg.maze.enabled = False
    env_cfg.maze.scene_mode = "none"
    env_cfg.camera.depth_backend = "fallback"
    env_cfg.camera.add_noise = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.random_start_yaw = False
    env_cfg.commands.resample_commands = False
    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(name=TASK, args=args, env_cfg=env_cfg)
    env.data_print = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=TASK, args=args, train_cfg=train_cfg, log_root=None
    )
    runner.load(str(checkpoint))
    return env, runner.get_inference_policy(device=env.device)


def _canonical_observation(env, gx, gy):
    zeros3 = torch.zeros((1, 3), dtype=torch.float32, device=env.device)
    zeros2 = torch.zeros((1, 2), dtype=torch.float32, device=env.device)
    gravity = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=env.device)
    goal = torch.tensor([[float(gx), float(gy)]], dtype=torch.float32, device=env.device)
    depth = torch.ones((1, 8, 32), dtype=torch.float32, device=env.device)
    return build_depth_local_observation(
        gravity, zeros3, zeros3, torch.zeros((1, 1), device=env.device),
        zeros2, goal, zeros2, depth,
    )


def run(checkpoint, output, physical_sign=1):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    env, policy = _make_runner(checkpoint)
    records = []
    try:
        with torch.no_grad():
            for gx in (1.0,):
                for gy in GY_VALUES:
                    action = policy(_canonical_observation(env, gx, gy))
                    records.append({
                        "sweep": "gx_1.0_primary", "gx": gx, "gy": gy,
                        "actor_mean_a0": float(action[0, 0].item()),
                        "actor_mean_a1": float(action[0, 1].item()),
                    })
            for gx in (0.5, 1.0, 1.5):
                for gy in GY_VALUES:
                    action = policy(_canonical_observation(env, gx, gy))
                    records.append({
                        "sweep": "gx_secondary", "gx": gx, "gy": gy,
                        "actor_mean_a0": float(action[0, 0].item()),
                        "actor_mean_a1": float(action[0, 1].item()),
                    })
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)
    report = {
        "task": TASK,
        "experiment": "B_policy_gy_response",
        "checkpoint": str(checkpoint),
        "deterministic": True,
        "physical_action1_sign_for_positive_gy": int(physical_sign),
        "observation": {
            "projected_gravity": [0.0, 0.0, -1.0], "lin_vel": [0.0, 0.0, 0.0],
            "ang_vel": [0.0, 0.0, 0.0], "joint2_position": 0.0,
            "dof_velocity": [0.0, 0.0], "previous_action": [0.0, 0.0],
            "depth": "all ones", "goal_normalization": "local_goal / 8.0",
        },
        "records": records,
        "metrics": policy_gy_metrics(records, physical_sign=physical_sign),
        "metrics_by_gx": group_policy_metrics(records, physical_sign=physical_sign),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path = output.with_name("policy_gy_summary.md")
    metrics = report["metrics"]
    lines = [
        "# Depth-local policy lateral-goal diagnostic (Experiment B)", "",
        f"Checkpoint: `{checkpoint}`", "",
        "Deterministic `policy.act_inference` probe with canonical proprioception and all-ones depth.", "",
        "| gx | gy | a0 mean | a1 mean |", "|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['gx']:.1f} | {row['gy']:.1f} | {row['actor_mean_a0']:.6f} | {row['actor_mean_a1']:.6f} |"
        )
    lines.extend([
        "", "Metrics (all 52 probes):", "",
        f"- a1 response span: `{metrics['a1_response_span']:.6f}`",
        f"- a0 response span: `{metrics['a0_response_span']:.6f}`",
        f"- sign agreement for |gy|≥0.2: `{metrics['sign_agreement_rate']:.6f}`",
        f"- Pearson(gy, a1): `{metrics['pearson_gy_a1']:.6f}`",
        f"- Spearman(gy, a1): `{metrics['spearman_gy_a1']:.6f}`",
        f"- symmetry error mean|a1(gy)+a1(-gy)|: `{metrics['symmetry_error']:.6f}`",
    ])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Experiment B metrics:", json.dumps(metrics, sort_keys=True))
    print("Saved:", output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--physical-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--output", default="logs/depth_local_diagnostics/policy_gy_sweep.json")
    args = parser.parse_args(argv)
    run(args.checkpoint, Path(args.output), physical_sign=args.physical_sign)


if __name__ == "__main__":
    main()
