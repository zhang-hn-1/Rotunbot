"""Print and record the reward terms actually used by the V0 local-depth task."""

import argparse
import json
import sys
from pathlib import Path


def audit(steps=20, report_path="logs/rotunbot_maze_local_depth/reward_audit.json"):
    import isaacgym  # noqa: F401 - must precede torch in Isaac Gym Preview 4
    import numpy as np

    if not hasattr(np, "float"):
        np.float = float
    import torch

    import legged_gym.envs  # noqa: F401 - registration side effects
    from legged_gym.utils import get_args, task_registry

    old_argv = sys.argv
    sys.argv = [old_argv[0], "--headless"]
    try:
        args = get_args()
    finally:
        sys.argv = old_argv
    args.task = "rotunbot_maze_local_depth"
    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.camera.depth_backend = "fallback"
    env_cfg.camera.add_noise = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.data_print = False
    original_functions = list(env.reward_functions)
    calls = {}

    def wrap(function, name, scale):
        def wrapped():
            value = function()
            value_tensor = torch.as_tensor(value, device=env.device)
            calls[name] = float((value_tensor.reshape(-1)[0] * scale).detach().cpu().item())
            return value
        return wrapped

    env.reward_functions = [
        wrap(function, name, env.reward_scales[name])
        for function, name in zip(original_functions, env.reward_names)
    ]
    obs, _ = env.reset()
    print("reward_names:", env.reward_names)
    print("reward_scales:", {name: float(scale) for name, scale in env.reward_scales.items()})
    records = []
    try:
        with torch.no_grad():
            for step in range(1, int(steps) + 1):
                action = torch.tensor(
                    [[0.35, 0.0] if step <= steps // 2 else [0.35, 0.25]],
                    dtype=torch.float32,
                    device=env.device,
                )
                calls.clear()
                obs, _, rewards, dones, _ = env.step(action)
                record = {
                    "step": step,
                    "goal_dist": float(torch.linalg.vector_norm(
                        env.global_goal_xy_world[0] - env.root_states[0, :2]
                    ).item()),
                    "active_local_goal_dist": float(torch.linalg.vector_norm(
                        env.active_local_goal_xy_robot[0]
                    ).item()),
                    "rew_close_to_target": calls.get("close_to_target", 0.0),
                    "rew_stop": calls.get("stop", 0.0),
                    "rew_balance": calls.get("balance", 0.0),
                    "rew_local_progress": calls.get("local_progress", 0.0),
                    "rew_local_reach": calls.get("local_reach", 0.0),
                    "total_reward": float(rewards[0].item()),
                    "reward_terms": dict(calls),
                    "done": bool(dones[0].item()),
                }
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)

    report = {
        "reward_names": env.reward_names,
        "reward_scales": {name: float(scale) for name, scale in env.reward_scales.items()},
        "steps": records,
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--report", default="logs/rotunbot_maze_local_depth/reward_audit.json")
    args = parser.parse_args()
    audit(args.steps, args.report)


if __name__ == "__main__":
    main()
