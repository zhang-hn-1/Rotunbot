"""Verify that the real V1 runner holds one action for one macro period."""

import argparse
import csv
import os
import sys
from pathlib import Path

import isaacgym  # noqa: F401 - must precede torch
import torch
from isaacgym import gymtorch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.navigation.direct_velocity import normalized_action_to_velocity_command
from legged_gym.navigation.high_level_action_timing import timing_row
from legged_gym.utils import get_args, task_registry


def _framework_args(remaining):
    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + list(remaining)
    try:
        return get_args()
    finally:
        os.sys.argv = original


def run(output, framework_args=()):
    args = _framework_args(framework_args)
    args.task = "rotunbot_sru_visual_corridor_v1"
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.camera.add_noise = False
    env_cfg.noise.add_noise = False
    env_cfg.commands.v1_goal_curriculum_enabled = False
    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    rows = []
    original_step = env.step
    repeat = None
    try:
        runner, _ = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
        )
        repeat = runner.action_repeat
        runner.alg.transition.values = torch.zeros(1, 1, device=env.device)
        env.reset()
        action = torch.tensor([[0.5, 0.2]], device=env.device)
        requested = normalized_action_to_velocity_command(
            action,
            env.cfg.commands.max_forward_speed,
            env.cfg.commands.max_yaw_rate,
            env.cfg.commands.minimum_turn_radius,
            env.cfg.commands.feasible_envelope_fraction,
        )
        primitive_counter = {"value": 0}

        def logged_step(step_actions):
            primitive_step = primitive_counter["value"] % repeat
            sample_id = primitive_counter["value"] // repeat
            result = original_step(step_actions)
            rows.append(
                timing_row(
                    sample_id,
                    primitive_step,
                    action[0].detach().cpu().tolist(),
                    requested[0].detach().cpu().tolist(),
                    env.applied_feasible_command[0].detach().cpu().tolist(),
                )
            )
            primitive_counter["value"] += 1
            return result

        env.step = logged_step
        runner._step_high_level(action)
        runner._step_high_level(action)
    finally:
        env.step = original_step
        if hasattr(env, "close"):
            env.close()
        else:
            env.gym.destroy_sim(env.sim)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("repeat=%d rows=%d output=%s" % (repeat, len(rows), output))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="logs/diagnostics/high_level_action_timing.csv")
    known, remaining = parser.parse_known_args(argv)
    return run(known.output, remaining)


if __name__ == "__main__":
    main()
