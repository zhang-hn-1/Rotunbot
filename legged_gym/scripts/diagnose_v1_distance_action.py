"""Deterministic V1 parent-policy distance OOD and clipped-goal diagnostic."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import isaacgym  # noqa: F401 - must precede torch
import torch
from isaacgym import gymtorch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import project_velocity_commands
from legged_gym.navigation.direct_velocity import normalized_action_to_velocity_command
from legged_gym.navigation.v1_distance_diagnostics import (
    SCAN_FIELDS,
    causal_pair,
    distance_grid,
    first_zero_crossing,
    scan_row,
)
from legged_gym.utils import get_args, task_registry


def _framework_args(remaining):
    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + list(remaining)
    try:
        return get_args()
    finally:
        os.sys.argv = original


def _raw_command(actions, cfg):
    command = actions.clamp(-1.0, 1.0).clone()
    command[:, 0] *= float(cfg.commands.max_forward_speed)
    command[:, 1] *= float(cfg.commands.max_yaw_rate)
    return command


def _project(command, env):
    return project_velocity_commands(
        command,
        env.cfg.commands.max_forward_speed,
        env.cfg.commands.max_yaw_rate,
        env.cfg.commands.minimum_turn_radius,
        env.cfg.commands.feasible_envelope_fraction,
        stationary_threshold=env.cfg.rewards.stationary_command_threshold,
        preserve_curvature_when_saturating=bool(
            getattr(env.cfg.commands, "preserve_curvature_when_saturating", False)
        ),
        curvature_fraction_breakpoints=getattr(
            env.cfg.commands, "stable_curvature_fraction_breakpoints", None
        ),
        curvature_max_speed_values=getattr(
            env.cfg.commands, "stable_curvature_max_speed_values", None
        ),
    )


def _set_fixed_state(env):
    env.root_states[0, :2] = env.env_origins[0, :2]
    env.root_states[0, 3:7] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device=env.device
    )
    env.root_states[0, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env._all_root_states)
    )


def _set_goal_and_observe(env, distance):
    goal = env.env_origins[0, :2].clone()
    goal[0] += float(distance)
    env.global_goal_xy_world[0] = goal
    env.goal_dist[0] = float(distance)
    env.previous_goal_distance[0] = float(distance)
    env.previous_velocity_command[0] = 0.0
    env.last_velocity_command[0] = 0.0
    env.goal_recovery_active[0] = False
    env.compute_observations()
    return env.get_observations()


def _causal_observation(env, physical_distance, visible_distance):
    observations = _set_goal_and_observe(env, physical_distance).clone()
    observations[0, 12] = float(visible_distance) / float(
        env.cfg.commands.maximum_goal_distance
    )
    return observations


def run(checkpoint, output, plot_output, causal_output=None, framework_args=()):
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    plot_output = Path(plot_output).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    args = _framework_args(framework_args)
    args.task = "rotunbot_sru_visual_corridor_v1"
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.camera.add_noise = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.v1_goal_curriculum_enabled = False
    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    try:
        runner, _ = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
        )
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        env.reset()
        _set_fixed_state(env)
        distances = distance_grid(0.50, 6.00, 0.25)
        rows = []
        with torch.no_grad():
            for distance in distances:
                observations = _set_goal_and_observe(env, distance)
                action = policy(observations)
                mapped = _raw_command(action, env.cfg)
                projected = normalized_action_to_velocity_command(
                    action,
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
                        env.cfg.commands, "stable_curvature_fraction_breakpoints", None
                    ),
                    curvature_max_speed_values=getattr(
                        env.cfg.commands, "stable_curvature_max_speed_values", None
                    ),
                )
                # Keep an explicit independent projection call in the artifact
                # path so the raw-to-projected boundary remains auditable.
                projected = _project(mapped, env)
                rows.append(
                    scan_row(
                        distance,
                        distance / float(env.cfg.commands.maximum_goal_distance),
                        action[0].cpu().tolist(),
                        mapped[0].cpu().tolist(),
                        projected[0].cpu().tolist(),
                    )
                )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCAN_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        plot_output.parent.mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        distance_values = [row["distance_m"] for row in rows]
        figure, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        axes[0].plot(distance_values, [row["raw_policy_mean_a_v"] for row in rows], label="raw a_v")
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].legend()
        axes[1].plot(distance_values, [row["mapped_v_cmd"] for row in rows], label="mapped v_cmd")
        axes[1].plot(distance_values, [row["projected_v_cmd"] for row in rows], label="projected v_cmd")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].legend()
        axes[2].plot(distance_values, [row["mapped_w_cmd"] for row in rows], label="mapped w_cmd")
        axes[2].plot(distance_values, [row["projected_w_cmd"] for row in rows], label="projected w_cmd")
        axes[2].axhline(0.0, color="black", linewidth=0.8)
        axes[2].set_xlabel("goal distance (m)")
        axes[2].legend()
        figure.tight_layout()
        figure.savefig(str(plot_output), dpi=140)
        plt.close(figure)
        summary = {
            "checkpoint": str(checkpoint),
            "deterministic": True,
            "physical_goal_distance_m": "varied only in actor observation",
            "distance_count": len(rows),
            "first_raw_a_v_zero_crossing_m": first_zero_crossing(
                [(row["distance_m"], row["raw_policy_mean_a_v"]) for row in rows]
            ),
            "first_mapped_v_cmd_zero_crossing_m": first_zero_crossing(
                [(row["distance_m"], row["mapped_v_cmd"]) for row in rows]
            ),
            "csv": str(output),
            "plot": str(plot_output),
        }
        if causal_output is not None:
            causal_output = Path(causal_output).expanduser().resolve()
            causal_rows = []
            with torch.no_grad():
                for visible_distance in (6.0, 2.0):
                    observation = _causal_observation(env, 6.0, visible_distance)
                    action = policy(observation)
                    mapped = _raw_command(action, env.cfg)
                    causal_rows.append(
                        causal_pair(
                            6.0,
                            visible_distance,
                            action[0].cpu().tolist(),
                            mapped[0].cpu().tolist(),
                        )
                    )
            causal_output.parent.mkdir(parents=True, exist_ok=True)
            with causal_output.open("w", newline="") as handle:
                fields = tuple(causal_rows[0])
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(causal_rows)
            summary["causal_output"] = str(causal_output)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary
    finally:
        if hasattr(env, "close"):
            env.close()
        else:
            env.gym.destroy_sim(env.sim)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="logs/diagnostics/v1_distance_action_scan.csv")
    parser.add_argument("--plot", default="logs/diagnostics/v1_distance_action_scan.png")
    parser.add_argument("--causal-output", default="logs/diagnostics/v1_clipped_goal_causal.csv")
    known, remaining = parser.parse_known_args(argv)
    return run(known.checkpoint, known.output, known.plot, known.causal_output, remaining)


if __name__ == "__main__":
    main()
