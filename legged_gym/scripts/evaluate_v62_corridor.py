"""Evaluate frozen V62 spatial motion in deterministic corridor scenarios."""

import argparse
import math
import os
from pathlib import Path

import isaacgym  # noqa: F401 - must precede torch
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import project_velocity_commands
from legged_gym.navigation.v62_corridor_task import (
    make_wall_segments,
    register_v62_corridor_eval_task,
)
from legged_gym.navigation.corridor_artifacts import EpisodeLogger, GateResult
from legged_gym.navigation.corridor_plotting import plot_corridor_artifacts
from legged_gym.navigation.corridor_scenarios import (
    make_double_turn_scenario,
    make_l_scenario,
    make_straight_scenario,
)
from legged_gym.navigation.v62_corridor_controller import PoseBasedCorridorController
from legged_gym.utils import get_args, task_registry


DEFAULT_CHECKPOINT = (
    "/home/jason/Rotunbot_SRU50_V62_SafeYaw_Final_Verified_20260829/model/model_150.pt"
)
LOW_LEVEL_HZ = 50.0
UPPER_COMMAND_HZ = 5.0
UPPER_HOLD_STEPS = 10
ROBOT_RADIUS_M = 0.40
CORRIDOR_TASK_NAME = register_v62_corridor_eval_task()


def _scenario_for_family(family, seed):
    if family == "A0":
        return make_straight_scenario(2.0, 5.0, seed)
    if family == "A1":
        return make_l_scenario(2.0, 3.0, 2.0, seed)
    if family == "A2":
        return make_double_turn_scenario(
            2.0, 2.0, "left_right" if int(seed) % 2 == 0 else "right_left", seed
        )
    raise ValueError("family must be A0, A1 or A2")


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--corridor_family", choices=("A0", "A1", "A2"), default="A0")
    parser.add_argument("--corridor_episodes", type=int, default=None)
    parser.add_argument("--corridor_seed", type=int, default=20260829)
    parser.add_argument("--corridor_output_dir", type=str, required=True)
    parser.add_argument("--corridor_checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--corridor_max_steps", type=int, default=None)
    original = list(os.sys.argv)
    diagnostic, remaining = parser.parse_known_args()
    os.sys.argv = [original[0]] + remaining
    try:
        args = get_args()
    finally:
        os.sys.argv = original
    args.corridor_family = diagnostic.corridor_family
    args.corridor_episodes = diagnostic.corridor_episodes or {"A0": 20, "A1": 20, "A2": 30}[diagnostic.corridor_family]
    args.corridor_seed = diagnostic.corridor_seed
    args.corridor_output_dir = Path(diagnostic.corridor_output_dir).expanduser().resolve()
    args.corridor_checkpoint = Path(diagnostic.corridor_checkpoint).expanduser().resolve()
    args.corridor_max_steps = diagnostic.corridor_max_steps
    if args.corridor_episodes < 1 or not args.corridor_checkpoint.is_file():
        raise ValueError("positive episodes and an existing checkpoint are required")
    args.task = CORRIDOR_TASK_NAME
    args.num_envs = 1
    return args


def _distance_to_polyline(position, centerline):
    points = np.asarray(centerline, dtype=np.float64)
    start = points[:-1]
    delta = points[1:] - start
    denominator = np.sum(delta * delta, axis=1)
    denominator[denominator < 1.0e-12] = 1.0
    fraction = np.clip(np.sum((position - start) * delta, axis=1) / denominator, 0.0, 1.0)
    projections = start + fraction[:, None] * delta
    distances = np.linalg.norm(projections - position, axis=1)
    return float(np.min(distances))


def _configure_env(env_cfg, scenario):
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 240.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.commands.target_curriculum = False
    env_cfg.commands.resampling_time = 10000.0
    env_cfg.commands.upper_level_command_frequency_hz = UPPER_COMMAND_HZ
    env_cfg.corridor_wall_width_m = scenario.width_m
    env_cfg.corridor_wall_segments = make_wall_segments(scenario.centerline)


def _new_controller(env):
    cfg = env.cfg.commands
    return PoseBasedCorridorController(
        maximum_forward_speed=cfg.max_forward_speed,
        maximum_yaw_rate=cfg.max_yaw_rate,
        minimum_turn_radius=cfg.minimum_turn_radius,
        envelope_fraction=cfg.feasible_envelope_fraction,
    )


def run_corridor(args, scenario, episodes, output_dir, enforce_gate=True, max_steps=None):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _configure_env(env_cfg, scenario)
    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.data_print = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    runner.load(str(args.corridor_checkpoint))
    policy = runner.get_inference_policy(device=env.device)
    logger = EpisodeLogger(output_dir)
    controller = _new_controller(env)
    try:
        if max_steps is None:
            max_steps = int(math.ceil(max(240.0, scenario.path_length_m / 0.05) / env.dt))
        all_records = []
        for episode_id in range(1, episodes + 1):
            obs, _ = env.reset()
            controller.reset()
            position0 = env.root_states[0, :2].detach().cpu().numpy() - env.env_origins[0, :2].detach().cpu().numpy()
            controller._scenario_key = None
            target = scenario.goal_xy.copy()
            trajectory = []
            min_distance = float(np.linalg.norm(position0 - target))
            max_lateral_error = 0.0
            path_length = 0.0
            previous_position = position0.copy()
            previous_applied = np.zeros(2, dtype=np.float64)
            rate_violations = 0
            domain_violations = 0
            hidden_jumps = 0
            transition_active_steps = 0
            collision = False
            success = False
            timeout = False
            for step in range(max_steps):
                position = env.root_states[0, :2].detach().cpu().numpy() - env.env_origins[0, :2].detach().cpu().numpy()
                quat = env.base_quat[0]
                yaw = float(torch.atan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]), 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)).item())
                if step % UPPER_HOLD_STEPS == 0:
                    command = controller.update(position, yaw, scenario)
                    env.set_command_targets(torch.as_tensor(command, device=env.device))
                    env.compute_observations()
                applied = env.commands[0, :2].detach().cpu().numpy()
                actual = np.asarray((env.tracking_lin_vel[0, 0].item(), env.tracking_ang_vel[0, 2].item()))
                delta = np.abs(applied - previous_applied)
                v_limit = float(env.cfg.commands.maximum_linear_acceleration) * float(env.dt)
                w_limit = float(env.cfg.commands.maximum_yaw_acceleration) * float(env.dt)
                rate_bad = bool(delta[0] > v_limit + 1.0e-6 or delta[1] > w_limit + 1.0e-7)
                projected = project_velocity_commands(
                    torch.as_tensor(applied, device=env.device).reshape(1, 2),
                    env.cfg.commands.max_forward_speed,
                    env.cfg.commands.max_yaw_rate,
                    env.cfg.commands.minimum_turn_radius,
                    env.cfg.commands.feasible_envelope_fraction,
                )[0].detach().cpu().numpy()
                domain_bad = bool(np.max(np.abs(projected - applied)) > 3.0e-6)
                rate_violations += int(rate_bad)
                domain_violations += int(domain_bad)
                hidden_jumps += int(rate_bad)
                transition_active_steps += int(bool(env.transition_active[0].item()))
                obs, _, _, dones, _ = env.step(policy(obs))
                next_position = env.root_states[0, :2].detach().cpu().numpy() - env.env_origins[0, :2].detach().cpu().numpy()
                path_length += float(np.linalg.norm(next_position - previous_position))
                previous_position = next_position
                distance = float(np.linalg.norm(next_position - target))
                lateral_error = _distance_to_polyline(next_position, scenario.centerline)
                min_distance = min(min_distance, distance)
                max_lateral_error = max(max_lateral_error, lateral_error)
                collision_now = lateral_error > scenario.width_m / 2.0 - ROBOT_RADIUS_M
                collision = collision or collision_now
                trajectory.append({
                    "episode_id": episode_id,
                    "time_s": (step + 1) * float(env.dt),
                    "x": float(next_position[0]),
                    "y": float(next_position[1]),
                    "goal_distance": distance,
                    "v_cmd": float(applied[0]),
                    "v_actual": float(actual[0]),
                    "w_cmd": float(applied[1]),
                    "w_actual": float(actual[1]),
                    "collision": int(collision_now),
                    "rate_violation": int(rate_bad),
                    "feasible_domain_violation": int(domain_bad),
                    "hidden_projection_jump": int(rate_bad),
                    "transition_active": int(bool(env.transition_active[0].item())),
                })
                previous_applied = applied.copy()
                if distance <= 0.35:
                    success = True
                    break
                if bool(dones[0].item()):
                    timeout = True
                    break
            if not success and not timeout and step + 1 >= max_steps:
                timeout = True
            divergent = (not success) and (collision or min_distance > 1.0)
            record = {
                "episode_id": episode_id,
                "seed": int(scenario.seed) + episode_id - 1,
                "scenario_family": scenario.family,
                "scenario_parameters": {"width_m": scenario.width_m, "path_length_m": scenario.path_length_m},
                "success": success,
                "timeout": timeout and not success,
                "divergent": divergent,
                "collision": collision,
                "duration_s": len(trajectory) * float(env.dt),
                "path_length_m": path_length,
                "min_goal_distance_m": min_distance,
                "max_lateral_error_m": max_lateral_error,
                "rate_violation_count": rate_violations,
                "feasible_domain_violation_count": domain_violations,
                "hidden_projection_jump_count": hidden_jumps,
                "transition_activation_count": transition_active_steps,
            }
            logger.write_episode(record)
            logger.write_trajectory(trajectory)
            all_records.append(record)
            print("corridor episode=%d success=%s timeout=%s divergence=%s min_distance=%.3f" % (episode_id, success, timeout, divergent, min_distance), flush=True)
        total = float(max(len(all_records), 1))
        summary = {
            "family": scenario.family,
            "episodes": len(all_records),
            "success_rate": sum(int(row["success"]) for row in all_records) / total,
            "collision_rate": sum(int(row["collision"]) for row in all_records) / total,
            "timeout_rate": sum(int(row["timeout"]) for row in all_records) / total,
            "divergence_rate": sum(int(row["divergent"]) for row in all_records) / total,
            "rate_violation_count": sum(row["rate_violation_count"] for row in all_records),
            "feasible_domain_violation_count": sum(row["feasible_domain_violation_count"] for row in all_records),
            "hidden_projection_jump_count": sum(row["hidden_projection_jump_count"] for row in all_records),
            "transition_activation_count": sum(row["transition_activation_count"] for row in all_records),
            "max_lateral_error_m": max(row["max_lateral_error_m"] for row in all_records),
        }
        logger.write_summary(summary)
        if (output_dir / "trajectory.csv").is_file():
            plot_corridor_artifacts(output_dir / "trajectory.csv", output_dir / "plots")
        if scenario.family == "straight":
            current_rules = {"success_rate": (">=", 1.0), "collision_rate": ("==", 0.0), "rate_violation_count": ("==", 0), "feasible_domain_violation_count": ("==", 0), "hidden_projection_jump_count": ("==", 0), "max_lateral_error_m": ("<", 0.20)}
        elif scenario.family == "l":
            current_rules = {"success_rate": (">=", 0.95), "collision_rate": ("==", 0.0), "rate_violation_count": ("==", 0), "feasible_domain_violation_count": ("==", 0), "hidden_projection_jump_count": ("==", 0), "transition_activation_count": (">", 0), "max_lateral_error_m": ("<", 0.30)}
        else:
            current_rules = {"success_rate": (">=", 0.90), "collision_rate": ("<=", 1.0 / max(episodes, 1)), "rate_violation_count": ("==", 0), "feasible_domain_violation_count": ("==", 0), "hidden_projection_jump_count": ("==", 0)}
        gate = GateResult.evaluate(summary, current_rules, current_rules)
        summary["gate"] = gate
        logger.write_summary(summary)
        print("CORRIDOR SUMMARY %s" % summary, flush=True)
        if enforce_gate and not gate["pass"]:
            raise RuntimeError("%s Gate failed: %s" % (scenario.family, gate["failures"]))
        return summary
    finally:
        if hasattr(env, "close"):
            env.close()


def main():
    args = _parse_args()
    scenario = _scenario_for_family(args.corridor_family, args.corridor_seed)
    output_dir = args.corridor_output_dir / args.corridor_family.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_corridor(args, scenario, args.corridor_episodes, output_dir, enforce_gate=True, max_steps=args.corridor_max_steps)


if __name__ == "__main__":
    main()
