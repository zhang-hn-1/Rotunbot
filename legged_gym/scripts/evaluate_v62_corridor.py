"""Evaluate frozen V62 spatial motion in deterministic corridor scenarios."""

import argparse
import json
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
from legged_gym.navigation.corridor_artifacts import (
    CheckpointMetadata,
    EpisodeLogger,
    GateResult,
)
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
    parser.add_argument("--corridor_resume", action="store_true")
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
    args.corridor_resume = diagnostic.corridor_resume
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


def _p95(values):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95.0))


def _yaw_from_quaternion(quat):
    return float(
        torch.atan2(
            2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
            1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2),
        ).item()
    )


def _yaw_error_to_path(position, yaw, scenario):
    nearest = int(
        np.argmin(np.linalg.norm(scenario.centerline - position.reshape(1, 2), axis=1))
    )
    left = max(0, nearest - 1)
    right = min(len(scenario.centerline) - 1, nearest + 1)
    path_yaw = math.atan2(
        float(scenario.centerline[right, 1] - scenario.centerline[left, 1]),
        float(scenario.centerline[right, 0] - scenario.centerline[left, 0]),
    )
    error = (path_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
    return abs(float(error))


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
    # S0 has one external scripted command source.  The V62 training task
    # normally samples smooth/random-walk profiles at the same upper tick;
    # leaving those enabled would overwrite the corridor controller's target.
    env_cfg.commands.smooth_profile_fraction = 0.0
    env_cfg.commands.random_walk_profile_fraction = 0.0
    env_cfg.commands.independent_smooth_profile_fraction = 0.0
    env_cfg.corridor_wall_width_m = scenario.width_m
    env_cfg.corridor_wall_segments = make_wall_segments(scenario.centerline)


def _coerce_record(record):
    """Restore the scalar types needed when aggregating resumed CSV records."""
    converted = {}
    for key, value in record.items():
        if value == "True":
            converted[key] = True
        elif value == "False":
            converted[key] = False
        elif key == "scenario_parameters":
            converted[key] = json.loads(value)
        else:
            try:
                converted[key] = int(value)
            except (TypeError, ValueError):
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
    return converted


def _new_controller(env):
    cfg = env.cfg.commands
    return PoseBasedCorridorController(
        maximum_forward_speed=cfg.max_forward_speed,
        maximum_yaw_rate=cfg.max_yaw_rate,
        minimum_turn_radius=cfg.minimum_turn_radius,
        envelope_fraction=cfg.feasible_envelope_fraction,
        straight_speed=0.20,
    )


def run_corridor(args, scenario, episodes, output_dir, enforce_gate=True, max_steps=None, resume=False):
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
    logger = EpisodeLogger(output_dir, append=resume)
    metadata = CheckpointMetadata.from_path(
        args.corridor_checkpoint,
        parent=args.corridor_checkpoint,
        stage="S0_%s" % scenario.family,
        seed=scenario.seed,
        iterations=0,
    )
    (output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    controller = _new_controller(env)
    try:
        if max_steps is None:
            max_steps = int(math.ceil(max(240.0, scenario.path_length_m / 0.05) / env.dt))
        all_records = [_coerce_record(row) for row in logger.episodes]
        existing_ids = sorted(int(row["episode_id"]) for row in all_records)
        if existing_ids and existing_ids != list(range(1, len(existing_ids) + 1)):
            raise ValueError("resume artifacts must contain contiguous episode IDs starting at 1")
        start_episode = (existing_ids[-1] + 1) if existing_ids else 1
        if start_episode > episodes:
            raise ValueError("resume artifacts already contain the requested episode count")
        for episode_id in range(start_episode, episodes + 1):
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
            previous_raw = np.zeros(2, dtype=np.float64)
            raw_command = np.zeros(2, dtype=np.float64)
            command_corrections = []
            rate_violations = 0
            domain_violations = 0
            hidden_jumps = 0
            transition_active_steps = 0
            governor_active_steps = 0
            projection_active_steps = 0
            transition_activation_events = 0
            sign_reversal_count = 0
            collision = False
            success = False
            timeout = False
            for step in range(max_steps):
                position = env.root_states[0, :2].detach().cpu().numpy() - env.env_origins[0, :2].detach().cpu().numpy()
                quat = env.base_quat[0]
                yaw = _yaw_from_quaternion(quat)
                if step % UPPER_HOLD_STEPS == 0:
                    previous_target = previous_raw.copy()
                    command = controller.update(position, yaw, scenario)
                    raw_command = np.asarray(command, dtype=np.float64)
                    target_changed = bool(
                        np.max(np.abs(raw_command - previous_target)) > 3.0e-6
                    )
                    transition_activation_events += int(target_changed)
                    if abs(raw_command[1]) > 1.0e-6 and abs(previous_raw[1]) > 1.0e-6:
                        if np.sign(raw_command[1]) != np.sign(previous_raw[1]):
                            sign_reversal_count += 1
                    previous_raw = raw_command.copy()
                    projected_raw = project_velocity_commands(
                        torch.as_tensor(raw_command, device=env.device).reshape(1, 2),
                        env.cfg.commands.max_forward_speed,
                        env.cfg.commands.max_yaw_rate,
                        env.cfg.commands.minimum_turn_radius,
                        env.cfg.commands.feasible_envelope_fraction,
                    )[0].detach().cpu().numpy()
                    projection_active_steps += int(
                        np.max(np.abs(projected_raw - raw_command)) > 3.0e-6
                    )
                    env.set_command_targets(torch.as_tensor(raw_command, device=env.device))
                    env.compute_observations()
                applied = env.commands[0, :2].detach().cpu().numpy()
                actual = np.asarray((env.tracking_lin_vel[0, 0].item(), env.tracking_ang_vel[0, 2].item()))
                command_corrections.append(float(np.linalg.norm(raw_command - applied)))
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
                transition_active = int(bool(env.transition_active[0].item()))
                transition_active_steps += transition_active
                governor_active_steps += int(np.linalg.norm(raw_command - applied) > 3.0e-6)
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
                    "goal_bearing": float(math.atan2(float(target[1] - next_position[1]), float(target[0] - next_position[0])) - _yaw_from_quaternion(env.base_quat[0])),
                    "robot_yaw": _yaw_from_quaternion(env.base_quat[0]),
                    "goal_x": float(target[0]),
                    "goal_y": float(target[1]),
                    "raw_sru_action_v": float(raw_command[0]),
                    "raw_sru_action_w": float(raw_command[1]),
                    "v_cmd": float(raw_command[0]),
                    "v_actual": float(actual[0]),
                    "w_cmd": float(raw_command[1]),
                    "w_actual": float(actual[1]),
                    "v_after_transition": float(applied[0]),
                    "w_after_transition": float(applied[1]),
                    "collision": int(collision_now),
                    "wall_clearance": float(scenario.width_m / 2.0 - ROBOT_RADIUS_M - lateral_error),
                    "rate_violation": int(rate_bad),
                    "feasible_domain_violation": int(domain_bad),
                    "hidden_projection_jump": int(rate_bad),
                    "transition_active": transition_active,
                    "transition_activation_event": int(
                        target_changed if step % UPPER_HOLD_STEPS == 0 else 0
                    ),
                    "governor_active": int(np.linalg.norm(raw_command - applied) > 3.0e-6),
                    "projection_active": int(np.max(np.abs(projected_raw - raw_command)) > 3.0e-6) if step % UPPER_HOLD_STEPS == 0 else 0,
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
            final_position = previous_position
            final_yaw = _yaw_from_quaternion(env.base_quat[0])
            final_lateral_error = _distance_to_polyline(final_position, scenario.centerline)
            final_yaw_error_deg = math.degrees(
                _yaw_error_to_path(final_position, final_yaw, scenario)
            )
            near_miss = (not success) and min_distance <= 1.0
            stuck = (not success) and len(trajectory) >= 100 and (
                np.linalg.norm(
                    np.asarray((trajectory[-1]["x"], trajectory[-1]["y"]))
                    - np.asarray((trajectory[-100]["x"], trajectory[-100]["y"]))
                )
                < 0.05
            )
            oscillation = sign_reversal_count >= 4
            divergent = (not success) and collision
            record = {
                "episode_id": episode_id,
                "seed": int(scenario.seed) + episode_id - 1,
                "scenario_family": scenario.family,
                "stage": "S0",
                "scenario_parameters": {
                    "width_m": scenario.width_m,
                    "path_length_m": scenario.path_length_m,
                    "turn_count": len(scenario.turns),
                },
                "robot_x": float(final_position[0]),
                "robot_y": float(final_position[1]),
                "robot_yaw": float(final_yaw),
                "goal_x": float(target[0]),
                "goal_y": float(target[1]),
                "goal_distance": float(np.linalg.norm(final_position - target)),
                "success": success,
                "timeout": timeout and not success,
                "divergent": divergent,
                "collision": collision,
                "near_miss": near_miss,
                "stuck": stuck,
                "oscillation": oscillation,
                "duration_s": len(trajectory) * float(env.dt),
                "path_length_m": path_length,
                "min_goal_distance_m": min_distance,
                "final_lateral_error_m": final_lateral_error,
                "final_yaw_error_deg": final_yaw_error_deg,
                "max_lateral_error_m": max_lateral_error,
                "min_wall_clearance_m": scenario.width_m / 2.0 - ROBOT_RADIUS_M - max_lateral_error,
                "rate_violation_count": rate_violations,
                "feasible_domain_violation_count": domain_violations,
                "hidden_projection_jump_count": hidden_jumps,
                "transition_activation_count": transition_activation_events,
                "transition_active_step_count": transition_active_steps,
                "transition_activation_event_count": transition_activation_events,
                "governor_activation_count": governor_active_steps,
                "projection_activation_count": projection_active_steps,
                "mean_command_correction": float(np.mean(command_corrections)) if command_corrections else 0.0,
                "p95_command_correction": _p95(command_corrections),
                "sign_reversal_count": sign_reversal_count,
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
            "near_miss_rate": sum(int(row["near_miss"]) for row in all_records) / total,
            "stuck_rate": sum(int(row["stuck"]) for row in all_records) / total,
            "oscillation_rate": sum(int(row["oscillation"]) for row in all_records) / total,
            "rate_violation_count": sum(row["rate_violation_count"] for row in all_records),
            "feasible_domain_violation_count": sum(row["feasible_domain_violation_count"] for row in all_records),
            "hidden_projection_jump_count": sum(row["hidden_projection_jump_count"] for row in all_records),
            "transition_activation_count": sum(row["transition_activation_event_count"] for row in all_records),
            "transition_active_step_count": sum(row["transition_active_step_count"] for row in all_records),
            "transition_activation_event_count": sum(row["transition_activation_event_count"] for row in all_records),
            "governor_activation_count": sum(row["governor_activation_count"] for row in all_records),
            "projection_activation_count": sum(row["projection_activation_count"] for row in all_records),
            "max_lateral_error_m": max(row["max_lateral_error_m"] for row in all_records),
            "max_final_lateral_error_m": max(row["final_lateral_error_m"] for row in all_records),
            "max_final_yaw_error_deg": max(row["final_yaw_error_deg"] for row in all_records),
            "min_wall_clearance_m": min(row["min_wall_clearance_m"] for row in all_records),
            "mean_command_correction": float(np.mean([row["mean_command_correction"] for row in all_records])),
            "p95_command_correction": _p95([row["p95_command_correction"] for row in all_records]),
            "sign_reversal_count": sum(row["sign_reversal_count"] for row in all_records),
        }
        total_steps = sum(
            int(round(row["duration_s"] / float(env.dt))) for row in all_records
        )
        summary["transition_activation_ratio"] = summary["transition_active_step_count"] / max(total_steps, 1)
        summary["governor_activation_ratio"] = summary["governor_activation_count"] / max(total_steps, 1)
        summary["projection_activation_ratio"] = summary["projection_activation_count"] / max(episodes * UPPER_HOLD_STEPS, 1)
        logger.write_summary(summary)
        if (output_dir / "trajectory.csv").is_file():
            plot_corridor_artifacts(output_dir / "trajectory.csv", output_dir / "plots")
        if scenario.family == "straight":
            current_rules = {"success_rate": (">=", 1.0), "collision_rate": ("==", 0.0), "rate_violation_count": ("==", 0), "feasible_domain_violation_count": ("==", 0), "hidden_projection_jump_count": ("==", 0), "max_final_lateral_error_m": ("<", 0.20), "max_final_yaw_error_deg": ("<", 5.0)}
        elif scenario.family == "l":
            current_rules = {"success_rate": (">=", 0.95), "collision_rate": ("==", 0.0), "rate_violation_count": ("==", 0), "feasible_domain_violation_count": ("==", 0), "hidden_projection_jump_count": ("==", 0), "transition_activation_count": (">", 0), "max_final_lateral_error_m": ("<", 0.30)}
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
    run_corridor(
        args,
        scenario,
        args.corridor_episodes,
        output_dir,
        enforce_gate=True,
        max_steps=args.corridor_max_steps,
        resume=args.corridor_resume,
    )


if __name__ == "__main__":
    main()
