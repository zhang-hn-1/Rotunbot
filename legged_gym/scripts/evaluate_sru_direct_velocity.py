"""Formally evaluate direct SRU velocity checkpoints for Task 7."""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import isaacgym  # noqa: F401 - must precede torch
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import project_velocity_commands
from legged_gym.navigation.corridor_artifacts import EpisodeLogger
from legged_gym.navigation.corridor_plotting import plot_corridor_artifacts
from legged_gym.navigation.direct_velocity import normalized_action_to_velocity_command
from legged_gym.navigation.direct_velocity_curriculum import configure_direct_velocity_stage
from legged_gym.navigation.direct_velocity_evaluation import (
    CommandDiagnostics,
    DEFAULT_EVALUATION_SEEDS,
    build_fixed_goal_specs,
    load_checkpoint_identity,
    select_step_telemetry,
    summarize_evaluation,
    write_failure_artifacts,
)
from legged_gym.utils import get_args, task_registry


def _parse_framework_args(remaining):
    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + list(remaining)
    try:
        return get_args()
    finally:
        os.sys.argv = original


def _project_applied(env, command):
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


def _raw_velocity_command(actions, maximum_forward_speed, maximum_yaw_rate):
    raw = actions.clamp(-1.0, 1.0).clone()
    raw[:, 0] *= float(maximum_forward_speed)
    raw[:, 1] *= float(maximum_yaw_rate)
    return raw


def _assign_goal(env, env_index, spec):
    index = int(env_index)
    yaw = float(
        env._yaw_from_quaternion(env.root_states[index:index + 1, 3:7])[0].item()
    )
    world_bearing = yaw + float(spec["bearing_rad"])
    distance = float(spec["distance_m"])
    env.global_goal_xy_world[index, 0] = (
        env.root_states[index, 0] + distance * math.cos(world_bearing)
    )
    env.global_goal_xy_world[index, 1] = (
        env.root_states[index, 1] + distance * math.sin(world_bearing)
    )
    env.goal_dist[index] = distance
    env.terminal_goal_distance[index] = distance
    env.previous_goal_distance[index] = distance
    env.goal_reached_buf[index] = False
    env.success_buf[index] = False


def _new_episode_state(env, env_index, spec):
    position = env.root_states[env_index, :2].detach().cpu().numpy().copy()
    return {
        "spec": spec,
        "steps": 0,
        "path_length_m": 0.0,
        "previous_position": position,
        "min_goal_distance_m": float(spec["distance_m"]),
        "last_goal_distance_m": float(spec["distance_m"]),
        "diagnostics": CommandDiagnostics(
            policy_dt=float(env.dt),
            maximum_linear_acceleration=env.cfg.commands.maximum_linear_acceleration,
            maximum_yaw_acceleration=env.cfg.commands.maximum_yaw_acceleration,
            projection_jump_threshold=(0.02, 0.01),
        ),
        "trajectory": [],
    }


def _episode_record(env, env_index, state, success, collision, timeout, divergent):
    spec = state["spec"]
    diagnostics = state["diagnostics"].summary()
    record = {
        "episode_id": int(spec["episode_id"]),
        "seed": int(spec["seed"]),
        "stage": str(spec.get("evaluation_stage", "")),
        "component": str(spec["component"]),
        "initial_goal_distance_m": float(spec["distance_m"]),
        "initial_goal_bearing_deg": math.degrees(float(spec["bearing_rad"])),
        "success": bool(success),
        "collision": bool(collision),
        "timeout": bool(timeout),
        "divergent": bool(divergent),
        "duration_s": state["steps"] * float(env.dt),
        "path_length_m": float(state["path_length_m"]),
        "min_goal_distance_m": float(state["min_goal_distance_m"]),
        "terminal_goal_distance_m": float(state["last_goal_distance_m"]),
    }
    record.update(diagnostics)
    return record


def evaluate_velocity_local_goal(
    checkpoint,
    stage,
    seed_list=None,
    episodes=100,
    num_envs=16,
    output_dir=None,
    max_steps=None,
    parent_checkpoint=None,
    enforce_gate=False,
    framework_args=(),
):
    """Evaluate one B-stage set and emit the complete Task 7 artifact bundle."""
    started = time.monotonic()
    stage = str(stage).upper()
    seeds = tuple(DEFAULT_EVALUATION_SEEDS if seed_list is None else seed_list)
    specs = build_fixed_goal_specs(stage, episodes=episodes, seed_list=seeds)
    for spec in specs:
        spec["evaluation_stage"] = stage
    output_dir = Path(output_dir or ("logs/phase_b/%s" % stage.lower())).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = load_checkpoint_identity(checkpoint, parent_checkpoint=parent_checkpoint)
    if identity["parent_checkpoint"] is None:
        raise ValueError("formal evaluation requires checkpoint parent metadata")

    random.seed(int(seeds[0]))
    np.random.seed(int(seeds[0]))
    torch.manual_seed(int(seeds[0]))
    args = _parse_framework_args(framework_args)
    args.task = "rotunbot_sru_direct_velocity"
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    configure_direct_velocity_stage(env_cfg, stage)
    env_cfg.env.num_envs = min(max(1, int(num_envs)), int(episodes))
    env_cfg.noise.add_noise = False
    env_cfg.camera.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.commands.random_start_yaw = False
    formal_steps = int(
        round(30.0 / (env_cfg.sim.dt * env_cfg.control.decimation))
    )
    max_steps = formal_steps if max_steps is None else int(max_steps)
    env_cfg.env.episode_length_s = max(
        31.0, (max_steps + 10) * env_cfg.sim.dt * env_cfg.control.decimation
    )
    train_cfg.seed = int(seeds[0])
    train_cfg.runner.resume = False

    env = None
    logger = EpisodeLogger(output_dir)
    try:
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
        runner, _ = task_registry.make_alg_runner(
            env=env,
            name=args.task,
            args=args,
            train_cfg=train_cfg,
            log_root=None,
        )
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        obs, _ = env.reset()

        next_spec = 0
        active = {}
        held_actions = torch.zeros(env.num_envs, 2, device=env.device)
        held_raw_commands = torch.zeros_like(held_actions)
        held_requested_commands = torch.zeros_like(held_actions)
        for env_index in range(env.num_envs):
            spec = specs[next_spec]
            next_spec += 1
            _assign_goal(env, env_index, spec)
            active[env_index] = _new_episode_state(env, env_index, spec)
        env.compute_observations()
        obs = env.get_observations()

        records = []
        all_trajectory_rows = []
        print(
            "Formal evaluation: stage=%s episodes=%d seeds=%s checkpoint=%s"
            % (stage, episodes, seeds, identity["checkpoint"]),
            flush=True,
        )
        with torch.no_grad():
            while len(records) < episodes:
                actions = policy(obs)
                upper_tick = (
                    env.common_step_counter
                    % env.upper_level_command_interval_steps
                    == 0
                )
                if upper_tick:
                    held_actions.copy_(actions.clamp(-1.0, 1.0))
                    held_raw_commands.copy_(
                        _raw_velocity_command(
                            actions,
                            env.cfg.commands.max_forward_speed,
                            env.cfg.commands.max_yaw_rate,
                        )
                    )
                    held_requested_commands.copy_(
                        normalized_action_to_velocity_command(
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
                    )
                obs, _, _, dones, _ = env.step(actions)
                done_mask = dones.flatten().bool()
                post_applied = env.applied_feasible_command.detach().clone()
                post_actual = torch.stack(
                    (env.tracking_lin_vel[:, 0], env.tracking_ang_vel[:, 2]),
                    dim=1,
                ).detach().clone()
                post_position = env.root_states[:, :2].detach().clone()
                post_target = env.command_targets.detach().clone()
                post_goal_xy = env._goal_xy_robot().detach().clone()
                post_goal_distance = env.goal_dist.detach().clone()
                post_transition_active = env.transition_active.detach().clone()
                post_transition_state = env.transition_state.detach().clone()
                post_recovery_active = env.goal_recovery_active.detach().clone()
                assigned_new_goal = False

                for env_index in list(active):
                    state = active[env_index]
                    state["steps"] += 1
                    forced_timeout = (
                        state["steps"] >= max_steps
                        and not bool(done_mask[env_index].item())
                    )
                    auto_done = bool(done_mask[env_index].item())
                    post_step = {
                        "applied_command": post_applied[env_index],
                        "actual_velocity": post_actual[env_index],
                        "position": post_position[env_index],
                        "command_target": post_target[env_index],
                        "goal_xy_robot": post_goal_xy[env_index],
                        "goal_distance": post_goal_distance[env_index],
                        "transition_active": post_transition_active[env_index],
                        "transition_state": post_transition_state[env_index],
                        "goal_recovery_active": post_recovery_active[env_index],
                        "success": env.success_buf[env_index],
                        "collision": env.step_collision_buf[env_index],
                        "timeout": env.time_out_buf[env_index],
                    }
                    terminal_post_step = None
                    if auto_done and hasattr(
                        env, "terminal_applied_feasible_command"
                    ):
                        terminal_post_step = {
                            "applied_command": env.terminal_applied_feasible_command[
                                env_index
                            ],
                            "actual_velocity": env.terminal_tracking_velocity[env_index],
                            "position": env.terminal_position[env_index],
                            "command_target": env.terminal_command_target[env_index],
                            "goal_xy_robot": env.terminal_goal_xy_robot[env_index],
                            "goal_distance": env.terminal_goal_distance[env_index],
                            "transition_active": env.terminal_transition_active[
                                env_index
                            ],
                            "transition_state": env.terminal_transition_state[env_index],
                            "goal_recovery_active": (
                                env.terminal_goal_recovery_active[env_index]
                            ),
                            "success": env.terminal_success[env_index],
                            "collision": env.terminal_collision[env_index],
                            "timeout": env.terminal_timeout[env_index],
                        }
                    telemetry = select_step_telemetry(
                        auto_done=auto_done,
                        post_step=post_step,
                        terminal_post_step=terminal_post_step,
                    )
                    applied_tensor = telemetry["applied_command"]
                    actual_tensor = telemetry["actual_velocity"]
                    position_tensor = telemetry["position"]
                    projected_applied = _project_applied(
                        env, applied_tensor.reshape(1, 2)
                    )[0]
                    transition_active = bool(telemetry["transition_active"].item())
                    diagnostic_row = state["diagnostics"].record(
                        held_raw_commands[env_index].detach().cpu().tolist(),
                        held_requested_commands[env_index].detach().cpu().tolist(),
                        applied_tensor.detach().cpu().tolist(),
                        projected_applied.detach().cpu().tolist(),
                        transition_active,
                    )
                    position = position_tensor.detach().cpu().numpy()
                    state["path_length_m"] += float(
                        np.linalg.norm(position - state["previous_position"])
                    )
                    state["previous_position"] = position.copy()
                    goal_xy = telemetry["goal_xy_robot"]
                    goal_xy_robot = (
                        float(goal_xy[0].item()),
                        float(goal_xy[1].item()),
                    )
                    goal_distance = float(telemetry["goal_distance"].item())
                    goal_bearing = math.atan2(
                        goal_xy_robot[1], goal_xy_robot[0]
                    )
                    state["last_goal_distance_m"] = goal_distance
                    state["min_goal_distance_m"] = min(
                        state["min_goal_distance_m"], goal_distance
                    )
                    target = telemetry["command_target"].detach().cpu().tolist()
                    row = {
                        "episode_id": int(state["spec"]["episode_id"]),
                        "seed": int(state["spec"]["seed"]),
                        "component": state["spec"]["component"],
                        "step": state["steps"],
                        "time_s": state["steps"] * float(env.dt),
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "goal_x_robot": goal_xy_robot[0],
                        "goal_y_robot": goal_xy_robot[1],
                        "goal_bearing_rad": goal_bearing,
                        "goal_distance": goal_distance,
                        "raw_action_v": float(held_actions[env_index, 0].item()),
                        "raw_action_w": float(held_actions[env_index, 1].item()),
                        "raw_v_cmd": float(
                            held_raw_commands[env_index, 0].item()
                        ),
                        "raw_w_cmd": float(
                            held_raw_commands[env_index, 1].item()
                        ),
                        "requested_v_cmd": float(
                            held_requested_commands[env_index, 0].item()
                        ),
                        "requested_w_cmd": float(
                            held_requested_commands[env_index, 1].item()
                        ),
                        "target_v_cmd": float(target[0]),
                        "target_w_cmd": float(target[1]),
                        "v_cmd": float(applied_tensor[0].item()),
                        "w_cmd": float(applied_tensor[1].item()),
                        "v_actual": float(actual_tensor[0].item()),
                        "w_actual": float(actual_tensor[1].item()),
                        "transition_state": int(telemetry["transition_state"].item()),
                        "goal_recovery_active": int(
                            telemetry["goal_recovery_active"].item()
                        ),
                    }
                    row.update(diagnostic_row)
                    state["trajectory"].append(row)

                    if not auto_done and not forced_timeout:
                        continue
                    success = (
                        bool(telemetry["success"].item())
                        and not forced_timeout
                    )
                    collision = (
                        bool(telemetry["collision"].item())
                        and not success
                    )
                    timeout = forced_timeout or (
                        bool(telemetry["timeout"].item())
                        and not collision
                    )
                    divergent = not success and not collision and not timeout
                    record = _episode_record(
                        env,
                        env_index,
                        state,
                        success=success,
                        collision=collision,
                        timeout=timeout,
                        divergent=divergent,
                    )
                    logger.write_episode(record)
                    records.append(record)
                    all_trajectory_rows.extend(state["trajectory"])
                    if not success:
                        write_failure_artifacts(
                            output_dir, record, state["trajectory"]
                        )
                    print(
                        "episode %3d: component=%s success=%d collision=%d "
                        "timeout=%d final_dist=%.3f raw_reverse=%d "
                        "applied_reverse=%d transitions=%d"
                        % (
                            record["episode_id"],
                            record["component"],
                            int(success),
                            int(collision),
                            int(timeout),
                            record["terminal_goal_distance_m"],
                            record["raw_reverse_command_count"],
                            record["applied_reverse_command_count"],
                            record["transition_activation_count"],
                        ),
                        flush=True,
                    )
                    if forced_timeout:
                        env.reset_idx(
                            torch.as_tensor(
                                [env_index],
                                device=env.device,
                                dtype=torch.long,
                            )
                        )
                    del active[env_index]
                    if next_spec < len(specs):
                        spec = specs[next_spec]
                        next_spec += 1
                        _assign_goal(env, env_index, spec)
                        active[env_index] = _new_episode_state(
                            env, env_index, spec
                        )
                        assigned_new_goal = True
                if assigned_new_goal:
                    env.compute_observations()
                    obs = env.get_observations()

        records.sort(key=lambda row: int(row["episode_id"]))
        all_trajectory_rows.sort(
            key=lambda row: (int(row["episode_id"]), int(row["step"]))
        )
        logger.write_trajectory(all_trajectory_rows)
        if all_trajectory_rows:
            plot_corridor_artifacts(
                output_dir / "trajectory.csv", output_dir / "plots"
            )
        summary = summarize_evaluation(
            records,
            stage=stage,
            seed_list=seeds,
            checkpoint_identity=identity,
            wall_clock_seconds=time.monotonic() - started,
        )
        summary["artifact_root"] = str(output_dir)
        summary["failure_artifact_count"] = sum(
            not row["success"] for row in records
        )
        logger.write_summary(summary)
        print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
        if enforce_gate and not summary["gate"]["pass"]:
            raise RuntimeError(
                "%s Gate failed: %s" % (stage, summary["gate"]["failures"])
            )
        return summary
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()


def evaluate(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--stage", choices=("S1", "S2", "S2B"), default="S1"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--parent_checkpoint", default=None)
    parser.add_argument("--seed_list", type=int, nargs="+", default=None)
    parser.add_argument("--enforce_gate", action="store_true")
    stage_args, remaining = parser.parse_known_args(
        sys.argv[1:] if argv is None else argv
    )
    return evaluate_velocity_local_goal(
        checkpoint=stage_args.checkpoint,
        stage=stage_args.stage,
        seed_list=stage_args.seed_list,
        episodes=stage_args.episodes,
        num_envs=stage_args.num_envs,
        output_dir=stage_args.output_dir,
        max_steps=stage_args.max_steps,
        parent_checkpoint=stage_args.parent_checkpoint,
        enforce_gate=stage_args.enforce_gate,
        framework_args=remaining,
    )


if __name__ == "__main__":
    evaluate()
