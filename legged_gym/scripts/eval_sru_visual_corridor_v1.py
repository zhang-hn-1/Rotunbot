"""Independent V1 curriculum and fixed-distance evaluator."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import isaacgym  # noqa: F401 - must precede torch
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.navigation.corridor_artifacts import EpisodeLogger
from legged_gym.navigation.direct_velocity_evaluation import (
    CommandDiagnostics,
    select_step_telemetry,
    write_failure_artifacts,
)
from legged_gym.navigation.v1_evaluation import (
    build_fixed_distance_specs,
    curriculum_gate,
    summarize_v1_episodes,
)
from legged_gym.scripts.evaluate_sru_direct_velocity import (
    _assign_goal,
    _parse_framework_args,
    _project_applied,
    _raw_velocity_command,
)
from legged_gym.utils import task_registry


def evaluation_targets(current_distance, next_distance=None, episodes=30):
    """Return the requested curriculum pair or the fixed 6 m formal set."""
    current_distance = float(current_distance)
    episodes = int(episodes)
    if next_distance is None and math.isclose(current_distance, 6.0) and episodes == 100:
        return [("fixed_6m", 6.0, 100)]
    if next_distance is None:
        return [("current", current_distance, episodes)]
    return [
        ("current", current_distance, episodes),
        ("next", float(next_distance), episodes),
    ]


def close_environment(env):
    """Destroy Isaac Gym resources for tasks that do not expose close()."""
    if env is None:
        return
    viewer = getattr(env, "viewer", None)
    if viewer is not None:
        env.gym.destroy_viewer(viewer)
    sim = getattr(env, "sim", None)
    if sim is not None:
        env.gym.destroy_sim(sim)


def reset_recurrent_hidden(actor_critic, done_mask):
    """Reset only finished vectorized environments between episode actions."""
    reset = getattr(actor_critic, "reset", None)
    if callable(reset):
        reset(done_mask.flatten().bool())


def _parse_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--current-distance", type=float, default=6.0)
    parser.add_argument("--next-distance", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--depth-backend", choices=("fallback", "isaacgym"), default="fallback"
    )
    parser.add_argument("--output-dir", required=True)
    parsed, remaining = parser.parse_known_args(argv)
    return parsed, remaining


def _episode_state(env, spec, env_index=0):
    position = env.root_states[env_index, :2].detach().cpu().numpy().copy()
    return {
        "spec": spec,
        "steps": 0,
        "path_length_m": 0.0,
        "previous_position": position,
        "last_goal_distance_m": float(spec["distance_m"]),
        "reverse_steps": 0,
        "forward_velocity_sum": 0.0,
        "absolute_yaw_velocity_sum": 0.0,
        "diagnostics": CommandDiagnostics(
            policy_dt=float(env.dt),
            maximum_linear_acceleration=env.cfg.commands.maximum_linear_acceleration,
            maximum_yaw_acceleration=env.cfg.commands.maximum_yaw_acceleration,
            projection_jump_threshold=(0.02, 0.01),
        ),
        "trajectory": [],
    }


def _record(env, env_index, state, telemetry, forced_timeout):
    success = bool(telemetry["success"].item()) and not forced_timeout
    collision = bool(telemetry["collision"].item()) and not success
    timeout = forced_timeout or (bool(telemetry["timeout"].item()) and not collision)
    divergent = not success and not collision and not timeout
    diagnostics = state["diagnostics"].summary()
    return {
        "episode_id": int(state["spec"]["episode_id"]),
        "seed": int(state["spec"]["seed"]),
        "distance_m": float(state["spec"]["distance_m"]),
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "divergent": divergent,
        "steps": int(state["steps"]),
        "episode_length": int(state["steps"]),
        "path_length_m": float(state["path_length_m"]),
        "initial_goal_distance_m": float(state["spec"]["distance_m"]),
        "terminal_goal_distance_m": float(state["last_goal_distance_m"]),
        "mean_forward_velocity": (
            state["forward_velocity_sum"] / max(state["steps"], 1)
        ),
        "mean_absolute_yaw_velocity": (
            state["absolute_yaw_velocity_sum"] / max(state["steps"], 1)
        ),
        "reverse_steps": int(state["reverse_steps"]),
        "trajectory_rows": len(state["trajectory"]),
        **diagnostics,
    }


def _trajectory_row(
    episode_id,
    step,
    distance_m,
    position,
    goal_distance,
    raw_v,
    raw_w,
    requested_v,
    requested_w,
    applied_v,
    applied_w,
    actual_v,
    actual_w,
    dt,
):
    """Build the shared corridor-plot trajectory schema."""
    return {
        "episode_id": int(episode_id),
        "step": int(step),
        "time_s": float(step) * float(dt),
        "distance_m": float(distance_m),
        "x": float(position[0]),
        "y": float(position[1]),
        "goal_distance": float(goal_distance),
        "raw_v_cmd": float(raw_v),
        "raw_w_cmd": float(raw_w),
        "requested_v_cmd": float(requested_v),
        "requested_w_cmd": float(requested_w),
        "v_cmd": float(applied_v),
        "w_cmd": float(applied_w),
        "v_actual": float(actual_v),
        "w_actual": float(actual_w),
    }


def evaluate_distance(
    checkpoint,
    distance,
    episodes,
    seed,
    output_dir,
    num_envs=16,
    max_steps=None,
    depth_backend="fallback",
    framework_args=(),
):
    """Run one independent fixed-distance set and write episode artifacts."""
    started = time.monotonic()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_fixed_distance_specs(distance, episodes, seed)
    args = _parse_framework_args(framework_args)
    args.task = "rotunbot_sru_visual_corridor_v1"
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = min(max(1, int(num_envs)), int(episodes))
    env_cfg.commands.v1_goal_curriculum_enabled = False
    env_cfg.commands.v1_performance_curriculum_enabled = False
    env_cfg.commands.goal_distance = (float(distance), float(distance))
    env_cfg.commands.goal_bearing = (0.0, 0.0)
    env_cfg.noise.add_noise = False
    env_cfg.camera.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.camera.depth_backend = str(depth_backend)
    if env_cfg.camera.depth_backend == "isaacgym":
        env_cfg.enable_camera_sensors_in_headless = True
    train_cfg.seed = int(seed)
    train_cfg.runner.resume = False
    formal_steps = int(
        round(45.0 / (env_cfg.sim.dt * env_cfg.control.decimation))
    )
    max_steps = formal_steps if max_steps is None else int(max_steps)

    env = None
    logger = EpisodeLogger(output_dir)
    try:
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
        runner, _ = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
        )
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        actor_critic = runner.alg.actor_critic
        obs, _ = env.reset()
        reset_recurrent_hidden(
            actor_critic,
            torch.ones(env.num_envs, dtype=torch.bool, device=env.device),
        )
        held_actions = torch.zeros(env.num_envs, 2, device=env.device)
        held_raw = torch.zeros_like(held_actions)
        held_requested = torch.zeros_like(held_actions)
        active = {}
        next_spec = 0
        for env_index in range(env.num_envs):
            spec = specs[next_spec]
            next_spec += 1
            _assign_goal(env, env_index, spec)
            active[env_index] = _episode_state(env, spec, env_index)
        env.compute_observations()
        obs = env.get_observations()
        records = []
        trajectories = []
        with torch.inference_mode():
            while len(records) < episodes:
                if env.common_step_counter % env.upper_level_command_interval_steps == 0:
                    actions = policy(obs).clamp(-1.0, 1.0)
                    held_actions.copy_(actions)
                    held_raw.copy_(_raw_velocity_command(
                        actions,
                        env.cfg.commands.max_forward_speed,
                        env.cfg.commands.max_yaw_rate,
                    ))
                    # This is the command before V62's existing transition,
                    # governor, and feasible-projection layers.
                    from legged_gym.navigation.direct_velocity import normalized_action_to_velocity_command
                    held_requested.copy_(normalized_action_to_velocity_command(
                        actions,
                        env.cfg.commands.max_forward_speed,
                        env.cfg.commands.max_yaw_rate,
                        env.cfg.commands.minimum_turn_radius,
                        env.cfg.commands.feasible_envelope_fraction,
                        preserve_curvature_when_saturating=bool(
                            getattr(env.cfg.commands, "preserve_curvature_when_saturating", False)
                        ),
                    ))
                obs, _, _, dones, _ = env.step(held_actions)
                done_mask = dones.flatten().bool()
                for env_index in list(active):
                    state = active[env_index]
                    state["steps"] += 1
                    forced_timeout = state["steps"] >= max_steps and not bool(done_mask[env_index].item())
                    post = {
                        "applied_command": env.applied_feasible_command[env_index],
                        "actual_velocity": torch.stack((env.tracking_lin_vel[env_index, 0], env.tracking_ang_vel[env_index, 2])),
                        "position": env.root_states[env_index, :2],
                        "command_target": env.command_targets[env_index],
                        "goal_xy_robot": env._goal_xy_robot()[env_index],
                        "goal_distance": env.goal_dist[env_index],
                        "transition_active": env.transition_active[env_index],
                        "transition_state": env.transition_state[env_index],
                        "goal_recovery_active": env.goal_recovery_active[env_index],
                        "success": env.success_buf[env_index],
                        "collision": env.step_collision_buf[env_index],
                        "timeout": env.time_out_buf[env_index],
                    }
                    terminal = None
                    if bool(done_mask[env_index].item()):
                        terminal = {
                            "applied_command": env.terminal_applied_feasible_command[env_index],
                            "actual_velocity": env.terminal_tracking_velocity[env_index],
                            "position": env.terminal_position[env_index],
                            "command_target": env.terminal_command_target[env_index],
                            "goal_xy_robot": env.terminal_goal_xy_robot[env_index],
                            "goal_distance": env.terminal_goal_distance[env_index],
                            "transition_active": env.terminal_transition_active[env_index],
                            "transition_state": env.terminal_transition_state[env_index],
                            "goal_recovery_active": env.terminal_goal_recovery_active[env_index],
                            "success": env.terminal_success[env_index],
                            "collision": env.terminal_collision[env_index],
                            "timeout": env.terminal_timeout[env_index],
                        }
                    telemetry = select_step_telemetry(
                        bool(done_mask[env_index].item()), post, terminal
                    )
                    applied = telemetry["applied_command"]
                    projected = _project_applied(env, applied.reshape(1, 2))[0]
                    state["diagnostics"].record(
                        held_raw[env_index].detach().cpu().tolist(),
                        held_requested[env_index].detach().cpu().tolist(),
                        applied.detach().cpu().tolist(),
                        projected.detach().cpu().tolist(),
                        bool(telemetry["transition_active"].item()),
                    )
                    position = telemetry["position"].detach().cpu().numpy()
                    state["path_length_m"] += float(
                        ((position - state["previous_position"]) ** 2).sum() ** 0.5
                    )
                    state["previous_position"] = position.copy()
                    actual = telemetry["actual_velocity"]
                    state["forward_velocity_sum"] += float(actual[0].item())
                    state["absolute_yaw_velocity_sum"] += abs(float(actual[1].item()))
                    state["reverse_steps"] += int(float(actual[0].item()) < -3.0e-6)
                    state["last_goal_distance_m"] = float(telemetry["goal_distance"].item())
                    state["trajectory"].append(_trajectory_row(
                        episode_id=state["spec"]["episode_id"],
                        step=state["steps"],
                        distance_m=state["spec"]["distance_m"],
                        position=position,
                        goal_distance=state["last_goal_distance_m"],
                        raw_v=held_raw[env_index, 0].item(),
                        raw_w=held_raw[env_index, 1].item(),
                        requested_v=held_requested[env_index, 0].item(),
                        requested_w=held_requested[env_index, 1].item(),
                        applied_v=applied[0].item(),
                        applied_w=applied[1].item(),
                        actual_v=actual[0].item(),
                        actual_w=actual[1].item(),
                        dt=env.dt,
                    ))
                    if not bool(done_mask[env_index].item()) and not forced_timeout:
                        continue
                    record = _record(env, env_index, state, telemetry, forced_timeout)
                    logger.write_episode(record)
                    records.append(record)
                    trajectories.extend(state["trajectory"])
                    if not record["success"]:
                        write_failure_artifacts(output_dir, record, state["trajectory"])
                    del active[env_index]
                    held_actions[env_index] = 0.0
                    if forced_timeout:
                        env.reset_idx(torch.as_tensor([env_index], device=env.device, dtype=torch.long))
                    if next_spec < len(specs):
                        spec = specs[next_spec]
                        next_spec += 1
                        _assign_goal(env, env_index, spec)
                        active[env_index] = _episode_state(env, spec, env_index)
                    if len(records) >= episodes:
                        break
                reset_recurrent_hidden(actor_critic, done_mask)
                if next_spec > 0 and not active:
                    break
                env.compute_observations()
                obs = env.get_observations()
        logger.write_trajectory(trajectories)
        summary = summarize_v1_episodes(records)
        summary.update({
            "distance_m": float(distance),
            "seed": int(seed),
            "max_steps": int(max_steps),
            "checkpoint": str(Path(checkpoint).resolve()),
            "depth_backend_requested": env.depth_backend_requested,
            "depth_backend_actual": env.depth_backend_actual,
            "wall_clock_seconds": time.monotonic() - started,
            "artifact_root": str(output_dir),
        })
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary
    finally:
        close_environment(env)


def main(argv=None):
    stage_args, remaining = _parse_args(sys.argv[1:] if argv is None else argv)
    targets = evaluation_targets(
        stage_args.current_distance,
        stage_args.next_distance,
        stage_args.episodes,
    )
    root = Path(stage_args.output_dir).resolve()
    summaries = {}
    for name, distance, episodes in targets:
        summaries[name] = evaluate_distance(
            stage_args.checkpoint,
            distance,
            episodes,
            stage_args.seed,
            root / name,
            num_envs=stage_args.num_envs,
            max_steps=stage_args.max_steps,
            depth_backend=stage_args.depth_backend,
            framework_args=remaining,
        )
    result = {"targets": summaries, "seed": stage_args.seed}
    if "current" in summaries and "next" in summaries:
        result["curriculum_gate"] = curriculum_gate(
            summaries["current"], summaries["next"]
        )
    (root / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
