"""Evaluate a recurrent V1 visual policy on the symmetric T-junction MVP."""

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path


def _scenario(value):
    value = str(value).upper()
    if value not in ("T_LEFT", "T_RIGHT"):
        raise ValueError("scenario must be T_LEFT or T_RIGHT")
    return value


def _expected_branch(scenario):
    return "LEFT" if _scenario(scenario) == "T_LEFT" else "RIGHT"


def goal_for_observation(normal_goal, swapped_goal, mode="normal"):
    """Return the goal supplied to the actor while preserving terminal goal state."""
    mode = str(mode).lower()
    if mode == "normal":
        goal = normal_goal
    elif mode == "zero":
        goal = (0.0, 0.0)
    elif mode == "swapped":
        goal = swapped_goal
    else:
        raise ValueError("goal mode must be normal, zero, or swapped")
    values = tuple(float(value) for value in goal)
    if len(values) != 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("goal must contain two finite values")
    return values


def make_t_student_episode_record(
    *,
    scenario,
    episode_id,
    pair_id,
    seed,
    goal,
    initial_pose,
    initial_yaw,
    horizon,
    episode_steps,
    success,
    collision,
    timeout,
    branch_prediction,
    wrong_turn,
    turn_completion,
    exit_reached,
    failure_trace=None,
    depth_backend_actual="isaacgym",
    goal_mode="normal",
):
    """Create one auditable student release record."""
    scenario = _scenario(scenario)
    if depth_backend_actual != "isaacgym":
        raise RuntimeError("T-junction student requires real Isaac Gym IMAGE_DEPTH")
    expected = _expected_branch(scenario)
    mode = str(goal_mode).lower()
    if mode not in ("normal", "zero", "swapped"):
        raise ValueError("goal mode must be normal, zero, or swapped")
    trace = list(failure_trace or ())
    if not bool(success) and not trace:
        trace.append({
            "terminal": {
                "success": bool(success),
                "collision": bool(collision),
                "timeout": bool(timeout),
                "wrong_turn": bool(wrong_turn),
            }
        })
    return {
        "episode_id": episode_id,
        "pair_id": str(pair_id),
        "scenario": scenario,
        "policy_role": "student",
        "seed": int(seed),
        "goal": tuple(float(value) for value in goal),
        "goal_mode": mode,
        "initial_pose": tuple(float(value) for value in initial_pose),
        "initial_yaw": float(initial_yaw),
        "horizon": int(horizon),
        "episode_steps": int(episode_steps),
        "success": bool(success),
        "collision": bool(collision),
        "timeout": bool(timeout),
        "expected_branch": expected,
        "branch_prediction": str(branch_prediction).upper(),
        "wrong_turn": bool(wrong_turn),
        "turn_completion": bool(turn_completion),
        "exit_reached": bool(exit_reached),
        "exit": bool(exit_reached),
        "depth_backend_actual": depth_backend_actual,
        "failure_trace": trace,
    }


def build_counterfactual_pairs(records):
    """Build exact left/right pairs from records sharing a pair_id.

    Each record must occur exactly once in one pair.  Pair metadata is checked
    when present, and pair IDs themselves must not be reused ambiguously.
    """
    records = list(records)
    if not records or len(records) % 2:
        raise ValueError("counterfactual records must contain complete pairs")
    grouped = {}
    by_episode = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("counterfactual records must be mappings")
        episode_id = record.get("episode_id")
        pair_id = record.get("pair_id")
        if episode_id is None or pair_id is None:
            raise ValueError("each counterfactual record needs episode_id and pair_id")
        if episode_id in by_episode:
            raise ValueError("episode_id values must be unique")
        by_episode[episode_id] = record
        grouped.setdefault(pair_id, []).append(record)
    pairs = []
    metadata_fields = ("seed", "initial_pose", "initial_yaw", "horizon")
    for pair_id, pair_records in grouped.items():
        if len(pair_records) != 2:
            raise ValueError("each pair_id must identify exactly two records")
        scenarios = {_scenario(row.get("scenario")) for row in pair_records}
        if scenarios != {"T_LEFT", "T_RIGHT"}:
            raise ValueError("each counterfactual pair needs one T_LEFT and one T_RIGHT")
        for field in metadata_fields:
            if not all(field in row for row in pair_records):
                raise ValueError("paired records must declare %s" % field)
            if pair_records[0][field] != pair_records[1][field]:
                raise ValueError("paired records must share %s" % field)
        left = next(row for row in pair_records if _scenario(row["scenario"]) == "T_LEFT")
        right = next(row for row in pair_records if _scenario(row["scenario"]) == "T_RIGHT")
        pairs.append((left["episode_id"], right["episode_id"]))
    if len({episode_id for pair in pairs for episode_id in pair}) != len(records):
        raise ValueError("counterfactual pairs must cover each record exactly once")
    return pairs


def _commit_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _close_environment(env):
    if env is None:
        return
    if getattr(env, "viewer", None) is not None:
        env.gym.destroy_viewer(env.viewer)
    if getattr(env, "sim", None) is not None:
        env.gym.destroy_sim(env.sim)


def _parse_framework_args(remaining):
    from legged_gym.utils import get_args

    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + list(remaining)
    try:
        return get_args()
    finally:
        os.sys.argv = original


def _local_pose(env, torch):
    position = env.root_states[0, :2] - env.env_origins[0, :2]
    yaw = env._yaw_from_quaternion(env.root_states[0:1, 3:7])[0]
    return position, yaw


def _assign_terminal_goal(env, geometry, torch):
    goal = torch.as_tensor(
        geometry.scenario.goal_xy, dtype=env.root_states.dtype, device=env.device
    ) + env.env_origins[0, :2]
    env.global_goal_xy_world[0] = goal
    distance = torch.linalg.vector_norm(goal - env.root_states[0, :2])
    env.goal_dist[0] = distance
    env.terminal_goal_distance[0] = distance
    env.previous_goal_distance[0] = distance
    env.goal_reached_buf[0] = False
    env.success_buf[0] = False


def _install_goal_mode(env, geometry, mode, torch):
    from legged_gym.scripts.collect_sru_visual_t_junction_teacher import install_observation_goal

    normal = tuple(float(value) for value in geometry.scenario.goal_xy)
    swapped_geometry = geometry._swapped_geometry if hasattr(geometry, "_swapped_geometry") else None
    swapped = tuple(float(value) for value in getattr(swapped_geometry, "scenario", geometry).goal_xy)
    local_goal = goal_for_observation(normal, swapped, mode)
    world = torch.as_tensor(local_goal, dtype=env.root_states.dtype, device=env.device)
    return install_observation_goal(env, env.env_origins[0, :2] + world.reshape(1, 2))


def _record_occupancy(occupancies, local_xy, classify_t_branch):
    branch = classify_t_branch(local_xy)
    return tuple(occupancies) + (branch,)


def _evaluate_scene(env, policy, actor_critic, scene, geometry, opposite_geometry,
                    episodes, seed, max_steps, goal_mode, torch):
    from legged_gym.navigation.v1_t_junction import classify_t_branch
    from legged_gym.navigation.v1_waypoint_manager import V1WaypointManager
    from legged_gym.navigation.direct_velocity import normalized_action_to_velocity_command
    from legged_gym.scripts.evaluate_sru_direct_velocity import _raw_velocity_command
    from legged_gym.scripts.eval_sru_visual_corridor_v1 import reset_recurrent_hidden
    from legged_gym.scripts.collect_sru_visual_t_junction_teacher import (
        classify_t_episode_progress,
        terminal_local_xy,
    )

    if getattr(env, "depth_backend_actual", None) != "isaacgym":
        raise RuntimeError("T-junction student requires env.depth_backend_actual=isaacgym")
    env.reset()
    _assign_terminal_goal(env, geometry, torch)
    reset_recurrent_hidden(actor_critic, torch.ones(1, dtype=torch.bool, device=env.device))
    from legged_gym.scripts.collect_sru_visual_t_junction_teacher import t_navigation_waypoints
    waypoints = t_navigation_waypoints(geometry)
    manager = V1WaypointManager(waypoints, reach_radius=geometry.reach_radius_m)
    held_action = torch.zeros(1, 2, device=env.device)
    held_raw = torch.zeros(1, 2, device=env.device)
    held_requested = torch.zeros(1, 2, device=env.device)
    records, trajectories = [], []
    for episode_index in range(int(episodes)):
        if episode_index:
            env.reset_idx(torch.tensor([0], dtype=torch.long, device=env.device))
            _assign_terminal_goal(env, geometry, torch)
            manager.reset()
            reset_recurrent_hidden(actor_critic, torch.ones(1, dtype=torch.bool, device=env.device))
        position, yaw = _local_pose(env, torch)
        initial_pose = (float(position[0].item()), float(position[1].item()), float(env.root_states[0, 2].item()))
        initial_yaw = float(yaw.item())
        pair_id = "pair-%02d" % episode_index
        observed = []
        failure_trace = []
        path_length = 0.0
        previous = position.detach().cpu().clone()
        min_clearance = float("inf")
        v_sum = 0.0
        abs_w_sum = 0.0
        steps = 0
        done = False
        while not done and steps < int(max_steps):
            if env.common_step_counter % env.upper_level_command_interval_steps == 0:
                position, yaw = _local_pose(env, torch)
                pose = (float(position[0].item()), float(position[1].item()), float(yaw.item()))
                manager.update(pose)
                waypoint = manager.get_current_waypoint()
                waypoint_world = env.env_origins[0, :2] + torch.as_tensor(
                    waypoint, dtype=env.root_states.dtype, device=env.device
                ).reshape(1, 2)
                if goal_mode == "normal":
                    env.set_observation_goal_world(waypoint_world)
                    env.compute_observations()
                elif goal_mode == "zero":
                    env.set_observation_goal_world(torch.zeros_like(waypoint_world))
                    env.compute_observations()
                elif goal_mode == "swapped":
                    swapped_waypoint = opposite_geometry.waypoints[manager.current_index]
                    env.set_observation_goal_world(
                        env.env_origins[0, :2] + torch.as_tensor(
                            swapped_waypoint, dtype=env.root_states.dtype, device=env.device
                        ).reshape(1, 2)
                    )
                    env.compute_observations()
                else:
                    raise ValueError("goal mode must be normal, zero, or swapped")
                action = policy(env.get_observations()).clamp(-1.0, 1.0)
                held_action.copy_(action)
                held_raw.copy_(_raw_velocity_command(
                    action, env.cfg.commands.max_forward_speed, env.cfg.commands.max_yaw_rate
                ))
                held_requested.copy_(normalized_action_to_velocity_command(
                    action,
                    env.cfg.commands.max_forward_speed,
                    env.cfg.commands.max_yaw_rate,
                    env.cfg.commands.minimum_turn_radius,
                    env.cfg.commands.feasible_envelope_fraction,
                    preserve_curvature_when_saturating=bool(
                        getattr(env.cfg.commands, "preserve_curvature_when_saturating", False)
                    ),
                ))
            before = env.root_states[0, :2].detach().cpu().clone()
            _, _, _, dones, _ = env.step(held_action)
            steps += 1
            done = bool(dones.flatten()[0].item())
            current = terminal_local_xy(env) if done else tuple(
                (env.root_states[0, :2] - env.env_origins[0, :2]).detach().cpu().tolist()
            )
            current_tensor = torch.as_tensor(current, dtype=torch.float32)
            path_length += float(torch.linalg.vector_norm(current_tensor - previous).item())
            previous = current_tensor
            observed = list(_record_occupancy(observed, current, classify_t_branch))
            if any(branch in ("LEFT", "RIGHT") and branch != _expected_branch(scene) for branch in observed):
                wrong_turn_latched = True
            else:
                wrong_turn_latched = False
            actual_v = float((env.terminal_tracking_velocity[0, 0] if done else env.tracking_lin_vel[0, 0]).item())
            actual_w = float((env.terminal_tracking_velocity[0, 1] if done else env.tracking_ang_vel[0, 2]).item())
            v_sum += actual_v
            abs_w_sum += abs(actual_w)
            min_clearance = min(min_clearance, float(env.obstacle_clearance[0].item()))
            trajectories.append({
                "episode_id": "%s-%02d" % (scene, episode_index),
                "pair_id": pair_id,
                "step": steps,
                "scene": scene,
                "goal_mode": goal_mode,
                "x": float(current[0]), "y": float(current[1]),
                "waypoint_index": int(manager.current_index),
                "raw_v_cmd": float(held_raw[0, 0].item()),
                "raw_w_cmd": float(held_raw[0, 1].item()),
                "requested_v_cmd": float(held_requested[0, 0].item()),
                "requested_w_cmd": float(held_requested[0, 1].item()),
                "v_actual": actual_v, "w_actual": actual_w,
                "clearance_m": min_clearance,
            })
            if not done and steps >= int(max_steps):
                break
        forced_timeout = not done
        local_xy = current
        progress = classify_t_episode_progress(
            scene, terminal_local_xy=local_xy, waypoint_index=manager.current_index,
            exit_reached=bool(env.terminal_success[0].item()) if done else False,
            observed_branches=observed,
        )
        wrong_turn = bool(progress["wrong_turn"] or any(
            branch in ("LEFT", "RIGHT") and branch != _expected_branch(scene)
            for branch in observed
        ))
        success = bool(env.terminal_success[0].item()) if done else False
        collision = bool(env.terminal_collision[0].item()) if done else False
        timeout = bool(forced_timeout or (bool(env.terminal_timeout[0].item()) if done else False))
        if not success:
            failure_trace.append({
                "step": steps,
                "terminal_xy": [float(local_xy[0]), float(local_xy[1])],
                "observed_branches": list(observed),
                "min_clearance_m": min_clearance,
            })
        records.append(make_t_student_episode_record(
            scenario=scene,
            episode_id="%s-%02d" % (scene, episode_index),
            pair_id=pair_id,
            seed=seed,
            goal=geometry.scenario.goal_xy,
            initial_pose=initial_pose,
            initial_yaw=initial_yaw,
            horizon=max_steps,
            episode_steps=steps,
            success=success,
            collision=collision,
            timeout=timeout,
            branch_prediction=progress["branch_prediction"],
            wrong_turn=wrong_turn,
            turn_completion=progress["turn_completed"],
            exit_reached=progress["exit_reached"],
            failure_trace=failure_trace,
            goal_mode=goal_mode,
        ))
    return records, trajectories


def _write_records(output, scene, records, trajectories):
    scene_root = output / scene
    scene_root.mkdir(parents=True, exist_ok=True)
    with (scene_root / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (scene_root / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectories[0]) if trajectories else ["episode_id"])
        writer.writeheader()
        if trajectories:
            writer.writerows(trajectories)
    (scene_root / "failure_traces.json").write_text(
        json.dumps([row for row in records if row["failure_trace"]], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=2250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", required=True)
    parsed, remaining = parser.parse_known_args(argv)
    return parsed, remaining


def main(argv=None):
    stage_args, remaining = _parse_args(argv)
    if stage_args.episodes <= 0 or stage_args.max_steps <= 0:
        raise ValueError("episodes and max-steps must be positive")
    import isaacgym  # noqa: F401
    import torch
    import legged_gym.envs  # noqa: F401
    from legged_gym.navigation.v1_t_junction import build_t_junction_geometry
    from legged_gym.navigation.v1_t_junction_metrics import aggregate_t_gate
    from legged_gym.scripts.eval_sru_visual_l_turn import _close_environment as close_env
    from legged_gym.utils import task_registry

    args = _parse_framework_args(remaining)
    args.task = "rotunbot_sru_visual_corridor_v1"
    args.seed = int(stage_args.seed)
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.seed = int(stage_args.seed)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = float(stage_args.max_steps) * float(
        env_cfg.sim.dt * env_cfg.control.decimation
    )
    env_cfg.enable_camera_sensors_in_headless = True
    env_cfg.camera.depth_backend = "isaacgym"
    env_cfg.camera.add_noise = False
    env_cfg.init_state.random_start_lateral = 0.0
    env_cfg.init_state.random_start_yaw = 0.0
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.v1_goal_curriculum_enabled = False
    env_cfg.commands.v1_performance_curriculum_enabled = False
    output = Path(stage_args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    normal_records, normal_trajectories = [], []
    ablation_payload = {}
    env = None
    for scene in ("T_LEFT", "T_RIGHT"):
        geometry = build_t_junction_geometry(scene)
        opposite = build_t_junction_geometry("T_RIGHT" if scene == "T_LEFT" else "T_LEFT")
        env_cfg.corridor_width_m = geometry.scenario.width_m
        env_cfg.corridor_wall_width_m = geometry.scenario.width_m
        env_cfg.corridor_wall_segments = ()
        env_cfg.corridor_explicit_wall_segments = geometry.wall_segments
        env_cfg.direct_obstacle_aabbs = geometry.obstacle_aabbs
        try:
            env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
            # The camera backend is marked ``unavailable`` until the first
            # reset/step capture; _evaluate_scene checks the actual backend
            # after that capture and fails closed if it is not Isaac Gym.
            train_cfg.runner.resume = False
            runner, _ = task_registry.make_alg_runner(
                env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
            )
            runner.load(str(stage_args.checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=env.device)
            for mode in ("normal", "zero", "swapped"):
                mode_records, mode_trajectories = _evaluate_scene(
                    env, policy, runner.alg.actor_critic, scene, geometry, opposite,
                    stage_args.episodes, stage_args.seed, stage_args.max_steps, mode, torch,
                )
                if mode == "normal":
                    normal_records.extend(mode_records)
                    normal_trajectories.extend(mode_trajectories)
                    _write_records(output, scene, mode_records, mode_trajectories)
                ablation_payload.setdefault(mode, {})[scene] = {
                    "episodes": len(mode_records),
                    "success_rate": sum(bool(row["success"]) for row in mode_records) / max(len(mode_records), 1),
                    "collision_rate": sum(bool(row["collision"]) for row in mode_records) / max(len(mode_records), 1),
                    "timeout_rate": sum(bool(row["timeout"]) for row in mode_records) / max(len(mode_records), 1),
                    "turn_completion_rate": sum(bool(row["turn_completion"]) for row in mode_records) / max(len(mode_records), 1),
                }
        finally:
            close_env(env)
            env = None

    pairs = build_counterfactual_pairs(normal_records)
    gate = aggregate_t_gate(normal_records, pairs=pairs, ablations=ablation_payload)
    payload = {
        "stage": "T_JUNCTION_STUDENT",
        "status": "PASS" if gate["pass"] else "FAIL",
        "commit": _commit_sha(),
        "checkpoint": str(Path(stage_args.checkpoint).resolve()),
        "seed": int(stage_args.seed),
        "episodes_per_scene": int(stage_args.episodes),
        "max_steps": int(stage_args.max_steps),
        "depth_backend_requested": "isaacgym",
        "depth_backend_actual": "isaacgym",
        "pairs": pairs,
        "pair_count": len(pairs),
        "ablations": ablation_payload,
        "gate": gate,
    }
    (output / "t_junction_student_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "paired_counterfactuals.json").write_text(
        json.dumps(pairs, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
