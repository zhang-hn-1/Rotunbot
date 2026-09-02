"""D1 Random Obstacle Oracle Feasibility Gate.

The teacher follows an 8-connected oracle path with lookahead pure pursuit.
Commands go through the frozen V62 velocity port. No SRU is loaded or trained.
Each map is a separate env built from explicit corridor wall segments
(arena boundary + box obstacle perimeters) so the actor topology is rebuilt
per map before make_env. Real IsaacGym IMAGE_DEPTH is asserted.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np

from legged_gym.navigation.random_obstacle_navigation import (
    ARENA_MAX,
    ARENA_MIN,
    DEFAULT_SAFETY_MARGIN_M,
    RandomObstacleSplitConfig,
    group_scenarios_by_topology,
    sample_random_obstacle_scenario,
    scenario_to_metadata,
    validate_random_scenario,
)
from legged_gym.navigation.v62_turn_reachability import (
    point_aabb_raw_distance,
    polyline_clearance,
)


def _perimeter_segments(center_xy, size_xy, yaw_rad):
    """Return the four thin-wall segments outlining a rotated box (OBB)."""
    cx, cy = center_xy
    length, width = size_xy
    cos = math.cos(yaw_rad)
    sin = math.sin(yaw_rad)
    ux, uy = cos * length / 2.0, sin * length / 2.0
    vx, vy = -sin * width / 2.0, cos * width / 2.0
    corners = [
        (cx - ux - vx, cy - uy - vy),
        (cx + ux - vx, cy + uy - vy),
        (cx + ux + vx, cy + uy + vy),
        (cx - ux + vx, cy - uy + vy),
    ]
    return [
        (corners[index], corners[(index + 1) % 4])
        for index in range(4)
    ]


def scenario_wall_segments(scenario):
    """Arena boundary walls plus obstacle perimeters, as explicit segments."""
    segments = [
        ((ARENA_MIN, ARENA_MIN), (ARENA_MAX, ARENA_MIN)),
        ((ARENA_MAX, ARENA_MIN), (ARENA_MAX, ARENA_MAX)),
        ((ARENA_MAX, ARENA_MAX), (ARENA_MIN, ARENA_MAX)),
        ((ARENA_MIN, ARENA_MAX), (ARENA_MIN, ARENA_MIN)),
    ]
    for box in scenario.obstacles:
        segments.extend(_perimeter_segments(box.center_xy, box.size_xy, box.yaw_rad))
    return tuple(segments)


def scenario_physics_aabbs(scenario):
    """Uninflated AABB list for collision/clearance (conservative w.r.t OBB)."""
    half = 0.025
    aabbs = [
        ((3.0, 0.0), (3.0, half)),
        ((6.0, 3.0), (half, 3.0)),
        ((3.0, 6.0), (3.0, half)),
        ((0.0, 3.0), (half, 3.0)),
    ]
    for box in scenario.obstacles:
        aabbs.append(box.to_aabb())
    return tuple(aabbs)


def _close(env):
    if env is None:
        return
    if getattr(env, "viewer", None) is not None:
        env.gym.destroy_viewer(env.viewer)
    if getattr(env, "sim", None) is not None:
        env.gym.destroy_sim(env.sim)


def _yaw_from_quat(env, index=0):
    quat = env.root_states[index, 3:7]
    return float(
        env._yaw_from_quaternion(quat.reshape(1, 4)).item()
    )


def _robot_position(env, index=0):
    return (env.root_states[index, :2] - env.env_origins[index, :2]).detach().cpu().numpy().copy()


def _set_robot_pose(env, scenario, torch):
    device = env.device
    index = 0
    xy = np.asarray(scenario.spawn_xy, dtype=np.float64)
    yaw = float(scenario.initial_yaw_rad)
    env.root_states[index, 0] = env.env_origins[index, 0].item() + float(xy[0])
    env.root_states[index, 1] = env.env_origins[index, 1].item() + float(xy[1])
    half = yaw / 2.0
    env.root_states[index, 3] = 0.0
    env.root_states[index, 4] = 0.0
    env.root_states[index, 5] = math.sin(half)
    env.root_states[index, 6] = math.cos(half)
    env.root_states[index, 7:13] = 0.0
    from isaacgym import gymtorch

    actor_id = env._robot_actor_ids(torch.zeros(1, dtype=torch.long, device=device))[0]
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env._all_root_states),
        gymtorch.unwrap_tensor(actor_id.reshape(1).to(torch.int32)),
        1,
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)


def _set_final_goal(env, scenario, torch):
    goal = np.asarray(scenario.goal_xy, dtype=np.float64)
    env.global_goal_xy_world[0, 0] = env.env_origins[0, 0].item() + float(goal[0])
    env.global_goal_xy_world[0, 1] = env.env_origins[0, 1].item() + float(goal[1])
    env.goal_dist[0] = torch.linalg.vector_norm(
        env.global_goal_xy_world[0] - env.root_states[0, :2]
    )
    env.terminal_goal_distance[0] = env.goal_dist[0].clone()
    env.previous_goal_distance[0] = env.goal_dist[0].clone()
    env.goal_reached_buf[0] = False
    env.success_buf[0] = False


def _path_points(scenario):
    return np.asarray(scenario.oracle_path, dtype=np.float64)


def _lookahead_waypoint(scenario, position_xy, lookahead_m):
    points = _path_points(scenario)
    if points.shape[0] < 2:
        return np.asarray(scenario.goal_xy, dtype=np.float64)
    differences = points[1:] - points[:-1]
    lengths = np.linalg.norm(differences, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    distance = np.linalg.norm(points - np.asarray(position_xy, dtype=np.float64), axis=1)
    start_index = int(np.argmin(distance))
    target_distance = min(float(cumulative[start_index]) + float(lookahead_m), total)
    index = int(np.searchsorted(cumulative, target_distance))
    if index <= 0:
        return points[0].copy()
    if index >= len(points):
        return points[-1].copy()
    fraction = (target_distance - cumulative[index - 1]) / max(lengths[index - 1], 1.0e-9)
    return (points[index - 1] + fraction * differences[index - 1]).copy()


def _goal_xy_robot(pose, waypoint):
    x, y, yaw = pose
    dx = float(waypoint[0]) - x
    dy = float(waypoint[1]) - y
    cos, sin = math.cos(yaw), math.sin(yaw)
    return np.asarray((cos * dx + sin * dy, -sin * dx + cos * dy), dtype=np.float64)


def run_episode(env, scenario, runtime, teacher_cfg, torch, lookahead_m):
    """Run one teacher rollout; returns summary and trajectory rows."""
    from legged_gym.navigation.v1_velocity_teacher import teacher_velocity_diagnostics

    env.reset()
    require_real_depth(env)
    _set_robot_pose(env, scenario, torch)
    _set_final_goal(env, scenario, torch)
    position = _robot_position(env)
    yaw = _yaw_from_quat(env)
    actual = torch.stack((env.tracking_lin_vel[:, 0], env.tracking_ang_vel[:, 2]), dim=1)
    rows = []
    previous_position = position.copy()
    path_length = 0.0
    min_clearance = float("inf")
    min_physics_clearance = float("inf")
    min_planning_clearance = float("inf")
    steps = 0
    done = False
    timeout = False
    max_steps = int(runtime["max_steps"])
    action = torch.zeros(1, 2, device=env.device)
    with torch.inference_mode():
        while not done and steps < max_steps:
            first_tick = steps == 0
            if first_tick or env.common_step_counter % int(env.upper_level_command_interval_steps) == 0:
                pose = (float(position[0]), float(position[1]), float(yaw))
                waypoint = _lookahead_waypoint(scenario, pose[:2], lookahead_m)
                goal_robot = _goal_xy_robot(pose, waypoint)
                goal_tensor = torch.as_tensor(goal_robot.reshape(1, 2), dtype=env.root_states.dtype, device=env.device)
                actual = torch.stack((env.tracking_lin_vel[:, 0], env.tracking_ang_vel[:, 2]), dim=1)
                teacher = teacher_velocity_diagnostics(
                    goal_tensor,
                    actual,
                    env.obstacle_clearance,
                    teacher_cfg,
                )
                action = teacher["applied_command"].clone()
                action[:, 0] = action[:, 0] / teacher_cfg.max_forward_speed
                action[:, 1] = action[:, 1] / teacher_cfg.max_yaw_rate
                action.clamp_(-1.0, 1.0)
            _, _, _, dones, _ = env.step(action)
            steps += 1
            done = bool(dones.flatten()[0].item())
            position_now = (
                env.terminal_position[0, :2].detach().cpu().numpy().copy()
                if done else _robot_position(env)
            )
            yaw = _yaw_from_quat(env)
            actual_v = float(
                env.terminal_tracking_velocity[0, 0].item()
                if done else env.tracking_lin_vel[0, 0].item()
            )
            actual_w = float(
                env.terminal_tracking_velocity[0, 1].item()
                if done else env.tracking_ang_vel[0, 2].item()
            )
            path_length += float(np.linalg.norm(position_now - previous_position))
            previous_position = position_now
            clearance = float(env.obstacle_clearance[0].item())
            min_clearance = min(min_clearance, clearance)
            point = tuple(float(value) for value in position_now)
            min_physics_clearance = min(min_physics_clearance, min(
                point_aabb_raw_distance(point, center, half) - float(runtime["robot_radius_m"])
                for center, half in scenario.raw_physics_aabbs()
            ))
            min_planning_clearance = min(min_planning_clearance, scenario.point_planning_clearance(point))
            rows.append({
                "step": steps,
                "time_s": steps * float(runtime["primitive_dt"]),
                "x_m": float(position_now[0]),
                "y_m": float(position_now[1]),
                "yaw_rad": yaw,
                "v_actual": actual_v,
                "w_actual": actual_w,
                "raw_clearance_m": clearance,
                "physics_free_space_clearance_m": clearance - float(runtime["robot_radius_m"]),
            })
            if steps >= max_steps:
                timeout = True
                break
    success = bool(env.terminal_success[0].item()) if done else False
    collision = bool(env.terminal_collision[0].item()) if done else False
    final_distance = float(env.terminal_goal_distance[0].item()) if done else float(env.goal_dist[0].item())
    return {
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "steps": steps,
        "path_length_m": path_length,
        "min_clearance_m": min_clearance if math.isfinite(min_clearance) else None,
        "min_physics_free_space_clearance_m": min_physics_clearance if math.isfinite(min_physics_clearance) else None,
        "min_planning_clearance_m": min_planning_clearance if math.isfinite(min_planning_clearance) else None,
        "final_goal_distance_m": final_distance,
        "terminal_x_m": float(position_now[0]),
        "terminal_y_m": float(position_now[1]),
    }, rows


def require_real_depth(env):
    actual = getattr(env, "depth_backend_actual", None)
    if actual != "isaacgym":
        raise RuntimeError("random-obstacle D1 requires real IsaacGym depth; got %s" % actual)


def _framework_args(remaining):
    from legged_gym.utils import get_args

    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + list(remaining)
    try:
        return get_args()
    finally:
        os.sys.argv = original


def _build_env_cfg(task_name, args, env_cfg, scenario, robot_radius_m, max_steps, torch):
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = max(
        40.0, (int(max_steps) + 10) * float(env_cfg.sim.dt * env_cfg.control.decimation)
    )
    env_cfg.noise.add_noise = False
    env_cfg.camera.add_noise = False
    env_cfg.camera.depth_backend = "isaacgym"
    env_cfg.enable_camera_sensors_in_headless = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.commands.random_start_yaw = False
    env_cfg.commands.v1_goal_curriculum_enabled = False
    env_cfg.commands.v1_performance_curriculum_enabled = False
    env_cfg.commands.resample_commands = False
    env_cfg.commands.random_start_yaw = False
    env_cfg.commands.smooth_profile_fraction = 0.0
    env_cfg.commands.random_walk_profile_fraction = 0.0
    env_cfg.commands.independent_smooth_profile_fraction = 0.0
    env_cfg.corridor_wall_segments = ()
    env_cfg.corridor_explicit_wall_segments = scenario_wall_segments(scenario)
    env_cfg.direct_obstacle_aabbs = scenario_physics_aabbs(scenario)
    return env_cfg


def run_scenario_evaluation(
    scenario,
    output_dir,
    run_id,
    max_steps,
    lookahead_m,
    robot_radius_m,
    seed,
    remaining,
):
    """Build one env and run one teacher episode."""
    import isaacgym  # noqa: F401
    import torch
    import legged_gym.envs  # noqa: F401
    from legged_gym.navigation.v1_velocity_teacher import (
        V1VelocityTeacherConfig,
        teacher_velocity_diagnostics,
    )
    from legged_gym.utils import task_registry

    args = _framework_args(remaining)
    args.task = "rotunbot_sru_visual_corridor_v1"
    args.seed = int(seed)
    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg = _build_env_cfg(args.task, args, env_cfg, scenario, robot_radius_m, max_steps, torch)
    env_cfg.seed = int(seed)
    teacher_cfg = V1VelocityTeacherConfig(
        max_forward_speed=float(env_cfg.commands.max_forward_speed),
        max_yaw_rate=float(env_cfg.commands.max_yaw_rate),
        minimum_turn_radius=float(env_cfg.commands.minimum_turn_radius),
        feasible_envelope_fraction=float(env_cfg.commands.feasible_envelope_fraction),
        goal_radius=float(env_cfg.commands.goal_radius),
    )
    runtime = {
        "primitive_dt": float(env_cfg.sim.dt * env_cfg.control.decimation),
        "max_steps": int(max_steps),
        "robot_radius_m": float(robot_radius_m),
    }
    env = None
    try:
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
        # depth_backend_actual becomes isaacgym after the first reset/step
        # capture; run_episode performs the fail-closed check at that point.
        summary, rows = run_episode(env, scenario, runtime, teacher_cfg, torch, lookahead_m)
    finally:
        _close(env)
    summary.update({
        "run_id": run_id,
        "map_seed": int(scenario.map_seed),
        "attempt_index": int(scenario.attempt_index),
        "obstacle_count": int(scenario.obstacle_count),
        "spawn_xy": list(scenario.spawn_xy),
        "initial_yaw_rad": float(scenario.initial_yaw_rad),
        "goal_xy": list(scenario.goal_xy),
        "oracle_path_length_m": float(scenario.oracle_path_length_m),
        "obstacles": [scenario_to_metadata(scenario)["obstacles"]],
    })
    spl = (
        float(scenario.oracle_path_length_m) / max(scenario.oracle_path_length_m, float(summary["path_length_m"]))
        if summary["success"] else 0.0
    )
    summary["spl"] = spl
    episode_dir = output_dir / ("map_%d" % scenario.map_seed)
    episode_dir.mkdir(parents=True, exist_ok=True)
    with (episode_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        import csv
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (episode_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return summary


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maps", type=int, default=20)
    parser.add_argument("--obstacle-counts", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-steps", type=int, default=2250)
    parser.add_argument("--lookahead-m", type=float, default=1.0)
    parser.add_argument("--robot-radius", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--map-seed-file", default=None)
    parser.add_argument("--headless", action="store_true")
    parsed, remaining = parser.parse_known_args(argv)
    return parsed, remaining


def main(argv=None):
    stage, remaining = _parse_args(argv)
    split_cfg = RandomObstacleSplitConfig()
    robot_radius = stage.robot_radius or 0.4
    output = Path(stage.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    obstacle_counts = (
        [int(value) for value in stage.obstacle_counts.split(",")]
        if stage.obstacle_counts
        else [2] * (int(stage.maps) // 2) + [3] * (int(stage.maps) - int(stage.maps) // 2)
    )
    if len(obstacle_counts) != int(stage.maps):
        raise ValueError("obstacle-counts length must equal maps")
    map_seeds = []
    if stage.map_seed_file:
        with open(stage.map_seed_file, "r", encoding="utf-8") as handle:
            map_seeds = [int(line.strip()) for line in handle if line.strip()]
    else:
        rng = random.Random(int(stage.seed))
        pool = list(range(split_cfg.test_seed_range[0], split_cfg.test_seed_range[1] + 1))
        rng.shuffle(pool)
        map_seeds = pool[: int(stage.maps)]
    scenarios = []
    for count, map_seed in zip(obstacle_counts, map_seeds):
        scenario = sample_random_obstacle_scenario(map_seed, count, robot_radius_m=robot_radius, split_name=stage.split)
        validate_random_scenario(scenario)
        scenarios.append(scenario)
    summaries = []
    for index, scenario in enumerate(scenarios):
        summary = run_scenario_evaluation(
            scenario,
            output,
            run_id=index,
            max_steps=stage.max_steps,
            lookahead_m=stage.lookahead_m,
            robot_radius_m=robot_radius,
            seed=int(stage.seed) + index,
            remaining=remaining,
        )
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    counts = sorted({int(item["obstacle_count"]) for item in summaries})
    grouped = {}
    for count in counts:
        rows = [item for item in summaries if int(item["obstacle_count"]) == count]
        grouped[count] = {
            "maps": len(rows),
            "success": int(sum(bool(item["success"]) for item in rows)),
            "collision": int(sum(bool(item["collision"]) for item in rows)),
            "timeout": int(sum(bool(item["timeout"]) for item in rows)),
            "success_rate": float(np.mean([bool(item["success"]) for item in rows])),
            "collision_rate": float(np.mean([bool(item["collision"]) for item in rows])),
            "mean_spl": float(np.mean([float(item["spl"]) for item in rows])),
        }
    overall_success = int(sum(bool(item["success"]) for item in summaries))
    overall_collision = int(sum(bool(item["collision"]) for item in summaries))
    payload = {
        "stage": "D1_RANDOM_OBSTACLE_ORACLE_TEACHER",
        "maps": len(summaries),
        "split": stage.split,
        "max_steps": int(stage.max_steps),
        "lookahead_m": float(stage.lookahead_m),
        "robot_radius_m": float(robot_radius),
        "overall_success_rate": float(overall_success / max(len(summaries), 1)),
        "overall_collision_rate": float(overall_collision / max(len(summaries), 1)),
        "overall_success": overall_success,
        "overall_collision": overall_collision,
        "by_obstacle_count": grouped,
        "episodes": summaries,
    }
    (output / "D1_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return payload


if __name__ == "__main__":
    main()
