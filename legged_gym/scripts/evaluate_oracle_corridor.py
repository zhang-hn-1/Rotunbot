"""Phase D1 V2 Oracle/Teacher feasibility evaluator.

This is intentionally not an SRU evaluation.  A deterministic geometric
oracle supplies local waypoints, a scripted teacher supplies physical velocity
commands through ``set_command_targets``, and the existing Frozen V62
command-to-actuator path executes them.  The visual corridor environment is
used only to keep the real IsaacGym IMAGE_DEPTH lifecycle and the frozen
observation/actuation stack available; no visual checkpoint is loaded.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import traceback
from pathlib import Path

import numpy as np

from legged_gym.navigation.random_obstacle_navigation import (
    BOUNDARY_WALL_THICKNESS_M,
    EVALUATION_VERSION,
    RandomObstacleConfig,
    RandomObstacleSplitConfig,
    build_seed_inventory,
    config_hash,
    frozen_inventory_hash,
    sample_random_obstacle_scenario,
    scenario_to_metadata,
    validate_random_scenario,
)
from legged_gym.navigation.phase_d_contracts import (
    classify_phase_d_failure,
    require_isaacgym_depth,
    resolve_phase_d_timing,
    terminal_convergence_evidence,
    transition_manager_stall_evidence,
)
from legged_gym.navigation.phase_d_diagnostics import command_loss_breakdown


DEFAULT_MAX_STEPS = 2250
DEFAULT_LOOKAHEAD_M = 0.8
DEFAULT_OUTPUT = "logs/phase_d/d1_v2_progressive_20260902"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _perimeter_segments(center_xy, size_xy, yaw_rad):
    cx, cy = center_xy
    length, width = size_xy
    c, s = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    ux, uy = c * length / 2.0, s * length / 2.0
    vx, vy = -s * width / 2.0, c * width / 2.0
    corners = (
        (cx - ux - vx, cy - uy - vy),
        (cx + ux - vx, cy + uy - vy),
        (cx + ux + vx, cy + uy + vy),
        (cx - ux + vx, cy - uy + vy),
    )
    return tuple((corners[index], corners[(index + 1) % 4]) for index in range(4))


def scenario_wall_segments(scenario):
    """Create real explicit box walls from the scenario's configured bounds."""
    low_x, low_y, high_x, high_y = scenario.bounds_xy
    segments = [
        ((low_x, low_y), (high_x, low_y)),
        ((high_x, low_y), (high_x, high_y)),
        ((high_x, high_y), (low_x, high_y)),
        ((low_x, high_y), (low_x, low_y)),
    ]
    for obstacle in scenario.obstacles:
        segments.extend(_perimeter_segments(obstacle.center_xy, obstacle.size_xy, obstacle.yaw_rad))
    return tuple(segments)


def scenario_physics_aabbs(scenario):
    """Return the conservative AABB proxy matching the wall actor geometry."""
    return tuple(scenario.raw_physics_aabbs())


def _point_aabb_distance(point, center, half):
    point = np.asarray(point, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(half, dtype=np.float64)
    return float(np.linalg.norm(np.maximum(np.abs(point - center) - half, 0.0)))


def _yaw_from_env(env):
    return float(env._yaw_from_quaternion(env.root_states[0:1, 3:7])[0].item())


def _position_from_env(env):
    return (env.root_states[0, :2] - env.env_origins[0, :2]).detach().cpu().numpy().copy()


def _set_robot_pose(env, scenario, torch):
    """Reset only the robot actor to the scenario pose before first rollout step."""
    state = env.root_states[0]
    state[0] = env.env_origins[0, 0] + float(scenario.spawn_xy[0])
    state[1] = env.env_origins[0, 1] + float(scenario.spawn_xy[1])
    half_yaw = float(scenario.initial_yaw_rad) / 2.0
    state[3:7] = torch.as_tensor(
        (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
        dtype=state.dtype,
        device=state.device,
    )
    state[7:13] = 0.0
    from isaacgym import gymtorch

    actor_id = env._robot_actor_ids(torch.zeros(1, dtype=torch.long, device=env.device))
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env._all_root_states),
        gymtorch.unwrap_tensor(actor_id.to(dtype=torch.int32)),
        1,
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.base_quat[0] = env.root_states[0, 3:7]
    env.base_lin_vel[0] = env.root_states[0, 7:10]
    env.base_ang_vel[0] = env.root_states[0, 10:13]
    physical_yaw = float(_yaw_from_env(env))
    if hasattr(env, "tracking_heading"):
        env.tracking_heading[0] = physical_yaw
        heading_error = math.atan2(
            math.sin(float(env.tracking_heading[0].item()) - physical_yaw),
            math.cos(float(env.tracking_heading[0].item()) - physical_yaw),
        )
        if abs(heading_error) >= 1.0e-6:
            raise RuntimeError("tracking_heading is inconsistent with reset pose")
    if hasattr(env, "_update_tracking_motion"):
        env._update_tracking_motion(integrate_heading=False)
    else:
        env.tracking_lin_vel[0] = 0.0
        env.tracking_ang_vel[0] = 0.0
    for name in ("previous_velocity_command", "previous_actual_velocity", "last_velocity_command"):
        if hasattr(env, name):
            getattr(env, name)[0] = 0.0
    for name in ("command_targets", "commands", "applied_feasible_command"):
        if hasattr(env, name):
            getattr(env, name)[0, :2] = 0.0
    if getattr(env, "feasible_transition_manager", None) is not None:
        env.feasible_transition_manager.reset(torch.zeros(1, dtype=torch.long, device=env.device))
    if hasattr(env, "observation_goal_active"):
        env.observation_goal_active[0] = False
    if hasattr(env, "previous_goal_distance"):
        env.previous_goal_distance[0] = 0.0


def _set_final_goal(env, scenario, torch):
    goal = torch.as_tensor(scenario.goal_xy, dtype=env.root_states.dtype, device=env.device)
    env.global_goal_xy_world[0] = env.env_origins[0, :2] + goal
    distance = torch.linalg.vector_norm(env.global_goal_xy_world[0] - env.root_states[0, :2])
    env.goal_dist[0] = distance
    env.previous_goal_distance[0] = distance
    env.terminal_goal_distance[0] = distance
    env.goal_reached_buf[0] = False
    env.success_buf[0] = False


def _path_progress(path, position):
    points = np.asarray(path, dtype=np.float64)
    if len(points) < 2:
        return 0.0, 0.0
    deltas = points[1:] - points[:-1]
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    best_distance = float("inf")
    best_arc = 0.0
    point = np.asarray(position, dtype=np.float64)
    for index, (start, delta, length) in enumerate(zip(points[:-1], deltas, lengths)):
        if length <= 1.0e-9:
            continue
        fraction = float(np.clip(np.dot(point - start, delta) / (length * length), 0.0, 1.0))
        projection = start + fraction * delta
        distance = float(np.linalg.norm(point - projection))
        if distance < best_distance:
            best_distance = distance
            best_arc = float(cumulative[index] + fraction * length)
    return best_arc, float(cumulative[-1])


def _lookahead_waypoint(scenario, position, lookahead_m):
    points = np.asarray(scenario.oracle_path, dtype=np.float64)
    if len(points) < 2:
        return np.asarray(scenario.goal_xy, dtype=np.float64), 0.0
    progress, total = _path_progress(points, position)
    target_arc = min(progress + float(lookahead_m), total)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    index = max(1, min(len(points) - 1, int(np.searchsorted(cumulative, target_arc))))
    previous = float(cumulative[index - 1])
    delta = points[index] - points[index - 1]
    fraction = (target_arc - previous) / max(float(np.linalg.norm(delta)), 1.0e-9)
    return points[index - 1] + fraction * delta, max(0.0, total - progress)


def _goal_xy_robot(position, yaw, waypoint):
    delta = np.asarray(waypoint, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray((c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]), dtype=np.float64)


def schedule_terminal_speed(raw_speed, remaining_path_m, config):
    """Return a radial Cruise/Approach/Terminal command scale."""
    remaining = max(0.0, float(remaining_path_m))
    start = float(config.terminal_slowdown_start_m)
    radius = float(config.goal_success_radius_m)
    if remaining > start:
        return {"phase": "Cruise", "scale": 1.0, "active": False}
    if remaining > radius:
        scale = (remaining - radius) / max(start - radius, 1.0e-6)
        return {"phase": "Approach", "scale": float(np.clip(scale, 0.0, 1.0)), "active": True}
    return {"phase": "Terminal", "scale": 0.0, "active": True}


def _classify_failure(summary, trajectory=()):
    """Classify only from explicit terminal facts and sustained telemetry."""
    if summary.get("success"):
        return "SUCCESS"
    if summary.get("process_failure"):
        return "PROCESS_FAILURE"
    if summary.get("collision"):
        return "COLLISION"
    stall = transition_manager_stall_evidence(trajectory)
    terminal_convergence = terminal_convergence_evidence(trajectory)
    return classify_phase_d_failure(
        success=bool(summary.get("success")),
        collision=bool(summary.get("collision")),
        process_failure=bool(summary.get("process_failure")),
        transition_manager_stall=bool(stall),
        terminal_convergence_failure=bool(terminal_convergence),
        goal_progress_failure=(
            float(summary.get("final_goal_distance_m", float("inf"))) > 2.0
        ),
        timeout=bool(summary.get("timeout")),
    )


def require_real_depth(env):
    require_isaacgym_depth(
        getattr(env, "depth_backend_requested", None),
        getattr(env, "depth_backend_actual", None),
    )


def _collision_sources(position, scenario, radius):
    distances = [_point_aabb_distance(position, center, half) for center, half in scenario.raw_physics_aabbs()]
    boundary = min(distances[:4]) if distances else float("inf")
    obstacle = min(distances[4:]) if len(distances) > 4 else float("inf")
    return boundary <= radius, obstacle <= radius, boundary, obstacle


def _select_terminal_value(done, live_value, terminal_value):
    """Select pre-reset terminal telemetry when the step ended an episode."""
    return terminal_value if bool(done) else live_value


def run_episode(env, scenario, torch, max_steps, lookahead_m, config, depth_output_dir=None, direct_global_goal=False, final_goal_switch_distance=None):
    """Run one scripted Teacher episode through Frozen V62."""
    from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import RotunbotVel
    from legged_gym.navigation.v1_velocity_teacher import V1VelocityTeacherConfig, teacher_velocity_diagnostics

    env.reset()
    _set_robot_pose(env, scenario, torch)
    _set_final_goal(env, scenario, torch)
    env.compute_observations()
    # Backend becomes authoritative only after the first real IMAGE_DEPTH capture.
    require_real_depth(env)
    teacher_cfg = V1VelocityTeacherConfig(
        max_forward_speed=float(env.cfg.commands.max_forward_speed),
        max_yaw_rate=float(env.cfg.commands.max_yaw_rate),
        minimum_turn_radius=float(env.cfg.commands.minimum_turn_radius),
        feasible_envelope_fraction=float(env.cfg.commands.feasible_envelope_fraction),
        goal_radius=float(env.cfg.commands.goal_radius),
    )
    policy_interval = int(env.upper_level_command_interval_steps)
    timing = resolve_phase_d_timing(
        env.sim_params.dt,
        env.cfg.control.decimation,
        env.cfg.commands.upper_level_command_frequency_hz,
    )
    policy_dt = float(timing.policy_dt_s)
    position = _position_from_env(env)
    yaw = _yaw_from_env(env)
    previous_position = position.copy()
    path_length = 0.0
    raw_clearances = []
    planning_clearances = []
    rows = []
    command = torch.zeros(1, 2, dtype=env.root_states.dtype, device=env.device)
    desired_v_raw = 0.0
    desired_w_raw = 0.0
    desired_v_projected = 0.0
    desired_w_projected = 0.0
    desired_v_scheduled = 0.0
    desired_w_scheduled = 0.0
    remaining = float("inf")
    current_slowdown = False
    done = False
    collision = False
    boundary_collision = False
    obstacle_collision = False
    timeout = False
    steps = 0
    terminal_slowdown_steps = 0
    projection_interventions = 0
    transition_active_steps = 0
    min_goal_distance = float("inf")
    last_depth_stats = None
    last_action_snapshot = None
    last_command_target = (0.0, 0.0)
    last_raw_clearance = float(config.robot_radius_m)

    with torch.inference_mode():
        while not done and steps < int(max_steps):
            if steps == 0 or env.common_step_counter % policy_interval == 0:
                global_distance_now = float(np.linalg.norm(np.asarray(scenario.goal_xy) - position))
                use_final_goal = bool(
                    direct_global_goal
                    or (final_goal_switch_distance is not None and global_distance_now <= float(final_goal_switch_distance))
                )
                if use_final_goal:
                    waypoint = np.asarray(scenario.goal_xy, dtype=np.float64)
                    _, total_path = _path_progress(scenario.oracle_path, position)
                    remaining = max(0.0, total_path)
                else:
                    waypoint, remaining = _lookahead_waypoint(scenario, position, lookahead_m)
                env.set_observation_goal_world(
                    env.env_origins[0, :2]
                    + torch.as_tensor(waypoint, dtype=env.root_states.dtype, device=env.device).reshape(1, 2)
                )
                env.compute_observations()
                if depth_output_dir is not None:
                    depth_dir = Path(depth_output_dir)
                    depth_dir.mkdir(parents=True, exist_ok=True)
                    frame_id = len(list(depth_dir.glob("*.npy")))
                    raw_depth = getattr(env, "_last_depth_raw", None)
                    if raw_depth is not None:
                        np.save(depth_dir / ("step_%06d_raw.npy" % steps), raw_depth[0].detach().cpu().numpy())
                    np.save(depth_dir / ("step_%06d_normalized.npy" % steps), env.depth_observation[0].detach().cpu().numpy())
                goal_robot = torch.as_tensor(
                    _goal_xy_robot(position, yaw, waypoint).reshape(1, 2),
                    dtype=env.root_states.dtype,
                    device=env.device,
                )
                actual = torch.stack((env.tracking_lin_vel[:, 0], env.tracking_ang_vel[:, 2]), dim=1)
                teacher = teacher_velocity_diagnostics(goal_robot, actual, env.obstacle_clearance, teacher_cfg)
                raw = teacher["raw_command"].clone()
                projected = teacher["applied_command"].clone()
                # D0 has one terminal controller only: V1VelocityTeacher plus
                # the frozen V62 projection/transition path.  Keep the legacy
                # scheduler as diagnostics, never as a second command scale.
                schedule = schedule_terminal_speed(float(projected[0, 0].item()), remaining, config)
                current_slowdown = bool(schedule["active"])
                terminal_slowdown_steps += int(current_slowdown)
                command.copy_(projected)
                env.set_command_targets(command)
                desired_v_raw = float(raw[0, 0].item())
                desired_w_raw = float(raw[0, 1].item())
                desired_v_projected = float(projected[0, 0].item())
                desired_w_projected = float(projected[0, 1].item())
                desired_v_scheduled = desired_v_projected
                desired_w_scheduled = desired_w_projected

            last_command_target = (
                float(env.command_targets[0, 0].item()),
                float(env.command_targets[0, 1].item()),
            )
            last_raw_clearance = float(env.obstacle_clearance[0].item())
            last_applied_command = env.applied_feasible_command[0].detach().clone()
            last_depth_stats = {
                "finite_ratio": float(torch.isfinite(env.depth_observation[0]).float().mean().item()),
                "min": float(env.depth_observation[0].min().item()),
                "mean": float(env.depth_observation[0].mean().item()),
                "max": float(env.depth_observation[0].max().item()),
                "std": float(env.depth_observation[0].std(unbiased=False).item()),
            }
            last_action_snapshot = {
                "nominal_action_1": float(env.nominal_policy_actions[0, 0].item()),
                "nominal_action_2": float(env.nominal_policy_actions[0, 1].item()),
                "feedback_action_1": float(env.feedback_policy_actions[0, 0].item()),
                "feedback_action_2": float(env.feedback_policy_actions[0, 1].item()),
                "combined_action_1": float(env.combined_policy_actions[0, 0].item()),
                "combined_action_2": float(env.combined_policy_actions[0, 1].item()),
                "output_action_1": float(env.output_actions[0, 0].item()),
                "output_action_2": float(env.output_actions[0, 1].item()),
            }

            # RotunbotVel.step keeps the command target above intact and runs
            # the frozen deterministic V62 command-to-actuator controller.
            _, _, _, dones, _ = RotunbotVel.step(env, torch.zeros_like(command))
            steps += 1
            done = bool(dones.flatten()[0].item())
            position_now = (
                env.terminal_position[0, :2].detach().cpu().numpy().copy()
                if done else _position_from_env(env)
            )
            yaw_now = float(_select_terminal_value(done, _yaw_from_env(env), env.terminal_yaw[0].item() if hasattr(env, "terminal_yaw") else _yaw_from_env(env)))
            actual_v = float(_select_terminal_value(done, env.tracking_lin_vel[0, 0], env.terminal_tracking_velocity[0, 0]).item())
            actual_w = float(_select_terminal_value(done, env.tracking_ang_vel[0, 2], env.terminal_tracking_velocity[0, 1]).item())
            applied = _select_terminal_value(done, env.applied_feasible_command[0], env.terminal_applied_feasible_command[0])
            goal_distance_row = float(_select_terminal_value(done, env.goal_dist[0], env.terminal_goal_distance[0]).item())
            transition_state_row = int(_select_terminal_value(done, env.transition_state[0], env.terminal_transition_state[0]).item())
            transition_active_row = bool(_select_terminal_value(done, env.transition_active[0], env.terminal_transition_active[0]).item())
            transition_progress_row = float(env.transition_progress[0].item())
            transition_settle_counter_row = int(env.transition_settle_counter[0].item())
            if done:
                transition_progress_row = 0.0
                transition_settle_counter_row = 0

            if done:
                applied = last_applied_command
            applied_v, applied_w = float(applied[0].item()), float(applied[1].item())
            raw_clearance = last_raw_clearance if done else float(env.obstacle_clearance[0].item())
            if done and last_depth_stats is not None:
                depth_stats_row = last_depth_stats
            else:
                depth_stats_row = {
                    "finite_ratio": float(torch.isfinite(env.depth_observation[0]).float().mean().item()),
                    "min": float(env.depth_observation[0].min().item()),
                    "mean": float(env.depth_observation[0].mean().item()),
                    "max": float(env.depth_observation[0].max().item()),
                    "std": float(env.depth_observation[0].std(unbiased=False).item()),
                }
            boundary_now, obstacle_now, boundary_distance, obstacle_distance = _collision_sources(
                position_now, scenario, float(config.robot_radius_m)
            )
            collision_now = bool(env.terminal_collision[0].item()) if done else bool(env.step_collision_buf[0].item())
            collision = collision or collision_now
            boundary_collision = boundary_collision or boundary_now
            obstacle_collision = obstacle_collision or obstacle_now
            planning_clearance = scenario.point_planning_clearance(position_now)
            raw_clearances.append(raw_clearance - float(config.robot_radius_m))
            planning_clearances.append(planning_clearance)
            path_length += float(np.linalg.norm(position_now - previous_position))
            previous_position = position_now
            min_goal_distance = min(min_goal_distance, float(np.linalg.norm(position_now - np.asarray(scenario.goal_xy))))
            transition_active = transition_active_row
            transition_active_steps += int(transition_active)
            projection_interventions += int(
                abs(applied_v - desired_v_scheduled) > 3.0e-5
                or abs(applied_w - desired_w_scheduled) > 3.0e-5
            )
            rows.append({
                "step": steps,
                "time_s": steps * policy_dt,
                "x_m": float(position_now[0]), "y_m": float(position_now[1]), "yaw_rad": yaw_now,
                "global_goal_x_m": float(scenario.goal_xy[0]),
                "global_goal_y_m": float(scenario.goal_xy[1]),
                "waypoint_x_m": float(waypoint[0]),
                "waypoint_y_m": float(waypoint[1]),
                "waypoint_distance_m": float(np.linalg.norm(np.asarray(waypoint) - position_now)),
                "remaining_path_m": remaining,
                "global_goal_distance_m": goal_distance_row,
                "goal_success_radius_m": float(config.goal_success_radius_m),
                "desired_v_raw_mps": desired_v_raw,
                "desired_w_raw_rps": desired_w_raw,
                "desired_v_projected_mps": desired_v_projected,
                "desired_w_projected_rps": desired_w_projected,
                "desired_v_scheduled_mps": desired_v_scheduled,
                "desired_w_scheduled_rps": desired_w_scheduled,
                "command_target_v_mps": last_command_target[0],
                "command_target_w_rps": last_command_target[1],
                "applied_v_mps": applied_v, "applied_w_rps": applied_w,
                "actual_v_mps": actual_v, "actual_w_rps": actual_w,
                "raw_physics_clearance_m": raw_clearance,
                "physics_free_space_clearance_m": raw_clearance - float(config.robot_radius_m),
                "planning_clearance_m": planning_clearance,
                "boundary_clearance_m": boundary_distance - float(config.robot_radius_m),
                "obstacle_clearance_m": obstacle_distance - float(config.robot_radius_m),
                "terminal_slowdown_active": int(current_slowdown),
                "transition_active": int(transition_active_row),
                "transition_state": transition_state_row,
                "transition_progress": transition_progress_row,
                "transition_settle_counter": transition_settle_counter_row,
                "collision": int(collision_now),
                "depth_finite_ratio": depth_stats_row["finite_ratio"],
                "depth_min": depth_stats_row["min"],
                "depth_mean": depth_stats_row["mean"],
                "depth_max": depth_stats_row["max"],
                "depth_std": depth_stats_row["std"],
                **last_action_snapshot,
            })
            position, yaw = position_now, yaw_now
            if steps >= int(max_steps) and not done:
                timeout = True
                break

    success = bool(env.terminal_success[0].item()) if done else False
    if done:
        collision = bool(env.terminal_collision[0].item())
        timeout = bool(env.terminal_timeout[0].item())
        final_distance = float(env.terminal_goal_distance[0].item())
    else:
        final_distance = float(env.goal_dist[0].item())
    path_points = np.asarray(scenario.oracle_path, dtype=np.float64)
    if len(path_points) >= 2:
        terminal_delta = path_points[-1] - path_points[-2]
    else:
        terminal_delta = np.asarray(scenario.goal_xy, dtype=np.float64) - np.asarray(scenario.spawn_xy, dtype=np.float64)
    terminal_direction = terminal_delta / max(float(np.linalg.norm(terminal_delta)), 1.0e-9)
    terminal_overshoot = max(0.0, float(np.dot(position - np.asarray(scenario.goal_xy), terminal_direction)))
    summary = {
        "evaluation_version": EVALUATION_VERSION,
        "map_seed": int(scenario.map_seed), "attempt_index": int(scenario.attempt_index),
        "obstacle_count": int(scenario.obstacle_count),
        "spawn_xy": list(scenario.spawn_xy), "goal_xy": list(scenario.goal_xy),
        "goal_heading_rad": float(scenario.goal_heading_rad), "initial_yaw_rad": float(scenario.initial_yaw_rad),
        "initial_heading_error_rad": float((scenario.initial_yaw_rad - scenario.goal_heading_rad + math.pi) % (2.0 * math.pi) - math.pi),
        "oracle_path_length_m": float(scenario.oracle_path_length_m), "actual_path_length_m": path_length,
        "success": bool(success), "collision": bool(collision), "collision_with_boundary": bool(boundary_collision),
        "collision_with_obstacle": bool(obstacle_collision), "timeout": bool(timeout),
        "steps": int(steps), "time_to_goal_s": steps * policy_dt if success else None,
        "final_goal_distance_m": final_distance, "minimum_goal_distance_m": min_goal_distance,
        "spl": float(scenario.oracle_path_length_m / max(scenario.oracle_path_length_m, path_length)) if success else 0.0,
        "minimum_raw_physics_clearance_m": float(min(raw_clearances)) if raw_clearances else None,
        "p05_raw_physics_clearance_m": float(np.percentile(raw_clearances, 5.0)) if raw_clearances else None,
        "median_raw_physics_clearance_m": float(np.median(raw_clearances)) if raw_clearances else None,
        "minimum_planning_clearance_m": float(min(planning_clearances)) if planning_clearances else None,
        "p05_planning_clearance_m": float(np.percentile(planning_clearances, 5.0)) if planning_clearances else None,
        "median_planning_clearance_m": float(np.median(planning_clearances)) if planning_clearances else None,
        "terminal_overshoot_m": terminal_overshoot,
        "terminal_tangent_rad": math.atan2(float(terminal_direction[1]), float(terminal_direction[0])),
        "terminal_slowdown_steps": int(terminal_slowdown_steps),
        "projection_intervention_count": int(projection_interventions),
        "transition_active_steps": int(transition_active_steps),
        "physics_dt_s": float(timing.physics_dt_s),
        "control_decimation": int(timing.control_decimation),
        "policy_dt_s": float(timing.policy_dt_s),
        "hold_policy_steps": int(timing.hold_policy_steps),
        "hold_physics_steps": int(timing.hold_physics_steps),
        "upper_command_hz": float(timing.upper_command_hz),
        "depth_backend_requested": str(getattr(env, "depth_backend_requested", "unknown")),
        "depth_backend_actual": str(getattr(env, "depth_backend_actual", "unknown")),
        "depth_capture_metadata": getattr(env, "depth_capture_metadata", lambda: {})(),
        "oracle_pass_side": getattr(scenario, "oracle_pass_side", "none"),
        "failure_flags": [],
        "direct_global_goal": bool(direct_global_goal),
        "final_goal_switch_distance_m": None if final_goal_switch_distance is None else float(final_goal_switch_distance),
        "command_loss_breakdown": command_loss_breakdown(rows),
    }
    summary["failure_reason"] = _classify_failure(summary, rows)
    if not success:
        summary["failure_flags"].append(summary["failure_reason"])
    if summary["terminal_overshoot_m"] > 0.5:
        summary["failure_flags"].append("TERMINAL_OVERSHOOT")
    if summary["terminal_slowdown_steps"] > 0 and not success:
        summary["failure_flags"].append("TERMINAL_REGION_ENTERED")
    if boundary_collision:
        summary["failure_flags"].append("COLLISION_WITH_BOUNDARY")
    if obstacle_collision:
        summary["failure_flags"].append("COLLISION_WITH_OBSTACLE")
    return summary, rows


def _configure_env(env_cfg, scenario, max_steps):
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = float(max_steps) * float(env_cfg.sim.dt * env_cfg.control.decimation) + 1.0
    env_cfg.enable_camera_sensors_in_headless = True
    env_cfg.camera.depth_backend = "isaacgym"
    env_cfg.camera.add_noise = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.init_state.random_start_lateral = 0.0
    env_cfg.init_state.random_start_yaw = 0.0
    env_cfg.commands.v1_goal_curriculum_enabled = False
    env_cfg.commands.v1_performance_curriculum_enabled = False
    env_cfg.commands.resample_commands = False
    env_cfg.commands.smooth_profile_fraction = 0.0
    env_cfg.commands.random_walk_profile_fraction = 0.0
    env_cfg.commands.independent_smooth_profile_fraction = 0.0
    env_cfg.corridor_wall_segments = ()
    env_cfg.corridor_explicit_wall_segments = scenario_wall_segments(scenario)
    env_cfg.direct_obstacle_aabbs = scenario_physics_aabbs(scenario)
    return env_cfg


def _framework_args(remaining):
    from legged_gym.utils import get_args
    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + list(remaining)
    try:
        return get_args()
    finally:
        os.sys.argv = original


def _write_plot(summary, trajectory, scenario, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 7))
    low_x, low_y, high_x, high_y = scenario.bounds_xy
    axis.plot([low_x, high_x, high_x, low_x, low_x], [low_y, low_y, high_y, high_y, low_y], "k-")
    for obstacle in scenario.obstacles:
        corners = list(obstacle.corners()) + [obstacle.corners()[0]]
        axis.plot([point[0] for point in corners], [point[1] for point in corners], "r-")
    if scenario.oracle_path:
        points = np.asarray(scenario.oracle_path)
        axis.plot(points[:, 0], points[:, 1], "b--", label="oracle path")
    if trajectory:
        axis.plot([row["x_m"] for row in trajectory], [row["y_m"] for row in trajectory], "g-", label="robot")
        axis.scatter([trajectory[-1]["x_m"]], [trajectory[-1]["y_m"]], c="m", label="terminal")
    axis.scatter([scenario.spawn_xy[0]], [scenario.spawn_xy[1]], c="black", label="spawn")
    axis.scatter([scenario.goal_xy[0]], [scenario.goal_xy[1]], c="orange", label="goal")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (m)"); axis.set_ylabel("y (m)")
    axis.set_title("%s map=%d %s" % (summary.get("failure_reason"), scenario.map_seed, "PASS" if summary.get("success") else "FAIL"))
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_one_map(scenario, output_dir, max_steps, lookahead_m, remaining, direct_global_goal=False, final_goal_switch_distance=None):
    import isaacgym  # noqa: F401
    import torch
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils import task_registry

    args = _framework_args(remaining)
    args.task = "rotunbot_sru_visual_corridor_v1"
    args.seed = int(scenario.map_seed)
    env_cfg, _ = task_registry.get_cfgs(args.task)
    _configure_env(env_cfg, scenario, max_steps)
    env_cfg.seed = int(scenario.map_seed)
    env = None
    try:
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
        map_dir = Path(output_dir).resolve() / ("map_%d" % int(scenario.map_seed))
        summary, trajectory = run_episode(
            env, scenario, torch, max_steps, lookahead_m, scenario.config,
            depth_output_dir=map_dir / "depth_samples",
            direct_global_goal=bool(direct_global_goal),
            final_goal_switch_distance=final_goal_switch_distance,
        )
    except Exception as error:
        summary = {
            "evaluation_version": EVALUATION_VERSION, "map_seed": int(scenario.map_seed),
            "obstacle_count": int(scenario.obstacle_count), "success": False,
            "collision": False, "timeout": False, "process_failure": True,
            "failure_reason": "PROCESS_FAILURE",
            "failure_flags": ["PROCESS_FAILURE"], "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        trajectory = []
    finally:
        _close(env)
    output_dir = Path(output_dir).resolve()
    map_dir = output_dir / ("map_%d" % int(scenario.map_seed))
    map_dir.mkdir(parents=True, exist_ok=True)
    (map_dir / "scenario.json").write_text(json.dumps(scenario_to_metadata(scenario), indent=2, sort_keys=True), encoding="utf-8")
    (map_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    (map_dir / "command_loss_breakdown.json").write_text(json.dumps(summary.get("command_loss_breakdown", {}), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    if summary.get("failure_reason") != "SUCCESS":
        diagnosis = {
            "map_seed": int(scenario.map_seed),
            "failure_reason": summary.get("failure_reason"),
            "final_goal_distance_m": summary.get("final_goal_distance_m"),
            "timeout": bool(summary.get("timeout", False)),
            "collision": bool(summary.get("collision", False)),
            "depth_backend_requested": summary.get("depth_backend_requested"),
            "depth_backend_actual": summary.get("depth_backend_actual"),
            "trajectory_rows": len(trajectory),
            "mean_actual_v_mps": (float(np.mean([row["actual_v_mps"] for row in trajectory])) if trajectory else None),
            "mean_applied_v_mps": (float(np.mean([row["applied_v_mps"] for row in trajectory])) if trajectory else None),
            "transition_active_steps": summary.get("transition_active_steps", 0),
            "error": summary.get("error"),
        }
        (map_dir / "failure_diagnosis.json").write_text(json.dumps(diagnosis, indent=2, sort_keys=True), encoding="utf-8")
    if trajectory:
        with (map_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trajectory[0]))
            writer.writeheader(); writer.writerows(trajectory)
        _write_plot(summary, trajectory, scenario, map_dir / "topdown.png")
    return summary


def _close(env):
    if env is None:
        return
    if getattr(env, "viewer", None) is not None:
        env.gym.destroy_viewer(env.viewer)
    if getattr(env, "sim", None) is not None:
        env.gym.destroy_sim(env.sim)


def _stage_defaults(stage):
    values = {
        "d0": (10, [0] * 10),
        "d1.1-smoke": (5, [1] * 5),
        "d1.1": (20, [1] * 20),
        "d1.2-smoke": (10, [2] * 10),
        "d1.2": (40, [2] * 40),
    }
    try:
        return values[str(stage).lower()]
    except KeyError as error:
        raise ValueError("unknown stage") from error


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=("d0", "d1.1-smoke", "d1.1", "d1.2-smoke", "d1.2"), default="d0")
    parser.add_argument("--mode", choices=("d0-c", "d0-b", "final-goal-switch"), default="d0-c")
    parser.add_argument("--final-goal-switch-distance-m", type=float, default=None)
    parser.add_argument("--maps", type=int, default=None)
    parser.add_argument("--obstacle-counts", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--map-seed-file", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--lookahead-m", type=float, default=DEFAULT_LOOKAHEAD_M)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--single-map", action="store_true")
    stage, remaining = parser.parse_known_args(argv)
    default_maps, default_counts = _stage_defaults(stage.stage)
    maps = int(stage.maps or default_maps)
    counts = [int(value) for value in stage.obstacle_counts.split(",")] if stage.obstacle_counts else default_counts[:maps]
    if len(counts) != maps:
        raise ValueError("obstacle-counts length must equal maps")
    split_config = RandomObstacleSplitConfig()
    if stage.map_seed_file:
        seeds = [int(line.strip()) for line in Path(stage.map_seed_file).read_text().splitlines() if line.strip()]
        if len(seeds) != maps:
            raise ValueError("map seed file length must equal maps")
    else:
        low, high = split_config.range_for(stage.split)
        seeds = random.Random(int(stage.seed)).sample(list(range(low, high + 1)), maps)
    config = RandomObstacleConfig(obstacle_count_min=0, obstacle_count_max=2, max_episode_steps=int(stage.max_steps))
    scenarios = []
    for count, map_seed in zip(counts, seeds):
        scenario = sample_random_obstacle_scenario(map_seed, count, config=config, split=stage.split)
        validate_random_scenario(scenario)
        scenarios.append(scenario)
    output = Path(stage.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = {
        "evaluation_version": EVALUATION_VERSION,
        "stage": stage.stage,
        "split": stage.split,
        "config_hash": config_hash(config),
        "inventory_hash": frozen_inventory_hash(scenarios),
        "maps": [scenario_to_metadata(item) for item in scenarios],
    }
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")
    if maps > 1 and not stage.single_map:
        raise ValueError("run one map per IsaacGym process; pass --single-map for one-map invocations")
    if maps != 1:
        raise ValueError("--single-map invocation must contain exactly one map")
    direct_global_goal = stage.mode == "d0-b"
    final_goal_switch_distance = (
        stage.final_goal_switch_distance_m
        if stage.mode == "final-goal-switch"
        else None
    )
    summary = run_one_map(
        scenarios[0], output, stage.max_steps, stage.lookahead_m, remaining,
        direct_global_goal=direct_global_goal,
        final_goal_switch_distance=final_goal_switch_distance,
    )
    (output / "D1_summary.json").write_text(json.dumps({"evaluation_version": EVALUATION_VERSION, "stage": stage.stage, "mode": stage.mode, "episodes": [summary]}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
