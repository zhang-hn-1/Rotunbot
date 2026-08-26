"""Non-training C1/C2/C3 Frozen-4150 collision control gates."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from legged_gym.navigation.isaac_compat import install_isaac_gym_compat

install_isaac_gym_compat()

from legged_gym.navigation.baseline import (  # noqa: E402
    CHECKPOINT_RELATIVE_PATH,
    LOCAL_WAYPOINT_DISTANCE_M,
    SUCCESS_DISTANCE_M,
    SUCCESS_SPEED_MPS,
)
from legged_gym.navigation.control_diagnostics import (  # noqa: E402
    C2_INITIAL_SPEEDS_MPS,
    select_corner_case,
    select_detour_case,
    select_straight_case,
)
from legged_gym.navigation.evaluation_logging import EpisodeLogger  # noqa: E402
from legged_gym.navigation.frozen_p2p import (  # noqa: E402
    action_was_clipped,
    frozen_inference_policy,
    load_frozen_runner,
    refresh_observation_after_goal_change,
    robot_pose,
    robot_speed,
    set_temporary_world_goal,
)
from legged_gym.navigation.goal_switch import GoalSwitchController  # noqa: E402
from legged_gym.navigation.hierarchical_maze import (  # noqa: E402
    HierarchicalMazeCfg,
    HierarchicalMazeP2P,
)
from legged_gym.navigation.local_goal_adapter import world_to_local  # noqa: E402
from legged_gym.navigation.oracle_diagnostics import (  # noqa: E402
    classify_collision,
    local_goal_polar,
    nearest_wall_clearance,
    point_to_segment_distance,
    reachability_clip_ratio,
)
from legged_gym.navigation.bfs_planner import cell_center, world_to_cell  # noqa: E402
from legged_gym.navigation.frozen_p2p import enforce_frozen_control_config  # noqa: E402
from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed  # noqa: E402


def _parse_script_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT_RELATIVE_PATH)
    parser.add_argument("--output-dir", default="logs/hierarchical_navigation/control_diagnostics")
    parser.add_argument("--max-steps", type=int, default=3002)
    return parser.parse_args()


def _isaac_args():
    from legged_gym import envs  # noqa: F401
    from legged_gym.utils import get_args

    saved = sys.argv
    sys.argv = [saved[0], "--headless", "--rl_device=cuda:0", "--sim_device=cuda:0"]
    try:
        return get_args()
    finally:
        sys.argv = saved


def _load_maze(args, checkpoint):
    from legged_gym.envs import task_registry

    env_cfg = HierarchicalMazeCfg()
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    enforce_frozen_control_config(env_cfg)
    env_cfg.latency.enabled = False
    _, train_cfg = task_registry.get_cfgs(name="rotunbot_target_repro")
    train_cfg.runner.resume = False
    set_seed(env_cfg.maze.seed)
    sim_params = parse_sim_params(args, {"sim": class_to_dict(env_cfg.sim)})
    env = HierarchicalMazeP2P(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless,
    )
    runner = load_frozen_runner(args, env, train_cfg, checkpoint)
    return env, frozen_inference_policy(runner, args.rl_device)


def _set_initial_forward_speed(env, speed_mps):
    import torch
    from isaacgym import gymtorch

    _, yaw = robot_pose(env)
    velocity = torch.tensor(
        [float(speed_mps) * np.cos(yaw), float(speed_mps) * np.sin(yaw)],
        dtype=env.root_states.dtype,
        device=env.root_states.device,
    )
    env.root_states[0, 7:9] = velocity
    env.root_states[0, 9:13] = 0.0
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.actor_root_state),
        gymtorch.unwrap_tensor(env.robot_actor_indices),
        1,
    )


def _cell_world(env, cell):
    origin = env.env_origins[0, :2].detach().cpu().numpy().astype(np.float64)
    return cell_center(cell, env.maze_layout.shape, env.cfg.maze.cell_size) + origin


def _run_case(env, policy, name, cells, initial_speed_mps, script_args, index, output_dir):
    obs, _ = env.reset()
    from legged_gym.scripts.evaluate_oracle_maze import _set_center_start

    _set_center_start(env)
    _set_initial_forward_speed(env, initial_speed_mps)
    switcher = GoalSwitchController(env)
    switcher.set_intermediate_goal_mode(True)
    targets = tuple(cells[1:])
    global_goal = _cell_world(env, targets[-1])
    logger = EpisodeLogger({
        "gate": name,
        "protocol": "non_training_control_diagnostic",
        "checkpoint": str(script_args.checkpoint),
        "initial_speed_mps": float(initial_speed_mps),
        "cells": [list(cell) for cell in cells],
    })
    current_target_index = 0
    current_world_goal = None
    current_raw_local_goal = None
    current_steps_since_switch = 0
    success = False
    reason = "timeout"
    last_xy, _ = robot_pose(env)
    min_clearance = float("inf")
    max_cross_track = 0.0
    collision_diagnostic = None

    for step in range(script_args.max_steps):
        robot_xy, robot_yaw = robot_pose(env)
        if current_world_goal is None:
            target_cell = targets[current_target_index]
            current_world_goal = _cell_world(env, target_cell)
            current_raw_local_goal = world_to_local(robot_xy, robot_yaw, current_world_goal)
            switcher.update_world_goal(current_world_goal, time_s=step * float(env.dt))
            obs = refresh_observation_after_goal_change(env)
            current_steps_since_switch = 0

        action = policy(obs)
        obs, _privileged, _reward, dones, _infos = env.step(action)
        done = bool(dones[0].item())
        control_xy, control_yaw = robot_pose(env)
        control_speed = robot_speed(env)
        collision = bool(env.maze_collision_buf[0].item())
        diagnostic_xy = control_xy
        diagnostic_yaw = control_yaw
        diagnostic_speed = control_speed
        if collision or done:
            diagnostic_xy = env.terminal_position[0].detach().cpu().numpy().astype(np.float64)
            diagnostic_yaw = float(env.terminal_yaw[0].detach().cpu().item())
            diagnostic_speed = float(env.terminal_speed[0].detach().cpu().item())
        local_goal = world_to_local(diagnostic_xy, diagnostic_yaw, current_world_goal)
        local_distance, local_bearing = local_goal_polar(local_goal)
        current_cell = world_to_cell(
            diagnostic_xy - env.env_origins[0, :2].detach().cpu().numpy(),
            env.maze_layout.shape,
            env.cfg.maze.cell_size,
        )
        waypoint_cell = targets[current_target_index]
        next_cell = targets[current_target_index + 1] if current_target_index + 1 < len(targets) else None
        current_segment = [_cell_world(env, current_cell).tolist(), _cell_world(env, waypoint_cell).tolist()]
        next_segment = (
            [_cell_world(env, waypoint_cell).tolist(), _cell_world(env, next_cell).tolist()]
            if next_cell is not None else None
        )
        cross_track = point_to_segment_distance(diagnostic_xy, current_segment[0], current_segment[1])
        origin = env.env_origins[0, :2].detach().cpu().numpy().astype(np.float64)
        wall_surface, clearance = nearest_wall_clearance(
            diagnostic_xy - origin,
            env._maze_wall_centers,
            [float(env.cfg.maze.cell_size)] * 2,
            float(env.cfg.maze.robot_collision_radius),
        )
        min_clearance = min(min_clearance, clearance)
        max_cross_track = max(max_cross_track, cross_track)
        logger.record_step(
            step=step,
            time_s=(step + 1) * float(env.dt),
            robot_xy=diagnostic_xy,
            robot_yaw=diagnostic_yaw,
            robot_speed=diagnostic_speed,
            waypoint_cell=waypoint_cell,
            next_bfs_cell=next_cell,
            local_goal_distance=local_distance,
            local_goal_bearing=local_bearing,
            waypoint_distance=float(np.linalg.norm(current_world_goal - diagnostic_xy)),
            steps_since_goal_switch=current_steps_since_switch,
            turn_aware_triggered=False,
            reachability_filtered=False,
            raw_local_goal_xy=current_raw_local_goal,
            filtered_local_goal_xy=current_raw_local_goal,
            reachability_clip_ratio=0.0,
            nearest_wall_distance=wall_surface,
            nearest_wall_surface_distance=wall_surface,
            robot_clearance=clearance,
            cross_track_error_to_current_bfs_segment=cross_track,
            collision=collision,
            action=action[0].detach().cpu().numpy(),
            action_clipped=action_was_clipped(env, action),
        )
        if collision:
            labels = classify_collision(
                phase="NAVIGATE",
                steps_since_goal_switch=current_steps_since_switch,
                delta_bearing_deg=(90.0 if next_cell is not None and current_target_index == 0 and name.startswith("C2") else 0.0),
                waypoint_reached=False,
                current_cell=current_cell,
                waypoint_cell=waypoint_cell,
                next_bfs_cell=next_cell,
            )
            collision_diagnostic = {
                **labels,
                "collision_step": step,
                "collision_xy": diagnostic_xy.tolist(),
                "collision_phase": "NAVIGATE",
                "steps_since_goal_switch": current_steps_since_switch,
                "collision_local_goal_bearing": local_bearing,
                "collision_local_goal_distance": local_distance,
                "current_bfs_segment": current_segment,
                "next_bfs_segment": next_segment,
                "nearest_wall_surface_distance": wall_surface,
                "robot_clearance": clearance,
                "cross_track_error_to_current_bfs_segment": cross_track,
            }
            reason = "collision"
            break
        if current_target_index == len(targets) - 1:
            if (
                float(np.linalg.norm(global_goal - control_xy)) <= SUCCESS_DISTANCE_M
                and control_speed <= SUCCESS_SPEED_MPS
            ):
                success = True
                reason = "global_success"
                break
        elif float(np.linalg.norm(current_world_goal - control_xy)) <= LOCAL_WAYPOINT_DISTANCE_M:
            current_target_index += 1
            current_world_goal = None
        if done:
            reason = "timeout" if bool(env.terminal_timeout[0].item()) else "unstable"
            break
        current_steps_since_switch += 1

    logger.finish(
        success=success,
        reason=reason,
        completion_time_s=(step + 1) * float(env.dt),
        minimum_robot_clearance_m=min_clearance,
        maximum_cross_track_error_m=max_cross_track,
        collision_diagnostic=collision_diagnostic,
        collision_step=collision_diagnostic["collision_step"] if collision_diagnostic else None,
        collision_xy=collision_diagnostic["collision_xy"] if collision_diagnostic else None,
    )
    logger.write_json(output_dir / f"episode_{index:04d}.json")
    return logger.summary


def run_gate(args, script_args):
    output_dir = Path(script_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env, policy = _load_maze(args, script_args.checkpoint)
    start_cell = tuple(int(value) for value in np.asarray(env.maze_layout.shape) // 2)
    scenarios = {
        "C1_straight_corridor": select_straight_case(
            env.maze_layout, start_cell, minimum_edges=2, maximum_edges=2
        ),
        "C2_single_90_corner": select_corner_case(env.maze_layout, start_cell),
        "C3_single_wall_detour": select_detour_case(env.maze_layout, start_cell),
    }
    results = []
    index = 0
    try:
        for name, scenario in scenarios.items():
            speeds = C2_INITIAL_SPEEDS_MPS if name.startswith("C2") else (0.0,)
            for speed in speeds:
                result = _run_case(
                    env, policy, name, scenario["cells"], speed, script_args, index, output_dir
                )
                result["scenario"] = name
                result["initial_speed_mps"] = float(speed)
                results.append(result)
                index += 1
    finally:
        env.gym.destroy_sim(env.sim)
    summary = {
        "gate": "control_diagnostics",
        "checkpoint": str(script_args.checkpoint),
        "scenarios": {name: {key: value for key, value in scenario.items()} for name, scenario in scenarios.items()},
        "c2_initial_speeds_mps": list(C2_INITIAL_SPEEDS_MPS),
        "results": results,
        "c1": {
            "success_rate": float(np.mean([row["success"] for row in results if row["scenario"] == "C1_straight_corridor"])),
            "collision_rate": float(np.mean([row["reason"] == "collision" for row in results if row["scenario"] == "C1_straight_corridor"])),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    script_args = _parse_script_args()
    run_gate(_isaac_args(), script_args)
