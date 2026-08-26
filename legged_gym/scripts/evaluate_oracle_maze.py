"""Gate 3: ground-truth BFS waypoints executed by frozen uniform 4150."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from legged_gym.navigation.isaac_compat import install_isaac_gym_compat

# The maze class imports the legacy environment stack, so install the runtime
# boundary before importing it (the P2P implementation itself is untouched).
install_isaac_gym_compat()

from legged_gym.navigation.baseline import (
    CHECKPOINT_RELATIVE_PATH,
    LOCAL_WAYPOINT_DISTANCE_M,
    SUCCESS_DISTANCE_M,
    SUCCESS_SPEED_MPS,
)
from legged_gym.navigation.evaluation_logging import EpisodeLogger
from legged_gym.navigation.frozen_p2p import (
    enforce_frozen_control_config,
    action_was_clipped,
    frozen_inference_policy,
    load_frozen_runner,
    refresh_observation_after_goal_change,
    robot_pose,
    robot_speed,
    set_temporary_world_goal,
)
from legged_gym.navigation.goal_switch import GoalSwitchController
from legged_gym.navigation.hierarchical_maze import HierarchicalMazeCfg, HierarchicalMazeP2P
from legged_gym.navigation.local_goal_adapter import local_to_world, world_to_local
from legged_gym.navigation.oracle_episode import OracleEpisodePlanner
from legged_gym.navigation.oracle_episode import waypoint_reached
from legged_gym.navigation.oracle_metrics import summarize_oracle_results, maze_spl
from legged_gym.navigation.bfs_planner import cell_center, plan_cells, world_to_cell
from legged_gym.navigation.oracle_diagnostics import (
    classify_collision,
    local_goal_polar,
    nearest_wall_clearance,
    reachability_clip_ratio,
    point_to_segment_distance,
)
from legged_gym.navigation.reachability import load_envelope


def _parse_script_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT_RELATIVE_PATH)
    parser.add_argument("--output-dir", default="logs/hierarchical_navigation/oracle_maze")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=6002)
    parser.add_argument("--waypoint-radius", type=float, default=LOCAL_WAYPOINT_DISTANCE_M)
    parser.add_argument("--reachability-envelope")
    parser.add_argument("--smoke", action="store_true", help="10-episode software-integrity smoke mode")
    parser.add_argument("--episode-manifest")
    parser.add_argument("--turn-aware", action="store_true")
    return parser.parse_args()


def _isaac_args():
    install_isaac_gym_compat()
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
    from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed

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
    return env, runner, frozen_inference_policy(runner, args.rl_device)


def _set_center_start(env):
    import torch
    from isaacgym import gymtorch

    env.root_states[0, :2] = env.env_origins[0, :2]
    env.root_states[0, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.actor_root_state),
        gymtorch.unwrap_tensor(env.robot_actor_indices),
        1,
    )
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = 0.0
    env.base_ang_vel[:] = 0.0
    env.compute_observations()


def _state_snapshot(env):
    history = getattr(env, "obs_history", None)
    previous = getattr(env, "last_actions", None)
    episode = getattr(env, "episode_length_buf", None)
    reset = getattr(env, "reset_buf", None)
    values = {}
    for name, value in (("previous", previous), ("episode", episode), ("reset", reset)):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy().copy()
        elif value is not None:
            value = np.asarray(value).copy()
        values[name] = value
    values["history_length"] = len(history) if history is not None else None
    return values


def _state_is_continuous(before, after):
    return (
        before["history_length"] == after["history_length"]
        and np.array_equal(before["previous"], after["previous"])
        and np.array_equal(before["episode"], after["episode"])
        and np.array_equal(before["reset"], after["reset"])
    )


def _episode_manifest(path, reachable, episodes, seed=0):
    path = Path(path)
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if len(manifest) != int(episodes):
            raise ValueError("episode manifest length does not match --episodes")
        for expected_episode_id, entry in enumerate(manifest):
            goal_index = int(entry["goal_index"])
            if not 0 <= goal_index < len(reachable):
                raise ValueError(f"manifest goal_index out of range: {goal_index}")
            if int(entry["episode_id"]) != expected_episode_id:
                raise ValueError("episode manifest ids must be contiguous and ordered")
            expected_goal = np.asarray(reachable[goal_index], dtype=np.float64)
            actual_goal = np.asarray(entry["goal_xy"], dtype=np.float64)
            if actual_goal.shape != (2,) or not np.allclose(actual_goal, expected_goal, atol=1.0e-8):
                raise ValueError("episode manifest goal_xy does not match the maze goal list")
        return manifest
    rng = np.random.default_rng(seed)
    manifest = []
    for episode_id in range(int(episodes)):
        goal_index = int(rng.integers(0, len(reachable)))
        manifest.append({
            "episode_id": episode_id,
            "goal_index": goal_index,
            "goal_xy": np.asarray(reachable[goal_index], dtype=np.float64).tolist(),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _failure_from_done(env):
    if bool(env.terminal_unstable[0].item()):
        return "unstable"
    if bool(env.terminal_out_of_bounds[0].item()):
        return "out_of_bounds"
    if bool(env.terminal_timeout[0].item()):
        return "timeout"
    return "waypoint_failure"


def _diagnostic_bfs_context(env, robot_xy, waypoint_cell, next_bfs_cell):
    """Return world-space BFS segments without changing planner state."""
    origin = env.env_origins[0, :2].detach().cpu().numpy().astype(np.float64)
    shape = env.maze_layout.shape
    size = float(env.cfg.maze.cell_size)
    current_cell = world_to_cell(robot_xy - origin, shape, size)
    current_center = cell_center(current_cell, shape, size) + origin
    waypoint_center = cell_center(waypoint_cell, shape, size) + origin
    current_segment = [current_center.tolist(), waypoint_center.tolist()]
    next_segment = None
    if next_bfs_cell is not None:
        next_center = cell_center(next_bfs_cell, shape, size) + origin
        next_segment = [waypoint_center.tolist(), next_center.tolist()]
    return current_cell, current_segment, next_segment


def _release_episode_cache():
    """Return temporary PyTorch allocations before the next long episode."""
    import gc
    import torch

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def _run_episode(env, policy, planner, script_args, episode_id, global_goal, goal_index):
    obs, _ = env.reset()
    _set_center_start(env)
    global_goal = np.asarray(global_goal, dtype=np.float64)
    switcher = GoalSwitchController(env)
    switcher.set_intermediate_goal_mode(True)
    logger = EpisodeLogger({
        "gate": "oracle_maze",
        "episode_id": episode_id,
        "goal_index": goal_index,
        "maze_seed": int(env.cfg.maze.seed),
        "protocol": "oracle_maze_120s",
        "low_level_protocol": "uniform_4150_original_60s_p2p",
        "turn_aware": bool(script_args.turn_aware),
    })
    waypoint_records = []
    current_world_goal = None
    current_local_goal = None
    current_raw_local_goal = None
    current_delta_bearing_deg = 0.0
    current_waypoint_cell = None
    current_next_bfs_cell = None
    current_reachability_filtered = False
    current_reachability_clip_ratio = 0.0
    steps_since_goal_switch = 0
    final_approach_entered = False
    final_approach_success = False
    final_approach_timeout = False
    final_approach_escape = False
    final_goal_cell = world_to_cell(global_goal, env.maze_layout.shape, env.cfg.maze.cell_size)
    collisions = 0
    success = False
    reason = "timeout"
    coordinate_errors = 0
    state_continuity_violations = 0
    actual_path_length = 0.0
    waypoint_reached_count = 0
    collision_diagnostic = None
    steps_taken = 0
    last_xy, _ = robot_pose(env)
    global_distance = float(np.linalg.norm(global_goal - last_xy))
    try:
        start_cell = world_to_cell(last_xy, env.maze_layout.shape, env.cfg.maze.cell_size)
        goal_cell = world_to_cell(global_goal, env.maze_layout.shape, env.cfg.maze.cell_size)
        bfs_path = plan_cells(env.maze_layout, start_cell, goal_cell)
        bfs_shortest_path_length = float(max(len(bfs_path) - 1, 0) * env.cfg.maze.cell_size)
    except Exception as error:
        reason = "planner_error"
        bfs_shortest_path_length = 0.0
        logger.finish(
            success=False,
            reason=reason,
            global_goal_xy=global_goal,
            waypoint_count=0,
            local_waypoint_reached_count=0,
            actual_path_length_m=0.0,
            bfs_shortest_path_length_m=bfs_shortest_path_length,
            maze_spl=0.0,
            completion_time_s=0.0,
            coordinate_error_count=0,
            state_continuity_violation_count=0,
            checkpoint_control_configuration_error_count=0,
            planner_error=str(error),
        )
        return logger
    for step in range(script_args.max_steps):
            steps_taken = step + 1
            robot_xy, robot_yaw = robot_pose(env)
            if current_world_goal is None:
                try:
                    waypoint = planner.next_local_waypoint(robot_xy, robot_yaw, global_goal)
                except Exception as error:
                    reason = "planner_error"
                    logger.record_step(step=step, planner_error=str(error))
                    break
                current_local_goal = waypoint.filtered_local_goal_xy
                current_raw_local_goal = waypoint.local_goal_xy
                current_world_goal = np.asarray(waypoint.temporary_world_goal_xy)
                current_delta_bearing_deg = float(waypoint.delta_bearing_deg)
                current_waypoint_cell = tuple(waypoint.cell)
                current_next_bfs_cell = None
                if waypoint.is_final_approach:
                    final_approach_entered = True
                    current_world_goal = global_goal.copy()
                    current_local_goal = waypoint.local_goal_xy
                    current_delta_bearing_deg = 0.0
                else:
                    try:
                        active_path = plan_cells(
                            env.maze_layout,
                            world_to_cell(robot_xy - env.env_origins[0, :2].detach().cpu().numpy(), env.maze_layout.shape, env.cfg.maze.cell_size),
                            world_to_cell(global_goal - env.env_origins[0, :2].detach().cpu().numpy(), env.maze_layout.shape, env.cfg.maze.cell_size),
                        )
                        current_next_bfs_cell = (
                            tuple(active_path[2]) if len(active_path) > 2 else None
                        )
                    except Exception:
                        current_next_bfs_cell = None
                current_reachability_filtered = bool(
                    np.linalg.norm(np.asarray(waypoint.local_goal_xy) - np.asarray(waypoint.filtered_local_goal_xy))
                    > 1.0e-8
                )
                current_reachability_clip_ratio = reachability_clip_ratio(
                    waypoint.local_goal_xy, waypoint.filtered_local_goal_xy
                )
                steps_since_goal_switch = 0
                reconstructed_world = local_to_world(robot_xy, robot_yaw, waypoint.local_goal_xy)
                reconstructed_temporary = local_to_world(
                    robot_xy, robot_yaw, waypoint.filtered_local_goal_xy
                )
                if (
                    np.linalg.norm(reconstructed_world - np.asarray(waypoint.world_goal_xy)) > 1.0e-8
                    or np.linalg.norm(reconstructed_temporary - current_world_goal) > 1.0e-8
                ):
                    coordinate_errors += 1
                before = _state_snapshot(env)
                try:
                    switcher.update_world_goal(current_world_goal, time_s=step * float(env.dt))
                    obs = refresh_observation_after_goal_change(env)
                except Exception as error:
                    reason = "goal_switch_error"
                    logger.record_step(step=step, goal_switch_error=str(error))
                    break
                after = _state_snapshot(env)
                if not _state_is_continuous(before, after):
                    state_continuity_violations += 1
                    reason = "goal_switch_error"
                    break
                if not waypoint.is_final_approach:
                    waypoint_records.append({
                        "step": step,
                        "cell": list(waypoint.cell),
                        "local_goal_xy": list(current_local_goal),
                        "world_goal_xy": current_world_goal.tolist(),
                        "delta_bearing_deg": current_delta_bearing_deg,
                        "reached": False,
                    })

            action = policy(obs)
            obs, _privileged, _reward, dones, _infos = env.step(action)
            robot_xy, _ = robot_pose(env)
            done = bool(dones[0].item())
            actual_path_length += float(np.linalg.norm(robot_xy - last_xy))
            last_xy = robot_xy
            global_distance = float(np.linalg.norm(global_goal - robot_xy))
            waypoint_distance = float(np.linalg.norm(current_world_goal - robot_xy))
            speed = robot_speed(env)
            collision = bool(env.maze_collision_buf[0].item())
            diagnostic_xy = robot_xy
            diagnostic_yaw = robot_pose(env)[1]
            diagnostic_speed = speed
            if collision or done:
                diagnostic_xy = (
                    env.terminal_position[0].detach().cpu().numpy().astype(np.float64)
                )
                diagnostic_yaw = float(env.terminal_yaw[0].detach().cpu().item())
                diagnostic_speed = float(env.terminal_speed[0].detach().cpu().item())
            measured_local_goal = np.asarray(
                world_to_local(diagnostic_xy, diagnostic_yaw, current_world_goal),
                dtype=np.float64,
            )
            local_goal_distance, local_goal_bearing = local_goal_polar(measured_local_goal)
            current_cell = None
            current_segment = None
            next_segment = None
            if current_waypoint_cell is not None:
                try:
                    current_cell, current_segment, next_segment = _diagnostic_bfs_context(
                        env, diagnostic_xy, current_waypoint_cell, current_next_bfs_cell
                    )
                except (TypeError, ValueError, IndexError):
                    pass
            origin = env.env_origins[0, :2].detach().cpu().numpy().astype(np.float64)
            nearest_surface_distance, robot_clearance = nearest_wall_clearance(
                diagnostic_xy - origin,
                env._maze_wall_centers,
                [float(env.cfg.maze.cell_size), float(env.cfg.maze.cell_size)],
                float(env.cfg.maze.robot_collision_radius),
            )
            cross_track_error = (
                point_to_segment_distance(diagnostic_xy, current_segment[0], current_segment[1])
                if current_segment is not None else None
            )
            collisions += int(collision)
            step_since_goal_switch = int(steps_since_goal_switch)
            logger.record_step(
                step=step,
                time_s=(step + 1) * float(env.dt),
                robot_xy=diagnostic_xy,
                global_goal_xy=global_goal,
                world_goal_xy=current_world_goal,
                local_goal_xy=current_local_goal,
                global_distance=global_distance,
                waypoint_distance=waypoint_distance,
                speed=diagnostic_speed,
                current_cell=(list(current_cell) if current_cell is not None else None),
                waypoint_cell=(list(current_waypoint_cell) if current_waypoint_cell is not None else None),
                next_bfs_cell=(list(current_next_bfs_cell) if current_next_bfs_cell is not None else None),
                robot_yaw=diagnostic_yaw,
                robot_speed=diagnostic_speed,
                local_goal_distance=local_goal_distance,
                local_goal_bearing=local_goal_bearing,
                delta_bearing_deg=current_delta_bearing_deg,
                collision=collision,
                steps_since_goal_switch=step_since_goal_switch,
                turn_aware_triggered=bool(
                    script_args.turn_aware and abs(current_delta_bearing_deg) >= 45.0
                ),
                reachability_filtered=current_reachability_filtered,
                raw_local_goal_xy=current_raw_local_goal,
                filtered_local_goal_xy=current_local_goal,
                reachability_clip_ratio=current_reachability_clip_ratio,
                nearest_wall_distance=nearest_surface_distance,
                nearest_wall_surface_distance=nearest_surface_distance,
                robot_clearance=robot_clearance,
                cross_track_error_to_current_bfs_segment=cross_track_error,
                action=action[0].detach().cpu().numpy(),
                action_clipped=action_was_clipped(env, action),
            )
            steps_since_goal_switch += 1

            if global_distance <= SUCCESS_DISTANCE_M and speed <= SUCCESS_SPEED_MPS:
                success = True
                reason = "global_success"
                final_approach_success = final_approach_entered
                break
            if collision:
                labels = classify_collision(
                    phase=planner.phase,
                    steps_since_goal_switch=step_since_goal_switch,
                    delta_bearing_deg=current_delta_bearing_deg,
                    waypoint_reached=False,
                    current_cell=current_cell,
                    waypoint_cell=current_waypoint_cell,
                    next_bfs_cell=current_next_bfs_cell,
                )
                collision_diagnostic = {
                    **labels,
                    "collision_step": int(step),
                    "collision_xy": diagnostic_xy.tolist(),
                    "collision_phase": planner.phase,
                    "steps_since_goal_switch": step_since_goal_switch,
                    "collision_local_goal_bearing": local_goal_bearing,
                    "collision_local_goal_distance": local_goal_distance,
                    "current_cell": list(current_cell) if current_cell is not None else None,
                    "waypoint_cell": list(current_waypoint_cell) if current_waypoint_cell is not None else None,
                    "next_bfs_cell": list(current_next_bfs_cell) if current_next_bfs_cell is not None else None,
                    "current_bfs_segment": current_segment,
                    "next_bfs_segment": next_segment,
                    "nearest_wall_surface_distance": nearest_surface_distance,
                    "robot_clearance": robot_clearance,
                    "cross_track_error_to_current_bfs_segment": cross_track_error,
                }
                reason = "collision"
                break
            if bool(dones[0].item()):
                reason = _failure_from_done(env)
                final_approach_timeout = final_approach_entered and reason == "timeout"
                break
            if final_approach_entered:
                if world_to_cell(robot_xy, env.maze_layout.shape, env.cfg.maze.cell_size) != final_goal_cell:
                    final_approach_escape = True
                    reason = "final_approach_escape"
                    break
                continue
            if waypoint_reached(
                waypoint_distance,
                speed,
                current_delta_bearing_deg,
                turn_aware=script_args.turn_aware,
            ):
                waypoint_reached_count += 1
                if waypoint_records:
                    waypoint_records[-1]["reached"] = True
                current_world_goal = None
                current_local_goal = None
    if final_approach_entered and not success and reason == "timeout":
        final_approach_timeout = True
    completion_time = steps_taken * float(env.dt)
    logger.finish(
        success=success,
        reason=reason,
        global_goal_xy=global_goal,
        waypoint_count=len(waypoint_records),
        local_waypoint_reached_count=waypoint_reached_count,
        waypoint_sequence=waypoint_records,
        collision_count=collisions,
        actual_path_length_m=actual_path_length,
        bfs_shortest_path_length_m=bfs_shortest_path_length,
        maze_spl=maze_spl(success, bfs_shortest_path_length, actual_path_length),
        completion_time_s=completion_time,
        final_distance=global_distance,
        coordinate_error_count=coordinate_errors,
        state_continuity_violation_count=state_continuity_violations,
        checkpoint_control_configuration_error_count=0,
        final_approach_entered=final_approach_entered,
        final_approach_success=final_approach_success,
        final_approach_timeout=final_approach_timeout,
        final_approach_escape=final_approach_escape,
        collision_diagnostic=(collision_diagnostic if collision else None),
        collision_step=(collision_diagnostic["collision_step"] if collision else None),
        collision_xy=(collision_diagnostic["collision_xy"] if collision else None),
        collision_phase=(collision_diagnostic["collision_phase"] if collision else None),
        collision_steps_since_goal_switch=(
            collision_diagnostic["steps_since_goal_switch"] if collision else None
        ),
        collision_local_goal_bearing=(
            collision_diagnostic["collision_local_goal_bearing"] if collision else None
        ),
        collision_local_goal_distance=(
            collision_diagnostic["collision_local_goal_distance"] if collision else None
        ),
        collision_current_bfs_segment=(
            collision_diagnostic["current_bfs_segment"] if collision else None
        ),
        collision_next_bfs_segment=(
            collision_diagnostic["next_bfs_segment"] if collision else None
        ),
    )
    return logger


def run_gate(args, script_args):
    output_dir = Path(script_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if script_args.smoke and script_args.reachability_envelope:
        raise ValueError("Oracle smoke must run without a reachability filter")
    # Envelope loading is explicit so a future reachability-aware run cannot
    # silently substitute a guessed geometry. It is deliberately disabled in
    # the Raw smoke/100-episode gates.
    envelope = (
        None
        if script_args.smoke
        else load_envelope(script_args.reachability_envelope)
        if script_args.reachability_envelope
        else None
    )
    env, _runner, policy = _load_maze(args, script_args.checkpoint)
    reachable = np.asarray(env._maze_goal_positions, dtype=np.float64)
    manifest_path = (
        Path(script_args.episode_manifest)
        if script_args.episode_manifest
        else output_dir / "episode_manifest.json"
    )
    manifest = _episode_manifest(manifest_path, reachable, script_args.episodes)
    results = []
    # Reuse one simulator and reset the robot between manifest entries. This
    # keeps episode state isolated while avoiding repeated PhysX context
    # allocation, which fragments the CUDA allocator over long Raw runs.
    origin = env.env_origins[0, :2].detach().cpu().numpy()
    try:
        for episode_id, entry in enumerate(manifest):
            global_goal = np.asarray(entry["goal_xy"], dtype=np.float64) + origin
            planner = OracleEpisodePlanner(
                env.maze_layout,
                env.maze_layout.shape,
                env.cfg.maze.cell_size,
                reachability=envelope,
            )
            logger = _run_episode(
                env,
                policy,
                planner,
                script_args,
                episode_id,
                global_goal,
                int(entry["goal_index"]),
            )
            logger.write_json(output_dir / f"episode_{episode_id:04d}.json")
            results.append(logger.summary)
            del logger
            _release_episode_cache()
    finally:
        env.gym.destroy_sim(env.sim)
    summary = summarize_oracle_results(results, protocol="oracle_maze_120s")
    summary.update({
        "gate": "oracle_maze_smoke" if script_args.smoke else "oracle_maze_raw",
        "smoke": bool(script_args.smoke),
        "reachability_filter": bool(envelope is not None),
        "turn_aware": bool(script_args.turn_aware),
        "maze_episode_budget_s": 120.0,
        "low_level_episode_budget_s": 60.0,
        "episode_manifest": str(manifest_path),
    })
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    script_args = _parse_script_args()
    run_gate(_isaac_args(), script_args)
