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
    action_was_clipped,
    load_frozen_runner,
    refresh_observation_after_goal_change,
    robot_pose,
    robot_speed,
    set_temporary_world_goal,
)
from legged_gym.navigation.goal_switch import GoalSwitchController
from legged_gym.navigation.hierarchical_maze import HierarchicalMazeCfg, HierarchicalMazeP2P
from legged_gym.navigation.oracle_episode import OracleEpisodePlanner
from legged_gym.navigation.reachability import load_envelope


def _parse_script_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT_RELATIVE_PATH)
    parser.add_argument("--output-dir", default="logs/hierarchical_navigation/oracle_maze")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=6002)
    parser.add_argument("--waypoint-radius", type=float, default=LOCAL_WAYPOINT_DISTANCE_M)
    parser.add_argument("--reachability-envelope")
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
    env_cfg.control.direct_velocity_gain_randomize = False
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
    return env, runner, runner.get_inference_policy(device=args.rl_device)


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


def _run_episode(env, policy, planner, rng, script_args, episode_id):
    obs, _ = env.reset()
    _set_center_start(env)
    reachable = np.asarray(env._maze_goal_positions)
    goal_index = int(rng.integers(0, len(reachable)))
    global_goal = reachable[goal_index] + env.env_origins[0, :2].detach().cpu().numpy()
    switcher = GoalSwitchController(env)
    switcher.set_intermediate_goal_mode(True)
    logger = EpisodeLogger({"gate": "oracle_maze", "episode_id": episode_id, "maze_seed": int(env.cfg.maze.seed)})
    waypoint_records = []
    current_world_goal = None
    current_local_goal = None
    collisions = 0
    success = False
    reason = "timeout"
    try:
        for step in range(script_args.max_steps):
            robot_xy, robot_yaw = robot_pose(env)
            if current_world_goal is None:
                waypoint = planner.next_local_waypoint(robot_xy, robot_yaw, global_goal)
                current_local_goal = waypoint.filtered_local_goal_xy
                current_world_goal = np.asarray(waypoint.temporary_world_goal_xy)
                switcher.update_world_goal(current_world_goal, time_s=step * float(env.dt))
                waypoint_records.append({"step": step, "cell": list(waypoint.cell), "local_goal_xy": list(current_local_goal), "world_goal_xy": current_world_goal.tolist()})
                obs = refresh_observation_after_goal_change(env)

            action = policy(obs)
            obs, _privileged, _reward, dones, _infos = env.step(action)
            robot_xy, _ = robot_pose(env)
            global_distance = float(np.linalg.norm(global_goal - robot_xy))
            waypoint_distance = float(np.linalg.norm(current_world_goal - robot_xy))
            speed = robot_speed(env)
            collision = bool(env.maze_collision_buf[0].item())
            collisions += int(collision)
            logger.record_step(
                step=step,
                time_s=(step + 1) * float(env.dt),
                robot_xy=robot_xy,
                global_goal_xy=global_goal,
                world_goal_xy=current_world_goal,
                local_goal_xy=current_local_goal,
                global_distance=global_distance,
                waypoint_distance=waypoint_distance,
                speed=speed,
                collision=collision,
                action=action[0].detach().cpu().numpy(),
                action_clipped=action_was_clipped(env, action),
            )

            if global_distance <= SUCCESS_DISTANCE_M and speed <= SUCCESS_SPEED_MPS:
                success = True
                reason = "global_success"
                break
            if collision:
                reason = "collision"
                break
            if waypoint_distance <= script_args.waypoint_radius:
                waypoint = planner.next_local_waypoint(robot_xy, robot_pose(env)[1], global_goal)
                current_local_goal = waypoint.filtered_local_goal_xy
                current_world_goal = np.asarray(waypoint.temporary_world_goal_xy)
                switcher.update_world_goal(current_world_goal, time_s=(step + 1) * float(env.dt))
                waypoint_records.append({"step": step + 1, "cell": list(waypoint.cell), "local_goal_xy": list(current_local_goal), "world_goal_xy": current_world_goal.tolist()})
                obs = refresh_observation_after_goal_change(env)
            if bool(dones[0].item()):
                reason = "unstable_or_timeout"
                break
    finally:
        env.gym.destroy_sim(env.sim)
    logger.finish(
        success=success,
        reason=reason,
        global_goal_xy=global_goal,
        waypoint_count=len(waypoint_records),
        waypoint_sequence=waypoint_records,
        collision_count=collisions,
        completion_time_s=(step + 1) * float(env.dt),
        final_distance=global_distance,
    )
    return logger


def run_gate(args, script_args):
    output_dir = Path(script_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Envelope loading is explicit so a future reachability-aware run cannot
    # silently substitute a guessed geometry. It is optional for the first map gate.
    envelope = load_envelope(script_args.reachability_envelope) if script_args.reachability_envelope else None
    env, _runner, policy = _load_maze(args, script_args.checkpoint)
    planner = OracleEpisodePlanner(
        env.maze_layout,
        env.maze_layout.shape,
        env.cfg.maze.cell_size,
        reachability=envelope,
    )
    rng = np.random.default_rng(0)
    results = []
    # Each episode gets a fresh simulator to avoid Isaac Gym state leakage.
    for episode_id in range(script_args.episodes):
        if episode_id:
            env, _runner, policy = _load_maze(args, script_args.checkpoint)
            planner = OracleEpisodePlanner(env.maze_layout, env.maze_layout.shape, env.cfg.maze.cell_size, envelope)
        logger = _run_episode(env, policy, planner, rng, script_args, episode_id)
        logger.write_json(output_dir / f"episode_{episode_id:04d}.json")
        results.append(logger.summary)
    summary = {
        "gate": "oracle_maze",
        "episodes": len(results),
        "global_success_rate": float(np.mean([row["success"] for row in results])) if results else 0.0,
        "collision_rate": float(np.mean([row["reason"] == "collision" for row in results])) if results else 0.0,
        "timeout_rate": float(np.mean([row["reason"] in ("timeout", "unstable_or_timeout") for row in results])) if results else 0.0,
        "average_completion_time_s": float(np.mean([row["completion_time_s"] for row in results])) if results else 0.0,
        "average_waypoint_count": float(np.mean([row["waypoint_count"] for row in results])) if results else 0.0,
        "episodes_detail": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    script_args = _parse_script_args()
    run_gate(_isaac_args(), script_args)
