"""Closed-loop Oracle collector; a depth provider is injected at runtime."""

import argparse
import importlib
import sys

import numpy as np

from legged_gym.navigation.isaac_compat import install_isaac_gym_compat
install_isaac_gym_compat()

from legged_gym.navigation.dataset import ClosedLoopDatasetWriter, OracleSample
from legged_gym.navigation.frozen_p2p import (
    load_frozen_runner,
    refresh_observation_after_goal_change,
    robot_pose,
    robot_speed,
    set_temporary_world_goal,
)
from legged_gym.navigation.goal_switch import GoalSwitchController
from legged_gym.navigation.hierarchical_maze import HierarchicalMazeCfg, HierarchicalMazeP2P
from legged_gym.navigation.oracle_episode import OracleEpisodePlanner


def _parse_script_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provider", required=True, help="module:function returning a DepthFrameProvider")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=6002)
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


def _load_provider(spec):
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    provider = factory()
    if not callable(getattr(provider, "get_frame", None)):
        raise TypeError("depth provider must expose get_frame()")
    return provider


def _load_env(args, checkpoint):
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
    env = HierarchicalMazeP2P(env_cfg, sim_params, args.physics_engine, args.sim_device, args.headless)
    runner = load_frozen_runner(args, env, train_cfg, checkpoint)
    return env, runner.get_inference_policy(device=args.rl_device)


def collect(args, script_args):
    provider = _load_provider(script_args.provider)
    env, policy = _load_env(args, script_args.checkpoint)
    planner = OracleEpisodePlanner(env.maze_layout, env.maze_layout.shape, env.cfg.maze.cell_size)
    rng = np.random.default_rng(0)
    output = ClosedLoopDatasetWriter(script_args.output_dir)
    try:
        for episode_id in range(script_args.episodes):
            obs, _ = env.reset()
            from legged_gym.scripts.evaluate_oracle_maze import _set_center_start

            _set_center_start(env)
            reachable = np.asarray(env._maze_goal_positions)
            goal = reachable[int(rng.integers(0, len(reachable)))] + env.env_origins[0, :2].detach().cpu().numpy()
            switcher = GoalSwitchController(env)
            switcher.set_intermediate_goal_mode(True)
            current_waypoint = None
            previous_local = (0.0, 0.0)
            for step in range(script_args.max_steps):
                robot_xy, robot_yaw = robot_pose(env)
                if current_waypoint is None:
                    current_waypoint = planner.next_local_waypoint(robot_xy, robot_yaw, goal)
                    set_temporary_world_goal(env, current_waypoint.temporary_world_goal_xy)
                    obs = refresh_observation_after_goal_change(env)
                depth = np.asarray(provider.get_frame())
                sample = OracleSample(
                    depth=depth,
                    robot_xy=robot_xy,
                    robot_yaw=robot_yaw,
                    global_goal_xy=goal,
                    local_goal_xy=current_waypoint.filtered_local_goal_xy,
                    temporary_world_goal_xy=current_waypoint.temporary_world_goal_xy,
                    previous_local_goal_xy=previous_local,
                    collision=bool(env.maze_collision_buf[0].item()),
                    timestamp_s=step * float(env.dt),
                    episode_id=episode_id,
                    waypoint_index=len(switcher.switches),
                )
                output.append(sample)
                action = policy(obs)
                obs, _privileged, _reward, dones, _infos = env.step(action)
                robot_xy, _ = robot_pose(env)
                if np.linalg.norm(goal - robot_xy) <= 0.20 and robot_speed(env) <= 0.10:
                    break
                if np.linalg.norm(np.asarray(current_waypoint.temporary_world_goal_xy) - robot_xy) <= 0.35:
                    previous_local = current_waypoint.filtered_local_goal_xy
                    current_waypoint = None
                if bool(dones[0].item()):
                    break
    finally:
        output.close()
        env.gym.destroy_sim(env.sim)
    print(f"Closed-loop Oracle dataset written to {script_args.output_dir}", flush=True)


if __name__ == "__main__":
    script_args = _parse_script_args()
    collect(_isaac_args(), script_args)
