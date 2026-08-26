"""Coverage Gate for the frozen uniform-4150 local-goal executor."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from legged_gym.navigation.baseline import (
    CHECKPOINT_RELATIVE_PATH,
    LOCAL_WAYPOINT_DISTANCE_M,
)
from legged_gym.navigation.evaluation_logging import EpisodeLogger
from legged_gym.navigation.frozen_p2p import (
    action_was_clipped,
    load_frozen_p2p,
    refresh_observation_after_goal_change,
    robot_pose,
    robot_speed,
    set_temporary_world_goal,
)
from legged_gym.navigation.local_goal_adapter import local_to_world


DISTANCES_M = (0.5, 1.0, 1.5, 2.0)
BEARINGS_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0)


def _parse_script_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT_RELATIVE_PATH)
    parser.add_argument("--output-dir", default="logs/hierarchical_navigation/coverage")
    parser.add_argument("--episodes-per-case", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=3002)
    return parser.parse_args()


def _isaac_args():
    from legged_gym.navigation.isaac_compat import install_isaac_gym_compat

    install_isaac_gym_compat()
    from legged_gym import envs  # noqa: F401
    from legged_gym.utils import get_args

    saved = sys.argv
    sys.argv = [saved[0], "--headless", "--rl_device=cuda:0", "--sim_device=cuda:0"]
    try:
        return get_args()
    finally:
        sys.argv = saved


def build_success_rate_matrix(case_results, distances=DISTANCES_M, bearings=BEARINGS_DEG):
    """Return a deterministic distance-by-bearing success-rate matrix."""
    lookup = {
        (float(row["distance_m"]), float(row["bearing_deg"])): float(row["success_rate"])
        for row in case_results
    }
    return [
        [lookup[(float(distance), float(bearing))] for bearing in bearings]
        for distance in distances
    ]


def _run_case(env, policy, distance, bearing_deg, repetition, script_args, output_dir, index):
    obs, _ = env.reset()
    robot_xy, robot_yaw = robot_pose(env)
    local_goal = float(distance) * np.array([
        np.cos(np.deg2rad(bearing_deg)),
        np.sin(np.deg2rad(bearing_deg)),
    ])
    world_goal = local_to_world(robot_xy, robot_yaw, local_goal)
    set_temporary_world_goal(env, world_goal)
    obs = refresh_observation_after_goal_change(env)
    logger = EpisodeLogger({
        "gate": "single_local_goal_coverage",
        "distance_m": float(distance),
        "bearing_deg": float(bearing_deg),
        "repetition": int(repetition),
    })
    success = False
    reason = "timeout"
    done = False
    current_distance = float("inf")
    for step in range(script_args.max_steps):
        action = policy(obs)
        obs, _privileged, _reward, dones, _infos = env.step(action)
        current_xy, _ = robot_pose(env)
        current_distance = float(np.linalg.norm(world_goal - current_xy))
        speed = robot_speed(env)
        logger.record_step(
            step=step,
            time_s=(step + 1) * float(env.dt),
            robot_xy=current_xy,
            world_goal_xy=world_goal,
            distance=current_distance,
            speed=speed,
            action=action[0].detach().cpu().numpy(),
            action_clipped=action_was_clipped(env, action),
        )
        done = bool(dones[0].item())
        terminal_distance = float(env.terminal_goal_dist[0].item())
        local_reached = (
            terminal_distance <= LOCAL_WAYPOINT_DISTANCE_M
            if done else current_distance <= LOCAL_WAYPOINT_DISTANCE_M
        )
        if local_reached:
            success = True
            reason = "local_goal"
            break
        if done:
            if bool(env.terminal_unstable[0].item()):
                reason = "unstable"
            elif bool(env.terminal_timeout[0].item()):
                reason = "timeout"
            elif bool(env.terminal_out_of_bounds[0].item()):
                reason = "out_of_bounds"
            else:
                reason = "divergence"
            break
    logger.finish(
        success=success,
        reason=reason,
        completion_time_s=(step + 1) * float(env.dt),
        final_distance=(
            float(env.terminal_goal_dist[0].item()) if done else current_distance
        ),
        minimum_distance=min(row["distance"] for row in logger.trajectory),
        action_clipping=any(row["action_clipped"] for row in logger.trajectory),
    )
    logger.write_json(output_dir / f"episode_{index:04d}.json")
    return logger.summary


def run_gate(args, script_args):
    if script_args.episodes_per_case < 1:
        raise ValueError("episodes-per-case must be positive")
    if script_args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    output_dir = Path(script_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env, _runner, policy = load_frozen_p2p(args, script_args.checkpoint)
    episode_results = []
    case_summaries = []
    try:
        index = 0
        for distance in DISTANCES_M:
            for bearing_deg in BEARINGS_DEG:
                case_results = []
                for repetition in range(script_args.episodes_per_case):
                    result = _run_case(
                        env, policy, distance, bearing_deg, repetition,
                        script_args, output_dir, index,
                    )
                    index += 1
                    case_results.append(result)
                    episode_results.append(result)
                case = {
                    "distance_m": float(distance),
                    "bearing_deg": float(bearing_deg),
                    "episodes": len(case_results),
                    "success_rate": float(np.mean([row["success"] for row in case_results])),
                    "timeout_count": int(sum(row["reason"] == "timeout" for row in case_results)),
                    "minimum_distances_m": [row["minimum_distance"] for row in case_results],
                    "completion_times_s": [row["completion_time_s"] for row in case_results],
                }
                case_summaries.append(case)
    finally:
        env.gym.destroy_sim(env.sim)

    summary = {
        "gate": "single_local_goal_coverage",
        "distances_m": list(DISTANCES_M),
        "bearings_deg": list(BEARINGS_DEG),
        "episodes_per_case": int(script_args.episodes_per_case),
        "local_waypoint_distance_m": LOCAL_WAYPOINT_DISTANCE_M,
        "success_rate_matrix": build_success_rate_matrix(case_summaries),
        "case_summaries": case_summaries,
        "episodes": episode_results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    script_args = _parse_script_args()
    run_gate(_isaac_args(), script_args)
