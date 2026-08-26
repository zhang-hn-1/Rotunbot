"""Gate 1: evaluate the frozen uniform-4150 policy on local goals."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from legged_gym.navigation.baseline import (
    CHECKPOINT_RELATIVE_PATH,
    SUCCESS_DISTANCE_M,
    SUCCESS_SPEED_MPS,
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


def _parse_script_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT_RELATIVE_PATH)
    parser.add_argument("--output-dir", default="logs/hierarchical_navigation/gate1")
    parser.add_argument("--episodes-per-goal", type=int, default=1)
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


def run_gate(args, script_args):
    env, _runner, policy = load_frozen_p2p(args, script_args.checkpoint)
    output_dir = Path(script_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    cases = []
    for distance in (0.5, 1.0, 1.5):
        for bearing_deg in (0.0, 30.0, -30.0, 45.0, -45.0):
            cases.append((distance, bearing_deg))
    results = []
    try:
        for case_index, (distance, bearing_deg) in enumerate(cases):
            for repetition in range(script_args.episodes_per_goal):
                obs, _ = env.reset()
                robot_xy, robot_yaw = robot_pose(env)
                # Randomize the commanded local bearing and preserve the
                # simulator's own reset position/yaw randomization.
                local_goal = distance * np.array(
                    [np.cos(np.deg2rad(bearing_deg)), np.sin(np.deg2rad(bearing_deg))]
                )
                world_goal = local_to_world(robot_xy, robot_yaw, local_goal)
                set_temporary_world_goal(env, world_goal)
                obs = refresh_observation_after_goal_change(env)
                logger = EpisodeLogger(
                    {
                        "gate": "single_local_goal",
                        "case_index": case_index,
                        "distance_m": distance,
                        "bearing_deg": bearing_deg,
                        "repetition": repetition,
                    }
                )
                done = False
                success = False
                reason = "timeout"
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
                    if done:
                        success = bool(env.terminal_goal_dist[0].item() <= SUCCESS_DISTANCE_M and env.terminal_speed[0].item() <= SUCCESS_SPEED_MPS)
                        if success:
                            reason = "local_goal"
                        elif bool(env.terminal_unstable[0].item()):
                            reason = "unstable"
                        elif bool(env.terminal_timeout[0].item()):
                            reason = "timeout"
                        else:
                            reason = "divergence"
                        break
                logger.finish(
                    success=success,
                    reason=reason,
                    completion_time_s=(step + 1) * float(env.dt),
                    final_distance=float(env.terminal_goal_dist[0].item()) if done else current_distance,
                    minimum_distance=min(row["distance"] for row in logger.trajectory),
                    action_clipping=any(row["action_clipped"] for row in logger.trajectory),
                )
                episode_path = output_dir / f"episode_{len(results):04d}.json"
                logger.write_json(episode_path)
                results.append(logger.summary)
    finally:
        env.gym.destroy_sim(env.sim)
    summary = {
        "gate": "single_local_goal",
        "episodes": len(results),
        "success_rate": float(np.mean([row["success"] for row in results])) if results else 0.0,
        "timeout_rate": float(np.mean([row["reason"] == "timeout" for row in results])) if results else 0.0,
        "divergence_rate": float(np.mean([row["reason"] in ("divergence", "unstable") for row in results])) if results else 0.0,
        "episodes_detail": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    script_args = _parse_script_args()
    run_gate(_isaac_args(), script_args)
