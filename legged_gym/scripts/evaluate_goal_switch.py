"""Gate 2: switch local goals without resetting the frozen P2P episode."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from legged_gym.navigation.baseline import (
    CHECKPOINT_RELATIVE_PATH,
    LOCAL_WAYPOINT_DISTANCE_M,
    SUCCESS_DISTANCE_M,
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
from legged_gym.navigation.goal_switch import GoalSwitchController
from legged_gym.navigation.local_goal_adapter import local_to_world


SEQUENCES = {
    "straight": ((1.0, 0.0), (1.0, 0.0), (1.0, 0.0)),
    "l_shape": ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
    "s_shape": ((1.0, 0.6), (1.0, -1.2), (1.0, 0.6)),
    "rectangle": ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)),
    "forward_then_lateral": ((1.0, 0.0), (0.0, 1.0)),
    "lateral_then_forward": ((0.0, 1.0), (1.0, 0.0)),
    "sharp_direction_change": ((1.0, 0.0), (-1.0, 0.0)),
}


def _parse_script_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT_RELATIVE_PATH)
    parser.add_argument("--output-dir", default="logs/hierarchical_navigation/gate2")
    parser.add_argument("--max-steps-per-waypoint", type=int, default=3002)
    parser.add_argument("--radius", type=float, default=LOCAL_WAYPOINT_DISTANCE_M)
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


def _snapshot(env):
    history = getattr(env, "obs_history", None)
    # The P2P policy history stores normalized actions in last_actions;
    # last_output_actions is a physical actuator-target diagnostic buffer.
    previous = getattr(env, "last_actions", None)
    if hasattr(previous, "detach"):
        previous = previous.detach().cpu().numpy().copy()
    elif previous is not None:
        previous = np.asarray(previous).copy()
    episode = getattr(env, "episode_length_buf", None)
    if hasattr(episode, "detach"):
        episode = episode.detach().cpu().numpy().copy()
    elif episode is not None:
        episode = np.asarray(episode).copy()
    reset = getattr(env, "reset_buf", None)
    if hasattr(reset, "detach"):
        reset = reset.detach().cpu().numpy().copy()
    elif reset is not None:
        reset = np.asarray(reset).copy()
    return {
        "episode_length": episode,
        "history_length": len(history) if history is not None else None,
        "previous_action": previous,
        "reset": reset,
    }


def _same_snapshot(before, after):
    return (
        np.array_equal(before["episode_length"], after["episode_length"])
        and before["history_length"] == after["history_length"]
        and np.array_equal(before["previous_action"], after["previous_action"])
        and np.array_equal(before["reset"], after["reset"])
    )


def _run_sequence(env, policy, name, local_sequence, script_args, sequence_id):
    obs, _ = env.reset()
    switcher = GoalSwitchController(env)
    switcher.set_intermediate_goal_mode(True)
    logger = EpisodeLogger({"gate": "goal_switch", "sequence": name, "sequence_id": sequence_id})
    waypoint_results = []
    all_success = True
    total_steps = 0
    previous_local = None
    try:
        for waypoint_index, local_goal in enumerate(local_sequence):
            robot_xy, robot_yaw = robot_pose(env)
            world_goal = local_to_world(robot_xy, robot_yaw, local_goal)
            before = _snapshot(env)
            event = switcher.update_world_goal(world_goal, time_s=total_steps * float(env.dt))
            after = _snapshot(env)
            continuity = _same_snapshot(before, after)
            obs = refresh_observation_after_goal_change(env)
            reached = False
            reason = "timeout"
            stop_duration_s = 0.0
            for _ in range(script_args.max_steps_per_waypoint):
                action = policy(obs)
                obs, _privileged, _reward, dones, _infos = env.step(action)
                total_steps += 1
                current_xy, _ = robot_pose(env)
                distance = float(np.linalg.norm(world_goal - current_xy))
                speed = robot_speed(env)
                logger.record_step(
                    waypoint_index=waypoint_index,
                    step=total_steps,
                    time_s=total_steps * float(env.dt),
                    robot_xy=current_xy,
                    world_goal_xy=world_goal,
                    distance=distance,
                    speed=speed,
                    action=action[0].detach().cpu().numpy(),
                    action_clipped=action_was_clipped(env, action),
                )
                if distance <= script_args.radius:
                    reached = True
                    reason = "waypoint_reached"
                    break
                if bool(dones[0].item()):
                    reason = "unstable_or_timeout"
                    break
            if not reached:
                all_success = False
            waypoint_results.append(
                {
                    "waypoint_index": waypoint_index,
                    "local_goal_xy": list(local_goal),
                    "world_goal_xy": world_goal.tolist(),
                    "reached": reached,
                    "reason": reason,
                    "switch_index": event.switch_index,
                    "state_continuous": continuity,
                    "action_discontinuity": switcher.measure_action_discontinuity(action) if "action" in locals() else None,
                    "stop_duration_s": stop_duration_s,
                }
            )
            previous_local = list(local_goal)
            if not reached:
                break
    finally:
        env.gym.destroy_sim(env.sim)
    logger.finish(
        success=all_success,
        reason="sequence_complete" if all_success else "waypoint_failure",
        waypoint_count=len(waypoint_results),
        waypoint_results=waypoint_results,
        goal_switch_latency_s=(total_steps / max(len(waypoint_results), 1)) * float(env.dt),
        previous_local_goal_xy=previous_local,
    )
    return logger


def run_gate(args, script_args):
    output_dir = Path(script_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for sequence_id, (name, sequence) in enumerate(SEQUENCES.items()):
        env, _runner, policy = load_frozen_p2p(args, script_args.checkpoint)
        logger = _run_sequence(env, policy, name, sequence, script_args, sequence_id)
        logger.write_json(output_dir / f"{name}.json")
        results.append(logger.summary)
    summary = {
        "gate": "goal_switch",
        "sequence_count": len(results),
        "sequence_success_rate": float(np.mean([row["success"] for row in results])) if results else 0.0,
        "sequences": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    script_args = _parse_script_args()
    run_gate(_isaac_args(), script_args)
