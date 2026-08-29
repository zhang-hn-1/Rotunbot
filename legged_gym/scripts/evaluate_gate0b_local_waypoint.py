"""Gate0B: short-range local-waypoint tracking with a frozen P2P policy."""

import os
import math
from pathlib import Path

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


CHECKPOINT = (
    "/home/jason/SphericalRobot_LeggedGym-master-new-map/logs/"
    "rotunbot_target_repro/Aug11_16-44-07_/model_2050.pt"
)
WAYPOINT_RADIUS = float(os.environ.get("GATE0B_WAYPOINT_RADIUS", "0.35"))
MIN_WAYPOINT_DISTANCE = float(os.environ.get("GATE0B_MIN_DISTANCE", "0.5"))
MAX_WAYPOINT_DISTANCE = float(os.environ.get("GATE0B_MAX_DISTANCE", "2.0"))
MIN_WAYPOINT_BEARING_DEG = float(os.environ.get("GATE0B_MIN_BEARING_DEG", "-180"))
MAX_WAYPOINT_BEARING_DEG = float(os.environ.get("GATE0B_MAX_BEARING_DEG", "180"))


def _yaw_from_quaternion(quat):
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def evaluate(args):
    checkpoint_path = Path(os.environ.get("GATE0B_P2P_CHECKPOINT", CHECKPOINT)).resolve()
    episodes = int(os.environ.get("GATE0B_EPISODES", "100"))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Gate0B checkpoint does not exist: {checkpoint_path}")

    args.task = os.environ.get("GATE0B_TASK", "rotunbot_target_repro")
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.commands.random_start_yaw = True
    env_cfg.commands.target_curriculum = False

    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    env.data_print = False
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    runner.load(str(checkpoint_path))
    policy = runner.get_inference_policy(device=env.device)

    def configure_waypoint():
        position = env.root_states[0, :2].clone()
        yaw = _yaw_from_quaternion(env.base_quat)[0]
        distance = torch.rand((), device=env.device) * (
            MAX_WAYPOINT_DISTANCE - MIN_WAYPOINT_DISTANCE
        ) + MIN_WAYPOINT_DISTANCE
        angle = torch.rand((), device=env.device) * math.radians(
            MAX_WAYPOINT_BEARING_DEG - MIN_WAYPOINT_BEARING_DEG
        ) + math.radians(MIN_WAYPOINT_BEARING_DEG)
        local_delta = torch.stack((distance * torch.cos(angle), distance * torch.sin(angle)))
        world_delta = torch.stack(
            (
                torch.cos(yaw) * local_delta[0] - torch.sin(yaw) * local_delta[1],
                torch.sin(yaw) * local_delta[0] + torch.cos(yaw) * local_delta[1],
            )
        )
        target = position + world_delta
        env.commands[0, :2] = target
        env.goal_dist[0] = distance
        env.last_goal_dist[0] = distance
        env.compute_observations()
        return env.get_observations(), target, distance

    env.reset()
    # reset() returns observations; overwrite the sampled global target with a
    # short local target before the first policy action.
    obs, target, initial_distance = configure_waypoint()

    reached = 0
    divergent = 0
    near_misses = 0
    lengths = []
    records = []
    episode_steps = 0
    episode_min_distance = float(initial_distance.item())
    completed = 0

    print(
        "Gate0B evaluation: "
        f"checkpoint={checkpoint_path}, episodes={episodes}, device={env.device}, "
        f"waypoint_distance=[{MIN_WAYPOINT_DISTANCE:.1f}, {MAX_WAYPOINT_DISTANCE:.1f}]m, "
        f"arrival_radius={WAYPOINT_RADIUS:.2f}m",
        flush=True,
    )

    with torch.no_grad():
        while completed < episodes:
            before_distance = float(
                torch.linalg.norm(target - env.root_states[0, :2]).item()
            )
            episode_min_distance = min(episode_min_distance, before_distance)
            actions = policy(obs)
            obs, _, _, dones, _ = env.step(actions)
            episode_steps += 1

            done = bool(dones[0].item())
            after_distance = float(
                torch.linalg.norm(target - env.root_states[0, :2]).item()
            )
            if not done:
                episode_min_distance = min(episode_min_distance, after_distance)
            reached_now = after_distance <= WAYPOINT_RADIUS
            if done and bool(env.success_buf[0].item()):
                reached_now = True

            if reached_now or done:
                final_distance = (
                    float(env.terminal_goal_dist[0].item())
                    if done
                    else after_distance
                )
                episode_min_distance = min(episode_min_distance, final_distance)
                is_reached = bool(reached_now)
                is_unstable = bool(
                    env.terminal_unstable[0].item()
                    or env.terminal_out_of_bounds[0].item()
                ) if done else False
                is_divergent = (not is_reached) and (
                    is_unstable or episode_min_distance > 1.0
                )
                reached += int(is_reached)
                divergent += int(is_divergent)
                near_misses += int(not is_reached and not is_divergent)
                lengths.append(episode_steps)
                completed += 1
                records.append(
                    {
                        "episode": completed,
                        "reached": is_reached,
                        "divergent": is_divergent,
                        "initial_distance": float(initial_distance.item()),
                        "min_distance": episode_min_distance,
                        "final_distance": final_distance,
                        "steps": episode_steps,
                    }
                )
                if not is_reached or completed <= 5 or completed % 20 == 0:
                    print(records[-1], flush=True)

                if completed >= episodes:
                    break
                episode_steps = 0
                if done:
                    # The simulator already reset the environment during step.
                    obs, target, initial_distance = configure_waypoint()
                else:
                    env.reset_idx(torch.tensor([0], device=env.device, dtype=torch.long))
                    obs, target, initial_distance = configure_waypoint()
                episode_min_distance = float(initial_distance.item())

    reach_rate = reached / episodes
    divergence_rate = divergent / episodes
    print(
        "GATE0B SUMMARY "
        f"episodes={episodes} waypoint_reach_rate={reach_rate:.2%} "
        f"divergence_rate={divergence_rate:.2%} "
        f"near_miss_rate={near_misses / episodes:.2%} "
        f"mean_steps={sum(lengths) / len(lengths):.1f}",
        flush=True,
    )
    if reach_rate < 0.95 or divergence_rate > 0.02:
        raise RuntimeError(
            "Gate0B failed: required waypoint reach >= 95% and divergence <= 2%"
        )
    print("Gate0B PASS", flush=True)


if __name__ == "__main__":
    evaluate(get_args())
