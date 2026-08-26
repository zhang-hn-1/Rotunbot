"""Evaluate a trained ``rotunbot_maze`` policy with task-native metrics.

This is intentionally separate from ``play.py``: the latter evaluates the
legacy point-to-point task and does not know about maze collisions or the
latched global goal used by :class:`RotunbotMaze`.

Environment variables:
    MAZE_EVAL_EPISODES: number of completed episodes (default: 20)
    MAZE_EVAL_MAX_STEPS: optional per-episode step cap
"""

import os

import isaacgym  # noqa: F401  (must be imported before torch)
import torch

from legged_gym.envs import *  # noqa: F401,F403  (registers tasks)
from legged_gym.utils import get_args, task_registry


def _as_bool(value):
    return bool(value.item()) if hasattr(value, "item") else bool(value)


def evaluate(args):
    args.task = "rotunbot_maze"
    episodes = int(os.environ.get("MAZE_EVAL_EPISODES", "20"))
    if episodes <= 0:
        raise ValueError("MAZE_EVAL_EPISODES must be positive")

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.maze.terminate_on_collision = True
    env_cfg.commands.random_start_yaw = False

    # A direct checkpoint evaluation must not create or append TensorBoard
    # runs.  make_alg_runner still resolves the checkpoint from logs/.
    train_cfg.runner.resume = True
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.reset()
    max_steps = int(os.environ.get("MAZE_EVAL_MAX_STEPS", str(int(env.max_episode_length))))
    successes = 0
    collisions = 0
    timeouts = 0
    other_failures = 0
    lengths = []
    episode_step = 0
    completed = 0

    print(
        "Evaluation: task=rotunbot_maze, episodes={}, checkpoint={}".format(
            episodes, train_cfg.runner.checkpoint
        ),
        flush=True,
    )
    with torch.no_grad():
        while completed < episodes:
            actions = policy(obs)
            obs, _, _, dones, _ = env.step(actions)
            episode_step += 1

            # The environment resets immediately inside step(), but these
            # buffers retain the terminal reason for the just-finished episode.
            done = _as_bool(dones[0])
            forced_timeout = episode_step >= max_steps
            if done or forced_timeout:
                success = _as_bool(env.success_buf[0]) and not forced_timeout
                collision = _as_bool(env.maze_collision_buf[0]) and not success
                timeout = _as_bool(env.time_out_buf[0]) or forced_timeout
                successes += int(success)
                collisions += int(collision)
                timeouts += int(timeout and not success and not collision)
                other_failures += int(not success and not collision and not timeout)
                lengths.append(episode_step)
                completed += 1
                print(
                    "episode {:>3}: success={} collision={} timeout={} steps={}".format(
                        completed, int(success), int(collision), int(timeout), episode_step
                    ),
                    flush=True,
                )
                episode_step = 0
                if forced_timeout and not done:
                    # Reset just this single environment so the next metric
                    # starts at a clean spawn state.
                    env.reset_idx(torch.tensor([0], device=env.device, dtype=torch.long))
                    obs = env.get_observations()

    print(
        "SUMMARY episodes={} success_rate={:.2%} collision_rate={:.2%} "
        "timeout_rate={:.2%} other_failure_rate={:.2%} mean_steps={:.1f}".format(
            episodes,
            successes / episodes,
            collisions / episodes,
            timeouts / episodes,
            other_failures / episodes,
            sum(lengths) / len(lengths),
        ),
        flush=True,
    )


if __name__ == "__main__":
    evaluate(get_args())
