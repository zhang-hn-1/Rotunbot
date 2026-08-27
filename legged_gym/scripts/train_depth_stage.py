"""Train a bounded depth-local curriculum stage without changing the V0 task."""

import argparse
import sys

import isaacgym  # noqa: F401 - must precede torch in Isaac Gym Preview 4
import numpy as np

if not hasattr(np, "float"):
    np.float = float

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", type=int, choices=(0, 1), default=0)
    parser.add_argument("--depth-backend", choices=("fallback", "isaacgym"), default="fallback")
    stage_args, remaining = parser.parse_known_args(
        sys.argv[1:] if argv is None else argv
    )
    original_argv = sys.argv
    sys.argv = [original_argv[0]] + remaining
    try:
        args = get_args()
    finally:
        sys.argv = original_argv

    args.task = "rotunbot_maze_local_depth"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.camera.depth_backend = stage_args.depth_backend
    env_cfg.enable_camera_sensors_in_headless = stage_args.depth_backend == "isaacgym"
    env_cfg.commands.local_curriculum_stage = stage_args.stage
    if stage_args.stage == 1:
        env_cfg.maze.scene_mode = "corridor"
        env_cfg.maze.enabled = False
        train_cfg.runner.experiment_name = "rotunbot_maze_local_depth_stage1"
    else:
        train_cfg.runner.experiment_name = "rotunbot_maze_local_depth_stage0"
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.noise.add_noise = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root="default"
    )
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
