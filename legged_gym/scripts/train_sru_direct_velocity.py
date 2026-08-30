"""Train the direct SRU velocity policy through the S1/S2/S2B curriculum."""

import argparse
import json
import os
import sys

import isaacgym  # noqa: F401 - must precede torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.navigation.direct_velocity_curriculum import (
    configure_direct_velocity_stage,
)
from legged_gym.navigation.corridor_artifacts import CheckpointMetadata
from legged_gym.utils import get_args, task_registry


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=("S1", "S2", "S2B"), default="S1")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--resume_path", default=None)
    parser.add_argument("--parent_checkpoint", default=None)
    stage_args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + remaining
    try:
        args = get_args()
    finally:
        os.sys.argv = original

    args.task = "rotunbot_sru_direct_velocity"
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    configure_direct_velocity_stage(env_cfg, stage_args.stage)
    if stage_args.num_envs is not None:
        env_cfg.env.num_envs = int(stage_args.num_envs)
    train_cfg.runner.experiment_name = "rotunbot_sru_direct_velocity_%s" % stage_args.stage.lower()
    if stage_args.iterations is not None:
        train_cfg.runner.max_iterations = int(stage_args.iterations)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root="default",
    )
    if stage_args.resume_path:
        runner.load(stage_args.resume_path)
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )
    if runner.log_dir is not None and stage_args.parent_checkpoint:
        checkpoint = os.path.join(
            runner.log_dir, "model_{}.pt".format(runner.current_learning_iteration)
        )
        metadata = CheckpointMetadata.from_path(
            checkpoint,
            parent=stage_args.parent_checkpoint,
            stage=stage_args.stage,
            seed=train_cfg.seed,
            iterations=runner.current_learning_iteration,
        )
        with open(os.path.join(runner.log_dir, "checkpoint_metadata.json"), "w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        print("Checkpoint metadata: {}".format(os.path.join(runner.log_dir, "checkpoint_metadata.json")))


if __name__ == "__main__":
    main()
