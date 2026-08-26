"""Train the explicit Robot-frame Local P2P controller from scratch."""

import os
from pathlib import Path

import isaacgym  # noqa: F401

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def train(args):
    args.task = "rotunbot_local_goal"
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = int(os.environ.get("LOCAL_GOAL_NUM_ENVS", env_cfg.env.num_envs))
    env_cfg.commands.local_curriculum_stage = os.environ.get("LOCAL_GOAL_STAGE", "A").upper()
    env_cfg.noise.add_noise = os.environ.get("LOCAL_GOAL_ADD_NOISE", "0") == "1"
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    train_cfg.runner.max_iterations = int(
        os.environ.get("LOCAL_GOAL_MAX_ITERATIONS", train_cfg.runner.max_iterations)
    )
    train_cfg.runner.resume = False

    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root="default",
    )

    checkpoint_value = os.environ.get("LOCAL_GOAL_CHECKPOINT")
    if checkpoint_value:
        checkpoint = Path(checkpoint_value).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Local Goal checkpoint does not exist: {checkpoint}")
        print(f"Loading compatible Local Goal checkpoint: {checkpoint}", flush=True)
        runner.load(str(checkpoint))
    else:
        print("Starting Local Goal PPO from scratch; no old P2P checkpoint is loaded.", flush=True)

    print(
        "Local Goal training: "
        f"stage={env_cfg.commands.local_curriculum_stage} "
        f"iterations={train_cfg.runner.max_iterations} "
        f"envs={env.num_envs} device={env.device} obs={env.num_obs}",
        flush=True,
    )
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    train(get_args())
