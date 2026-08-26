"""Fine-tune the existing P2P actor on empty-map local waypoints."""

import os
from pathlib import Path

import isaacgym  # noqa: F401

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


DEFAULT_CHECKPOINT = (
    "/home/jason/SphericalRobot_LeggedGym-master-new-map/logs/"
    "rotunbot_target_repro/Aug11_16-44-07_/model_2050.pt"
)


def train(args):
    checkpoint = Path(os.environ.get("LOCAL_P2P_INIT_CHECKPOINT", DEFAULT_CHECKPOINT)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Local P2P initialization checkpoint does not exist: {checkpoint}")

    args.task = "rotunbot_local_p2p"
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = int(os.environ.get("LOCAL_P2P_NUM_ENVS", "64"))
    env_cfg.commands.local_curriculum_stage = int(
        os.environ.get("LOCAL_P2P_CURRICULUM_STAGE", "1")
    )
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    train_cfg.runner.max_iterations = int(
        os.environ.get("LOCAL_P2P_MAX_ITERATIONS", str(train_cfg.runner.max_iterations))
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
    print(f"Loading frozen P2P initialization: {checkpoint}", flush=True)
    runner.load(str(checkpoint))
    # The source checkpoint carries a large exploration std and an old
    # adaptive optimizer state.  Fine-tuning a controller needs a gentler,
    # explicit continuation setting; this does not alter the actor network.
    action_std = float(os.environ.get("LOCAL_P2P_ACTION_STD", "0.30"))
    learning_rate = float(os.environ.get("LOCAL_P2P_LEARNING_RATE", "1.0e-4"))
    if hasattr(runner.alg.actor_critic, "std"):
        runner.alg.actor_critic.std.data.fill_(action_std)
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = learning_rate
    print(
        f"Fine-tune optimizer settings: action_std={action_std}, "
        f"learning_rate={learning_rate}",
        flush=True,
    )
    print(
        "Starting local P2P fine-tuning: "
        f"iterations={train_cfg.runner.max_iterations}, "
        f"envs={env.num_envs}, stage={env_cfg.commands.local_curriculum_stage}, "
        f"device={env.device}",
        flush=True,
    )
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    train(get_args())
