"""Phase1-4: frozen Actor probe for mirrored local goals."""

import os
from pathlib import Path

import isaacgym  # noqa: F401
import torch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


CHECKPOINT = "/home/jason/SphericalRobot_LeggedGym-master-new-map/logs/rotunbot_local_p2p/Aug24_18-18-41_/model_3050.pt"


def configure(args):
    args.task = "rotunbot_target_repro"
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 2
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.commands.random_start_yaw = False
    env_cfg.commands.target_curriculum = False
    env_cfg.commands.resample_commands = False
    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    env.data_print = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    checkpoint = Path(os.environ.get("PHASE1_ACTOR_CHECKPOINT", CHECKPOINT)).resolve()
    runner.load(str(checkpoint))
    return env, runner.get_inference_policy(device=env.device)


def main(args):
    env, policy = configure(args)
    try:
        env.reset()
        # Make all non-target state identical; keep the two goals mirrored in
        # robot/world XY at yaw=0.  Histories are cleared so both inputs differ
        # only in the current target command.
        from isaacgym import gymtorch
        env.root_states[1] = env.root_states[0]
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
        env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
        for history in env.obs_history:
            history.zero_()
        for history in env.critic_history:
            history.zero_()
        env.actions.zero_()
        env.last_actions.zero_()
        env.commands.zero_()
        env.commands[0, :2] = torch.tensor((1.0, 1.0), device=env.device)
        env.commands[1, :2] = torch.tensor((1.0, -1.0), device=env.device)
        env.compute_observations()
        with torch.no_grad():
            actions = policy(env.get_observations())
        print(f"checkpoint={os.environ.get('PHASE1_ACTOR_CHECKPOINT', CHECKPOINT)}")
        print(f"goal_left=(1,+1) action={actions[0].detach().cpu().tolist()}")
        print(f"goal_right=(1,-1) action={actions[1].detach().cpu().tolist()}")
        print(f"action_sum={float((actions[0, 0] + actions[1, 0]).item()):+.6e}")
        print(f"steering_difference={float((actions[0, 1] - actions[1, 1]).item()):+.6e}")
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main(get_args())
