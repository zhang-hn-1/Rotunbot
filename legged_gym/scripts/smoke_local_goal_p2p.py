"""GPU smoke test for the explicit Robot-frame Local P2P task."""

import math

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def main(args):
    args.task = "rotunbot_local_goal"
    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 2
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.commands.local_curriculum_stage = "A"

    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    env.data_print = False
    try:
        obs = env.reset()
        local_goal = torch.tensor([[1.0, 0.5], [1.0, 0.5]], device=env.device)
        qx, qy, qz, qw = env.base_quat.unbind(dim=1)
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy.square() + qz.square()),
        )
        c, s = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)
        world_delta = torch.cat(
            (c * local_goal[:, 0:1] - s * local_goal[:, 1:2],
             s * local_goal[:, 0:1] + c * local_goal[:, 1:2]),
            dim=1,
        )
        env.world_goal[:] = env.root_states[:, :2] + world_delta
        env.commands[:, :2] = env.world_goal
        env.compute_observations()
        obs = env.get_observations()

        assert tuple(obs.shape) == (2, 17), tuple(obs.shape)
        assert torch.isfinite(obs).all().item()
        torch.testing.assert_close(obs[0, :2], obs[1, :2], atol=1e-5, rtol=0.0)
        torch.testing.assert_close(obs[:, :2], local_goal / 3.0, atol=1e-5, rtol=0.0)

        zero_actions = torch.zeros((2, 2), device=env.device)
        obs, _, rewards, dones, _ = env.step(zero_actions)
        assert tuple(obs.shape) == (2, 17), tuple(obs.shape)
        assert torch.isfinite(obs).all().item()
        assert torch.isfinite(rewards).all().item()
        assert not dones.any().item(), dones
        assert env.action_count.item() == 8
        print(
            f"LOCAL_GOAL_SMOKE PASS obs_shape={tuple(obs.shape)} "
            f"local_goal={env.local_goal.detach().cpu().tolist()} "
            f"reward={rewards.detach().cpu().tolist()} "
            f"clip_ratio={env.clip_count.item() / max(env.action_count.item(), 1):.6f}"
        )
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main(get_args())
