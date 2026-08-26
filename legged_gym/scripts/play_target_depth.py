"""Run a trained rotunbot_target_depth policy."""

import isaacgym  # noqa: F401  (must be imported before task registration)
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def play(args):
    args.task = "rotunbot_target_depth"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.camera.add_noise = False
    # The current depth policy is trained from Isaac Gym GPU camera tensors.
    # Keep viewer and headless play paths on the same sensor distribution.
    env_cfg.camera.enable = True
    env_cfg.camera.policy_source = "camera"
    env_cfg.enable_camera_sensors_in_headless = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    train_cfg.runner.resume = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    policy = runner.get_inference_policy(device=env.device)
    # make_env constructs the simulator but does not perform the task reset.
    # Explicitly initialize the fixed start/goal and all stacked observations.
    obs, _ = env.reset()
    print("Depth source: Isaac Gym GPU camera (same as training)")

    target_episodes = 10
    completed_episodes = 0
    success_count = 0
    collision_count = 0
    timeout_count = 0
    unstable_count = 0
    out_of_bounds_count = 0

    for step in range(target_episodes * (int(env.max_episode_length) + 1)):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, rewards, dones, infos = env.step(actions)
        if bool(torch.any(dones)):
            success = bool(env.success_buf[0].item())
            collision = bool(env.step_collision_buf[0].item())
            timeout = bool(env.terminal_timeout[0].item())
            unstable = bool(env.terminal_unstable[0].item())
            out_of_bounds = bool(env.terminal_out_of_bounds[0].item())
            completed_episodes += 1
            success_count += int(success)
            collision_count += int(collision)
            timeout_count += int(timeout)
            unstable_count += int(unstable)
            out_of_bounds_count += int(out_of_bounds)
            print(
                "step=",
                step,
                "success=",
                int(success),
                "collision=",
                int(collision),
            )
            if completed_episodes >= target_episodes:
                break

    if completed_episodes == 0:
        raise RuntimeError("No completed episode during depth-policy evaluation.")
    print(
        "Depth evaluation: "
        f"SR={success_count}/{completed_episodes}="
        f"{success_count / completed_episodes:.2%}, "
        f"collision={collision_count / completed_episodes:.2%}, "
        f"timeout={timeout_count / completed_episodes:.2%}, "
        f"unstable={unstable_count / completed_episodes:.2%}, "
        f"out_of_bounds={out_of_bounds_count / completed_episodes:.2%}"
    )


if __name__ == "__main__":
    play(get_args())
