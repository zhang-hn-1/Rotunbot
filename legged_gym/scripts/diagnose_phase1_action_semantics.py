"""Phase1-2: measure physical motion caused by fixed action channels."""

import math

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


SCENARIOS = (
    ("action0_positive", 0.5, 0.0),
    ("action0_negative", -0.5, 0.0),
    ("action1_positive", 0.0, 0.5),
    ("action1_negative", 0.0, -0.5),
)
STEPS = 100


def yaw_from_quaternion(quat):
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def configure(args):
    args.task = "rotunbot_target_repro"
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
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
    return env


def run_scenario(env, name, action0, action1):
    env.reset()
    # Keep the target far away so this probe cannot terminate on success.
    env.commands[0, :2] = torch.tensor((5.0, 5.0), device=env.device)
    env.goal_dist[0] = torch.linalg.norm(env.commands[0, :2] - env.root_states[0, :2])
    env.last_goal_dist[0] = env.goal_dist[0]
    start_xy = env.root_states[0, :2].clone()
    start_yaw = float(yaw_from_quaternion(env.base_quat[0]).item())
    action = torch.tensor([[action0, action1]], device=env.device)
    with torch.no_grad():
        for _ in range(STEPS):
            _, _, _, dones, _ = env.step(action)
            if bool(dones[0].item()):
                raise RuntimeError(f"unexpected termination during {name}")
    end_xy = env.root_states[0, :2].clone()
    end_yaw = float(yaw_from_quaternion(env.base_quat[0]).item())
    delta_xy = end_xy - start_xy
    delta_yaw = math.atan2(math.sin(end_yaw - start_yaw), math.cos(end_yaw - start_yaw))
    c, s = math.cos(start_yaw), math.sin(start_yaw)
    body_delta = (c * float(delta_xy[0]) + s * float(delta_xy[1]),
                  -s * float(delta_xy[0]) + c * float(delta_xy[1]))
    print(
        f"{name}: action=({action0:+.2f},{action1:+.2f}) "
        f"start_yaw={math.degrees(start_yaw):+.3f}deg "
        f"delta_world=({float(delta_xy[0]):+.6f},{float(delta_xy[1]):+.6f}) "
        f"delta_body=({body_delta[0]:+.6f},{body_delta[1]:+.6f}) "
        f"delta_yaw={math.degrees(delta_yaw):+.6f}deg"
    )


def main(args):
    env = configure(args)
    try:
        for scenario in SCENARIOS:
            run_scenario(env, *scenario)
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main(get_args())
