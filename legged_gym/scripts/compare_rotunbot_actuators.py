#!/usr/bin/env python3
"""Compare one fixed action under R torque and DIRECT_VP drive execution."""

import distutils.version
import json
import sys

import numpy as np

np.float = float

import isaacgym  # noqa: E402
import torch  # noqa: E402

from legged_gym.envs import *  # noqa: F401,F403,E402
from legged_gym.utils import get_args, task_registry  # noqa: E402


TASK = "rotunbot_target_repro"
STEPS = 20
ACTION = (0.8, 0.4)


def _make_args():
    sys.argv = [
        sys.argv[0],
        "--headless",
        "--sim_device=cpu",
        "--rl_device=cpu",
    ]
    return get_args()


def _run(control_type, seed=3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = _make_args()
    env_cfg, _ = task_registry.get_cfgs(name=TASK)
    env_cfg.env.num_envs = 1
    env_cfg.commands.target_curriculum = False
    env_cfg.control.control_type = control_type
    env_cfg.control.decimation = 1
    # Apply the same target slew limits to both paths so this test isolates
    # actuator execution rather than the policy/action limiter.
    env_cfg.control.rate_limit_1 = 0.02
    env_cfg.control.rate_limit_2 = 0.04
    env_cfg.control.direct_use_rate_limit = True
    env, _ = task_registry.make_env(name=TASK, args=args, env_cfg=env_cfg)
    try:
        env.reset()
        action = torch.tensor([ACTION], dtype=torch.float32, device=env.device)
        rows = []
        for step in range(STEPS + 1):
            rows.append({
                "step": step,
                "dof_pos": env.dof_pos[0].detach().cpu().tolist(),
                "dof_vel": env.dof_vel[0].detach().cpu().tolist(),
                "base_lin_vel": env.base_lin_vel[0].detach().cpu().tolist(),
                "root_xy": env.root_states[0, :2].detach().cpu().tolist(),
                "torques": env.torques[0].detach().cpu().tolist(),
                "output_actions": env.output_actions[0].detach().cpu().tolist(),
            })
            if step < STEPS:
                env.step(action)
        return rows
    finally:
        env.gym.destroy_sim(env.sim)


def main():
    direct = _run("DIRECT_VP")
    direct_torque = _run("DIRECT_VP_TORQUE")
    effort = _run("R")
    print(json.dumps({
        "action": ACTION,
        "direct_vp": direct,
        "direct_vp_torque": direct_torque,
        "r_torque": effort,
    }, indent=2))


if __name__ == "__main__":
    main()
