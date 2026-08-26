"""GPU smoke test for the Phase-1 Oracle hierarchical navigation loop.

The planner uses maze ground truth and latches a local target at 2 Hz. The
default low-level adapter is deterministic and validates the planner/task
contract until a blind P2P checkpoint is available. Replace
``AnalyticP2PAdapter`` with the frozen P2P policy without changing the planner.
"""

import os

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403 - registers all tasks
from legged_gym.planners import OracleLocalSubgoalPlanner
from legged_gym.utils import get_args, task_registry


PLANNER_HZ = 2.0
DEFAULT_EPISODES = 1
DEFAULT_MAX_STEPS = 3000


def _yaw_from_quaternion(quat):
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class AnalyticP2PAdapter:
    """Deterministic stand-in for a frozen blind P2P policy in smoke tests."""

    def __init__(self, env):
        self.env = env

    def __call__(self, local_goal_xy):
        angle = torch.atan2(local_goal_xy[:, 1], local_goal_xy[:, 0])
        distance = torch.linalg.vector_norm(local_goal_xy, dim=1)
        # A Rotunbot changes heading through steering while rolling. Do not
        # zero the forward command for a waypoint that is currently lateral.
        forward = torch.clamp(0.45 + 0.30 * distance, 0.0, 1.0)
        steering = torch.clamp(1.8 * angle, -1.0, 1.0)
        stop = distance < 0.35
        forward = torch.where(stop, torch.zeros_like(forward), forward)
        return torch.stack((forward, steering), dim=1)


def _world_to_robot(world_goal, robot_position, yaw):
    delta = world_goal - robot_position
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    return torch.stack(
        (
            cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1],
            -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1],
        ),
        dim=1,
    )


def smoke_test(args):
    args.task = "rotunbot_maze"
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.maze.terminate_on_collision = True
    env_cfg.commands.random_start_yaw = False
    env_cfg.init_state.randomize_initial_velocity = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.data_print = False
    planner = OracleLocalSubgoalPlanner(
        env.maze_layout,
        cell_size=env.cfg.maze.cell_size,
        lookahead_cells=int(os.environ.get("ORACLE_LOOKAHEAD_CELLS", "1")),
    )
    adapter = AnalyticP2PAdapter(env)
    obs, _ = env.reset()

    planner_steps = max(1, int(round(1.0 / (PLANNER_HZ * env.dt))))
    episodes = int(os.environ.get("ORACLE_SMOKE_EPISODES", DEFAULT_EPISODES))
    max_steps = int(os.environ.get("ORACLE_SMOKE_MAX_STEPS", DEFAULT_MAX_STEPS))
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("smoke episode and step counts must be positive")

    successes = collisions = timeouts = 0
    completed = 0
    local_goal = None
    for step in range(episodes * max_steps):
        if step % planner_steps == 0 or local_goal is None:
            position = (env.root_states[:, :2] - env.env_origins[:, :2]).detach().cpu().numpy()
            goal = (env.maze_global_goals - env.env_origins[:, :2]).detach().cpu().numpy()
            waypoints = [planner.plan(position[i], goal[i])[0] for i in range(env.num_envs)]
            local_goal = torch.as_tensor(
                np.asarray(waypoints), dtype=torch.float32, device=env.device
            )
            env.set_maze_local_goals(local_goal)

        robot_position = env.root_states[:, :2] - env.env_origins[:, :2]
        yaw = _yaw_from_quaternion(env.base_quat)
        local_goal_robot = _world_to_robot(local_goal, robot_position, yaw)
        actions = adapter(local_goal_robot)
        obs, _, _, dones, _ = env.step(actions)

        if bool(torch.any(dones)):
            completed += 1
            success = bool(env.success_buf[0].item())
            collision = bool(env.maze_collision_buf[0].item())
            timeout = bool(env.time_out_buf[0].item())
            successes += int(success)
            collisions += int(collision)
            timeouts += int(timeout)
            print(
                f"episode={completed} success={int(success)} "
                f"collision={int(collision)} timeout={int(timeout)}"
            )
            if completed >= episodes:
                break
            local_goal = None

    if completed == 0:
        raise RuntimeError("Oracle planner smoke test did not complete an episode")
    if not torch.isfinite(env.root_states).all():
        raise RuntimeError("non-finite Rotunbot state detected during Oracle smoke test")
    print(
        "Oracle planner smoke test passed: "
        f"device={env.device}, planner_hz={PLANNER_HZ:.1f}, "
        f"planner_steps={planner_steps}, episodes={completed}, "
        f"SR={successes / completed:.2%}, collision={collisions / completed:.2%}, "
        f"timeout={timeouts / completed:.2%}"
    )


if __name__ == "__main__":
    smoke_test(get_args())
