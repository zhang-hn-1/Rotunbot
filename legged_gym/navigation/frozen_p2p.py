"""Runtime-only loader and state helpers for the frozen uniform-4150 skill."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .baseline import (
    ACTION_DIM,
    CHECKPOINT_RELATIVE_PATH,
    CONTROL_TYPE,
    FRAME_STACK,
    OBSERVATION_DIM,
    P2P_TASK_NAME,
    POSITION_GAIN,
    SUCCESS_DISTANCE_M,
    SUCCESS_SPEED_MPS,
    VELOCITY_GAIN,
    require_checkpoint,
)


@dataclass(frozen=True)
class FrozenP2PConfig:
    checkpoint: Path
    task_name: str = P2P_TASK_NAME
    control_type: str = CONTROL_TYPE
    device: str = "cuda:0"

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cuda:0"):
        return cls(checkpoint=require_checkpoint(checkpoint), device=str(device))


def set_temporary_world_goal(env, world_goal_xy, env_index=0):
    """Replace only the world-frame target stored in ``env.commands``."""
    goal = np.asarray(world_goal_xy, dtype=np.float64)
    if goal.shape != (2,) or not np.all(np.isfinite(goal)):
        raise ValueError("world_goal_xy must contain two finite values")
    commands = env.commands
    if hasattr(commands, "device"):
        import torch

        commands[env_index, :2] = torch.as_tensor(
            goal, dtype=commands.dtype, device=commands.device
        )
    else:
        commands[env_index, :2] = goal


def refresh_observation_after_goal_change(env):
    """Refresh goal channels in-place without appending a history frame.

    The frozen actor consumes 20 historical 19-D frames. A temporary goal
    switch changes the semantic target for the whole actor input, while the
    measured robot-state/action portions of those frames remain intact.
    """
    history = getattr(env, "obs_history", None)
    if history is None or len(history) == 0:
        raise AttributeError("environment must expose a non-empty obs_history")

    commands = env.commands[:, :2]
    command_scale = getattr(getattr(env, "obs_scales", None), "command", 1.0)
    target = commands * command_scale
    first = history[0]
    if hasattr(first, "device"):
        import torch

        for frame in history:
            frame[:, :2] = target
        env.obs_buf = torch.stack(list(history), dim=1).reshape(env.num_envs, -1)
        privileged_history = getattr(env, "critic_history", None)
        if privileged_history is not None and len(privileged_history) > 0:
            for frame in privileged_history:
                frame[:, :2] = target
            env.privileged_obs_buf = torch.cat(list(privileged_history), dim=1)
    else:
        target = np.asarray(target, dtype=np.float64)
        for frame in history:
            frame[:, :2] = target
        env.obs_buf = np.stack(list(history), axis=1).reshape(len(target), -1)
        privileged_history = getattr(env, "critic_history", None)
        if privileged_history is not None and len(privileged_history) > 0:
            for frame in privileged_history:
                frame[:, :2] = target
            env.privileged_obs_buf = np.concatenate(list(privileged_history), axis=1)
    return env.get_observations()


def enforce_frozen_control_config(env_cfg):
    """Disable executor gain randomization and assert the trained gains."""
    env_cfg.control.direct_velocity_gain_randomize = False
    assert not env_cfg.control.direct_velocity_gain_randomize
    assert float(env_cfg.control.direct_velocity_gain) == VELOCITY_GAIN
    assert float(env_cfg.control.direct_position_gain) == POSITION_GAIN
    return env_cfg


def robot_pose(env, env_index=0):
    """Return the current measured world XY position and yaw."""
    position = env.root_states[env_index, :2].detach().cpu().numpy().astype(np.float64)
    quaternion = env.root_states[env_index, 3:7].detach().cpu().numpy()
    x, y, z, w = [float(value) for value in quaternion]
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return position, yaw


def robot_speed(env, env_index=0):
    if hasattr(env, "base_lin_vel"):
        velocity = env.base_lin_vel[env_index]
    else:
        velocity = env.root_states[env_index, 7:10]
    return float(velocity.detach().norm().cpu().item())


def action_was_clipped(env, action, env_index=0):
    # ``output_actions`` is in physical target units (m/s and radians), while
    # the policy action is normalized.  Compare the policy input with the
    # environment's clipped normalized tensor instead of mixing units.
    if not hasattr(env, "actions"):
        return False
    clipped = env.actions[env_index].detach().cpu().numpy()
    raw = action[env_index].detach().cpu().numpy()
    return bool(np.any(np.abs(clipped[:ACTION_DIM] - raw[:ACTION_DIM]) > 1.0e-6))


def load_frozen_p2p(args, checkpoint):
    """Create the original P2P environment and load a verified checkpoint."""
    checkpoint = require_checkpoint(checkpoint)
    from legged_gym.envs import task_registry
    from legged_gym.navigation.hierarchical_p2p_env import HierarchicalP2P
    from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed

    env_cfg, train_cfg = task_registry.get_cfgs(name=P2P_TASK_NAME)
    env_cfg.env.num_envs = 1
    env_cfg.commands.target_curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    enforce_frozen_control_config(env_cfg)
    if hasattr(env_cfg, "latency"):
        env_cfg.latency.enabled = False
    # Avoid the config's training resume path; the explicit path below is the
    # only allowed source of model weights for all evaluation gates.
    train_cfg.runner.resume = False
    args.num_envs = 1
    set_seed(env_cfg.seed)
    sim_params = parse_sim_params(args, {"sim": class_to_dict(env_cfg.sim)})
    env = HierarchicalP2P(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless,
    )
    runner = load_frozen_runner(args, env, train_cfg, checkpoint)
    policy = runner.get_inference_policy(device=args.rl_device)
    return env, runner, policy


def load_frozen_runner(args, env, train_cfg, checkpoint):
    """Attach the frozen policy to an already-created compatible environment."""
    checkpoint = require_checkpoint(checkpoint)
    import torch

    from legged_gym.envs import task_registry

    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    state = torch.load(str(checkpoint), map_location=args.rl_device)
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint}")
    runner.load(str(checkpoint), load_optimizer=False)
    print(
        "Loaded frozen uniform-4150 checkpoint: "
        f"{checkpoint} (task={P2P_TASK_NAME}, obs={OBSERVATION_DIM}x{FRAME_STACK}, "
        f"actions={ACTION_DIM}, control={CONTROL_TYPE}, "
        f"gains=({VELOCITY_GAIN},{POSITION_GAIN}))",
        flush=True,
    )
    return runner
