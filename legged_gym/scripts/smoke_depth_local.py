"""Two-environment V0 smoke for explicit fallback or real camera backends."""

import sys

def _extract_backend(argv):
    backend = "fallback"
    remaining = []
    skip = False
    for index, value in enumerate(argv):
        if skip:
            skip = False
            continue
        if value == "--depth-backend":
            if index + 1 >= len(argv):
                raise ValueError("--depth-backend requires fallback or isaacgym")
            backend = argv[index + 1]
            skip = True
        else:
            remaining.append(value)
    if backend not in ("fallback", "isaacgym"):
        raise ValueError("--depth-backend must be fallback or isaacgym")
    return backend, remaining


def smoke(argv=None):
    import isaacgym  # noqa: F401 - must precede torch in Isaac Gym Preview 4
    import numpy as np

    if not hasattr(np, "float"):
        np.float = float
    import torch

    from legged_gym.dwl.actor_critic_depth_local import ActorCriticDepthLocal
    import legged_gym.envs  # noqa: F401 - registration side effects
    from legged_gym.utils import get_args, task_registry

    backend, remaining = _extract_backend(list(sys.argv[1:] if argv is None else argv))
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    args.task = "rotunbot_maze_local_depth"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 2
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.camera.depth_backend = backend
    if backend == "isaacgym":
        # Offscreen graphics remain enabled by the task config even when the
        # viewer is disabled with --headless.
        env_cfg.enable_camera_sensors_in_headless = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    try:
        obs, privileged = env.reset()
        if tuple(obs.shape) != (2, 272) or tuple(privileged.shape) != (2, 18):
            raise RuntimeError(f"unexpected observation shapes: {obs.shape}, {privileged.shape}")
        policy = ActorCriticDepthLocal(272, 272, 18, 2).to(env.device)
        actions = policy.act_inference(obs)
        obs, privileged, rewards, dones, _ = env.step(actions)
        if tuple(actions.shape) != (2, 2):
            raise RuntimeError("unexpected action shape")
        if not torch.isfinite(rewards).all() or not torch.isfinite(obs).all():
            raise RuntimeError("non-finite V0 smoke output")
        if float(env.depth_observation.min()) < 0.0 or float(env.depth_observation.max()) > 1.0:
            raise RuntimeError("depth observation escaped [0, 1]")
        expected = "isaacgym" if backend == "isaacgym" else "fallback"
        if env.depth_backend_requested != backend or env.depth_backend_actual != expected:
            raise RuntimeError(
                f"backend audit failed: requested={env.depth_backend_requested}, "
                f"actual={env.depth_backend_actual}"
            )
        print(f"Depth local smoke passed: backend={backend}, obs={tuple(obs.shape)}, critic={tuple(privileged.shape)}")
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    smoke()
