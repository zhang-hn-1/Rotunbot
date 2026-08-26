"""Preview the Isaac Gym depth sensor without loading or training a policy.

Run this script without ``--headless``.  The Isaac Gym viewer shows the robot
and obstacle scene; a second matplotlib window shows a high-resolution
normalized depth image from the camera.  The policy observation remains at
the task's configured resolution and is downsampled separately.
"""

import isaacgym  # noqa: F401  (must be imported before task registration)
from isaacgym import gymapi
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def _quat_to_matrix(quaternion):
    """Return a 3x3 rotation matrix for an Isaac Gym [x, y, z, w] quat."""
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1.0e-8:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _draw_camera_coordinate_frame(env, env_cfg, axis_length=0.45):
    """Draw the camera frame in env-local coordinates in the Isaac viewer."""
    if env.viewer is None:
        return

    # base_link is the root body for Rotunbot, so root_states gives its pose.
    root_position = env.root_states[0, :3].detach().cpu().numpy()
    root_quaternion = env.root_states[0, 3:7].detach().cpu().numpy()
    env_origin = env.env_origins[0].detach().cpu().numpy()
    base_rotation = _quat_to_matrix(root_quaternion)

    camera_position_local = np.asarray(env_cfg.camera.position, dtype=np.float64)
    camera_rotation_local = _quat_to_matrix(env_cfg.camera.rotation)
    camera_position_env = root_position - env_origin + base_rotation @ camera_position_local
    camera_rotation_env = base_rotation @ camera_rotation_local

    # Isaac Gym lines use pairs of points in the selected environment's local frame.
    segments = []
    colors = []
    axis_colors = (
        np.asarray((1.0, 0.0, 0.0), dtype=np.float32),  # +X, red
        np.asarray((0.0, 1.0, 0.0), dtype=np.float32),  # +Y, green
        np.asarray((0.0, 0.3, 1.0), dtype=np.float32),  # +Z, blue
    )
    for axis_index, color in enumerate(axis_colors):
        endpoint = camera_position_env + axis_length * camera_rotation_env[:, axis_index]
        segments.extend((camera_position_env, endpoint))
        colors.append(color)

    # The optical direction is drawn separately so the camera convention is visible.
    # Isaac Gym camera depth is measured along the camera view direction; -Z is
    # shown here as the conventional graphics-camera forward axis.
    optical_endpoint = camera_position_env - 1.25 * axis_length * camera_rotation_env[:, 2]
    segments.extend((camera_position_env, optical_endpoint))
    colors.append(np.asarray((1.0, 0.0, 1.0), dtype=np.float32))  # optical axis, magenta

    vertices = np.ascontiguousarray(np.asarray(segments, dtype=np.float32))
    line_colors = np.ascontiguousarray(np.asarray(colors, dtype=np.float32))
    env.gym.add_lines(
        env.viewer,
        env.envs[0],
        line_colors.shape[0],
        vertices,
        line_colors,
    )


def _read_display_depth(env, env_cfg):
    """Read the full-resolution Isaac Gym depth image for display only."""
    height = int(env_cfg.camera.height)
    width = int(env_cfg.camera.width)
    near = float(env_cfg.camera.near_plane)
    far = float(env_cfg.camera.far_plane)

    if getattr(env, "_camera_ready", False) and env._camera_handles:
        raw_depth = env.gym.get_camera_image(
            env.sim,
            env.envs[0],
            env._camera_handles[0],
            gymapi.IMAGE_DEPTH,
        )
        raw_depth = np.asarray(raw_depth, dtype=np.float32).reshape(height, width)
        raw_depth = np.abs(raw_depth)
        raw_depth = np.where(np.isfinite(raw_depth), raw_depth, far)
        return (np.clip(raw_depth, near, far) - near) / max(far - near, 1.0e-6)

    # Fallback mode has only the policy-resolution image; enlarge it for the
    # preview window while keeping the observation path unchanged.
    low_resolution = env.depth_observation[0].detach().view(1, 1, *env.depth_observation.shape[-2:])
    enlarged = F.interpolate(low_resolution, size=(height, width), mode="nearest")
    return enlarged[0, 0].cpu().numpy()


def preview(args):
    if args.headless:
        raise ValueError(
            "preview_target_depth_camera.py needs a viewer; remove --headless"
        )

    args.task = "rotunbot_target_depth"
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    # This is a sensor preview, not the randomized training protocol.  Keep
    # the initial heading fixed so an automatic reset cannot look like a
    # spontaneous turn while the zero-action robot is standing still.
    env_cfg.commands.random_start_yaw = False
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.env.episode_length_s = 3600.0
    env_cfg.camera.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    using_real_camera = bool(getattr(env, "_camera_ready", False))
    print("camera attached body: base_link")
    print("camera local position: (0.42, 0.0, 0.0)")
    print("camera local rotation: (0.0, 0.0, 0.0, 1.0)")
    print("camera source:", "Isaac Gym IMAGE_DEPTH" if using_real_camera else "fallback")
    print("camera/policy resolution: 160x90")
    print("depth display: normalized distance, near=dark, far/open=bright")

    plt.ion()
    figure, axis = plt.subplots(figsize=(12, 7))
    frame = np.zeros(
        (int(env_cfg.camera.height), int(env_cfg.camera.width)),
        dtype=np.float32,
    )
    image = axis.imshow(
        frame,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    axis.set_xlabel("camera horizontal pixels")
    axis.set_ylabel("camera vertical pixels")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("normalized distance")
    figure.tight_layout()

    zero_actions = torch.zeros(
        env.num_envs, env.num_actions, device=env.device, requires_grad=False
    )

    try:
        env.reset()
        for step in range(1000000):
            _, _, _, dones, _ = env.step(zero_actions)
            env.gym.clear_lines(env.viewer)
            # The environment draws the markers during env.step(), but this
            # preview clears viewer lines before drawing the camera frame.
            # Redraw them here so they remain visible in every frame.
            env._draw_maze_markers()
            _draw_camera_coordinate_frame(env, env_cfg)
            frame = _read_display_depth(env, env_cfg)
            image.set_data(frame)
            axis.set_title(
                f"rotunbot_target_depth | step={step} | "
                f"min={frame.min():.3f}, max={frame.max():.3f}"
            )
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            if bool(torch.any(dones)):
                yaw = float(env.base_euler_tensor[0, 2].detach().cpu())
                print(
                    f"episode reset at step {step}; current yaw={yaw:+.3f} rad"
                )

            if not plt.fignum_exists(figure.number):
                break
    finally:
        plt.close(figure)
        if getattr(env, "viewer", None) is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    preview(get_args())
