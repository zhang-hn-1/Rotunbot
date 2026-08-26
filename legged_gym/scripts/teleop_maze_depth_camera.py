"""Drive one Rotunbot through the procedural maze while viewing depth."""

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch
from isaacgym import gymapi
import matplotlib.pyplot as plt
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403 - registers all tasks
from legged_gym.teleop import RotunbotKeyboardController
from legged_gym.utils import get_args, task_registry


KEY_BINDINGS = (
    (gymapi.KEY_W, "teleop_forward"),
    (gymapi.KEY_S, "teleop_reverse"),
    (gymapi.KEY_A, "teleop_left"),
    (gymapi.KEY_D, "teleop_right"),
    (gymapi.KEY_SPACE, "teleop_stop"),
    (gymapi.KEY_R, "teleop_reset"),
)

CAMERA_POSITION = (0.42, 0.0, 0.0)
CAMERA_ROTATION = (0.0, 0.0, 0.0, 1.0)
CAMERA_WIDTH = 160
CAMERA_HEIGHT = 90
CAMERA_HORIZONTAL_FOV = 105.0
CAMERA_NEAR_PLANE = 0.05
CAMERA_FAR_PLANE = 8.0


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


def _attach_camera(env):
    camera_props = gymapi.CameraProperties()
    camera_props.width = CAMERA_WIDTH
    camera_props.height = CAMERA_HEIGHT
    camera_props.horizontal_fov = CAMERA_HORIZONTAL_FOV
    camera_props.near_plane = CAMERA_NEAR_PLANE
    camera_props.far_plane = CAMERA_FAR_PLANE

    local_transform = gymapi.Transform()
    local_transform.p = gymapi.Vec3(*CAMERA_POSITION)
    local_transform.r = gymapi.Quat(*CAMERA_ROTATION)

    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    body_handle = env.gym.find_actor_rigid_body_handle(
        env.envs[0], env.actor_handles[0], "base_link"
    )
    if body_handle < 0:
        raise RuntimeError("Could not find base_link for the maze camera")
    env.gym.attach_camera_to_body(
        camera_handle,
        env.envs[0],
        body_handle,
        local_transform,
        gymapi.FOLLOW_TRANSFORM,
    )
    return camera_handle


def _read_depth(env, camera_handle):
    """Read and normalize the current camera depth image for display."""
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    raw_depth = env.gym.get_camera_image(
        env.sim, env.envs[0], camera_handle, gymapi.IMAGE_DEPTH
    )
    raw_depth = np.asarray(raw_depth, dtype=np.float32).reshape(
        CAMERA_HEIGHT, CAMERA_WIDTH
    )
    raw_depth = np.abs(raw_depth)
    raw_depth = np.where(np.isfinite(raw_depth), raw_depth, CAMERA_FAR_PLANE)
    return (np.clip(raw_depth, CAMERA_NEAR_PLANE, CAMERA_FAR_PLANE) - CAMERA_NEAR_PLANE) / (
        CAMERA_FAR_PLANE - CAMERA_NEAR_PLANE
    )


def _draw_camera_frame(env, axis_length=0.45):
    """Draw camera +X/+Y/+Z and optical-forward axes in the maze viewer."""
    root_position = env.root_states[0, :3].detach().cpu().numpy()
    root_quaternion = env.root_states[0, 3:7].detach().cpu().numpy()
    env_origin = env.env_origins[0].detach().cpu().numpy()
    base_rotation = _quat_to_matrix(root_quaternion)
    camera_position_local = np.asarray(CAMERA_POSITION, dtype=np.float64)
    camera_rotation_local = _quat_to_matrix(CAMERA_ROTATION)
    camera_position_env = root_position - env_origin + base_rotation @ camera_position_local
    camera_rotation_env = base_rotation @ camera_rotation_local

    axis_colors = (
        np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
        np.asarray((0.0, 1.0, 0.0), dtype=np.float32),
        np.asarray((0.0, 0.3, 1.0), dtype=np.float32),
    )
    segments = []
    colors = []
    for axis_index, color in enumerate(axis_colors):
        endpoint = camera_position_env + axis_length * camera_rotation_env[:, axis_index]
        segments.extend((camera_position_env, endpoint))
        colors.append(color)

    # For the graphics-camera convention, optical forward is -Z.
    optical_endpoint = camera_position_env - 1.25 * axis_length * camera_rotation_env[:, 2]
    segments.extend((camera_position_env, optical_endpoint))
    colors.append(np.asarray((1.0, 0.0, 1.0), dtype=np.float32))

    vertices = np.ascontiguousarray(np.asarray(segments, dtype=np.float32))
    line_colors = np.ascontiguousarray(np.asarray(colors, dtype=np.float32))
    env.gym.add_lines(
        env.viewer,
        env.envs[0],
        line_colors.shape[0],
        vertices,
        line_colors,
    )


def teleop(args):
    if args.headless:
        raise ValueError(
            "teleop_maze_depth_camera.py requires a viewer; remove --headless"
        )

    args.task = "rotunbot_maze"
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 3600.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.stop_distance = 0.0
    env_cfg.rewards.scales.to_target = 0.0

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if env.viewer is None:
        raise RuntimeError("Isaac Gym viewer was not created")
    camera_handle = _attach_camera(env)

    controller = RotunbotKeyboardController(
        forward_speed=env_cfg.teleop.forward_speed,
        steering_position=env_cfg.teleop.steering_position,
    )
    for key, action_name in KEY_BINDINGS:
        env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action_name)

    def handle_viewer_event(event):
        if controller.handle_event(event.action, event.value):
            print(controller.status())

    env.add_viewer_action_handler(handle_viewer_event)
    print("W/S: forward/reverse | A/D: left/right | Space: stop | R: reset | Esc: quit")
    print("camera attached body: base_link")
    print("camera local pose: position=(0.42, 0.0, 0.0), rotation=identity")
    print("depth display: normalized distance, near=dark, far/open=bright")
    print(controller.status())

    plt.ion()
    figure, axis = plt.subplots(figsize=(12, 7))
    image = axis.imshow(
        np.ones((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.float32),
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

    all_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    step_count = 0
    try:
        env.reset()
        while not env.gym.query_viewer_has_closed(env.viewer):
            if controller.consume_reset_request():
                env.reset_idx(all_env_ids)

            command = controller.normalized_action(
                env_cfg.control.first_actionScale,
                env_cfg.control.second_actionScale,
            )
            actions[:] = torch.as_tensor(
                command, dtype=torch.float32, device=env.device
            )
            _, _, _, dones, _ = env.step(actions)

            env.gym.clear_lines(env.viewer)
            _draw_camera_frame(env)
            frame = _read_depth(env, camera_handle)
            image.set_data(frame)
            position = env.root_states[0, :3].detach().cpu().numpy()
            axis.set_title(
                f"rotunbot_maze | step={step_count} | "
                f"position=({position[0]:+.2f}, {position[1]:+.2f}) | "
                f"min={frame.min():.3f}, max={frame.max():.3f}"
            )
            figure.canvas.draw_idle()
            figure.canvas.flush_events()

            if bool(torch.any(dones)):
                print("episode reset at step", step_count)
            if not plt.fignum_exists(figure.number):
                break

            step_count += 1
            if step_count % int(env_cfg.teleop.status_interval_steps) == 0:
                speed = torch.linalg.vector_norm(env.base_lin_vel[0]).item()
                print(
                    f"position=({position[0]:+.2f}, {position[1]:+.2f}) "
                    f"speed={speed:.2f} m/s"
                )
    finally:
        plt.close(figure)
        if getattr(env, "viewer", None) is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    teleop(get_args())
