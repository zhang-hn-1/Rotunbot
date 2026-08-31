"""Run the V1 real-camera one-wall physical depth sanity test."""

import argparse
import json
import os
import sys
from pathlib import Path

import isaacgym  # noqa: F401 - must precede torch
import numpy as np
import torch

from legged_gym.envs.rotunbot.maze.rotunbot_maze_camera import (
    capture_isaac_depth_tensors,
)


def validate_physical_sanity(measurements):
    """Validate monotonic wall distance and finite encoder input statistics."""
    rows = sorted(measurements, key=lambda row: float(row["wall_distance_m"]))
    expected = [0.5, 2.0, 5.0]
    failures = []
    if [float(row["wall_distance_m"]) for row in rows] != expected:
        failures.append("wall_distance_set_incomplete")
    center = [float(row["center_distance_m"]) for row in rows]
    if not (len(center) == 3 and center[0] < center[1] < center[2]):
        failures.append("center_depth_not_monotonic")
    if any(float(row["finite_ratio"]) < 1.0 for row in rows):
        failures.append("encoder_input_not_finite")
    return {
        "pass": not failures,
        "failures": failures,
        "ordered_wall_distances_m": [float(row["wall_distance_m"]) for row in rows],
        "center_distances_m": center,
        "finite_ratios": [float(row["finite_ratio"]) for row in rows],
    }


def _parse_framework_args(framework_args):
    from legged_gym.utils import get_args

    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + list(framework_args)
    try:
        return get_args()
    finally:
        os.sys.argv = original


def _cpu_depth(env, handle):
    from isaacgym import gymapi

    raw = env.gym.get_camera_image(
        env.sim, env.envs[0], handle, gymapi.IMAGE_DEPTH
    )
    return np.asarray(raw, dtype=np.float32).reshape(
        int(env.cfg.camera.height), int(env.cfg.camera.width)
    )


def _measurement(env, distance):
    raw, normalized = capture_isaac_depth_tensors(
        env.gym,
        env.sim,
        env._camera_depth_tensors,
        env.device,
        env.cfg.camera.near_plane,
        env.cfg.camera.far_plane,
    )
    cpu_raw = _cpu_depth(env, env._camera_handles[0])
    raw_frame = raw[0]
    height, width = raw_frame.shape
    center_pixel = raw_frame[height // 2, width // 2].detach().float()
    center_distance = (
        float(center_pixel.abs().item())
        if torch.isfinite(center_pixel) and center_pixel != 0.0
        else float("inf")
    )
    finite_ratio = float(torch.isfinite(normalized).float().mean().item())
    finite_pixels = torch.isfinite(raw_frame)
    finite_indices = finite_pixels.nonzero(as_tuple=False)
    camera_transform = env.gym.get_camera_transform(
        env.sim, env.envs[0], env._camera_handles[0]
    )
    return {
        "wall_distance_m": float(distance),
        "wall_actor_positions_m": env._all_root_states[1:, :3].detach().cpu().tolist(),
        "raw_shape": list(raw.shape),
        "raw_finite_ratio": float(torch.isfinite(raw).float().mean().item()),
        "raw_min": float(raw.min().item()),
        "raw_max": float(raw.max().item()),
        "raw_mean": float(raw.float().mean().item()),
        "raw_median": float(raw.float().median().item()),
        "raw_center_pixel": float(raw_frame[height // 2, width // 2].item()),
        "raw_finite_bbox": (
            [int(finite_indices[:, 0].min()), int(finite_indices[:, 1].min()),
             int(finite_indices[:, 0].max()), int(finite_indices[:, 1].max())]
            if finite_indices.numel() else None
        ),
        "raw_finite_values": raw_frame[finite_pixels].detach().cpu().tolist(),
        "cpu_raw_finite_ratio": float(np.isfinite(cpu_raw).mean()),
        "cpu_raw_center_pixel": float(cpu_raw[height // 2, width // 2]),
        "center_distance_m": center_distance,
        "finite_ratio": finite_ratio,
        "normalized_min": float(normalized.min().item()),
        "normalized_max": float(normalized.max().item()),
        "normalized_mean": float(normalized.mean().item()),
        "normalized_median": float(normalized.median().item()),
        "camera_transform_world": {
            "position": [
                float(camera_transform.p.x),
                float(camera_transform.p.y),
                float(camera_transform.p.z),
            ],
            "rotation": [
                float(camera_transform.r.x),
                float(camera_transform.r.y),
                float(camera_transform.r.z),
                float(camera_transform.r.w),
            ],
        },
    }


def _make_sanity_env(args, distance, camera_rotation):
    from legged_gym.envs import task_registry

    args.task = "rotunbot_sru_visual_corridor_v1"
    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.camera.depth_backend = "isaacgym"
    # Use the production V1 mount pose.  The physical probe must not pass by
    # changing the sensor geometry relative to the policy/evaluation path.
    env_cfg.camera.position = (0.42, 0.0, 0.0)
    env_cfg.camera.rotation = tuple(float(value) for value in camera_rotation)
    env_cfg.camera.add_noise = False
    env_cfg.enable_camera_sensors_in_headless = True
    env_cfg.corridor_wall_segments = (((float(distance), -0.5), (float(distance), 0.5)),)
    env_cfg.corridor_wall_width_m = 0.0
    env_cfg.direct_obstacle_aabbs = ()
    return task_registry.make_env(args.task, args=args, env_cfg=env_cfg)[0], env_cfg


def run_physical_sanity(
    framework_args=(), output=None, wall_distance=None,
    camera_rotation=(0.0, 0.0, 0.0, 1.0),
):
    import legged_gym.envs  # noqa: F401 - registration side effects

    args = _parse_framework_args(framework_args)
    measurements = []
    camera_cfg = None
    distances = (0.5, 2.0, 5.0) if wall_distance is None else (float(wall_distance),)
    for distance in distances:
        env, env_cfg = _make_sanity_env(args, distance, camera_rotation)
        camera_cfg = env_cfg.camera
        try:
            env.reset()
            measurements.append(_measurement(env, distance))
        finally:
            if getattr(env, "viewer", None) is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)
    result = {
        "backend": "isaacgym",
        "camera_position_m": list(camera_cfg.position),
        "camera_rotation_quat": list(camera_cfg.rotation),
        "near_plane": float(camera_cfg.near_plane),
        "far_plane": float(camera_cfg.far_plane),
        "measurements": measurements,
    }
    result["sanity"] = (
        validate_physical_sanity(measurements)
        if wall_distance is None
        else {"pass": None, "failures": ["single_distance_measurement"]}
    )
    if output:
        path = Path(output).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="logs/depth_backend_calibration/v1_physical_sanity.json")
    parser.add_argument("--wall-distance", type=float, default=None)
    parser.add_argument("--camera-rotation", nargs=4, type=float, default=(0.0, 0.0, 0.0, 1.0))
    stage_args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return run_physical_sanity(
        remaining, stage_args.output, stage_args.wall_distance, stage_args.camera_rotation
    )


if __name__ == "__main__":
    main()
