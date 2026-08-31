"""Audit raw depth conventions and the tensor delivered to the V1 encoder."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import isaacgym  # noqa: F401 - must precede torch
import torch

from legged_gym.envs.rotunbot.maze.rotunbot_maze_camera import (
    DepthCameraMixin,
    normalize_depth_image,
)


def camera_forward_rotation():
    """Rotate Isaac's optical -Z axis onto the robot-frame +X axis."""
    return (0.0, -0.70710678, 0.0, 0.70710678)


def summarize_depth_pipeline(raw, normalized, near_plane, far_plane, metadata=None):
    """Return JSON-safe raw/encoder-input statistics without mutating tensors."""
    raw = raw.detach().float()
    normalized = normalized.detach().float()
    invalid = (~torch.isfinite(raw)) | (raw <= 0.0)
    nonfinite = ~torch.isfinite(raw)
    nonpositive = torch.isfinite(raw) & (raw <= 0.0)
    finite = bool(torch.isfinite(normalized).all().item())
    if normalized.numel():
        encoder_range = [float(normalized.min().item()), float(normalized.max().item())]
        mean = float(normalized.mean().item())
        std = float(normalized.std(unbiased=False).item())
    else:
        encoder_range = [None, None]
        mean = std = None
    result = {
        "raw_shape": list(raw.shape),
        "raw_dtype": str(raw.dtype),
        "raw_invalid_count": int(invalid.sum().item()),
        "raw_nonfinite_count": int(nonfinite.sum().item()),
        "raw_nonpositive_count": int(nonpositive.sum().item()),
        "raw_min": float(raw.min().item()) if raw.numel() else None,
        "raw_max": float(raw.max().item()) if raw.numel() else None,
        "raw_finite": bool(torch.isfinite(raw).all().item()),
        "near_plane": float(near_plane),
        "far_plane": float(far_plane),
        "encoder_input_shape": list(normalized.shape),
        "encoder_input_dtype": str(normalized.dtype),
        "encoder_input_range": encoder_range,
        "encoder_input_mean": mean,
        "encoder_input_std": std,
        "encoder_input_finite": finite,
    }
    if metadata:
        result.update(dict(metadata))
    return result


class _DirectionProbe(DepthCameraMixin):
    def __init__(self, center):
        from types import SimpleNamespace

        self.num_envs = 1
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(
            env=SimpleNamespace(depth_height=8, depth_width=32),
            camera=SimpleNamespace(
                horizontal_fov=105.0,
                near_plane=0.05,
                far_plane=8.0,
                position=(0.0, 0.0, 0.0),
            ),
        )
        self.base_euler_tensor = torch.zeros(1, 3)
        self.root_states = torch.zeros(1, 13)
        self.env_origins = torch.zeros(1, 3)
        self.center = torch.as_tensor([center], dtype=torch.float32)

    def _get_depth_fallback_aabbs(self):
        return self.center, torch.full_like(self.center, 0.05)


def fallback_audit():
    near_depth = _DirectionProbe((1.0, 0.0))._fallback_depth()
    far_depth = _DirectionProbe((3.0, 0.0))._fallback_depth()
    raw = torch.tensor([[-1.0, 0.0, 0.05, 4.0, 8.0, float("inf"), float("nan")]])
    normalized = normalize_depth_image(raw, 0.05, 8.0)
    result = summarize_depth_pipeline(
        raw,
        normalized,
        0.05,
        8.0,
        {
            "backend": "fallback",
            "horizontal_fov_deg": 105.0,
            "resolution": [8, 32],
            "near_obstacle_min": float(near_depth.min().item()),
            "far_obstacle_min": float(far_depth.min().item()),
            "near_is_smaller_than_far": bool(near_depth.min() < far_depth.min()),
            "camera_forward_direction": "+x in robot frame",
            "normalization": "clamp/fill invalid to far, then (depth-near)/(far-near)",
        },
    )
    return result


def isaacgym_audit(framework_args, forward_camera=False):
    from legged_gym.envs import task_registry
    from legged_gym.utils import get_args

    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + list(framework_args)
    try:
        args = get_args()
    finally:
        os.sys.argv = original
    args.task = "rotunbot_sru_visual_corridor_v1"
    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.camera.depth_backend = "isaacgym"
    env_cfg.enable_camera_sensors_in_headless = True
    if forward_camera:
        env_cfg.camera.rotation = camera_forward_rotation()
    env_cfg.camera.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    try:
        env.reset()
        normalized = env.capture_depth()
        raw = torch.stack(env._camera_depth_tensors, dim=0).to(env.device)
        return summarize_depth_pipeline(
            raw,
            normalized,
            env_cfg.camera.near_plane,
            env_cfg.camera.far_plane,
            {
                "backend": "isaacgym",
                "backend_requested": env.depth_backend_requested,
                "backend_actual": env.depth_backend_actual,
                "horizontal_fov_deg": float(env_cfg.camera.horizontal_fov),
                "resolution": [int(env_cfg.camera.height), int(env_cfg.camera.width)],
                "camera_position_m": list(env_cfg.camera.position),
                "camera_rotation_quat": list(env_cfg.camera.rotation),
                "camera_forward_direction": "+x in robot frame (configured pose)",
                "forward_camera_rotation_applied": bool(forward_camera),
                "normalization": "clamp/fill invalid to far, then (depth-near)/(far-near)",
            },
        )
    finally:
        if hasattr(env, "close"):
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("fallback", "isaacgym"), default="fallback")
    parser.add_argument("--forward-camera", action="store_true")
    parser.add_argument("--output", default="logs/depth_backend_calibration/v1_pipeline_audit.json")
    stage_args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    report = (
        fallback_audit()
        if stage_args.backend == "fallback"
        else isaacgym_audit(remaining, forward_camera=stage_args.forward_camera)
    )
    output = Path(stage_args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
