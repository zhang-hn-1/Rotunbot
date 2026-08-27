"""Sensor-only Gate1.5 diagnostic for side-obstacle visibility."""

import argparse
import json
import math
from pathlib import Path

import isaacgym  # noqa: F401 - must precede torch in this repository
import torch
from types import SimpleNamespace

from legged_gym.envs.rotunbot.maze.rotunbot_maze_camera import DepthCameraMixin


class _Probe(DepthCameraMixin):
    def __init__(self, center, horizontal_fov):
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(
            env=SimpleNamespace(depth_height=8, depth_width=32),
            camera=SimpleNamespace(
                horizontal_fov=horizontal_fov,
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
        return self.center, torch.full_like(self.center, 0.15)


def measure(bearings=(45, 60, 75, 90, -60, -90), radius=1.5, fov=105.0):
    result = {}
    for bearing in bearings:
        angle = math.radians(bearing)
        center = (radius * math.cos(angle), radius * math.sin(angle))
        depth = _Probe(center, fov)._fallback_depth()
        result[str(bearing)] = {
            "visible_fraction": float((depth < 0.95).float().mean().item()),
            "min_normalized_depth": float(depth.min().item()),
        }
    return {
        "horizontal_fov_deg": fov,
        "radius_m": radius,
        "obstacle_half_extent_m": 0.15,
        "bearings_deg": list(bearings),
        "measurements": result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="logs/depth_gate1_5_observability.json")
    args = parser.parse_args()
    report = measure()
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
