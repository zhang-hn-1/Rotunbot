"""Provider-neutral closed-loop Oracle dataset serialization."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def _pair(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain two finite values")
    return (float(result[0]), float(result[1]))


@dataclass(frozen=True)
class OracleSample:
    depth: np.ndarray
    robot_xy: tuple
    robot_yaw: float
    global_goal_xy: tuple
    local_goal_xy: tuple
    temporary_world_goal_xy: tuple
    previous_local_goal_xy: tuple
    collision: bool
    timestamp_s: float
    episode_id: int
    waypoint_index: int

    def __post_init__(self):
        depth = np.asarray(self.depth)
        if depth.ndim < 2 or not np.all(np.isfinite(depth)):
            raise ValueError("depth must be a finite image array")
        object.__setattr__(self, "depth", depth.copy())
        for field in (
            "robot_xy",
            "global_goal_xy",
            "local_goal_xy",
            "temporary_world_goal_xy",
            "previous_local_goal_xy",
        ):
            object.__setattr__(self, field, _pair(getattr(self, field), field))
        object.__setattr__(self, "robot_yaw", float(self.robot_yaw))
        object.__setattr__(self, "collision", bool(self.collision))
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        object.__setattr__(self, "episode_id", int(self.episode_id))
        object.__setattr__(self, "waypoint_index", int(self.waypoint_index))


class ClosedLoopDatasetWriter:
    """Write depth arrays and append-only labels for actual closed-loop steps."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._records = self.root.joinpath("records.jsonl").open(
            "w", encoding="utf-8"
        )
        self._index = 0

    def append(self, sample):
        if not isinstance(sample, OracleSample):
            raise TypeError("sample must be an OracleSample")
        depth_name = f"depth_{self._index:06d}.npy"
        np.save(self.root / depth_name, sample.depth)
        record = {
            "depth_file": depth_name,
            "robot_xy": list(sample.robot_xy),
            "robot_yaw": sample.robot_yaw,
            "global_goal_xy": list(sample.global_goal_xy),
            "local_goal_xy": list(sample.local_goal_xy),
            "temporary_world_goal_xy": list(sample.temporary_world_goal_xy),
            "previous_local_goal_xy": list(sample.previous_local_goal_xy),
            "collision": sample.collision,
            "timestamp_s": sample.timestamp_s,
            "episode_id": sample.episode_id,
            "waypoint_index": sample.waypoint_index,
        }
        self._records.write(json.dumps(record) + "\n")
        self._records.flush()
        self._index += 1
        return record

    def close(self):
        if not self._records.closed:
            self._records.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
