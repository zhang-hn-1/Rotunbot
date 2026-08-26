"""Measured action reachability and deterministic local-goal filtering."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


def _pair(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain two finite values")
    return (float(result[0]), float(result[1]))


@dataclass(frozen=True)
class ReachabilitySample:
    action0: float
    action1: float
    displacement_body_xy: tuple
    steady_state_velocity_body_xy: tuple
    rise_time_s: float
    cross_axis_coupling: float
    action_clipping: bool
    joint_response: tuple

    def __post_init__(self):
        object.__setattr__(self, "action0", float(self.action0))
        object.__setattr__(self, "action1", float(self.action1))
        object.__setattr__(
            self, "displacement_body_xy", _pair(self.displacement_body_xy, "displacement_body_xy")
        )
        object.__setattr__(
            self,
            "steady_state_velocity_body_xy",
            _pair(self.steady_state_velocity_body_xy, "steady_state_velocity_body_xy"),
        )
        object.__setattr__(self, "rise_time_s", float(self.rise_time_s))
        object.__setattr__(self, "cross_axis_coupling", float(self.cross_axis_coupling))
        object.__setattr__(self, "action_clipping", bool(self.action_clipping))
        object.__setattr__(self, "joint_response", _pair(self.joint_response, "joint_response"))


@dataclass(frozen=True)
class ReachabilityEnvelope:
    """Radial limits indexed by measured bearing, not an assumed ellipse."""

    bearings_rad: tuple
    max_radius_m: tuple

    def __post_init__(self):
        bearings = tuple(float(value) for value in self.bearings_rad)
        radii = tuple(float(value) for value in self.max_radius_m)
        if not bearings or len(bearings) != len(radii):
            raise ValueError("bearings_rad and max_radius_m must have equal non-zero length")
        if not all(np.isfinite(value) for value in bearings + radii) or any(value < 0 for value in radii):
            raise ValueError("reachability envelope values must be finite and non-negative")
        object.__setattr__(self, "bearings_rad", bearings)
        object.__setattr__(self, "max_radius_m", radii)

    @classmethod
    def from_samples(cls, samples, angular_bins=16):
        samples = tuple(samples)
        if int(angular_bins) < 1:
            raise ValueError("angular_bins must be positive")
        if not samples:
            raise ValueError("at least one reachability sample is required")
        bins = int(angular_bins)
        bearings = np.linspace(0.0, 2.0 * np.pi, bins, endpoint=False)
        radii = np.zeros(bins, dtype=np.float64)
        observed = np.zeros(bins, dtype=bool)
        for sample in samples:
            displacement = np.asarray(sample.displacement_body_xy, dtype=np.float64)
            radius = float(np.linalg.norm(displacement))
            if radius <= 1.0e-12:
                continue
            angle = float(np.mod(np.arctan2(displacement[1], displacement[0]), 2.0 * np.pi))
            distance = np.abs(np.angle(np.exp(1j * (angle - bearings))))
            index = int(np.argmin(distance))
            radii[index] = max(radii[index], radius)
            observed[index] = True
        if not np.any(observed):
            raise ValueError("samples contain no non-zero displacement")
        # A bin with no direct measurement inherits the nearest measured bin;
        # this is interpolation of measured data, not an ellipse assumption.
        for index in np.flatnonzero(~observed):
            distances = np.abs(np.angle(np.exp(1j * (bearings[index] - bearings[observed]))))
            radii[index] = radii[observed][int(np.argmin(distances))]
        return cls(tuple(bearings.tolist()), tuple(radii.tolist()))

    def _limit_for_angle(self, angle):
        bearings = np.asarray(self.bearings_rad, dtype=np.float64)
        distance = np.abs(np.angle(np.exp(1j * (float(angle) - bearings))))
        return float(self.max_radius_m[int(np.argmin(distance))])

    def filter(self, local_goal_xy):
        """Clip an out-of-envelope goal along its measured bearing."""
        goal = np.asarray(local_goal_xy, dtype=np.float64)
        if goal.shape != (2,) or not np.all(np.isfinite(goal)):
            raise ValueError("local_goal_xy must contain two finite values")
        radius = float(np.linalg.norm(goal))
        if radius <= 1.0e-12:
            return goal.copy()
        limit = self._limit_for_angle(np.arctan2(goal[1], goal[0]))
        return goal.copy() if radius <= limit else goal * (limit / radius)


def save_samples(path, samples):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(sample) for sample in samples]
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_samples(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ReachabilitySample(**item) for item in payload]


def save_envelope(path, envelope):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(envelope), indent=2), encoding="utf-8")


def load_envelope(path):
    return ReachabilityEnvelope(**json.loads(Path(path).read_text(encoding="utf-8")))
