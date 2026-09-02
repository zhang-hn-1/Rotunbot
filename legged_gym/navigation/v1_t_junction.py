"""Pure, deterministic geometry for the V1 T-junction MVP."""

from dataclasses import dataclass
import math

import numpy as np

from .corridor_scenarios import CorridorScenario


V1_WALL_THICKNESS_M = 0.10


@dataclass(frozen=True)
class TJunctionGeometry:
    scenario: CorridorScenario
    waypoints: np.ndarray
    wall_segments: tuple
    obstacle_aabbs: tuple
    branch_direction: int
    reach_radius_m: float


def _finite_positive(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _normalize_branch(branch):
    value = str(branch).upper()
    if value in ("LEFT", "T_LEFT"):
        return 1
    if value in ("RIGHT", "T_RIGHT"):
        return -1
    raise ValueError("branch must be LEFT, RIGHT, T_LEFT, or T_RIGHT")


def _obstacle_aabbs(segments):
    half_thickness = V1_WALL_THICKNESS_M / 2.0
    return tuple(
        (
            tuple(np.minimum(start, end) - half_thickness),
            tuple(np.maximum(start, end) + half_thickness),
        )
        for start, end in segments
    )


def build_t_junction_geometry(
    branch: str,
    width_m=3.0,
    stem_length_m=2.5,
    branch_length_m=2.5,
    reach_radius_m=0.35,
):
    """Build a symmetric T; only the selected goal/waypoint side changes."""
    width = _finite_positive("width_m", width_m)
    stem = _finite_positive("stem_length_m", stem_length_m)
    branch_length = _finite_positive("branch_length_m", branch_length_m)
    reach = _finite_positive("reach_radius_m", reach_radius_m)
    direction = _normalize_branch(branch)
    half_width = width / 2.0
    junction = np.asarray((stem, 0.0), dtype=np.float64)
    goal = np.asarray((stem, direction * branch_length), dtype=np.float64)
    waypoints = np.asarray(((0.0, 0.0), junction, goal), dtype=np.float64)
    segments = (
        (np.asarray((0.0, half_width)), junction + (0.0, half_width)),
        (np.asarray((0.0, -half_width)), junction + (0.0, -half_width)),
        (junction + (-half_width, 0.0), junction + (-half_width, branch_length)),
        (junction + (half_width, 0.0), junction + (half_width, branch_length)),
        (junction + (-half_width, 0.0), junction + (-half_width, -branch_length)),
        (junction + (half_width, 0.0), junction + (half_width, -branch_length)),
    )
    scenario = CorridorScenario(
        family="t_junction",
        width_m=width,
        centerline=waypoints.copy(),
        start_xy=waypoints[0].copy(),
        goal_xy=waypoints[-1].copy(),
        turns=(),
        seed=0,
    )
    return TJunctionGeometry(
        scenario=scenario,
        waypoints=waypoints,
        wall_segments=segments,
        obstacle_aabbs=_obstacle_aabbs(segments),
        branch_direction=direction,
        reach_radius_m=reach,
    )


def classify_t_branch(goal_xy, deadband_m=0.35):
    """Classify the branch from a goal/terminal XY point."""
    point = np.asarray(goal_xy, dtype=np.float64)
    deadband = _finite_positive("deadband_m", deadband_m)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("goal_xy must contain two finite values")
    if abs(float(point[1])) <= deadband:
        return "UNDECIDED"
    return "LEFT" if point[1] > 0.0 else "RIGHT"
