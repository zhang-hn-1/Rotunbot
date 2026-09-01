"""Deterministic mirrored L-turn geometry for the V1 visual MVP."""

from dataclasses import dataclass

import numpy as np

from .corridor_scenarios import CorridorScenario, CorridorTurn, make_l_scenario
from .v62_corridor_task import make_wall_segments
from .visual_corridor_v1 import V1_WALL_THICKNESS_M


@dataclass(frozen=True)
class LTurnGeometry:
    scenario: CorridorScenario
    waypoints: np.ndarray
    wall_segments: tuple
    obstacle_aabbs: tuple
    turn_direction: int
    corner_clearance_m: float


def _mirror_scenario(scenario, direction):
    if direction == 1:
        return scenario
    centerline = scenario.centerline.copy()
    centerline[:, 1] *= -1.0
    turns = tuple(
        CorridorTurn(
            -int(turn.direction),
            np.asarray((turn.center_xy[0], -turn.center_xy[1])),
            turn.start_index,
            turn.end_index,
        )
        for turn in scenario.turns
    )
    return CorridorScenario(
        family=scenario.family,
        width_m=scenario.width_m,
        centerline=centerline,
        start_xy=centerline[0].copy(),
        goal_xy=centerline[-1].copy(),
        turns=turns,
        seed=scenario.seed,
    )


def _wall_obstacle_aabbs(segments, width_m):
    half_width = float(width_m) / 2.0
    half_thickness = float(V1_WALL_THICKNESS_M) / 2.0
    obstacles = []
    for start, end in segments:
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        delta = end - start
        length = float(np.linalg.norm(delta))
        normal = np.asarray((-delta[1], delta[0])) / length
        midpoint = 0.5 * (start + end)
        half_extent = np.maximum(np.abs(delta) / 2.0, half_thickness)
        for side in (-1.0, 1.0):
            center = midpoint + side * half_width * normal
            obstacles.append((tuple(center), tuple(half_extent)))
    return tuple(obstacles)


def build_l_turn_geometry(
    turn_direction,
    width_m=3.0,
    first_segment_length_m=1.5,
    second_segment_length_m=1.5,
    turn_radius_m=2.0,
    corner_clearance_m=0.60,
    reach_radius_m=0.35,
):
    """Build a fixed, wide, mirrored L with safe pre/post-turn waypoints."""
    direction_name = str(turn_direction).lower()
    if direction_name not in ("left", "right"):
        raise ValueError("turn_direction must be 'left' or 'right'")
    if float(corner_clearance_m) <= 0.0:
        raise ValueError("corner_clearance_m must be positive")
    direction = 1 if direction_name == "left" else -1
    base = make_l_scenario(
        width_m=float(width_m),
        straight_m=float(first_segment_length_m),
        turn_radius_m=float(turn_radius_m),
        seed=0,
    )
    scenario = _mirror_scenario(base, direction)
    segments = make_wall_segments(scenario.centerline)
    corner_x = float(first_segment_length_m) + float(turn_radius_m)
    corner_y = direction * float(turn_radius_m)
    clearance = float(corner_clearance_m)
    waypoints = np.asarray(
        (
            scenario.start_xy,
            (float(first_segment_length_m) - clearance, 0.0),
            (corner_x, direction * (float(turn_radius_m) + clearance)),
            scenario.goal_xy,
        ),
        dtype=np.float64,
    )
    if float(reach_radius_m) <= 0.0:
        raise ValueError("reach_radius_m must be positive")
    # Keep the named corner values in the construction for auditability and
    # assert that the post-turn waypoint is on the intended side.
    if not np.isfinite(corner_y) or direction * waypoints[2, 1] <= 0.0:
        raise ValueError("invalid L-turn corner geometry")
    return LTurnGeometry(
        scenario=scenario,
        waypoints=waypoints,
        wall_segments=segments,
        obstacle_aabbs=_wall_obstacle_aabbs(segments, width_m),
        turn_direction=direction,
        corner_clearance_m=clearance,
    )
