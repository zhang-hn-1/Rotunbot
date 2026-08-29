"""Deterministic centerline geometries for the corridor curriculum."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CorridorTurn:
    """One quarter-turn in a corridor centerline."""

    direction: int
    center_xy: np.ndarray
    start_index: int
    end_index: int


@dataclass(frozen=True)
class CorridorScenario:
    """Sampled centerline and metadata for one deterministic scenario."""

    family: str
    width_m: float
    centerline: np.ndarray
    start_xy: np.ndarray
    goal_xy: np.ndarray
    turns: tuple
    seed: int

    @property
    def path_length_m(self):
        if len(self.centerline) < 2:
            return 0.0
        return float(
            np.linalg.norm(np.diff(self.centerline, axis=0), axis=1).sum()
        )


def _validate(width_m, turn_radius_m=None, straight_m=None):
    if float(width_m) <= 0.0:
        raise ValueError("width_m must be positive")
    if turn_radius_m is not None and float(turn_radius_m) <= 0.0:
        raise ValueError("turn_radius_m must be positive")
    if straight_m is not None and float(straight_m) <= 0.0:
        raise ValueError("straight_m must be positive")


def _line(start, end, step=0.05, include_start=True):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    distance = float(np.linalg.norm(end - start))
    count = max(1, int(math.ceil(distance / step)))
    values = np.linspace(start, end, count + 1)
    return values if include_start else values[1:]


def _arc(center, radius, start_angle, end_angle, step=0.05, include_start=True):
    arc_length = abs(end_angle - start_angle) * radius
    count = max(1, int(math.ceil(arc_length / step)))
    angles = np.linspace(start_angle, end_angle, count + 1)
    values = np.stack(
        (center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)),
        axis=1,
    )
    return values if include_start else values[1:]


def _scenario(family, width_m, points, turns, seed):
    centerline = np.asarray(points, dtype=np.float64)
    return CorridorScenario(
        family=family,
        width_m=float(width_m),
        centerline=centerline,
        start_xy=centerline[0].copy(),
        goal_xy=centerline[-1].copy(),
        turns=tuple(turns),
        seed=int(seed),
    )


def make_straight_scenario(width_m, length_m, seed):
    _validate(width_m, straight_m=length_m)
    points = _line((0.0, 0.0), (float(length_m), 0.0))
    return _scenario("straight", width_m, points, (), seed)


def make_l_scenario(width_m, straight_m, turn_radius_m, seed):
    _validate(width_m, turn_radius_m, straight_m)
    straight_m = float(straight_m)
    radius = float(turn_radius_m)
    first = _line((0.0, 0.0), (straight_m, 0.0))
    center = np.asarray((straight_m, radius), dtype=np.float64)
    arc = _arc(center, radius, -math.pi / 2.0, 0.0, include_start=False)
    end = arc[-1]
    second = _line(end, (end[0], end[1] + straight_m), include_start=False)
    points = np.concatenate((first, arc, second), axis=0)
    turn = CorridorTurn(1, center, len(first) - 1, len(first) + len(arc) - 1)
    return _scenario("l", width_m, points, (turn,), seed)


def make_double_turn_scenario(width_m, turn_radius_m, handedness, seed):
    _validate(width_m, turn_radius_m, straight_m=3.0)
    if handedness not in ("left_right", "right_left"):
        raise ValueError("handedness must be 'left_right' or 'right_left'")

    radius = float(turn_radius_m)
    directions = (1, -1) if handedness == "left_right" else (-1, 1)
    points = [_line((0.0, 0.0), (3.0, 0.0))]
    turns = []
    heading = 0.0
    current = points[0][-1]
    for index, direction in enumerate(directions):
        center = current + np.asarray(
            (-direction * radius * math.sin(heading), direction * radius * math.cos(heading))
        )
        start_angle = math.atan2(current[1] - center[1], current[0] - center[0])
        end_angle = start_angle + direction * math.pi / 2.0
        arc = _arc(center, radius, start_angle, end_angle, include_start=False)
        start_index = sum(len(part) for part in points) - 1
        points.append(arc)
        end_index = start_index + len(arc)
        turns.append(CorridorTurn(direction, center, start_index, end_index))
        current = arc[-1]
        heading += direction * math.pi / 2.0
        straight_end = current + np.asarray(
            (3.0 * math.cos(heading), 3.0 * math.sin(heading))
        )
        line = _line(current, straight_end, include_start=False)
        points.append(line)
        current = line[-1]
    return _scenario("double_turn", width_m, np.concatenate(points), turns, seed)
