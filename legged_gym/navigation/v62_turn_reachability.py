"""Pure metrics and evidence helpers for Frozen V62 turn feasibility audits."""

from dataclasses import dataclass
import ast
import csv
import math
from pathlib import Path

import numpy as np


FAILURE_REASONS = (
    "INNER_WALL_COLLISION",
    "OUTER_WALL_COLLISION",
    "END_WALL_COLLISION",
    "GOVERNOR_STALL",
    "TURN_RADIUS_INFEASIBLE",
    "WAYPOINT_PROGRESS_FAILURE",
    "TIMEOUT",
    "UNKNOWN",
)


@dataclass(frozen=True)
class FrozenV62Runtime:
    """Resolved runtime values recorded by a command-only V62 audit."""

    policy_dt_s: float
    upper_command_hz: float
    hold_steps: int
    max_forward_speed_mps: float
    max_yaw_rate_rps: float
    minimum_turn_radius_m: float
    maximum_linear_acceleration_mps2: float
    maximum_yaw_acceleration_rps2: float
    robot_radius_m: float

    @property
    def command_period_s(self):
        return float(self.policy_dt_s) * int(self.hold_steps)


@dataclass(frozen=True)
class TurnTrialSpec:
    trial_id: str
    side: str
    mode: str
    requested_v_mps: float
    requested_w_rps: float
    trigger_step: int
    repeat: int
    horizon_steps: int


@dataclass(frozen=True)
class TurnTrialResult:
    trial_id: str
    side: str
    mode: str
    requested_v_mps: float
    requested_w_rps: float
    projected_v_mps: float
    projected_w_rps: float
    actual_v_mps: float
    actual_w_rps: float
    instantaneous_radius_m: float
    path_radius_m: float
    heading_change_rad: float
    time_to_90_s: float
    stabilization_distance_m: float
    max_forward_overshoot_m: float
    max_lateral_excursion_m: float
    min_clearance_m: float
    p05_clearance_m: float
    median_clearance_m: float
    collision: bool
    timeout: bool
    governor_stall: bool
    turn_radius_infeasible: bool
    failure_reason: str


def instantaneous_radius(v_actual_mps, w_actual_rps, yaw_epsilon=1.0e-6):
    """Return ``abs(v / w)``; near-zero yaw rates have infinite radius."""
    v = float(v_actual_mps)
    w = float(w_actual_rps)
    if not math.isfinite(v) or not math.isfinite(w):
        raise ValueError("actual velocity must be finite")
    if abs(w) <= float(yaw_epsilon):
        return float("inf")
    return abs(v / w)


def point_aabb_raw_distance(point_xy, center, half_extent):
    point = np.asarray(point_xy, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    extent = np.asarray(half_extent, dtype=np.float64)
    if point.shape != (2,) or center.shape != (2,) or extent.shape != (2,):
        raise ValueError("point, center, and half_extent must have shape (2,)")
    if not np.isfinite(np.concatenate((point, center, extent))).all() or np.any(extent < 0.0):
        raise ValueError("AABB values must be finite and nonnegative")
    delta = np.maximum(np.abs(point - center) - extent, 0.0)
    return float(np.linalg.norm(delta))


def point_clearance(point_xy, aabbs, robot_radius_m=0.0):
    """Return exact Euclidean distance to the nearest AABB minus robot radius."""
    radius = float(robot_radius_m)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("robot_radius_m must be finite and nonnegative")
    aabbs = tuple(aabbs)
    if not aabbs:
        return float("inf")
    raw = min(point_aabb_raw_distance(point_xy, center, extent) for center, extent in aabbs)
    return raw - radius


def inflate_aabbs(aabbs, robot_radius_m, extra_margin_m=0.0):
    """Conservatively inflate AABB half-extents for robot-center feasibility."""
    amount = float(robot_radius_m) + float(extra_margin_m)
    if not math.isfinite(amount) or amount < 0.0:
        raise ValueError("inflation amount must be finite and nonnegative")
    return tuple(
        (
            tuple(float(value) for value in center),
            tuple(float(value) + amount for value in half_extent),
        )
        for center, half_extent in aabbs
    )


def segment_min_clearance(start_xy, end_xy, aabbs, robot_radius_m=0.0, samples=128):
    """Return a conservative swept-segment clearance by uniform subdivision."""
    count = int(samples)
    if count < 2:
        raise ValueError("samples must be at least two")
    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    if start.shape != (2,) or end.shape != (2,) or not np.isfinite(np.vstack((start, end))).all():
        raise ValueError("segment endpoints must be finite XY points")
    points = np.linspace(start, end, count)
    return float(min(point_clearance(point, aabbs, robot_radius_m) for point in points))


def polyline_clearance(points_xy, aabbs, robot_radius_m=0.0, samples_per_segment=64):
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("points_xy must be a nonempty finite [N, 2] array")
    values = [point_clearance(point, aabbs, robot_radius_m) for point in points]
    for start, end in zip(points[:-1], points[1:]):
        values.append(segment_min_clearance(start, end, aabbs, robot_radius_m, samples_per_segment))
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "p05": float(np.percentile(values, 5.0)),
        "median": float(np.median(values)),
    }


def fit_path_radius(points_xy, turn_start_index=0, turn_end_index=None):
    """Fit a circle to measured turn points and return radius and RMSE."""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("at least three finite XY points are required")
    start = max(0, int(turn_start_index))
    end = len(points) if turn_end_index is None else min(len(points), int(turn_end_index))
    selected = points[start:end]
    if len(selected) < 3:
        raise ValueError("turn interval must contain at least three points")
    x = selected[:, 0]
    y = selected[:, 1]
    matrix = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(selected))))
    target = x * x + y * y
    solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    cx, cy, constant = solution
    radius_sq = float(constant + cx * cx + cy * cy)
    if radius_sq <= 0.0 or not math.isfinite(radius_sq):
        raise ValueError("circle fit produced an invalid radius")
    radius = math.sqrt(radius_sq)
    residual = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius
    return {
        "radius_m": float(radius),
        "center_xy": (float(cx), float(cy)),
        "fit_rmse_m": float(np.sqrt(np.mean(residual * residual))),
        "sample_count": int(len(selected)),
    }


def classify_failure_reason(*, collision=False, collision_wall=None, timeout=False,
                            governor_stall=False, turn_radius_infeasible=False,
                            waypoint_progress_failure=False):
    """Return the primary reason using collision-first release precedence."""
    if collision:
        wall = str(collision_wall or "").upper()
        if wall in ("INNER", "INNER_WALL"):
            return "INNER_WALL_COLLISION"
        if wall in ("OUTER", "OUTER_WALL"):
            return "OUTER_WALL_COLLISION"
        if wall in ("END", "END_WALL"):
            return "END_WALL_COLLISION"
        return "UNKNOWN"
    if governor_stall:
        return "GOVERNOR_STALL"
    if turn_radius_infeasible:
        return "TURN_RADIUS_INFEASIBLE"
    if waypoint_progress_failure:
        return "WAYPOINT_PROGRESS_FAILURE"
    if timeout:
        return "TIMEOUT"
    return "UNKNOWN"


def aggregate_turn_trials(rows):
    """Aggregate measured rows while retaining failure and clearance counts."""
    rows = list(rows)
    if not rows:
        raise ValueError("at least one trial row is required")
    clearance = np.asarray([float(row["min_clearance_m"]) for row in rows], dtype=np.float64)
    path_radius = np.asarray([float(row["path_radius_m"]) for row in rows], dtype=np.float64)
    instant_radius = np.asarray([float(row["instantaneous_radius_m"]) for row in rows], dtype=np.float64)
    return {
        "trial_count": len(rows),
        "success_count": sum(not bool(row.get("collision")) and not bool(row.get("timeout")) for row in rows),
        "collision_count": sum(bool(row.get("collision")) for row in rows),
        "timeout_count": sum(bool(row.get("timeout")) for row in rows),
        "failure_reasons": {
            reason: sum(str(row.get("failure_reason")) == reason for row in rows)
            for reason in FAILURE_REASONS
        },
        "min_clearance_m": float(np.min(clearance)),
        "p05_clearance_m": float(np.percentile(clearance, 5.0)),
        "median_clearance_m": float(np.median(clearance)),
        "mean_path_radius_m": float(np.mean(path_radius)),
        "mean_instantaneous_radius_m": float(np.mean(instant_radius[np.isfinite(instant_radius)])) if np.isfinite(instant_radius).any() else float("inf"),
    }


def sorted_geometry_candidates(candidates):
    """Order candidates from narrow/low-margin to larger geometry."""
    return sorted(
        list(candidates),
        key=lambda item: (
            float(item.get("branch_half_width", float("inf"))),
            float(item.get("stem_half_width", float("inf"))),
            float(item.get("junction_opening_length", float("inf"))),
            float(item.get("terminal_margin", float("inf"))),
        ),
    )


def _parse_trace(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    return value or []


def extract_t_failure_baseline(left_csv, right_csv):
    """Extract available baseline evidence without inventing missing fields."""
    result = {}
    for side, path in (("T_LEFT", left_csv), ("T_RIGHT", right_csv)):
        rows = load_csv_rows(path)
        if not rows:
            raise ValueError("no rows in %s baseline CSV" % side)
        collision_positions = []
        collision_steps = []
        clearance = []
        turn_positions = []
        overshoot = []
        for row in rows:
            trace = _parse_trace(row.get("failure_trace"))
            if not trace:
                continue
            terminal = trace[-1]
            position = terminal.get("position_xy")
            if position is not None:
                collision_positions.append(tuple(float(value) for value in position))
                overshoot.append(float(position[0]) - 2.5)
            collision_steps.append(int(row.get("episode_steps", 0)))
            clearance.extend(float(item["clearance_m"]) for item in trace if "clearance_m" in item)
            turn = next((item for item in trace if int(item.get("waypoint_index", 0)) >= 2), None)
            if turn and turn.get("position_xy") is not None:
                turn_positions.append(tuple(float(value) for value in turn["position_xy"]))
        signed = 1.0 if side == "T_LEFT" else -1.0
        lateral = [signed * point[1] for point in collision_positions]
        result[side] = {
            "episodes": len(rows),
            "collision_count": sum(str(row.get("collision")).lower() == "true" for row in rows),
            "success_count": sum(str(row.get("success")).lower() == "true" for row in rows),
            "wrong_turn_count": sum(str(row.get("wrong_turn")).lower() == "true" for row in rows),
            "collision_position_mean": tuple(float(np.mean([point[index] for point in collision_positions])) for index in (0, 1)) if collision_positions else None,
            "collision_position_range": [(float(np.min([point[index] for point in collision_positions])), float(np.max([point[index] for point in collision_positions]))) for index in (0, 1)] if collision_positions else None,
            "collision_step_mean": float(np.mean(collision_steps)) if collision_steps else None,
            "collision_velocity": None,
            "collision_yaw": None,
            "turn_start_position_mean": tuple(float(np.mean([point[index] for point in turn_positions])) for index in (0, 1)) if turn_positions else None,
            "maximum_lateral_excursion_m": float(np.max(lateral)) if lateral else None,
            "maximum_forward_overshoot_m": float(np.max(overshoot)) if overshoot else None,
            "minimum_raw_clearance_m": float(np.min(clearance)) if clearance else None,
            "terminal_waypoint_index": max((int(item.get("waypoint_index", 0)) for row in rows for item in _parse_trace(row.get("failure_trace"))), default=None),
            "commanded_velocity": None,
            "actual_velocity": None,
            "missing_fields": ["collision_velocity", "collision_yaw", "commanded_velocity", "actual_velocity"],
        }
    return result


def load_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
