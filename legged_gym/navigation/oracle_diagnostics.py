"""Pure geometry and deterministic labels for Oracle collision diagnostics."""

import numpy as np


APPROACH_COLLISION = "APPROACH_COLLISION"
POST_SWITCH_COLLISION = "POST_SWITCH_COLLISION"
STRAIGHT_CORRIDOR_COLLISION = "STRAIGHT_CORRIDOR_COLLISION"
CORNER_CUT_COLLISION = "CORNER_CUT_COLLISION"
FINAL_APPROACH_COLLISION = "FINAL_APPROACH_COLLISION"
OTHER_COLLISION = "OTHER"

COLLISION_CLASSES = (
    APPROACH_COLLISION,
    POST_SWITCH_COLLISION,
    STRAIGHT_CORRIDOR_COLLISION,
    CORNER_CUT_COLLISION,
    FINAL_APPROACH_COLLISION,
    OTHER_COLLISION,
)
POST_SWITCH_WINDOWS = (5, 10, 20)
TURN_THRESHOLD_DEG = 45.0


def _xy(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain two finite values")
    return result


def point_to_segment_distance(point_xy, segment_start_xy, segment_end_xy):
    point = _xy(point_xy, "point_xy")
    start = _xy(segment_start_xy, "segment_start_xy")
    end = _xy(segment_end_xy, "segment_end_xy")
    delta = end - start
    length_squared = float(delta.dot(delta))
    if length_squared <= 1.0e-12:
        return float(np.linalg.norm(point - start))
    projection = float((point - start).dot(delta) / length_squared)
    projection = min(max(projection, 0.0), 1.0)
    return float(np.linalg.norm(point - (start + projection * delta)))


def local_goal_polar(local_goal_xy):
    goal = _xy(local_goal_xy, "local_goal_xy")
    return float(np.linalg.norm(goal)), float(np.degrees(np.arctan2(goal[1], goal[0])))


def reachability_clip_ratio(raw_local_goal_xy, filtered_local_goal_xy):
    raw = _xy(raw_local_goal_xy, "raw_local_goal_xy")
    filtered = _xy(filtered_local_goal_xy, "filtered_local_goal_xy")
    raw_radius = float(np.linalg.norm(raw))
    if raw_radius <= 1.0e-12:
        return 0.0
    return float(max(0.0, 1.0 - np.linalg.norm(filtered) / raw_radius))


def nearest_wall_clearance(robot_xy, wall_centers_xy, wall_size_xy, robot_collision_radius):
    """Return nearest wall surface distance and exterior robot clearance.

    Wall geometry is modeled as axis-aligned rectangles centered at
    ``wall_centers_xy``.  The surface distance is measured from the robot
    center to the rectangle, then the robot radius is subtracted exactly once.
    """
    robot = _xy(robot_xy, "robot_xy")
    centers = np.asarray(wall_centers_xy, dtype=np.float64)
    if centers.size == 0:
        return float("inf"), float("inf")
    if centers.ndim != 2 or centers.shape[1] != 2 or not np.all(np.isfinite(centers)):
        raise ValueError("wall_centers_xy must have shape (N, 2)")
    size = np.asarray(wall_size_xy, dtype=np.float64)
    if size.ndim == 0:
        size = np.repeat(size, 2)
    if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        raise ValueError("wall_size_xy must contain two positive finite values")
    radius = float(robot_collision_radius)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("robot_collision_radius must be finite and non-negative")
    offset = np.maximum(np.abs(centers - robot[None, :]) - size[None, :] / 2.0, 0.0)
    surface_distance = float(np.min(np.linalg.norm(offset, axis=1)))
    return surface_distance, surface_distance - radius


def _is_corner(current_cell, waypoint_cell, next_bfs_cell):
    if current_cell is None or waypoint_cell is None or next_bfs_cell is None:
        return False
    first = np.asarray(waypoint_cell, dtype=np.float64) - np.asarray(current_cell, dtype=np.float64)
    second = np.asarray(next_bfs_cell, dtype=np.float64) - np.asarray(waypoint_cell, dtype=np.float64)
    if np.linalg.norm(first) <= 1.0e-12 or np.linalg.norm(second) <= 1.0e-12:
        return False
    angle = abs(float(np.degrees(np.arctan2(
        first[0] * second[1] - first[1] * second[0], first.dot(second)
    ))))
    return angle >= TURN_THRESHOLD_DEG


def classify_collision(
    *,
    phase,
    steps_since_goal_switch,
    delta_bearing_deg,
    waypoint_reached,
    current_cell,
    waypoint_cell,
    next_bfs_cell,
):
    """Return overlapping collision labels and one deterministic primary class."""
    final_approach = str(phase) == "FINAL_APPROACH"
    post_switch = 0 <= int(steps_since_goal_switch) <= 10
    corner = _is_corner(current_cell, waypoint_cell, next_bfs_cell) or (
        abs(float(delta_bearing_deg)) >= TURN_THRESHOLD_DEG
    )
    straight = not corner and waypoint_cell is not None and current_cell != waypoint_cell
    approach = str(phase) == "NAVIGATE" and waypoint_cell is not None
    if final_approach:
        primary = FINAL_APPROACH_COLLISION
    elif post_switch:
        primary = POST_SWITCH_COLLISION
    elif corner and not bool(waypoint_reached):
        primary = CORNER_CUT_COLLISION
    elif straight and not bool(waypoint_reached):
        primary = STRAIGHT_CORRIDOR_COLLISION
    elif approach:
        primary = APPROACH_COLLISION
    else:
        primary = OTHER_COLLISION
    return {
        "collision_class_primary": primary,
        "is_final_approach": final_approach,
        "is_post_switch": post_switch,
        "is_corner": corner,
        "is_straight_corridor": straight,
        "is_approach": approach,
    }


def summarize_collision_diagnostics(collision_records, episode_count):
    records = tuple(collision_records)
    episodes = max(int(episode_count), 1)
    counts = {name: sum(row.get("collision_class_primary") == name for row in records) for name in COLLISION_CLASSES}
    collision_count = len(records)
    overlap_counts = {
        name: sum(bool(row.get(name, False)) for row in records)
        for name in ("is_final_approach", "is_post_switch", "is_corner", "is_straight_corridor", "is_approach")
    }
    return {
        "collision_count": collision_count,
        "collision_class_counts": counts,
        "collision_class_rates": {
            name: float(count / collision_count) if collision_count else 0.0
            for name, count in counts.items()
        },
        "overlap_label_counts": overlap_counts,
        "collision_post_switch_window_counts": {
            str(window): sum(
                0 <= int(row.get("steps_since_goal_switch", -1)) <= window
                for row in records
            )
            for window in POST_SWITCH_WINDOWS
        },
    }
