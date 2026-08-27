"""Pure geometry and deterministic labels for Oracle collision diagnostics."""

import numpy as np


PLANNED_STRAIGHT_COLLISION = "PLANNED_STRAIGHT_COLLISION"
PLANNED_CORNER_COLLISION = "PLANNED_CORNER_COLLISION"
OFF_PATH_COLLISION = "OFF_PATH_COLLISION"
POST_SWITCH_COLLISION = "POST_SWITCH_COLLISION"
FINAL_APPROACH_COLLISION = "FINAL_APPROACH_COLLISION"
OTHER_COLLISION = "OTHER"

# Compatibility aliases for consumers of Diagnostics v1.
APPROACH_COLLISION = PLANNED_STRAIGHT_COLLISION
STRAIGHT_CORRIDOR_COLLISION = PLANNED_STRAIGHT_COLLISION
CORNER_CUT_COLLISION = PLANNED_CORNER_COLLISION

COLLISION_CLASSES = (
    PLANNED_STRAIGHT_COLLISION,
    PLANNED_CORNER_COLLISION,
    OFF_PATH_COLLISION,
    POST_SWITCH_COLLISION,
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


def _is_corner(planned_from_cell, planned_waypoint_cell, planned_next_cell, delta_bearing_deg):
    if planned_from_cell is None or planned_waypoint_cell is None or planned_next_cell is None:
        return abs(float(delta_bearing_deg)) >= TURN_THRESHOLD_DEG
    first = np.asarray(planned_waypoint_cell, dtype=np.float64) - np.asarray(planned_from_cell, dtype=np.float64)
    second = np.asarray(planned_next_cell, dtype=np.float64) - np.asarray(planned_waypoint_cell, dtype=np.float64)
    if np.linalg.norm(first) <= 1.0e-12 or np.linalg.norm(second) <= 1.0e-12:
        return False
    angle = abs(float(np.degrees(np.arctan2(
        first[0] * second[1] - first[1] * second[0], first.dot(second)
    ))))
    return angle >= TURN_THRESHOLD_DEG or abs(float(delta_bearing_deg)) >= TURN_THRESHOLD_DEG


def classify_collision(
    *,
    phase,
    steps_since_goal_switch,
    delta_bearing_deg,
    waypoint_reached,
    actual_current_cell=None,
    planned_from_cell=None,
    planned_waypoint_cell=None,
    planned_next_cell=None,
    current_cell=None,
    waypoint_cell=None,
    next_bfs_cell=None,
):
    """Classify using frozen planned geometry and separately measured drift."""
    if actual_current_cell is None:
        actual_current_cell = current_cell
    if planned_waypoint_cell is None:
        planned_waypoint_cell = waypoint_cell
    if planned_next_cell is None:
        planned_next_cell = next_bfs_cell
    if planned_from_cell is None:
        planned_from_cell = actual_current_cell
    final_approach = str(phase) == "FINAL_APPROACH"
    post_switch = 0 <= int(steps_since_goal_switch) <= 10
    corner = _is_corner(
        planned_from_cell, planned_waypoint_cell, planned_next_cell, delta_bearing_deg
    )
    straight = not corner and planned_waypoint_cell is not None
    off_path = (
        planned_waypoint_cell is not None
        and actual_current_cell is not None
        and tuple(actual_current_cell) not in {
            tuple(planned_from_cell), tuple(planned_waypoint_cell)
        }
    )
    approach = str(phase) == "NAVIGATE" and planned_waypoint_cell is not None
    if final_approach:
        primary = FINAL_APPROACH_COLLISION
    elif post_switch:
        primary = POST_SWITCH_COLLISION
    elif off_path:
        primary = OFF_PATH_COLLISION
    elif corner and not bool(waypoint_reached):
        primary = PLANNED_CORNER_COLLISION
    elif straight and not bool(waypoint_reached):
        primary = PLANNED_STRAIGHT_COLLISION
    elif approach:
        primary = APPROACH_COLLISION
    else:
        primary = OTHER_COLLISION
    return {
        "collision_class_primary": primary,
        "is_final_approach": final_approach,
        "is_post_switch": post_switch,
        "is_planned_corner": corner,
        "is_planned_straight": straight,
        "is_off_path": off_path,
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
        name: sum(
            bool(row.get(name, row.get("is_corner", False) if name == "is_planned_corner"
                       else row.get("is_straight_corridor", False) if name == "is_planned_straight"
                       else False))
            for row in records
        )
        for name in (
            "is_final_approach", "is_post_switch", "is_planned_corner",
            "is_planned_straight", "is_off_path", "is_approach",
        )
    }
    overlap_counts["is_corner"] = overlap_counts["is_planned_corner"]
    overlap_counts["is_straight_corridor"] = overlap_counts["is_planned_straight"]
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
