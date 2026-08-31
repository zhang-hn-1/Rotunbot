"""Pure helpers for deterministic V1 distance-action diagnostics."""

import math


SCAN_FIELDS = (
    "distance_m",
    "normalized_goal_x",
    "raw_policy_mean_a_v",
    "raw_policy_mean_a_w",
    "mapped_v_cmd",
    "mapped_w_cmd",
    "projected_v_cmd",
    "projected_w_cmd",
)


def distance_grid(start, stop, step):
    """Return an inclusive, rounded distance grid without floating drift."""
    if step <= 0 or stop < start:
        raise ValueError("distance grid requires stop >= start and step > 0")
    count = int(math.floor((float(stop) - float(start)) / float(step) + 1.0e-9))
    values = tuple(round(float(start) + index * float(step), 2) for index in range(count + 1))
    if not values or values[-1] != round(float(stop), 2):
        raise ValueError("distance range must be an integer multiple of step")
    return values


def scan_row(
    distance_m,
    normalized_goal_x,
    raw_action,
    mapped_command,
    projected_command,
):
    """Build the stable CSV row contract for one deterministic probe."""
    return {
        "distance_m": float(distance_m),
        "normalized_goal_x": float(normalized_goal_x),
        "raw_policy_mean_a_v": float(raw_action[0]),
        "raw_policy_mean_a_w": float(raw_action[1]),
        "mapped_v_cmd": float(mapped_command[0]),
        "mapped_w_cmd": float(mapped_command[1]),
        "projected_v_cmd": float(projected_command[0]),
        "projected_w_cmd": float(projected_command[1]),
    }


def causal_pair(physical_goal_distance, visible_goal_distance, raw_action, mapped_command):
    """Build one row for the observation-only clipped-goal comparison."""
    return {
        "physical_goal_distance_m": float(physical_goal_distance),
        "visible_goal_distance_m": float(visible_goal_distance),
        "raw_policy_mean_a_v": float(raw_action[0]),
        "raw_policy_mean_a_w": float(raw_action[1]),
        "mapped_v_cmd": float(mapped_command[0]),
        "mapped_w_cmd": float(mapped_command[1]),
    }


def first_zero_crossing(points):
    """Interpolate the first x where a sampled y series crosses zero."""
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if y0 == 0.0:
            return float(x0)
        if y0 * y1 < 0.0:
            fraction = -float(y0) / (float(y1) - float(y0))
            return float(x0) + fraction * (float(x1) - float(x0))
    if points and points[-1][1] == 0.0:
        return float(points[-1][0])
    return None
