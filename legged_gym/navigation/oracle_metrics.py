"""Pure aggregation helpers for the Oracle Maze evaluation protocol."""

import numpy as np

from .oracle_diagnostics import summarize_collision_diagnostics


FAILURE_REASONS = (
    "global_success",
    "collision",
    "timeout",
    "unstable",
    "out_of_bounds",
    "waypoint_failure",
    "planner_error",
    "goal_switch_error",
    "final_approach_escape",
)


def maze_spl(success, shortest_path_length_m, actual_path_length_m):
    if not success or actual_path_length_m <= 0.0:
        return 0.0
    return float(shortest_path_length_m / max(actual_path_length_m, shortest_path_length_m))


def summarize_oracle_results(results, protocol="oracle_maze_120s"):
    rows = tuple(results)
    count = len(rows)
    if count == 0:
        raise ValueError("at least one Oracle result is required")
    waypoint_attempts = sum(int(row.get("waypoint_count", 0)) for row in rows)
    waypoint_reached = sum(int(row.get("local_waypoint_reached_count", 0)) for row in rows)
    reason_counts = {reason: sum(row.get("reason") == reason for row in rows) for reason in FAILURE_REASONS}
    final_approach_entry_count = sum(bool(row.get("final_approach_entered", False)) for row in rows)
    final_approach_success_count = sum(bool(row.get("final_approach_success", False)) for row in rows)
    final_approach_timeout_count = sum(bool(row.get("final_approach_timeout", False)) for row in rows)
    final_approach_escape_count = sum(bool(row.get("final_approach_escape", False)) for row in rows)
    collision_records = [row["collision_diagnostic"] for row in rows if row.get("collision_diagnostic")]

    def mean(field):
        return float(np.mean([float(row.get(field, 0.0)) for row in rows]))

    global_success_rate = float(reason_counts["global_success"] / count)
    summary = {
        "protocol": protocol,
        "low_level_protocol": "uniform_4150_original_60s_p2p",
        "episodes": count,
        "global_success_rate": global_success_rate,
        "global_sr": global_success_rate,
        "collision_rate": float(reason_counts["collision"] / count),
        "timeout_rate": float(reason_counts["timeout"] / count),
        "waypoint_failure_rate": float(reason_counts["waypoint_failure"] / count),
        "unstable_rate": float(reason_counts["unstable"] / count),
        "out_of_bounds_rate": float(reason_counts["out_of_bounds"] / count),
        "local_waypoint_reach_rate": float(waypoint_reached / waypoint_attempts) if waypoint_attempts else 0.0,
        "actual_path_length_m": mean("actual_path_length_m"),
        "bfs_shortest_path_length_m": mean("bfs_shortest_path_length_m"),
        "maze_spl": mean("maze_spl"),
        "completion_time_s": mean("completion_time_s"),
        "waypoint_count": mean("waypoint_count"),
        "final_approach_entry_count": final_approach_entry_count,
        "final_approach_success_count": final_approach_success_count,
        "final_approach_timeout_count": final_approach_timeout_count,
        "final_approach_escape_count": final_approach_escape_count,
        "failure_reason_counts": reason_counts,
        "planner_error_count": reason_counts["planner_error"],
        "goal_switch_error_count": reason_counts["goal_switch_error"],
        "coordinate_error_count": sum(int(row.get("coordinate_error_count", 0)) for row in rows),
        "state_continuity_violation_count": sum(int(row.get("state_continuity_violation_count", 0)) for row in rows),
        "checkpoint_control_configuration_error_count": sum(
            int(row.get("checkpoint_control_configuration_error_count", 0)) for row in rows
        ),
        "episodes_detail": list(rows),
    }
    summary.update(summarize_collision_diagnostics(collision_records, episode_count=count))
    return summary
