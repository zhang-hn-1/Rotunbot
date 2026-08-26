"""Closed-loop Oracle waypoint selection for a frozen local skill."""

from dataclasses import dataclass

import numpy as np

from .bfs_planner import cell_center, plan_cells, select_next_waypoint, world_to_cell
from .local_goal_adapter import local_to_world, world_to_local


NAVIGATE = "NAVIGATE"
FINAL_APPROACH = "FINAL_APPROACH"
TURN_AWARE_BEARING_THRESHOLD_DEG = 45.0
TURN_AWARE_SPEED_MPS = 0.30
LOCAL_WAYPOINT_DISTANCE_M = 0.35


def waypoint_reached(
    distance_m,
    speed_mps,
    delta_bearing_deg=0.0,
    turn_aware=False,
):
    """Return whether a waypoint may be switched under the selected policy."""
    if float(distance_m) > LOCAL_WAYPOINT_DISTANCE_M:
        return False
    return not (
        bool(turn_aware)
        and abs(float(delta_bearing_deg)) >= TURN_AWARE_BEARING_THRESHOLD_DEG
        and float(speed_mps) > TURN_AWARE_SPEED_MPS
    )


def _signed_turn_degrees(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    return float(np.degrees(np.arctan2(
        first[0] * second[1] - first[1] * second[0],
        first.dot(second),
    )))


def _xy(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain two finite values")
    return result


@dataclass(frozen=True)
class LocalWaypoint:
    cell: tuple
    local_goal_xy: tuple
    filtered_local_goal_xy: tuple
    world_goal_xy: tuple
    temporary_world_goal_xy: tuple
    delta_bearing_deg: float = 0.0
    is_final_approach: bool = False


class OracleEpisodePlanner:
    """Plan from the current measured robot pose on the ground-truth map."""

    def __init__(self, occupancy, maze_shape, cell_size, reachability=None):
        self.occupancy = np.asarray(occupancy)
        self.maze_shape = tuple(np.asarray(maze_shape, dtype=np.int64).tolist())
        self.cell_size = float(cell_size)
        self.reachability = reachability
        self.phase = NAVIGATE
        self.final_approach_entry_count = 0
        self.final_approach_escape_count = 0
        self._final_approach_escaped = False

    def _final_waypoint(self, robot, robot_yaw, global_goal, goal_cell):
        local_goal = world_to_local(robot, robot_yaw, global_goal)
        return LocalWaypoint(
            cell=tuple(goal_cell),
            local_goal_xy=tuple(local_goal.tolist()),
            # FINAL_APPROACH is intentionally never reachability-filtered.
            filtered_local_goal_xy=tuple(local_goal.tolist()),
            world_goal_xy=tuple(global_goal.tolist()),
            temporary_world_goal_xy=tuple(global_goal.tolist()),
            delta_bearing_deg=0.0,
            is_final_approach=True,
        )

    def next_local_waypoint(self, robot_xy, robot_yaw, global_goal_xy):
        robot = _xy(robot_xy, "robot_xy")
        global_goal = _xy(global_goal_xy, "global_goal_xy")
        start_cell = world_to_cell(robot, self.maze_shape, self.cell_size)
        goal_cell = world_to_cell(global_goal, self.maze_shape, self.cell_size)
        if self.phase == FINAL_APPROACH:
            if start_cell != goal_cell and not self._final_approach_escaped:
                self.final_approach_escape_count += 1
                self._final_approach_escaped = True
            return self._final_waypoint(robot, robot_yaw, global_goal, goal_cell)
        if start_cell == goal_cell:
            self.phase = FINAL_APPROACH
            self.final_approach_entry_count += 1
            return self._final_waypoint(robot, robot_yaw, global_goal, goal_cell)
        path = plan_cells(self.occupancy, start_cell, goal_cell)
        waypoint_cell = select_next_waypoint(path, 0)
        world_waypoint = cell_center(waypoint_cell, self.maze_shape, self.cell_size)
        current_segment = np.asarray(path[1], dtype=np.float64) - np.asarray(path[0], dtype=np.float64)
        next_segment = (
            np.asarray(path[2], dtype=np.float64) - np.asarray(path[1], dtype=np.float64)
            if len(path) > 2 else current_segment
        )
        delta_bearing_deg = _signed_turn_degrees(current_segment, next_segment)
        local_goal = world_to_local(robot, robot_yaw, world_waypoint)
        filtered = (
            self.reachability.filter(local_goal)
            if self.reachability is not None
            else local_goal
        )
        temporary = local_to_world(robot, robot_yaw, filtered)
        return LocalWaypoint(
            cell=tuple(waypoint_cell),
            local_goal_xy=tuple(local_goal.tolist()),
            filtered_local_goal_xy=tuple(filtered.tolist()),
            world_goal_xy=tuple(world_waypoint.tolist()),
            temporary_world_goal_xy=tuple(temporary.tolist()),
            delta_bearing_deg=delta_bearing_deg,
            is_final_approach=False,
        )
