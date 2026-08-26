"""Closed-loop Oracle waypoint selection for a frozen local skill."""

from dataclasses import dataclass

import numpy as np

from .bfs_planner import cell_center, plan_cells, select_next_waypoint, world_to_cell
from .local_goal_adapter import local_to_world, world_to_local


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


class OracleEpisodePlanner:
    """Plan from the current measured robot pose on the ground-truth map."""

    def __init__(self, occupancy, maze_shape, cell_size, reachability=None):
        self.occupancy = np.asarray(occupancy)
        self.maze_shape = tuple(np.asarray(maze_shape, dtype=np.int64).tolist())
        self.cell_size = float(cell_size)
        self.reachability = reachability

    def next_local_waypoint(self, robot_xy, robot_yaw, global_goal_xy):
        robot = _xy(robot_xy, "robot_xy")
        global_goal = _xy(global_goal_xy, "global_goal_xy")
        start_cell = world_to_cell(robot, self.maze_shape, self.cell_size)
        goal_cell = world_to_cell(global_goal, self.maze_shape, self.cell_size)
        path = plan_cells(self.occupancy, start_cell, goal_cell)
        waypoint_cell = select_next_waypoint(path, 0)
        world_waypoint = cell_center(waypoint_cell, self.maze_shape, self.cell_size)
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
        )
