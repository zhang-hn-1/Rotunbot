"""Map-only Oracle local-subgoal planner for Phase-1 experiments."""

from collections import deque

import numpy as np

from legged_gym.maps import FREE, cell_centers_to_world


class OracleLocalSubgoalPlanner:
    """Convert a shortest occupancy-grid path into a local waypoint."""

    def __init__(self, maze, cell_size, lookahead_cells=1):
        self.maze = np.asarray(maze, dtype=np.uint8)
        if self.maze.ndim != 2:
            raise ValueError("maze must be a 2-D occupancy grid")
        self.cell_size = float(cell_size)
        if self.cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        self.lookahead_cells = int(lookahead_cells)
        if self.lookahead_cells < 1:
            raise ValueError("lookahead_cells must be >= 1")
        self._path_cache = {}

    def world_to_cell(self, position_xy):
        position_xy = np.asarray(position_xy, dtype=np.float64)
        if position_xy.shape != (2,):
            raise ValueError("position_xy must have shape [2]")
        maze_shape = np.asarray(self.maze.shape, dtype=np.float64)
        cell = np.rint(
            position_xy / self.cell_size + maze_shape / 2.0 - 0.5
        ).astype(np.int64)
        cell = np.clip(cell, [0, 0], np.asarray(self.maze.shape) - 1)
        return tuple(int(value) for value in cell)

    def cell_to_world(self, cell):
        cell = np.asarray(cell, dtype=np.int64).reshape(1, 2)
        return cell_centers_to_world(cell, self.maze.shape, self.cell_size)[0]

    def _nearest_free_cell(self, cell):
        cell = tuple(int(value) for value in cell)
        if self.maze[cell] == FREE:
            return cell
        queue = deque([cell])
        visited = {cell}
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (x + dx, y + dy)
                if neighbor in visited:
                    continue
                if not (0 <= neighbor[0] < self.maze.shape[0]):
                    continue
                if not (0 <= neighbor[1] < self.maze.shape[1]):
                    continue
                if self.maze[neighbor] == FREE:
                    return neighbor
                visited.add(neighbor)
                queue.append(neighbor)
        raise ValueError("maze contains no free cell near requested position")

    def _shortest_path(self, start, goal):
        key = (start, goal)
        if key in self._path_cache:
            return self._path_cache[key]
        queue = deque([start])
        parent = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            x, y = current
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (x + dx, y + dy)
                if neighbor in parent:
                    continue
                if not (0 <= neighbor[0] < self.maze.shape[0]):
                    continue
                if not (0 <= neighbor[1] < self.maze.shape[1]):
                    continue
                if self.maze[neighbor] != FREE:
                    continue
                parent[neighbor] = current
                queue.append(neighbor)
        if goal not in parent:
            raise ValueError(f"no free path from {start} to {goal}")
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent[current]
        path = tuple(reversed(path))
        self._path_cache[key] = path
        return path

    def plan(self, position_xy, goal_xy):
        """Return ``(map_local_waypoint_xy, path_cells)``."""
        start = self._nearest_free_cell(self.world_to_cell(position_xy))
        goal = self._nearest_free_cell(self.world_to_cell(goal_xy))
        path = self._shortest_path(start, goal)
        index = min(self.lookahead_cells, len(path) - 1)
        return self.cell_to_world(path[index]), path
