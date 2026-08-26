"""Oracle four-neighbor BFS and maze-cell geometry."""

from collections import deque

import numpy as np


def _cell(value, name):
    array = np.asarray(value)
    if array.shape != (2,):
        raise ValueError(f"{name} must contain exactly two indices")
    if not np.all(np.isfinite(array.astype(np.float64))):
        raise ValueError(f"{name} must contain finite values")
    result = (int(array[0]), int(array[1]))
    if result != (array[0], array[1]):
        raise ValueError(f"{name} must contain integer indices")
    return result


def _grid(occupancy):
    grid = np.asarray(occupancy)
    if grid.ndim != 2:
        raise ValueError("occupancy must be a two-dimensional array")
    return grid


def _check_free(grid, cell, name):
    x, y = cell
    if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
        raise ValueError(f"{name} lies outside the occupancy grid: {cell}")
    if grid[cell] != 0:
        raise ValueError(f"{name} is a wall cell: {cell}")


def plan_cells(occupancy, start_cell, goal_cell):
    """Return a shortest four-neighbor free-cell path, including endpoints."""
    grid = _grid(occupancy)
    start = _cell(start_cell, "start_cell")
    goal = _cell(goal_cell, "goal_cell")
    _check_free(grid, start, "start_cell")
    _check_free(grid, goal, "goal_cell")

    queue = deque([start])
    predecessor = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        x, y = current
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            nx, ny = neighbor
            if (
                0 <= nx < grid.shape[0]
                and 0 <= ny < grid.shape[1]
                and grid[neighbor] == 0
                and neighbor not in predecessor
            ):
                predecessor[neighbor] = current
                queue.append(neighbor)

    if goal not in predecessor:
        raise ValueError(f"goal_cell is unreachable from start_cell: {goal}")

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = predecessor[current]
    path.reverse()
    return tuple(path)


def cell_center(cell, maze_shape, cell_size):
    """Convert an occupancy-grid cell index to map-local XY coordinates."""
    cell = _cell(cell, "cell")
    shape = np.asarray(maze_shape, dtype=np.float64)
    if shape.shape != (2,) or np.any(shape <= 0):
        raise ValueError("maze_shape must contain two positive dimensions")
    size = float(cell_size)
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("cell_size must be positive and finite")
    return (np.asarray(cell, dtype=np.float64) - shape / 2.0 + 0.5) * size


def world_to_cell(world_xy, maze_shape, cell_size):
    """Map local XY coordinates to the containing occupancy-grid cell."""
    world = np.asarray(world_xy, dtype=np.float64)
    if world.shape != (2,) or not np.all(np.isfinite(world)):
        raise ValueError("world_xy must contain two finite values")
    shape = np.asarray(maze_shape, dtype=np.float64)
    if shape.shape != (2,) or np.any(shape <= 0):
        raise ValueError("maze_shape must contain two positive dimensions")
    size = float(cell_size)
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("cell_size must be positive and finite")
    index = np.floor(world / size + shape / 2.0).astype(np.int64)
    return int(index[0]), int(index[1])


def select_next_waypoint(path, current_index=0):
    """Select the next cell center target, or the current endpoint if done."""
    if not path:
        raise ValueError("path must contain at least one cell")
    index = int(current_index)
    if index < 0 or index >= len(path):
        raise IndexError(f"current_index out of range: {index}")
    return tuple(path[min(index + 1, len(path) - 1)])
