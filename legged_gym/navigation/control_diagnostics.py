"""Deterministic fixed scenarios for non-training controller diagnostics."""

import numpy as np

from .bfs_planner import plan_cells


C2_INITIAL_SPEEDS_MPS = (0.0, 0.2, 0.4, 0.6)
_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _free(layout, cell):
    x, y = cell
    return 0 <= x < layout.shape[0] and 0 <= y < layout.shape[1] and layout[cell] == 0


def _cell_add(cell, direction):
    return (int(cell[0] + direction[0]), int(cell[1] + direction[1]))


def select_straight_case(layout, start_cell, minimum_edges=3, maximum_edges=None):
    grid = np.asarray(layout)
    start = tuple(int(value) for value in start_cell)
    for direction in _DIRECTIONS:
        cells = [start]
        while _free(grid, _cell_add(cells[-1], direction)):
            cells.append(_cell_add(cells[-1], direction))
        if len(cells) - 1 >= int(minimum_edges):
            if maximum_edges is not None:
                cells = cells[: int(maximum_edges) + 1]
            return {"kind": "C1", "cells": tuple(cells)}
    raise ValueError("no straight corridor satisfies minimum_edges")


def select_corner_case(layout, start_cell):
    grid = np.asarray(layout)
    start = tuple(int(value) for value in start_cell)
    for first in _DIRECTIONS:
        corner = _cell_add(start, first)
        if not _free(grid, corner):
            continue
        for second in _DIRECTIONS:
            if second == first or second == (-first[0], -first[1]):
                continue
            goal = _cell_add(corner, second)
            if _free(grid, goal):
                return {"kind": "C2", "cells": (start, corner, goal)}
    raise ValueError("no single 90-degree corner satisfies the start cell")


def select_detour_case(layout, start_cell):
    grid = np.asarray(layout)
    start = tuple(int(value) for value in start_cell)
    if not _free(grid, start):
        raise ValueError("detour start cell must be free")
    candidates = []
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            goal = (x, y)
            if not _free(grid, goal) or goal == start:
                continue
            manhattan = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
            try:
                path = plan_cells(grid, start, goal)
            except ValueError:
                continue
            if len(path) - 1 > manhattan and len(path) - 1 >= 4:
                candidates.append((len(path), goal, path))
    if not candidates:
        raise ValueError("no reachable wall-detour case exists")
    _, goal, path = sorted(candidates, key=lambda item: (item[1][0], item[1][1], item[0]))[0]
    return {"kind": "C3", "start_cell": start, "goal_cell": goal, "cells": tuple(path)}
