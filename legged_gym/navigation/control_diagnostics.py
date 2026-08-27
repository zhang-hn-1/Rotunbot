"""Deterministic fixed scenarios for non-training controller diagnostics."""

import numpy as np

from .bfs_planner import plan_cells


C2_INITIAL_SPEEDS_MPS = (0.0, 0.2, 0.4, 0.6)
_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def synthetic_diagnostic_layout(shape=(15, 15), center_cell=(7, 7), center_radius=2):
    """Build a bounded wall-only layout for an isolated control diagnostic.

    The free center is retained so the maze environment can initialize its
    normal goal list.  The two diagnostic paths are intentionally separate
    from that cleared area and remain within the P2P +/-10m bounds.
    """
    grid = np.ones(tuple(int(value) for value in shape), dtype=np.int8)
    center_x, center_y = (int(center_cell[0]), int(center_cell[1]))
    grid[
        center_x - int(center_radius): center_x + int(center_radius) + 1,
        center_y - int(center_radius): center_y + int(center_radius) + 1,
    ] = 0
    # C1: four free cells with continuous walls at x=2 and x=4.
    c1 = ((3, 3), (3, 4), (3, 5), (3, 6))
    grid[tuple(np.asarray(c1).T)] = 0
    # C2: east then south, with its inner and outer wall continuations.
    c2 = ((10, 3), (10, 4), (11, 4))
    grid[tuple(np.asarray(c2).T)] = 0
    return grid


def _free(layout, cell):
    x, y = cell
    return 0 <= x < layout.shape[0] and 0 <= y < layout.shape[1] and layout[cell] == 0


def _cell_add(cell, direction):
    return (int(cell[0] + direction[0]), int(cell[1] + direction[1]))


def _center_excluded(cell, center_cell, radius):
    if center_cell is None or radius is None:
        return False
    return max(
        abs(int(cell[0]) - int(center_cell[0])),
        abs(int(cell[1]) - int(center_cell[1])),
    ) <= int(radius)


def _boundary_excluded(cell, shape, margin):
    margin = int(margin)
    return (
        margin > 0
        and (
            int(cell[0]) <= margin
            or int(cell[1]) <= margin
            or int(cell[0]) >= int(shape[0]) - 1 - margin
            or int(cell[1]) >= int(shape[1]) - 1 - margin
        )
    )


def _wall(grid, cell):
    x, y = cell
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[cell] == 1


def _straight_topology(grid, cells):
    direction = (cells[1][0] - cells[0][0], cells[1][1] - cells[0][1])
    normal = (-direction[1], direction[0])
    side_pairs = []
    for cell in cells:
        left = _cell_add(cell, normal)
        right = _cell_add(cell, (-normal[0], -normal[1]))
        if not (_wall(grid, left) and _wall(grid, right)):
            return None
        side_pairs.append((left, right))
    return {
        "wall_cells": tuple(cell for pair in side_pairs for cell in pair),
        "side_wall_pairs": tuple(side_pairs),
    }


def _corner_topology(grid, cells):
    first = (cells[1][0] - cells[0][0], cells[1][1] - cells[0][1])
    second = (cells[2][0] - cells[1][0], cells[2][1] - cells[1][1])
    if first == second or first == (-second[0], -second[1]):
        return None
    if first[0] * second[0] + first[1] * second[1] != 0:
        return None

    # Require a wall on each side of both legs, then explicitly validate the
    # inner diagonal and the two outer wall continuations at the corner.
    def endpoint_sides(cell, direction):
        normal = (-direction[1], direction[0])
        sides = (_cell_add(cell, normal), _cell_add(cell, (-normal[0], -normal[1])))
        return sides if all(_wall(grid, side) for side in sides) else None

    start_sides = endpoint_sides(cells[0], first)
    goal_sides = endpoint_sides(cells[2], second)
    if start_sides is None or goal_sides is None:
        return None
    corner = cells[1]
    inner = _cell_add(corner, (second[0] - first[0], second[1] - first[1]))
    inner = (inner[0], inner[1])
    # For the tested orientation this is P1 + d2 - d1.  The opposite turn
    # orientation is handled by the signed vector above; it is the quadrant
    # between the incoming and outgoing legs.
    if not _wall(grid, inner):
        return None
    left1 = (-first[1], first[0])
    left2 = (-second[1], second[0])
    outer = (
        _cell_add(cells[0], left1),
        _cell_add(corner, left1),
        _cell_add(corner, left2),
        _cell_add(cells[2], left2),
    )
    if not all(_wall(grid, cell) for cell in outer):
        return None
    return {
        "inner_wall_cells": (inner,),
        "outer_wall_cells": tuple(outer),
        "wall_cells": tuple(dict.fromkeys(start_sides + goal_sides + (inner,) + outer)),
    }


def _candidate_starts(grid, start_cell):
    if start_cell is not None:
        yield tuple(int(value) for value in start_cell)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            cell = (x, y)
            if start_cell is None or cell != tuple(start_cell):
                yield cell


def select_real_straight_case(
    layout, start_cell=None, *, center_cell=None, center_clearance_radius=0,
    minimum_edges=3, maximum_edges=None, boundary_margin=0
):
    """Select a non-cleared corridor whose two sides are continuous walls."""
    grid = np.asarray(layout)
    for start in _candidate_starts(grid, start_cell):
        if (
            not _free(grid, start)
            or _center_excluded(start, center_cell, center_clearance_radius)
            or _boundary_excluded(start, grid.shape, boundary_margin)
        ):
            continue
        for direction in _DIRECTIONS:
            cells = [start]
            while _free(grid, _cell_add(cells[-1], direction)):
                candidate = _cell_add(cells[-1], direction)
                if (
                    _center_excluded(candidate, center_cell, center_clearance_radius)
                    or _boundary_excluded(candidate, grid.shape, boundary_margin)
                ):
                    break
                cells.append(candidate)
            if len(cells) - 1 < int(minimum_edges):
                continue
            if maximum_edges is not None:
                cells = cells[: int(maximum_edges) + 1]
            topology = _straight_topology(grid, cells)
            if topology is not None:
                return {
                    "kind": "C1",
                    "cells": tuple(cells),
                    **topology,
                    "topology_validated": True,
                    "center_clearance_radius": int(center_clearance_radius),
                }
    raise ValueError("no real straight corridor satisfies wall and center-clearance constraints")


def select_real_corner_case(
    layout, start_cell=None, *, center_cell=None, center_clearance_radius=0,
    boundary_margin=0
):
    """Select a 90-degree path with verified inner/outer wall topology."""
    grid = np.asarray(layout)
    for start in _candidate_starts(grid, start_cell):
        if (
            not _free(grid, start)
            or _center_excluded(start, center_cell, center_clearance_radius)
            or _boundary_excluded(start, grid.shape, boundary_margin)
        ):
            continue
        for first in _DIRECTIONS:
            corner = _cell_add(start, first)
            if (
                not _free(grid, corner)
                or _center_excluded(corner, center_cell, center_clearance_radius)
                or _boundary_excluded(corner, grid.shape, boundary_margin)
            ):
                continue
            for second in _DIRECTIONS:
                goal = _cell_add(corner, second)
                cells = (start, corner, goal)
                if (
                    not _free(grid, goal)
                    or _center_excluded(goal, center_cell, center_clearance_radius)
                    or _boundary_excluded(goal, grid.shape, boundary_margin)
                ):
                    continue
                topology = _corner_topology(grid, cells)
                if topology is not None:
                    return {
                        "kind": "C2",
                        "cells": cells,
                        **topology,
                        "topology_validated": True,
                        "center_clearance_radius": int(center_clearance_radius),
                    }
    raise ValueError("no real 90-degree corner satisfies wall and center-clearance constraints")


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
