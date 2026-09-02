"""Deterministic random-obstacle scenario generation for Phase D (D0/D1).

Physics uses rotated boxes (OBB). Planning uses a conservative axis-aligned
envelope inflated by the robot radius plus a safety margin; the invariant is
``planning_geometry >= physics_geometry``.
"""

from dataclasses import dataclass, field
import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

from .v62_turn_reachability import (
    point_aabb_raw_distance,
    point_clearance,
    polyline_clearance,
)

# Source of truth: resources/robots/Rotunbot/urdf/Rotunbot.urdf link1
# collision sphere radius; corroborated by direct-velocity config
# maze.robot_collision_radius.
DEFAULT_ROBOT_RADIUS_M = 0.4
DEFAULT_SAFETY_MARGIN_M = 0.05

ARENA_MIN = 0.0
ARENA_MAX = 6.0
BOUNDARY_HALF_WIDTH_M = 0.05

TRAIN_SEED_RANGE = (0, 9999)
VALIDATION_SEED_RANGE = (10000, 11999)
TEST_SEED_RANGE = (20000, 21999)
OOD_SEED_RANGE = (30000, 31999)

DEFAULT_MIN_GOAL_DISTANCE_M = 2.0
DEFAULT_MIN_OBSTACLE_GAP_M = 0.10
DEFAULT_GRID_CELL_M = 0.10
DEFAULT_MAX_ATTEMPTS = 64


@dataclass(frozen=True)
class RandomObstacleSplitConfig:
    """Single source of truth for train/validation/test/OOD seed ranges."""

    train_seed_range: Tuple[int, int] = TRAIN_SEED_RANGE
    validation_seed_range: Tuple[int, int] = VALIDATION_SEED_RANGE
    test_seed_range: Tuple[int, int] = TEST_SEED_RANGE
    ood_seed_range: Tuple[int, int] = OOD_SEED_RANGE

    def __post_init__(self):
        ranges = [
            ("train", self.train_seed_range),
            ("validation", self.validation_seed_range),
            ("test", self.test_seed_range),
            ("ood", self.ood_seed_range),
        ]
        for name, (low, high) in ranges:
            if not (isinstance(low, int) and isinstance(high, int) and 0 <= low <= high):
                raise ValueError("%s seed range must be a valid [low, high] int range" % name)
        for index, (name, (low, high)) in enumerate(ranges):
            for other_name, (other_low, other_high) in ranges[index + 1:]:
                if not (high < other_low or other_high < low):
                    raise ValueError("%s and %s seed ranges overlap" % (name, other_name))

    def in_split(self, map_seed, split_name):
        low, high = {
            "train": self.train_seed_range,
            "validation": self.validation_seed_range,
            "test": self.test_seed_range,
            "ood": self.ood_seed_range,
        }[split_name]
        return int(low) <= int(map_seed) <= int(high)


@dataclass(frozen=True)
class ObstacleBox:
    center_xy: Tuple[float, float]
    size_xy: Tuple[float, float]  # (length, width) along local x/y axes
    yaw_rad: float

    def corners(self):
        cx, cy = self.center_xy
        length, width = self.size_xy
        cos = math.cos(self.yaw_rad)
        sin = math.sin(self.yaw_rad)
        local = [(-length / 2.0, -width / 2.0), (length / 2.0, -width / 2.0),
                 (length / 2.0, width / 2.0), (-length / 2.0, width / 2.0)]
        return [
            (cx + cos * lx - sin * ly, cy + sin * lx + cos * ly)
            for lx, ly in local
        ]

    def to_aabb(self):
        """Conservative axis-aligned envelope of this rotated box."""
        xs = [point[0] for point in self.corners()]
        ys = [point[1] for point in self.corners()]
        center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
        half = ((max(xs) - min(xs)) / 2.0, (max(ys) - min(ys)) / 2.0)
        return (center, half)


@dataclass(frozen=True)
class RandomObstacleScenario:
    map_seed: int
    attempt_index: int
    obstacle_count: int
    obstacles: Tuple[ObstacleBox, ...]
    spawn_xy: Tuple[float, float]
    initial_yaw_rad: float
    goal_xy: Tuple[float, float]
    robot_radius_m: float
    safety_margin_m: float
    oracle_path: Tuple[Tuple[float, float], ...] = ()
    oracle_path_length_m: float = float("nan")
    bounds_xy: Tuple[float, float, float, float] = (ARENA_MIN, ARENA_MIN, ARENA_MAX, ARENA_MAX)
    boundary_clearance_m: float = 1.2

    def planning_aabbs(self):
        """Conservative inflated AABBs: arena boundary + obstacle envelopes."""
        radius = float(self.robot_radius_m)
        margin = float(self.safety_margin_m)
        low_x, low_y, high_x, high_y = self.bounds_xy
        half = float(BOUNDARY_HALF_WIDTH_M)
        boundary = (
            (((low_x + high_x) / 2.0, low_y - half), ((high_x - low_x) / 2.0 + half, half)),
            (((low_x + high_x) / 2.0, high_y + half), ((high_x - low_x) / 2.0 + half, half)),
            ((low_x - half, (low_y + high_y) / 2.0), (half, (high_y - low_y) / 2.0 + half)),
            ((high_x + half, (low_y + high_y) / 2.0), (half, (high_y - low_y) / 2.0 + half)),
        )
        obstacles = [box.to_aabb() for box in self.obstacles]
        inflated = []
        for center, half_extent in list(boundary) + obstacles:
            inflated.append(
                (
                    (float(center[0]), float(center[1])),
                    (float(half_extent[0]) + radius + margin,
                     float(half_extent[1]) + radius + margin),
                )
            )
        return tuple(inflated)

    def raw_physics_aabbs(self):
        low_x, low_y, high_x, high_y = self.bounds_xy
        half = float(BOUNDARY_HALF_WIDTH_M)
        boundary = (
            (((low_x + high_x) / 2.0, low_y - half), ((high_x - low_x) / 2.0 + half, half)),
            (((low_x + high_x) / 2.0, high_y + half), ((high_x - low_x) / 2.0 + half, half)),
            ((low_x - half, (low_y + high_y) / 2.0), (half, (high_y - low_y) / 2.0 + half)),
            ((high_x + half, (low_y + high_y) / 2.0), (half, (high_y - low_y) / 2.0 + half)),
        )
        return boundary + tuple(box.to_aabb() for box in self.obstacles)

    def physics_obstacle_centers_and_extents(self):
        """Return (centers, half_extents) of the boundary boxes for sim/clearance."""
        centers = []
        extents = []
        low_x, low_y, high_x, high_y = self.bounds_xy
        half = float(BOUNDARY_HALF_WIDTH_M)
        for center, half_extent in (
            (((low_x + high_x) / 2.0, low_y - half), ((high_x - low_x) / 2.0 + half, half)),
            (((low_x + high_x) / 2.0, high_y + half), ((high_x - low_x) / 2.0 + half, half)),
            ((low_x - half, (low_y + high_y) / 2.0), (half, (high_y - low_y) / 2.0 + half)),
            ((high_x + half, (low_y + high_y) / 2.0), (half, (high_y - low_y) / 2.0 + half)),
        ):
            centers.append(center)
            extents.append(half_extent)
        for box in self.obstacles:
            center, half_extent = box.to_aabb()
            centers.append(center)
            extents.append(half_extent)
        return tuple(centers), tuple(extents)

    def physics_rotated_boxes(self):
        """Return OBBs (as parameterised boxes) for creating real rotated actors."""
        return self.obstacles

    def point_physics_clearance(self, point_xy):
        """Raw Euclidean distance from a point to the nearest real boundary/box."""
        return min(
            point_aabb_raw_distance(point_xy, center, half_extent)
            for center, half_extent in self.raw_physics_aabbs()
        )

    def point_planning_clearance(self, point_xy):
        """Distance to the nearest inflated planning AABB.

        The planning AABBs already include robot_radius + safety_margin, so
        zero marks the boundary of the robot-centre-feasible region.
        """
        return point_clearance(
            point_xy,
            tuple((center, half_extent) for center, half_extent in self.planning_aabbs()),
            robot_radius_m=0.0,
        )


def scenario_to_metadata(scenario):
    return {
        "map_seed": int(scenario.map_seed),
        "attempt_index": int(scenario.attempt_index),
        "obstacle_count": int(scenario.obstacle_count),
        "obstacles": [
            {
                "center_xy": list(box.center_xy),
                "size_xy": list(box.size_xy),
                "yaw_rad": float(box.yaw_rad),
            }
            for box in scenario.obstacles
        ],
        "spawn_xy": list(scenario.spawn_xy),
        "initial_yaw_rad": float(scenario.initial_yaw_rad),
        "goal_xy": list(scenario.goal_xy),
        "robot_radius_m": float(scenario.robot_radius_m),
        "safety_margin_m": float(scenario.safety_margin_m),
        "oracle_path": [list(point) for point in scenario.oracle_path],
        "oracle_path_length_m": float(scenario.oracle_path_length_m),
        "bounds_xy": list(scenario.bounds_xy),
        "boundary_clearance_m": float(scenario.boundary_clearance_m),
    }


def scenario_from_metadata(metadata):
    return RandomObstacleScenario(
        map_seed=int(metadata["map_seed"]),
        attempt_index=int(metadata["attempt_index"]),
        obstacle_count=int(metadata["obstacle_count"]),
        obstacles=tuple(
            ObstacleBox(tuple(box["center_xy"]), tuple(box["size_xy"]), float(box["yaw_rad"]))
            for box in metadata["obstacles"]
        ),
        spawn_xy=tuple(metadata["spawn_xy"]),
        initial_yaw_rad=float(metadata["initial_yaw_rad"]),
        goal_xy=tuple(metadata["goal_xy"]),
        robot_radius_m=float(metadata["robot_radius_m"]),
        safety_margin_m=float(metadata["safety_margin_m"]),
        oracle_path=tuple(tuple(point) for point in metadata.get("oracle_path", ())),
        oracle_path_length_m=float(metadata.get("oracle_path_length_m", float("nan"))),
        bounds_xy=tuple(metadata.get("bounds_xy", (ARENA_MIN, ARENA_MIN, ARENA_MAX, ARENA_MAX))),
        boundary_clearance_m=float(metadata.get("boundary_clearance_m", 1.2)),
    )


def build_occupancy_grid(scenario, cell_size_m=DEFAULT_GRID_CELL_M):
    """Grid with 1 = blocked by inflated obstacle/boundary, 0 = free."""
    cell = float(cell_size_m)
    if cell <= 0.0:
        raise ValueError("cell_size_m must be positive")
    low_x, low_y, high_x, high_y = scenario.bounds_xy
    width = int(math.ceil((high_x - low_x) / cell))
    height = int(math.ceil((high_y - low_y) / cell))
    grid = np.zeros((height, width), dtype=np.int8)
    inflated = scenario.planning_aabbs()
    for gy in range(height):
        for gx in range(width):
            x = low_x + (gx + 0.5) * cell
            y = low_y + (gy + 0.5) * cell
            # Blocked if the whole robot-centred footprint is unsafe: compare
            # the cell centre against each inflated AABB plus half a cell.
            blocked = False
            for center, half_extent in inflated:
                if point_aabb_raw_distance((x, y), center, half_extent) <= math.sqrt(2.0) * cell / 2.0:
                    blocked = True
                    break
            grid[gy, gx] = 1 if blocked else 0
    return grid, cell


def dijkstra_8connected(grid, start_cell, goal_cell, cell_size_m):
    """8-connected shortest path with orthogonal cost 1 and diagonal sqrt(2)."""
    height, width = grid.shape
    sy, sx = start_cell
    gy, gx = goal_cell
    if not (0 <= sy < height and 0 <= sx < width):
        raise ValueError("start cell out of bounds")
    if not (0 <= gy < height and 0 <= gx < width):
        raise ValueError("goal cell out of bounds")
    if grid[sy, sx] == 1 or grid[gy, gx] == 1:
        return None, float("inf")
    scale = float(cell_size_m)
    cost_ortho = 1.0
    cost_diag = math.sqrt(2.0)
    distance = np.full((height, width), float("inf"), dtype=np.float64)
    previous = np.full((height, width, 2), -1, dtype=np.int64)
    distance[sy, sx] = 0.0
    queue = [(0.0, sy, sx)]
    heapq.heapify(queue)
    directions = [
        (-1, -1, cost_diag), (-1, 0, cost_ortho), (-1, 1, cost_diag),
        (0, -1, cost_ortho), (0, 1, cost_ortho),
        (1, -1, cost_diag), (1, 0, cost_ortho), (1, 1, cost_diag),
    ]
    while queue:
        current_distance, cy, cx = heapq.heappop(queue)
        if current_distance > distance[cy, cx] + 1.0e-9:
            continue
        if (cy, cx) == (gy, gx):
            break
        for dy, dx, cost in directions:
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if grid[ny, nx] == 1:
                continue
            # Prevent corner cutting between diagonally touching blocked cells.
            if dx != 0 and dy != 0:
                if grid[cy + dy, cx] == 1 or grid[cy, cx + dx] == 1:
                    continue
            candidate = current_distance + cost
            if candidate + 1.0e-9 < distance[ny, nx]:
                distance[ny, nx] = candidate
                previous[ny, nx, 0] = cy
                previous[ny, nx, 1] = cx
                heapq.heappush(queue, (candidate, ny, nx))
    if distance[gy, gx] == float("inf"):
        return None, float("inf")
    cells = []
    cy, cx = gy, gx
    while (cy, cx) != (sy, sx):
        cells.append((cy, cx))
        ncy, ncx = previous[cy, cx]
        if ncy < 0:
            raise RuntimeError("shortest-path reconstruction failed")
        cy, cx = int(ncy), int(ncx)
    cells.append((sy, sx))
    cells.reverse()
    low_x, low_y, _, _ = (float(ARENA_MIN), float(ARENA_MIN), float(ARENA_MAX), float(ARENA_MAX))
    world = [((low_x + (cx + 0.5) * scale), (low_y + (cy + 0.5) * scale)) for cy, cx in cells]
    return world, float(distance[gy, gx] * scale)


def chaikin_smooth(points, iterations=2, ratio=0.25):
    """Chaikin corner cutting that preserves endpoints and stays near the path."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("points must be a finite [N, 2] array with N >= 2")
    if float(ratio) <= 0.0 or float(ratio) >= 0.5:
        raise ValueError("ratio must be in (0, 0.5)")
    current = points
    for _ in range(int(iterations)):
        output = [current[0]]
        for start, end in zip(current[:-1], current[1:]):
            output.append((1.0 - ratio) * start + ratio * end)
            output.append(ratio * start + (1.0 - ratio) * end)
        output.append(current[-1])
        current = np.asarray(output, dtype=np.float64)
    return current


def path_max_turn_degrees(points):
    """Maximum unsigned deflection angle (degrees) between consecutive segments."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return 0.0
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    lengths[lengths < 1.0e-9] = 1.0
    directions = vectors / lengths[:, None]
    maximum = 0.0
    for first, second in zip(directions[:-1], directions[1:]):
        cos = float(np.clip(np.dot(first, second), -1.0, 1.0))
        maximum = max(maximum, math.degrees(math.acos(cos)))
    return float(maximum)


def smooth_path_length_m(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return float("nan")
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _inflated_overlap(obstacles):
    """Return True if any two inflated obstacle AABBs overlap."""
    inflated = [box.to_aabb() for box in obstacles]
    for index, (center_a, half_a) in enumerate(inflated):
        for center_b, half_b in inflated[index + 1:]:
            overlap_x = abs(center_a[0] - center_b[0]) < (half_a[0] + half_b[0])
            overlap_y = abs(center_a[1] - center_b[1]) < (half_a[1] + half_b[1])
            if overlap_x and overlap_y:
                return True
    return False


def _point_inside_inflated(point_xy, aabbs):
    return any(
        point_aabb_raw_distance(point_xy, center, half_extent) <= 1.0e-9
        for center, half_extent in aabbs
    )


def sample_random_obstacle_scenario(
    map_seed,
    obstacle_count,
    robot_radius_m=DEFAULT_ROBOT_RADIUS_M,
    safety_margin_m=DEFAULT_SAFETY_MARGIN_M,
    min_goal_distance_m=DEFAULT_MIN_GOAL_DISTANCE_M,
    min_obstacle_gap_m=DEFAULT_MIN_OBSTACLE_GAP_M,
    grid_cell_m=DEFAULT_GRID_CELL_M,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
    size_range_m=(0.4, 1.1),
    split_name="test",
    max_path_turn_degrees=40.0,
    smooth_iterations=2,
    initial_yaw_offset_rad=math.radians(30.0),
    boundary_clearance_m=1.2,
):
    """Deterministically sample an accepted random-obstacle scenario.

    Unaccepted draws increment ``attempt_index``; each attempt uses
    ``default_rng(map_seed * 1009 + attempt_index)`` so results are
    reproducible across processes. Acceptance uses only static rules.
    """
    split = RandomObstacleSplitConfig()
    if split_name not in ("train", "validation", "test", "ood"):
        raise ValueError("split_name must be train/validation/test/ood")
    if not split.in_split(map_seed, split_name):
        raise ValueError("map_seed %d is outside %s seed range" % (map_seed, split_name))
    if int(obstacle_count) < 2 or int(obstacle_count) > 5:
        raise ValueError("obstacle_count must be between 2 and 5")
    radius = float(robot_radius_m)
    margin = float(safety_margin_m)
    if radius <= 0.0 or margin < 0.0:
        raise ValueError("radius must be positive and margin nonnegative")
    min_goal = float(min_goal_distance_m)
    size_min, size_max = float(size_range_m[0]), float(size_range_m[1])
    obstacle_low = ARENA_MIN + radius + margin + 0.2
    obstacle_high = ARENA_MAX - radius - margin - 0.2
    boundary_clearance = float(boundary_clearance_m)
    if boundary_clearance <= 0.0 or boundary_clearance >= (ARENA_MAX - ARENA_MIN) / 2.0:
        raise ValueError("boundary_clearance_m must be within the arena half-width")
    spawn_low = ARENA_MIN + boundary_clearance
    spawn_high = ARENA_MAX - boundary_clearance
    for attempt in range(int(max_attempts)):
        rng = np.random.default_rng(int(map_seed) * 1009 + attempt)
        spawn_xy = (
            float(rng.uniform(spawn_low, spawn_high)),
            float(rng.uniform(spawn_low, spawn_high)),
        )
        initial_yaw = float(rng.uniform(-math.pi, math.pi))
        obstacles = []
        for _ in range(int(obstacle_count)):
            placed = False
            for _ in range(64):
                cx = float(rng.uniform(obstacle_low, obstacle_high))
                cy = float(rng.uniform(obstacle_low, obstacle_high))
                length = float(rng.uniform(size_min, size_max))
                width = float(rng.uniform(size_min, size_max))
                yaw = float(rng.uniform(0.0, math.pi))
                candidate = ObstacleBox((cx, cy), (length, width), yaw)
                # Reject overlap with the robot's inflated footprint.
                inflated_check = obstacles + [candidate]
                if _inflated_overlap(inflated_check):
                    continue
                if _point_inside_inflated(spawn_xy, [box.to_aabb() for box in inflated_check]):
                    continue
                obstacles.append(candidate)
                placed = True
                break
            if not placed:
                break
        if len(obstacles) != int(obstacle_count):
            continue
        inflated = [box.to_aabb() for box in obstacles]
        goal_xy = None
        for _ in range(64):
            gx = float(rng.uniform(spawn_low, spawn_high))
            gy = float(rng.uniform(spawn_low, spawn_high))
            distance = math.hypot(gx - spawn_xy[0], gy - spawn_xy[1])
            if distance < min_goal:
                continue
            if _point_inside_inflated((gx, gy), inflated):
                continue
            goal_xy = (gx, gy)
            break
        if goal_xy is None:
            continue
        scenario = RandomObstacleScenario(
            map_seed=int(map_seed),
            attempt_index=attempt,
            obstacle_count=int(obstacle_count),
            obstacles=tuple(obstacles),
            spawn_xy=spawn_xy,
            initial_yaw_rad=initial_yaw,
            goal_xy=goal_xy,
            robot_radius_m=radius,
            safety_margin_m=margin,
            boundary_clearance_m=boundary_clearance,
        )
        grid, cell = build_occupancy_grid(scenario, grid_cell_m)
        low_x = ARENA_MIN
        low_y = ARENA_MIN
        start_cell = (int(math.floor((spawn_xy[1] - low_y) / cell)), int(math.floor((spawn_xy[0] - low_x) / cell)))
        goal_cell = (int(math.floor((goal_xy[1] - low_y) / cell)), int(math.floor((goal_xy[0] - low_x) / cell)))
        if grid[start_cell[0], start_cell[1]] == 1 or grid[goal_cell[0], goal_cell[1]] == 1:
            continue
        path, path_length = dijkstra_8connected(grid, start_cell, goal_cell, cell)
        if path is None:
            continue
        smooth = chaikin_smooth(path, iterations=int(smooth_iterations))
        if path_max_turn_degrees(smooth) > float(max_path_turn_degrees):
            continue
        smooth_clearance = polyline_clearance(
            smooth, scenario.planning_aabbs(), robot_radius_m=0.0
        )
        if smooth_clearance["minimum"] < -1.0e-6:
            continue
        delta = smooth[1] - smooth[0]
        path_heading = math.atan2(float(delta[1]), float(delta[0]))
        offset = float(rng.uniform(-float(initial_yaw_offset_rad), float(initial_yaw_offset_rad)))
        aligned_yaw = (path_heading + offset + math.pi) % (2.0 * math.pi) - math.pi
        scenario = RandomObstacleScenario(
            map_seed=int(map_seed),
            attempt_index=attempt,
            obstacle_count=int(obstacle_count),
            obstacles=tuple(obstacles),
            spawn_xy=spawn_xy,
            initial_yaw_rad=aligned_yaw,
            goal_xy=goal_xy,
            robot_radius_m=radius,
            safety_margin_m=margin,
            boundary_clearance_m=boundary_clearance,
            oracle_path=tuple(smooth),
            oracle_path_length_m=float(smooth_path_length_m(smooth)),
        )
        return scenario
    raise RuntimeError(
        "unable to sample an accepted map for seed %d after %d attempts"
        % (int(map_seed), int(max_attempts))
    )


def validate_random_scenario(scenario, min_goal_distance_m=DEFAULT_MIN_GOAL_DISTANCE_M,
                             initial_yaw_offset_rad=math.radians(30.0)):
    """Static acceptance checks; raises ValueError with a reason on failure."""
    if not isinstance(scenario, RandomObstacleScenario):
        raise ValueError("expected RandomObstacleScenario")
    if not (2 <= int(scenario.obstacle_count) <= 5):
        raise ValueError("obstacle_count must be in [2, 5]")
    if len(scenario.obstacles) != int(scenario.obstacle_count):
        raise ValueError("obstacles do not match obstacle_count")
    radius = float(scenario.robot_radius_m)
    if radius <= 0.0:
        raise ValueError("robot_radius_m must be positive")
    inflated = [box.to_aabb() for box in scenario.obstacles]
    for box in scenario.obstacles:
        length, width = box.size_xy
        if length <= 0.0 or width <= 0.0:
            raise ValueError("obstacle dimensions must be positive")
        for point in box.corners():
            if not (ARENA_MIN <= point[0] <= ARENA_MAX and ARENA_MIN <= point[1] <= ARENA_MAX):
                raise ValueError("obstacle corner outside arena bounds")
    if _inflated_overlap(scenario.obstacles):
        raise ValueError("inflated obstacle AABBs overlap")
    for point in (scenario.spawn_xy, scenario.goal_xy):
        if not (ARENA_MIN < point[0] < ARENA_MAX and ARENA_MIN < point[1] < ARENA_MAX):
            raise ValueError("spawn/goal must lie strictly inside arena bounds")
        boundary_distance = min(
            float(point[0]) - ARENA_MIN,
            ARENA_MAX - float(point[0]),
            float(point[1]) - ARENA_MIN,
            ARENA_MAX - float(point[1]),
        )
        if boundary_distance < float(scenario.boundary_clearance_m) - 1.0e-6:
            raise ValueError("spawn/goal too close to the arena boundary")
    if _point_inside_inflated(scenario.spawn_xy, inflated):
        raise ValueError("spawn is inside an inflated obstacle")
    if _point_inside_inflated(scenario.goal_xy, inflated):
        raise ValueError("goal is inside an inflated obstacle")
    if math.hypot(scenario.goal_xy[0] - scenario.spawn_xy[0],
                  scenario.goal_xy[1] - scenario.spawn_xy[1]) < float(min_goal_distance_m):
        raise ValueError("spawn-goal distance below minimum")
    grid, cell = build_occupancy_grid(scenario)
    start_cell = (int(math.floor((scenario.spawn_xy[1] - ARENA_MIN) / cell)),
                  int(math.floor((scenario.spawn_xy[0] - ARENA_MIN) / cell)))
    goal_cell = (int(math.floor((scenario.goal_xy[1] - ARENA_MIN) / cell)),
                 int(math.floor((scenario.goal_xy[0] - ARENA_MIN) / cell)))
    path, length = dijkstra_8connected(grid, start_cell, goal_cell, cell)
    if path is None:
        raise ValueError("no inflated free-space path between spawn and goal")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("invalid oracle path length")
    if len(scenario.oracle_path) >= 2:
        points = np.asarray(scenario.oracle_path, dtype=np.float64)
        delta = points[1] - points[0]
        path_heading = math.atan2(float(delta[1]), float(delta[0]))
        error = (float(scenario.initial_yaw_rad) - path_heading + math.pi) % (2.0 * math.pi) - math.pi
        if abs(error) > float(initial_yaw_offset_rad) + 1.0e-6:
            raise ValueError("initial_yaw deviates beyond the route alignment limit")
    return True


def group_scenarios_by_topology(scenarios):
    """Group scenarios by obstacle count so each IsaacGym batch is homogeneous."""
    grouped = {}
    for scenario in scenarios:
        grouped.setdefault(int(scenario.obstacle_count), []).append(scenario)
    return {count: tuple(items) for count, items in sorted(grouped.items())}
