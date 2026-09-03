"""Deterministic 12 m Phase-D random-obstacle scenarios."""

from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Tuple

import numpy as np


EVALUATION_VERSION = "D1_V2_12M"
GENERATOR_VERSION = "random-obstacle-v2"
ARENA_BOUNDS_M = (0.0, 0.0, 12.0, 12.0)
ARENA_MIN = 0.0
ARENA_MAX = 12.0
BOUNDARY_WALL_THICKNESS_M = 0.05
ROBOT_URDF_RELATIVE = Path("resources/robots/Rotunbot/urdf/Rotunbot.urdf")
DEFAULT_SAFETY_MARGIN_M = 0.05
DEFAULT_TERMINAL_MOTION_MARGIN_M = 1.0
DEFAULT_BOUNDARY_CLEARANCE_M = 1.5
DEFAULT_GRID_CELL_M = 0.10
DEFAULT_MIN_OBSTACLE_GAP_M = 2.5
DEFAULT_MIN_START_GOAL_DISTANCE_M = 6.0
DEFAULT_MAX_START_GOAL_DISTANCE_M = 10.5
DEFAULT_MAX_ATTEMPTS = 128
TRAIN_SEED_RANGE = (0, 9999)
VALIDATION_SEED_RANGE = (10000, 11999)
TEST_SEED_RANGE = (20000, 21999)
OOD_SEED_RANGE = (30000, 31999)


def resolve_robot_effective_radius(urdf_path=None):
    """Read the largest spherical collision radius from the active URDF."""
    path = Path(urdf_path) if urdf_path is not None else Path(__file__).resolve().parents[2] / ROBOT_URDF_RELATIVE
    path = path.resolve()
    root = ET.parse(path).getroot()
    radii = [float(item.attrib["radius"]) for item in root.findall(".//collision/geometry/sphere")]
    radii = [value for value in radii if math.isfinite(value) and value > 0.0]
    if not radii:
        raise ValueError("active Rotunbot URDF has no positive sphere collision radius")
    return max(radii), str(path)


@dataclass(frozen=True)
class RandomObstacleConfig:
    evaluation_version: str = EVALUATION_VERSION
    generator_version: str = GENERATOR_VERSION
    arena_bounds: Tuple[float, float, float, float] = ARENA_BOUNDS_M
    start_region: Tuple[float, float, float, float] = (1.5, 3.0, 2.0, 10.0)
    goal_region: Tuple[float, float, float, float] = (9.0, 10.5, 2.0, 10.0)
    obstacle_region: Tuple[float, float, float, float] = (4.0, 8.0, 1.5, 10.5)
    obstacle_count_min: int = 0
    obstacle_count_max: int = 2
    long_side_range_m: Tuple[float, float] = (0.8, 1.4)
    short_side_range_m: Tuple[float, float] = (0.6, 1.0)
    obstacle_yaw_range_rad: Tuple[float, float] = (-math.pi / 4.0, math.pi / 4.0)
    initial_heading_error_range_rad: Tuple[float, float] = (-math.radians(20.0), math.radians(20.0))
    min_start_goal_distance_m: float = DEFAULT_MIN_START_GOAL_DISTANCE_M
    max_start_goal_distance_m: float = DEFAULT_MAX_START_GOAL_DISTANCE_M
    min_obstacle_gap_m: float = DEFAULT_MIN_OBSTACLE_GAP_M
    robot_radius_m: float = field(default_factory=lambda: resolve_robot_effective_radius()[0])
    robot_effective_radius_source: str = field(default_factory=lambda: resolve_robot_effective_radius()[1])
    safety_margin_m: float = DEFAULT_SAFETY_MARGIN_M
    terminal_motion_margin_m: float = DEFAULT_TERMINAL_MOTION_MARGIN_M
    boundary_clearance_m: float = DEFAULT_BOUNDARY_CLEARANCE_M
    grid_cell_m: float = DEFAULT_GRID_CELL_M
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    max_path_turn_degrees: float = 40.0
    smooth_iterations: int = 2
    terminal_slowdown_start_m: float = 1.5
    goal_success_radius_m: float = 0.35
    max_episode_steps: int = 2250
    upper_command_hz: float = 5.0
    physics_dt_s: float = 0.005
    policy_dt_s: float = 0.020
    hold_policy_steps: int = 10
    hold_physics_steps: int = 40

    def __post_init__(self):
        low_x, low_y, high_x, high_y = self.arena_bounds
        if high_x <= low_x or high_y <= low_y:
            raise ValueError("arena_bounds must have positive dimensions")
        if self.obstacle_count_min < 0 or self.obstacle_count_max < self.obstacle_count_min:
            raise ValueError("invalid obstacle count range")
        if self.robot_radius_m <= 0.0 or self.safety_margin_m < 0.0:
            raise ValueError("invalid robot radius or safety margin")
        if self.boundary_clearance_m <= 0.0:
            raise ValueError("boundary_clearance_m must be positive")

    def to_dict(self):
        result = dict(self.__dict__)
        for key in ("arena_bounds", "start_region", "goal_region", "obstacle_region",
                    "long_side_range_m", "short_side_range_m", "obstacle_yaw_range_rad",
                    "initial_heading_error_range_rad"):
            result[key] = list(result[key])
        return result


@dataclass(frozen=True)
class RandomObstacleSplitConfig:
    train_seed_range: Tuple[int, int] = TRAIN_SEED_RANGE
    validation_seed_range: Tuple[int, int] = VALIDATION_SEED_RANGE
    test_seed_range: Tuple[int, int] = TEST_SEED_RANGE
    ood_seed_range: Tuple[int, int] = OOD_SEED_RANGE

    def __post_init__(self):
        ranges = (self.train_seed_range, self.validation_seed_range,
                  self.test_seed_range, self.ood_seed_range)
        for low, high in ranges:
            if not (isinstance(low, int) and isinstance(high, int) and 0 <= low <= high):
                raise ValueError("seed ranges must be nonnegative integer intervals")
        for first, second in zip(ranges[:-1], ranges[1:]):
            if first[1] >= second[0]:
                raise ValueError("seed ranges must be disjoint")

    def range_for(self, split):
        try:
            return {"train": self.train_seed_range, "validation": self.validation_seed_range,
                    "test": self.test_seed_range, "ood": self.ood_seed_range}[str(split)]
        except KeyError as error:
            raise ValueError("split must be train, validation, test, or ood") from error

    def in_split(self, seed, split):
        low, high = self.range_for(split)
        return int(low) <= int(seed) <= int(high)


@dataclass(frozen=True)
class ObstacleBox:
    center_xy: Tuple[float, float]
    size_xy: Tuple[float, float]
    yaw_rad: float

    def corners(self):
        cx, cy = self.center_xy
        length, width = self.size_xy
        cosine, sine = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        local = ((-length / 2.0, -width / 2.0), (length / 2.0, -width / 2.0),
                 (length / 2.0, width / 2.0), (-length / 2.0, width / 2.0))
        return tuple((cx + cosine * x - sine * y, cy + sine * x + cosine * y) for x, y in local)

    def to_aabb(self):
        corners = self.corners()
        xs, ys = zip(*corners)
        return ((0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))),
                (0.5 * (max(xs) - min(xs)), 0.5 * (max(ys) - min(ys))))


@dataclass(frozen=True)
class RandomObstacleScenario:
    map_seed: int
    attempt_index: int
    split: str
    config: RandomObstacleConfig
    obstacles: Tuple[ObstacleBox, ...]
    spawn_xy: Tuple[float, float]
    goal_xy: Tuple[float, float]
    goal_heading_rad: float
    initial_yaw_rad: float
    oracle_path: Tuple[Tuple[float, float], ...]
    oracle_path_length_m: float

    @property
    def obstacle_count(self):
        return len(self.obstacles)

    @property
    def bounds_xy(self):
        return self.config.arena_bounds

    @property
    def robot_radius_m(self):
        return self.config.robot_radius_m

    @property
    def safety_margin_m(self):
        return self.config.safety_margin_m

    def _boundary_aabbs(self, extra=0.0):
        low_x, low_y, high_x, high_y = self.bounds_xy
        physical_half = BOUNDARY_WALL_THICKNESS_M / 2.0
        half = physical_half + float(extra)
        return (
            (((low_x + high_x) / 2.0, low_y - physical_half), ((high_x - low_x) / 2.0, half)),
            (((low_x + high_x) / 2.0, high_y + physical_half), ((high_x - low_x) / 2.0, half)),
            ((low_x - physical_half, (low_y + high_y) / 2.0), (half, (high_y - low_y) / 2.0)),
            ((high_x + physical_half, (low_y + high_y) / 2.0), (half, (high_y - low_y) / 2.0)),
        )

    def raw_physics_aabbs(self):
        return self._boundary_aabbs() + tuple(box.to_aabb() for box in self.obstacles)

    def planning_aabbs(self, extra_gap=0.0):
        amount = self.robot_radius_m + self.safety_margin_m + float(extra_gap)
        return self._boundary_aabbs(amount) + tuple(
            (center, (half[0] + amount, half[1] + amount))
            for center, half in (box.to_aabb() for box in self.obstacles)
        )

    def point_physics_clearance(self, point_xy):
        return min(_point_aabb_distance(point_xy, center, half)
                   for center, half in self.raw_physics_aabbs()) - self.robot_radius_m

    @property
    def oracle_pass_side(self):
        """Return the static side used to pass the first obstacle."""
        if not self.obstacles or len(self.oracle_path) < 2:
            return "none"
        obstacle = self.obstacles[0]
        center = np.asarray(obstacle.center_xy, dtype=np.float64)
        points = np.asarray(self.oracle_path, dtype=np.float64)
        closest = points[np.argmin(np.linalg.norm(points - center, axis=1))]
        offset = float(closest[1] - center[1])
        if abs(offset) <= 1.0e-6:
            return "unknown"
        return "left" if offset > 0.0 else "right"

    def point_planning_clearance(self, point_xy):
        return min(_point_aabb_distance(point_xy, center, half)
                   for center, half in self.planning_aabbs())


def _point_aabb_distance(point_xy, center, half_extent):
    point = np.asarray(point_xy, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(half_extent, dtype=np.float64)
    return float(np.linalg.norm(np.maximum(np.abs(point - center) - half, 0.0)))


def chaikin_smooth(points, iterations=2, ratio=0.25):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("points must be a [N,2] array")
    if not 0.0 < float(ratio) < 0.5:
        raise ValueError("ratio must be in (0,0.5)")
    current = points
    for _ in range(int(iterations)):
        result = [current[0]]
        for start, end in zip(current[:-1], current[1:]):
            result.extend(((1.0 - ratio) * start + ratio * end,
                           ratio * start + (1.0 - ratio) * end))
        result.append(current[-1])
        current = np.asarray(result, dtype=np.float64)
    return current


def smooth_path_length_m(points):
    return float(np.linalg.norm(np.diff(np.asarray(points, dtype=np.float64), axis=0), axis=1).sum())


def path_max_turn_degrees(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    vectors = np.diff(points, axis=0)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1.0e-9)
    return float(max((math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))
                      for a, b in zip(vectors[:-1], vectors[1:])), default=0.0))


def build_occupancy_grid(scenario, cell_size_m=None):
    cell = float(cell_size_m or scenario.config.grid_cell_m)
    if cell <= 0.0:
        raise ValueError("cell_size_m must be positive")
    low_x, low_y, high_x, high_y = scenario.bounds_xy
    width = int(math.ceil((high_x - low_x) / cell))
    height = int(math.ceil((high_y - low_y) / cell))
    grid = np.zeros((height, width), dtype=np.int8)
    blocked = scenario.planning_aabbs()
    for iy in range(height):
        y = low_y + (iy + 0.5) * cell
        for ix in range(width):
            x = low_x + (ix + 0.5) * cell
            grid[iy, ix] = int(any(_point_aabb_distance((x, y), center, half) <= math.sqrt(2.0) * cell / 2.0
                                   for center, half in blocked))
    return grid, cell


def dijkstra_8connected(grid, start_cell, goal_cell, cell_size_m, bounds_xy=ARENA_BOUNDS_M):
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("grid must be two-dimensional")
    height, width = grid.shape
    sy, sx = map(int, start_cell)
    gy, gx = map(int, goal_cell)
    if not (0 <= sy < height and 0 <= sx < width and 0 <= gy < height and 0 <= gx < width):
        raise ValueError("path cell out of bounds")
    if grid[sy, sx] or grid[gy, gx]:
        return None, float("inf")
    distances = np.full((height, width), float("inf"), dtype=np.float64)
    parents = np.full((height, width, 2), -1, dtype=np.int64)
    distances[sy, sx] = 0.0
    queue = [(0.0, sy, sx)]
    neighbors = ((-1, -1, math.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, math.sqrt(2.0),
                 ), (0, -1, 1.0), (0, 1, 1.0), (1, -1, math.sqrt(2.0)), (1, 0, 1.0), (1, 1, math.sqrt(2.0)))
    while queue:
        distance, y, x = heapq.heappop(queue)
        if distance > distances[y, x] + 1.0e-9:
            continue
        if (y, x) == (gy, gx):
            break
        for dy, dx, cost in neighbors:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width) or grid[ny, nx]:
                continue
            if dx and dy and (grid[y, nx] or grid[ny, x]):
                continue
            candidate = distance + cost
            if candidate + 1.0e-9 < distances[ny, nx]:
                distances[ny, nx] = candidate
                parents[ny, nx] = (y, x)
                heapq.heappush(queue, (candidate, ny, nx))
    if not math.isfinite(float(distances[gy, gx])):
        return None, float("inf")
    cells = []
    current = (gy, gx)
    while current != (sy, sx):
        cells.append(current)
        current = tuple(int(value) for value in parents[current])
    cells.append((sy, sx))
    cells.reverse()
    low_x, low_y, _, _ = bounds_xy
    cell = float(cell_size_m)
    world = tuple((low_x + (x + 0.5) * cell, low_y + (y + 0.5) * cell) for y, x in cells)
    return world, float(distances[gy, gx] * cell)


def _expanded_aabb(box, amount):
    center, half = box.to_aabb()
    return center, (half[0] + amount, half[1] + amount)


def _aabb_gap(first, second):
    (cx, cy), (hx, hy) = first
    (dx, dy), (ux, uy) = second
    return math.hypot(max(abs(cx - dx) - hx - ux, 0.0), max(abs(cy - dy) - hy - uy, 0.0))


def _point_in_aabb(point, box):
    return _point_aabb_distance(point, box[0], box[1]) <= 1.0e-9


def _inside_region(point, region):
    return region[0] <= point[0] <= region[1] and region[2] <= point[1] <= region[3]


def _cell_for(point, bounds, cell):
    low_x, low_y, _, _ = bounds
    return int(math.floor((point[1] - low_y) / cell)), int(math.floor((point[0] - low_x) / cell))


def reachable_4connected(grid, start_cell, goal_cell):
    """Return whether free cells connect under the strict 4-neighbor rule."""
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("grid must be two-dimensional")
    start = tuple(int(value) for value in start_cell)
    goal = tuple(int(value) for value in goal_cell)
    height, width = grid.shape
    if not all(0 <= cell[0] < height and 0 <= cell[1] < width for cell in (start, goal)):
        return False
    if grid[start] or grid[goal]:
        return False
    queue = [start]
    visited = {start}
    while queue:
        y, x = queue.pop(0)
        if (y, x) == goal:
            return True
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (y + dy, x + dx)
            if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width and not grid[neighbor] and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def segment_swept_clearance(start, end, aabbs, samples=32):
    """Conservatively check every sampled point along a path segment."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    values = []
    for fraction in np.linspace(0.0, 1.0, max(2, int(samples))):
        point = start + float(fraction) * (end - start)
        values.append(min(_point_aabb_distance(point, center, half) for center, half in aabbs))
    return float(min(values))


def terminal_approach_clearance(scenario):
    """Minimum planning clearance in the terminal motion envelope."""
    cfg = scenario.config
    envelope = float(cfg.robot_radius_m + cfg.safety_margin_m + cfg.terminal_motion_margin_m)
    goal = np.asarray(scenario.goal_xy, dtype=np.float64)
    return min(
        _point_aabb_distance(goal, center, half) - envelope
        for center, half in scenario.raw_physics_aabbs()
    )


def sample_random_obstacle_scenario(map_seed, obstacle_count, config=None, split="test", split_name=None, robot_radius_m=None):
    """Sample an accepted map using only deterministic pre-rollout rules."""
    if split_name is not None:
        split = split_name
    if config is None:
        config = RandomObstacleConfig(robot_radius_m=(float(robot_radius_m) if robot_radius_m is not None else resolve_robot_effective_radius()[0]))
    if robot_radius_m is not None and abs(float(robot_radius_m) - config.robot_radius_m) > 1.0e-9:
        raise ValueError("robot_radius_m conflicts with config")
    if not RandomObstacleSplitConfig().in_split(map_seed, split):
        raise ValueError("map_seed outside requested split")
    if not config.obstacle_count_min <= int(obstacle_count) <= config.obstacle_count_max:
        raise ValueError("obstacle_count outside configured curriculum")
    for attempt in range(int(config.max_attempts)):
        rng = np.random.default_rng(int(map_seed) * 1009 + attempt)
        spawn = (float(rng.uniform(config.start_region[0], config.start_region[1])),
                 float(rng.uniform(config.start_region[2], config.start_region[3])))
        goal = (float(rng.uniform(config.goal_region[0], config.goal_region[1])),
                float(rng.uniform(config.goal_region[2], config.goal_region[3])))
        distance = math.hypot(goal[0] - spawn[0], goal[1] - spawn[1])
        if not config.min_start_goal_distance_m <= distance <= config.max_start_goal_distance_m:
            continue
        obstacles = []
        inflation = config.robot_radius_m + config.safety_margin_m + config.min_obstacle_gap_m / 2.0
        failed = False
        for _ in range(int(obstacle_count)):
            placed = False
            for _ in range(96):
                long_side = float(rng.uniform(*config.long_side_range_m))
                short_side = float(rng.uniform(*config.short_side_range_m))
                if short_side > long_side:
                    long_side, short_side = short_side, long_side
                candidate = ObstacleBox(
                    (float(rng.uniform(config.obstacle_region[0], config.obstacle_region[1])),
                     float(rng.uniform(config.obstacle_region[2], config.obstacle_region[3]))),
                    (long_side, short_side),
                    float(rng.uniform(*config.obstacle_yaw_range_rad)),
                )
                low_x, low_y, high_x, high_y = config.arena_bounds
                if any(not (low_x <= x <= high_x and low_y <= y <= high_y) for x, y in candidate.corners()):
                    continue
                candidate_inflated = _expanded_aabb(candidate, inflation)
                if _point_in_aabb(spawn, candidate_inflated) or _point_in_aabb(goal, candidate_inflated):
                    continue
                if any(_aabb_gap(candidate_inflated, _expanded_aabb(previous, inflation)) < config.min_obstacle_gap_m - 1.0e-9 for previous in obstacles):
                    continue
                obstacles.append(candidate)
                placed = True
                break
            if not placed:
                failed = True
                break
        if failed:
            continue
        goal_heading = math.atan2(goal[1] - spawn[1], goal[0] - spawn[0])
        provisional = RandomObstacleScenario(int(map_seed), attempt, str(split), config, tuple(obstacles), spawn, goal, goal_heading, goal_heading, (), float("nan"))
        grid, cell = build_occupancy_grid(provisional)
        path, path_length = dijkstra_8connected(grid, _cell_for(spawn, config.arena_bounds, cell), _cell_for(goal, config.arena_bounds, cell), cell, config.arena_bounds)
        if path is None:
            continue
        start_cell = _cell_for(spawn, config.arena_bounds, cell)
        goal_cell = _cell_for(goal, config.arena_bounds, cell)
        if not reachable_4connected(grid, start_cell, goal_cell):
            continue
        raw_path = np.asarray((spawn,) + tuple(path) + (goal,), dtype=np.float64)
        smooth = chaikin_smooth(raw_path, config.smooth_iterations)
        if path_max_turn_degrees(smooth) > config.max_path_turn_degrees:
            continue
        planning_aabbs = provisional.planning_aabbs()
        if min(provisional.point_planning_clearance(point) for point in smooth) <= 0.0:
            continue
        if any(segment_swept_clearance(start, end, planning_aabbs) <= 0.0 for start, end in zip(smooth[:-1], smooth[1:])):
            continue
        initial_yaw = goal_heading + float(rng.uniform(*config.initial_heading_error_range_rad))
        scenario = RandomObstacleScenario(int(map_seed), attempt, str(split), config, tuple(obstacles), spawn, goal, goal_heading, initial_yaw, tuple(tuple(float(value) for value in point) for point in smooth), smooth_path_length_m(smooth))
        if terminal_approach_clearance(scenario) < 0.0:
            continue
        validate_random_scenario(scenario)
        return scenario
    raise RuntimeError("no valid random-obstacle scenario after deterministic attempts")


def validate_random_scenario(scenario):
    if not isinstance(scenario, RandomObstacleScenario):
        raise ValueError("expected RandomObstacleScenario")
    cfg = scenario.config
    if not cfg.obstacle_count_min <= scenario.obstacle_count <= cfg.obstacle_count_max:
        raise ValueError("obstacle count invalid")
    if not _inside_region(scenario.spawn_xy, cfg.start_region) or not _inside_region(scenario.goal_xy, cfg.goal_region):
        raise ValueError("spawn/goal outside configured regions")
    distance = math.hypot(scenario.goal_xy[0] - scenario.spawn_xy[0], scenario.goal_xy[1] - scenario.spawn_xy[1])
    if not cfg.min_start_goal_distance_m <= distance <= cfg.max_start_goal_distance_m:
        raise ValueError("start-goal distance outside configured range")
    low_x, low_y, high_x, high_y = cfg.arena_bounds
    inflation = cfg.robot_radius_m + cfg.safety_margin_m + cfg.min_obstacle_gap_m / 2.0
    previous = []
    for box in scenario.obstacles:
        if box.size_xy[0] < box.size_xy[1] or box.size_xy[1] <= 0.0:
            raise ValueError("obstacle long/short dimensions invalid")
        if any(not (low_x <= x <= high_x and low_y <= y <= high_y) for x, y in box.corners()):
            raise ValueError("obstacle corner outside arena")
        inflated = _expanded_aabb(box, inflation)
        if any(_aabb_gap(inflated, prior) < cfg.min_obstacle_gap_m - 1.0e-9 for prior in previous):
            raise ValueError("inflated obstacle spacing below configured gap")
        previous.append(inflated)
        if _point_in_aabb(scenario.spawn_xy, inflated) or _point_in_aabb(scenario.goal_xy, inflated):
            raise ValueError("spawn/goal inside inflated obstacle")
    for point in (scenario.spawn_xy, scenario.goal_xy):
        if not (low_x < point[0] < high_x and low_y < point[1] < high_y):
            raise ValueError("spawn/goal outside arena")
        if min(point[0] - low_x, high_x - point[0], point[1] - low_y, high_y - point[1]) < cfg.boundary_clearance_m - 1.0e-9:
            raise ValueError("spawn/goal too close to boundary")
    grid, cell = build_occupancy_grid(scenario)
    path, length = dijkstra_8connected(grid, _cell_for(scenario.spawn_xy, cfg.arena_bounds, cell), _cell_for(scenario.goal_xy, cfg.arena_bounds, cell), cell, cfg.arena_bounds)
    if path is None or not math.isfinite(length):
        raise ValueError("inflated free-space is not connected")
    if not scenario.oracle_path:
        raise ValueError("oracle path is missing")
    if not math.isfinite(float(scenario.oracle_path_length_m)) or scenario.oracle_path_length_m <= 0.0:
        raise ValueError("oracle path length must be positive and finite")
    planning_aabbs = scenario.planning_aabbs()
    if any(segment_swept_clearance(start, end, planning_aabbs) <= 0.0 for start, end in zip(scenario.oracle_path[:-1], scenario.oracle_path[1:])):
        raise ValueError("oracle path segment violates planning clearance")
    if terminal_approach_clearance(scenario) < 0.0:
        raise ValueError("terminal approach envelope lacks clearance")
    heading_error = (scenario.initial_yaw_rad - scenario.goal_heading_rad + math.pi) % (2.0 * math.pi) - math.pi
    if abs(heading_error) > max(abs(value) for value in cfg.initial_heading_error_range_rad) + 1.0e-6:
        raise ValueError("initial heading error outside configured range")
    return True


def scenario_to_metadata(scenario):
    return {
        "evaluation_version": scenario.config.evaluation_version,
        "generator_version": scenario.config.generator_version,
        "map_seed": scenario.map_seed,
        "attempt_index": scenario.attempt_index,
        "split": scenario.split,
        "config": scenario.config.to_dict(),
        "obstacle_count": scenario.obstacle_count,
        "obstacles": [{"center_xy": list(box.center_xy), "size_xy": list(box.size_xy), "yaw_rad": box.yaw_rad} for box in scenario.obstacles],
        "spawn_xy": list(scenario.spawn_xy),
        "goal_xy": list(scenario.goal_xy),
        "goal_heading_rad": scenario.goal_heading_rad,
        "initial_yaw_rad": scenario.initial_yaw_rad,
        "initial_heading_error_rad": (scenario.initial_yaw_rad - scenario.goal_heading_rad + math.pi) % (2.0 * math.pi) - math.pi,
        "oracle_pass_side": scenario.oracle_pass_side,
        "oracle_path": [list(point) for point in scenario.oracle_path],
        "oracle_path_length_m": scenario.oracle_path_length_m,
    }


def scenario_from_metadata(metadata):
    values = dict(metadata["config"])
    for key in ("arena_bounds", "start_region", "goal_region", "obstacle_region", "long_side_range_m", "short_side_range_m", "obstacle_yaw_range_rad", "initial_heading_error_range_rad"):
        if key in values:
            values[key] = tuple(values[key])
    config = RandomObstacleConfig(**values)
    return RandomObstacleScenario(
        int(metadata["map_seed"]), int(metadata["attempt_index"]), str(metadata["split"]), config,
        tuple(ObstacleBox(tuple(item["center_xy"]), tuple(item["size_xy"]), float(item["yaw_rad"])) for item in metadata["obstacles"]),
        tuple(metadata["spawn_xy"]), tuple(metadata["goal_xy"]), float(metadata["goal_heading_rad"]), float(metadata["initial_yaw_rad"]),
        tuple(tuple(point) for point in metadata["oracle_path"]), float(metadata["oracle_path_length_m"]),
    )


def config_hash(config):
    return hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def frozen_inventory_hash(scenarios):
    return hashlib.sha256(json.dumps([scenario_to_metadata(item) for item in scenarios], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_seed_inventory(counts, seeds, config=None, split="test"):
    counts, seeds = list(counts), list(seeds)
    if len(counts) != len(seeds):
        raise ValueError("counts and seeds must have identical lengths")
    scenarios = tuple(sample_random_obstacle_scenario(seed, count, config=config, split=split) for count, seed in zip(counts, seeds))
    if len({item.map_seed for item in scenarios}) != len(scenarios):
        raise ValueError("inventory seeds must be unique")
    return scenarios


def group_scenarios_by_topology(scenarios):
    result = {}
    for scenario in scenarios:
        result.setdefault(scenario.obstacle_count, []).append(scenario)
    return {key: tuple(value) for key, value in sorted(result.items())}
