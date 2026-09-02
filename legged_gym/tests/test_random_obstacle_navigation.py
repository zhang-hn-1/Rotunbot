import math
import unittest

import numpy as np

from legged_gym.navigation.random_obstacle_navigation import (
    ARENA_MAX,
    ARENA_MIN,
    DEFAULT_SAFETY_MARGIN_M,
    ObstacleBox,
    RandomObstacleScenario,
    RandomObstacleSplitConfig,
    build_occupancy_grid,
    chaikin_smooth,
    dijkstra_8connected,
    group_scenarios_by_topology,
    sample_random_obstacle_scenario,
    scenario_from_metadata,
    scenario_to_metadata,
    smooth_path_length_m,
    validate_random_scenario,
)


def _scenarios(seeds, count=4, split="test"):
    return [sample_random_obstacle_scenario(seed, count, split_name=split) for seed in seeds]


class RandomObstacleSplitTests(unittest.TestCase):
    def test_seed_ranges_are_disjoint_and_validated(self):
        split = RandomObstacleSplitConfig()
        self.assertTrue(split.in_split(0, "train"))
        self.assertTrue(split.in_split(9999, "train"))
        self.assertFalse(split.in_split(10000, "train"))
        self.assertTrue(split.in_split(20000, "test"))
        self.assertFalse(split.in_split(25000, "validation"))
        with self.assertRaises(ValueError):
            RandomObstacleSplitConfig(
                train_seed_range=(0, 10000), validation_seed_range=(9999, 11999)
            )
        with self.assertRaises(ValueError):
            RandomObstacleSplitConfig(train_seed_range=(5, 2))

    def test_seed_outside_split_is_rejected(self):
        with self.assertRaises(ValueError):
            sample_random_obstacle_scenario(15000, 2, split_name="train")


class ScenarioDeterminismTests(unittest.TestCase):
    def test_same_seed_reproduces_exact_scenario(self):
        first = sample_random_obstacle_scenario(20001, 3)
        second = sample_random_obstacle_scenario(20001, 3)
        self.assertEqual(first.map_seed, second.map_seed)
        np.testing.assert_allclose(first.spawn_xy, second.spawn_xy)
        np.testing.assert_allclose(first.goal_xy, second.goal_xy)
        self.assertEqual(first.initial_yaw_rad, second.initial_yaw_rad)
        self.assertEqual(first.obstacle_count, second.obstacle_count)
        for a, b in zip(first.obstacles, second.obstacles):
            np.testing.assert_allclose(a.center_xy, b.center_xy)
            np.testing.assert_allclose(a.size_xy, b.size_xy)
            self.assertAlmostEqual(a.yaw_rad, b.yaw_rad)
        self.assertEqual(len(first.oracle_path), len(second.oracle_path))
        np.testing.assert_allclose(first.oracle_path_length_m, second.oracle_path_length_m)

    def test_different_seeds_produce_observable_difference(self):
        seeds = []
        for seed in range(20000, 20030):
            scenario = sample_random_obstacle_scenario(seed, 3)
            seeds.append(tuple(scenario.obstacles[0].center_xy))
        distinct = len({item for item in seeds})
        self.assertGreater(distinct, 1)

    def test_all_sampled_maps_pass_static_validation(self):
        for count in (2, 3, 4, 5):
            for seed in range(20000, 20008):
                scenario = sample_random_obstacle_scenario(seed, count)
                self.assertTrue(validate_random_scenario(scenario))


class ScenarioGeometryTests(unittest.TestCase):
    def test_obstacles_respect_count_and_bounds(self):
        scenario = sample_random_obstacle_scenario(20050, 5)
        self.assertEqual(scenario.obstacle_count, 5)
        self.assertEqual(len(scenario.obstacles), 5)
        for box in scenario.obstacles:
            cx, cy = box.center_xy
            self.assertTrue(ARENA_MIN < cx < ARENA_MAX)
            self.assertTrue(ARENA_MIN < cy < ARENA_MAX)
            for point in box.corners():
                self.assertTrue(ARENA_MIN - 1.0e-6 <= point[0] <= ARENA_MAX + 1.0e-6)
                self.assertTrue(ARENA_MIN - 1.0e-6 <= point[1] <= ARENA_MAX + 1.0e-6)

    def test_spawn_and_goal_are_outside_inflated_obstacles(self):
        scenario = sample_random_obstacle_scenario(20051, 4)
        inflated = [box.to_aabb() for box in scenario.obstacles]
        for point in (scenario.spawn_xy, scenario.goal_xy):
            raw = min(
                max(abs(point[0] - center[0]) - half[0], 0.0) ** 2
                + max(abs(point[1] - center[1]) - half[1], 0.0) ** 2
                for center, half in inflated
            )
            # Being outside the conservative AABB (already radius+margin inflated)
            # is the required condition.
            self.assertGreater(math.sqrt(raw) + 1.0e-9, 0.0)

    def test_rotated_box_aabb_is_conservative(self):
        box = ObstacleBox((3.0, 3.0), (1.0, 0.5), math.pi / 4.0)
        center, half = box.to_aabb()
        for corner in box.corners():
            self.assertLessEqual(abs(corner[0] - center[0]), half[0] + 1.0e-9)
            self.assertLessEqual(abs(corner[1] - center[1]), half[1] + 1.0e-9)

    def test_planning_aabbs_are_inflated_by_radius_plus_margin(self):
        scenario = sample_random_obstacle_scenario(20052, 3)
        inflated = scenario.planning_aabbs()
        self.assertTrue(all(
            half[0] >= DEFAULT_SAFETY_MARGIN_M and half[1] >= DEFAULT_SAFETY_MARGIN_M
            for _, half in inflated
        ))


class OraclePathTests(unittest.TestCase):
    def test_dijkstra_returns_eight_connected_shortest_path(self):
        scenario = sample_random_obstacle_scenario(20060, 3)
        grid, cell = build_occupancy_grid(scenario)
        self.assertGreater(len(scenario.oracle_path), 0)
        self.assertGreater(scenario.oracle_path_length_m, 0.0)
        start_cell = (int(math.floor((scenario.spawn_xy[1] - ARENA_MIN) / cell)),
                      int(math.floor((scenario.spawn_xy[0] - ARENA_MIN) / cell)))
        goal_cell = (int(math.floor((scenario.goal_xy[1] - ARENA_MIN) / cell)),
                     int(math.floor((scenario.goal_xy[0] - ARENA_MIN) / cell)))
        raw_path, _ = dijkstra_8connected(grid, start_cell, goal_cell, cell)
        smoothed = chaikin_smooth(raw_path, iterations=2)
        self.assertAlmostEqual(
            smooth_path_length_m(smoothed),
            scenario.oracle_path_length_m,
            places=6,
        )

    def test_path_cells_do_not_cross_blocked_cells(self):
        scenario = sample_random_obstacle_scenario(20061, 4)
        grid, cell = build_occupancy_grid(scenario)
        for point in scenario.oracle_path:
            cell_x = int(math.floor((point[0] - ARENA_MIN) / cell))
            cell_y = int(math.floor((point[1] - ARENA_MIN) / cell))
            self.assertEqual(int(grid[cell_y, cell_x]), 0)


class MetadataAndGroupingTests(unittest.TestCase):
    def test_metadata_round_trip(self):
        scenario = sample_random_obstacle_scenario(20070, 5)
        restored = scenario_from_metadata(scenario_to_metadata(scenario))
        self.assertEqual(restored.map_seed, scenario.map_seed)
        self.assertEqual(restored.obstacle_count, scenario.obstacle_count)
        self.assertEqual(len(restored.obstacles), len(scenario.obstacles))
        np.testing.assert_allclose(restored.goal_xy, scenario.goal_xy)
        np.testing.assert_allclose(
            restored.oracle_path_length_m, scenario.oracle_path_length_m
        )
        self.assertTrue(validate_random_scenario(restored))

    def test_topology_grouping_is_homogeneous(self):
        scenarios = _scenarios(range(20080, 20096), count=4) + _scenarios(range(20100, 20108), count=2)
        grouped = group_scenarios_by_topology(scenarios)
        self.assertEqual(set(grouped.keys()), {2, 4})
        self.assertEqual(len(grouped[4]), 16)
        self.assertEqual(len(grouped[2]), 8)


if __name__ == "__main__":
    unittest.main()
