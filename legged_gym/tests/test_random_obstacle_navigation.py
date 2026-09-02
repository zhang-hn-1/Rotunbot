import math
import unittest

import numpy as np

from legged_gym.navigation.random_obstacle_navigation import (
    ARENA_BOUNDS_M,
    DEFAULT_SAFETY_MARGIN_M,
    ObstacleBox,
    RandomObstacleConfig,
    RandomObstacleSplitConfig,
    build_occupancy_grid,
    build_seed_inventory,
    config_hash,
    dijkstra_8connected,
    frozen_inventory_hash,
    group_scenarios_by_topology,
    resolve_robot_effective_radius,
    sample_random_obstacle_scenario,
    scenario_from_metadata,
    scenario_to_metadata,
    validate_random_scenario,
)

from legged_gym.scripts.evaluate_oracle_corridor import schedule_terminal_speed



class RandomObstacleSplitTests(unittest.TestCase):
    def test_seed_ranges_are_disjoint_and_selected(self):
        split = RandomObstacleSplitConfig()
        self.assertTrue(split.in_split(0, "train"))
        self.assertTrue(split.in_split(10000, "validation"))
        self.assertTrue(split.in_split(20000, "test"))
        self.assertTrue(split.in_split(30000, "ood"))
        self.assertFalse(split.in_split(20000, "train"))
        with self.assertRaises(ValueError):
            RandomObstacleSplitConfig(train_seed_range=(0, 10000), validation_seed_range=(9999, 11999))

    def test_scenario_seed_must_belong_to_requested_split(self):
        with self.assertRaises(ValueError):
            sample_random_obstacle_scenario(15000, 1, split="train")


class ScenarioDeterminismTests(unittest.TestCase):
    def test_same_seed_reproduces_one_and_two_obstacle_maps(self):
        for count in (0, 1, 2):
            first = sample_random_obstacle_scenario(20001 + count, count)
            second = sample_random_obstacle_scenario(20001 + count, count)
            self.assertEqual(scenario_to_metadata(first), scenario_to_metadata(second))
            self.assertTrue(validate_random_scenario(first))

    def test_different_seeds_change_observable_map_parameters(self):
        scenarios = [sample_random_obstacle_scenario(seed, 1) for seed in range(20000, 20005)]
        self.assertGreater(len({tuple(s.spawn_xy) for s in scenarios}), 1)
        self.assertEqual({scenario.obstacle_count for scenario in scenarios}, {1})

    def test_config_and_inventory_hashes_are_deterministic(self):
        config = RandomObstacleConfig()
        first = build_seed_inventory([1, 1], [20010, 20011], config=config)
        second = build_seed_inventory([1, 1], [20010, 20011], config=config)
        self.assertEqual(config_hash(config), config_hash(config))
        self.assertEqual(frozen_inventory_hash(first), frozen_inventory_hash(second))


class ScenarioGeometryTests(unittest.TestCase):
    def test_v2_uses_12m_bounds_and_regions(self):
        scenario = sample_random_obstacle_scenario(20020, 1)
        self.assertEqual(scenario.bounds_xy, ARENA_BOUNDS_M)
        self.assertEqual(scenario.config.evaluation_version, "D1_V2_12M")
        self.assertTrue(1.5 <= scenario.spawn_xy[0] <= 3.0)
        self.assertTrue(2.0 <= scenario.spawn_xy[1] <= 10.0)
        self.assertTrue(9.0 <= scenario.goal_xy[0] <= 10.5)
        self.assertTrue(2.0 <= scenario.goal_xy[1] <= 10.0)
        self.assertEqual(scenario.obstacle_count, 1)

    def test_count_policy_allows_only_zero_one_two(self):
        config = RandomObstacleConfig()
        for count in (0, 1, 2):
            scenario = sample_random_obstacle_scenario(20030 + count, count, config=config)
            self.assertEqual(scenario.obstacle_count, count)
        with self.assertRaises(ValueError):
            sample_random_obstacle_scenario(20040, 3, config=config)

    def test_robot_radius_is_read_from_active_urdf(self):
        radius, source = resolve_robot_effective_radius()
        self.assertAlmostEqual(radius, 0.4)
        self.assertTrue(source.endswith("resources/robots/Rotunbot/urdf/Rotunbot.urdf"))

    def test_box_size_and_yaw_ranges(self):
        scenario = sample_random_obstacle_scenario(20050, 1)
        box = scenario.obstacles[0]
        self.assertGreaterEqual(box.size_xy[0], 0.8)
        self.assertLessEqual(box.size_xy[0], 1.4)
        self.assertGreaterEqual(box.size_xy[1], 0.6)
        self.assertLessEqual(box.size_xy[1], 1.0)
        self.assertGreaterEqual(box.size_xy[0], box.size_xy[1])
        self.assertLessEqual(abs(box.yaw_rad), math.pi / 4.0 + 1.0e-9)
        for x, y in box.corners():
            self.assertTrue(0.0 <= x <= 12.0)
            self.assertTrue(0.0 <= y <= 12.0)

    def test_planning_aabb_contains_obb_and_is_inflated(self):
        scenario = sample_random_obstacle_scenario(20051, 1)
        box = scenario.obstacles[0]
        center, half = box.to_aabb()
        planning = scenario.planning_aabbs()
        obstacle_planning = planning[4]
        for point in box.corners():
            self.assertLessEqual(abs(point[0] - center[0]), half[0] + 1.0e-9)
            self.assertLessEqual(abs(point[1] - center[1]), half[1] + 1.0e-9)
        self.assertGreaterEqual(obstacle_planning[1][0], half[0] + scenario.robot_radius_m + DEFAULT_SAFETY_MARGIN_M)
        self.assertGreaterEqual(obstacle_planning[1][1], half[1] + scenario.robot_radius_m + DEFAULT_SAFETY_MARGIN_M)

    def test_heading_error_is_within_plus_minus_20_degrees(self):
        scenario = sample_random_obstacle_scenario(20052, 1)
        error = (scenario.initial_yaw_rad - scenario.goal_heading_rad + math.pi) % (2.0 * math.pi) - math.pi
        self.assertLessEqual(abs(error), math.radians(20.0) + 1.0e-9)

    def test_region_distance_and_boundary_margin_are_explicit(self):
        scenario = sample_random_obstacle_scenario(20053, 1)
        distance = math.hypot(
            scenario.goal_xy[0] - scenario.spawn_xy[0],
            scenario.goal_xy[1] - scenario.spawn_xy[1],
        )
        self.assertGreaterEqual(distance, scenario.config.min_start_goal_distance_m)
        self.assertLessEqual(distance, scenario.config.max_start_goal_distance_m)
        for point in (scenario.spawn_xy, scenario.goal_xy):
            self.assertGreaterEqual(
                min(point[0], point[1], 12.0 - point[0], 12.0 - point[1]),
                scenario.config.boundary_clearance_m,
            )

    def test_terminal_speed_scheduler_has_cruise_approach_terminal(self):
        config = RandomObstacleConfig()
        self.assertEqual(schedule_terminal_speed(0.2, 2.0, config)["phase"], "Cruise")
        approach = schedule_terminal_speed(0.2, 0.8, config)
        self.assertEqual(approach["phase"], "Approach")
        self.assertLess(approach["scale"], 1.0)
        terminal = schedule_terminal_speed(0.2, 0.2, config)
        self.assertEqual(terminal["phase"], "Terminal")
        self.assertEqual(terminal["scale"], 0.0)


class OraclePathTests(unittest.TestCase):
    def test_dijkstra_uses_scenario_bounds_for_world_conversion(self):
        scenario = sample_random_obstacle_scenario(20060, 1)
        grid, cell = build_occupancy_grid(scenario)
        start = (int(math.floor((scenario.spawn_xy[1] - scenario.bounds_xy[1]) / cell)), int(math.floor((scenario.spawn_xy[0] - scenario.bounds_xy[0]) / cell)))
        goal = (int(math.floor((scenario.goal_xy[1] - scenario.bounds_xy[1]) / cell)), int(math.floor((scenario.goal_xy[0] - scenario.bounds_xy[0]) / cell)))
        path, length = dijkstra_8connected(grid, start, goal, cell, scenario.bounds_xy)
        self.assertIsNotNone(path)
        self.assertGreater(length, 0.0)
        self.assertGreater(scenario.oracle_path_length_m, 0.0)
        self.assertTrue(all(0.0 <= point[0] <= 12.0 and 0.0 <= point[1] <= 12.0 for point in path))

    def test_static_validator_rejects_tampered_map(self):
        scenario = sample_random_obstacle_scenario(20061, 2)
        metadata = scenario_to_metadata(scenario)
        metadata["goal_xy"] = [1.0, 1.0]
        with self.assertRaises(ValueError):
            validate_random_scenario(scenario_from_metadata(metadata))


class MetadataAndGroupingTests(unittest.TestCase):
    def test_metadata_round_trip_and_inventory_grouping(self):
        scenarios = build_seed_inventory([1, 2], [20070, 20071])
        restored = [scenario_from_metadata(scenario_to_metadata(item)) for item in scenarios]
        self.assertEqual([scenario_to_metadata(item) for item in restored], [scenario_to_metadata(item) for item in scenarios])
        grouped = group_scenarios_by_topology(scenarios)
        self.assertEqual(set(grouped), {1, 2})
        self.assertTrue(all(item.obstacle_count == key for key, values in grouped.items() for item in values))


if __name__ == "__main__":
    unittest.main()
