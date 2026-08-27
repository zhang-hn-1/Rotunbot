import unittest

from legged_gym.navigation.oracle_diagnostics import (
    OFF_PATH_COLLISION,
    PLANNED_CORNER_COLLISION,
    PLANNED_STRAIGHT_COLLISION,
    FINAL_APPROACH_COLLISION,
    POST_SWITCH_COLLISION,
    classify_collision,
    nearest_wall_clearance,
    point_to_segment_distance,
    local_goal_polar,
    reachability_clip_ratio,
    summarize_collision_diagnostics,
)


class OracleDiagnosticsTests(unittest.TestCase):
    def test_point_to_segment_distance_clamps_to_segment(self):
        self.assertAlmostEqual(
            point_to_segment_distance([2.0, 1.0], [0.0, 0.0], [1.0, 0.0]),
            2.0 ** 0.5,
        )

    def test_wall_clearance_is_surface_distance_minus_robot_radius(self):
        surface_distance, clearance = nearest_wall_clearance(
            robot_xy=[1.2, 0.0],
            wall_centers_xy=[[0.0, 0.0]],
            wall_size_xy=[2.0, 2.0],
            robot_collision_radius=0.3,
        )
        self.assertAlmostEqual(surface_distance, 0.2)
        self.assertAlmostEqual(clearance, -0.1)

    def test_local_goal_polar_and_clip_ratio_use_raw_and_filtered_vectors(self):
        distance, bearing = local_goal_polar([0.0, 2.0])
        self.assertAlmostEqual(distance, 2.0)
        self.assertAlmostEqual(bearing, 90.0)
        self.assertAlmostEqual(reachability_clip_ratio([3.0, 0.0], [2.0, 0.0]), 1.0 / 3.0)
        self.assertEqual(reachability_clip_ratio([1.0, 0.0], [1.0, 0.0]), 0.0)

    def test_corner_and_post_switch_are_both_labeled_but_post_switch_is_primary(self):
        result = classify_collision(
            phase="NAVIGATE",
            steps_since_goal_switch=5,
            delta_bearing_deg=90.0,
            waypoint_reached=False,
            actual_current_cell=(1, 1),
            planned_from_cell=(1, 1),
            planned_waypoint_cell=(1, 2),
            planned_next_cell=(2, 2),
        )
        self.assertEqual(result["collision_class_primary"], POST_SWITCH_COLLISION)
        self.assertTrue(result["is_post_switch"])
        self.assertTrue(result["is_corner"])
        self.assertTrue(result["is_approach"])
        self.assertTrue(result["is_planned_corner"])

    def test_final_approach_has_highest_primary_priority(self):
        result = classify_collision(
            phase="FINAL_APPROACH",
            steps_since_goal_switch=1,
            delta_bearing_deg=90.0,
            waypoint_reached=False,
            actual_current_cell=(1, 1),
            planned_from_cell=(1, 1),
            planned_waypoint_cell=(1, 2),
            planned_next_cell=(2, 2),
        )
        self.assertEqual(result["collision_class_primary"], FINAL_APPROACH_COLLISION)
        self.assertTrue(result["is_final_approach"])

    def test_straight_corridor_primary_class(self):
        result = classify_collision(
            phase="NAVIGATE",
            steps_since_goal_switch=20,
            delta_bearing_deg=0.0,
            waypoint_reached=False,
            actual_current_cell=(1, 1),
            planned_from_cell=(1, 1),
            planned_waypoint_cell=(1, 2),
            planned_next_cell=(1, 3),
        )
        self.assertEqual(result["collision_class_primary"], PLANNED_STRAIGHT_COLLISION)

    def test_off_path_primary_preserves_planned_corner_overlap(self):
        result = classify_collision(
            phase="NAVIGATE",
            steps_since_goal_switch=20,
            delta_bearing_deg=90.0,
            waypoint_reached=False,
            actual_current_cell=(3, 3),
            planned_from_cell=(1, 1),
            planned_waypoint_cell=(1, 2),
            planned_next_cell=(2, 2),
        )
        self.assertEqual(result["collision_class_primary"], OFF_PATH_COLLISION)
        self.assertTrue(result["is_off_path"])
        self.assertTrue(result["is_planned_corner"])

    def test_collision_summary_reports_primary_rates_overlap_and_window_sensitivity(self):
        records = [
            {"collision_class_primary": POST_SWITCH_COLLISION, "steps_since_goal_switch": 4, "is_planned_corner": True},
            {"collision_class_primary": PLANNED_CORNER_COLLISION, "steps_since_goal_switch": 12, "is_planned_corner": True},
            {"collision_class_primary": FINAL_APPROACH_COLLISION, "steps_since_goal_switch": 30, "is_final_approach": True},
        ]
        summary = summarize_collision_diagnostics(records, episode_count=10)
        self.assertEqual(summary["collision_count"], 3)
        self.assertEqual(summary["collision_class_counts"][POST_SWITCH_COLLISION], 1)
        self.assertAlmostEqual(summary["collision_class_rates"][PLANNED_CORNER_COLLISION], 1.0 / 3.0)
        self.assertEqual(summary["overlap_label_counts"]["is_planned_corner"], 2)
        self.assertEqual(summary["collision_post_switch_window_counts"], {"5": 1, "10": 1, "20": 2})


if __name__ == "__main__":
    unittest.main()
