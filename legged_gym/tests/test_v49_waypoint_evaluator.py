"""Contracts for deterministic Stage 1 waypoint evaluation helpers."""

import unittest

from legged_gym.scripts.evaluate_v49_waypoint_sequence import (
    INITIAL_YAWS_DEG,
    TRAJECTORIES,
    initial_pose_for_episode,
    trajectory_waypoints,
)


class V49WaypointEvaluatorTests(unittest.TestCase):
    def test_trajectories_match_stage1_spec(self):
        self.assertEqual(trajectory_waypoints("A"), TRAJECTORIES["A"])
        self.assertEqual(TRAJECTORIES["A"], ((1.0, 0.0), (2.0, 0.0), (3.0, 0.0)))
        self.assertEqual(TRAJECTORIES["B"], ((1.0, 0.0), (2.0, 0.25), (3.0, 0.0)))

    def test_initial_pose_is_seeded_and_uses_declared_yaw_set(self):
        first = initial_pose_for_episode(20260828, 0)
        again = initial_pose_for_episode(20260828, 0)
        self.assertEqual(first, again)
        self.assertIn(first[2], [yaw * 3.141592653589793 / 180.0 for yaw in INITIAL_YAWS_DEG])
        self.assertGreaterEqual(first[0], -0.05)
        self.assertLessEqual(first[0], 0.05)
        self.assertGreaterEqual(first[1], -0.05)
        self.assertLessEqual(first[1], 0.05)


if __name__ == "__main__":
    unittest.main()
