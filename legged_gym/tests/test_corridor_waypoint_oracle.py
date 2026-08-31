"""Behavioral tests for the continuous corridor waypoint Oracle."""

import math
import unittest

import numpy as np

from legged_gym.navigation.corridor_scenarios import (
    make_l_scenario,
    make_straight_scenario,
)
from legged_gym.navigation.corridor_waypoint_oracle import CorridorWaypointOracle
from legged_gym.planners.oracle_local_subgoal import CorridorWaypointAdapter


class CorridorWaypointOracleTests(unittest.TestCase):
    """The Oracle must supply geometry only, within the trained B3 envelope."""

    def test_straight_waypoint_advances_on_continuous_centerline(self):
        scenario = make_straight_scenario(2.0, 5.0, seed=1)
        oracle = CorridorWaypointOracle(
            scenario, local_distance_limit=1.0, bearing_limit_deg=45.0,
        )

        waypoint = oracle.next_waypoint((1.0, 0.0, 0.0))

        np.testing.assert_allclose(waypoint, (1.6, 0.0), atol=1.0e-8)

    def test_turn_near_waypoint_shortens_before_centerline_turn(self):
        scenario = make_l_scenario(2.0, 3.0, 2.0, seed=2)
        oracle = CorridorWaypointOracle(
            scenario, local_distance_limit=1.0, bearing_limit_deg=60.0,
        )

        waypoint = oracle.next_waypoint((2.7, 0.0, 0.0))

        np.testing.assert_allclose(waypoint, (3.0, 0.0), atol=1.0e-8)
        self.assertLess(np.linalg.norm(waypoint - np.array((2.7, 0.0))), 0.6)

    def test_waypoint_is_clamped_to_b3_distance_and_bearing(self):
        scenario = make_straight_scenario(2.0, 5.0, seed=3)
        oracle = CorridorWaypointOracle(
            scenario, local_distance_limit=0.4, bearing_limit_deg=30.0,
        )
        pose = np.array((1.0, 0.0, math.pi / 2.0))

        waypoint = oracle.next_waypoint(pose)
        delta = waypoint - pose[:2]
        local = np.array((
            math.cos(pose[2]) * delta[0] + math.sin(pose[2]) * delta[1],
            -math.sin(pose[2]) * delta[0] + math.cos(pose[2]) * delta[1],
        ))

        self.assertLessEqual(np.linalg.norm(local), 0.4 + 1.0e-8)
        self.assertLessEqual(abs(math.atan2(local[1], local[0])), math.radians(30.0) + 1.0e-8)

    def test_approaching_turn_does_not_create_a_right_angle_waypoint_jump(self):
        scenario = make_l_scenario(2.0, 3.0, 2.0, seed=4)
        oracle = CorridorWaypointOracle(
            scenario, local_distance_limit=1.0, bearing_limit_deg=80.0,
        )
        poses = ((2.5, 0.0, 0.0), (2.6, 0.0, 0.0), (2.7, 0.0, 0.0))

        waypoints = [oracle.next_waypoint(pose) for pose in poses]

        for previous, current in zip(waypoints[:-1], waypoints[1:]):
            self.assertLess(np.linalg.norm(current - previous), 0.11)
            self.assertLessEqual(abs(float(current[1] - previous[1])), 1.0e-8)

    def test_crossing_turn_start_keeps_waypoint_continuous(self):
        """Catch a full-lookahead jump immediately after a turn starts."""
        scenario = make_l_scenario(2.0, 3.0, 2.0, seed=41)
        oracle = CorridorWaypointOracle(
            scenario, local_distance_limit=1.0, bearing_limit_deg=80.0,
        )
        poses = ((2.999, 0.0, 0.0), (3.0, 0.0, 0.0), (3.001, 0.0, 0.0))

        waypoints = [oracle.next_waypoint(pose) for pose in poses]

        for previous, current in zip(waypoints[:-1], waypoints[1:]):
            self.assertLess(np.linalg.norm(current - previous), 0.01)
        self.assertLess(waypoints[-1][1], 0.01)

    def test_infeasible_heading_returns_explicit_bounded_local_goal_fallback(self):
        """Catch a fallback that is outside B3 capability or non-finite."""
        scenario = make_straight_scenario(2.0, 5.0, seed=6)
        oracle = CorridorWaypointOracle(
            scenario, local_distance_limit=0.4, bearing_limit_deg=30.0,
        )

        waypoint = oracle.next_waypoint((1.0, 0.0, math.pi / 2.0))

        self.assertTrue(np.isfinite(waypoint).all())
        self.assertGreater(abs(float(waypoint[1])), 0.1)
        self.assertLessEqual(np.linalg.norm(waypoint - np.array((1.0, 0.0))), 0.4 + 1.0e-8)

    def test_adapter_exposes_only_a_world_waypoint_not_an_actuator_command(self):
        scenario = make_straight_scenario(2.0, 5.0, seed=5)
        oracle = CorridorWaypointOracle(
            scenario, local_distance_limit=1.0, bearing_limit_deg=45.0,
        )
        adapter = CorridorWaypointAdapter(oracle)

        waypoint = adapter.next_waypoint((0.0, 0.0, 0.0))

        self.assertEqual(waypoint.shape, (2,))
        self.assertTrue(np.isfinite(waypoint).all())
        self.assertFalse(hasattr(adapter, "command"))


if __name__ == "__main__":
    unittest.main()
