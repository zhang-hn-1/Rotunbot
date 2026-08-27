import unittest

from legged_gym.tests.test_depth_local_rewards import fake_env


class WaypointSwitchingTests(unittest.TestCase):
    def test_stage_four_reaches_waypoint_without_episode_reset(self):
        env = fake_env(stage=4)
        env.active_local_goal_xy_world[:] = 0.1
        env.active_local_goal_xy_robot[:] = 0.1
        env.check_termination()
        self.assertTrue(bool(env.waypoint_reached.all()))
        self.assertTrue(bool(env.waypoint_changed.all()))
        self.assertFalse(bool(env.reset_buf.any()))

    def test_feasibility_filter_has_explicit_forward_and_lateral_limits(self):
        env = fake_env(stage=4)
        values = env.filter_feasible_waypoints(
            [[1.0, 0.0], [0.1, 0.0], [1.0, 1.0], [-1.0, 0.0]]
        )
        self.assertEqual(values.tolist(), [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
