import unittest

from legged_gym.tests.test_depth_local_rewards import fake_env
import torch


class WaypointSwitchingTests(unittest.TestCase):
    def test_stage_four_reaches_waypoint_without_episode_reset(self):
        env = fake_env(stage=4)
        env.active_local_goal_xy_world[:] = 0.1
        env.active_local_goal_xy_robot[:] = 0.1
        env.check_termination()
        self.assertTrue(bool(env.waypoint_reached.all()))
        self.assertTrue(bool(env.needs_new_waypoint.all()))
        self.assertFalse(bool(env.reset_buf.any()))

    def test_feasibility_filter_has_explicit_forward_and_lateral_limits(self):
        env = fake_env(stage=4)
        values = env.filter_feasible_waypoints(
            [[1.0, 0.0], [0.1, 0.0], [1.0, 1.0], [-1.0, 0.0]]
        )
        self.assertEqual(values.tolist(), [True, False, False, False])

    def test_stage_four_waits_for_planner_without_global_goal_pollution(self):
        env = fake_env(stage=4)
        env.active_local_goal_xy_world[:] = torch.tensor([[0.1, 0.0], [0.1, 0.0]])
        env.active_local_goal_xy_robot[:] = env.active_local_goal_xy_world
        global_goal_before = env.global_goal_xy_world.clone()

        env.check_termination()

        self.assertTrue(bool(env.waypoint_reached.all()))
        self.assertTrue(bool(env.needs_new_waypoint.all()))
        self.assertFalse(bool(env.reset_buf.any()))
        self.assertTrue(torch.equal(env.active_local_goal_xy_world, torch.tensor([[0.1, 0.0], [0.1, 0.0]])))
        self.assertFalse(torch.equal(env.active_local_goal_xy_world, global_goal_before))

    def test_planner_injection_is_the_only_way_to_switch_goal(self):
        env = fake_env(stage=4)
        env.needs_new_waypoint[:] = True
        next_waypoint = torch.tensor([[1.0, 0.2], [1.0, -0.2]])

        env.set_active_waypoint(next_waypoint)

        self.assertTrue(torch.equal(env.active_local_goal_xy_world, next_waypoint))
        self.assertFalse(bool(env.needs_new_waypoint.any()))
        self.assertTrue(bool(env.waypoint_changed.all()))
        self.assertTrue(torch.allclose(env.obs_buf[:, 12:14], next_waypoint / 8.0))


if __name__ == "__main__":
    unittest.main()
