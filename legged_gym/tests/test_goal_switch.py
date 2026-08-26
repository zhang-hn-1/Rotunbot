import unittest

import numpy as np

from legged_gym.navigation.goal_switch import GoalSwitchController


class FakeEnv:
    def __init__(self):
        self.commands = np.zeros((1, 3), dtype=np.float64)
        self.episode_length_buf = np.array([17], dtype=np.int64)
        self.last_output_actions = np.array([[0.3, -0.2]])
        self.last_actions = np.array([[0.3, -0.2]])
        self.reset_buf = np.array([False])
        self.obs_history = ["h0", "h1"]


class GoalSwitchTests(unittest.TestCase):
    def test_update_changes_only_world_goal_and_preserves_state(self):
        env = FakeEnv()
        controller = GoalSwitchController(env, env_index=0)
        before = {
            "episode": env.episode_length_buf.copy(),
            "action": env.last_output_actions.copy(),
            "reset": env.reset_buf.copy(),
            "history": list(env.obs_history),
        }
        event = controller.update_world_goal([2.0, -1.0], time_s=3.5)
        np.testing.assert_allclose(env.commands[0, :2], [2.0, -1.0])
        np.testing.assert_array_equal(env.episode_length_buf, before["episode"])
        np.testing.assert_allclose(env.last_output_actions, before["action"])
        np.testing.assert_array_equal(env.reset_buf, before["reset"])
        self.assertEqual(env.obs_history, before["history"])
        self.assertEqual(event.switch_index, 0)
        self.assertEqual(event.time_s, 3.5)

    def test_action_discontinuity_is_measured_not_modified(self):
        env = FakeEnv()
        controller = GoalSwitchController(env)
        controller.update_world_goal([1.0, 0.0], time_s=0.0)
        discontinuity = controller.measure_action_discontinuity([0.8, -0.2])
        self.assertAlmostEqual(discontinuity, 0.5)
        np.testing.assert_allclose(env.last_output_actions[0], [0.3, -0.2])


if __name__ == "__main__":
    unittest.main()
