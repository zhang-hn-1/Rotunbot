import unittest

import torch


class _FakeEnv:
    num_envs = 2

    def __init__(self):
        self.calls = []
        self.index = 0

    def step(self, actions):
        self.calls.append(actions.clone())
        dones = torch.tensor([self.index == 1, False])
        self.index += 1
        return (
            torch.zeros(2, 3),
            None,
            torch.ones(2),
            dones,
            {"time_outs": torch.zeros(2, dtype=torch.bool)},
        )


class HighLevelActionTimingTests(unittest.TestCase):
    def test_repeat_is_derived_from_env_dt_and_frequency(self):
        from legged_gym.navigation.high_level_action_timing import derive_action_repeat

        self.assertEqual(derive_action_repeat(0.04, 5.0), 5)

    def test_non_integer_frequency_ratio_is_rejected(self):
        from legged_gym.navigation.high_level_action_timing import derive_action_repeat

        with self.assertRaises(ValueError):
            derive_action_repeat(0.04, 7.0)

    def test_macro_reward_uses_discounted_primitive_rewards(self):
        from legged_gym.navigation.high_level_action_timing import MacroStepAccumulator

        accumulator = MacroStepAccumulator(1, repeat=3, primitive_gamma=0.9)
        for index, reward in enumerate((1.0, 2.0, 3.0)):
            accumulator.add(
                torch.tensor([reward]),
                torch.tensor([False]),
                torch.tensor([False]),
                torch.tensor([0.0]),
                index,
            )
        result = accumulator.result()
        self.assertAlmostEqual(float(result.rewards[0]), 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0)
        self.assertFalse(bool(result.dones[0]))

    def test_done_mid_repeat_excludes_reset_episode_rewards(self):
        from legged_gym.navigation.high_level_action_timing import MacroStepAccumulator

        accumulator = MacroStepAccumulator(1, repeat=5, primitive_gamma=0.9)
        for index, reward in enumerate((1.0, 2.0, 100.0, 100.0, 100.0)):
            accumulator.add(
                torch.tensor([reward]),
                torch.tensor([index == 1]),
                torch.tensor([False]),
                torch.tensor([0.0]),
                index,
            )
        result = accumulator.result()
        self.assertAlmostEqual(float(result.rewards[0]), 1.0 + 0.9 * 2.0)
        self.assertTrue(bool(result.dones[0]))

    def test_timeout_bootstrap_uses_primitive_terminal_discount(self):
        from legged_gym.navigation.high_level_action_timing import MacroStepAccumulator

        accumulator = MacroStepAccumulator(1, repeat=5, primitive_gamma=0.9)
        for index in range(5):
            accumulator.add(
                torch.tensor([1.0]),
                torch.tensor([index == 3]),
                torch.tensor([index == 3]),
                torch.tensor([10.0]),
                index,
            )
        result = accumulator.result()
        self.assertAlmostEqual(float(result.timeout_bootstrap[0]), 0.9**4 * 10.0, places=5)

    def test_parallel_done_env_does_not_stop_active_env(self):
        from legged_gym.navigation.high_level_action_timing import MacroStepAccumulator

        accumulator = MacroStepAccumulator(2, repeat=3, primitive_gamma=1.0)
        accumulator.add(
            torch.tensor([1.0, 10.0]),
            torch.tensor([True, False]),
            torch.tensor([False, False]),
            torch.zeros(2),
            0,
        )
        accumulator.add(
            torch.tensor([100.0, 20.0]),
            torch.tensor([False, True]),
            torch.tensor([False, False]),
            torch.zeros(2),
            1,
        )
        result = accumulator.result()
        self.assertEqual(result.rewards.tolist(), [[1.0], [30.0]])
        self.assertEqual(result.dones.tolist(), [[True], [True]])

    def test_policy_action_is_held_and_done_env_is_masked(self):
        from legged_gym.dwl.on_policy_runner_dwl import DWLOnPolicyRunner

        fake_env = _FakeEnv()
        runner = object.__new__(DWLOnPolicyRunner)
        runner.env = fake_env
        runner.device = "cpu"
        runner.action_repeat = 3
        runner.primitive_gamma = 1.0
        runner.alg = type(
            "FakeAlgorithm",
            (),
            {"transition": type("FakeTransition", (), {"values": torch.zeros(2, 1)})()},
        )()
        action = torch.tensor([[0.5, 0.2], [-0.5, -0.2]])
        _, _, rewards, dones, _ = runner._step_high_level(action)
        self.assertEqual(len(fake_env.calls), 3)
        self.assertTrue(torch.equal(fake_env.calls[0], action))
        self.assertTrue(torch.equal(fake_env.calls[1][1], action[1]))
        self.assertTrue(torch.equal(fake_env.calls[2][1], action[1]))
        self.assertTrue(torch.equal(fake_env.calls[2][0], torch.zeros(2)))
        self.assertEqual(rewards.tolist(), [2.0, 3.0])
        self.assertEqual(dones.tolist(), [True, False])

    def test_timing_row_exposes_policy_and_primitive_identity(self):
        from legged_gym.navigation.high_level_action_timing import timing_row

        row = timing_row(2, 3, (0.5, 0.2), (0.1, 0.02), (0.1, 0.02))
        self.assertEqual(row["policy_sample_id"], 2)
        self.assertEqual(row["primitive_step"], 3)
        self.assertEqual(row["raw_action_v"], 0.5)
        self.assertEqual(row["requested_v_cmd"], 0.1)


if __name__ == "__main__":
    unittest.main()
