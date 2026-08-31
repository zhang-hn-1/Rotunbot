import unittest

import torch

from legged_gym.dwl.actor_critic_depth import SpatialRecurrentUnit
from legged_gym.dwl.rollout_storage_dwl import RolloutStorage


class RecurrentSRUVelocityTests(unittest.TestCase):
    def test_sru_accepts_carry_hidden_and_matches_concatenated_sequence(self):
        torch.manual_seed(3)
        unit = SpatialRecurrentUnit(5, 7)
        first = torch.randn(2, 3, 5)
        second = torch.randn(2, 2, 5)
        whole = unit(torch.cat((first, second), dim=1))
        first_hidden = unit(first)
        continued = unit(second, hidden=first_hidden)
        self.assertTrue(torch.allclose(whole, continued, atol=1.0e-6))

    def test_recurrent_storage_keeps_time_axis_and_builds_done_masks(self):
        storage = RolloutStorage(
            2, 4, [3], [2], [2], device="cpu", recurrent=True
        )
        for step in range(4):
            transition = RolloutStorage.Transition()
            transition.observations = torch.full((2, 3), float(step))
            transition.critic_observations = torch.zeros(2, 2)
            transition.actions = torch.zeros(2, 2)
            transition.rewards = torch.ones(2)
            transition.dones = torch.tensor([step == 1, False])
            transition.values = torch.zeros(2, 1)
            transition.actions_log_prob = torch.zeros(2, 1)
            transition.action_mean = torch.zeros(2, 2)
            transition.action_sigma = torch.ones(2, 2)
            transition.hidden_states = (torch.full((2, 7), float(step)), None)
            storage.add_transitions(transition)
        storage.compute_returns(torch.zeros(2, 1), 0.99, 0.95)
        batch = next(storage.mini_batch_generator(1, 1))
        observations, _, _, _, _, _, _, _, _, hidden, masks = batch
        self.assertEqual(tuple(observations.shape), (4, 2, 3))
        self.assertEqual(tuple(hidden[0].shape), (2, 7))
        expected_done = torch.tensor([1.0, 1.0, 0.0, 1.0])
        self.assertTrue(
            any(torch.equal(masks[:, column], expected_done) for column in range(2))
        )
        self.assertTrue(
            any(torch.equal(masks[:, column], torch.ones(4)) for column in range(2))
        )
        self.assertEqual(storage.last_sequence_metadata["sequence_length"], 4)
        self.assertEqual(storage.last_sequence_metadata["hidden_shape"], [2, 7])


if __name__ == "__main__":
    unittest.main()
