import unittest

import torch

from legged_gym.dwl.actor_critic_direct_velocity import ActorCriticDirectVelocity


class V1SRUStateAuditTests(unittest.TestCase):
    def _policy(self):
        policy = ActorCriticDirectVelocity(
            num_short_obs=272,
            num_proprio_obs=272,
            num_critic_obs=18,
            num_actions=2,
            depth_height=8,
            depth_width=32,
            proprio_dim=12,
            hidden_dim=16,
            encoder_dim=16,
            attention_heads=4,
            actor_hidden_dims=(16,),
            critic_hidden_dims=(16,),
        )
        policy.eval()
        return policy

    def test_direct_v1_policy_declares_stateless_single_step_sru(self):
        policy = self._policy()
        self.assertFalse(policy.is_recurrent)
        self.assertIsNone(policy.reset(torch.tensor([True])))

    def test_previous_inference_does_not_change_next_inference_hidden_input(self):
        torch.manual_seed(7)
        policy = self._policy()
        first = torch.randn(1, 272)
        second = torch.randn(1, 272)
        expected = policy.act_inference(second)
        policy.act_inference(first)
        observed = policy.act_inference(second)
        self.assertTrue(torch.allclose(observed, expected, atol=1.0e-6))

    def test_single_step_call_does_not_expose_hidden_state_to_ppo_storage(self):
        from legged_gym.dwl.ppo_dwl import PPODWL

        policy = self._policy()
        algorithm = PPODWL(policy, device="cpu")
        algorithm.act(torch.zeros(1, 272), torch.zeros(1, 18))
        self.assertIsNone(algorithm.transition.hidden_states)


if __name__ == "__main__":
    unittest.main()
