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

    def test_direct_v1_policy_declares_true_recurrent_single_step_sru(self):
        policy = self._policy()
        self.assertTrue(policy.is_recurrent)
        self.assertIsNone(policy.reset(torch.tensor([True])))

    def test_done_reset_clears_only_the_finished_environment_hidden(self):
        torch.manual_seed(7)
        policy = self._policy()
        first = torch.randn(2, 272)
        policy.act_inference(first)
        before = policy.get_hidden_states()[0].clone()
        policy.reset(torch.tensor([True, False]))
        after = policy.get_hidden_states()[0]
        self.assertTrue(torch.equal(after[0], torch.zeros_like(after[0])))
        self.assertTrue(torch.equal(after[1], before[1]))

    def test_single_step_call_exposes_pre_action_hidden_state_to_ppo(self):
        from legged_gym.dwl.ppo_dwl import PPODWL

        policy = self._policy()
        algorithm = PPODWL(policy, device="cpu")
        algorithm.init_storage(1, 2, [272], [18], [2])
        algorithm.act(torch.zeros(1, 272), torch.zeros(1, 18))
        algorithm.process_env_step(
            torch.zeros(1), torch.zeros(1, dtype=torch.bool), {}
        )
        self.assertIsNone(algorithm.storage.saved_hidden_states_a)
        algorithm.act(torch.zeros(1, 272), torch.zeros(1, 18))
        self.assertIsNotNone(algorithm.transition.hidden_states[0])


if __name__ == "__main__":
    unittest.main()
