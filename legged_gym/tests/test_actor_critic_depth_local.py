import unittest

import torch

from legged_gym.dwl.actor_critic_depth_local import ActorCriticDepthLocal


class ActorCriticDepthLocalTests(unittest.TestCase):
    def make_policy(self):
        return ActorCriticDepthLocal(272, 272, 18, 2)

    def test_shapes_and_finite_outputs(self):
        policy = self.make_policy()
        observations = torch.zeros(4, 272)
        actions = policy.act_inference(observations)
        self.assertEqual(tuple(actions.shape), (4, 2))
        values = policy.evaluate(torch.zeros(4, 18))
        self.assertEqual(tuple(values.shape), (4, 1))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(torch.isfinite(values).all())

    def test_observation_split_is_sixteen_state_and_256_depth(self):
        policy = self.make_policy()
        state, depth = policy.split_observation(torch.zeros(2, 272))
        self.assertEqual(tuple(state.shape), (2, 16))
        self.assertEqual(tuple(depth.shape), (2, 1, 8, 32))

    def test_rejects_wrong_actor_observation_size(self):
        with self.assertRaises(ValueError):
            ActorCriticDepthLocal(271, 271, 18, 2)


if __name__ == "__main__":
    unittest.main()
