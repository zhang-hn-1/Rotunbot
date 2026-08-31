import tempfile
import unittest
from pathlib import Path

import isaacgym  # noqa: F401 - Isaac Gym must precede torch
import torch

from legged_gym.dwl.actor_critic_direct_velocity import (
    ActorCriticDirectVelocity,
    load_direct_velocity_warm_start,
    migrate_direct_velocity_state_dict,
)
from legged_gym.envs.rotunbot.direct_velocity.rotunbot_direct_velocity import (
    RotunbotDirectVelocity,
)
from legged_gym.envs.rotunbot.direct_velocity.rotunbot_direct_velocity_config import (
    RotunbotDirectVelocityCfg,
)
from legged_gym.navigation.direct_velocity_observation import (
    build_direct_velocity_observation,
)


def _policy(num_obs, num_critic_obs):
    return ActorCriticDirectVelocity(
        num_short_obs=num_obs,
        num_proprio_obs=num_obs,
        num_critic_obs=num_critic_obs,
        num_actions=2,
        depth_height=8,
        depth_width=32,
        proprio_dim=12,
        hidden_dim=32,
        encoder_dim=32,
        attention_heads=4,
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
    )


class DirectVelocityRecoveryObservationTests(unittest.TestCase):
    def test_recovery_bit_appends_to_legacy_actor_prefix(self):
        proprio = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        goal = torch.tensor([[4.0, 1.0], [2.0, -1.0]])
        previous = torch.tensor([[0.1, 0.02], [0.2, -0.03]])
        depth = torch.ones(2, 8, 32)
        legacy = build_direct_velocity_observation(proprio, goal, previous, depth)

        observation = build_direct_velocity_observation(
            proprio,
            goal,
            previous,
            depth,
            recovery_active=torch.tensor([False, True]),
        )

        self.assertEqual(tuple(observation.shape), (2, 273))
        self.assertTrue(torch.equal(observation[:, :272], legacy))
        self.assertTrue(torch.equal(observation[:, 272], torch.tensor([0.0, 1.0])))

    def test_environment_publishes_recovery_bit_to_actor_critic_and_terminal_snapshot(self):
        env = RotunbotDirectVelocity.__new__(RotunbotDirectVelocity)
        env.num_envs = 2
        env.cfg = type(
            "Cfg",
            (),
            {"commands": type("Commands", (), {"maximum_goal_distance": 8.0})()},
        )()
        proprio = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        goal = torch.tensor([[4.0, 1.0], [2.0, -1.0]])
        previous = torch.tensor([[0.1, 0.02], [0.2, -0.03]])
        depth = torch.ones(2, 8, 32)
        env.capture_depth = lambda: depth
        env._proprioception = lambda: proprio
        env._goal_xy_robot = lambda: goal
        env.previous_velocity_command = previous
        env.goal_recovery_active = torch.tensor([False, True])
        env.obstacle_clearance = torch.tensor([3.0, 2.0])
        env.step_collision_buf = torch.tensor([False, True])
        env.depth_observation = torch.zeros_like(depth)
        env.obs_buf = torch.zeros(2, 273)
        env.privileged_obs_buf = torch.zeros(2, 19)
        env.terminal_privileged_obs = torch.zeros(2, 19)

        env.compute_observations()
        env._snapshot_terminal_privileged_observation(torch.tensor([1]))

        legacy = build_direct_velocity_observation(proprio, goal, previous, depth)
        expected_critic_prefix = torch.cat(
            (
                proprio,
                goal,
                previous,
                torch.tensor([[3.0], [2.0]]),
                torch.tensor([[0.0], [1.0]]),
            ),
            dim=1,
        )
        self.assertTrue(torch.equal(env.obs_buf[:, :272], legacy))
        self.assertTrue(torch.equal(env.obs_buf[:, 272], torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.equal(env.privileged_obs_buf[:, :18], expected_critic_prefix))
        self.assertTrue(torch.equal(env.privileged_obs_buf[:, 18], torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.equal(env.terminal_privileged_obs[1], env.privileged_obs_buf[1]))

    def test_direct_velocity_config_declares_recovery_aware_observation_dimensions(self):
        self.assertEqual(RotunbotDirectVelocityCfg.env.num_observations, 273)
        self.assertEqual(RotunbotDirectVelocityCfg.env.num_single_obs, 273)
        self.assertEqual(RotunbotDirectVelocityCfg.env.num_short_obs, 273)
        self.assertEqual(RotunbotDirectVelocityCfg.env.num_privileged_obs, 19)
        self.assertEqual(RotunbotDirectVelocityCfg.env.single_num_privileged_obs, 19)

    def test_migration_preserves_old_actor_and_critic_outputs_when_recovery_is_zero(self):
        torch.manual_seed(7)
        old_policy = _policy(272, 18)
        new_policy = _policy(273, 19)

        migrated = migrate_direct_velocity_state_dict(
            old_policy.state_dict(), new_policy.state_dict()
        )
        new_policy.load_state_dict(migrated, strict=True)
        old_actor_observation = torch.randn(3, 272)
        old_critic_observation = torch.randn(3, 18)
        new_actor_observation = torch.cat(
            (old_actor_observation, torch.zeros(3, 1)), dim=1
        )
        new_critic_observation = torch.cat(
            (old_critic_observation, torch.zeros(3, 1)), dim=1
        )

        self.assertTrue(
            torch.equal(
                old_policy.act_inference(old_actor_observation),
                new_policy.act_inference(new_actor_observation),
            )
        )
        self.assertTrue(
            torch.equal(
                old_policy.evaluate(old_critic_observation),
                new_policy.evaluate(new_critic_observation),
            )
        )

    def test_migrated_recovery_columns_start_inert_and_can_learn(self):
        torch.manual_seed(11)
        old_policy = _policy(272, 18)
        new_policy = _policy(273, 19)
        new_policy.load_state_dict(
            migrate_direct_velocity_state_dict(old_policy.state_dict(), new_policy.state_dict())
        )
        legacy = torch.randn(2, 272)
        zero_bit = torch.cat((legacy, torch.zeros(2, 1)), dim=1)
        one_bit = torch.cat((legacy, torch.ones(2, 1)), dim=1)

        self.assertTrue(torch.equal(new_policy.act_inference(zero_bit), new_policy.act_inference(one_bit)))
        self.assertTrue(torch.equal(new_policy.evaluate(torch.zeros(2, 19)), new_policy.evaluate(torch.tensor([[0.0] * 18 + [1.0]] * 2))))
        self.assertTrue(torch.equal(new_policy.depth_encoder.cross_query.weight[:, -1], torch.zeros(32)))
        self.assertTrue(torch.equal(new_policy.memory.input_projection.weight[:, -1], torch.zeros(32)))
        self.assertTrue(torch.equal(new_policy.memory.spatial_projection.weight[:, -1], torch.zeros(32)))
        self.assertTrue(torch.equal(new_policy.memory.gates.weight[:, 48], torch.zeros(64)))
        self.assertTrue(torch.equal(new_policy.velocity_head[0].weight[:, -1], torch.zeros(32)))
        self.assertTrue(torch.equal(new_policy.critic[0].weight[:, -1], torch.zeros(32)))

        with torch.no_grad():
            new_policy.velocity_head[0].weight[:, -1].fill_(0.25)
        self.assertFalse(torch.equal(new_policy.act_inference(zero_bit), new_policy.act_inference(one_bit)))

    def test_migration_rejects_unrecognized_weight_mismatch(self):
        old_policy = _policy(272, 18)
        new_policy = _policy(273, 19)
        broken = dict(old_policy.state_dict())
        broken["velocity_output.weight"] = broken["velocity_output.weight"][:1]

        with self.assertRaisesRegex(RuntimeError, "unsupported direct-velocity checkpoint mismatch"):
            migrate_direct_velocity_state_dict(broken, new_policy.state_dict())

    def test_warm_start_loads_only_migrated_model_weights_and_keeps_new_optimizer_fresh(self):
        torch.manual_seed(13)
        old_policy = _policy(272, 18)
        old_optimizer = torch.optim.Adam(old_policy.parameters(), lr=0.01)
        old_policy.act_inference(torch.randn(2, 272)).sum().backward()
        old_optimizer.step()
        new_policy = _policy(273, 19)
        new_optimizer = torch.optim.Adam(new_policy.parameters(), lr=0.02)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model_800.pt"
            torch.save(
                {
                    "model_state_dict": old_policy.state_dict(),
                    "optimizer_state_dict": old_optimizer.state_dict(),
                    "iter": 800,
                },
                checkpoint,
            )
            result = load_direct_velocity_warm_start(new_policy, checkpoint)

        self.assertTrue(result["migrated"])
        self.assertEqual(result["source_iteration"], 800)
        self.assertEqual(new_optimizer.state_dict()["state"], {})
        old_actor_observation = torch.randn(2, 272)
        self.assertTrue(
            torch.equal(
                old_policy.act_inference(old_actor_observation),
                new_policy.act_inference(
                    torch.cat((old_actor_observation, torch.zeros(2, 1)), dim=1)
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
