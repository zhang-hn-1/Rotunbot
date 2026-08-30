import unittest
import math

import torch

from legged_gym.dwl.actor_critic_direct_velocity import ActorCriticDirectVelocity
from legged_gym.navigation.direct_velocity import (
    normalized_action_to_velocity_command,
    velocity_command_rate_penalty,
)
from legged_gym.navigation.direct_velocity_curriculum import configure_direct_velocity_stage
from legged_gym.navigation.direct_velocity_observation import (
    build_direct_velocity_observation,
)


class DirectVelocityPolicyTests(unittest.TestCase):
    def test_curriculum_is_explicit_and_has_no_local_goal_mode(self):
        cfg = type("Cfg", (), {})()
        cfg.commands = type("Commands", (), {})()
        cfg.camera = type("Camera", (), {})()
        cfg.maze = type("Maze", (), {})()
        configure_direct_velocity_stage(cfg, "S2B")
        self.assertEqual(cfg.commands.goal_distance, (0.5, 2.0))
        self.assertAlmostEqual(cfg.commands.goal_bearing[0], -math.radians(45.0))
        self.assertAlmostEqual(cfg.commands.goal_bearing[1], math.radians(45.0))
        self.assertEqual(len(cfg.commands.replay_goal_specs), 2)
        self.assertAlmostEqual(cfg.commands.replay_goal_specs[0][0], 0.20)
        self.assertAlmostEqual(cfg.commands.replay_goal_specs[1][0], 0.10)
        self.assertTrue(cfg.camera.add_noise)
        self.assertFalse(cfg.maze.enabled)

        configure_direct_velocity_stage(cfg, "S1")
        self.assertEqual(cfg.commands.goal_distance, (0.5, 1.0))
        self.assertAlmostEqual(cfg.commands.goal_bearing[0], -math.radians(10.0))
        self.assertAlmostEqual(cfg.commands.goal_bearing[1], math.radians(10.0))

        configure_direct_velocity_stage(cfg, "S2")
        self.assertEqual(cfg.commands.goal_distance, (0.5, 1.5))
        self.assertAlmostEqual(cfg.commands.goal_bearing[0], -math.radians(30.0))
        self.assertAlmostEqual(cfg.commands.goal_bearing[1], math.radians(30.0))
        self.assertEqual(len(cfg.commands.replay_goal_specs), 1)
        self.assertAlmostEqual(cfg.commands.replay_goal_specs[0][0], 0.30)

    def test_navigation_reward_keeps_only_navigation_terms_active(self):
        from legged_gym.envs.rotunbot.direct_velocity.rotunbot_direct_velocity_config import (
            RotunbotDirectVelocityCfg,
        )
        from legged_gym.utils.helpers import class_to_dict

        scales = class_to_dict(RotunbotDirectVelocityCfg.rewards.scales)
        active = {name for name, scale in scales.items() if scale != 0.0}
        self.assertEqual(
            active,
            {"termination", "goal_progress", "goal_reach", "collision", "action_rate"},
        )

    def test_reset_goal_yaw_can_use_fresh_root_quaternion(self):
        from legged_gym.envs.rotunbot.direct_velocity.rotunbot_direct_velocity import (
            RotunbotDirectVelocity,
        )

        env = RotunbotDirectVelocity.__new__(RotunbotDirectVelocity)
        yaw = math.pi / 3.0
        quaternion = torch.tensor([[0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]])
        measured = env._yaw_from_quaternion(quaternion)
        self.assertAlmostEqual(float(measured[0]), yaw, places=5)

    def test_observation_layout_is_goal_and_previous_command_not_local_waypoint(self):
        proprio = torch.zeros(2, 12)
        goal = torch.tensor([[4.0, 1.0], [2.0, -1.0]])
        previous = torch.tensor([[0.1, 0.02], [0.2, -0.03]])
        depth = torch.ones(2, 8, 32)
        observation = build_direct_velocity_observation(proprio, goal, previous, depth)

        self.assertEqual(tuple(observation.shape), (2, 272))
        self.assertTrue(torch.equal(observation[:, 12:14], goal / 8.0))
        self.assertTrue(torch.equal(observation[:, 14:16], previous))
        self.assertTrue(torch.equal(observation[:, 16:].reshape(2, 8, 32), depth))

    def test_velocity_head_is_two_dimensional_and_maps_to_feasible_commands(self):
        policy = ActorCriticDirectVelocity(
            num_short_obs=272,
            num_proprio_obs=272,
            num_critic_obs=18,
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
        observation = torch.randn(3, 272)
        action = policy.act_inference(observation)
        self.assertEqual(tuple(action.shape), (3, 2))
        self.assertTrue(torch.all(action.abs() <= 1.0 + 1.0e-6))

        command = normalized_action_to_velocity_command(
            action,
            maximum_forward_speed=0.25,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=2.0,
            envelope_fraction=1.0,
        )
        self.assertEqual(tuple(command.shape), (3, 2))
        self.assertTrue(torch.all(command[:, 0].abs() <= 0.25 + 1.0e-6))
        self.assertTrue(torch.all(command[:, 1].abs() <= 0.10 + 1.0e-6))
        self.assertTrue(torch.all(command[:, 1].abs() <= command[:, 0].abs() / 2.0 + 1.0e-6))

    def test_action_rate_uses_previous_physical_velocity_command(self):
        current = torch.tensor([[0.10, 0.02], [0.0, -0.01]])
        previous = torch.tensor([[0.04, 0.01], [0.0, -0.01]])
        penalty = velocity_command_rate_penalty(current, previous)
        self.assertTrue(torch.allclose(penalty, torch.tensor([0.0037, 0.0])))


if __name__ == "__main__":
    unittest.main()
