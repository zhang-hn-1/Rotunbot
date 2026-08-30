import unittest
import math

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch
import torch

from legged_gym.dwl.actor_critic_direct_velocity import ActorCriticDirectVelocity
from legged_gym.navigation.direct_velocity import (
    goal_turn_alignment,
    goal_speed_alignment,
    goal_kinematic_recovery,
    inside_minimum_radius_turn_circle,
    normalized_action_to_velocity_command,
    update_goal_recovery_phase,
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
            {
                "termination", "goal_progress", "goal_reach", "collision",
                "action_rate", "goal_turn_alignment", "goal_speed_alignment",
                "goal_kinematic_recovery",
            },
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

    def test_velocity_head_can_apply_the_complete_v62_curvature_projection(self):
        action = torch.tensor([[1.0, 1.0]])
        command = normalized_action_to_velocity_command(
            action,
            maximum_forward_speed=0.25,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=2.0,
            envelope_fraction=1.0,
            preserve_curvature_when_saturating=True,
            curvature_fraction_breakpoints=(0.0, 0.25, 0.50, 1.0),
            curvature_max_speed_values=(0.25, 0.20, 0.15, 0.10),
        )
        self.assertTrue(
            torch.allclose(command, torch.tensor([[0.12, 0.048]]), atol=1.0e-5)
        )

    def test_direct_velocity_mapping_is_symmetric_and_bounded(self):
        action = torch.tensor([[-1.0, -1.0], [-0.5, 1.0], [0.5, -1.0], [1.0, 1.0]])
        command = normalized_action_to_velocity_command(
            action,
            maximum_forward_speed=0.25,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=2.0,
            envelope_fraction=1.0,
        )

        self.assertTrue(
            torch.allclose(command[:, 0], torch.tensor([-0.25, -0.125, 0.125, 0.25]))
        )
        self.assertTrue(torch.allclose(command[0], -command[3]))
        self.assertTrue(torch.allclose(command[1], -command[2]))

    def test_action_rate_uses_previous_physical_velocity_command(self):
        current = torch.tensor([[0.10, 0.02], [0.0, -0.01]])
        previous = torch.tensor([[0.04, 0.01], [0.0, -0.01]])
        penalty = velocity_command_rate_penalty(current, previous)
        self.assertTrue(torch.allclose(penalty, torch.tensor([0.0037, 0.0])))

    def test_goal_turn_alignment_prefers_command_toward_goal_side(self):
        goal = torch.tensor([[1.0, 0.5], [1.0, -0.5]])
        toward = torch.tensor([[0.0, 0.05], [0.0, -0.05]])
        away = -toward
        self.assertTrue(torch.all(goal_turn_alignment(goal, toward) > 0.0))
        self.assertTrue(torch.all(goal_turn_alignment(goal, away) < 0.0))

    def test_goal_speed_alignment_requests_braking_inside_stopping_band(self):
        goal = torch.tensor([[2.0, 0.0], [0.5, 0.0], [-1.0, 0.0], [0.5, 0.5]])
        fast = torch.tensor([[0.25, 0.0], [0.25, 0.0]])
        slow = torch.tensor([[0.25, 0.0], [0.08, 0.0]])
        fast = torch.cat((fast, torch.tensor([[0.25, 0.0], [0.20, 0.0]])), dim=0)
        slow = torch.cat((slow, torch.tensor([[-0.20, 0.0], [-0.20, 0.0]])), dim=0)
        fast_reward = goal_speed_alignment(goal, fast)
        slow_reward = goal_speed_alignment(goal, slow)
        self.assertAlmostEqual(float(fast_reward[0]), 0.0, places=6)
        self.assertGreater(float(slow_reward[1]), float(fast_reward[1]))
        self.assertGreater(float(slow_reward[2]), float(fast_reward[2]))
        self.assertGreater(float(slow_reward[3]), float(fast_reward[3]))

    def test_goal_kinematic_recovery_prefers_reverse_in_nonconvergent_geometry(self):
        goal = torch.tensor([[0.5, 0.5], [2.0, 0.0]])
        reverse = torch.tensor([[-0.20, 0.0], [-0.20, 0.0]])
        forward = -reverse
        self.assertGreater(
            float(goal_kinematic_recovery(goal, reverse)[0]),
            float(goal_kinematic_recovery(goal, forward)[0]),
        )
        self.assertAlmostEqual(float(goal_kinematic_recovery(goal, reverse)[1]), 0.0, places=6)

    def test_direct_turn_circle_uses_hand_derived_chord_boundary(self):
        sqrt_three = math.sqrt(3.0)
        goal = torch.tensor(
            [
                [sqrt_three / 2.0, 0.5],
                [sqrt_three, 1.0],
                [3.0 * sqrt_three / 2.0, 1.5],
                [1.0, 0.0],
            ]
        )

        inside = inside_minimum_radius_turn_circle(goal, minimum_turn_radius=2.0)

        self.assertTrue(torch.equal(inside, torch.tensor([True, False, False, False])))

    def test_recovery_phase_enters_on_corrected_geometry_and_exits_with_hysteresis(self):
        def polar(distance, bearing_deg):
            bearing = math.radians(bearing_deg)
            return [distance * math.cos(bearing), distance * math.sin(bearing)]

        active = torch.tensor([False, False, True, True, True, True])
        goals = torch.tensor(
            [
                polar(1.50, 30.0),
                polar(1.00, 15.0),
                polar(2.05, 30.0),
                polar(2.20, 30.0),
                polar(1.00, 5.0),
                polar(0.30, 30.0),
            ]
        )

        updated = update_goal_recovery_phase(
            active,
            goals,
            minimum_turn_radius=2.0,
            goal_radius=0.35,
            enter_bearing=math.radians(20.0),
            exit_bearing=math.radians(10.0),
            exit_distance_margin=0.10,
        )

        self.assertTrue(
            torch.equal(updated, torch.tensor([True, False, True, False, False, False]))
        )

    def test_stateful_recovery_prefers_reverse_inside_factor_two_boundary(self):
        bearing = math.radians(30.0)
        goal = torch.tensor([[1.5 * math.cos(bearing), 1.5 * math.sin(bearing)]])
        reverse = torch.tensor([[-0.20, 0.0]])
        forward = torch.tensor([[0.20, 0.0]])
        active = torch.tensor([True])

        self.assertGreater(
            float(goal_speed_alignment(goal, reverse, recovery_active=active)[0]),
            float(goal_speed_alignment(goal, forward, recovery_active=active)[0]),
        )
        self.assertGreater(
            float(goal_kinematic_recovery(goal, reverse, recovery_active=active)[0]),
            float(goal_kinematic_recovery(goal, forward, recovery_active=active)[0]),
        )

    def test_environment_reward_consumes_latched_recovery_phase(self):
        from legged_gym.envs.rotunbot.direct_velocity.rotunbot_direct_velocity import (
            RotunbotDirectVelocity,
        )

        env = RotunbotDirectVelocity.__new__(RotunbotDirectVelocity)
        env.cfg = type(
            "Cfg",
            (),
            {
                "commands": type(
                    "Commands",
                    (),
                    {
                        "max_forward_speed": 0.25,
                        "goal_radius": 0.35,
                        "minimum_turn_radius": 2.0,
                    },
                )()
            },
        )()
        bearing = math.radians(30.0)
        goal = torch.tensor(
            [
                [1.5 * math.cos(bearing), 1.5 * math.sin(bearing)],
                [1.5 * math.cos(bearing), 1.5 * math.sin(bearing)],
            ]
        )
        env._goal_xy_robot = lambda: goal
        env.previous_velocity_command = torch.tensor([[-0.20, 0.0], [-0.20, 0.0]])
        env.goal_recovery_active = torch.tensor([True, False])

        reward = env._reward_goal_kinematic_recovery()

        self.assertGreater(float(reward[0]), 0.0)
        self.assertEqual(float(reward[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
