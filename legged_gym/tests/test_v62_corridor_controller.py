import math
import unittest

import isaacgym  # noqa: F401 - import before torch-backed navigation modules
import numpy as np

from legged_gym.navigation.corridor_scenarios import (
    make_l_scenario,
    make_straight_scenario,
)
from legged_gym.navigation.v62_corridor_controller import (
    CorridorControllerState,
    PoseBasedCorridorController,
)
from legged_gym.navigation.v62_corridor_task import make_wall_segments
from legged_gym.scripts.evaluate_v62_corridor import _configure_env


class V62CorridorControllerTests(unittest.TestCase):
    def test_external_s0_evaluator_disables_internal_command_profiles(self):
        scenario = make_straight_scenario(2.0, 5.0, 1)
        cfg = type("Cfg", (), {})()
        cfg.env = type("Env", (), {})()
        cfg.noise = type("Noise", (), {})()
        cfg.domain_rand = type("DomainRand", (), {})()
        cfg.init_state = type("InitState", (), {})()
        cfg.commands = type("Commands", (), {})()
        cfg.noise.add_noise = True
        cfg.domain_rand.randomize_friction = True
        cfg.domain_rand.randomize_base_mass = True
        cfg.domain_rand.push_robots = True
        cfg.init_state.randomize_initial_velocity = True
        cfg.commands.target_curriculum = True
        cfg.commands.resampling_time = 1.0
        cfg.commands.upper_level_command_frequency_hz = None
        _configure_env(cfg, scenario)
        self.assertEqual(cfg.commands.smooth_profile_fraction, 0.0)
        self.assertEqual(cfg.commands.random_walk_profile_fraction, 0.0)
        self.assertEqual(cfg.commands.independent_smooth_profile_fraction, 0.0)

    def test_wall_segments_coalesce_straight_and_facet_turn(self):
        straight = make_straight_scenario(2.0, 5.0, 1)
        self.assertEqual(len(make_wall_segments(straight.centerline)), 1)
        corner = make_l_scenario(2.0, 3.0, 2.0, 1)
        segments = make_wall_segments(corner.centerline)
        self.assertGreater(len(segments), 2)
        self.assertLess(len(segments), 20)

    def setUp(self):
        self.controller = PoseBasedCorridorController(
            maximum_forward_speed=0.25,
            maximum_yaw_rate=0.10,
            minimum_turn_radius=2.0,
            envelope_fraction=1.0,
        )

    def test_straight_corridor_starts_with_forward_command(self):
        scenario = make_straight_scenario(2.0, 5.0, seed=1)
        command = self.controller.update(np.array([0.0, 0.0]), 0.0, scenario)

        self.assertEqual(self.controller.state, CorridorControllerState.STRAIGHT)
        self.assertGreater(command[0], 0.0)
        self.assertAlmostEqual(command[1], 0.0)

    def test_l_corridor_decelerates_before_corner_and_then_turns(self):
        scenario = make_l_scenario(2.0, 3.0, 2.0, seed=2)
        before_corner = self.controller.update(np.array([2.55, 0.0]), 0.0, scenario)
        turn = self.controller.update(np.array([3.0, 0.0]), 0.0, scenario)

        self.assertEqual(self.controller.state, CorridorControllerState.TURN)
        self.assertLessEqual(before_corner[0], 0.10 + 1.0e-6)
        self.assertGreater(turn[1], 0.0)
        self.assertTrue(self.controller.transition_activation_count > 0)

    def test_turn_exit_restores_forward_motion_from_pose(self):
        scenario = make_l_scenario(2.0, 3.0, 2.0, seed=3)
        self.controller.update(np.array([3.0, 0.0]), 0.0, scenario)
        turn_end = scenario.centerline[scenario.turns[0].end_index]
        command = self.controller.update(turn_end, math.pi / 2.0, scenario)

        self.assertIn(
            self.controller.state,
            (CorridorControllerState.ACCELERATE, CorridorControllerState.STRAIGHT),
        )
        self.assertGreater(command[0], 0.0)
        self.assertAlmostEqual(command[1], 0.0)

    def test_every_command_is_inside_configured_feasible_domain(self):
        scenario = make_l_scenario(2.0, 3.0, 2.0, seed=4)
        poses = (
            (np.array([0.0, 0.0]), 0.0),
            (np.array([2.8, 0.0]), 0.0),
            (np.array([3.0, 0.0]), 0.0),
            (np.array([3.8, 0.3]), 0.4),
            (scenario.centerline[scenario.turns[0].end_index], math.pi / 2.0),
        )
        for position, yaw in poses:
            command = self.controller.update(position, yaw, scenario)
            self.assertLessEqual(abs(command[0]), 0.25 + 1e-8)
            self.assertLessEqual(abs(command[1]), 0.10 + 1e-8)
            self.assertLessEqual(abs(command[1]), abs(command[0]) / 2.0 + 1e-8)

    def test_direction_change_requires_new_transition_activation(self):
        scenario = make_l_scenario(2.0, 3.0, 2.0, seed=5)
        self.controller.update(np.array([3.0, 0.0]), 0.0, scenario)
        first_count = self.controller.transition_activation_count
        self.controller.update(np.array([3.0, 0.0]), 0.0, scenario)
        self.assertEqual(self.controller.transition_activation_count, first_count)


if __name__ == "__main__":
    unittest.main()
