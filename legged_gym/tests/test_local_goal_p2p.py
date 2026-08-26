"""Pure tests for the Robot-frame Local P2P data contract."""

import math
import unittest

import isaacgym  # noqa: F401
import torch

from legged_gym.envs.rotunbot.local_goal_p2p.local_goal_utils import (
    build_local_observation,
    world_to_robot_xy,
)
from legged_gym.utils import task_registry


class LocalGoalP2PContractTests(unittest.TestCase):
    def test_new_task_is_registered_with_single_frame_observation(self):
        env_cfg, _ = task_registry.get_cfgs("rotunbot_local_goal")
        self.assertEqual(env_cfg.env.num_single_obs, 17)
        self.assertEqual(env_cfg.env.num_observations, 17)

    def test_world_delta_is_rotated_into_robot_frame(self):
        world = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        yaw = torch.tensor(
            [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]
        )
        expected = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, -1.0],
                [-1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        torch.testing.assert_close(world_to_robot_xy(world, yaw), expected, atol=1e-6, rtol=0.0)

    def test_observation_has_declared_order_and_shape(self):
        local_goal = torch.tensor([[3.0, -1.5], [-3.0, 3.0]])
        base_lin_vel = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        base_ang_vel = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        projected_gravity = torch.tensor([[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])
        dof_pos = torch.tensor([[19.0, 20.0], [21.0, 22.0]])
        dof_vel = torch.tensor([[23.0, 24.0], [25.0, 26.0]])
        previous_actions = torch.tensor([[27.0, 28.0], [29.0, 30.0]])

        observation = build_local_observation(
            local_goal,
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            dof_pos,
            dof_vel,
            previous_actions,
            max_goal_distance=3.0,
        )

        expected = torch.tensor(
            [
                [1.0, -0.5, 1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 13.0, 14.0, 15.0, 19.0, 20.0, 23.0, 24.0, 27.0, 28.0],
                [-1.0, 1.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0, 16.0, 17.0, 18.0, 21.0, 22.0, 25.0, 26.0, 29.0, 30.0],
            ]
        )
        self.assertEqual(tuple(observation.shape), (2, 17))
        torch.testing.assert_close(observation, expected, atol=1e-6, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
