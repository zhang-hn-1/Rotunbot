"""Pose-based scripted commands for spatial validation of the frozen V62 stack."""

from enum import Enum
import math

import numpy as np
import torch

from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import (
    project_velocity_commands,
)


class CorridorControllerState(str, Enum):
    STRAIGHT = "straight"
    DECELERATION = "deceleration"
    TURN = "turn"
    ACCELERATE = "accelerate"


def _wrap_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class PoseBasedCorridorController:
    """Generate feasible velocity targets from a known corridor centerline."""

    def __init__(
        self,
        maximum_forward_speed,
        maximum_yaw_rate,
        minimum_turn_radius,
        envelope_fraction=1.0,
        straight_speed=0.10,
        turn_speed=0.10,
        turn_yaw_rate=0.05,
        deceleration_distance=0.60,
        turn_exit_distance=0.50,
    ):
        self.maximum_forward_speed = float(maximum_forward_speed)
        self.maximum_yaw_rate = float(maximum_yaw_rate)
        self.minimum_turn_radius = float(minimum_turn_radius)
        self.envelope_fraction = float(envelope_fraction)
        self.straight_speed = float(straight_speed)
        self.turn_speed = float(turn_speed)
        self.turn_yaw_rate = float(turn_yaw_rate)
        self.deceleration_distance = float(deceleration_distance)
        self.turn_exit_distance = float(turn_exit_distance)
        self.reset()

    def reset(self):
        self.state = CorridorControllerState.STRAIGHT
        self.turn_index = 0
        self.transition_activation_count = 0
        self._scenario_key = None

    def _set_state(self, state):
        if state != self.state:
            self.state = state
            self.transition_activation_count += 1

    def _scenario_changed(self, scenario):
        key = (scenario.family, int(scenario.seed), len(scenario.centerline))
        if key != self._scenario_key:
            self.reset()
            self._scenario_key = key

    def _nearest_index(self, position_xy, centerline):
        distances = np.linalg.norm(centerline - position_xy.reshape(1, 2), axis=1)
        return int(np.argmin(distances))

    def _heading_at(self, centerline, index):
        left = max(0, index - 1)
        right = min(len(centerline) - 1, index + 1)
        delta = centerline[right] - centerline[left]
        return math.atan2(float(delta[1]), float(delta[0]))

    def _project(self, command):
        tensor = torch.as_tensor(np.asarray(command, dtype=np.float32)).reshape(1, 2)
        projected = project_velocity_commands(
            tensor,
            self.maximum_forward_speed,
            self.maximum_yaw_rate,
            self.minimum_turn_radius,
            self.envelope_fraction,
        )
        return projected[0].cpu().numpy().astype(np.float64)

    def update(self, position_xy, yaw, scenario):
        position = np.asarray(position_xy, dtype=np.float64)
        if position.shape != (2,):
            raise ValueError("position_xy must have shape (2,)")
        self._scenario_changed(scenario)
        nearest = self._nearest_index(position, scenario.centerline)

        if self.turn_index >= len(scenario.turns):
            self._set_state(CorridorControllerState.STRAIGHT)
            return self._project((self.straight_speed, 0.0))

        turn = scenario.turns[self.turn_index]
        turn_start = scenario.centerline[turn.start_index]
        turn_end = scenario.centerline[turn.end_index]
        distance_to_start = float(np.linalg.norm(position - turn_start))
        distance_to_end = float(np.linalg.norm(position - turn_end))
        desired_heading = self._heading_at(scenario.centerline, nearest)
        heading_error = _wrap_angle(desired_heading - yaw)

        if nearest >= turn.end_index or distance_to_end <= self.turn_exit_distance:
            self.turn_index += 1
            self._set_state(CorridorControllerState.ACCELERATE)
            return self._project((self.straight_speed, 0.0))

        if nearest < turn.start_index and distance_to_start > self.deceleration_distance:
            self._set_state(CorridorControllerState.STRAIGHT)
            return self._project((self.straight_speed, 0.0))

        if nearest < turn.start_index:
            self._set_state(CorridorControllerState.DECELERATION)
            return self._project((min(self.turn_speed, 0.10), 0.0))

        self._set_state(CorridorControllerState.TURN)
        direction = float(turn.direction)
        if abs(heading_error) > 0.08:
            direction = math.copysign(1.0, heading_error)
        return self._project((self.turn_speed, direction * self.turn_yaw_rate))
