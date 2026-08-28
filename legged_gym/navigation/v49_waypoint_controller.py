"""Pure 5 Hz geometric waypoint control for the frozen V49 interface."""

from dataclasses import dataclass
import math

import torch

from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import (
    project_velocity_commands,
)


@dataclass(frozen=True)
class V49WaypointConfig:
    """Stage-local gains; V49 release limits remain explicit and unchanged."""

    reach_radius_m: float = 0.25
    slow_radius_m: float = 0.75
    maximum_forward_speed: float = 0.13
    maximum_yaw_rate: float = 0.10
    minimum_turn_radius: float = 3.148148148148148
    envelope_fraction: float = 0.85
    stationary_threshold: float = 0.02
    turn_authority_start_speed: float = 0.08
    turn_authority_full_speed: float = 0.10
    yaw_gain: float = 0.35
    high_bearing_speed_scale: float = 0.25


@dataclass
class WaypointCommand:
    raw_command: torch.Tensor
    projected_command: torch.Tensor
    distance: torch.Tensor
    bearing_error: torch.Tensor


@dataclass
class WaypointTick:
    active_waypoint_index: int
    raw_command: torch.Tensor
    projected_command: torch.Tensor
    distance: torch.Tensor
    bearing_error: torch.Tensor
    waypoint_reached: bool
    waypoint_switched: bool
    sequence_complete: bool

    @property
    def done(self):
        return self.sequence_complete


class V49WaypointController:
    """Compute a rolling geometric command and delegate feasibility to V49."""

    def __init__(self, config=None):
        self.config = config or V49WaypointConfig()

    @staticmethod
    def wrap_to_pi(angle):
        return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi

    def command(self, robot_xy, robot_yaw, waypoint_xy):
        robot_xy = torch.as_tensor(robot_xy, dtype=torch.float32)
        device = robot_xy.device
        robot_yaw = torch.as_tensor(
            robot_yaw, dtype=robot_xy.dtype, device=device
        ).reshape(-1)
        waypoint_xy = torch.as_tensor(
            waypoint_xy, dtype=robot_xy.dtype, device=device
        )
        if robot_xy.ndim != 2 or robot_xy.shape[1] != 2:
            raise ValueError("robot_xy must have shape [batch, 2]")
        if waypoint_xy.shape != robot_xy.shape:
            raise ValueError("waypoint_xy must have the same shape as robot_xy")
        delta = waypoint_xy - robot_xy
        distance = torch.linalg.vector_norm(delta, dim=1)
        goal_heading = torch.atan2(delta[:, 1], delta[:, 0])
        bearing_error = self.wrap_to_pi(goal_heading - robot_yaw)

        distance_scale = torch.clamp(
            distance / float(self.config.slow_radius_m), 0.0, 1.0
        )
        bearing_scale = torch.where(
            torch.abs(bearing_error) > (math.pi / 2.0),
            torch.full_like(bearing_error, self.config.high_bearing_speed_scale),
            torch.ones_like(bearing_error),
        )
        raw_v = self.config.maximum_forward_speed * distance_scale * bearing_scale
        raw_w = self.config.yaw_gain * bearing_error
        raw_command = torch.stack((raw_v, raw_w), dim=1)
        projected_command = project_velocity_commands(
            raw_command,
            self.config.maximum_forward_speed,
            self.config.maximum_yaw_rate,
            self.config.minimum_turn_radius,
            self.config.envelope_fraction,
            self.config.stationary_threshold,
            self.config.turn_authority_start_speed,
            self.config.turn_authority_full_speed,
        )
        return WaypointCommand(
            raw_command=raw_command,
            projected_command=projected_command,
            distance=distance,
            bearing_error=bearing_error,
        )


class WaypointSequenceController:
    """Ordered waypoint state machine advanced only by high-level ticks."""

    def __init__(self, waypoints, config=None, policy_steps_per_tick=10):
        waypoints = torch.as_tensor(waypoints, dtype=torch.float32)
        if waypoints.ndim != 2 or waypoints.shape[1] != 2 or not len(waypoints):
            raise ValueError("waypoints must have shape [N, 2] with N >= 1")
        if policy_steps_per_tick != 10:
            raise ValueError("V49 waypoint control requires ten policy steps per tick")
        self.waypoints = waypoints
        self.controller = V49WaypointController(config)
        self.policy_steps_per_tick = policy_steps_per_tick
        self.active_waypoint_index = 0
        self.waypoint_reached = False
        self.sequence_complete = False
        self.switch_count = 0
        self._last_policy_tick = None
        self._last_tick = None

    def tick(self, robot_xy, robot_yaw):
        if self.sequence_complete:
            zero = torch.zeros(1, 2, dtype=torch.float32)
            return WaypointTick(
                self.active_waypoint_index, zero, zero, torch.zeros(1),
                torch.zeros(1), False, False, True,
            )

        target = self.waypoints[self.active_waypoint_index].to(
            device=robot_xy.device, dtype=torch.float32
        ).unsqueeze(0)
        command = self.controller.command(robot_xy, robot_yaw, target)
        reached = bool(torch.all(command.distance <= self.controller.config.reach_radius_m))
        switched = False
        self.waypoint_reached = reached
        if reached:
            if self.active_waypoint_index == len(self.waypoints) - 1:
                self.sequence_complete = True
                command = WaypointCommand(
                    command.raw_command,
                    torch.zeros_like(command.projected_command),
                    command.distance,
                    command.bearing_error,
                )
            else:
                self.active_waypoint_index += 1
                self.switch_count += 1
                switched = True
                target = self.waypoints[self.active_waypoint_index].to(
                    device=robot_xy.device, dtype=torch.float32
                ).unsqueeze(0)
                command = self.controller.command(robot_xy, robot_yaw, target)
        result = WaypointTick(
            self.active_waypoint_index,
            command.raw_command,
            command.projected_command,
            command.distance,
            command.bearing_error,
            reached,
            switched,
            self.sequence_complete,
        )
        self._last_tick = result
        return result

    def command_for_policy_step(self, policy_step, robot_xy, robot_yaw):
        """Return the held command; only policy steps 0, 10, ... run a tick."""
        tick = int(policy_step) // self.policy_steps_per_tick
        if self._last_policy_tick != tick:
            self._last_policy_tick = tick
            self._last_tick = self.tick(robot_xy, robot_yaw)
        return self._last_tick.projected_command
