"""Explainable V1 velocity teacher in the measured V62 command domain."""

from dataclasses import dataclass

import torch

from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import (
    project_velocity_commands,
)


@dataclass(frozen=True)
class V1VelocityTeacherConfig:
    """Conservative teacher parameters, expressed in physical units."""

    max_forward_speed: float = 0.25
    max_yaw_rate: float = 0.10
    minimum_turn_radius: float = 2.0
    feasible_envelope_fraction: float = 1.0
    goal_radius: float = 0.35
    goal_stop_margin: float = 0.05
    stopping_distance: float = 0.80
    heading_gain: float = 1.25
    yaw_rate_damping: float = 0.35
    velocity_feedback_gain: float = 0.20
    obstacle_stop_distance: float = 0.45
    # V1's centered side-wall clearance is about 0.60--0.70 m.  Treat that
    # as open corridor; only clearance approaching the robot radius brakes.
    obstacle_slow_distance: float = 0.70


def _validate_inputs(goal_xy_robot, actual_velocity, obstacle_distance):
    if goal_xy_robot.ndim != 2 or goal_xy_robot.shape[1] != 2:
        raise ValueError("goal_xy_robot must have shape [N, 2]")
    if actual_velocity.shape != goal_xy_robot.shape:
        raise ValueError("actual_velocity must have shape [N, 2]")
    if obstacle_distance.ndim != 1 or obstacle_distance.shape[0] != goal_xy_robot.shape[0]:
        raise ValueError("obstacle_distance must have shape [N]")
    if not torch.isfinite(goal_xy_robot).all() or not torch.isfinite(actual_velocity).all():
        raise ValueError("teacher inputs must be finite")


def _parameter(config, name, default=None):
    value = getattr(config, name, default)
    if value is None:
        raise ValueError("teacher config is missing %s" % name)
    return float(value)


def teacher_velocity_diagnostics(
    goal_xy_robot, actual_velocity, obstacle_distance, config
):
    """Return raw, requested, applied and projection-correction commands.

    ``actual_velocity`` is ``[v_actual, w_actual]`` in the robot frame and
    ``obstacle_distance`` is the nearest measured clearance in metres.  The
    teacher never bypasses V62: ``applied_command`` is the unchanged V62
    feasible projection of the raw teacher request.
    """
    _validate_inputs(goal_xy_robot, actual_velocity, obstacle_distance)
    dtype = goal_xy_robot.dtype
    device = goal_xy_robot.device
    distance = torch.linalg.vector_norm(goal_xy_robot, dim=1)
    bearing = torch.atan2(goal_xy_robot[:, 1], goal_xy_robot[:, 0])

    goal_radius = _parameter(config, "goal_radius")
    # The V62 plant has a small command/velocity lag.  Continue gently past
    # the nominal radius so the closed loop actually enters the environment's
    # success disk instead of settling a fraction of a millimetre outside it.
    control_goal_radius = max(
        0.0, goal_radius - _parameter(config, "goal_stop_margin", 0.05)
    )
    stopping_distance = _parameter(config, "stopping_distance", 0.80)
    if stopping_distance <= control_goal_radius:
        raise ValueError("stopping_distance must exceed goal_radius")
    approach = torch.clamp(
        (distance - control_goal_radius)
        / (stopping_distance - control_goal_radius),
        0.0,
        1.0,
    )
    # Forward-only V1 teacher: lateral goals reduce speed while heading turns
    # toward the target; the already measured V62 projection limits curvature.
    heading_factor = torch.clamp(torch.cos(bearing), min=0.0, max=1.0)

    obstacle_distance = obstacle_distance.to(device=device, dtype=dtype)
    finite_obstacle = torch.isfinite(obstacle_distance)
    safe_obstacle_distance = torch.where(
        finite_obstacle,
        obstacle_distance,
        torch.full_like(
            obstacle_distance, _parameter(config, "obstacle_slow_distance", 0.70)
        ),
    )
    obstacle_scale = torch.clamp(
        (
            safe_obstacle_distance
            - _parameter(config, "obstacle_stop_distance", 0.45)
        )
        / (
            _parameter(config, "obstacle_slow_distance", 0.70)
            - _parameter(config, "obstacle_stop_distance", 0.45)
        ),
        0.0,
        1.0,
    )
    desired_speed = (
        _parameter(config, "max_forward_speed")
        * approach
        * heading_factor
        * obstacle_scale
    )
    # Feedback is deliberately braking-only.  It uses measured velocity to
    # slow an overspeeding plant without making normal acceleration request a
    # command larger than the reliable forward domain.
    raw_speed = desired_speed - _parameter(
        config, "velocity_feedback_gain", 0.20
    ) * torch.relu(actual_velocity[:, 0] - desired_speed)
    raw_yaw = _parameter(config, "heading_gain", 1.25) * bearing - _parameter(
        config, "yaw_rate_damping", 0.35
    ) * actual_velocity[:, 1]
    raw_command = torch.stack((raw_speed, raw_yaw), dim=1)

    applied_command = project_velocity_commands(
        raw_command,
        _parameter(config, "max_forward_speed"),
        _parameter(config, "max_yaw_rate"),
        _parameter(config, "minimum_turn_radius"),
        _parameter(config, "feasible_envelope_fraction"),
    )
    correction = applied_command - raw_command
    return {
        "raw_command": raw_command,
        "requested_command": raw_command.clone(),
        "applied_command": applied_command,
        "projection_correction": correction,
        "projection_correction_norm": torch.linalg.vector_norm(correction, dim=1),
        "goal_distance": distance,
        "goal_bearing": bearing,
        "desired_speed": desired_speed,
    }


def teacher_velocity_command(
    goal_xy_robot, actual_velocity, obstacle_distance, config
):
    """Return the feasible teacher ``[v, w]`` command for each environment."""
    return teacher_velocity_diagnostics(
        goal_xy_robot, actual_velocity, obstacle_distance, config
    )["applied_command"]
