"""Shared normalized-action to feasible ``(v, w)`` conversion."""

import torch

from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import project_velocity_commands


def normalized_action_to_velocity_command(
    actions,
    maximum_forward_speed,
    maximum_yaw_rate,
    minimum_turn_radius,
    envelope_fraction=1.0,
    preserve_curvature_when_saturating=False,
    curvature_fraction_breakpoints=None,
    curvature_max_speed_values=None,
):
    """Map SRU velocity-head outputs into the V62 command domain."""
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("actions must have shape [N, 2]")
    normalized = actions.clamp(-1.0, 1.0)
    command = torch.empty_like(normalized)
    command[:, 0] = normalized[:, 0] * float(maximum_forward_speed)
    command[:, 1] = normalized[:, 1] * float(maximum_yaw_rate)
    return project_velocity_commands(
        command,
        maximum_forward_speed,
        maximum_yaw_rate,
        minimum_turn_radius,
        envelope_fraction,
        preserve_curvature_when_saturating=preserve_curvature_when_saturating,
        curvature_fraction_breakpoints=curvature_fraction_breakpoints,
        curvature_max_speed_values=curvature_max_speed_values,
    )


def velocity_command_rate_penalty(command, previous_command):
    """Return the squared change between two physical velocity commands."""
    if command.shape != previous_command.shape:
        raise ValueError(
            "command and previous_command must have the same shape; "
            f"received {tuple(command.shape)} and {tuple(previous_command.shape)}"
        )
    if command.ndim != 2 or command.shape[1] != 2:
        raise ValueError(
            "command tensors must have shape [batch, 2]; "
            f"received {tuple(command.shape)}"
        )
    return torch.sum(torch.square(command - previous_command), dim=1)


def goal_turn_alignment(goal_xy_robot, command, bearing_threshold=0.05):
    """Reward yaw-command sign that turns toward the robot-frame goal."""
    if goal_xy_robot.ndim != 2 or goal_xy_robot.shape[1] != 2:
        raise ValueError("goal_xy_robot must have shape [batch, 2]")
    if command.ndim != 2 or command.shape != goal_xy_robot.shape:
        raise ValueError("command must have the same shape [batch, 2] as goal_xy_robot")
    bearing = torch.atan2(goal_xy_robot[:, 1], goal_xy_robot[:, 0])
    turning = torch.abs(bearing) >= float(bearing_threshold)
    signed_command = torch.sign(bearing) * command[:, 1]
    # Keep the signal unsaturated through the V62 yaw-command range so a
    # large-bearing goal still prefers stronger feasible turning authority.
    return turning.float() * torch.tanh(signed_command / 0.10)


def goal_speed_alignment(
    goal_xy_robot,
    command,
    maximum_forward_speed=0.25,
    goal_radius=0.35,
    stopping_distance=0.80,
):
    """Prefer a distance-dependent forward speed near the goal.

    Progress alone rewards moving quickly until the robot has already passed a
    small goal.  This bounded shaping term asks for full speed outside a
    stopping band and linearly reduces the requested forward speed to zero at
    the goal radius.  It only shapes the SRU output; V62 still owns execution.
    """
    if goal_xy_robot.ndim != 2 or goal_xy_robot.shape[1] != 2:
        raise ValueError("goal_xy_robot must have shape [batch, 2]")
    if command.ndim != 2 or command.shape != goal_xy_robot.shape:
        raise ValueError("command must have the same shape [batch, 2] as goal_xy_robot")
    if stopping_distance <= goal_radius:
        raise ValueError("stopping_distance must be greater than goal_radius")
    distance = torch.linalg.vector_norm(goal_xy_robot, dim=1)
    desired_speed = torch.clamp(
        (distance - float(goal_radius))
        / float(stopping_distance - goal_radius),
        min=0.0,
        max=1.0,
    ) * float(maximum_forward_speed)
    bearing = torch.atan2(goal_xy_robot[:, 1], goal_xy_robot[:, 0])
    # A target that has moved behind the robot requires a recovery command.
    # Keep normal forward approach unchanged, but make the desired speed
    # negative in the rear half-plane so the policy can reverse instead of
    # orbiting the target indefinitely.
    rear_half_plane = torch.cos(bearing) < 0.0
    desired_speed[rear_half_plane] *= torch.cos(bearing[rear_half_plane])
    return -torch.abs(command[:, 0] - desired_speed) / float(maximum_forward_speed)
