"""Shared normalized-action to feasible ``(v, w)`` conversion."""

import torch

from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import project_velocity_commands


def normalized_action_to_velocity_command(
    actions,
    maximum_forward_speed,
    maximum_yaw_rate,
    minimum_turn_radius,
    envelope_fraction=1.0,
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
    return turning.float() * torch.tanh(signed_command / 0.03)
