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
