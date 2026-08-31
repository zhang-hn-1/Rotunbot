"""Observation contract for direct SRU velocity navigation."""

import torch


def build_direct_velocity_observation(
    proprioception,
    goal_xy_robot,
    previous_command,
    depth,
    max_goal_distance=8.0,
    recovery_active=None,
    previous_actual_velocity=None,
):
    """Pack the legacy SRU input and an optional recovery-phase bit.

    ``goal_xy_robot`` is the transformed global goal, not a planner waypoint.
    The fixed layout keeps the policy interface auditable and deployment-safe.
    """
    if proprioception.ndim != 2 or proprioception.shape[1] != 12:
        raise ValueError("proprioception must have shape [N, 12]")
    batch = proprioception.shape[0]
    for name, value in (("goal_xy_robot", goal_xy_robot), ("previous_command", previous_command)):
        if value.ndim != 2 or value.shape != (batch, 2):
            raise ValueError("%s must have shape [%d, 2]" % (name, batch))
    if previous_actual_velocity is not None:
        if (
            previous_actual_velocity.ndim != 2
            or previous_actual_velocity.shape != (batch, 2)
        ):
            raise ValueError(
                "previous_actual_velocity must have shape [%d, 2]" % batch
            )
    if depth.ndim != 3 or tuple(depth.shape[1:]) != (8, 32) or depth.shape[0] != batch:
        raise ValueError("depth must have shape [%d, 8, 32]" % batch)
    fields = [
        proprioception,
        goal_xy_robot / float(max_goal_distance),
        previous_command,
    ]
    if previous_actual_velocity is not None:
        fields.append(previous_actual_velocity)
    fields.append(depth.reshape(batch, -1))
    legacy_observation = torch.cat(fields, dim=1)
    if recovery_active is None:
        return legacy_observation
    if recovery_active.ndim != 1 or recovery_active.shape[0] != batch:
        raise ValueError("recovery_active must have shape [%d]" % batch)
    return torch.cat(
        (legacy_observation, recovery_active.float().unsqueeze(1)), dim=1
    )
