"""Observation contract for direct SRU velocity navigation."""

import torch


def build_direct_velocity_observation(
    proprioception,
    goal_xy_robot,
    previous_command,
    depth,
    max_goal_distance=8.0,
):
    """Pack ``proprio + global goal + previous command + depth`` for SRU.

    ``goal_xy_robot`` is the transformed global goal, not a planner waypoint.
    The fixed layout keeps the policy interface auditable and deployment-safe.
    """
    if proprioception.ndim != 2 or proprioception.shape[1] != 12:
        raise ValueError("proprioception must have shape [N, 12]")
    batch = proprioception.shape[0]
    for name, value in (("goal_xy_robot", goal_xy_robot), ("previous_command", previous_command)):
        if value.ndim != 2 or value.shape != (batch, 2):
            raise ValueError("%s must have shape [%d, 2]" % (name, batch))
    if depth.ndim != 3 or tuple(depth.shape[1:]) != (8, 32) or depth.shape[0] != batch:
        raise ValueError("depth must have shape [%d, 8, 32]" % batch)
    return torch.cat(
        (
            proprioception,
            goal_xy_robot / float(max_goal_distance),
            previous_command,
            depth.reshape(batch, -1),
        ),
        dim=1,
    )
