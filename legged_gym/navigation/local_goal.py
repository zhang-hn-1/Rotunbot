"""Robot-frame local goal transformations."""

import torch


def world_goal_to_robot_xy(robot_xy, robot_yaw, goal_xy_world):
    """Transform world-frame XY goals into each robot's planar body frame."""
    if robot_xy.ndim != 2 or robot_xy.shape[-1] != 2:
        raise ValueError("robot_xy must have shape [N, 2]")
    if goal_xy_world.shape != robot_xy.shape:
        raise ValueError("goal_xy_world must have the same shape as robot_xy")
    if robot_yaw.ndim != 1 or robot_yaw.shape[0] != robot_xy.shape[0]:
        raise ValueError("robot_yaw must have shape [N]")
    delta = goal_xy_world - robot_xy
    c = torch.cos(robot_yaw)
    s = torch.sin(robot_yaw)
    return torch.stack(
        (c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]),
        dim=-1,
    )
