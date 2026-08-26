"""Robot-frame and world-frame 2-D goal geometry."""

import numpy as np


def _vector(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _yaw(value):
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("robot_yaw must be finite")
    return result


def _rotation(robot_yaw):
    cosine = np.cos(robot_yaw)
    sine = np.sin(robot_yaw)
    return np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float64,
    )


def local_to_world(robot_xy, robot_yaw, local_goal_xy):
    """Convert a robot-frame offset into an absolute world-frame goal."""
    position = _vector(robot_xy, "robot_xy")
    local_goal = _vector(local_goal_xy, "local_goal_xy")
    return position + _rotation(_yaw(robot_yaw)).dot(local_goal)


def world_to_local(robot_xy, robot_yaw, world_goal_xy):
    """Convert an absolute world-frame goal into a robot-frame offset."""
    position = _vector(robot_xy, "robot_xy")
    world_goal = _vector(world_goal_xy, "world_goal_xy")
    return _rotation(_yaw(robot_yaw)).T.dot(world_goal - position)
