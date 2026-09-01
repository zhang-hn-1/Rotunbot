"""Minimal episode-local waypoint manager for V1 L-turn experiments."""

import math

import numpy as np


class V1WaypointManager:
    """Own waypoint progression, never velocity generation or obstacle logic."""

    def __init__(self, waypoints, reach_radius=0.35):
        values = np.asarray(waypoints, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
            raise ValueError("waypoints must have shape [N, 2] with N >= 1")
        if not np.isfinite(values).all():
            raise ValueError("waypoints must be finite")
        if float(reach_radius) <= 0.0:
            raise ValueError("reach_radius must be positive")
        self.waypoints = values.copy()
        self.reach_radius = float(reach_radius)
        self.current_index = 0
        self.reset()

    def reset(self, waypoints=None):
        if waypoints is not None:
            values = np.asarray(waypoints, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
                raise ValueError("waypoints must have shape [N, 2] with N >= 1")
            if not np.isfinite(values).all():
                raise ValueError("waypoints must be finite")
            self.waypoints = values.copy()
        self.current_index = 0
        return self.get_current_waypoint()

    def get_current_waypoint(self):
        return self.waypoints[self.current_index].copy()

    def update(self, robot_pose):
        pose = np.asarray(robot_pose, dtype=np.float64)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("robot_pose must be finite [x, y, yaw]")
        distance = float(np.linalg.norm(pose[:2] - self.get_current_waypoint()))
        reached = distance <= self.reach_radius
        if reached and self.current_index < len(self.waypoints) - 1:
            self.current_index += 1
        return reached

    def is_final_goal_reached(self, robot_xy):
        position = np.asarray(robot_xy, dtype=np.float64)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError("robot_xy must be finite [x, y]")
        return bool(
            np.linalg.norm(position - self.waypoints[-1]) <= self.reach_radius
        )

    def get_current_waypoint_robot(self, robot_pose):
        pose = np.asarray(robot_pose, dtype=np.float64)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("robot_pose must be finite [x, y, yaw]")
        delta = self.get_current_waypoint() - pose[:2]
        c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
        return np.asarray((c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]))
