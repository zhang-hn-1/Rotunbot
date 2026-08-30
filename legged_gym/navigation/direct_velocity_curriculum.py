"""Curriculum presets for direct global-goal velocity navigation."""

import math


def configure_direct_velocity_stage(env_cfg, stage):
    """Apply an explicit S1/S2/S2B training preset to a direct-velocity cfg."""
    stage = str(stage).upper()
    presets = {
        "S1": {
            "goal_distance": (0.5, 1.0),
            "goal_bearing": (-math.radians(10.0), math.radians(10.0)),
            "camera_noise": False,
            "random_start_yaw": False,
        },
        "S2": {
            "goal_distance": (0.5, 1.5),
            "goal_bearing": (-math.radians(30.0), math.radians(30.0)),
            "camera_noise": False,
            "random_start_yaw": True,
        },
        "S2B": {
            "goal_distance": (0.5, 2.0),
            "goal_bearing": (-math.radians(45.0), math.radians(45.0)),
            "camera_noise": True,
            "random_start_yaw": True,
        },
    }
    if stage not in presets:
        raise ValueError("stage must be S1, S2 or S2B")
    preset = presets[stage]
    env_cfg.commands.goal_distance = preset["goal_distance"]
    env_cfg.commands.goal_bearing = preset["goal_bearing"]
    env_cfg.commands.random_start_yaw = preset["random_start_yaw"]
    env_cfg.camera.add_noise = preset["camera_noise"]
    env_cfg.maze.enabled = False
    env_cfg.maze.scene_mode = "none"
    return env_cfg
