"""Curriculum presets for direct global-goal velocity navigation."""


def configure_direct_velocity_stage(env_cfg, stage):
    """Apply an explicit S1/S2/S2B training preset to a direct-velocity cfg."""
    stage = str(stage).upper()
    presets = {
        "S1": {
            "goal_distance": (1.5, 3.0),
            "goal_bearing": (-0.60, 0.60),
            "camera_noise": False,
            "random_start_yaw": False,
        },
        "S2": {
            "goal_distance": (2.0, 5.0),
            "goal_bearing": (-1.20, 1.20),
            "camera_noise": False,
            "random_start_yaw": True,
        },
        "S2B": {
            "goal_distance": (2.0, 6.0),
            "goal_bearing": (-3.141592653589793, 3.141592653589793),
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
