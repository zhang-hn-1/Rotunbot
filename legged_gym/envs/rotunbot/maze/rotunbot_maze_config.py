"""Configuration for the procedural-maze Rotunbot task."""

from ..target_point.rotunbot_target_obstacle_config import (
    RotunbotTargetObstacleCfg,
    RotunbotTargetObstacleCfgPPO,
)


class RotunbotMazeCfg(RotunbotTargetObstacleCfg):
    class env(RotunbotTargetObstacleCfg.env):
        # A 15x15 maze contains many static wall actors.  This conservative
        # default avoids the actor explosion caused by the legacy value 2048.
        num_envs = 64

    class maze:
        grid_size = (15, 15)
        cell_size = 2.0
        wall_height = 1.5
        center_clearance_radius = 2
        min_goal_distance = 2.0
        seed = 0
        wall_color = (0.32, 0.38, 0.48)
        robot_collision_radius = 0.4
        terminate_on_collision = False

    class teleop:
        forward_speed = 1.0
        steering_position = 0.2
        status_interval_steps = 120

    class control(RotunbotTargetObstacleCfg.control):
        # The keyboard and learned policy both command first-axis velocity and
        # second-axis position through this same low-level controller.
        first_velocity_kp = 35.0
        second_position_kp = 200.0
        second_velocity_kd = 100.0

    class rewards(RotunbotTargetObstacleCfg.rewards):
        # Preserve negative shaping/termination signals during validation.
        only_positive_rewards = False
        class scales(RotunbotTargetObstacleCfg.rewards.scales):
            # The legacy obstacle reward prints every step. It is not needed
            # by the planner smoke and would flood the validation log.
            to_target = 0.0

    class commands(RotunbotTargetObstacleCfg.commands):
        stop_vel = 0.1
        # Global goals are sampled by reset_idx() and remain fixed until the
        # episode ends; waypoints may still be latched explicitly by planners.
        resample_commands = False
        random_start_yaw = False
        class ranges(RotunbotTargetObstacleCfg.commands.ranges):
            # Boundary wall centers are at +/-14 m; reachable goals are inside.
            pos_x = [-12.0, 12.0]
            pos_y = [-12.0, 12.0]


class RotunbotMazeCfgPPO(RotunbotTargetObstacleCfgPPO):
    class runner(RotunbotTargetObstacleCfgPPO.runner):
        experiment_name = "rotunbot_maze"
