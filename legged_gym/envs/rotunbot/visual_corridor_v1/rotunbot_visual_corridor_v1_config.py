"""Configuration for V1 Depth Straight Corridor training."""

import math

from legged_gym.navigation.visual_corridor_v1 import (
    V1_CORRIDOR_LENGTH_M,
    V1_CORRIDOR_WIDTH_M,
    build_v1_straight_geometry,
)
from ..direct_velocity.rotunbot_direct_velocity_config import (
    RotunbotDirectVelocityCfg,
    RotunbotDirectVelocityCfgPPO,
)


_V1_SEGMENTS, _V1_OBSTACLES = build_v1_straight_geometry()


class RotunbotVisualCorridorV1Cfg(RotunbotDirectVelocityCfg):
    """Randomized-start straight corridor with direct velocity actions."""

    visual_stage = "V1"
    corridor_width_m = V1_CORRIDOR_WIDTH_M
    corridor_length_m = V1_CORRIDOR_LENGTH_M
    corridor_wall_width_m = V1_CORRIDOR_WIDTH_M
    corridor_wall_segments = _V1_SEGMENTS
    direct_obstacle_aabbs = _V1_OBSTACLES

    class env(RotunbotDirectVelocityCfg.env):
        num_envs = 64
        episode_length_s = 45.0

    class init_state(RotunbotDirectVelocityCfg.init_state):
        randomize_initial_velocity = False
        random_start_lateral = 0.30
        random_start_yaw = math.radians(10.0)

    class commands(RotunbotDirectVelocityCfg.commands):
        resample_commands = False
        random_start_yaw = False
        goal_distance = (V1_CORRIDOR_LENGTH_M, V1_CORRIDOR_LENGTH_M)
        goal_bearing = (0.0, 0.0)
        # Formal evaluation leaves this disabled and always uses the full
        # corridor. The V1 training entry point enables it for warm-start
        # transfer from the 0.5--2 m S2/S2B distribution.
        v1_goal_curriculum_enabled = False
        v1_curriculum_start_distance = 2.0
        v1_curriculum_horizon_steps = 12000

    class rewards(RotunbotDirectVelocityCfg.rewards):
        only_positive_rewards = False

        class scales(RotunbotDirectVelocityCfg.rewards.scales):
            goal_progress = 0.0
            path_progress = 20.0
            wall_clearance = -0.05


class RotunbotVisualCorridorV1CfgPPO(RotunbotDirectVelocityCfgPPO):
    class runner(RotunbotDirectVelocityCfgPPO.runner):
        experiment_name = "rotunbot_sru_visual_corridor_v1"
        max_iterations = 1500
