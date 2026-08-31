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
        # V1 recurrent policy ABI: proprio + goal + previous requested
        # velocity + previous actual velocity + depth + recovery bit.
        num_single_obs = 275
        num_short_obs = 275
        num_observations = 275
        num_privileged_obs = 21
        single_num_privileged_obs = 21

    class init_state(RotunbotDirectVelocityCfg.init_state):
        randomize_initial_velocity = False
        random_start_lateral = 0.30
        random_start_yaw = math.radians(10.0)

    class commands(RotunbotDirectVelocityCfg.commands):
        resample_commands = False
        random_start_yaw = False
        goal_distance = (V1_CORRIDOR_LENGTH_M, V1_CORRIDOR_LENGTH_M)
        goal_bearing = (0.0, 0.0)
        # Formal evaluation leaves both curricula disabled and always uses the
        # full corridor. The V1 training entry point enables the performance-
        # gated sampler for warm-start transfer from the S2 distribution.
        v1_goal_curriculum_enabled = False
        v1_curriculum_start_distance = 2.0
        v1_curriculum_horizon_steps = 12000
        v1_performance_curriculum_enabled = False
        v1_curriculum_seed = 4

    class rewards(RotunbotDirectVelocityCfg.rewards):
        only_positive_rewards = False

        class scales(RotunbotDirectVelocityCfg.rewards.scales):
            goal_progress = 20.0
            path_progress = 0.0
            wall_clearance = -0.05


class RotunbotVisualCorridorV1CfgPPO(RotunbotDirectVelocityCfgPPO):
    class policy(RotunbotDirectVelocityCfgPPO.policy):
        previous_actual_velocity_dim = 2

    class runner(RotunbotDirectVelocityCfgPPO.runner):
        experiment_name = "rotunbot_sru_visual_corridor_v1"
        max_iterations = 1500
