"""Configuration for direct SRU navigation through the V62 velocity port."""

from ..maze.rotunbot_maze_local_depth_config import (
    RotunbotMazeLocalDepthCfg,
)
from ..vel_tracking.rotunbot_vel_config import (
    RotunbotVelSRU50SafeYawResidualV62TransitionCfg,
    RotunbotVelSRU50SafeYawResidualV62TransitionCfgPPO,
)


class RotunbotDirectVelocityCfg(RotunbotVelSRU50SafeYawResidualV62TransitionCfg):
    """Depth + global-goal policy with a two-channel velocity action."""

    class env(RotunbotVelSRU50SafeYawResidualV62TransitionCfg.env):
        num_envs = 64
        num_actions = 2
        num_single_obs = 272
        num_short_obs = 272
        num_observations = 272
        num_privileged_obs = 18
        single_num_privileged_obs = 18
        depth_height = 8
        depth_width = 32
        depth_dim = 256
        proprio_dim = 12
        goal_dim = 2
        previous_command_dim = 2
        episode_length_s = 30.0

    class init_state(RotunbotVelSRU50SafeYawResidualV62TransitionCfg.init_state):
        randomize_initial_velocity = False

    class maze:
        enabled = False
        scene_mode = "none"
        terminate_on_collision = True
        robot_collision_radius = 0.4
        safety_clearance = 0.8

    class camera:
        enable = True
        depth_backend = "fallback"
        width = 32
        height = 8
        horizontal_fov = 105.0
        near_plane = 0.05
        far_plane = 8.0
        position = (0.42, 0.0, 0.0)
        rotation = (0.0, 0.0, 0.0, 1.0)
        body_name = "base_link"
        add_noise = False
        noise_std = 0.025
        dropout_probability = 0.015
        quantization = 0.01

    class commands(RotunbotVelSRU50SafeYawResidualV62TransitionCfg.commands):
        resample_commands = False
        random_start_yaw = False
        goal_distance = (0.5, 1.0)
        goal_bearing = (-0.17453292519943295, 0.17453292519943295)
        goal_radius = 0.35
        maximum_goal_distance = 8.0
        # Policy actions are sampled at 5 Hz and held for ten 50 Hz updates.
        upper_level_command_frequency_hz = 5.0
        feasible_transition_manager_enabled = True
        smooth_profile_fraction = 0.0
        random_walk_profile_fraction = 0.0

    class rewards(RotunbotVelSRU50SafeYawResidualV62TransitionCfg.rewards):
        only_positive_rewards = False
        class scales(RotunbotVelSRU50SafeYawResidualV62TransitionCfg.rewards.scales):
            # Navigation owns the objective.  V62 tracking rewards are not
            # active because the SRU command is the target, not a reference
            # supplied by a separate velocity-tracking policy.
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            curvature_tracking = 0.0
            angular_tracking_error = 0.0
            angular_acceleration_error = 0.0
            yaw_wrong_direction = 0.0
            yaw_direction = 0.0
            linear_wrong_direction = 0.0
            lateral_velocity = 0.0
            stationary_yaw = 0.0
            straight_yaw = 0.0
            action_saturation = 0.0
            dof_pos_limits = 0.0
            torques = 0.0
            # These are physical metres and one-step rewards are multiplied
            # by dt in LeggedRobot.  Use navigation-scale weights so that a
            # 2--3 m goal produces a learnable return over a 30 s episode.
            goal_progress = 20.0
            goal_reach = 50.0
            collision = -50.0
            goal_turn_alignment = 1.0
            goal_speed_alignment = 2.0
            goal_kinematic_recovery = 3.0
            action_rate = -0.01
            residual_action = 0.0

    class evaluation:
        target_error_threshold = 0.35
        stop_velocity_threshold = 0.10


class RotunbotDirectVelocityCfgPPO(
    RotunbotVelSRU50SafeYawResidualV62TransitionCfgPPO
):
    runner_class_name = "DWLOnPolicyRunner"

    class policy(RotunbotVelSRU50SafeYawResidualV62TransitionCfgPPO.policy):
        policy_class_name = "ActorCriticDirectVelocity"
        depth_height = 8
        depth_width = 32
        proprio_dim = 12
        goal_dim = 2
        previous_command_dim = 2
        encoder_dim = 64
        attention_heads = 4
        hidden_dim = 128
        init_noise_std = 0.20
        min_noise_std = 0.05
        max_noise_std = 0.80
        actor_hidden_dims = [256, 128]
        critic_hidden_dims = [256, 128]

    class runner(RotunbotVelSRU50SafeYawResidualV62TransitionCfgPPO.runner):
        policy_class_name = "ActorCriticDirectVelocity"
        algorithm_class_name = "PPODWL"
        experiment_name = "rotunbot_sru_direct_velocity"
        max_iterations = 15000
        save_interval = 50
