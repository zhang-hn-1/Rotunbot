"""Configuration for the V0 depth-aware local executor."""

from .rotunbot_maze_config import RotunbotMazeCfg, RotunbotMazeCfgPPO


class RotunbotMazeLocalDepthCfg(RotunbotMazeCfg):
    # Headless viewer suppression and offscreen camera rendering are separate
    # concerns.  Keep graphics enabled for the explicit real-camera smoke.
    enable_camera_sensors_in_headless = True

    class env(RotunbotMazeCfg.env):
        num_envs = 16
        num_actions = 2
        num_single_obs = 272
        num_short_obs = 272
        frame_stack = 1
        short_frame_stack = 1
        c_frame_stack = 1
        num_observations = 272
        num_privileged_obs = 18
        single_num_privileged_obs = 18
        depth_height = 8
        depth_width = 32
        depth_dim = 256
        state_dim = 16
        episode_length_s = 30.0

    class terrain(RotunbotMazeCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False

    class maze(RotunbotMazeCfg.maze):
        enabled = False
        terminate_on_collision = True
        safety_clearance = 0.8

    class commands(RotunbotMazeCfg.commands):
        resample_commands = False
        random_start_yaw = True
        local_curriculum_stage = 0
        local_goal_distance = (0.4, 1.5)
        local_goal_lateral = (-0.6, 0.6)
        local_waypoint_radius = 0.25
        global_goal_radius = 0.35
        stop_vel = 0.1
        distance_limit = (0.25, 2.0)
        lateral_limit = 0.8
        minimum_forward_component = 0.15
        bearing_limit_deg = 120.0
        class ranges(RotunbotMazeCfg.commands.ranges):
            pos_x = [-10.0, 10.0]
            pos_y = [-10.0, 10.0]

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
        add_noise = True
        noise_std = 0.025
        dropout_probability = 0.015
        quantization = 0.01

    class rewards(RotunbotMazeCfg.rewards):
        only_positive_rewards = False
        class scales(RotunbotMazeCfg.rewards.scales):
            local_progress = 3.0
            local_reach = 20.0
            wall_penalty = 0.5
            collision = -20.0
            action_rate = -0.01

    class evaluation:
        target_error_threshold = 0.35
        stop_velocity_threshold = 0.1


class RotunbotMazeLocalDepthCfgPPO(RotunbotMazeCfgPPO):
    class policy(RotunbotMazeCfgPPO.policy):
        policy_class_name = "ActorCriticDepthLocal"
        depth_height = RotunbotMazeLocalDepthCfg.env.depth_height
        depth_width = RotunbotMazeLocalDepthCfg.env.depth_width
        init_noise_std = 0.3
        min_noise_std = 0.1
        max_noise_std = 0.8
        actor_hidden_dims = [256, 128]
        critic_hidden_dims = [256, 128]
        activation = "elu"

    class algorithm(RotunbotMazeCfgPPO.algorithm):
        entropy_coef = 0.003
        num_learning_epochs = 2
        num_mini_batches = 1

    class runner(RotunbotMazeCfgPPO.runner):
        policy_class_name = "ActorCriticDepthLocal"
        algorithm_class_name = "PPODWL"
        num_steps_per_env = 32
        max_iterations = 15000
        save_interval = 50
        experiment_name = "rotunbot_maze_local_depth"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
