"""Depth-camera obstacle-navigation configuration built from target_repro."""

from .rotunbot_target_repro_config import (
    RotunbotTargetReproCfg,
    RotunbotTargetReproCfgPPO,
)


class RotunbotTargetDepthCfg(RotunbotTargetReproCfg):
    """Mapless target navigation with static collision geometry and depth input."""

    # Render the actual Isaac Gym depth camera in headless mode.  The policy
    # must see the same sensor modality during training and deployment; the
    # deterministic ray/AABB fallback remains available for diagnostics.
    enable_camera_sensors_in_headless = True

    class env(RotunbotTargetReproCfg.env):
        # Camera rendering is substantially more expensive than blind RL.
        num_envs = 16
        frame_stack = 8
        short_frame_stack = 2
        c_frame_stack = 4
        # Compact depth input shared by the simulator sensor, preview and policy.
        # The lower resolution keeps the attention encoder within the available
        # GPU memory while retaining the maze-wall structure needed for control.
        depth_height = 32
        depth_width = 64
        depth_dim = depth_height * depth_width
        proprio_dim = 19
        num_single_obs = proprio_dim + depth_dim
        num_observations = frame_stack * num_single_obs
        # Critic-only maze information: normalized wall clearance and the
        # current collision flag.  The actor remains depth + proprioception.
        single_num_privileged_obs = proprio_dim + 2
        num_privileged_obs = c_frame_stack * single_num_privileged_obs
        episode_length_s = 60

    class control(RotunbotTargetReproCfg.control):
        # Keep the same motor interface while reducing camera/ policy rate to
        # 25 Hz; the physics step remains 50 Hz.
        decimation = 2

    class commands(RotunbotTargetReproCfg.commands):
        # Keep the maze layout fixed, but sample a valid road-center goal on
        # every reset and randomize the initial yaw.
        target_curriculum = False
        random_start_yaw = True

        class ranges(RotunbotTargetReproCfg.commands.ranges):
            # The inherited target command bounds remain wide enough for all
            # valid maze-cell-center goals.
            pos_x = [-10.0, 10.0]
            pos_y = [-10.0, 10.0]

    class obstacles:
        # (x, y) centers and (width, depth, height) box dimensions in metres.
        centers = ((2.0, 1.5), (-1.5, -2.0), (-2.8, 2.6), (2.8, -2.7))
        sizes = ((0.8, 3.0, 1.2), (3.0, 0.8, 1.2), (0.8, 2.2, 1.2), (0.8, 2.0, 1.2))
        robot_collision_radius = 0.45
        goal_clearance = 0.70
        safety_clearance = 1.10
        terminate_on_collision = True
        colors = (
            (0.75, 0.20, 0.08),
            (0.08, 0.32, 0.75),
            (0.75, 0.55, 0.08),
            (0.45, 0.12, 0.60),
        )

    class maze:
        # Reuse the existing rotunbot_target_depth task name while replacing
        # its four boxes with the deterministic procedural maze.
        enabled = True
        grid_size = (15, 15)
        cell_size = 2.0
        wall_height = 1.5
        center_clearance_radius = 2
        seed = 0
        start_cell = (7, 7)
        # Candidate goals must be at least this far from the fixed start.
        min_goal_distance = 8.0
        robot_collision_radius = 0.4
        safety_clearance = 0.8
        terminate_on_collision = True
        wall_color = (0.32, 0.38, 0.48)
        visualize_start_goal = True
        marker_radius = 0.18

    class init_state(RotunbotTargetReproCfg.init_state):
        # Fixed initial pose and zero root velocity at every reset.
        rot = [0.0, 0.0, 0.0, 1.0]
        randomize_initial_velocity = False

    class camera:
        enable = True
        # Training policy input.  The evaluator can override this at runtime
        # for fallback-vs-camera comparisons without adding another task.
        policy_source = "fallback"
        width = 64
        height = 32
        horizontal_fov = 105.0
        near_plane = 0.05
        far_plane = 8.0
        # Camera pose expressed in the base_link frame.
        position = (0.42, 0.0, 0.0)
        rotation = (0.0, 0.0, 0.0, 1.0)
        # Isaac Gym camera tensors require a graphics device; if none is
        # available, the task uses the same ray/AABB depth model as fallback.
        allow_headless_fallback = True
        # Runtime diagnostic flag; the evaluator enables this explicitly.
        capture_fallback = False
        add_noise = True
        noise_std = 0.025
        dropout_probability = 0.015
        quantization = 0.01

    class noise(RotunbotTargetReproCfg.noise):
        add_noise = True
        noise_level = 0.15

    class rewards(RotunbotTargetReproCfg.rewards):
        only_positive_rewards = False
        # Normalize one control-step change in goal distance before applying
        # the existing approaching_target reward scale.
        progress_normalization = 0.05
        # The maze shortest-path term handles turns and detours; retain a
        # small Euclidean component for smooth progress inside the goal cell.
        maze_euclidean_progress_weight = 0.25
        stall_speed_threshold = 0.4
        # Match the maze safety clearance.  Low-speed turning is exempt when
        # the robot is within this distance of a wall.
        stall_clearance_threshold = 0.8
        # At the 25 Hz policy rate, eight cycles are about 0.32 seconds.
        stall_progress_window = 8

        class scales(RotunbotTargetReproCfg.rewards.scales):
            collision = -12.0
            # The previous progress signal was too small compared with the
            # constant time cost, so the policy learned to stop safely after
            # reaching the first obstacle.  Strengthen path progress while
            # retaining the existing collision and clearance penalties.
            approaching_target = 1.5
            # The base reward scales are multiplied by dt.  Increase this
            # term so the wall-distance gradient is useful before collision.
            obstacle_clearance = -4.0
            time = -0.05
            # Soft anti-stall term; it is inactive near walls and near the
            # goal, so low-speed turns and final braking remain possible.
            stall_far_from_goal = -0.10

    class evaluation(RotunbotTargetReproCfg.evaluation):
        # Fixed-maze success is defined as reaching within one metre and then
        # stopping.  The terminal stop speed threshold remains 0.10 m/s.
        target_error_threshold = 1.0


class RotunbotTargetDepthCfgPPO(RotunbotTargetReproCfgPPO):
    class policy(RotunbotTargetReproCfgPPO.policy):
        actor_hidden_dims = [256, 128]
        critic_hidden_dims = [256, 128]
        # The camera run drove the learned exploration std to its upper bound,
        # causing most rollout episodes to collide.  Keep exploration useful
        # without allowing near-random motor commands.
        init_noise_std = 0.30
        min_noise_std = 0.10
        max_noise_std = 0.80
        in_channels = RotunbotTargetDepthCfg.env.frame_stack
        depth_dim = RotunbotTargetDepthCfg.env.depth_dim
        proprio_dim = RotunbotTargetDepthCfg.env.proprio_dim
        depth_height = RotunbotTargetDepthCfg.env.depth_height
        depth_width = RotunbotTargetDepthCfg.env.depth_width
        encoder_dim = 64
        attention_heads = 4
        hidden_dim = 128

    class algorithm(RotunbotTargetReproCfgPPO.algorithm):
        # A smaller entropy bonus prevents the action std from growing to the
        # maximum throughout the depth-camera run.
        entropy_coef = 0.003
        num_learning_epochs = 5
        # Keep the effective global minibatch near 512 when using 256 total
        # environments across four DDP ranks (256 * 64 / 32 / 4 = 128 local
        # samples per rank and 512 samples after gradient synchronization).
        # With 128 total environments and four DDP ranks this gives 256 local
        # samples per minibatch, instead of 64, while retaining five epochs.
        num_mini_batches = 8
        learning_rate = 3.0e-4
        # The inherited adaptive schedule changes the learning rate once per
        # PPO minibatch.  With depth observations and distributed training it
        # can grow far above the configured value and then collapse to 1e-5.
        # Keep the update scale stable for this task.
        schedule = "fixed"

    class runner(RotunbotTargetReproCfgPPO.runner):
        policy_class_name = "ActorCriticDepth"
        algorithm_class_name = "PPODWL"
        # Reduce rollout memory for depth-image training.
        num_steps_per_env = 64
        max_iterations = 10000
        save_interval = 50
        experiment_name = "rotunbot_target_depth"
        resume = True
        load_run = "Aug17_12-03-12_maze_depth_geodesic_v4"
        checkpoint = 5350
        run_name = "maze_depth_geodesic_v4_continue_fallback_rtx4070"
        # Preserve optimizer moments when continuing this same depth policy.
        # A blind-policy fine-tune can still explicitly disable this in the
        # runner configuration before loading.
        load_optimizer = True
