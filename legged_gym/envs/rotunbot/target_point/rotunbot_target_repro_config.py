"""Paper-reproduction conditions with the existing 19-D DWL-CNN policy unchanged."""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO

from .rotunbot_target_lh_config import RotunbotTargetLHCfg


class RotunbotTargetReproCfg(RotunbotTargetLHCfg):
    """Flat, obstacle-free point-to-point task for code reproduction."""

    class env(RotunbotTargetLHCfg.env):
        # Keep the Graduation training batch/environment setting.  The number
        # of parallel environments does not change the per-episode test rule.
        # Isaac Gym Preview 4 does not expose the GPU aggregate-pair capacity
        # requested by 2048 colocated environments.  At 1024 environments the
        # required broadphase pairs fit, avoiding silently missed contacts.
        num_envs = 1024
        num_actions = 2
        num_single_obs = 19
        frame_stack = 20
        short_frame_stack = 5
        c_frame_stack = 3
        num_observations = frame_stack * num_single_obs
        num_privileged_obs = c_frame_stack * RotunbotTargetLHCfg.env.single_num_privileged_obs
        episode_length_s = 60

    class terrain(RotunbotTargetLHCfg.terrain):
        # The requested reproduction is the planar, no-obstacle experiment.
        mesh_type = "plane"
        measure_heights = False
        curriculum = False

    class control(RotunbotTargetLHCfg.control):
        # Reproduction control period is dt=0.02 s.  LeggedRobot uses
        # cfg.sim.dt * cfg.control.decimation as the policy period.
        # Keep the checkpoint's 2-D action interface.  The velocity command
        # is tracked by an explicit torque law so its executor matches the
        # original R-controller dynamics.
        control_type = "DIRECT_VP_TORQUE"
        decimation = 1
        rate_limit_1 = 0.02
        rate_limit_2 = 0.04
        first_vel_limits = 3.0
        second_pos_limits = 0.45
        torque_limits_1 = 100.0
        torque_limits_2 = 100.0
        direct_velocity_scale = 1.0
        direct_velocity_limit = 3.0
        direct_position_scale = 0.5
        direct_position_limit = 0.45
        direct_drive_gain_scale = 1.0
        # Safe continuation from the old R-controller checkpoint.
        direct_use_rate_limit = True
        direct_velocity_rate_limit = 0.02
        direct_position_rate_limit = 0.04

    class commands(RotunbotTargetLHCfg.commands):
        # Compatibility aliases retained for inherited LH code.  The formal
        # The reproduction success check in rotunbot_target_repro.py reads evaluation.* below.
        random_start_yaw = True
        stop_distance = 0.20
        stop_vel = 0.1

        # Nominal-metric stage uses the same strict 0.20 m success radius as
        # evaluation.  The resumed checkpoint is stuck at a 0.40 m curriculum
        # radius because noisy rollout success never reaches the 80% gate.
        target_curriculum = False
        curriculum_success_distance_start = 1.0
        curriculum_success_distance_min = 0.20
        curriculum_success_distance_step = 0.20
        target_curriculum_window = 2048
        target_curriculum_success_rate = 0.80

        # Preserve the full-map distribution while increasing exposure to the
        # lateral 1--4 m region that dominates seed-11 F1/F4 failures.
        hard_side_target_probability = 0.35
        hard_side_distance_min = 1.0
        hard_side_distance_max = 4.0
        hard_side_bearing_min_deg = 60.0
        hard_side_bearing_max_deg = 110.0

        class ranges(RotunbotTargetLHCfg.commands.ranges):
            pos_x = [-5.0, 5.0]
            pos_y = [-5.0, 5.0]

    class asset(RotunbotTargetLHCfg.asset):
        # Keep the Graduation robot model; it is mechanically equivalent for
        # this comparison and keeps the trained policy/model pairing intact.
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/Rotunbot/urdf/Rotunbot_test2.urdf"

    class init_state(RotunbotTargetLHCfg.init_state):
        randomize_initial_velocity = False

    class noise(RotunbotTargetLHCfg.noise):
        # Keep the Graduation training observation-noise setting. play.py
        # disables it during the clean paper-style evaluation.
        add_noise = True

    class domain_rand(RotunbotTargetLHCfg.domain_rand):
        # Nominal-metric stage: retain mild friction and latency variation so
        # the accepted controller is not forgotten, but stop spending policy
        # updates on active push recovery.
        randomize_friction = True
        randomize_base_mass = False
        push_robots = False
        push_interval_s = 3.0
        max_push_vel_xy = 0.3

    class latency:
        # Stage 1 of paper-style latency adaptation.  At 50 Hz, two steps are
        # 0.04 s.  Later stages can raise these maxima to five steps (0.1 s)
        # after a checkpoint passes the clean and delayed paired evaluations.
        enabled = True
        min_observation_steps = 0
        max_observation_steps = 2
        min_action_steps = 0
        max_action_steps = 2

    class rewards(RotunbotTargetLHCfg.rewards):
        debug_print = False
        only_positive_rewards = False
        close_para = 1.0
        # A narrow distance-shaping scale is necessary at the initial
        # [-1, 1] curriculum stage.  sigma=8 makes the distance reward almost
        # constant, so a stationary policy can receive reward without moving.
        tracking_sigma_main = 0.5
        progress_target_speed = 0.6
        soft_dof_pos_limit = 1.0
        stop_reward_multiplier = 1.0

        class scales(RotunbotTargetLHCfg.rewards.scales):
            # Paper-inspired point-to-point shaping.  The asymmetric
            # close/away pair is disabled because it can make a loop around
            # the target have positive net reward.
            termination = -0.0
            close_to_target = 0.0
            away_to_target = 0.0
            approaching_target = 0.5
            to_target = 1.5
            stop = 20.0
            balance = 0.1
            torques = -1.0e-5
            action_rate = -0.004
            time = -0.5
            overturn = -0.5
            lin_vel_x_limit = -1.0
            ang_vel_z_limit = -0.2

            # Additional small braking term to reduce overspeed inside the
            # formal 0.20 m stopping region.
            near_goal_speed = -0.2

            # Graduation-only terms are disabled in this paper-inspired run.
            ang_vel_xy = 0.0
            lin_vel_z = 0.0
            dof_acc = 0.0
            dof_pos_limits = 0.0


    class evaluation:
        # Evaluation protocol requested for the paper-style comparison.
        # The paper explicitly reports 40 repeated trials per method and
        # environment in the real-robot experiments.
        num_eval_episodes = 40
        target_error_threshold = 0.20
        stop_velocity_threshold = 0.10


class RotunbotTargetReproCfgPPO(LeggedRobotCfgPPO):
    """Checkpoint-compatible PPO/DWL-CNN settings for continued training.

    The paper's exact actor uses a Transformer LH Encoder; this class keeps
    the existing 19-D DWL-CNN so its checkpoints remain loadable.
    """

    seed = 11
    # Keep the existing DWL runner so the 19-D checkpoint remains loadable.
    runner_class_name = "DWLOnPolicyRunner"

    class policy:
        init_noise_std = 0.3
        min_noise_std = 0.15
        # The resumed checkpoint carries std=1.5.  Under frequent pushes this
        # made the three-update probe destructive; clamp exploration before
        # retrying the same disturbance curriculum.
        max_noise_std = 0.3
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        kernel_size = [3, 2]
        filter_size = [16, 8]
        stride_size = [1, 1]
        lh_output_dim = 16
        in_channels = RotunbotTargetReproCfg.env.frame_stack
        # Keep the old encoder/policy fixed initially and learn only a
        # 2-D identity-initialized action remapping for DIRECT_VP.
        action_adapter_enabled = False
        action_adapter_train_only = False

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.002
        num_learning_epochs = 5
        num_mini_batches = 16
        # Conservative fine-tuning from model_3800.  Earlier 1e-3 probes
        # moved the policy too far in only a few updates.
        # Strict-success precision fine-tuning uses a fresh optimizer, so the
        # resumed checkpoint cannot silently restore its old, larger step.
        learning_rate = 5.0e-5
        schedule = "fixed"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCriticDWL"
        algorithm_class_name = "PPODWL"
        num_steps_per_env = 96
        max_iterations = 3
        save_interval = 1
        experiment_name = "rotunbot_target_repro"
        run_name = "nominal_strict020_hardside35_seed11_stage2"
        # Continue from the existing checkpoint without changing the policy
        # input/output dimensions.
        resume = True
        load_optimizer = True
        load_run = "Aug14_21-59-52_nominal_strict020_hardside35_seed11_from3809"
        checkpoint = 3812
        resume_path = None
