"""Configuration for the explicit Robot-frame Local P2P task."""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO

from ..target_point.rotunbot_target_repro_config import RotunbotTargetReproCfg


class RotunbotLocalGoalCfg(RotunbotTargetReproCfg):
    class env(RotunbotTargetReproCfg.env):
        num_envs = 2048
        num_actions = 2
        num_single_obs = 17
        frame_stack = 1
        short_frame_stack = 1
        c_frame_stack = 1
        single_num_privileged_obs = 17
        num_observations = 17
        num_privileged_obs = 17
        episode_length_s = 6.0

    class commands(RotunbotTargetReproCfg.commands):
        num_commands = 2
        resampling_time = 30.0
        resample_commands = False
        random_start_yaw = True
        target_curriculum = False
        local_curriculum_stage = "A"
        local_goal_radius = 0.35
        local_goal_max_distance = 3.0

        class stage_a:
            distance = (0.5, 2.0)
            bearing_deg = (-45.0, 45.0)

        class stage_b:
            distance = (0.5, 2.5)
            bearing_deg = (-90.0, 90.0)

        class stage_c:
            distance = (0.5, 3.0)
            bearing_deg = (-180.0, 180.0)

    class noise(RotunbotTargetReproCfg.noise):
        add_noise = False

    class rewards(RotunbotTargetReproCfg.rewards):
        only_positive_rewards = False

        class scales(RotunbotTargetReproCfg.rewards.scales):
            termination = 0.0
            close_to_target = 0.0
            away_to_target = 0.0
            approaching_target = 0.0
            to_target = 0.0
            stop = 0.0
            balance = 0.0
            torques = 0.0
            action_rate = 0.0
            time = 0.0
            overturn = 0.0
            lin_vel_x_limit = 0.0
            ang_vel_z_limit = 0.0
            near_goal_speed = 0.0
            local_progress = 1.0
            local_reach = 5.0
            local_time = -0.01
            local_action_smooth = -0.001

    class evaluation:
        target_error_threshold = 0.35
        stop_velocity_threshold = 0.10


class RotunbotLocalGoalCfgPPO(LeggedRobotCfgPPO):
    seed = 3
    runner_class_name = "OnPolicyRunner"

    class policy(LeggedRobotCfgPPO.policy):
        # The local-goal task is sensitive to lateral action sign.  Keep the
        # initial exploration bounded so PPO does not spend the rollout on
        # saturated opposite-direction actions.
        init_noise_std = 0.20
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        num_mini_batches = 4
        learning_rate = 3.0e-4

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        run_name = ""
        experiment_name = "rotunbot_local_goal_p2p"
        num_steps_per_env = 96
        max_iterations = 10000
        save_interval = 50
        resume = False
        load_run = -1
        checkpoint = -1
