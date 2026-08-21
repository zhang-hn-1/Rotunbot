"""SRU policy configs for the procedural-maze Rotunbot task.

Same maze environment (15x15 grid, 2 m cells) but with frame-stacked
repro-style observations (20 x 35 = 19 base + 16 wall rays) so both the SRU
memory encoder and (for 方案 B) the frozen uniform-4150 DWL base on its
19-D slice can be used.

  * ``RotunbotMazeSRUCfgPPO``     -- 方案 A style: SRU directly controls.
  * ``RotunbotMazeSRUModCfgPPO``  -- 方案 B style: frozen 4150 base on the
        19-D slice + SRU residual modulation on the full 35-D frame.
"""

from .rotunbot_maze_config import RotunbotMazeCfg, RotunbotMazeCfgPPO


class RotunbotMazeSRUCfg(RotunbotMazeCfg):
    class env(RotunbotMazeCfg.env):
        num_envs = 128
        num_actions = 2
        num_single_obs = 35  # 19 base + 16 wall-ray distances
        frame_stack = 20
        short_frame_stack = 5
        num_observations = int(frame_stack * num_single_obs)  # 700
        num_privileged_obs = int(frame_stack * num_single_obs)  # 700 (critic = same stacked obs)
        # Longer horizon for long maze paths; tune if episodes time out.
        episode_length_s = 120

    class normalization(RotunbotMazeCfg.normalization):
        # LH / paper-reproduction obs scales so the first 19 channels are
        # byte-compatible with the accepted flat-plane policy (4150).
        class obs_scales:
            command = 1.0
            lin_vel = 1.0
            ang_vel = 0.5
            quat = 1.0
            dof_pos = 2.0
            dof_vel = 1.0
            pos = 0.2
            height_measurements = 5.0

    class maze(RotunbotMazeCfg.maze):
        # Collisions are penalized by the reward but do NOT end the episode,
        # so the policy can learn to recover from wall bumps (stage 1 used
        # terminate_on_collision=True and stuck at 0% success).
        terminate_on_collision = False


class _MazeSRUCommonPolicy:
    in_channels = RotunbotMazeSRUCfg.env.frame_stack  # 20
    sru_hidden_size = 128
    sru_memory_size = 32
    sru_num_layers = 1
    spatial_feature_mode = "rotunbot_18d"  # quaternion layout, matches base
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"
    init_noise_std = 0.3
    min_noise_std = 0.15
    max_noise_std = 0.3


class RotunbotMazeSRUCfgPPO(RotunbotMazeCfgPPO):
    """方案 A style: SRU directly controls the maze robot (from scratch)."""

    seed = 11
    runner_class_name = "DWLOnPolicyRunner"

    class policy(_MazeSRUCommonPolicy):
        pass

    class algorithm(RotunbotMazeCfgPPO.algorithm):
        learning_rate = 5.0e-5
        schedule = "fixed"
        entropy_coef = 0.002
        num_learning_epochs = 5
        num_mini_batches = 8

    class runner(RotunbotMazeCfgPPO.runner):
        policy_class_name = "ActorCriticSRULH"
        algorithm_class_name = "PPODWL"
        num_steps_per_env = 96
        max_iterations = 300
        save_interval = 25
        experiment_name = "rotunbot_maze_sru"
        run_name = "sru_maze_direct"
        resume = False
        load_optimizer = False


class RotunbotMazeSRUModCfgPPO(RotunbotMazeCfgPPO):
    """方案 B style: frozen uniform-4150 base on the 19-D slice + SRU
    residual modulation on the full 35-D frame (base + wall rays).

    Zero-initialized modulator: training starts from the flat-plane policy's
    behavior and the SRU learns wall avoidance / maze rerouting on top.
    """

    seed = 11
    runner_class_name = "DWLOnPolicyRunner"

    class policy(_MazeSRUCommonPolicy):
        base_path = (
            "{LEGGED_GYM_ROOT_DIR}/logs/rotunbot_target_repro/"
            "Aug16_02-57-06_uniform_t1_long500_from3809/model_4150.pt"
        )
        base_trainable = False
        base_proprio_obs = 19  # base consumes channels 0:19 of each frame
        mod_hidden_dims = [256, 128]

    class algorithm(RotunbotMazeCfgPPO.algorithm):
        learning_rate = 5.0e-5
        schedule = "fixed"
        entropy_coef = 0.002
        num_learning_epochs = 5
        num_mini_batches = 8
        teacher_path = None
        distill_weight = 0.0

    class runner(RotunbotMazeCfgPPO.runner):
        policy_class_name = "ActorCriticSRUModulate"
        algorithm_class_name = "PPODWL"
        num_steps_per_env = 96
        max_iterations = 400
        save_interval = 25
        experiment_name = "rotunbot_maze_sru"
        run_name = "sru_maze_mod4150_stage2"
        # Continue stage-1 model_300 with the relaxed env (no collision
        # termination, 120 s episodes).
        resume = True
        load_optimizer = False
        load_run = "Aug21_23-44-54_sru_maze_mod4150"
        checkpoint = 300
