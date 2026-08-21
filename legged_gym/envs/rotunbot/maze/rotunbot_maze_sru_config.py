"""SRU policy config for the procedural-maze Rotunbot task.

Same maze environment (15x15 grid, 2 m cells) but with frame-stacked
observations (20 x 19) so the SRU memory encoder can scan history.  The maze
frame uses Euler angles (layout in rotunbot_maze_sru.py), so the SRU policy
uses ``spatial_feature_mode="rotunbot_maze_19d"``.
"""

from .rotunbot_maze_config import RotunbotMazeCfg, RotunbotMazeCfgPPO


class RotunbotMazeSRUCfg(RotunbotMazeCfg):
    class env(RotunbotMazeCfg.env):
        num_envs = 128
        num_actions = 2
        num_single_obs = 19
        frame_stack = 20
        short_frame_stack = 5
        num_observations = int(frame_stack * num_single_obs)  # 380
        num_privileged_obs = None
        # Longer horizon for long maze paths; tune if episodes time out.
        episode_length_s = 60

    class maze(RotunbotMazeCfg.maze):
        # Collision ends the episode so the policy learns to avoid walls.
        terminate_on_collision = True


class RotunbotMazeSRUCfgPPO(RotunbotMazeCfgPPO):
    """SRU (方案 A style) directly controls the maze robot."""

    seed = 11
    runner_class_name = "DWLOnPolicyRunner"

    class policy:
        in_channels = RotunbotMazeSRUCfg.env.frame_stack  # 20
        sru_hidden_size = 128
        sru_memory_size = 32
        sru_num_layers = 1
        spatial_feature_mode = "rotunbot_maze_19d"
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        init_noise_std = 0.3
        min_noise_std = 0.15
        max_noise_std = 0.3

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
        max_iterations = 200
        save_interval = 20
        experiment_name = "rotunbot_maze_sru"
        run_name = "sru_maze"
        resume = False
        load_optimizer = False
