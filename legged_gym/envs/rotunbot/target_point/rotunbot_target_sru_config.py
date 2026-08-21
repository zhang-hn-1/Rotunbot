"""SRU policy configs for the Rotunbot paper-reproduction task.

Same environment as ``rotunbot_target_repro`` (19-D x 20 frames, 2-D action,
DIRECT_VP_TORQUE executor); only the policy network changes:

  * ``RotunbotTargetSRUCfgPPO``    -- 方案 A: ActorCriticSRULH directly
        outputs the 2-D action (SRU memory encoder replaces the DWL-CNN).
  * ``RotunbotTargetSRUModCfgPPO`` -- 方案 B: ActorCriticSRUModulate keeps the
        frozen accepted DWL policy (uniform 4150) and adds an SRU residual
        modulation on its 2-D action mean.

Register with the task registry (see envs/__init__.py) and launch with
``--task rotunbot_target_sru`` / ``--task rotunbot_target_sru_mod``.
"""

from legged_gym.envs.rotunbot.target_point.rotunbot_target_repro_config import (
    RotunbotTargetReproCfg,
    RotunbotTargetReproCfgPPO,
)

# Checkpoint of the accepted baseline (uniform 4150, new paper protocol).
SRU_BASE_CHECKPOINT = (
    "{LEGGED_GYM_ROOT_DIR}/logs/rotunbot_target_repro/"
    "Aug16_02-57-06_uniform_t1_long500_from3809/model_4150.pt"
)


class RotunbotTargetSRUCfg(RotunbotTargetReproCfg):
    """Environment identical to the accepted repro task (SRU policy only)."""


class _SRUCommonPolicy:
    """Shared SRU hyperparameters for both integration modes.

    in_channels = frame_stack (20), num_proprio_obs = num_single_obs (19)
    is derived by the runner from the environment, so the flattened actor
    observation stays 20 x 19 = 380 -- identical to the DWL-CNN policy.
    """

    in_channels = RotunbotTargetReproCfg.env.frame_stack  # 20
    sru_hidden_size = 128
    sru_memory_size = 32
    sru_num_layers = 1
    spatial_feature_mode = "rotunbot_18d"
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"
    init_noise_std = 0.3
    min_noise_std = 0.15
    max_noise_std = 0.3


class RotunbotTargetSRUCfgPPO(RotunbotTargetReproCfgPPO):
    """方案 A: SRU directly controls the robot (2-D action unchanged)."""

    seed = 11
    runner_class_name = "DWLOnPolicyRunner"

    class policy(_SRUCommonPolicy):
        pass

    class algorithm(RotunbotTargetReproCfgPPO.algorithm):
        # Isolate the SRU effect: no teacher distillation in the first runs.
        teacher_path = None
        distill_weight = 0.0

    class runner(RotunbotTargetReproCfgPPO.runner):
        policy_class_name = "ActorCriticSRULH"
        algorithm_class_name = "PPODWL"
        max_iterations = 50
        save_interval = 10
        experiment_name = "rotunbot_target_sru"
        run_name = "sru_direct_from4150"
        resume = False
        load_optimizer = False
        load_run = None
        checkpoint = -1


class RotunbotTargetSRUModCfgPPO(RotunbotTargetReproCfgPPO):
    """方案 B: frozen uniform-4150 base + SRU residual modulation.

    The SRU modulator starts from zero-ish random weights; the base policy is
    frozen, so the first rollouts behave exactly like the accepted model and
    training only shapes the residual.  Set ``base_trainable = True`` to
    jointly fine-tune the base after the residual has learned.
    """

    seed = 11
    runner_class_name = "DWLOnPolicyRunner"

    class policy(_SRUCommonPolicy):
        base_path = SRU_BASE_CHECKPOINT
        base_trainable = False
        mod_hidden_dims = [256, 128]

    class algorithm(RotunbotTargetReproCfgPPO.algorithm):
        teacher_path = None
        distill_weight = 0.0

    class runner(RotunbotTargetReproCfgPPO.runner):
        policy_class_name = "ActorCriticSRUModulate"
        algorithm_class_name = "PPODWL"
        max_iterations = 50
        save_interval = 10
        experiment_name = "rotunbot_target_sru"
        run_name = "sru_modulate_from4150"
        resume = False
        load_optimizer = False
        load_run = None
        checkpoint = -1
