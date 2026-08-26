"""Empty-map local-waypoint fine-tuning configuration."""

from ..target_point.rotunbot_target_repro_config import (
    RotunbotTargetReproCfg,
    RotunbotTargetReproCfgPPO,
)


class RotunbotLocalP2PCfg(RotunbotTargetReproCfg):
    class env(RotunbotTargetReproCfg.env):
        num_envs = 64
        episode_length_s = 6.0

    class commands(RotunbotTargetReproCfg.commands):
        # Stage A: learn short, mostly forward waypoints before widening the
        # bearing distribution in later fine-tuning runs.
        local_curriculum_stage = 1
        local_waypoint_radius = 0.35
        local_distance_stage_a = [1.0, 3.0]
        local_bearing_stage_a = [-60.0, 60.0]
        local_distance_stage_b = [0.5, 3.0]
        local_bearing_stage_b = [-120.0, 120.0]
        local_distance_stage_c = [0.5, 3.0]
        local_bearing_stage_c = [-180.0, 180.0]

    class rewards(RotunbotTargetReproCfg.rewards):
        # Preserve negative feedback.  The fine-tune reward has only local
        # waypoint responsibilities; no global-goal or collision shaping.
        only_positive_rewards = False

        class scales:
            local_progress = 1.0
            local_reach = 5.0
            local_time = -0.01
            local_action_smooth = -0.001


class RotunbotLocalP2PCfgPPO(RotunbotTargetReproCfgPPO):
    class runner(RotunbotTargetReproCfgPPO.runner):
        experiment_name = "rotunbot_local_p2p"
        max_iterations = 1000
        save_interval = 50

