"""Configuration for the Rotunbot low-level ``(v, yaw-rate)`` tracker.

This task deliberately contains no navigation, depth, map, or target-point
logic. A measured nominal inverse map converts each feasible command into the
two physical Rotunbot actuator targets. Measured velocity error supplies a
bounded braking correction, and the policy outputs only a small residual:

``nominal + feedback + residual_scale * policy_action``
"""

from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class RotunbotVelCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2048
        num_actions = 2
        # command(2), body linear velocity(3), body angular velocity(3),
        # projected gravity(3), joint-2 position(1), joint velocities(2),
        # previous normalized policy action(2)
        num_observations = 16
        num_privileged_obs = None
        env_spacing = 3.0
        episode_length_s = 20.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        # The generic legged-robot reset injects +/-0.5 velocity.  Start the
        # nominal tracker at rest; disturbance recovery belongs to a later
        # robustness stage.
        randomize_initial_velocity = False
        default_joint_angles = {
            "joint1": 0.0,
            "joint2": 0.0,
        }

    class sim(LeggedRobotCfg.sim):
        # Match the depth-navigation physics and 25 Hz policy interface.
        dt = 0.02

    class control(LeggedRobotCfg.control):
        control_type = "R"
        decimation = 2

        # The policy output is normalized. These are physical actuator-target
        # scales, not torque scales.
        joint1_velocity_scale = 1.0
        joint2_position_scale = 0.45
        joint1_velocity_limit = 1.0
        joint2_position_limit = 0.45

        # Residual RL: the nominal inverse already satisfies the deterministic
        # tracking contract, while PPO learns small state-dependent corrections.
        nominal_forward_speed_per_action = 0.40
        # Fixed-action measurements show that steering/yaw gain rises with
        # forward speed. These coefficients interpolate the nominal gain over
        # the trained 0.08--0.35 m/s feasible envelope.
        nominal_yaw_gain_intercept = 0.0915
        nominal_yaw_gain_speed_slope = 0.175
        # The nominal inverse is almost exact at steady state. Explicit bounded
        # velocity feedback removes the long inertial tail after abrupt command
        # changes; PPO learns only the remaining nonlinear residual.
        # Linear feedforward is already effectively exact; adding linear P
        # feedback fought the reversal governor in the first v13 sweep. Keep
        # only a deliberately mild yaw correction to damp rotational inertia.
        linear_feedback_gain = 0.0
        low_speed_linear_feedback_gain = None
        linear_feedback_transition_start_speed = 0.04
        linear_feedback_transition_full_speed = 0.08
        angular_feedback_gain = 0.20
        linear_feedback_action_limit = 0.0
        angular_feedback_action_limit = 0.15
        # Optional extra P feedback restricted by the causal target-gap mask.
        # Zero preserves every existing task.
        smooth_angular_feedback_gain = 0.0
        smooth_angular_feedback_action_limit = 0.0
        smooth_angular_feedback_minimum_command_rate = 1.0e-4
        # Optional filtered derivative feedback on velocity-tracking error.
        # Gains are time constants [s]. Zero preserves all existing tasks.
        linear_derivative_gain = 0.0
        angular_derivative_gain = 0.0
        linear_derivative_action_limit = 0.0
        angular_derivative_action_limit = 0.08
        error_derivative_filter_alpha = 0.20
        # Optional acceleration feedforward.  Zero preserves the v13 control
        # contract; the dynamic-tracking task selects calibrated non-zero values.
        linear_command_lead_time = 0.0
        angular_command_lead_time = 0.0
        # Optional larger linear lead for continuously reachable commands.
        # Zero disables it and preserves all legacy tasks.
        smooth_linear_command_lead_time = 0.0
        smooth_linear_command_minimum_rate = 1.0e-4
        # Optional independent yaw lead for continuously reachable commands.
        # Keeping this separate from ``angular_command_lead_time`` lets a task
        # retain aggressive step anticipation without over-driving a fast
        # sinusoidal reference.
        smooth_angular_command_lead_time = 0.0
        smooth_angular_command_minimum_rate = 1.0e-4
        smooth_angular_command_gain = 1.0
        require_explicit_smooth_profile_for_phase_lead = False
        # Optional actuator-space command-rate feedforward.  Unlike command
        # lead compensation this term is added after feasible-set projection,
        # so it can reduce dynamic phase lag without changing path curvature.
        linear_rate_feedforward_time = 0.0
        angular_rate_feedforward_time = 0.0
        linear_rate_feedforward_action_limit = 0.0
        angular_rate_feedforward_action_limit = 0.08
        # Infinite thresholds preserve legacy behavior. Finite values allow a
        # task to apply rate feedforward only to continuously reachable targets,
        # not to acceleration-limited steps or reversals.
        rate_feedforward_target_gap_threshold = [float("inf"), float("inf")]
        residual_action_scale = [0.05, 0.10]
        residual_error_alignment_filter = False
        disable_residual_during_braking = False
        # A residual safety projection normally checks the instantaneous
        # tracking error.  Non-zero preview times instead check the error to a
        # bounded future command, allowing a rate-aware policy to compensate
        # phase lag without permitting arbitrary error-increasing actions.
        residual_alignment_linear_preview_time = 0.0
        residual_alignment_angular_preview_time = 0.0
        # Optional leaky integral feedback removes repeatable low-frequency
        # tracking bias.  Zero gains preserve every pre-v17 controller.
        linear_integral_gain = 0.0
        angular_integral_gain = 0.0
        linear_integral_action_limit = 0.0
        angular_integral_action_limit = 0.0
        integral_leak_rate = 0.5
        linear_error_integral_limit = 0.50
        angular_error_integral_limit = 0.20
        # Infinite thresholds preserve ordinary PI. A dynamic task may use
        # finite thresholds so integral memory is active only at steady target.
        integral_command_rate_threshold = [float("inf"), float("inf")]

        # Same target slew limits as the 25 Hz depth-navigation controller.
        set_target_rate_limit = True
        joint1_target_rate_limit = 0.04
        joint2_target_rate_limit = 0.08

        joint1_velocity_kp = 35.0
        joint2_position_kp = 300.0
        joint2_velocity_kd = 150.0
        joint1_torque_limit = 100.0
        joint2_torque_limit = 100.0

    class commands(LeggedRobotCfg.commands):
        # command[:, 0] = body-frame forward velocity [m/s]
        # command[:, 1] = heading yaw rate [rad/s]
        num_commands = 2
        heading_command = False
        curriculum = False
        # ``None`` preserves the historical behavior where smooth reference
        # targets are refreshed on every low-level policy step.  SRU-compatible
        # tasks set this to 5 Hz and hold each high-level request between ticks.
        upper_level_command_frequency_hz = None
        # Hold (new_request - old_request) / upper_period between high-level
        # ticks so a 50 Hz controller observes the causal slope of a sampled
        # 5 Hz command instead of a one-step numerical impulse.
        hold_upper_command_rate = False
        # When true, the command presented by the upper layer is the exact
        # low-level tracking reference.  No acceleration governor or automatic
        # stop/restart phase may replace it.  The upper layer is responsible for
        # sending commands inside the measured reachable set.
        direct_command_tracking = False
        # Stage1.4 is opt-in.  The default command path and all legacy tasks
        # remain unchanged when this switch is false.
        dynamic_governor_enabled = False
        dynamic_governor_model_path = ""
        # Optionally cap steering authority using achieved rolling speed rather
        # than only the requested speed.  This prevents a high-level command
        # from requesting full yaw while the sphere is still accelerating from
        # rest.  Legacy tasks keep the old command-only projection.
        use_measured_turn_authority = False
        turn_authority_speed_preview_margin = 0.0
        # A reversal now contains an explicit braking phase.  Keep each request
        # long enough for brake, release, and tracking rather than replacing it
        # halfway through the maneuver.
        resampling_time = 8.0

        # The feasible command set is not rectangular. Turning commands obey
        # |w| <= min(max_yaw_rate, |v| / minimum_turn_radius).
        max_forward_speed = 0.35
        max_yaw_rate = 0.10
        minimum_turn_radius = 0.90
        minimum_turn_speed = 0.08
        feasible_envelope_fraction = 0.85
        # Optional empirically identified low-speed steering-authority fade.
        # Equal values disable it and preserve the geometric legacy envelope.
        turn_authority_start_speed = 0.0
        turn_authority_full_speed = 0.0

        # Raw upper-level targets may jump, but a spherical robot cannot
        # instantaneously reverse its linear/yaw momentum. These rates define
        # the reference trajectory tracked by the low-level policy.
        maximum_linear_acceleration = 0.30
        maximum_yaw_acceleration = 0.04

        # Legacy tasks execute either reversal as request -> full stop ->
        # opposite request.  A later task can enable yaw-only braking so the
        # sphere retains the translational motion needed for steering authority.
        yaw_only_braking = False
        reversal_detection_v = 0.05
        reversal_detection_w = 0.015
        # Full-stop braking is reserved for discontinuous upper-level requests.
        # Smooth ramps/sinusoids can cross zero safely under the acceleration
        # governor and must not be mistaken for abrupt reversals.
        reversal_minimum_request_jump_v = 0.10
        reversal_minimum_request_jump_w = 0.03
        # A large same-direction yaw reduction also needs a braking phase;
        # otherwise residual angular momentum overwhelms small new yaw targets.
        yaw_deceleration_brake_ratio = 0.50
        yaw_deceleration_brake_delta = 0.025
        # Disabled by default.  A later empirically validated task may require
        # a stop-and-restart for a large same-direction speed reduction.
        linear_deceleration_brake_ratio = None
        linear_deceleration_brake_delta = 0.0
        linear_deceleration_target_speed_max = None
        reversal_release_command_v = 0.005
        reversal_release_command_w = 0.005
        reversal_release_measured_v = 0.03
        reversal_release_measured_w = 0.015

        stop_fraction = 0.10
        straight_fraction = 0.25
        turn_fraction = 0.65
        # Cover the entire non-zero yaw range.  The previous 0.45 lower bound
        # omitted exactly the small-yaw commands that failed random evaluation.
        minimum_turn_command_fraction = 0.05
        # v12's remaining tail error was concentrated in high-speed, maximum-
        # yaw commands and in direction changes. Oversample those cases without
        # removing the low-yaw and straight-line coverage added in v11.
        extreme_turn_fraction = 0.35
        opposite_transition_fraction = 0.50
        # Probability of deliberately keeping the rolling direction while
        # reversing only yaw.  Zero preserves every task before v39.  This is
        # a distinct physical transition for a spherical robot: steering
        # authority must be retained while angular momentum changes sign.
        yaw_only_transition_fraction = 0.0

        # Disabled for the legacy v13 task.  The independent dynamic task turns
        # this on so PPO sees smooth ramps and sinusoidal zero crossings.
        smooth_profile_fraction = 0.0
        # Fraction of smooth-profile environments that use a bounded,
        # sign-preserving 5 Hz correlated random walk rather than a sinusoid.
        random_walk_profile_fraction = 0.0
        random_walk_linear_step = 0.008
        random_walk_yaw_step = 0.004
        random_walk_minimum_speed = 0.08
        smooth_profile_period_min_s = 12.0
        smooth_profile_period_max_s = 30.0
        smooth_profile_speed_amplitude_min = 0.12
        smooth_profile_speed_amplitude_max = 0.35
        smooth_profile_yaw_fraction_min = 0.15
        smooth_profile_yaw_fraction_max = 1.0
        # Optional independent smooth family. Zero keeps all legacy tasks on
        # the original constant-curvature profile w = curvature * v.
        independent_smooth_profile_fraction = 0.0
        independent_fixed_velocity_fraction = 0.60
        independent_profile_minimum_speed = 0.04
        independent_profile_yaw_fraction_min = 0.15
        independent_profile_yaw_fraction_max = 1.0
        independent_profile_yaw_frequency_ratios = [0.5, 1.0, 2.0]
        observe_command_rates = False
        observe_preview_tracking_errors = False
        observe_tracking_error_integrals = False
        observe_tracking_error_derivatives = False

        # The nominal inverse already provides stable motion, so residual PPO
        # can train on the full feasible command mixture from iteration zero.
        straight_only_policy_steps = 0
        mixed_policy_steps = 0
        mixed_turn_fraction = 0.40
        mixed_envelope_fraction = 0.50

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.35, 0.35]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-0.10, 0.10]

    class asset(LeggedRobotCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/Rotunbot/urdf/"
            "Rotunbot_test2.urdf"
        )
        name = "Rotunbot"
        terminate_after_contacts_on = ["base_link"]
        penalize_contacts_on = []
        self_collisions = 1

        # A perfect sphere/plane point contact has no torsional resistance.
        # Keep the same contact-patch yaw model used by depth navigation so the
        # learned tracker cannot exploit unrealistic in-place spin.
        angular_damping = 0.0
        contact_yaw_damping = True
        contact_yaw_damping_body = "link1"
        contact_yaw_damping_viscous = 2.0
        contact_yaw_damping_coulomb = 0.5
        contact_yaw_damping_transition = 0.02
        contact_yaw_damping_max_torque = 2.0
        contact_yaw_damping_force_threshold = 10.0
        contact_yaw_damping_speed_scale = 0.10
        contact_yaw_damping_speed_exponent = 4.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        # First establish the nominal tracking contract. Robustness
        # randomization is a later, separate stage.
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.5
            ang_vel = 2.0
            dof_pos = 1.0 / 0.45
            dof_vel = 0.5
            height_measurements = 1.0

        clip_observations = 10.0
        clip_actions = 1.0

    class noise(LeggedRobotCfg.noise):
        add_noise = False
        noise_level = 0.0

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0

        # Fixed physical error widths avoid the zero-command singularity in
        # the previous relative-error reward.
        linear_tracking_sigma = 0.12
        angular_tracking_sigma = 0.025
        # During smooth-reference training a small causal rate extrapolation
        # can be used as the reward target. Zero preserves legacy tasks.
        smooth_angular_reward_preview_time = 0.0
        angular_acceleration_error_sigma = 0.04
        # Width of the normalized perpendicular error between requested and
        # measured [v, yaw-rate].  This is a division-free turn-radius signal.
        curvature_tracking_sigma = 0.08
        stationary_command_threshold = 0.02
        turning_command_threshold = 0.01

        class scales:
            termination = -1.0
            tracking_lin_vel = 5.0
            tracking_ang_vel = 10.0
            curvature_tracking = 0.0
            angular_tracking_error = -2.0
            angular_acceleration_error = 0.0
            # The absolute angular tracking reward already has its unique
            # optimum at command_w. A positive direction bonus keeps growing
            # after the target and therefore encourages systematic overshoot.
            yaw_direction = 0.0
            yaw_wrong_direction = 0.0
            lateral_velocity = -0.25
            stationary_yaw = -0.50
            straight_yaw = -0.75
            action_rate = -0.005
            residual_action = -0.005
            action_saturation = -0.01
            torques = -2.0e-6
            dof_pos_limits = -0.20


class RotunbotVelCfgPPO(LeggedRobotCfgPPO):
    seed = 4

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.20
        actor_hidden_dims = [128, 64]
        critic_hidden_dims = [128, 64]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 3.0e-4
        entropy_coef = 0.0005
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "rotunbot_vel"
        run_name = ""
        num_steps_per_env = 32
        max_iterations = 1000
        save_interval = 50


class RotunbotVelDynamicCfg(RotunbotVelCfg):
    """No-noise v14 task combining abrupt requests and smooth references."""

    class control(RotunbotVelCfg.control):
        linear_feedback_gain = 1.0
        angular_feedback_gain = 0.40
        linear_feedback_action_limit = 0.60
        angular_feedback_action_limit = 0.25
        linear_command_lead_time = 0.25
        angular_command_lead_time = 0.75
        residual_action_scale = [0.05, 0.15]

    class commands(RotunbotVelCfg.commands):
        smooth_profile_fraction = 0.50

    class rewards(RotunbotVelCfg.rewards):
        class scales(RotunbotVelCfg.rewards.scales):
            tracking_lin_vel = 6.0
            tracking_ang_vel = 14.0
            angular_tracking_error = -3.0
            action_rate = -0.002
            residual_action = -0.003


class RotunbotVelDynamicCfgPPO(RotunbotVelCfgPPO):
    class policy(RotunbotVelCfgPPO.policy):
        init_noise_std = 0.15

    class algorithm(RotunbotVelCfgPPO.algorithm):
        learning_rate = 2.0e-4
        entropy_coef = 0.0002

    class runner(RotunbotVelCfgPPO.runner):
        experiment_name = "rotunbot_vel_dynamic"
        save_interval = 50


class RotunbotVelDynamicRateCfg(RotunbotVelDynamicCfg):
    """v15 task exposing governed acceleration to a safety-filtered residual."""

    class env(RotunbotVelDynamicCfg.env):
        num_observations = 18

    class control(RotunbotVelDynamicCfg.control):
        residual_action_scale = [0.04, 0.12]
        residual_error_alignment_filter = True
        disable_residual_during_braking = True

    class commands(RotunbotVelDynamicCfg.commands):
        smooth_profile_fraction = 0.75
        observe_command_rates = True

    class rewards(RotunbotVelDynamicCfg.rewards):
        class scales(RotunbotVelDynamicCfg.rewards.scales):
            tracking_lin_vel = 8.0
            tracking_ang_vel = 20.0
            angular_tracking_error = -5.0
            action_rate = -0.01
            residual_action = -0.25


class RotunbotVelDynamicRateCfgPPO(RotunbotVelDynamicCfgPPO):
    class policy(RotunbotVelDynamicCfgPPO.policy):
        init_noise_std = 0.08

    class algorithm(RotunbotVelDynamicCfgPPO.algorithm):
        learning_rate = 1.0e-4
        entropy_coef = 0.0

    class runner(RotunbotVelDynamicCfgPPO.runner):
        experiment_name = "rotunbot_vel_dynamic_rate"
        save_interval = 50


class RotunbotVelDynamicPreviewCfg(RotunbotVelDynamicRateCfg):
    """v16 task allowing bounded rate-aware anticipation of tracking error."""

    class env(RotunbotVelDynamicRateCfg.env):
        # v15 observations (18) + predicted linear/yaw tracking errors (2).
        num_observations = 20

    class control(RotunbotVelDynamicRateCfg.control):
        # The preview only changes the safety projection and the two explicit
        # error observations.  The nominal actuator map remains the calibrated
        # v15 controller, so abrupt-reversal behavior is preserved.
        residual_alignment_linear_preview_time = 0.25
        residual_alignment_angular_preview_time = 1.00
        residual_action_scale = [0.035, 0.10]

    class commands(RotunbotVelDynamicRateCfg.commands):
        smooth_profile_fraction = 0.85
        observe_preview_tracking_errors = True

    class rewards(RotunbotVelDynamicRateCfg.rewards):
        class scales(RotunbotVelDynamicRateCfg.rewards.scales):
            tracking_lin_vel = 8.0
            tracking_ang_vel = 24.0
            angular_tracking_error = -8.0
            action_rate = -0.005
            residual_action = -0.05


class RotunbotVelDynamicPreviewCfgPPO(RotunbotVelDynamicRateCfgPPO):
    class policy(RotunbotVelDynamicRateCfgPPO.policy):
        init_noise_std = 0.06

    class algorithm(RotunbotVelDynamicRateCfgPPO.algorithm):
        learning_rate = 7.5e-5
        entropy_coef = 0.0

    class runner(RotunbotVelDynamicRateCfgPPO.runner):
        experiment_name = "rotunbot_vel_dynamic_preview"
        save_interval = 25


class RotunbotVelDynamicPICfg(RotunbotVelDynamicPreviewCfg):
    """v17 task learning a bounded residual around the identified PI tracker."""

    class env(RotunbotVelDynamicPreviewCfg.env):
        # v16 observations (20) + normalized integrated v/yaw errors (2).
        num_observations = 22

    class control(RotunbotVelDynamicPreviewCfg.control):
        angular_feedback_gain = 0.30
        angular_command_lead_time = 1.10
        angular_integral_gain = 0.10
        angular_integral_action_limit = 0.08
        integral_leak_rate = 0.50
        residual_alignment_angular_preview_time = 1.20
        residual_action_scale = [0.03, 0.08]

    class commands(RotunbotVelDynamicPreviewCfg.commands):
        # Keep enough abrupt targets to recover the random-command coverage
        # that the 85%-smooth v16 distribution under-represented.
        smooth_profile_fraction = 0.70
        observe_tracking_error_integrals = True

    class rewards(RotunbotVelDynamicPreviewCfg.rewards):
        class scales(RotunbotVelDynamicPreviewCfg.rewards.scales):
            tracking_lin_vel = 8.0
            tracking_ang_vel = 24.0
            angular_tracking_error = -8.0
            action_rate = -0.005
            residual_action = -0.10


class RotunbotVelDynamicPICfgPPO(RotunbotVelDynamicPreviewCfgPPO):
    class policy(RotunbotVelDynamicPreviewCfgPPO.policy):
        init_noise_std = 0.05

    class algorithm(RotunbotVelDynamicPreviewCfgPPO.algorithm):
        learning_rate = 5.0e-5
        entropy_coef = 0.0

    class runner(RotunbotVelDynamicPreviewCfgPPO.runner):
        experiment_name = "rotunbot_vel_dynamic_pi"
        save_interval = 25


class RotunbotVelDynamicSteadyPICfg(RotunbotVelDynamicPICfg):
    """v18 tracker: PI removes steady bias but never fights a moving target."""

    class control(RotunbotVelDynamicPICfg.control):
        integral_command_rate_threshold = [1.0e-4, 1.0e-4]


class RotunbotVelDynamicSteadyPICfgPPO(RotunbotVelDynamicPICfgPPO):
    class runner(RotunbotVelDynamicPICfgPPO.runner):
        experiment_name = "rotunbot_vel_dynamic_steady_pi"
        save_interval = 25


class RotunbotVelDynamicAnticipatoryCfg(RotunbotVelDynamicSteadyPICfg):
    """v19: train the residual to anticipate a causal smooth yaw reference."""

    class control(RotunbotVelDynamicSteadyPICfg.control):
        # The v18 sweep gave the best joint sine/random compromise at 0.90 s.
        angular_command_lead_time = 0.90

    class commands(RotunbotVelDynamicSteadyPICfg.commands):
        # Retain 20% abrupt requests so phase improvement cannot erase step
        # and random-target behavior.
        smooth_profile_fraction = 0.80

    class rewards(RotunbotVelDynamicSteadyPICfg.rewards):
        smooth_angular_reward_preview_time = 0.16

        class scales(RotunbotVelDynamicSteadyPICfg.rewards.scales):
            action_rate = -0.002
            residual_action = -0.03


class RotunbotVelDynamicAnticipatoryCfgPPO(RotunbotVelDynamicSteadyPICfgPPO):
    class policy(RotunbotVelDynamicSteadyPICfgPPO.policy):
        init_noise_std = 0.06

    class algorithm(RotunbotVelDynamicSteadyPICfgPPO.algorithm):
        learning_rate = 7.5e-5

    class runner(RotunbotVelDynamicSteadyPICfgPPO.runner):
        experiment_name = "rotunbot_vel_dynamic_anticipatory"
        save_interval = 25


class RotunbotVelDynamicAnticipatoryV20Cfg(RotunbotVelDynamicAnticipatoryCfg):
    """v20: strengthen causal yaw preview without changing the controller."""

    class rewards(RotunbotVelDynamicAnticipatoryCfg.rewards):
        # v19 at 0.16 s improved the best measured yaw lag by one 0.04 s
        # policy step.  Use two additional steps of preview so the result is
        # robustly below the 1.00 s acceptance limit instead of sitting on it.
        smooth_angular_reward_preview_time = 0.32


class RotunbotVelDynamicAnticipatoryV20CfgPPO(
    RotunbotVelDynamicAnticipatoryCfgPPO
):
    class runner(RotunbotVelDynamicAnticipatoryCfgPPO.runner):
        experiment_name = "rotunbot_vel_dynamic_anticipatory_v20"
        save_interval = 25


class RotunbotVelDynamicSmoothFeedforwardV21Cfg(
    RotunbotVelDynamicAnticipatoryCfg
):
    """v21: causal smooth-only actuator-rate feedforward."""

    class control(RotunbotVelDynamicAnticipatoryCfg.control):
        angular_rate_feedforward_time = 0.60
        angular_rate_feedforward_action_limit = 0.12
        rate_feedforward_target_gap_threshold = [0.02, 0.005]


class RotunbotVelDynamicSmoothFeedforwardV21CfgPPO(
    RotunbotVelDynamicAnticipatoryCfgPPO
):
    class runner(RotunbotVelDynamicAnticipatoryCfgPPO.runner):
        experiment_name = "rotunbot_vel_dynamic_smooth_ff_v21"
        save_interval = 25


class RotunbotVelDynamicSmoothBrakeV22Cfg(
    RotunbotVelDynamicSmoothFeedforwardV21Cfg
):
    """v22: add causal yaw-momentum braking only for smooth requests."""

    class control(RotunbotVelDynamicSmoothFeedforwardV21Cfg.control):
        angular_rate_feedforward_time = -0.40
        angular_rate_feedforward_action_limit = 0.12
        smooth_angular_feedback_gain = 0.20
        smooth_angular_feedback_action_limit = 0.20


class RotunbotVelDynamicSmoothBrakeV22CfgPPO(
    RotunbotVelDynamicSmoothFeedforwardV21CfgPPO
):
    class runner(RotunbotVelDynamicSmoothFeedforwardV21CfgPPO.runner):
        experiment_name = "rotunbot_vel_dynamic_smooth_brake_v22"
        save_interval = 25


class RotunbotVelDynamicAccelerationV23Cfg(
    RotunbotVelDynamicAnticipatoryCfg
):
    """v23: expose and reward yaw-acceleration tracking for phase control."""

    class env(RotunbotVelDynamicAnticipatoryCfg.env):
        # v19 observations (22) + linear/yaw tracking-error derivatives (2).
        num_observations = 24

    class commands(RotunbotVelDynamicAnticipatoryCfg.commands):
        observe_tracking_error_derivatives = True

    class rewards(RotunbotVelDynamicAnticipatoryCfg.rewards):
        angular_acceleration_error_sigma = 0.04

        class scales(RotunbotVelDynamicAnticipatoryCfg.rewards.scales):
            angular_acceleration_error = -4.0


class RotunbotVelDynamicAccelerationV23CfgPPO(
    RotunbotVelDynamicAnticipatoryCfgPPO
):
    class runner(RotunbotVelDynamicAnticipatoryCfgPPO.runner):
        experiment_name = "rotunbot_vel_dynamic_acceleration_v23"
        save_interval = 25


class RotunbotVelDynamicDualLeadV24Cfg(RotunbotVelDynamicAnticipatoryCfg):
    """v24: preserve step lead while advancing smooth coupled v/w reversal."""

    class control(RotunbotVelDynamicAnticipatoryCfg.control):
        # The default 0.25 s remains active for abrupt/rate-limited targets.
        # A causal target-gap mask selects 0.80 s only for smooth requests.
        smooth_linear_command_lead_time = 0.80
        smooth_linear_command_minimum_rate = 1.0e-4
        rate_feedforward_target_gap_threshold = [0.005, 0.002]


class RotunbotVelDynamicDualLeadV24CfgPPO(
    RotunbotVelDynamicAnticipatoryCfgPPO
):
    class runner(RotunbotVelDynamicAnticipatoryCfgPPO.runner):
        experiment_name = "rotunbot_vel_dynamic_dual_lead_v24"
        save_interval = 25


class RotunbotVelDynamicDualPhaseV25Cfg(RotunbotVelDynamicDualLeadV24Cfg):
    """v25: use independent step and smooth-reference yaw anticipation."""

    class control(RotunbotVelDynamicDualLeadV24Cfg.control):
        # Abrupt/rate-limited requests retain the calibrated 0.90 s v19 lead.
        # Smooth references use less yaw lead to remove the 12 s sine overshoot.
        smooth_angular_command_lead_time = 0.75
        smooth_angular_command_minimum_rate = 1.0e-4
        # Complete a causal lead compensator: the derivative channel advances
        # the mid-slope response while the smooth-only gain limits amplitude.
        angular_rate_feedforward_time = 0.40
        angular_rate_feedforward_action_limit = 0.10
        smooth_angular_command_gain = 0.92


class RotunbotVelDynamicDualPhaseV25CfgPPO(
    RotunbotVelDynamicDualLeadV24CfgPPO
):
    class runner(RotunbotVelDynamicDualLeadV24CfgPPO.runner):
        experiment_name = "rotunbot_vel_dynamic_dual_phase_v25"
        save_interval = 25


class RotunbotVelDynamicExplicitPhaseV26Cfg(
    RotunbotVelDynamicDualPhaseV25Cfg
):
    """v26 deployment: isolate strong phase lead to explicit smooth references."""

    class control(RotunbotVelDynamicDualPhaseV25Cfg.control):
        angular_rate_feedforward_time = 0.80
        angular_rate_feedforward_action_limit = 0.15
        smooth_angular_command_gain = 0.70
        require_explicit_smooth_profile_for_phase_lead = True


class RotunbotVelDynamicExplicitPhaseV26CfgPPO(
    RotunbotVelDynamicDualPhaseV25CfgPPO
):
    class runner(RotunbotVelDynamicDualPhaseV25CfgPPO.runner):
        experiment_name = "rotunbot_vel_dynamic_explicit_phase_v26"
        save_interval = 25


class RotunbotVelIndependentV28Cfg(RotunbotVelDynamicExplicitPhaseV26Cfg):
    """v28: independently varying feasible v/w commands at a safe speed cap."""

    class env(RotunbotVelDynamicExplicitPhaseV26Cfg.env):
        # Long enough to retain complete 6-16 s profiles between resets.
        episode_length_s = 40.0

    class control(RotunbotVelDynamicExplicitPhaseV26Cfg.control):
        # The fixed-v calibration passed every bin through 0.15 m/s with a
        # speed-independent inverse gain.  At 0.20 m/s the plant entered a
        # delayed/saturated regime, so V28 deliberately stays below it.
        nominal_yaw_gain_speed_slope = 0.0
        residual_action_scale = [0.04, 0.14]

    class commands(RotunbotVelDynamicExplicitPhaseV26Cfg.commands):
        max_forward_speed = 0.15
        minimum_turn_speed = 0.04
        resampling_time = 16.0
        smooth_profile_fraction = 0.85
        smooth_profile_period_min_s = 6.0
        smooth_profile_period_max_s = 16.0
        smooth_profile_speed_amplitude_min = 0.04
        smooth_profile_speed_amplitude_max = 0.15
        independent_smooth_profile_fraction = 0.75
        independent_fixed_velocity_fraction = 0.60
        independent_profile_minimum_speed = 0.04
        independent_profile_yaw_fraction_min = 0.10
        independent_profile_yaw_fraction_max = 1.0
        independent_profile_yaw_frequency_ratios = [0.5, 1.0, 2.0]

        class ranges(RotunbotVelDynamicExplicitPhaseV26Cfg.commands.ranges):
            lin_vel_x = [-0.15, 0.15]

    class rewards(RotunbotVelDynamicExplicitPhaseV26Cfg.rewards):
        class scales(RotunbotVelDynamicExplicitPhaseV26Cfg.rewards.scales):
            # Penalize only an actual wrong direction; unlike the former
            # positive direction bonus this has no incentive to overshoot.
            yaw_wrong_direction = -3.0
            residual_action = -0.015


class RotunbotVelIndependentV28CfgPPO(
    RotunbotVelDynamicExplicitPhaseV26CfgPPO
):
    class policy(RotunbotVelDynamicExplicitPhaseV26CfgPPO.policy):
        init_noise_std = 0.08

    class algorithm(RotunbotVelDynamicExplicitPhaseV26CfgPPO.algorithm):
        learning_rate = 1.0e-4
        entropy_coef = 0.0002

    class runner(RotunbotVelDynamicExplicitPhaseV26CfgPPO.runner):
        experiment_name = "rotunbot_vel_independent_v28"
        max_iterations = 1000
        save_interval = 25


class RotunbotVelIndependentRobustV29Cfg(RotunbotVelIndependentV28Cfg):
    """v29: dynamically validated v/w envelope with decoupled lead calibration."""

    class control(RotunbotVelIndependentV28Cfg.control):
        # The V26/V28 fixed-speed sweep showed that the lead compensator is
        # calibrated with 0.20 m/s headroom even though robust commands stop at
        # 0.15 m/s.  Keep this internal projection scale separate from the
        # externally admissible command envelope below.
        lead_projection_max_forward_speed = 0.20
        disable_integral_for_explicit_smooth_profiles = True
        # The nominal inverse already passes the validated envelope.  PPO may
        # add only a small state-dependent correction and cannot dominate it.
        residual_action_scale = [0.02, 0.06]

    class commands(RotunbotVelIndependentV28Cfg.commands):
        # The geometric radius 0.90 m allowed 0.0472 rad/s at 0.05 m/s, but the
        # measured plant oscillated to +/-0.145 rad/s.  A 1.42 m robust dynamic
        # radius gives ~0.030 rad/s at 0.05 m/s, which passed both directions
        # with zero wrong-sign samples.  At 0.15 m/s the global 0.085 rad/s cap
        # remains active, so maximum-speed steering authority is retained.
        minimum_turn_radius = 1.42
        # Targets remain bounded by max_forward_speed=0.15.  The larger value
        # is only numerical headroom inside the one-step acceleration governor;
        # it prevents a target exactly on the external boundary from repeatedly
        # entering the projection/clamp branch.
        governor_projection_max_forward_speed = 0.20

    class rewards(RotunbotVelIndependentV28Cfg.rewards):
        class scales(RotunbotVelIndependentV28Cfg.rewards.scales):
            residual_action = -0.10


class RotunbotVelIndependentRobustV29CfgPPO(RotunbotVelIndependentV28CfgPPO):
    class policy(RotunbotVelIndependentV28CfgPPO.policy):
        init_noise_std = 0.05

    class runner(RotunbotVelIndependentV28CfgPPO.runner):
        experiment_name = "rotunbot_vel_independent_robust_v29"
        max_iterations = 1000
        save_interval = 25


class RotunbotVelIndependentFeasibleV30Cfg(
    RotunbotVelIndependentRobustV29Cfg
):
    """v30: final no-noise command set validated across 32 PhysX environments."""

    class commands(RotunbotVelIndependentRobustV29Cfg.commands):
        # Effective yaw slope is envelope_fraction / radius = 0.85 / 2.125
        # = 0.40 rad/m.  Long-horizon 32-env tests passed at (v,w) amplitudes
        # (0.05,0.02), (0.10,0.04), and (0.15,0.06) in both directions.
        minimum_turn_radius = 2.125


class RotunbotVelIndependentFeasibleV30CfgPPO(
    RotunbotVelIndependentRobustV29CfgPPO
):
    class runner(RotunbotVelIndependentRobustV29CfgPPO.runner):
        experiment_name = "rotunbot_vel_independent_feasible_v30"
        max_iterations = 1000
        save_interval = 25


class RotunbotVelIndependentDynamicEnvelopeV31Cfg(
    RotunbotVelIndependentFeasibleV30Cfg
):
    """v31: one shared empirically feasible envelope for train/control/eval."""

    class control(RotunbotVelIndependentFeasibleV30Cfg.control):
        # The seed-9123 sweep reduced the angular p95 tail with 0.70 s lead,
        # while derivative feedback and disabling PI both made it worse.
        angular_command_lead_time = 0.70
        # V30 PPO residuals did not beat the nominal controller.  Keep learning
        # authority deliberately small so a policy cannot erase the validated
        # inverse map while it learns the remaining state-dependent correction.
        residual_action_scale = [0.01, 0.04]

    class commands(RotunbotVelIndependentFeasibleV30Cfg.commands):
        # Physical traces show that straight motion remains accurate below
        # 0.04 m/s, but useful/reversible yaw does not.  Fade yaw authority from
        # zero at 0.04 m/s to the full V30 bound at 0.08 m/s.  This is not an
        # arbitrary score relaxation: commands outside this envelope are not
        # accepted as mechanically feasible commands anywhere in the stack.
        turn_authority_start_speed = 0.04
        turn_authority_full_speed = 0.08
        # Increase abrupt-command coverage from 15% to 35%; random transitions
        # were the remaining failure mode after smooth-sine tracking passed.
        smooth_profile_fraction = 0.65
        opposite_transition_fraction = 0.60


class RotunbotVelIndependentDynamicEnvelopeV31CfgPPO(
    RotunbotVelIndependentFeasibleV30CfgPPO
):
    class runner(RotunbotVelIndependentFeasibleV30CfgPPO.runner):
        experiment_name = "rotunbot_vel_independent_dynamic_envelope_v31"
        max_iterations = 1000
        save_interval = 25


class RotunbotVelIndependentBrakeGovernorV32Cfg(
    RotunbotVelIndependentDynamicEnvelopeV31Cfg
):
    """v32: brake before abrupt high-to-low speed requests."""

    class control(RotunbotVelIndependentDynamicEnvelopeV31Cfg.control):
        # V31 PPO screens consistently degraded a stronger nominal controller.
        # Retain only tiny residual authority for the next guarded probe.
        residual_action_scale = [0.002, 0.010]

    class commands(RotunbotVelIndependentDynamicEnvelopeV31Cfg.commands):
        # Seed-9123 failures were concentrated in abrupt 0.09--0.14 m/s to
        # 0.02--0.04 m/s requests.  At those points rolling momentum repeatedly
        # crossed zero even after seven seconds.  A full-stop phase is the same
        # physically causal contract already used for sign reversals.
        linear_deceleration_brake_ratio = 0.50
        linear_deceleration_brake_delta = 0.04
        reversal_minimum_request_jump_v = 0.06

    class rewards(RotunbotVelIndependentDynamicEnvelopeV31Cfg.rewards):
        class scales(
            RotunbotVelIndependentDynamicEnvelopeV31Cfg.rewards.scales
        ):
            residual_action = -0.50


class RotunbotVelIndependentBrakeGovernorV32CfgPPO(
    RotunbotVelIndependentDynamicEnvelopeV31CfgPPO
):
    class runner(RotunbotVelIndependentDynamicEnvelopeV31CfgPPO.runner):
        experiment_name = "rotunbot_vel_independent_brake_governor_v32"
        max_iterations = 1000
        save_interval = 25


class RotunbotVelIndependentTargetedBrakeV33Cfg(
    RotunbotVelIndependentBrakeGovernorV32Cfg
):
    """v33: restrict full-stop deceleration braking to low-speed targets."""

    class commands(RotunbotVelIndependentBrakeGovernorV32Cfg.commands):
        linear_deceleration_target_speed_max = 0.04


class RotunbotVelIndependentTargetedBrakeV33CfgPPO(
    RotunbotVelIndependentBrakeGovernorV32CfgPPO
):
    class runner(RotunbotVelIndependentBrakeGovernorV32CfgPPO.runner):
        experiment_name = "rotunbot_vel_independent_targeted_brake_v33"
        max_iterations = 1000
        save_interval = 25


class RotunbotVelIndependentTargetedBrakeV34Cfg(
    RotunbotVelIndependentTargetedBrakeV33Cfg
):
    """v34: include the measured 0.04--0.05 m/s deceleration failures."""

    class commands(RotunbotVelIndependentTargetedBrakeV33Cfg.commands):
        linear_deceleration_target_speed_max = 0.05


class RotunbotVelIndependentTargetedBrakeV34CfgPPO(
    RotunbotVelIndependentTargetedBrakeV33CfgPPO
):
    class runner(RotunbotVelIndependentTargetedBrakeV33CfgPPO.runner):
        experiment_name = "rotunbot_vel_independent_targeted_brake_v34"
        max_iterations = 1000
        save_interval = 25


class RotunbotVelIndependentScheduledFeedbackV35Cfg(
    RotunbotVelIndependentTargetedBrakeV34Cfg
):
    """v35: damp low-speed rolling oscillation without weakening fast turns."""

    class control(RotunbotVelIndependentTargetedBrakeV34Cfg.control):
        # A full-envelope seed-9123 sweep at gain=0.75 raised linear direction
        # coverage, but weakened high-speed yaw. Apply it only in the measured
        # low-speed failure band and recover gain=1.0 by 0.08 m/s.
        low_speed_linear_feedback_gain = 0.75
        linear_feedback_transition_start_speed = 0.04
        linear_feedback_transition_full_speed = 0.08


class RotunbotVelIndependentScheduledFeedbackV35CfgPPO(
    RotunbotVelIndependentTargetedBrakeV34CfgPPO
):
    class runner(RotunbotVelIndependentTargetedBrakeV34CfgPPO.runner):
        experiment_name = "rotunbot_vel_independent_scheduled_feedback_v35"
        max_iterations = 1000
        save_interval = 25


class RotunbotVelIndependentWaveRobustV36Cfg(
    RotunbotVelIndependentScheduledFeedbackV35Cfg
):
    """v36: waveform-neutral yaw tracking inside the measured feasible envelope."""

    class control(RotunbotVelIndependentScheduledFeedbackV35Cfg.control):
        # The V35 20 s waveform sweep showed that explicit angular-rate lead
        # over-amplified constant-slope triangle commands. With no explicit
        # lead, both sine and triangle tracking improved while the physical
        # yaw-rate projection and low-speed authority fade remained unchanged.
        angular_rate_feedforward_time = 0.0


class RotunbotVelIndependentWaveRobustV36CfgPPO(
    RotunbotVelIndependentScheduledFeedbackV35CfgPPO
):
    class runner(RotunbotVelIndependentScheduledFeedbackV35CfgPPO.runner):
        experiment_name = "rotunbot_vel_independent_wave_robust_v36"
        max_iterations = 1000
        save_interval = 25


class RotunbotVelIndependentSafeEnvelopeV37Cfg(
    RotunbotVelIndependentWaveRobustV36Cfg
):
    """v37: cap commands below the measured 0.14 m/s yaw-dynamics cliff."""

    class commands(RotunbotVelIndependentWaveRobustV36Cfg.commands):
        # Fixed-v sine tests were symmetric and accurate through 0.13 m/s
        # (w MAE <= 0.0038 rad/s, lag <= 0.36 s, no wrong-sign samples).
        # At 0.14 m/s lag jumped to 1.12 s and wrong-sign samples appeared.
        max_forward_speed = 0.13
        smooth_profile_speed_amplitude_max = 0.13

        class ranges(RotunbotVelIndependentWaveRobustV36Cfg.commands.ranges):
            lin_vel_x = [-0.13, 0.13]


class RotunbotVelIndependentSafeEnvelopeV37CfgPPO(
    RotunbotVelIndependentWaveRobustV36CfgPPO
):
    class runner(RotunbotVelIndependentWaveRobustV36CfgPPO.runner):
        experiment_name = "rotunbot_vel_independent_safe_envelope_v37"
        max_iterations = 1000
        save_interval = 25


class RotunbotVelIndependentYawBrakeV38Cfg(
    RotunbotVelIndependentSafeEnvelopeV37Cfg
):
    """v38: preserve rolling steering authority while reversing yaw."""

    class commands(RotunbotVelIndependentSafeEnvelopeV37Cfg.commands):
        # Linear reversals still use a full stop.  A yaw-only reversal instead
        # holds a straight, feasible rolling command until measured yaw is near
        # zero, then releases the requested opposite curvature.
        yaw_only_braking = True


class RotunbotVelIndependentYawBrakeV38CfgPPO(
    RotunbotVelIndependentSafeEnvelopeV37CfgPPO
):
    class runner(RotunbotVelIndependentSafeEnvelopeV37CfgPPO.runner):
        experiment_name = "rotunbot_vel_independent_yaw_brake_v38"
        max_iterations = 1000
        save_interval = 25


class RotunbotVelIndependentYawTransitionV39Cfg(
    RotunbotVelIndependentYawBrakeV38Cfg
):
    """v39: train the yaw-only transitions that dominate navigation control."""

    class commands(RotunbotVelIndependentYawBrakeV38Cfg.commands):
        # V38 random-evaluation failures were concentrated in discontinuous
        # 0.01--0.03 rad/s yaw reversals/reductions.  These are above the
        # evaluator's non-zero yaw threshold but below V38's brake latch.
        reversal_detection_w = 0.010
        reversal_minimum_request_jump_w = 0.015
        reversal_release_measured_w = 0.005
        yaw_deceleration_brake_delta = 0.010

        # V28--V38 flipped both v and w together.  Navigation usually keeps v
        # and asks curvature to change sign, so expose PPO to that transition
        # directly while retaining both smooth profiles and full reversals.
        smooth_profile_fraction = 0.45
        yaw_only_transition_fraction = 0.75
        opposite_transition_fraction = 0.35

    class rewards(RotunbotVelIndependentYawBrakeV38Cfg.rewards):
        class scales(RotunbotVelIndependentYawBrakeV38Cfg.rewards.scales):
            angular_tracking_error = -3.0
            yaw_wrong_direction = -5.0
            residual_action = -0.25


class RotunbotVelIndependentYawTransitionV39CfgPPO(
    RotunbotVelIndependentYawBrakeV38CfgPPO
):
    class runner(RotunbotVelIndependentYawBrakeV38CfgPPO.runner):
        experiment_name = "rotunbot_vel_independent_yaw_transition_v39"
        max_iterations = 1000
        save_interval = 25


class RotunbotVelIndependentTransitionEnvelopeV40Cfg(
    RotunbotVelIndependentYawTransitionV39Cfg
):
    """v40: exclude yaw requests in the measured weak low-speed band."""

    class commands(RotunbotVelIndependentYawTransitionV39Cfg.commands):
        # Across seed123/9123, 39/62 wrong-direction targets occurred at
        # |v|=0.06--0.08 m/s, while only four occurred at |v|>=0.10 m/s.
        # Straight motion remains admissible below 0.08 m/s; only requested
        # yaw is faded out because the sphere lacks reliable steering authority
        # there after arbitrary momentum transitions.
        minimum_turn_speed = 0.08
        turn_authority_start_speed = 0.08
        turn_authority_full_speed = 0.10
        independent_profile_minimum_speed = 0.08


class RotunbotVelIndependentTransitionEnvelopeV40CfgPPO(
    RotunbotVelIndependentYawTransitionV39CfgPPO
):
    class runner(RotunbotVelIndependentYawTransitionV39CfgPPO.runner):
        experiment_name = "rotunbot_vel_independent_transition_envelope_v40"
        max_iterations = 1000
        save_interval = 25


class RotunbotVelSRU50V41Cfg(RotunbotVelIndependentTransitionEnvelopeV40Cfg):
    """v41: 200 Hz physics, 50 Hz low level, and a 5 Hz SRU command port."""

    class sim(RotunbotVelIndependentTransitionEnvelopeV40Cfg.sim):
        dt = 0.005

    class control(RotunbotVelIndependentTransitionEnvelopeV40Cfg.control):
        # Four 200 Hz physics steps per 50 Hz policy action.
        decimation = 4
        # Preserve the physical target slew per second when moving from the
        # legacy 25 Hz controller to 50 Hz.
        joint1_target_rate_limit = 0.02
        joint2_target_rate_limit = 0.04
        # Preserve the derivative filter's approximate continuous-time pole
        # after halving policy dt from 40 ms to 20 ms.
        error_derivative_filter_alpha = 0.105572809

    class commands(RotunbotVelIndependentTransitionEnvelopeV40Cfg.commands):
        upper_level_command_frequency_hz = 5.0
        use_measured_turn_authority = True
        # One centimetre per second of preview avoids a hard dead band from
        # velocity-estimation jitter without allowing in-place steering.
        turn_authority_speed_preview_margin = 0.01
        # Most SRU outputs are temporally correlated.  Retain abrupt stops and
        # reversals for robustness, but make sampled smooth references dominant.
        smooth_profile_fraction = 0.65

    class rewards(RotunbotVelIndependentTransitionEnvelopeV40Cfg.rewards):
        class scales(
            RotunbotVelIndependentTransitionEnvelopeV40Cfg.rewards.scales
        ):
            # At twice the policy frequency, a physically identical smooth
            # action trajectory has roughly half the per-step delta.
            action_rate = -0.004


class RotunbotVelSRU50V41CfgPPO(
    RotunbotVelIndependentTransitionEnvelopeV40CfgPPO
):
    class algorithm(
        RotunbotVelIndependentTransitionEnvelopeV40CfgPPO.algorithm
    ):
        # Preserve approximately the same discount and GAE horizons in seconds
        # after halving the low-level policy period.
        gamma = 0.994987437
        lam = 0.974679434

    class runner(RotunbotVelIndependentTransitionEnvelopeV40CfgPPO.runner):
        experiment_name = "rotunbot_vel_sru50_v41"
        # 64 * 0.02 s matches the former 32 * 0.04 s rollout duration.
        num_steps_per_env = 64
        max_iterations = 1500
        save_interval = 25


class RotunbotVelSRU50LearnableV42Cfg(RotunbotVelSRU50V41Cfg):
    """v42: recalibrated 50 Hz inverse plus a genuinely learnable residual."""

    class control(RotunbotVelSRU50V41Cfg.control):
        # A 200/50/5 Hz no-policy sweep found 0.145 to be the best combined
        # step/random/sine inverse gain.  The inherited 0.0915 value produced
        # roughly 30--40 percent yaw-amplitude overshoot at this update rate.
        nominal_yaw_gain_intercept = 0.145
        # V41 unintentionally inherited V32's [0.002, 0.010] residual scale.
        # At that scale eight independently trained checkpoints were
        # indistinguishable from the zero-residual controller.  Restore bounded
        # authority while retaining the error-alignment safety projection.
        residual_action_scale = [0.04, 0.20]
        residual_error_alignment_filter = True
        # A reversal is exactly where angular-momentum correction is needed.
        # The alignment filter still removes an error-increasing residual.
        disable_residual_during_braking = False

    class rewards(RotunbotVelSRU50V41Cfg.rewards):
        class scales(RotunbotVelSRU50V41Cfg.rewards.scales):
            tracking_ang_vel = 32.0
            angular_tracking_error = -6.0
            yaw_wrong_direction = -8.0
            action_rate = -0.002
            residual_action = -0.02


class RotunbotVelSRU50LearnableV42CfgPPO(RotunbotVelSRU50V41CfgPPO):
    class policy(RotunbotVelSRU50V41CfgPPO.policy):
        init_noise_std = 0.10

    class algorithm(RotunbotVelSRU50V41CfgPPO.algorithm):
        learning_rate = 1.0e-4
        entropy_coef = 0.0002

    class runner(RotunbotVelSRU50V41CfgPPO.runner):
        experiment_name = "rotunbot_vel_sru50_v42"
        max_iterations = 1500
        save_interval = 25


class RotunbotVelSRU50AngularResidualV43Cfg(RotunbotVelSRU50LearnableV42Cfg):
    """v43: preserve calibrated linear control and learn yaw correction only."""

    class control(RotunbotVelSRU50LearnableV42Cfg.control):
        # V42 improved random yaw p95 by about 31 percent, but its learned
        # linear residual doubled linear p95 error. The calibrated linear
        # inverse already passes, so remove that unnecessary policy authority.
        # Increase only the error-aligned yaw authority to address the remaining
        # rebound after arbitrary momentum transitions.
        residual_action_scale = [0.0, 0.35]


class RotunbotVelSRU50AngularResidualV43CfgPPO(
    RotunbotVelSRU50LearnableV42CfgPPO
):
    class runner(RotunbotVelSRU50LearnableV42CfgPPO.runner):
        experiment_name = "rotunbot_vel_sru50_v43"
        max_iterations = 1500
        save_interval = 25


class RotunbotVelSRU50TransientV44Cfg(RotunbotVelSRU50AngularResidualV43Cfg):
    """v44: expose the angular residual to navigation-rate transients."""

    class env(RotunbotVelSRU50AngularResidualV43Cfg.env):
        # V43 had command rate, preview-error, and integral observations but no
        # error derivative.  The latter distinguishes continuing yaw momentum
        # from an already-braking state after a new SRU command.
        num_observations = 24

    class commands(RotunbotVelSRU50AngularResidualV43Cfg.commands):
        # V43 trained discontinuous requests only every eight seconds while the
        # random navigation stress test changes targets every 1.5 seconds.  Two
        # seconds covers repeated transients without pretending a sphere can
        # execute an unrelated maximum reversal on every 0.2 s SRU tick.
        resampling_time = 2.0
        smooth_profile_fraction = 0.45
        observe_tracking_error_derivatives = True


class RotunbotVelSRU50TransientV44CfgPPO(
    RotunbotVelSRU50AngularResidualV43CfgPPO
):
    class runner(RotunbotVelSRU50AngularResidualV43CfgPPO.runner):
        experiment_name = "rotunbot_vel_sru50_v44"
        max_iterations = 1500
        save_interval = 25


class RotunbotVelSRU50ReachableV45Cfg(RotunbotVelSRU50TransientV44Cfg):
    """v45: SRU port constrained by the measured dynamic reachable set."""

    class control(RotunbotVelSRU50TransientV44Cfg.control):
        # The no-residual P/D sweep found that the inherited 0.30 feedback gain
        # excites yaw rebound. P=0.15, D=0 was the stable transient baseline.
        angular_feedback_gain = 0.15
        angular_derivative_gain = 0.0
        # The deterministic baseline now passes random and sine tracking. PPO
        # receives only modest, error-aligned authority for the remaining
        # nonlinear state dependence and cannot replace the command governor.
        residual_action_scale = [0.0, 0.15]

    class commands(RotunbotVelSRU50TransientV44Cfg.commands):
        # Measured 200/50/5 Hz sweeps: this pair produced random p95 errors of
        # 0.0192 m/s and 0.0072 rad/s with 99.6/99.3 percent sign correctness.
        maximum_linear_acceleration = 0.08
        maximum_yaw_acceleration = 0.010

    class rewards(RotunbotVelSRU50TransientV44Cfg.rewards):
        class scales(RotunbotVelSRU50TransientV44Cfg.rewards.scales):
            residual_action = -0.10


class RotunbotVelSRU50ReachableV45CfgPPO(RotunbotVelSRU50TransientV44CfgPPO):
    class policy(RotunbotVelSRU50TransientV44CfgPPO.policy):
        init_noise_std = 0.05

    class algorithm(RotunbotVelSRU50TransientV44CfgPPO.algorithm):
        learning_rate = 5.0e-5
        entropy_coef = 0.0

    class runner(RotunbotVelSRU50TransientV44CfgPPO.runner):
        experiment_name = "rotunbot_vel_sru50_v45"
        max_iterations = 1500
        save_interval = 25


class RotunbotVelSRU50LinearResidualV46Cfg(RotunbotVelSRU50ReachableV45Cfg):
    """v46: learn a small linear correction for governed sign transitions."""

    class control(RotunbotVelSRU50ReachableV45Cfg.control):
        # V45 robust evaluation passed every error, step, yaw-direction, and
        # sinusoidal limit, but about four percent of linear random segments
        # crossed zero too late.  V45 gave PPO no linear authority at all.
        # Restore only the small V42 linear bound; the error-alignment filter,
        # V45 governor, and stronger residual penalty prevent steady-state drift.
        residual_action_scale = [0.04, 0.15]

    class commands(RotunbotVelSRU50ReachableV45Cfg.commands):
        # Make governed reversals common enough for PPO to optimize the rare
        # sign-transition failure without replacing the correlated SRU profiles.
        opposite_transition_fraction = 0.50

    class rewards(RotunbotVelSRU50ReachableV45Cfg.rewards):
        class scales(RotunbotVelSRU50ReachableV45Cfg.rewards.scales):
            linear_wrong_direction = -8.0
            residual_action = -0.10


class RotunbotVelSRU50LinearResidualV46CfgPPO(
    RotunbotVelSRU50ReachableV45CfgPPO
):
    class runner(RotunbotVelSRU50ReachableV45CfgPPO.runner):
        experiment_name = "rotunbot_vel_sru50_v46"
        max_iterations = 1500
        save_interval = 25


class RotunbotVelSRU50DirectV47Cfg(RotunbotVelSRU50LinearResidualV46Cfg):
    """v47: exact reachable-set ``(v, w)`` requests with no command governor."""

    class control(RotunbotVelSRU50LinearResidualV46Cfg.control):
        # Controller-side anticipation from older waveform experiments changed
        # the command semantics.  V47 tracks the present physical request only.
        linear_command_lead_time = 0.0
        angular_command_lead_time = 0.0
        smooth_linear_command_lead_time = 0.0
        smooth_angular_command_lead_time = 0.0
        smooth_angular_command_gain = 1.0
        linear_rate_feedforward_time = 0.0
        angular_rate_feedforward_time = 0.0
        linear_integral_gain = 0.0
        angular_integral_gain = 0.0
        residual_alignment_linear_preview_time = 0.0
        residual_alignment_angular_preview_time = 0.0

        # The identified inverse remains a stable baseline.  PPO has enough
        # bounded authority to correct both channels without replacing it.
        linear_feedback_gain = 0.75
        low_speed_linear_feedback_gain = 0.75
        linear_feedback_action_limit = 0.35
        angular_feedback_gain = 0.20
        angular_feedback_action_limit = 0.20
        residual_action_scale = [0.08, 0.30]
        residual_error_alignment_filter = False

    class commands(RotunbotVelSRU50LinearResidualV46Cfg.commands):
        direct_command_tracking = True
        # The explicit baseline scan passed |w| <= 0.30 |v| at 0.10 and
        # 0.13 m/s.  With the inherited 0.85 envelope fraction, 2.8333 m
        # produces exactly that conservative measured slope.
        minimum_turn_radius = 2.8333333333333335
        resampling_time = 3.0
        smooth_profile_fraction = 0.75
        extreme_turn_fraction = 0.20
        opposite_transition_fraction = 0.10
        yaw_only_transition_fraction = 0.25

    class rewards(RotunbotVelSRU50LinearResidualV46Cfg.rewards):
        smooth_angular_reward_preview_time = 0.0
        curvature_tracking_sigma = 0.08

        class scales(RotunbotVelSRU50LinearResidualV46Cfg.rewards.scales):
            tracking_lin_vel = 10.0
            tracking_ang_vel = 28.0
            curvature_tracking = 5.0
            angular_tracking_error = -5.0
            linear_wrong_direction = -4.0
            yaw_wrong_direction = -6.0
            action_rate = -0.002
            residual_action = -0.03


class RotunbotVelSRU50DirectV47CfgPPO(RotunbotVelSRU50LinearResidualV46CfgPPO):
    class policy(RotunbotVelSRU50LinearResidualV46CfgPPO.policy):
        init_noise_std = 0.08

    class algorithm(RotunbotVelSRU50LinearResidualV46CfgPPO.algorithm):
        learning_rate = 1.0e-4
        entropy_coef = 0.0001

    class runner(RotunbotVelSRU50LinearResidualV46CfgPPO.runner):
        experiment_name = "rotunbot_vel_sru50_v47"
        max_iterations = 1500
        save_interval = 25


class RotunbotVelSRU50DirectDynamicV48Cfg(RotunbotVelSRU50DirectV47Cfg):
    """v48: direct 5 Hz SRU commands with causal dynamic yaw tracking."""

    class control(RotunbotVelSRU50DirectV47Cfg.control):
        # The V47 closed-loop sweep showed P=0.40 is the best stable compromise:
        # P=0.60 worsened both steps and random transitions.  Feed forward the
        # measured upper-command slope in actuator space; the requested v/w is
        # never altered and the term is active only on continuous profiles.
        angular_feedback_gain = 0.40
        angular_feedback_action_limit = 0.30
        angular_rate_feedforward_time = 0.30
        angular_rate_feedforward_action_limit = 0.12
        residual_action_scale = [0.08, 0.30]

    class commands(RotunbotVelSRU50DirectV47Cfg.commands):
        hold_upper_command_rate = True
        # Train predominantly on the same 5 Hz correlated command family used
        # by SRU.  Abrupt but non-extreme steps remain for recovery robustness.
        smooth_profile_fraction = 0.85
        random_walk_profile_fraction = 0.40
        random_walk_linear_step = 0.008
        random_walk_yaw_step = 0.004
        random_walk_minimum_speed = 0.08
        extreme_turn_fraction = 0.10
        opposite_transition_fraction = 0.05
        yaw_only_transition_fraction = 0.15

    class rewards(RotunbotVelSRU50DirectV47Cfg.rewards):
        class scales(RotunbotVelSRU50DirectV47Cfg.rewards.scales):
            # Absolute v/w errors remain the primary objective.  Curvature is
            # deliberately auxiliary and cannot replace either tracking term.
            tracking_lin_vel = 12.0
            tracking_ang_vel = 36.0
            angular_tracking_error = -6.0
            curvature_tracking = 3.0
            yaw_wrong_direction = -8.0
            action_rate = -0.001
            residual_action = -0.04


class RotunbotVelSRU50DirectDynamicV48CfgPPO(
    RotunbotVelSRU50DirectV47CfgPPO
):
    class policy(RotunbotVelSRU50DirectV47CfgPPO.policy):
        init_noise_std = 0.08

    class algorithm(RotunbotVelSRU50DirectV47CfgPPO.algorithm):
        learning_rate = 8.0e-5
        entropy_coef = 0.0001

    class runner(RotunbotVelSRU50DirectV47CfgPPO.runner):
        experiment_name = "rotunbot_vel_sru50_v48"
        max_iterations = 1500
        save_interval = 25


class RotunbotVelSRU50ReleaseV49Cfg(RotunbotVelSRU50DirectDynamicV48Cfg):
    """v49 release: calibrated direct SRU tracker selected by held-out tests."""

    class control(RotunbotVelSRU50DirectDynamicV48Cfg.control):
        # Full non-extreme release sweeps at checkpoint 300:
        #   FF=0.60 -> sine yaw lag p95 0.60 s (no margin)
        #   FF=0.65 -> 0.56 s, random yaw p95 0.00987 rad/s
        #   FF=0.70 -> 0.54 s, random yaw p95 0.01015 rad/s
        # Select the smallest value with useful phase-lag margin.
        angular_feedback_gain = 0.40
        angular_feedback_action_limit = 0.30
        angular_rate_feedforward_time = 0.65
        angular_rate_feedforward_action_limit = 0.12

    class commands(RotunbotVelSRU50DirectDynamicV48Cfg.commands):
        # Long 250+250-step scans: slope 0.28 passed with only 0.000114
        # rad/s yaw-MAE margin, while 0.29 failed.  Freeze the release domain
        # one measured grid step inside that boundary: 0.85 / radius = 0.27.
        minimum_turn_radius = 3.148148148148148


class RotunbotVelSRU50ReleaseV49CfgPPO(
    RotunbotVelSRU50DirectDynamicV48CfgPPO
):
    class runner(RotunbotVelSRU50DirectDynamicV48CfgPPO.runner):
        experiment_name = "rotunbot_vel_sru50_v49_release"
        max_iterations = 1500
        save_interval = 25


class RotunbotVelSRU50V49IntegrationCfg(RotunbotVelSRU50ReleaseV49Cfg):
    """Isolated evaluation task for the frozen V49 low-level controller."""

    class asset(RotunbotVelSRU50ReleaseV49Cfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/Rotunbot/urdf/Rotunbot_test2.urdf"


class RotunbotVelSRU50V49IntegrationCfgPPO(RotunbotVelSRU50ReleaseV49CfgPPO):
    class runner(RotunbotVelSRU50ReleaseV49CfgPPO.runner):
        experiment_name = "rotunbot_vel_sru50_v49_integration"
