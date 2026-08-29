"""Low-level feasible forward-velocity/yaw-rate tracking task."""

import torch
from isaacgym import gymapi, gymtorch

from legged_gym.envs.base.legged_robot import LeggedRobot

from .rotunbot_vel_config import RotunbotVelCfg
from .feasible_transition_manager import (
    FeasibleVelocityTransitionManager,
    TransitionState,
)


def command_update_interval_steps(policy_dt, command_frequency_hz):
    """Return an exact policy-step interval for a sampled high-level command.

    A navigation command must remain constant for an integer number of low-level
    policy steps.  Silently rounding an incompatible frequency changes the
    deployed interface, so reject non-integral ratios instead.
    """
    if command_frequency_hz is None:
        return 1
    policy_dt = float(policy_dt)
    command_frequency_hz = float(command_frequency_hz)
    if policy_dt <= 0.0 or command_frequency_hz <= 0.0:
        raise ValueError("policy_dt and command_frequency_hz must be positive")
    exact_interval = 1.0 / (policy_dt * command_frequency_hz)
    interval = int(round(exact_interval))
    if interval < 1 or abs(exact_interval - interval) > 1.0e-6:
        raise ValueError(
            "command frequency must divide the low-level policy frequency: "
            f"policy_dt={policy_dt}, command_frequency_hz={command_frequency_hz}"
        )
    return interval


def planar_velocity_in_heading_frame(heading, world_linear_velocity):
    """Express world velocity in a gravity-aligned planar heading frame."""
    forward_xy = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
    lateral_xy = torch.stack((-torch.sin(heading), torch.cos(heading)), dim=1)
    forward_velocity = torch.sum(
        world_linear_velocity[:, :2] * forward_xy,
        dim=1,
    )
    lateral_velocity = torch.sum(
        world_linear_velocity[:, :2] * lateral_xy,
        dim=1,
    )
    return torch.stack(
        (forward_velocity, lateral_velocity, world_linear_velocity[:, 2]),
        dim=1,
    )


def yaw_from_quaternion(quaternion):
    """Return gravity-aligned yaw for quaternions stored as (x, y, z, w)."""
    qx, qy, qz, qw = quaternion.unbind(dim=1)
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy.square() + qz.square())
    return torch.atan2(sin_yaw, cos_yaw)


def feasible_yaw_rate_limit(
    forward_velocity,
    maximum_yaw_rate,
    minimum_turn_radius,
    envelope_fraction=1.0,
    turn_authority_start_speed=0.0,
    turn_authority_full_speed=0.0,
):
    """Return the speed-dependent nonholonomic yaw-rate bound.

    Legacy tasks use the geometric ``|w| <= |v| / radius`` envelope.  A task
    may additionally fade steering authority in between two empirically
    identified speeds.  This represents the Rotunbot's inability to generate
    useful yaw at (almost) zero rolling speed without pretending that it can
    rotate in place.  A smoothstep ramp keeps the reference and its first
    derivative continuous at both boundaries.
    """
    speed = torch.abs(forward_velocity)
    radius_bound = speed / float(minimum_turn_radius)
    yaw_limit = torch.minimum(
        radius_bound,
        torch.full_like(radius_bound, float(maximum_yaw_rate)),
    ) * float(envelope_fraction)
    start_speed = float(turn_authority_start_speed)
    full_speed = float(turn_authority_full_speed)
    if full_speed > start_speed:
        normalized_speed = torch.clamp(
            (speed - start_speed) / (full_speed - start_speed),
            0.0,
            1.0,
        )
        authority = normalized_speed.square() * (3.0 - 2.0 * normalized_speed)
        yaw_limit = yaw_limit * authority
    return yaw_limit


def speed_scheduled_value(
    forward_velocity,
    low_speed_value,
    full_speed_value,
    transition_start_speed,
    transition_full_speed,
):
    """Smoothly blend a calibrated low-speed value into its nominal value."""
    speed = torch.abs(forward_velocity)
    start_speed = float(transition_start_speed)
    full_speed = float(transition_full_speed)
    if full_speed <= start_speed:
        return torch.full_like(speed, float(full_speed_value))
    fraction = torch.clamp(
        (speed - start_speed) / (full_speed - start_speed), 0.0, 1.0
    )
    fraction = fraction.square() * (3.0 - 2.0 * fraction)
    return float(low_speed_value) + (
        float(full_speed_value) - float(low_speed_value)
    ) * fraction


def piecewise_linear_schedule(value, breakpoints, scheduled_values):
    """Linearly interpolate a tensor over a small monotone calibration table."""
    if breakpoints is None or scheduled_values is None:
        raise ValueError("breakpoints and scheduled_values must both be provided")
    if len(breakpoints) != len(scheduled_values) or len(breakpoints) < 2:
        raise ValueError("piecewise schedules need equally sized tables of length >= 2")
    points = [float(item) for item in breakpoints]
    values = [float(item) for item in scheduled_values]
    if any(right <= left for left, right in zip(points[:-1], points[1:])):
        raise ValueError("piecewise schedule breakpoints must be strictly increasing")

    query = torch.clamp(value, points[0], points[-1])
    result = torch.full_like(query, values[0])
    for index in range(len(points) - 1):
        left = points[index]
        right = points[index + 1]
        fraction = torch.clamp((query - left) / (right - left), 0.0, 1.0)
        segment = values[index] + (values[index + 1] - values[index]) * fraction
        result = torch.where(query >= left, segment, result)
    return result


def canonical_drive_direction(commanded_forward_velocity, measured_forward_velocity,
                              stationary_threshold=0.02):
    """Return a stable +/-1 drive sign for forward/reverse policy symmetry."""
    commanded = commanded_forward_velocity.reshape(-1)
    measured = measured_forward_velocity.reshape(-1)
    moving_command = torch.abs(commanded) >= float(stationary_threshold)
    direction = torch.where(moving_command, torch.sign(commanded), torch.sign(measured))
    return torch.where(direction == 0.0, torch.ones_like(direction), direction)


def canonicalize_velocity_policy_observations(
    observations,
    drive_direction,
    observe_command_rates=False,
    observe_preview_tracking_errors=False,
    observe_tracking_error_integrals=False,
    observe_tracking_error_derivatives=False,
):
    """Map reverse-driving observations into the equivalent forward problem.

    The command-like two-channel blocks are [v, w], so both entries change sign
    under the canonical map.  Body-forward velocity, body yaw rate, and joint-1
    velocity change sign as well.  Steering state and the actor's previous
    canonical residual are invariant.  This removes a learned forward/reverse
    asymmetry without changing the physical nominal controller.
    """
    canonical = observations.clone()
    sign = drive_direction.reshape(-1, 1).to(
        device=canonical.device, dtype=canonical.dtype
    )
    cursor = 0
    canonical[:, cursor : cursor + 2] *= sign
    cursor += 2
    optional_blocks = (
        observe_command_rates,
        observe_preview_tracking_errors,
        observe_tracking_error_integrals,
        observe_tracking_error_derivatives,
    )
    for enabled in optional_blocks:
        if enabled:
            canonical[:, cursor : cursor + 2] *= sign
            cursor += 2

    # tracking_lin_vel [x, y, z]
    canonical[:, cursor] *= sign[:, 0]
    cursor += 3
    # tracking_ang_vel [x, y, z]
    canonical[:, cursor + 2] *= sign[:, 0]
    cursor += 3
    # projected gravity [3] and steering position [1] are invariant.
    cursor += 4
    # dof_vel [joint1, joint2]
    canonical[:, cursor] *= sign[:, 0]
    return canonical


def map_canonical_residual_actions(actions, drive_direction):
    """Map canonical residuals back to physical forward/reverse actuator signs."""
    mapped = actions.clone()
    mapped[:, 0] *= drive_direction.to(
        device=mapped.device, dtype=mapped.dtype
    ).reshape(-1)
    return mapped


def rate_aligned_angular_residual_actions(
    actions,
    commands,
    command_rates,
    measured_forward_velocity,
    smooth_mask,
    stationary_threshold=0.02,
    minimum_angular_command_rate=1.0e-4,
    zero_angular_when_inactive=False,
    measured_yaw_rate=None,
    error_align_when_inactive=False,
    inactive_error_full_scale=None,
):
    """Constrain smooth yaw residuals to the physically correct lead sign.

    For this sphere, positive steering action creates negative yaw while driving
    forward and positive yaw while driving backward.  Therefore a residual that
    anticipates ``dw/dt`` must have sign ``-sign(v) * sign(dw/dt)``.  PPO still
    chooses the nonlinear magnitude from state, error and command-rate inputs;
    the filter removes only the sign ambiguity that produced the measured V55
    wrong-way phase compensation.
    """
    filtered = actions.clone()
    drive_direction = canonical_drive_direction(
        commands[:, 0], measured_forward_velocity, stationary_threshold
    )
    angular_rate = command_rates[:, 1]
    active = smooth_mask.reshape(-1).bool() & (
        torch.abs(angular_rate) >= float(minimum_angular_command_rate)
    )
    desired_sign = -drive_direction * torch.sign(angular_rate)
    aligned = desired_sign * torch.abs(filtered[:, 1])
    if bool(error_align_when_inactive):
        if measured_yaw_rate is None:
            raise ValueError(
                "measured_yaw_rate is required when error_align_when_inactive=True"
            )
        yaw_error = commands[:, 1] - measured_yaw_rate.reshape(-1)
        error_sign = -drive_direction * torch.sign(yaw_error)
        error_aligned = error_sign * torch.abs(filtered[:, 1])
        if inactive_error_full_scale is not None:
            full_scale = max(float(inactive_error_full_scale), 1.0e-8)
            error_gate = torch.clamp(torch.abs(yaw_error) / full_scale, 0.0, 1.0)
            error_aligned *= error_gate
        filtered[:, 1] = torch.where(active, aligned, error_aligned)
    elif bool(zero_angular_when_inactive):
        filtered[:, 1] = torch.where(
            active, aligned, torch.zeros_like(filtered[:, 1])
        )
    else:
        filtered[:, 1] = torch.where(active, aligned, filtered[:, 1])
    return filtered


def update_persistent_yaw_error_gate(
    yaw_error,
    previous_error_sign,
    persistent_time,
    cooldown_time,
    active,
    dt,
    activation_error,
    release_error,
    full_scale_error,
    activation_time,
    sign_flip_cooldown,
    sign_epsilon=1.0e-4,
):
    """Advance a hysteretic safety gate for the learned yaw residual.

    The learned residual is useful for sustained model mismatch, but it must not
    chase ordinary zero crossings.  A correction is therefore admitted only
    after a same-sign yaw error persists.  Crossing zero disables it immediately
    and starts a short cooldown.  The returned continuous gate ramps with error
    magnitude so activation cannot inject the full residual in one step.
    """
    dt = float(dt)
    activation_error = float(activation_error)
    release_error = float(release_error)
    full_scale_error = float(full_scale_error)
    activation_time = float(activation_time)
    sign_flip_cooldown = float(sign_flip_cooldown)
    sign_epsilon = float(sign_epsilon)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not 0.0 <= release_error < activation_error < full_scale_error:
        raise ValueError(
            "expected 0 <= release_error < activation_error < full_scale_error"
        )
    if activation_time < 0.0 or sign_flip_cooldown < 0.0:
        raise ValueError("gate times must be non-negative")

    yaw_error = yaw_error.reshape(-1)
    error_magnitude = torch.abs(yaw_error)
    current_sign = torch.sign(yaw_error)
    sign_is_valid = error_magnitude >= sign_epsilon
    sign_flip = (
        sign_is_valid
        & (previous_error_sign != 0.0)
        & (current_sign != previous_error_sign)
    )

    next_cooldown = torch.clamp(cooldown_time - dt, min=0.0)
    next_cooldown = torch.where(
        sign_flip,
        torch.full_like(next_cooldown, sign_flip_cooldown),
        next_cooldown,
    )
    eligible = (
        (error_magnitude >= activation_error)
        & (next_cooldown <= 0.0)
        & ~sign_flip
    )
    next_persistent_time = torch.where(
        eligible,
        persistent_time + dt,
        torch.zeros_like(persistent_time),
    )
    remain_active = (
        active
        & (error_magnitude > release_error)
        & (next_cooldown <= 0.0)
        & ~sign_flip
    )
    become_active = eligible & (
        next_persistent_time >= max(activation_time - 0.5 * dt, 0.0)
    )
    next_active = remain_active | become_active

    magnitude_gate = torch.clamp(
        (error_magnitude - release_error)
        / max(full_scale_error - release_error, 1.0e-8),
        0.0,
        1.0,
    )
    gate = next_active.to(dtype=yaw_error.dtype) * magnitude_gate
    next_error_sign = torch.where(
        sign_is_valid,
        current_sign,
        torch.where(
            error_magnitude <= release_error,
            torch.zeros_like(previous_error_sign),
            previous_error_sign,
        ),
    )
    return (
        next_error_sign,
        next_persistent_time,
        next_cooldown,
        next_active,
        gate,
    )


def persistent_error_gated_angular_residual_actions(
    actions,
    yaw_error,
    drive_direction,
    gate,
    force_error_alignment=True,
):
    """Apply a yaw gate and force the residual to reduce current yaw error."""
    filtered = actions.clone()
    if bool(force_error_alignment):
        desired_steering_sign = (
            -drive_direction.reshape(-1).to(filtered.dtype)
            * torch.sign(yaw_error.reshape(-1))
        )
        filtered[:, 1] = desired_steering_sign * torch.abs(filtered[:, 1])
    filtered[:, 1] *= gate.reshape(-1).to(filtered.dtype)
    return filtered


def blend_yaw_residual_gate_with_command_rate(
    error_gate,
    angular_command_rate,
    bypass_start_rate,
    bypass_full_rate,
):
    """Blend from the steady-error gate to full residual during yaw transients."""
    bypass_start_rate = float(bypass_start_rate)
    bypass_full_rate = float(bypass_full_rate)
    if bypass_full_rate <= bypass_start_rate:
        return error_gate
    if bypass_start_rate < 0.0:
        raise ValueError("bypass_start_rate must be non-negative")
    rate_gate = torch.clamp(
        (torch.abs(angular_command_rate.reshape(-1)) - bypass_start_rate)
        / (bypass_full_rate - bypass_start_rate),
        0.0,
        1.0,
    )
    return torch.maximum(error_gate, rate_gate.to(error_gate.dtype))


def project_velocity_commands(
    commands,
    maximum_forward_speed,
    maximum_yaw_rate,
    minimum_turn_radius,
    envelope_fraction=1.0,
    stationary_threshold=0.0,
    turn_authority_start_speed=0.0,
    turn_authority_full_speed=0.0,
    authority_forward_velocity=None,
    authority_speed_preview_margin=0.0,
    preserve_curvature_when_saturating=False,
    curvature_fraction_breakpoints=None,
    curvature_max_speed_values=None,
):
    """Project [v, w] commands into the Rotunbot feasible command set.

    When ``preserve_curvature_when_saturating`` is enabled, rectangular speed
    and yaw-rate limits are applied with one common scale factor.  Therefore a
    feasible requested ratio ``w / v`` is retained by slowing the robot down
    instead of clipping only one channel.  The minimum-turn-radius bound is
    still authoritative: an impossible curvature is clipped to the measured
    physical boundary, and a zero-speed request can never become in-place yaw.
    """
    projected = commands.clone()
    if preserve_curvature_when_saturating:
        epsilon = torch.finfo(projected.dtype).eps
        linear_scale = float(maximum_forward_speed) / torch.clamp(
            torch.abs(projected[:, 0]), min=epsilon
        )
        yaw_scale = float(maximum_yaw_rate) / torch.clamp(
            torch.abs(projected[:, 1]), min=epsilon
        )
        common_scale = torch.minimum(
            torch.ones_like(linear_scale), torch.minimum(linear_scale, yaw_scale)
        )
        projected[:, :2] *= common_scale.unsqueeze(1)
    else:
        projected[:, 0] = torch.clamp(
            projected[:, 0],
            -float(maximum_forward_speed),
            float(maximum_forward_speed),
        )
    authority_velocity = projected[:, 0]
    if authority_forward_velocity is not None:
        measured_authority = torch.abs(
            authority_forward_velocity.to(
                device=projected.device, dtype=projected.dtype
            ).reshape(-1)
        ) + float(authority_speed_preview_margin)
        authority_speed = torch.minimum(
            torch.abs(projected[:, 0]), measured_authority
        )
        # ``feasible_yaw_rate_limit`` uses only the magnitude, so a positive
        # authority speed is sufficient and remains compatible with Torch 1.10.
        authority_velocity = authority_speed
    yaw_limit = feasible_yaw_rate_limit(
        authority_velocity,
        maximum_yaw_rate,
        minimum_turn_radius,
        envelope_fraction,
        turn_authority_start_speed,
        turn_authority_full_speed,
    )
    projected[:, 1] = torch.maximum(
        torch.minimum(projected[:, 1], yaw_limit),
        -yaw_limit,
    )
    if (
        curvature_fraction_breakpoints is not None
        and curvature_max_speed_values is not None
    ):
        epsilon = torch.finfo(projected.dtype).eps
        curvature_fraction = torch.clamp(
            torch.abs(projected[:, 1])
            * float(minimum_turn_radius)
            / torch.clamp(torch.abs(projected[:, 0]), min=epsilon),
            0.0,
            1.0,
        )
        stable_speed_limit = piecewise_linear_schedule(
            curvature_fraction,
            curvature_fraction_breakpoints,
            curvature_max_speed_values,
        )
        stable_scale = torch.minimum(
            torch.ones_like(stable_speed_limit),
            stable_speed_limit
            / torch.clamp(torch.abs(projected[:, 0]), min=epsilon),
        )
        projected[:, :2] *= stable_scale.unsqueeze(1)
    if stationary_threshold > 0.0:
        stationary = torch.abs(projected[:, 0]) < float(stationary_threshold)
        projected[stationary, 1] = 0.0
    return projected


def advance_correlated_velocity_commands(
    commands,
    linear_step,
    yaw_step,
    minimum_speed,
    maximum_forward_speed,
    maximum_yaw_rate,
    minimum_turn_radius,
    envelope_fraction=1.0,
    stationary_threshold=0.0,
    turn_authority_start_speed=0.0,
    turn_authority_full_speed=0.0,
):
    """Advance a bounded 5 Hz command random walk without flipping drive sign.

    SRU outputs are temporally correlated, not independent full-range samples
    every 0.2 s.  This helper perturbs speed magnitude and yaw rate locally,
    then projects the result into the same physical ``(v, w)`` domain used by
    training and deployment.
    """
    updated = commands[:, :2].clone()
    drive_sign = torch.where(
        updated[:, 0] >= 0.0,
        torch.ones_like(updated[:, 0]),
        -torch.ones_like(updated[:, 0]),
    )
    speed_delta = (
        2.0 * torch.rand_like(updated[:, 0]) - 1.0
    ) * float(linear_step)
    speed = torch.clamp(
        torch.abs(updated[:, 0]) + speed_delta,
        float(minimum_speed),
        float(maximum_forward_speed),
    )
    updated[:, 0] = drive_sign * speed
    updated[:, 1] += (
        2.0 * torch.rand_like(updated[:, 1]) - 1.0
    ) * float(yaw_step)
    return project_velocity_commands(
        updated,
        maximum_forward_speed,
        maximum_yaw_rate,
        minimum_turn_radius,
        envelope_fraction,
        stationary_threshold=stationary_threshold,
        turn_authority_start_speed=turn_authority_start_speed,
        turn_authority_full_speed=turn_authority_full_speed,
    )


def nominal_actuator_actions(
    commands,
    forward_speed_per_action=0.40,
    yaw_gain_intercept=0.0915,
    yaw_gain_speed_slope=0.175,
    drive_speed_breakpoints=None,
    drive_action_values=None,
    steering_speed_breakpoints=None,
    steering_half_fraction_scales=None,
    steering_full_fraction_scales=None,
    maximum_yaw_rate=0.10,
    minimum_turn_radius=2.0,
):
    """Map feasible ``[v, w]`` commands to normalized Rotunbot actions.

    The map is the symmetric nominal inverse identified by the fixed-action
    tests.  It supplies a physically valid baseline; PPO learns only a bounded
    residual around it.
    """
    actions = torch.zeros_like(commands[:, :2])
    if drive_speed_breakpoints is None or drive_action_values is None:
        actions[:, 0] = torch.clamp(
            commands[:, 0] / float(forward_speed_per_action),
            -1.0,
            1.0,
        )
    else:
        drive_magnitude = piecewise_linear_schedule(
            torch.abs(commands[:, 0]),
            drive_speed_breakpoints,
            drive_action_values,
        )
        actions[:, 0] = torch.clamp(
            torch.sign(commands[:, 0]) * drive_magnitude, -1.0, 1.0
        )
    yaw_gain = float(yaw_gain_intercept) + float(yaw_gain_speed_slope) * torch.abs(
        commands[:, 0]
    )
    actions[:, 1] = torch.clamp(
        -torch.sign(commands[:, 0])
        * commands[:, 1]
        / yaw_gain,
        -1.0,
        1.0,
    )
    steering_tables = (
        steering_speed_breakpoints,
        steering_half_fraction_scales,
        steering_full_fraction_scales,
    )
    if all(table is not None for table in steering_tables):
        speed = torch.abs(commands[:, 0])
        yaw_limit = torch.minimum(
            torch.full_like(speed, float(maximum_yaw_rate)),
            speed / max(float(minimum_turn_radius), 1.0e-8),
        )
        yaw_fraction = torch.where(
            yaw_limit > 1.0e-8,
            torch.clamp(torch.abs(commands[:, 1]) / yaw_limit, 0.0, 1.0),
            torch.zeros_like(yaw_limit),
        )
        half_scale = piecewise_linear_schedule(
            speed,
            steering_speed_breakpoints,
            steering_half_fraction_scales,
        )
        full_scale = piecewise_linear_schedule(
            speed,
            steering_speed_breakpoints,
            steering_full_fraction_scales,
        )
        upper_fraction = torch.clamp((yaw_fraction - 0.5) / 0.5, 0.0, 1.0)
        steering_scale = half_scale + (full_scale - half_scale) * upper_fraction
        actions[:, 1] = torch.clamp(actions[:, 1] * steering_scale, -1.0, 1.0)
    return actions


def lead_compensated_velocity_commands(
    commands,
    command_rates,
    linear_lead_time,
    angular_lead_time,
    maximum_forward_speed,
    maximum_yaw_rate,
    minimum_turn_radius,
    envelope_fraction=1.0,
    stationary_threshold=0.0,
    turn_authority_start_speed=0.0,
    turn_authority_full_speed=0.0,
):
    """Lead the governed reference without leaving the feasible command set.

    The nominal inverse is a steady-state map and therefore lags a changing
    reference.  A bounded look-ahead based on the *governed* command rate adds
    the missing acceleration feedforward while retaining the sphere's speed,
    yaw-rate, and minimum-turn-radius constraints.
    """
    led = commands[:, :2].clone()
    led[:, 0] += float(linear_lead_time) * command_rates[:, 0]
    led[:, 1] += float(angular_lead_time) * command_rates[:, 1]
    return project_velocity_commands(
        led,
        maximum_forward_speed,
        maximum_yaw_rate,
        minimum_turn_radius,
        envelope_fraction,
        stationary_threshold=stationary_threshold,
        turn_authority_start_speed=turn_authority_start_speed,
        turn_authority_full_speed=turn_authority_full_speed,
    )


def velocity_error_feedback_actions(
    commands,
    measured_forward_velocity,
    measured_yaw_rate,
    forward_speed_per_action=0.40,
    yaw_gain_intercept=0.0915,
    yaw_gain_speed_slope=0.175,
    linear_feedback_gain=0.0,
    angular_feedback_gain=0.20,
    linear_action_limit=0.0,
    angular_action_limit=0.15,
    stationary_threshold=0.02,
    wrong_direction_angular_feedback_gain=None,
    wrong_direction_command_threshold=0.01,
):
    """Convert measured ``(v, w)`` error into bounded actuator corrections.

    The calibrated inverse map is accurate at steady state, but a sphere keeps
    substantial linear and yaw momentum after an abrupt command change.  This
    proportional feedback supplies the predictable braking/correction term;
    PPO remains responsible only for the smaller nonlinear residual.

    When the command governor temporarily requests a full stop, the measured
    forward direction selects the steering sign so yaw momentum is actively
    opposed instead of waiting for passive contact damping.
    """
    measured_forward_velocity = measured_forward_velocity.reshape(-1)
    measured_yaw_rate = measured_yaw_rate.reshape(-1)
    feedback = torch.zeros_like(commands[:, :2])

    linear_error = commands[:, 0] - measured_forward_velocity
    linear_gain = torch.as_tensor(
        linear_feedback_gain,
        dtype=commands.dtype,
        device=commands.device,
    )
    feedback[:, 0] = torch.clamp(
        linear_gain * linear_error / float(forward_speed_per_action),
        -float(linear_action_limit),
        float(linear_action_limit),
    )

    commanded_motion = torch.abs(commands[:, 0]) >= float(stationary_threshold)
    drive_direction = torch.where(
        commanded_motion,
        torch.sign(commands[:, 0]),
        torch.sign(measured_forward_velocity),
    )
    effective_speed = torch.maximum(
        torch.abs(commands[:, 0]),
        torch.abs(measured_forward_velocity),
    )
    yaw_gain = (
        float(yaw_gain_intercept)
        + float(yaw_gain_speed_slope) * effective_speed
    )
    yaw_error = commands[:, 1] - measured_yaw_rate
    angular_gain = torch.as_tensor(
        angular_feedback_gain,
        dtype=commands.dtype,
        device=commands.device,
    )
    if wrong_direction_angular_feedback_gain is not None:
        meaningful_command = (
            torch.abs(commands[:, 1])
            >= float(wrong_direction_command_threshold)
        )
        wrong_direction = meaningful_command & (
            commands[:, 1] * measured_yaw_rate < 0.0
        )
        wrong_direction_gain = torch.as_tensor(
            wrong_direction_angular_feedback_gain,
            dtype=commands.dtype,
            device=commands.device,
        )
        angular_gain = torch.where(
            wrong_direction,
            wrong_direction_gain,
            angular_gain,
        )
    feedback[:, 1] = torch.clamp(
        -drive_direction
        * angular_gain
        * yaw_error
        / yaw_gain,
        -float(angular_action_limit),
        float(angular_action_limit),
    )
    return feedback


def velocity_error_integral_actions(
    commands,
    measured_forward_velocity,
    error_integral,
    forward_speed_per_action=0.40,
    yaw_gain_intercept=0.0915,
    yaw_gain_speed_slope=0.175,
    linear_integral_gain=0.0,
    angular_integral_gain=0.0,
    linear_action_limit=0.0,
    angular_action_limit=0.0,
    stationary_threshold=0.02,
):
    """Convert bounded integrated ``(v, w)`` error into actuator correction."""
    measured_forward_velocity = measured_forward_velocity.reshape(-1)
    integral = torch.zeros_like(commands[:, :2])
    integral[:, 0] = torch.clamp(
        float(linear_integral_gain)
        * error_integral[:, 0]
        / float(forward_speed_per_action),
        -float(linear_action_limit),
        float(linear_action_limit),
    )
    commanded_motion = torch.abs(commands[:, 0]) >= float(stationary_threshold)
    drive_direction = torch.where(
        commanded_motion,
        torch.sign(commands[:, 0]),
        torch.sign(measured_forward_velocity),
    )
    effective_speed = torch.maximum(
        torch.abs(commands[:, 0]),
        torch.abs(measured_forward_velocity),
    )
    yaw_gain = (
        float(yaw_gain_intercept)
        + float(yaw_gain_speed_slope) * effective_speed
    )
    integral[:, 1] = torch.clamp(
        -drive_direction
        * float(angular_integral_gain)
        * error_integral[:, 1]
        / yaw_gain,
        -float(angular_action_limit),
        float(angular_action_limit),
    )
    return integral


def velocity_error_derivative_actions(
    commands,
    measured_forward_velocity,
    error_derivative,
    forward_speed_per_action=0.40,
    yaw_gain_intercept=0.0915,
    yaw_gain_speed_slope=0.175,
    linear_derivative_gain=0.0,
    angular_derivative_gain=0.0,
    linear_action_limit=0.0,
    angular_action_limit=0.0,
    stationary_threshold=0.02,
):
    """Convert filtered tracking-error derivatives into actuator correction."""
    measured_forward_velocity = measured_forward_velocity.reshape(-1)
    derivative = torch.zeros_like(commands[:, :2])
    derivative[:, 0] = torch.clamp(
        float(linear_derivative_gain)
        * error_derivative[:, 0]
        / float(forward_speed_per_action),
        -float(linear_action_limit),
        float(linear_action_limit),
    )
    commanded_motion = torch.abs(commands[:, 0]) >= float(stationary_threshold)
    drive_direction = torch.where(
        commanded_motion,
        torch.sign(commands[:, 0]),
        torch.sign(measured_forward_velocity),
    )
    effective_speed = torch.maximum(
        torch.abs(commands[:, 0]),
        torch.abs(measured_forward_velocity),
    )
    yaw_gain = (
        float(yaw_gain_intercept)
        + float(yaw_gain_speed_slope) * effective_speed
    )
    derivative[:, 1] = torch.clamp(
        -drive_direction
        * float(angular_derivative_gain)
        * error_derivative[:, 1]
        / yaw_gain,
        -float(angular_action_limit),
        float(angular_action_limit),
    )
    return derivative


def velocity_rate_feedforward_actions(
    commands,
    command_rates,
    measured_forward_velocity,
    forward_speed_per_action=0.40,
    yaw_gain_intercept=0.0915,
    yaw_gain_speed_slope=0.175,
    linear_preview_time=0.0,
    angular_preview_time=0.0,
    linear_action_limit=0.0,
    angular_action_limit=0.0,
    stationary_threshold=0.02,
):
    """Convert command derivatives into bounded actuator-space feedforward.

    This channel is deliberately separate from command lead compensation.
    Applying a future yaw command before the feasible-command projection can
    distort the requested curvature.  Here ``dw/dt`` is converted directly to
    the steering-action sign and scale required by the spherical robot.
    """
    measured_forward_velocity = measured_forward_velocity.reshape(-1)
    feedforward = torch.zeros_like(commands[:, :2])
    feedforward[:, 0] = torch.clamp(
        float(linear_preview_time)
        * command_rates[:, 0]
        / float(forward_speed_per_action),
        -float(linear_action_limit),
        float(linear_action_limit),
    )

    commanded_motion = torch.abs(commands[:, 0]) >= float(stationary_threshold)
    drive_direction = torch.where(
        commanded_motion,
        torch.sign(commands[:, 0]),
        torch.sign(measured_forward_velocity),
    )
    effective_speed = torch.maximum(
        torch.abs(commands[:, 0]),
        torch.abs(measured_forward_velocity),
    )
    yaw_gain = (
        float(yaw_gain_intercept)
        + float(yaw_gain_speed_slope) * effective_speed
    )
    feedforward[:, 1] = torch.clamp(
        -drive_direction
        * float(angular_preview_time)
        * command_rates[:, 1]
        / yaw_gain,
        -float(angular_action_limit),
        float(angular_action_limit),
    )
    return feedforward


def command_target_gap_mask(
    target_commands,
    governed_commands,
    linear_gap_threshold,
    angular_gap_threshold,
):
    """Identify causally smooth requests from the target/governor gap.

    A feasible continuously changing target stays close to the governed command,
    while an abrupt step remains far away until the acceleration-limited ramp has
    finished.  This lets dynamic feedforward help smooth references without
    injecting the same kick into step and reversal tests.
    """
    gap = torch.abs(target_commands[:, :2] - governed_commands[:, :2])
    return (gap[:, 0] <= float(linear_gap_threshold)) & (
        gap[:, 1] <= float(angular_gap_threshold)
    )


def error_aligned_residual_actions(
    actions,
    commands,
    measured_forward_velocity,
    measured_yaw_rate,
    stationary_threshold=0.02,
):
    """Remove residual components that would increase instantaneous error.

    This is an action-space safety projection, not a teacher action.  PPO still
    chooses the correction magnitude, while the fixed controller remains the
    fallback when a sampled residual points in the physically wrong direction.
    """
    filtered = actions.clone()
    linear_error = commands[:, 0] - measured_forward_velocity.reshape(-1)
    linear_aligned = filtered[:, 0] * linear_error > 0.0
    filtered[:, 0] = torch.where(
        linear_aligned, filtered[:, 0], torch.zeros_like(filtered[:, 0])
    )

    commanded_motion = torch.abs(commands[:, 0]) >= float(stationary_threshold)
    drive_direction = torch.where(
        commanded_motion,
        torch.sign(commands[:, 0]),
        torch.sign(measured_forward_velocity.reshape(-1)),
    )
    yaw_error = commands[:, 1] - measured_yaw_rate.reshape(-1)
    desired_steering_sign = -drive_direction * yaw_error
    angular_aligned = filtered[:, 1] * desired_steering_sign > 0.0
    filtered[:, 1] = torch.where(
        angular_aligned, filtered[:, 1], torch.zeros_like(filtered[:, 1])
    )
    return filtered


def rate_limit_velocity_commands(
    current_commands,
    target_commands,
    maximum_linear_acceleration,
    maximum_yaw_acceleration,
    dt,
    maximum_forward_speed,
    maximum_yaw_rate,
    minimum_turn_radius,
    envelope_fraction=1.0,
    stationary_threshold=0.0,
    turn_authority_start_speed=0.0,
    turn_authority_full_speed=0.0,
    authority_forward_velocity=None,
    authority_speed_preview_margin=0.0,
):
    """Advance raw targets by one physically feasible command-governor step."""
    delta_limit = torch.as_tensor(
        [
            float(maximum_linear_acceleration) * float(dt),
            float(maximum_yaw_acceleration) * float(dt),
        ],
        dtype=current_commands.dtype,
        device=current_commands.device,
    )
    governed = current_commands + torch.maximum(
        torch.minimum(target_commands - current_commands, delta_limit),
        -delta_limit,
    )
    return project_velocity_commands(
        governed,
        maximum_forward_speed,
        maximum_yaw_rate,
        minimum_turn_radius,
        envelope_fraction,
        stationary_threshold=stationary_threshold,
        turn_authority_start_speed=turn_authority_start_speed,
        turn_authority_full_speed=turn_authority_full_speed,
        authority_forward_velocity=authority_forward_velocity,
        authority_speed_preview_margin=authority_speed_preview_margin,
    )


def reversal_brake_mask(
    current_commands,
    target_commands,
    measured_forward_speed,
    measured_yaw_rate,
    linear_threshold,
    yaw_threshold,
    yaw_deceleration_ratio=None,
    yaw_deceleration_delta=0.0,
    linear_deceleration_ratio=None,
    linear_deceleration_delta=0.0,
    linear_deceleration_target_speed_max=None,
    include_yaw=True,
):
    """Identify direction changes or strong speed reductions needing a stop.

    The applied command is normally the best description of the current motion
    direction.  Near zero command, however, the sphere may still carry linear
    or yaw momentum, so the measured motion becomes authoritative.
    """
    linear_threshold = float(linear_threshold)
    yaw_threshold = float(yaw_threshold)
    source_v = torch.where(
        torch.abs(measured_forward_speed) > torch.abs(current_commands[:, 0]),
        measured_forward_speed,
        current_commands[:, 0],
    )
    source_w = torch.where(
        torch.abs(measured_yaw_rate) > torch.abs(current_commands[:, 1]),
        measured_yaw_rate,
        current_commands[:, 1],
    )
    reverse_v = (
        (torch.abs(source_v) >= linear_threshold)
        & (torch.abs(target_commands[:, 0]) >= linear_threshold)
        & (source_v * target_commands[:, 0] < 0.0)
    )
    reverse_w = torch.zeros_like(reverse_v)
    if include_yaw:
        reverse_w = (
            (torch.abs(source_w) >= yaw_threshold)
            & (torch.abs(target_commands[:, 1]) >= yaw_threshold)
            & (source_w * target_commands[:, 1] < 0.0)
        )
    brake_linear_reduction = torch.zeros_like(reverse_v)
    if linear_deceleration_ratio is not None:
        target_v = target_commands[:, 0]
        same_linear_direction = source_v * target_v >= 0.0
        brake_linear_reduction = (
            same_linear_direction
            & (torch.abs(source_v) >= linear_threshold)
            & (
                torch.abs(target_v)
                < float(linear_deceleration_ratio) * torch.abs(source_v)
            )
            & (
                torch.abs(target_v - source_v)
                >= float(linear_deceleration_delta)
            )
        )
        if linear_deceleration_target_speed_max is not None:
            brake_linear_reduction &= torch.abs(target_v) <= float(
                linear_deceleration_target_speed_max
            )
    brake_yaw_reduction = torch.zeros_like(reverse_w)
    if include_yaw and yaw_deceleration_ratio is not None:
        target_w = target_commands[:, 1]
        same_direction = source_w * target_w >= 0.0
        brake_yaw_reduction = (
            same_direction
            & (torch.abs(source_w) >= yaw_threshold)
            & (
                torch.abs(target_w)
                < float(yaw_deceleration_ratio) * torch.abs(source_w)
            )
            & (
                torch.abs(target_w - source_w)
                >= float(yaw_deceleration_delta)
            )
        )
    return reverse_v | reverse_w | brake_linear_reduction | brake_yaw_reduction


def yaw_reversal_brake_mask(
    current_commands,
    target_commands,
    measured_yaw_rate,
    yaw_threshold,
    yaw_deceleration_ratio=None,
    yaw_deceleration_delta=0.0,
):
    """Detect yaw changes that should pass through straight rolling first.

    Unlike a linear reversal, a spherical robot must retain translational
    motion to preserve useful steering authority.  This mask therefore drives
    only the governed yaw command to zero; it never requests an in-place turn.
    """
    yaw_threshold = float(yaw_threshold)
    source_w = torch.where(
        torch.abs(measured_yaw_rate) > torch.abs(current_commands[:, 1]),
        measured_yaw_rate,
        current_commands[:, 1],
    )
    target_w = target_commands[:, 1]
    reverse_w = (
        (torch.abs(source_w) >= yaw_threshold)
        & (torch.abs(target_w) >= yaw_threshold)
        & (source_w * target_w < 0.0)
    )
    brake_yaw_reduction = torch.zeros_like(reverse_w)
    if yaw_deceleration_ratio is not None:
        same_direction = source_w * target_w >= 0.0
        brake_yaw_reduction = (
            same_direction
            & (torch.abs(source_w) >= yaw_threshold)
            & (
                torch.abs(target_w)
                < float(yaw_deceleration_ratio) * torch.abs(source_w)
            )
            & (
                torch.abs(target_w - source_w)
                >= float(yaw_deceleration_delta)
            )
        )
    return reverse_w | brake_yaw_reduction


def command_request_jump_mask(
    previous_targets,
    new_targets,
    minimum_linear_jump,
    minimum_yaw_jump,
):
    """Return requests discontinuous enough to justify a full-stop brake."""
    delta = torch.abs(new_targets - previous_targets)
    return (delta[:, 0] >= float(minimum_linear_jump)) | (
        delta[:, 1] >= float(minimum_yaw_jump)
    )


def tracking_integral_reset_mask(request_jump, smooth_profile):
    """Reset PI memory only for genuinely discontinuous external requests.

    A continuously updated trajectory changes numerically every policy step,
    but those small changes are not command steps. Clearing the integral for
    each such update disables integral feedback in deployment and evaluation.
    """
    return request_jump & ~smooth_profile


def update_rate_gated_error_integral(
    error_integral,
    tracking_error,
    command_rates,
    dt,
    leak_rate,
    integral_limits,
    command_rate_thresholds,
):
    """Apply leaky PI memory only while each command channel is steady."""
    decay = max(0.0, 1.0 - max(0.0, float(leak_rate)) * float(dt))
    updated = error_integral * decay + tracking_error * float(dt)
    limits = torch.as_tensor(
        integral_limits, dtype=updated.dtype, device=updated.device
    )
    thresholds = torch.as_tensor(
        command_rate_thresholds, dtype=updated.dtype, device=updated.device
    )
    command_is_changing = torch.abs(command_rates) > thresholds
    updated = torch.where(command_is_changing, torch.zeros_like(updated), updated)
    return torch.maximum(torch.minimum(updated, limits), -limits)


def smooth_feasible_velocity_profile(
    phase,
    speed_amplitude,
    signed_curvature,
    maximum_forward_speed,
    maximum_yaw_rate,
    minimum_turn_radius,
    envelope_fraction=1.0,
    stationary_threshold=0.0,
    turn_authority_start_speed=0.0,
    turn_authority_full_speed=0.0,
):
    """Generate a smooth, zero-crossing-safe feasible ``[v, w]`` profile."""
    velocity = speed_amplitude * torch.sin(phase)
    yaw_rate = signed_curvature * velocity
    return project_velocity_commands(
        torch.stack((velocity, yaw_rate), dim=1),
        maximum_forward_speed,
        maximum_yaw_rate,
        minimum_turn_radius,
        envelope_fraction,
        stationary_threshold=stationary_threshold,
        turn_authority_start_speed=turn_authority_start_speed,
        turn_authority_full_speed=turn_authority_full_speed,
    )


def independent_feasible_velocity_profile(
    phase,
    velocity_offset,
    velocity_amplitude,
    yaw_amplitude,
    yaw_phase_offset,
    yaw_frequency_ratio,
    maximum_forward_speed,
    maximum_yaw_rate,
    minimum_turn_radius,
    envelope_fraction=1.0,
    stationary_threshold=0.0,
    turn_authority_start_speed=0.0,
    turn_authority_full_speed=0.0,
):
    """Generate independently phased smooth v/w commands, then project them.

    Unlike the legacy constant-curvature profile ``w = curvature * v``, this
    family includes constant-v/alternating-w motion and independent v/w phase
    and frequency. Projection is deliberately the final operation so every
    sample still obeys the spherical robot's nonholonomic feasible set.
    """
    velocity = velocity_offset + velocity_amplitude * torch.sin(phase)
    yaw_rate = yaw_amplitude * torch.sin(
        yaw_frequency_ratio * phase + yaw_phase_offset
    )
    return project_velocity_commands(
        torch.stack((velocity, yaw_rate), dim=1),
        maximum_forward_speed,
        maximum_yaw_rate,
        minimum_turn_radius,
        envelope_fraction,
        stationary_threshold=stationary_threshold,
        turn_authority_start_speed=turn_authority_start_speed,
        turn_authority_full_speed=turn_authority_full_speed,
    )


class RotunbotVel(LeggedRobot):
    """Track feasible body-forward speed and heading yaw-rate commands."""

    cfg: RotunbotVelCfg

    def _process_dof_props(self, props, env_id):
        # The Rotunbot URDF does not declare the effort drive used by the
        # custom torque controller.  Without this override PhysX keeps the
        # joint in its imported/default drive mode and ignores our torques.
        props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.0)
        return super()._process_dof_props(props, env_id)

    def _init_buffers(self):
        super()._init_buffers()
        self.upper_level_command_interval_steps = command_update_interval_steps(
            self.dt,
            getattr(
                self.cfg.commands,
                "upper_level_command_frequency_hz",
                None,
            ),
        )
        print(
            "[RotunbotVel] timing: "
            f"physics={1.0 / float(self.sim_params.dt):.1f} Hz, "
            f"low_level={1.0 / float(self.dt):.1f} Hz, "
            "upper_command="
            f"{1.0 / (float(self.dt) * self.upper_level_command_interval_steps):.1f} Hz, "
            f"hold={self.upper_level_command_interval_steps} low-level steps"
        )
        self.command_targets = self.commands[:, :2].clone()
        self.applied_feasible_command = self.commands[:, :2].clone()
        self.feasible_transition_manager = None
        self.transition_state = torch.full(
            (self.num_envs,), TransitionState.TRACK, dtype=torch.long, device=self.device
        )
        self.transition_anchor_command = torch.zeros_like(self.command_targets)
        self.transition_latest_target = torch.zeros_like(self.command_targets)
        self.transition_progress = torch.zeros(
            self.num_envs, dtype=self.command_targets.dtype, device=self.device
        )
        self.transition_settle_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.transition_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if bool(
            getattr(
                self.cfg.commands,
                "feasible_transition_manager_enabled",
                False,
            )
        ):
            self.feasible_transition_manager = FeasibleVelocityTransitionManager(
                num_envs=self.num_envs,
                device=self.device,
                dtype=self.command_targets.dtype,
                dt=self.dt,
                maximum_linear_acceleration=self.cfg.commands.maximum_linear_acceleration,
                maximum_yaw_acceleration=self.cfg.commands.maximum_yaw_acceleration,
                maximum_forward_speed=self.cfg.commands.governor_projection_max_forward_speed,
                maximum_yaw_rate=self.cfg.commands.max_yaw_rate,
                minimum_turn_radius=self.cfg.commands.minimum_turn_radius,
                envelope_fraction=self.cfg.commands.feasible_envelope_fraction,
                stationary_threshold=self.cfg.rewards.stationary_command_threshold,
                reversal_detection_v=self.cfg.commands.reversal_detection_v,
                reversal_detection_w=self.cfg.commands.reversal_detection_w,
                reversal_minimum_request_jump_v=self.cfg.commands.reversal_minimum_request_jump_v,
                reversal_minimum_request_jump_w=self.cfg.commands.reversal_minimum_request_jump_w,
                settle_v_threshold=self.cfg.commands.transition_settle_v_threshold,
                settle_w_threshold=self.cfg.commands.transition_settle_w_threshold,
                settle_time=self.cfg.commands.transition_settle_time,
                curvature_fraction_breakpoints=getattr(
                    self.cfg.commands,
                    "stable_curvature_fraction_breakpoints",
                    None,
                ),
                curvature_max_speed_values=getattr(
                    self.cfg.commands,
                    "stable_curvature_max_speed_values",
                    None,
                ),
            )
            self.transition_state = self.feasible_transition_manager.transition_state
            self.transition_anchor_command = (
                self.feasible_transition_manager.transition_anchor_command
            )
            self.transition_latest_target = (
                self.feasible_transition_manager.transition_latest_target
            )
            self.transition_progress = self.feasible_transition_manager.transition_progress
            self.transition_settle_counter = (
                self.feasible_transition_manager.transition_settle_counter
            )
            self.transition_active = self.feasible_transition_manager.transition_active
        self.command_brake_pending = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.command_yaw_brake_pending = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.command_rates = torch.zeros_like(self.command_targets)
        # Optional causal rate observation for an exact sampled 5 Hz command:
        # (new_request - old_request) / 0.2 s is held until the next upper tick.
        # Legacy tasks retain the historical one-step finite difference.
        self.held_upper_command_rates = torch.zeros_like(self.command_targets)
        self.command_rate_hold_steps_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.tracking_error_integral = torch.zeros_like(self.command_targets)
        self.last_tracking_error = torch.zeros_like(self.command_targets)
        self.tracking_error_derivative = torch.zeros_like(self.command_targets)
        self.residual_yaw_error_sign = torch.zeros(
            self.num_envs, device=self.device
        )
        self.residual_yaw_error_persistent_time = torch.zeros(
            self.num_envs, device=self.device
        )
        self.residual_yaw_error_cooldown_time = torch.zeros(
            self.num_envs, device=self.device
        )
        self.residual_yaw_error_gate_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.residual_yaw_error_gate = torch.zeros(
            self.num_envs, device=self.device
        )
        self.command_profile_is_smooth = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.command_profile_is_random_walk = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Separate the internal profile generator state from the controller's
        # reference classification.  External evaluators can mark a supplied
        # trajectory smooth without activating the built-in profile generator.
        self.command_reference_is_smooth = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.command_profile_phase = torch.zeros(self.num_envs, device=self.device)
        self.command_profile_period = torch.ones(self.num_envs, device=self.device)
        self.command_profile_speed_amplitude = torch.zeros(
            self.num_envs, device=self.device
        )
        self.command_profile_signed_curvature = torch.zeros(
            self.num_envs, device=self.device
        )
        self.command_profile_is_independent = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.command_profile_velocity_offset = torch.zeros(
            self.num_envs, device=self.device
        )
        self.command_profile_velocity_amplitude = torch.zeros(
            self.num_envs, device=self.device
        )
        self.command_profile_yaw_amplitude = torch.zeros(
            self.num_envs, device=self.device
        )
        self.command_profile_yaw_phase_offset = torch.zeros(
            self.num_envs, device=self.device
        )
        self.command_profile_yaw_frequency_ratio = torch.ones(
            self.num_envs, device=self.device
        )
        # Compatibility alias for reports produced before the v12 governor.
        self.command_reversal_pending = self.command_brake_pending
        self.requested_output_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            device=self.device,
        )
        self.output_actions = torch.zeros_like(self.requested_output_actions)
        self.last_output_actions = torch.zeros_like(self.requested_output_actions)
        self.nominal_policy_actions = torch.zeros_like(self.requested_output_actions)
        self.feedback_policy_actions = torch.zeros_like(self.requested_output_actions)
        self.derivative_feedback_policy_actions = torch.zeros_like(
            self.requested_output_actions
        )
        self.rate_feedforward_policy_actions = torch.zeros_like(
            self.requested_output_actions
        )
        self.combined_policy_actions = torch.zeros_like(self.requested_output_actions)
        self.applied_residual_actions = torch.zeros_like(self.requested_output_actions)
        self.tracking_heading = yaw_from_quaternion(self.base_quat).clone()
        self.tracking_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.tracking_ang_vel = torch.zeros_like(self.base_ang_vel)
        self._update_tracking_motion(integrate_heading=False)

    def _update_tracking_motion(self, integrate_heading=True):
        if integrate_heading:
            self.tracking_heading.add_(self.root_states[:, 12] * float(self.dt))
            self.tracking_heading.copy_(
                torch.atan2(
                    torch.sin(self.tracking_heading),
                    torch.cos(self.tracking_heading),
                )
            )
        self.tracking_lin_vel.copy_(
            planar_velocity_in_heading_frame(
                self.tracking_heading,
                self.root_states[:, 7:10],
            )
        )
        # Roll/pitch rates remain useful attitude feedback.  Heading rate is
        # the gravity-aligned world-Z rate, not body Z after the base has rolled.
        self.tracking_ang_vel.copy_(self.base_ang_vel)
        self.tracking_ang_vel[:, 2].copy_(self.root_states[:, 12])

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_tracking_motion()
        upper_command_tick = (
            self.common_step_counter % self.upper_level_command_interval_steps
            == 0
        )
        sine_profile = (
            self.command_profile_is_smooth
            & ~self.command_profile_is_random_walk
        )
        smooth_ids = sine_profile.nonzero(as_tuple=False).flatten()
        if smooth_ids.numel() > 0:
            next_phase = self.command_profile_phase[smooth_ids] + (
                2.0
                * torch.pi
                * float(self.dt)
                / self.command_profile_period[smooth_ids]
            )
            self.command_profile_phase[smooth_ids] = torch.remainder(
                next_phase, 2.0 * torch.pi
            )
            # The profile represents the high-level navigation request.  Its
            # phase evolves continuously, but the request is sampled and held
            # at the configured SRU frequency.  The 50 Hz governor below still
            # advances on every policy step.
            if not upper_command_tick:
                smooth_ids = smooth_ids[:0]
        if smooth_ids.numel() > 0:
            smooth_targets = smooth_feasible_velocity_profile(
                self.command_profile_phase[smooth_ids],
                self.command_profile_speed_amplitude[smooth_ids],
                self.command_profile_signed_curvature[smooth_ids],
                self.cfg.commands.max_forward_speed,
                self.cfg.commands.max_yaw_rate,
                self.cfg.commands.minimum_turn_radius,
                self.cfg.commands.feasible_envelope_fraction,
                stationary_threshold=self.cfg.rewards.stationary_command_threshold,
                turn_authority_start_speed=getattr(
                    self.cfg.commands, "turn_authority_start_speed", 0.0
                ),
                turn_authority_full_speed=getattr(
                    self.cfg.commands, "turn_authority_full_speed", 0.0
                ),
            )
            independent = self.command_profile_is_independent[smooth_ids]
            if torch.any(independent):
                independent_ids = smooth_ids[independent]
                smooth_targets[independent] = independent_feasible_velocity_profile(
                    self.command_profile_phase[independent_ids],
                    self.command_profile_velocity_offset[independent_ids],
                    self.command_profile_velocity_amplitude[independent_ids],
                    self.command_profile_yaw_amplitude[independent_ids],
                    self.command_profile_yaw_phase_offset[independent_ids],
                    self.command_profile_yaw_frequency_ratio[independent_ids],
                    self.cfg.commands.max_forward_speed,
                    self.cfg.commands.max_yaw_rate,
                    self.cfg.commands.minimum_turn_radius,
                    self.cfg.commands.feasible_envelope_fraction,
                    stationary_threshold=self.cfg.rewards.stationary_command_threshold,
                    turn_authority_start_speed=getattr(
                        self.cfg.commands, "turn_authority_start_speed", 0.0
                    ),
                    turn_authority_full_speed=getattr(
                        self.cfg.commands, "turn_authority_full_speed", 0.0
                    ),
                )
            self.set_command_targets(smooth_targets, smooth_ids)
        random_walk_ids = self.command_profile_is_random_walk.nonzero(
            as_tuple=False
        ).flatten()
        if not upper_command_tick:
            random_walk_ids = random_walk_ids[:0]
        if random_walk_ids.numel() > 0:
            random_walk_targets = advance_correlated_velocity_commands(
                self.command_targets[random_walk_ids],
                self.cfg.commands.random_walk_linear_step,
                self.cfg.commands.random_walk_yaw_step,
                self.cfg.commands.random_walk_minimum_speed,
                self.cfg.commands.max_forward_speed,
                self.cfg.commands.max_yaw_rate,
                self.cfg.commands.minimum_turn_radius,
                self.cfg.commands.feasible_envelope_fraction,
                stationary_threshold=self.cfg.rewards.stationary_command_threshold,
                turn_authority_start_speed=getattr(
                    self.cfg.commands, "turn_authority_start_speed", 0.0
                ),
                turn_authority_full_speed=getattr(
                    self.cfg.commands, "turn_authority_full_speed", 0.0
                ),
            )
            self.set_command_targets(random_walk_targets, random_walk_ids)
        previous_commands = self.commands[:, :2].clone()
        direct_tracking = bool(
            getattr(self.cfg.commands, "direct_command_tracking", False)
        )
        if self.feasible_transition_manager is not None:
            self.commands[:, :2] = self.feasible_transition_manager.advance(
                self.commands[:, :2],
                self.tracking_lin_vel[:, 0],
                self.tracking_ang_vel[:, 2],
            )[0]
            self.applied_feasible_command.copy_(self.commands[:, :2])
            self.command_brake_pending.copy_(
                self.transition_state == TransitionState.BRAKE_TO_ORIGIN
            )
            self.command_yaw_brake_pending.zero_()
        elif direct_tracking:
            # Exact upper/lower-layer contract: the requested reachable command
            # is the reference.  Mechanical inertia remains in the plant, but
            # no hidden governor substitutes a different v/w target.
            self.commands[:, :2].copy_(self.command_targets)
            self.command_brake_pending.zero_()
            self.command_yaw_brake_pending.zero_()
            self.applied_feasible_command.copy_(self.commands[:, :2])
        else:
            effective_targets = self.command_targets.clone()
            effective_targets[self.command_brake_pending] = 0.0
            effective_targets[self.command_yaw_brake_pending, 1] = 0.0
            governor_projection_max_forward_speed = getattr(
                self.cfg.commands,
                "governor_projection_max_forward_speed",
                None,
            )
            if governor_projection_max_forward_speed is None:
                governor_projection_max_forward_speed = (
                    self.cfg.commands.max_forward_speed
                )
            self.commands[:, :2] = rate_limit_velocity_commands(
                self.commands[:, :2],
                effective_targets,
                self.cfg.commands.maximum_linear_acceleration,
                self.cfg.commands.maximum_yaw_acceleration,
                self.dt,
                governor_projection_max_forward_speed,
                self.cfg.commands.max_yaw_rate,
                self.cfg.commands.minimum_turn_radius,
                self.cfg.commands.feasible_envelope_fraction,
                stationary_threshold=self.cfg.rewards.stationary_command_threshold,
                turn_authority_start_speed=getattr(
                    self.cfg.commands, "turn_authority_start_speed", 0.0
                ),
                turn_authority_full_speed=getattr(
                    self.cfg.commands, "turn_authority_full_speed", 0.0
                ),
                authority_forward_velocity=(
                    self.tracking_lin_vel[:, 0]
                    if bool(
                        getattr(
                            self.cfg.commands,
                            "use_measured_turn_authority",
                            False,
                        )
                    )
                    else None
                ),
                authority_speed_preview_margin=float(
                    getattr(
                        self.cfg.commands,
                        "turn_authority_speed_preview_margin",
                        0.0,
                    )
                ),
            )
            self.applied_feasible_command.copy_(self.commands[:, :2])
        self.command_rates.copy_(
            (self.commands[:, :2] - previous_commands) / float(self.dt)
        )
        if direct_tracking and bool(
            getattr(self.cfg.commands, "hold_upper_command_rate", False)
        ):
            active_rate = self.command_rate_hold_steps_remaining > 0
            self.command_rate_hold_steps_remaining[active_rate] -= 1
            active_rate = self.command_rate_hold_steps_remaining > 0
            self.command_rates.zero_()
            self.command_rates[active_rate] = self.held_upper_command_rates[
                active_rate
            ]
        if not direct_tracking:
            stopped = self.command_brake_pending & (
                torch.abs(self.commands[:, 0])
                <= float(self.cfg.commands.reversal_release_command_v)
            ) & (
                torch.abs(self.commands[:, 1])
                <= float(self.cfg.commands.reversal_release_command_w)
            ) & (
                torch.abs(self.tracking_lin_vel[:, 0])
                <= float(self.cfg.commands.reversal_release_measured_v)
            ) & (
                torch.abs(self.tracking_ang_vel[:, 2])
                <= float(self.cfg.commands.reversal_release_measured_w)
            )
            self.command_brake_pending[stopped] = False
            yaw_stopped = self.command_yaw_brake_pending & (
                torch.abs(self.commands[:, 1])
                <= float(self.cfg.commands.reversal_release_command_w)
            ) & (
                torch.abs(self.tracking_ang_vel[:, 2])
                <= float(self.cfg.commands.reversal_release_measured_w)
            )
            self.command_yaw_brake_pending[yaw_stopped] = False
        tracking_error = torch.stack(
            (
                self.commands[:, 0] - self.tracking_lin_vel[:, 0],
                self.commands[:, 1] - self.tracking_ang_vel[:, 2],
            ),
            dim=1,
        )
        raw_error_derivative = (
            tracking_error - self.last_tracking_error
        ) / float(self.dt)
        derivative_alpha = float(
            self.cfg.control.error_derivative_filter_alpha
        )
        derivative_alpha = min(1.0, max(0.0, derivative_alpha))
        self.tracking_error_derivative.mul_(1.0 - derivative_alpha).add_(
            raw_error_derivative, alpha=derivative_alpha
        )
        self.last_tracking_error.copy_(tracking_error)
        self.tracking_error_integral.copy_(
            update_rate_gated_error_integral(
                self.tracking_error_integral,
                tracking_error,
                self.command_rates,
                self.dt,
                self.cfg.control.integral_leak_rate,
                [
                    self.cfg.control.linear_error_integral_limit,
                    self.cfg.control.angular_error_integral_limit,
                ],
                self.cfg.control.integral_command_rate_threshold,
            )
        )
        if bool(
            getattr(
                self.cfg.control,
                "disable_integral_for_explicit_smooth_profiles",
                False,
            )
        ):
            # A sinusoid has zero instantaneous rate at every extremum, and a
            # feasibility-clipped sinusoid can have a long zero-rate plateau.
            # Those points are not steady commands.  Accumulating PI memory
            # there creates a cycle-to-cycle yaw bias and eventually reverses
            # the response.  Keep PI only for genuine constant/step targets.
            self.tracking_error_integral[self.command_reference_is_smooth] = 0.0
        self.tracking_error_integral[self.command_brake_pending] = 0.0
        self.tracking_error_derivative[self.command_brake_pending] = 0.0
        self.tracking_error_integral[self.command_yaw_brake_pending, 1] = 0.0
        self.tracking_error_derivative[self.command_yaw_brake_pending, 1] = 0.0

    def set_command_targets(self, target_commands, env_ids=None):
        """Set raw requests and latch a brake phase for genuine reversals."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        targets = target_commands.to(
            device=self.device, dtype=self.command_targets.dtype
        )
        if targets.ndim == 1:
            targets = targets.unsqueeze(0).expand(env_ids.numel(), -1)
        if targets.shape != (env_ids.numel(), 2):
            raise ValueError(
                "target_commands must have shape (len(env_ids), 2); "
                f"received {tuple(targets.shape)}"
            )

        if bool(
            getattr(
                self.cfg.commands,
                "project_external_commands_to_feasible_domain",
                False,
            )
        ):
            targets = project_velocity_commands(
                targets,
                self.cfg.commands.max_forward_speed,
                self.cfg.commands.max_yaw_rate,
                self.cfg.commands.minimum_turn_radius,
                self.cfg.commands.feasible_envelope_fraction,
                stationary_threshold=self.cfg.rewards.stationary_command_threshold,
                turn_authority_start_speed=getattr(
                    self.cfg.commands, "turn_authority_start_speed", 0.0
                ),
                turn_authority_full_speed=getattr(
                    self.cfg.commands, "turn_authority_full_speed", 0.0
                ),
                preserve_curvature_when_saturating=bool(
                    getattr(
                        self.cfg.commands,
                        "preserve_curvature_when_saturating",
                        False,
                    )
                ),
                curvature_fraction_breakpoints=getattr(
                    self.cfg.commands,
                    "stable_curvature_fraction_breakpoints",
                    None,
                ),
                curvature_max_speed_values=getattr(
                    self.cfg.commands,
                    "stable_curvature_max_speed_values",
                    None,
                ),
            )

        old_targets = self.command_targets[env_ids]
        changed = torch.any(torch.abs(targets - old_targets) > 1.0e-6, dim=1)
        if self.feasible_transition_manager is not None:
            if torch.any(changed) and bool(
                getattr(self.cfg.commands, "hold_upper_command_rate", False)
            ):
                changed_ids = env_ids[changed]
                upper_period = float(self.dt) * float(
                    self.upper_level_command_interval_steps
                )
                held_rate = (targets[changed] - old_targets[changed]) / max(
                    upper_period, 1.0e-8
                )
                self.held_upper_command_rates[changed_ids] = held_rate
                self.command_rate_hold_steps_remaining[changed_ids] = (
                    self.upper_level_command_interval_steps
                )
                self.command_rates[changed_ids] = held_rate
            self.feasible_transition_manager.update_target(
                targets,
                self.commands[env_ids, :2],
                self.tracking_lin_vel[env_ids, 0],
                self.tracking_ang_vel[env_ids, 2],
                env_ids=env_ids,
            )
            self.command_targets[env_ids] = targets
            self.command_brake_pending.copy_(
                self.transition_state == TransitionState.BRAKE_TO_ORIGIN
            )
            self.command_yaw_brake_pending.zero_()
            return

        if torch.any(changed):
            changed_ids = env_ids[changed]
            if bool(
                getattr(self.cfg.commands, "hold_upper_command_rate", False)
            ):
                upper_period = float(self.dt) * float(
                    self.upper_level_command_interval_steps
                )
                held_rate = (targets[changed] - old_targets[changed]) / max(
                    upper_period, 1.0e-8
                )
                self.held_upper_command_rates[changed_ids] = held_rate
                self.command_rate_hold_steps_remaining[changed_ids] = (
                    self.upper_level_command_interval_steps
                )
                # External high-level callers set the request immediately
                # before policy inference, so expose the causal slope in that
                # first observation as well as during the following hold.
                self.command_rates[changed_ids] = held_rate
            yaw_only_braking = bool(
                getattr(self.cfg.commands, "yaw_only_braking", False)
            )
            reversing = reversal_brake_mask(
                self.commands[changed_ids, :2],
                targets[changed],
                self.tracking_lin_vel[changed_ids, 0],
                self.tracking_ang_vel[changed_ids, 2],
                self.cfg.commands.reversal_detection_v,
                self.cfg.commands.reversal_detection_w,
                yaw_deceleration_ratio=(
                    self.cfg.commands.yaw_deceleration_brake_ratio
                ),
                yaw_deceleration_delta=(
                    self.cfg.commands.yaw_deceleration_brake_delta
                ),
                linear_deceleration_ratio=getattr(
                    self.cfg.commands, "linear_deceleration_brake_ratio", None
                ),
                linear_deceleration_delta=getattr(
                    self.cfg.commands, "linear_deceleration_brake_delta", 0.0
                ),
                linear_deceleration_target_speed_max=getattr(
                    self.cfg.commands,
                    "linear_deceleration_target_speed_max",
                    None,
                ),
                include_yaw=not yaw_only_braking,
            )
            yaw_braking = torch.zeros_like(reversing)
            if yaw_only_braking:
                yaw_braking = yaw_reversal_brake_mask(
                    self.commands[changed_ids, :2],
                    targets[changed],
                    self.tracking_ang_vel[changed_ids, 2],
                    self.cfg.commands.reversal_detection_w,
                    yaw_deceleration_ratio=(
                        self.cfg.commands.yaw_deceleration_brake_ratio
                    ),
                    yaw_deceleration_delta=(
                        self.cfg.commands.yaw_deceleration_brake_delta
                    ),
                )
            request_jump = command_request_jump_mask(
                old_targets[changed],
                targets[changed],
                self.cfg.commands.reversal_minimum_request_jump_v,
                self.cfg.commands.reversal_minimum_request_jump_w,
            )
            reversing &= request_jump
            yaw_braking &= request_jump & ~reversing
            self.command_brake_pending[changed_ids] = reversing
            self.command_yaw_brake_pending[changed_ids] = yaw_braking
            if hasattr(self, "tracking_error_integral"):
                reset_integral = tracking_integral_reset_mask(
                    request_jump,
                    self.command_profile_is_smooth[changed_ids],
                )
                discontinuous_ids = changed_ids[reset_integral]
                self.tracking_error_integral[discontinuous_ids] = 0.0
        self.command_targets[env_ids] = targets

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        if not bool(self.cfg.init_state.randomize_initial_velocity):
            self.root_states[env_ids, 7:13] = 0.0
            self.gym.set_actor_root_state_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
            )

    def step(self, actions):
        # Rate limits are defined per policy step, not per PhysX substep.
        self.last_output_actions.copy_(self.output_actions)
        cfg = self.cfg.control
        if bool(getattr(cfg, "residual_persistent_yaw_error_gate", False)):
            yaw_error = self.commands[:, 1] - self.tracking_ang_vel[:, 2]
            (
                next_sign,
                next_persistent_time,
                next_cooldown,
                next_active,
                next_gate,
            ) = update_persistent_yaw_error_gate(
                yaw_error,
                self.residual_yaw_error_sign,
                self.residual_yaw_error_persistent_time,
                self.residual_yaw_error_cooldown_time,
                self.residual_yaw_error_gate_active,
                self.dt,
                getattr(cfg, "residual_yaw_gate_activation_error", 0.010),
                getattr(cfg, "residual_yaw_gate_release_error", 0.004),
                getattr(cfg, "residual_yaw_gate_full_scale_error", 0.025),
                getattr(cfg, "residual_yaw_gate_activation_time", 0.20),
                getattr(cfg, "residual_yaw_gate_sign_flip_cooldown", 0.40),
                getattr(cfg, "residual_yaw_gate_sign_epsilon", 1.0e-4),
            )
            self.residual_yaw_error_sign.copy_(next_sign)
            self.residual_yaw_error_persistent_time.copy_(next_persistent_time)
            self.residual_yaw_error_cooldown_time.copy_(next_cooldown)
            self.residual_yaw_error_gate_active.copy_(next_active)
            next_gate = blend_yaw_residual_gate_with_command_rate(
                next_gate,
                self.command_rates[:, 1],
                getattr(cfg, "residual_yaw_gate_rate_bypass_start", float("inf")),
                getattr(cfg, "residual_yaw_gate_rate_bypass_full", float("inf")),
            )
            self.residual_yaw_error_gate.copy_(next_gate)
        else:
            self.residual_yaw_error_gate.fill_(1.0)
        return super().step(actions)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.requested_output_actions[env_ids] = 0.0
        self.output_actions[env_ids] = 0.0
        self.last_output_actions[env_ids] = 0.0
        self.nominal_policy_actions[env_ids] = 0.0
        self.feedback_policy_actions[env_ids] = 0.0
        self.derivative_feedback_policy_actions[env_ids] = 0.0
        self.rate_feedforward_policy_actions[env_ids] = 0.0
        self.combined_policy_actions[env_ids] = 0.0
        self.applied_residual_actions[env_ids] = 0.0
        self.tracking_heading[env_ids] = yaw_from_quaternion(
            self.root_states[env_ids, 3:7]
        )
        self.tracking_lin_vel[env_ids] = 0.0
        self.tracking_ang_vel[env_ids] = 0.0
        self.commands[env_ids, :2] = 0.0
        self.applied_feasible_command[env_ids] = 0.0
        self.command_rates[env_ids] = 0.0
        self.held_upper_command_rates[env_ids] = 0.0
        self.command_rate_hold_steps_remaining[env_ids] = 0
        self.tracking_error_integral[env_ids] = 0.0
        self.last_tracking_error[env_ids] = 0.0
        self.tracking_error_derivative[env_ids] = 0.0
        self.residual_yaw_error_sign[env_ids] = 0.0
        self.residual_yaw_error_persistent_time[env_ids] = 0.0
        self.residual_yaw_error_cooldown_time[env_ids] = 0.0
        self.residual_yaw_error_gate_active[env_ids] = False
        self.residual_yaw_error_gate[env_ids] = 0.0
        self.command_brake_pending[env_ids] = False
        self.command_yaw_brake_pending[env_ids] = False
        self.command_profile_is_random_walk[env_ids] = False
        if self.feasible_transition_manager is not None:
            self.feasible_transition_manager.reset(env_ids)

    def _resample_commands(self, env_ids):
        """Sample stop, straight, and feasible curved-motion commands."""
        count = int(len(env_ids))
        if count == 0:
            return

        cfg = self.cfg.commands
        previous_commands = self.command_targets[env_ids].clone()
        if self.common_step_counter < int(cfg.straight_only_policy_steps):
            turn_fraction = 0.0
            envelope_fraction = 0.0
        elif self.common_step_counter < int(cfg.mixed_policy_steps):
            turn_fraction = float(cfg.mixed_turn_fraction)
            envelope_fraction = float(cfg.mixed_envelope_fraction)
        else:
            turn_fraction = float(cfg.turn_fraction)
            envelope_fraction = float(cfg.feasible_envelope_fraction)
        stop_fraction = float(cfg.stop_fraction)
        straight_fraction = 1.0 - stop_fraction - turn_fraction

        random_kind = torch.rand(count, device=self.device)
        stop_end = stop_fraction
        straight_end = stop_end + straight_fraction
        stop_mask = random_kind < stop_end
        straight_mask = (random_kind >= stop_end) & (random_kind < straight_end)
        turn_mask = random_kind >= straight_end

        maximum_speed = float(cfg.max_forward_speed)
        velocity = (
            2.0 * torch.rand(count, device=self.device) - 1.0
        ) * maximum_speed

        # A low-speed straight command is physically valid and must remain in
        # the training distribution.  Only curved-motion samples need the
        # configured minimum rolling speed; the dynamic yaw-authority envelope
        # below then decides how much steering is actually attainable.
        small_turn = turn_mask & (
            torch.abs(velocity) < float(cfg.minimum_turn_speed)
        )
        moving_sign = torch.where(
            velocity >= 0.0,
            torch.ones_like(velocity),
            -torch.ones_like(velocity),
        )
        velocity[small_turn] = (
            moving_sign[small_turn] * float(cfg.minimum_turn_speed)
        )

        # Regularly expose the policy to the boundary of the feasible set.
        # Uniform sampling alone produces too few maximum-speed turns.
        extreme_mask = turn_mask & (
            torch.rand(count, device=self.device) < float(cfg.extreme_turn_fraction)
        )
        velocity[extreme_mask] = moving_sign[extreme_mask] * maximum_speed

        yaw_limit = feasible_yaw_rate_limit(
            velocity,
            cfg.max_yaw_rate,
            cfg.minimum_turn_radius,
            envelope_fraction,
            getattr(cfg, "turn_authority_start_speed", 0.0),
            getattr(cfg, "turn_authority_full_speed", 0.0),
        )
        steering_magnitude = float(cfg.minimum_turn_command_fraction) + (
            1.0 - float(cfg.minimum_turn_command_fraction)
        ) * torch.rand_like(yaw_limit)
        steering_magnitude[extreme_mask] = 1.0
        steering_sign = torch.where(
            torch.rand_like(yaw_limit) < 0.5,
            -torch.ones_like(yaw_limit),
            torch.ones_like(yaw_limit),
        )
        steering = steering_sign * steering_magnitude
        yaw_rate = torch.where(
            turn_mask,
            steering * yaw_limit,
            torch.zeros_like(velocity),
        )

        velocity[stop_mask] = 0.0
        yaw_rate[stop_mask | straight_mask] = 0.0
        sampled = torch.stack((velocity, yaw_rate), dim=-1)
        projected = project_velocity_commands(
            sampled,
            cfg.max_forward_speed,
            cfg.max_yaw_rate,
            cfg.minimum_turn_radius,
            envelope_fraction,
            stationary_threshold=self.cfg.rewards.stationary_command_threshold,
            turn_authority_start_speed=getattr(
                cfg, "turn_authority_start_speed", 0.0
            ),
            turn_authority_full_speed=getattr(
                cfg, "turn_authority_full_speed", 0.0
            ),
        )
        # Dynamic tracking requires deliberate sign reversals.  Yaw-only
        # transitions are sampled separately because keeping v while flipping
        # w is the dominant navigation maneuver and has different spherical-
        # robot dynamics from reversing both channels together.
        previous_moving = torch.abs(previous_commands[:, 0]) >= float(
            cfg.minimum_turn_speed
        )
        previous_turning = torch.abs(previous_commands[:, 1]) >= float(
            cfg.reversal_detection_w
        )
        yaw_only_mask = previous_moving & previous_turning & (
            torch.rand(count, device=self.device)
            < float(getattr(cfg, "yaw_only_transition_fraction", 0.0))
        )
        if torch.any(yaw_only_mask):
            yaw_only_targets = previous_commands[yaw_only_mask].clone()
            yaw_only_targets[:, 1] *= -1.0
            projected[yaw_only_mask] = project_velocity_commands(
                yaw_only_targets,
                cfg.max_forward_speed,
                cfg.max_yaw_rate,
                cfg.minimum_turn_radius,
                envelope_fraction,
                stationary_threshold=self.cfg.rewards.stationary_command_threshold,
                turn_authority_start_speed=getattr(
                    cfg, "turn_authority_start_speed", 0.0
                ),
                turn_authority_full_speed=getattr(
                    cfg, "turn_authority_full_speed", 0.0
                ),
            )

        opposite_mask = previous_moving & ~yaw_only_mask & (
            torch.rand(count, device=self.device)
            < float(cfg.opposite_transition_fraction)
        )
        projected[opposite_mask] = project_velocity_commands(
            -previous_commands[opposite_mask],
            cfg.max_forward_speed,
            cfg.max_yaw_rate,
            cfg.minimum_turn_radius,
            envelope_fraction,
            stationary_threshold=self.cfg.rewards.stationary_command_threshold,
            turn_authority_start_speed=getattr(
                cfg, "turn_authority_start_speed", 0.0
            ),
            turn_authority_full_speed=getattr(
                cfg, "turn_authority_full_speed", 0.0
            ),
        )
        profile_mask = torch.rand(count, device=self.device) < float(
            cfg.smooth_profile_fraction
        )
        random_walk_mask = profile_mask & (
            torch.rand(count, device=self.device)
            < float(getattr(cfg, "random_walk_profile_fraction", 0.0))
        )
        smooth_mask = profile_mask & ~random_walk_mask
        self.command_profile_is_independent[env_ids] = False
        self.command_profile_is_smooth[env_ids] = profile_mask
        self.command_profile_is_random_walk[env_ids] = random_walk_mask
        self.command_reference_is_smooth[env_ids] = profile_mask
        if torch.any(random_walk_mask):
            random_walk_initial = projected[random_walk_mask].clone()
            zero_or_slow = torch.abs(random_walk_initial[:, 0]) < float(
                cfg.random_walk_minimum_speed
            )
            if torch.any(zero_or_slow):
                random_sign = torch.where(
                    torch.rand(
                        int(zero_or_slow.sum().item()), device=self.device
                    ) < 0.5,
                    -torch.ones(
                        int(zero_or_slow.sum().item()), device=self.device
                    ),
                    torch.ones(
                        int(zero_or_slow.sum().item()), device=self.device
                    ),
                )
                random_walk_initial[zero_or_slow, 0] = (
                    random_sign * float(cfg.random_walk_minimum_speed)
                )
            random_walk_initial = project_velocity_commands(
                random_walk_initial,
                cfg.max_forward_speed,
                cfg.max_yaw_rate,
                cfg.minimum_turn_radius,
                cfg.feasible_envelope_fraction,
                stationary_threshold=self.cfg.rewards.stationary_command_threshold,
                turn_authority_start_speed=getattr(
                    cfg, "turn_authority_start_speed", 0.0
                ),
                turn_authority_full_speed=getattr(
                    cfg, "turn_authority_full_speed", 0.0
                ),
            )
            projected[random_walk_mask] = random_walk_initial
        if torch.any(smooth_mask):
            smooth_ids = env_ids[smooth_mask]
            smooth_count = int(smooth_ids.numel())
            phase = 2.0 * torch.pi * torch.rand(smooth_count, device=self.device)
            period = float(cfg.smooth_profile_period_min_s) + (
                float(cfg.smooth_profile_period_max_s)
                - float(cfg.smooth_profile_period_min_s)
            ) * torch.rand(smooth_count, device=self.device)
            speed_amplitude = float(cfg.smooth_profile_speed_amplitude_min) + (
                float(cfg.smooth_profile_speed_amplitude_max)
                - float(cfg.smooth_profile_speed_amplitude_min)
            ) * torch.rand(smooth_count, device=self.device)
            maximum_profile_yaw = torch.minimum(
                torch.full_like(speed_amplitude, float(cfg.max_yaw_rate)),
                speed_amplitude / float(cfg.minimum_turn_radius),
            ) * float(cfg.feasible_envelope_fraction)
            yaw_fraction = float(cfg.smooth_profile_yaw_fraction_min) + (
                float(cfg.smooth_profile_yaw_fraction_max)
                - float(cfg.smooth_profile_yaw_fraction_min)
            ) * torch.rand(smooth_count, device=self.device)
            curvature_sign = torch.where(
                torch.rand(smooth_count, device=self.device) < 0.5,
                -torch.ones(smooth_count, device=self.device),
                torch.ones(smooth_count, device=self.device),
            )
            signed_curvature = (
                curvature_sign
                * yaw_fraction
                * maximum_profile_yaw
                / torch.clamp(speed_amplitude, min=1.0e-6)
            )
            self.command_profile_phase[smooth_ids] = phase
            self.command_profile_period[smooth_ids] = period
            self.command_profile_speed_amplitude[smooth_ids] = speed_amplitude
            self.command_profile_signed_curvature[smooth_ids] = signed_curvature
            smooth_targets = smooth_feasible_velocity_profile(
                phase,
                speed_amplitude,
                signed_curvature,
                cfg.max_forward_speed,
                cfg.max_yaw_rate,
                cfg.minimum_turn_radius,
                cfg.feasible_envelope_fraction,
                stationary_threshold=self.cfg.rewards.stationary_command_threshold,
                turn_authority_start_speed=getattr(
                    cfg, "turn_authority_start_speed", 0.0
                ),
                turn_authority_full_speed=getattr(
                    cfg, "turn_authority_full_speed", 0.0
                ),
            )
            independent_fraction = float(
                getattr(cfg, "independent_smooth_profile_fraction", 0.0)
            )
            independent_local = (
                torch.rand(smooth_count, device=self.device) < independent_fraction
            )
            self.command_profile_is_independent[smooth_ids] = independent_local
            if torch.any(independent_local):
                independent_ids = smooth_ids[independent_local]
                independent_count = int(independent_ids.numel())
                fixed_velocity = (
                    torch.rand(independent_count, device=self.device)
                    < float(cfg.independent_fixed_velocity_fraction)
                )
                speed_magnitude = float(cfg.independent_profile_minimum_speed) + (
                    float(cfg.max_forward_speed)
                    - float(cfg.independent_profile_minimum_speed)
                ) * torch.rand(independent_count, device=self.device)
                speed_sign = torch.where(
                    torch.rand(independent_count, device=self.device) < 0.5,
                    -torch.ones(independent_count, device=self.device),
                    torch.ones(independent_count, device=self.device),
                )
                velocity_offset = torch.where(
                    fixed_velocity,
                    speed_sign * speed_magnitude,
                    torch.zeros_like(speed_magnitude),
                )
                velocity_amplitude = torch.where(
                    fixed_velocity,
                    torch.zeros_like(speed_magnitude),
                    speed_magnitude,
                )
                maximum_profile_yaw = feasible_yaw_rate_limit(
                    speed_magnitude,
                    cfg.max_yaw_rate,
                    cfg.minimum_turn_radius,
                    cfg.feasible_envelope_fraction,
                    getattr(cfg, "turn_authority_start_speed", 0.0),
                    getattr(cfg, "turn_authority_full_speed", 0.0),
                )
                yaw_fraction = float(cfg.independent_profile_yaw_fraction_min) + (
                    float(cfg.independent_profile_yaw_fraction_max)
                    - float(cfg.independent_profile_yaw_fraction_min)
                ) * torch.rand(independent_count, device=self.device)
                yaw_amplitude = yaw_fraction * maximum_profile_yaw
                yaw_phase_offset = (
                    2.0 * torch.pi * torch.rand(independent_count, device=self.device)
                )
                frequency_choices = torch.as_tensor(
                    cfg.independent_profile_yaw_frequency_ratios,
                    dtype=phase.dtype,
                    device=self.device,
                )
                frequency_index = torch.randint(
                    frequency_choices.numel(),
                    (independent_count,),
                    device=self.device,
                )
                yaw_frequency_ratio = frequency_choices[frequency_index]
                self.command_profile_velocity_offset[independent_ids] = velocity_offset
                self.command_profile_velocity_amplitude[
                    independent_ids
                ] = velocity_amplitude
                self.command_profile_yaw_amplitude[independent_ids] = yaw_amplitude
                self.command_profile_yaw_phase_offset[
                    independent_ids
                ] = yaw_phase_offset
                self.command_profile_yaw_frequency_ratio[
                    independent_ids
                ] = yaw_frequency_ratio
                smooth_targets[independent_local] = independent_feasible_velocity_profile(
                    phase[independent_local],
                    velocity_offset,
                    velocity_amplitude,
                    yaw_amplitude,
                    yaw_phase_offset,
                    yaw_frequency_ratio,
                    cfg.max_forward_speed,
                    cfg.max_yaw_rate,
                    cfg.minimum_turn_radius,
                    cfg.feasible_envelope_fraction,
                    stationary_threshold=self.cfg.rewards.stationary_command_threshold,
                    turn_authority_start_speed=getattr(
                        cfg, "turn_authority_start_speed", 0.0
                    ),
                    turn_authority_full_speed=getattr(
                        cfg, "turn_authority_full_speed", 0.0
                    ),
                )
            projected[smooth_mask] = smooth_targets
        self.set_command_targets(projected, env_ids)

    def compute_observations(self):
        cfg = self.cfg
        command_scale = torch.as_tensor(
            [
                1.0 / float(cfg.commands.max_forward_speed),
                1.0 / float(cfg.commands.max_yaw_rate),
            ],
            device=self.device,
        )
        components = [self.commands[:, :2] * command_scale]
        if bool(getattr(cfg.commands, "observe_command_rates", False)):
            rate_scale = torch.as_tensor(
                [
                    1.0 / float(cfg.commands.maximum_linear_acceleration),
                    1.0 / float(cfg.commands.maximum_yaw_acceleration),
                ],
                device=self.device,
            )
            components.append(self.command_rates * rate_scale)
        if bool(
            getattr(cfg.commands, "observe_preview_tracking_errors", False)
        ):
            preview_commands = lead_compensated_velocity_commands(
                self.commands,
                self.command_rates,
                cfg.control.residual_alignment_linear_preview_time,
                cfg.control.residual_alignment_angular_preview_time,
                cfg.commands.max_forward_speed,
                cfg.commands.max_yaw_rate,
                cfg.commands.minimum_turn_radius,
                cfg.commands.feasible_envelope_fraction,
                stationary_threshold=cfg.rewards.stationary_command_threshold,
                turn_authority_start_speed=getattr(
                    cfg.commands, "turn_authority_start_speed", 0.0
                ),
                turn_authority_full_speed=getattr(
                    cfg.commands, "turn_authority_full_speed", 0.0
                ),
            )
            measured_commands = torch.stack(
                (self.tracking_lin_vel[:, 0], self.tracking_ang_vel[:, 2]),
                dim=1,
            )
            components.append((preview_commands - measured_commands) * command_scale)
        if bool(
            getattr(cfg.commands, "observe_tracking_error_integrals", False)
        ):
            integral_scale = torch.as_tensor(
                [
                    1.0 / float(cfg.control.linear_error_integral_limit),
                    1.0 / float(cfg.control.angular_error_integral_limit),
                ],
                device=self.device,
            )
            components.append(self.tracking_error_integral * integral_scale)
        if bool(
            getattr(cfg.commands, "observe_tracking_error_derivatives", False)
        ):
            derivative_scale = torch.as_tensor(
                [
                    1.0 / float(cfg.commands.maximum_linear_acceleration),
                    1.0 / float(cfg.commands.maximum_yaw_acceleration),
                ],
                device=self.device,
            )
            components.append(self.tracking_error_derivative * derivative_scale)
        components.extend(
            (
                self.tracking_lin_vel * self.obs_scales.lin_vel,
                self.tracking_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.dof_pos[:, 1:2] * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            )
        )
        self.obs_buf = torch.cat(components, dim=-1)
        if bool(
            getattr(
                cfg.control,
                "canonicalize_policy_for_drive_reversal",
                False,
            )
        ):
            drive_direction = canonical_drive_direction(
                self.commands[:, 0],
                self.tracking_lin_vel[:, 0],
                cfg.rewards.stationary_command_threshold,
            )
            self.obs_buf = canonicalize_velocity_policy_observations(
                self.obs_buf,
                drive_direction,
                observe_command_rates=bool(
                    getattr(cfg.commands, "observe_command_rates", False)
                ),
                observe_preview_tracking_errors=bool(
                    getattr(cfg.commands, "observe_preview_tracking_errors", False)
                ),
                observe_tracking_error_integrals=bool(
                    getattr(cfg.commands, "observe_tracking_error_integrals", False)
                ),
                observe_tracking_error_derivatives=bool(
                    getattr(cfg.commands, "observe_tracking_error_derivatives", False)
                ),
            )
        if self.add_noise:
            self.obs_buf += (
                2.0 * torch.rand_like(self.obs_buf) - 1.0
            ) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        noise = torch.zeros_like(self.obs_buf[0])
        self.add_noise = bool(cfg.noise.add_noise)
        if not self.add_noise:
            return noise
        level = float(cfg.noise.noise_level)
        scales = cfg.noise.noise_scales
        state_offset = 2
        if bool(getattr(cfg.commands, "observe_command_rates", False)):
            state_offset += 2
        if bool(
            getattr(cfg.commands, "observe_preview_tracking_errors", False)
        ):
            state_offset += 2
        if bool(
            getattr(cfg.commands, "observe_tracking_error_integrals", False)
        ):
            state_offset += 2
        if bool(
            getattr(cfg.commands, "observe_tracking_error_derivatives", False)
        ):
            state_offset += 2
        noise[state_offset : state_offset + 3] = (
            scales.lin_vel * level * self.obs_scales.lin_vel
        )
        noise[state_offset + 3 : state_offset + 6] = (
            scales.ang_vel * level * self.obs_scales.ang_vel
        )
        noise[state_offset + 6 : state_offset + 9] = scales.gravity * level
        noise[state_offset + 9] = (
            scales.dof_pos * level * self.obs_scales.dof_pos
        )
        noise[state_offset + 10 : state_offset + 12] = (
            scales.dof_vel * level * self.obs_scales.dof_vel
        )
        return noise

    def _compute_torques(self, actions):
        cfg = self.cfg.control
        lead_projection_max_forward_speed = getattr(
            cfg,
            "lead_projection_max_forward_speed",
            None,
        )
        if lead_projection_max_forward_speed is None:
            lead_projection_max_forward_speed = self.cfg.commands.max_forward_speed
        lead_projection_max_forward_speed = float(
            lead_projection_max_forward_speed
        )
        gap_thresholds = getattr(
            cfg,
            "rate_feedforward_target_gap_threshold",
            [float("inf"), float("inf")],
        )
        smooth_tracking_enabled = command_target_gap_mask(
            self.command_targets,
            self.commands,
            gap_thresholds[0],
            gap_thresholds[1],
        )
        if bool(
            getattr(
                cfg,
                "require_explicit_smooth_profile_for_phase_lead",
                False,
            )
        ):
            smooth_tracking_enabled &= self.command_reference_is_smooth
        smooth_feedback_enabled = smooth_tracking_enabled & (
            torch.abs(self.command_rates[:, 1])
            > float(cfg.smooth_angular_feedback_minimum_command_rate)
        )
        nominal_commands = lead_compensated_velocity_commands(
            self.commands,
            self.command_rates,
            cfg.linear_command_lead_time,
            cfg.angular_command_lead_time,
            lead_projection_max_forward_speed,
            self.cfg.commands.max_yaw_rate,
            self.cfg.commands.minimum_turn_radius,
            self.cfg.commands.feasible_envelope_fraction,
            stationary_threshold=self.cfg.rewards.stationary_command_threshold,
            turn_authority_start_speed=getattr(
                self.cfg.commands, "turn_authority_start_speed", 0.0
            ),
            turn_authority_full_speed=getattr(
                self.cfg.commands, "turn_authority_full_speed", 0.0
            ),
        )
        smooth_linear_lead_time = float(
            getattr(cfg, "smooth_linear_command_lead_time", 0.0)
        )
        smooth_angular_lead_time = float(
            getattr(cfg, "smooth_angular_command_lead_time", 0.0)
        )
        if smooth_linear_lead_time > 0.0 or smooth_angular_lead_time > 0.0:
            smooth_nominal_commands = lead_compensated_velocity_commands(
                self.commands,
                self.command_rates,
                (
                    smooth_linear_lead_time
                    if smooth_linear_lead_time > 0.0
                    else cfg.linear_command_lead_time
                ),
                (
                    smooth_angular_lead_time
                    if smooth_angular_lead_time > 0.0
                    else cfg.angular_command_lead_time
                ),
                lead_projection_max_forward_speed,
                self.cfg.commands.max_yaw_rate,
                self.cfg.commands.minimum_turn_radius,
                self.cfg.commands.feasible_envelope_fraction,
                stationary_threshold=self.cfg.rewards.stationary_command_threshold,
                turn_authority_start_speed=getattr(
                    self.cfg.commands, "turn_authority_start_speed", 0.0
                ),
                turn_authority_full_speed=getattr(
                    self.cfg.commands, "turn_authority_full_speed", 0.0
                ),
            )
            smooth_angular_command_gain = float(
                getattr(cfg, "smooth_angular_command_gain", 1.0)
            )
            smooth_nominal_commands[:, 1].mul_(smooth_angular_command_gain)
            # A bounded smooth profile can contain zero-rate plateaus after
            # feasible-set projection.  Selecting this branch from the
            # instantaneous rate switched the angular gain from 0.70 to 1.00
            # at every plateau edge and excited a large wrong-sign yaw
            # oscillation.  The target-gap/explicit-profile mask already
            # separates smooth references from steps, so retain the smooth
            # controller continuously, including extrema and flat tops.
            smooth_lead_enabled = smooth_tracking_enabled
            nominal_commands = torch.where(
                smooth_lead_enabled.unsqueeze(1),
                smooth_nominal_commands,
                nominal_commands,
            )
        nominal = nominal_actuator_actions(
            nominal_commands,
            forward_speed_per_action=cfg.nominal_forward_speed_per_action,
            yaw_gain_intercept=cfg.nominal_yaw_gain_intercept,
            yaw_gain_speed_slope=cfg.nominal_yaw_gain_speed_slope,
            drive_speed_breakpoints=getattr(
                cfg, "nominal_drive_speed_breakpoints", None
            ),
            drive_action_values=getattr(
                cfg, "nominal_drive_action_values", None
            ),
            steering_speed_breakpoints=getattr(
                cfg, "nominal_steering_speed_breakpoints", None
            ),
            steering_half_fraction_scales=getattr(
                cfg, "nominal_steering_half_fraction_scales", None
            ),
            steering_full_fraction_scales=getattr(
                cfg, "nominal_steering_full_fraction_scales", None
            ),
            maximum_yaw_rate=self.cfg.commands.max_yaw_rate,
            minimum_turn_radius=self.cfg.commands.minimum_turn_radius,
        )
        linear_feedback_gain = cfg.linear_feedback_gain
        low_speed_linear_feedback_gain = getattr(
            cfg, "low_speed_linear_feedback_gain", None
        )
        if low_speed_linear_feedback_gain is not None:
            linear_feedback_gain = speed_scheduled_value(
                self.commands[:, 0],
                low_speed_linear_feedback_gain,
                cfg.linear_feedback_gain,
                getattr(cfg, "linear_feedback_transition_start_speed", 0.04),
                getattr(cfg, "linear_feedback_transition_full_speed", 0.08),
            )
        feedback = velocity_error_feedback_actions(
            self.commands,
            self.tracking_lin_vel[:, 0],
            self.tracking_ang_vel[:, 2],
            forward_speed_per_action=cfg.nominal_forward_speed_per_action,
            yaw_gain_intercept=cfg.nominal_yaw_gain_intercept,
            yaw_gain_speed_slope=cfg.nominal_yaw_gain_speed_slope,
            linear_feedback_gain=linear_feedback_gain,
            angular_feedback_gain=cfg.angular_feedback_gain,
            linear_action_limit=cfg.linear_feedback_action_limit,
            angular_action_limit=cfg.angular_feedback_action_limit,
            stationary_threshold=self.cfg.rewards.stationary_command_threshold,
            wrong_direction_angular_feedback_gain=getattr(
                cfg, "wrong_direction_angular_feedback_gain", None
            ),
            wrong_direction_command_threshold=getattr(
                cfg, "wrong_direction_command_threshold", 0.01
            ),
        )
        smooth_feedback = velocity_error_feedback_actions(
            self.commands,
            self.tracking_lin_vel[:, 0],
            self.tracking_ang_vel[:, 2],
            forward_speed_per_action=cfg.nominal_forward_speed_per_action,
            yaw_gain_intercept=cfg.nominal_yaw_gain_intercept,
            yaw_gain_speed_slope=cfg.nominal_yaw_gain_speed_slope,
            linear_feedback_gain=0.0,
            angular_feedback_gain=cfg.smooth_angular_feedback_gain,
            linear_action_limit=0.0,
            angular_action_limit=cfg.smooth_angular_feedback_action_limit,
            stationary_threshold=self.cfg.rewards.stationary_command_threshold,
        )
        feedback += smooth_feedback * smooth_feedback_enabled.unsqueeze(1)
        derivative_feedback = velocity_error_derivative_actions(
            self.commands,
            self.tracking_lin_vel[:, 0],
            self.tracking_error_derivative,
            forward_speed_per_action=cfg.nominal_forward_speed_per_action,
            yaw_gain_intercept=cfg.nominal_yaw_gain_intercept,
            yaw_gain_speed_slope=cfg.nominal_yaw_gain_speed_slope,
            linear_derivative_gain=cfg.linear_derivative_gain,
            angular_derivative_gain=cfg.angular_derivative_gain,
            linear_action_limit=cfg.linear_derivative_action_limit,
            angular_action_limit=cfg.angular_derivative_action_limit,
            stationary_threshold=self.cfg.rewards.stationary_command_threshold,
        )
        rate_feedforward = velocity_rate_feedforward_actions(
            self.commands,
            self.command_rates,
            self.tracking_lin_vel[:, 0],
            forward_speed_per_action=cfg.nominal_forward_speed_per_action,
            yaw_gain_intercept=cfg.nominal_yaw_gain_intercept,
            yaw_gain_speed_slope=cfg.nominal_yaw_gain_speed_slope,
            linear_preview_time=cfg.linear_rate_feedforward_time,
            angular_preview_time=cfg.angular_rate_feedforward_time,
            linear_action_limit=cfg.linear_rate_feedforward_action_limit,
            angular_action_limit=cfg.angular_rate_feedforward_action_limit,
            stationary_threshold=self.cfg.rewards.stationary_command_threshold,
        )
        rate_feedforward *= smooth_tracking_enabled.unsqueeze(1)
        integral_feedback = velocity_error_integral_actions(
            self.commands,
            self.tracking_lin_vel[:, 0],
            self.tracking_error_integral,
            forward_speed_per_action=cfg.nominal_forward_speed_per_action,
            yaw_gain_intercept=cfg.nominal_yaw_gain_intercept,
            yaw_gain_speed_slope=cfg.nominal_yaw_gain_speed_slope,
            linear_integral_gain=cfg.linear_integral_gain,
            angular_integral_gain=cfg.angular_integral_gain,
            linear_action_limit=cfg.linear_integral_action_limit,
            angular_action_limit=cfg.angular_integral_action_limit,
            stationary_threshold=self.cfg.rewards.stationary_command_threshold,
        )
        residual_actions = actions
        if bool(
            getattr(cfg, "canonicalize_policy_for_drive_reversal", False)
        ):
            residual_actions = map_canonical_residual_actions(
                residual_actions,
                canonical_drive_direction(
                    self.commands[:, 0],
                    self.tracking_lin_vel[:, 0],
                    self.cfg.rewards.stationary_command_threshold,
                ),
            )
        if bool(getattr(cfg, "residual_rate_alignment_filter", False)):
            residual_actions = rate_aligned_angular_residual_actions(
                residual_actions,
                self.commands,
                self.command_rates,
                self.tracking_lin_vel[:, 0],
                self.command_reference_is_smooth,
                stationary_threshold=self.cfg.rewards.stationary_command_threshold,
                minimum_angular_command_rate=float(
                    getattr(
                        cfg,
                        "residual_rate_alignment_minimum_command_rate",
                        1.0e-4,
                    )
                ),
                zero_angular_when_inactive=bool(
                    getattr(cfg, "residual_rate_alignment_zero_inactive", False)
                ),
                measured_yaw_rate=self.tracking_ang_vel[:, 2],
                error_align_when_inactive=bool(
                    getattr(
                        cfg,
                        "residual_rate_alignment_error_when_inactive",
                        False,
                    )
                ),
                inactive_error_full_scale=getattr(
                    cfg,
                    "residual_inactive_error_full_scale",
                    None,
                ),
            )
        if bool(getattr(cfg, "residual_error_alignment_filter", False)):
            alignment_commands = self.commands
            linear_preview = float(
                getattr(cfg, "residual_alignment_linear_preview_time", 0.0)
            )
            angular_preview = float(
                getattr(cfg, "residual_alignment_angular_preview_time", 0.0)
            )
            if linear_preview > 0.0 or angular_preview > 0.0:
                alignment_commands = lead_compensated_velocity_commands(
                    self.commands,
                    self.command_rates,
                    linear_preview,
                    angular_preview,
                    lead_projection_max_forward_speed,
                    self.cfg.commands.max_yaw_rate,
                    self.cfg.commands.minimum_turn_radius,
                    self.cfg.commands.feasible_envelope_fraction,
                    stationary_threshold=(
                        self.cfg.rewards.stationary_command_threshold
                    ),
                    turn_authority_start_speed=getattr(
                        self.cfg.commands, "turn_authority_start_speed", 0.0
                    ),
                    turn_authority_full_speed=getattr(
                        self.cfg.commands, "turn_authority_full_speed", 0.0
                    ),
                )
            residual_actions = error_aligned_residual_actions(
                residual_actions,
                alignment_commands,
                self.tracking_lin_vel[:, 0],
                self.tracking_ang_vel[:, 2],
                stationary_threshold=self.cfg.rewards.stationary_command_threshold,
            )
        if bool(getattr(cfg, "disable_residual_during_braking", False)):
            residual_actions = residual_actions.clone()
            any_braking = (
                self.command_brake_pending | self.command_yaw_brake_pending
            )
            residual_actions[any_braking] = 0.0
        if bool(getattr(cfg, "residual_persistent_yaw_error_gate", False)):
            residual_actions = persistent_error_gated_angular_residual_actions(
                residual_actions,
                self.commands[:, 1] - self.tracking_ang_vel[:, 2],
                canonical_drive_direction(
                    self.commands[:, 0],
                    self.tracking_lin_vel[:, 0],
                    self.cfg.rewards.stationary_command_threshold,
                ),
                self.residual_yaw_error_gate,
                force_error_alignment=bool(
                    getattr(cfg, "residual_yaw_gate_force_error_alignment", True)
                ),
            )
        residual_scale = torch.as_tensor(
            cfg.residual_action_scale,
            dtype=actions.dtype,
            device=actions.device,
        )
        combined_actions = torch.clamp(
            nominal
            + feedback
            + derivative_feedback
            + rate_feedforward
            + integral_feedback
            + residual_scale * residual_actions,
            -1.0,
            1.0,
        )
        self.nominal_policy_actions.copy_(nominal)
        self.feedback_policy_actions.copy_(feedback)
        self.derivative_feedback_policy_actions.copy_(derivative_feedback)
        self.rate_feedforward_policy_actions.copy_(rate_feedforward)
        self.applied_residual_actions.copy_(residual_actions)
        self.combined_policy_actions.copy_(combined_actions)

        requested = torch.empty_like(actions)
        requested[:, 0] = torch.clamp(
            combined_actions[:, 0] * float(cfg.joint1_velocity_scale),
            -float(cfg.joint1_velocity_limit),
            float(cfg.joint1_velocity_limit),
        )
        requested[:, 1] = torch.clamp(
            combined_actions[:, 1] * float(cfg.joint2_position_scale),
            -float(cfg.joint2_position_limit),
            float(cfg.joint2_position_limit),
        )
        self.requested_output_actions.copy_(requested)

        targets = requested
        if bool(cfg.set_target_rate_limit):
            target_delta = requested - self.last_output_actions
            limits = torch.as_tensor(
                [
                    float(cfg.joint1_target_rate_limit),
                    float(cfg.joint2_target_rate_limit),
                ],
                device=self.device,
            )
            targets = self.last_output_actions + torch.maximum(
                torch.minimum(target_delta, limits),
                -limits,
            )
        self.output_actions.copy_(targets)

        torques = torch.empty_like(actions)
        torques[:, 0] = float(cfg.joint1_velocity_kp) * (
            targets[:, 0] - self.dof_vel[:, 0]
        )
        torques[:, 1] = (
            float(cfg.joint2_position_kp)
            * (targets[:, 1] - self.dof_pos[:, 1])
            - float(cfg.joint2_velocity_kd) * self.dof_vel[:, 1]
        )
        torques[:, 0] = torch.clamp(
            torques[:, 0],
            -float(cfg.joint1_torque_limit),
            float(cfg.joint1_torque_limit),
        )
        torques[:, 1] = torch.clamp(
            torques[:, 1],
            -float(cfg.joint2_torque_limit),
            float(cfg.joint2_torque_limit),
        )
        return torques

    def _reward_tracking_lin_vel(self):
        error = self.commands[:, 0] - self.tracking_lin_vel[:, 0]
        sigma = float(self.cfg.rewards.linear_tracking_sigma)
        return torch.exp(-torch.abs(error) / sigma)

    def _angular_reward_target(self):
        """Return current yaw target, with causal preview for smooth profiles."""
        target = self.commands[:, 1]
        preview_time = float(
            getattr(self.cfg.rewards, "smooth_angular_reward_preview_time", 0.0)
        )
        if preview_time <= 0.0:
            return target
        smooth = self.command_profile_is_smooth
        if not torch.any(smooth):
            return target
        preview_commands = lead_compensated_velocity_commands(
            self.commands,
            self.command_rates,
            0.0,
            preview_time,
            self.cfg.commands.max_forward_speed,
            self.cfg.commands.max_yaw_rate,
            self.cfg.commands.minimum_turn_radius,
            self.cfg.commands.feasible_envelope_fraction,
            stationary_threshold=self.cfg.rewards.stationary_command_threshold,
            turn_authority_start_speed=getattr(
                self.cfg.commands, "turn_authority_start_speed", 0.0
            ),
            turn_authority_full_speed=getattr(
                self.cfg.commands, "turn_authority_full_speed", 0.0
            ),
        )
        return torch.where(smooth, preview_commands[:, 1], target)

    def _reward_tracking_ang_vel(self):
        error = self._angular_reward_target() - self.tracking_ang_vel[:, 2]
        sigma = float(self.cfg.rewards.angular_tracking_sigma)
        return torch.exp(-torch.abs(error) / sigma)

    def _reward_curvature_tracking(self):
        """Reward matching the requested v/w direction without division."""
        maximum_v = max(float(self.cfg.commands.max_forward_speed), 1.0e-6)
        maximum_w = max(float(self.cfg.commands.max_yaw_rate), 1.0e-6)
        requested_v = self.commands[:, 0] / maximum_v
        requested_w = self.commands[:, 1] / maximum_w
        measured_v = self.tracking_lin_vel[:, 0] / maximum_v
        measured_w = self.tracking_ang_vel[:, 2] / maximum_w
        requested_norm = torch.sqrt(requested_v.square() + requested_w.square())
        perpendicular_error = torch.abs(
            measured_v * requested_w - measured_w * requested_v
        ) / torch.clamp(requested_norm, min=1.0e-6)
        valid_turn = (
            torch.abs(self.commands[:, 0])
            >= float(self.cfg.commands.minimum_turn_speed)
        ) & (
            torch.abs(self.commands[:, 1])
            >= float(self.cfg.rewards.turning_command_threshold)
        )
        sigma = float(self.cfg.rewards.curvature_tracking_sigma)
        return valid_turn.float() * torch.exp(-perpendicular_error / sigma)

    def _reward_angular_tracking_error(self):
        """Non-saturating angular-error signal for large and small yaw errors."""
        error = self._angular_reward_target() - self.tracking_ang_vel[:, 2]
        sigma = float(self.cfg.rewards.angular_tracking_sigma)
        return torch.abs(error) / sigma

    def _reward_angular_acceleration_error(self):
        """Match yaw acceleration on smooth references to reduce phase lag."""
        smooth = self.command_profile_is_smooth.float()
        sigma = float(self.cfg.rewards.angular_acceleration_error_sigma)
        normalized_error = torch.abs(self.tracking_error_derivative[:, 1]) / sigma
        return smooth * torch.clamp(normalized_error, max=5.0)

    def _reward_straight_yaw(self):
        """Suppress learned yaw bias whenever the requested path is straight."""
        straight = torch.abs(self.commands[:, 1]) < float(
            self.cfg.rewards.turning_command_threshold
        )
        sigma = float(self.cfg.rewards.angular_tracking_sigma)
        return straight.float() * torch.square(self.tracking_ang_vel[:, 2] / sigma)

    def _reward_lateral_velocity(self):
        return torch.square(self.tracking_lin_vel[:, 1])

    def _reward_stationary_yaw(self):
        stationary = (
            torch.abs(self.commands[:, 0])
            < float(self.cfg.rewards.stationary_command_threshold)
        )
        return stationary.float() * torch.square(self.tracking_ang_vel[:, 2])

    def _reward_yaw_direction(self):
        command = self.commands[:, 1]
        turning = torch.abs(command) > float(
            self.cfg.rewards.turning_command_threshold
        )
        signed_rate = torch.sign(command) * self.tracking_ang_vel[:, 2]
        return turning.float() * torch.tanh(signed_rate / 0.02)

    def _reward_yaw_wrong_direction(self):
        """Penalize opposite yaw without rewarding same-direction overshoot."""
        command = self.commands[:, 1]
        turning = torch.abs(command) > float(
            self.cfg.rewards.turning_command_threshold
        )
        signed_rate = torch.sign(command) * self.tracking_ang_vel[:, 2]
        opposite = torch.clamp(-signed_rate / 0.02, min=0.0, max=5.0)
        return turning.float() * opposite

    def _reward_linear_wrong_direction(self):
        """Penalize governed forward-speed sign errors during reversals."""
        command = self.commands[:, 0]
        moving = torch.abs(command) > float(
            self.cfg.rewards.stationary_command_threshold
        )
        signed_speed = torch.sign(command) * self.tracking_lin_vel[:, 0]
        opposite = torch.clamp(-signed_speed / 0.04, min=0.0, max=5.0)
        return moving.float() * opposite

    def _reward_action_saturation(self):
        excess = torch.clamp(
            torch.abs(self.combined_policy_actions) - 0.90,
            min=0.0,
        )
        return torch.sum(torch.square(excess / 0.10), dim=1)

    def _reward_residual_action(self):
        return torch.sum(torch.square(self.actions), dim=1)
