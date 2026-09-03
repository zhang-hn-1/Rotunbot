"""Explainable V1 velocity teacher in the measured V62 command domain."""

from dataclasses import dataclass
import math

import torch

from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import (
    project_velocity_commands,
)


@dataclass(frozen=True)
class V1VelocityTeacherConfig:
    """Conservative teacher parameters, expressed in physical units."""

    max_forward_speed: float = 0.25
    max_yaw_rate: float = 0.10
    minimum_turn_radius: float = 2.0
    feasible_envelope_fraction: float = 1.0
    goal_radius: float = 0.35
    goal_stop_margin: float = 0.15
    stopping_distance: float = 0.80
    heading_gain: float = 0.35
    yaw_rate_damping: float = 0.35
    velocity_feedback_gain: float = 0.20
    obstacle_stop_distance: float = 0.45
    # V1's centered side-wall clearance is about 0.60--0.70 m.  Treat that
    # as open corridor; only clearance approaching the robot radius brakes.
    obstacle_slow_distance: float = 0.70
    # When the goal bearing is large, cosine speed reduction combined with the
    # measured |w| <= v/R envelope can create a low-speed turn deadlock.  In
    # open space, retain enough speed for the feasible yaw command to recover.
    turn_recovery_bearing: float = 0.65
    turn_recovery_speed_fraction: float = 0.80


def _aggregate_scalar(records, sum_key, count_key):
    total = sum(float(record.get(sum_key, 0.0)) for record in records)
    count = sum(float(record.get(count_key, 0.0)) for record in records)
    return total / count if count > 0.0 else 0.0


def _aggregate_distribution(records, prefix, count_key):
    count = sum(float(record.get(count_key, 0.0)) for record in records)
    total = sum(float(record.get(prefix + "_sum", 0.0)) for record in records)
    square_total = sum(
        float(record.get(prefix + "_sq_sum", 0.0)) for record in records
    )
    mean = total / count if count > 0.0 else 0.0
    variance = max(square_total / count - mean * mean, 0.0) if count > 0.0 else 0.0
    minimums = [record[prefix + "_min"] for record in records if prefix + "_min" in record]
    maximums = [record[prefix + "_max"] for record in records if prefix + "_max" in record]
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(minimums) if minimums else 0.0,
        "max": max(maximums) if maximums else 0.0,
    }


def summarize_teacher_episodes(records):
    """Aggregate episode and command-path evidence for the formal teacher gate."""
    records = list(records)
    if not records:
        raise ValueError("at least one teacher episode record is required")
    count = float(len(records))
    successes = sum(bool(record.get("success", False)) for record in records)
    collisions = sum(bool(record.get("collision", False)) for record in records)
    timeouts = sum(bool(record.get("timeout", False)) for record in records)
    initial = [
        float(record["initial_goal_distance_m"] if "initial_goal_distance_m" in record else record["distance_m"])
        for record in records
    ]
    final = [float(record["terminal_goal_distance_m"]) for record in records]
    paths = [float(record.get("path_length_m", 0.0)) for record in records]
    spl_values = [
        (goal / max(path, goal)) if bool(record.get("success", False)) else 0.0
        for record, goal, path in zip(records, initial, paths)
    ]
    teacher_commands = sum(float(record.get("teacher_command_count", 0.0)) for record in records)
    projection_count = sum(float(record.get("projection_activation_count", 0.0)) for record in records)
    governor_count = sum(float(record.get("governor_modification_count", 0.0)) for record in records)
    projection_sum = sum(float(record.get("projection_correction_sum", 0.0)) for record in records)
    projection_max = max((float(record.get("projection_correction_max", 0.0)) for record in records), default=0.0)
    reverse_count = sum(float(record.get("reverse_command_count", 0.0)) for record in records)
    tracking_count = sum(float(record.get("tracking_sample_count", 0.0)) for record in records)
    v_stats = _aggregate_distribution(records, "teacher_v", "teacher_command_count")
    w_stats = _aggregate_distribution(records, "teacher_w", "teacher_command_count")
    summary = {
        "episodes": len(records),
        "success_count": successes,
        "success_rate": successes / count,
        "collision_count": collisions,
        "collision_rate": collisions / count,
        "timeout_count": timeouts,
        "timeout_rate": timeouts / count,
        "mean_final_goal_distance_m": sum(final) / count,
        "mean_path_length_m": sum(paths) / count,
        "path_efficiency": sum(min(1.0, goal / path) for goal, path in zip(initial, paths) if path > 0.0) / max(sum(path > 0.0 for path in paths), 1),
        "spl": sum(spl_values) / count,
        "mean_teacher_v_mps": v_stats["mean"],
        "std_teacher_v_mps": v_stats["std"],
        "min_teacher_v_mps": v_stats["min"],
        "max_teacher_v_mps": v_stats["max"],
        "mean_teacher_w_rps": w_stats["mean"],
        "std_teacher_w_rps": w_stats["std"],
        "min_teacher_w_rps": w_stats["min"],
        "max_teacher_w_rps": w_stats["max"],
        "teacher_command_count": int(teacher_commands),
        "reverse_command_count": int(reverse_count),
        "reverse_command_ratio": reverse_count / max(teacher_commands, 1.0),
        "projection_activation_count": int(projection_count),
        "projection_activation_ratio": projection_count / max(teacher_commands, 1.0),
        "mean_projection_correction_norm": projection_sum / max(teacher_commands, 1.0),
        "max_projection_correction_norm": projection_max,
        "governor_modification_count": int(governor_count),
        "governor_modification_ratio": governor_count / max(tracking_count, 1.0),
        "tracking_sample_count": int(tracking_count),
        "tracking_v_mae_mps": _aggregate_scalar(records, "tracking_v_abs_error_sum", "tracking_sample_count"),
        "tracking_w_mae_rps": _aggregate_scalar(records, "tracking_w_abs_error_sum", "tracking_sample_count"),
    }
    numeric = [value for value in summary.values() if isinstance(value, (int, float))]
    summary["finite"] = all(math.isfinite(float(value)) for value in numeric)
    return summary


def evaluate_teacher_gate(
    summary,
    success_threshold,
    collision_threshold=0.0,
    reverse_threshold=0.01,
    projection_correction_threshold=0.05,
):
    """Apply the short-distance formal teacher gate and return explainable checks."""
    threshold = float(success_threshold)
    checks = {
        "finite": bool(summary.get("finite", False)),
        "success_rate": float(summary.get("success_rate", -1.0)) >= threshold,
        "collision_rate": float(summary.get("collision_rate", 1.0)) <= float(collision_threshold),
        "reverse_command_ratio": float(summary.get("reverse_command_ratio", 1.0)) <= float(reverse_threshold),
        "mean_projection_correction_norm": float(summary.get("mean_projection_correction_norm", math.inf)) <= float(projection_correction_threshold),
    }
    return {
        "pass": all(checks.values()),
        "success_threshold": threshold,
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def _validate_inputs(goal_xy_robot, actual_velocity, obstacle_distance):
    if goal_xy_robot.ndim != 2 or goal_xy_robot.shape[1] != 2:
        raise ValueError("goal_xy_robot must have shape [N, 2]")
    if actual_velocity.shape != goal_xy_robot.shape:
        raise ValueError("actual_velocity must have shape [N, 2]")
    if obstacle_distance.ndim != 1 or obstacle_distance.shape[0] != goal_xy_robot.shape[0]:
        raise ValueError("obstacle_distance must have shape [N]")
    if not torch.isfinite(goal_xy_robot).all() or not torch.isfinite(actual_velocity).all():
        raise ValueError("teacher inputs must be finite")
    if torch.isnan(obstacle_distance).any() or torch.isneginf(obstacle_distance).any():
        raise ValueError("obstacle_distance must not contain NaN or -inf")


def _parameter(config, name, default=None):
    value = getattr(config, name, default)
    if value is None:
        raise ValueError("teacher config is missing %s" % name)
    return float(value)


def teacher_velocity_diagnostics(
    goal_xy_robot, actual_velocity, obstacle_distance, config
):
    """Return raw, requested, applied and projection-correction commands.

    ``actual_velocity`` is ``[v_actual, w_actual]`` in the robot frame and
    ``obstacle_distance`` is the nearest measured clearance in metres.  The
    teacher never bypasses V62: ``applied_command`` is the unchanged V62
    feasible projection of the raw teacher request.
    """
    _validate_inputs(goal_xy_robot, actual_velocity, obstacle_distance)
    dtype = goal_xy_robot.dtype
    device = goal_xy_robot.device
    distance = torch.linalg.vector_norm(goal_xy_robot, dim=1)
    bearing = torch.atan2(goal_xy_robot[:, 1], goal_xy_robot[:, 0])

    goal_radius = _parameter(config, "goal_radius")
    # The V62 plant has a small command/velocity lag.  Continue gently past
    # the nominal radius so the closed loop actually enters the environment's
    # success disk instead of settling a fraction of a millimetre outside it.
    control_goal_radius = max(
        0.0, goal_radius - _parameter(config, "goal_stop_margin", 0.05)
    )
    stopping_distance = _parameter(config, "stopping_distance", 0.80)
    if stopping_distance <= control_goal_radius:
        raise ValueError("stopping_distance must exceed goal_radius")
    approach = torch.clamp(
        (distance - control_goal_radius)
        / (stopping_distance - control_goal_radius),
        0.0,
        1.0,
    )
    # Forward-only V1 teacher: lateral goals reduce speed while heading turns
    # toward the target; the already measured V62 projection limits curvature.
    heading_factor = torch.clamp(torch.cos(bearing), min=0.0, max=1.0)

    obstacle_distance = obstacle_distance.to(device=device, dtype=dtype)
    finite_obstacle = torch.isfinite(obstacle_distance)
    safe_obstacle_distance = torch.where(
        finite_obstacle,
        obstacle_distance,
        torch.full_like(
            obstacle_distance, _parameter(config, "obstacle_slow_distance", 0.70)
        ),
    )
    obstacle_scale = torch.clamp(
        (
            safe_obstacle_distance
            - _parameter(config, "obstacle_stop_distance", 0.45)
        )
        / (
            _parameter(config, "obstacle_slow_distance", 0.70)
            - _parameter(config, "obstacle_stop_distance", 0.45)
        ),
        0.0,
        1.0,
    )
    desired_speed = (
        _parameter(config, "max_forward_speed")
        * approach
        * heading_factor
        * obstacle_scale
    )
    recovery_start = _parameter(config, "turn_recovery_bearing", 0.65)
    recovery_progress = torch.clamp(
        (torch.abs(bearing) - recovery_start)
        / (math.pi / 2.0 - recovery_start),
        0.0,
        1.0,
    )
    recovery_speed = (
        _parameter(config, "max_forward_speed")
        * _parameter(config, "turn_recovery_speed_fraction", 0.80)
        * recovery_progress
        * approach
    )
    # Do not accelerate toward a close obstacle merely to turn.  The V1
    # straight corridor has no internal obstacle, while future obstacle-aware
    # stages retain the braking behavior above.
    desired_speed = torch.maximum(
        desired_speed,
        torch.where(obstacle_scale >= 0.999, recovery_speed, torch.zeros_like(recovery_speed)),
    )
    # Feedback is deliberately braking-only.  It uses measured velocity to
    # slow an overspeeding plant without making normal acceleration request a
    # command larger than the reliable forward domain.
    raw_speed = desired_speed - _parameter(
        config, "velocity_feedback_gain", 0.20
    ) * torch.relu(actual_velocity[:, 0] - desired_speed)
    raw_yaw = _parameter(config, "heading_gain", 1.25) * bearing - _parameter(
        config, "yaw_rate_damping", 0.35
    ) * actual_velocity[:, 1]
    raw_command = torch.stack((raw_speed, raw_yaw), dim=1)

    applied_command = project_velocity_commands(
        raw_command,
        _parameter(config, "max_forward_speed"),
        _parameter(config, "max_yaw_rate"),
        _parameter(config, "minimum_turn_radius"),
        _parameter(config, "feasible_envelope_fraction"),
    )
    correction = applied_command - raw_command
    return {
        "raw_command": raw_command,
        "requested_command": raw_command.clone(),
        "applied_command": applied_command,
        "projection_correction": correction,
        "projection_correction_norm": torch.linalg.vector_norm(correction, dim=1),
        "goal_distance": distance,
        "goal_bearing": bearing,
        "desired_speed": desired_speed,
    }


def teacher_velocity_command(
    goal_xy_robot, actual_velocity, obstacle_distance, config
):
    """Return the feasible teacher ``[v, w]`` command for each environment."""
    return teacher_velocity_diagnostics(
        goal_xy_robot, actual_velocity, obstacle_distance, config
    )["applied_command"]
