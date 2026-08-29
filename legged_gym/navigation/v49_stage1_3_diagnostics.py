"""Pure logging and aggregation helpers for the Stage1.3 sweep."""

import math
import statistics


STAGE13_HORIZONS_MS = (50, 100, 150, 200)
STAGE13_TRACE_FIELDS = (
    "seed", "env_id", "trial_id", "simulation_dt", "control_dt",
    "policy_step", "time_s", "initial_forward_velocity", "initial_yaw_rate",
    "forward_velocity_command", "yaw_rate_command", "projected_forward_velocity",
    "projected_yaw_rate", "actual_v", "actual_w", "yaw", "yaw_rate",
    "root_position_x", "root_position_y", "root_position_z",
    "body_linear_velocity_x", "body_linear_velocity_y", "body_linear_velocity_z",
    "body_angular_velocity_x", "body_angular_velocity_y", "body_angular_velocity_z",
    "joint1_position", "joint1_velocity", "joint2_position", "joint2_velocity",
    "nominal_action_0", "nominal_action_1", "feedback_action_0", "feedback_action_1",
    "final_action_0", "final_action_1", "simulation_unstable",
)

STAGE13_SUMMARY_FIELDS = (
    "initial_forward_velocity", "initial_yaw_rate", "forward_velocity_command",
    "yaw_rate_command", "projected_forward_velocity", "projected_yaw_rate",
    "actual_v_50ms", "actual_w_50ms", "actual_v_100ms", "actual_w_100ms",
    "actual_v_150ms", "actual_w_150ms", "actual_v_200ms", "actual_w_200ms",
    "cumulative_yaw_change_200ms", "body_displacement_x_200ms",
    "body_displacement_y_200ms", "forward_direction_reversed",
    "yaw_direction_reversed", "simulation_unstable",
)


def horizon_policy_step(horizon_ms, control_dt):
    """Return the first 1-indexed policy sample at/after a requested horizon."""
    horizon_ms = float(horizon_ms)
    control_dt = float(control_dt)
    if horizon_ms <= 0.0 or control_dt <= 0.0:
        raise ValueError("horizon_ms and control_dt must be positive")
    return int(math.ceil((horizon_ms / 1000.0) / control_dt - 1.0e-12))


def symmetric_yaw_grid(max_abs_yaw, step):
    """Create an exactly symmetric signed yaw-rate grid in rad/s."""
    maximum = float(max_abs_yaw)
    increment = float(step)
    if maximum <= 0.0 or increment <= 0.0:
        raise ValueError("max_abs_yaw and step must be positive")
    count = int(round(maximum / increment))
    if abs(count * increment - maximum) > 1.0e-8:
        raise ValueError("max_abs_yaw must be an integer multiple of step")
    return tuple(round(index * increment, 12) for index in range(-count, count + 1))


def _interpolate_samples(samples, horizon_ms, control_dt):
    """Interpolate 50 Hz samples to exact 50/100/150/200 ms timestamps."""
    if len(samples) < 1:
        raise ValueError("at least one policy sample is required")
    target_step = float(horizon_ms) / 1000.0 / float(control_dt)
    if target_step <= 0.0 or target_step > len(samples) + 1.0e-5:
        raise ValueError("horizon exceeds trace")
    target_step = min(target_step, float(len(samples)))
    lower = int(math.floor(target_step))
    upper = int(math.ceil(target_step))
    if lower == upper:
        return float(samples[lower - 1])
    fraction = target_step - lower
    return float(samples[lower - 1]) + fraction * (
        float(samples[upper - 1]) - float(samples[lower - 1])
    )


def detect_direction_reversal(initial_value, samples, epsilon=1.0e-5):
    """Return true if any measured sample opposes a nonzero initial direction."""
    initial = float(initial_value)
    if abs(initial) <= float(epsilon):
        return False
    return any(float(value) * initial < -(float(epsilon) ** 2) for value in samples)


def detect_command_direction_reversal(command_value, samples, epsilon=1.0e-5):
    """Return true if a nonzero command produces an opposing velocity sample."""
    command = float(command_value)
    if abs(command) <= float(epsilon):
        return False
    return any(float(value) * command < -(float(epsilon) ** 2) for value in samples)


def _mean(values):
    return sum(values) / float(len(values)) if values else None


def _std(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def aggregate_stage13_trials(rows):
    """Aggregate repeated transition summaries into one table row."""
    rows = list(rows)
    if not rows:
        raise ValueError("cannot aggregate empty trial list")
    result = {
        "initial_forward_velocity": _mean([float(row["initial_forward_velocity"]) for row in rows]),
        "initial_yaw_rate": _mean([float(row["initial_yaw_rate"]) for row in rows]),
        "forward_velocity_command": _mean([float(row["forward_velocity_command"]) for row in rows]),
        "yaw_rate_command": _mean([float(row["yaw_rate_command"]) for row in rows]),
        "projected_forward_velocity": _mean([float(row["projected_forward_velocity"]) for row in rows]),
        "projected_yaw_rate": _mean([float(row["projected_yaw_rate"]) for row in rows]),
        "repeat_count": len(rows),
        "simulation_instability_count": sum(bool(row["simulation_unstable"]) for row in rows),
        "forward_direction_reversal_count": sum(bool(row["forward_direction_reversed"]) for row in rows),
        "yaw_direction_reversal_count": sum(bool(row["yaw_direction_reversed"]) for row in rows),
    }
    for horizon in STAGE13_HORIZONS_MS:
        for axis in ("v", "w"):
            name = "actual_%s_%dms" % (axis, horizon)
            values = [float(row[name]) for row in rows]
            result["mean_" + name] = _mean(values)
            result["std_" + name] = _std(values)
    for name in ("cumulative_yaw_change_200ms", "body_displacement_x_200ms", "body_displacement_y_200ms"):
        values = [float(row[name]) for row in rows]
        result["mean_" + name] = _mean(values)
        result["std_" + name] = _std(values)
    return result


def summarize_trace(samples, initial, projected_command, control_dt):
    """Create the horizon summary for one completed transition."""
    v_samples = [float(row["actual_v"]) for row in samples]
    w_samples = [float(row["actual_w"]) for row in samples]
    yaw_samples = [float(row["yaw"]) for row in samples]
    displacement_x = [float(row["root_position_x"]) for row in samples]
    displacement_y = [float(row["root_position_y"]) for row in samples]
    initial_yaw = float(initial.get("initial_yaw", yaw_samples[0]))
    initial_x = float(initial.get("initial_root_position_x", displacement_x[0]))
    initial_y = float(initial.get("initial_root_position_y", displacement_y[0]))
    cumulative_yaw = math.atan2(
        math.sin(yaw_samples[-1] - initial_yaw),
        math.cos(yaw_samples[-1] - initial_yaw),
    )
    summary = {
        "initial_forward_velocity": float(initial["initial_forward_velocity"]),
        "initial_yaw_rate": float(initial["initial_yaw_rate"]),
        "forward_velocity_command": float(initial["forward_velocity_command"]),
        "yaw_rate_command": float(initial["yaw_rate_command"]),
        "projected_forward_velocity": float(projected_command[0]),
        "projected_yaw_rate": float(projected_command[1]),
        "cumulative_yaw_change_200ms": cumulative_yaw,
        "body_displacement_x_200ms": displacement_x[-1] - initial_x,
        "body_displacement_y_200ms": displacement_y[-1] - initial_y,
        # A zero-velocity initial state has no direction to reverse.  The
        # transition diagnostic therefore uses the signed projected request
        # for command-direction errors; the initial-state helper remains
        # available for audits that specifically need momentum reversal.
        "forward_direction_reversed": detect_command_direction_reversal(
            projected_command[0], v_samples
        ),
        "yaw_direction_reversed": detect_command_direction_reversal(
            projected_command[1], w_samples
        ),
        "simulation_unstable": any(
            not math.isfinite(value) for value in v_samples + w_samples + yaw_samples
        ),
    }
    for horizon in STAGE13_HORIZONS_MS:
        summary["actual_v_%dms" % horizon] = _interpolate_samples(
            v_samples, horizon, control_dt
        )
        summary["actual_w_%dms" % horizon] = _interpolate_samples(
            w_samples, horizon, control_dt
        )
    return summary
