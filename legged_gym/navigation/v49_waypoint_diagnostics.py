"""Pure, diagnostic-only calculations for the Stage 1.1 audit.

This module deliberately does not alter the V49 controller.  It reuses the
existing feasible-set projection when a diagnostic rolling floor is enabled
and exposes only measurements/classifications for the evaluator.
"""

from dataclasses import dataclass
import math

import torch

from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import (
    project_velocity_commands,
)


@dataclass(frozen=True)
class DiagnosticMode:
    """Optional audit switches; defaults are exactly the Stage 1 behavior."""

    smooth_reference: bool = False
    minimum_rolling_speed: float = None

    def __post_init__(self):
        if self.minimum_rolling_speed is not None and self.minimum_rolling_speed <= 0.0:
            raise ValueError("minimum_rolling_speed must be positive or None")


def _with_rolling_floor(raw_command, minimum_rolling_speed):
    adjusted = raw_command.clone()
    if minimum_rolling_speed is None:
        return adjusted
    floor = float(minimum_rolling_speed)
    moving_sign = torch.where(
        adjusted[:, 0] >= 0.0,
        torch.ones_like(adjusted[:, 0]),
        -torch.ones_like(adjusted[:, 0]),
    )
    below_floor = torch.abs(adjusted[:, 0]) < floor
    adjusted[below_floor, 0] = moving_sign[below_floor] * floor
    return adjusted


def apply_diagnostic_projection(
    raw_command,
    mode,
    maximum_forward_speed=0.13,
    maximum_yaw_rate=0.10,
    minimum_turn_radius=3.148148148148148,
    envelope_fraction=0.85,
    stationary_threshold=0.02,
    turn_authority_start_speed=0.08,
    turn_authority_full_speed=0.10,
):
    """Apply only the optional rolling floor, then the unchanged V49 projection."""
    if not isinstance(mode, DiagnosticMode):
        raise TypeError("mode must be DiagnosticMode")
    _, projected = apply_diagnostic_command(
        raw_command,
        mode,
        maximum_forward_speed,
        maximum_yaw_rate,
        minimum_turn_radius,
        envelope_fraction,
        stationary_threshold,
        turn_authority_start_speed,
        turn_authority_full_speed,
    )
    return projected


def apply_diagnostic_command(
    raw_command,
    mode,
    maximum_forward_speed=0.13,
    maximum_yaw_rate=0.10,
    minimum_turn_radius=3.148148148148148,
    envelope_fraction=0.85,
    stationary_threshold=0.02,
    turn_authority_start_speed=0.08,
    turn_authority_full_speed=0.10,
):
    """Return diagnostic raw/projected commands without changing V49 config."""
    if not isinstance(mode, DiagnosticMode):
        raise TypeError("mode must be DiagnosticMode")
    adjusted = _with_rolling_floor(
        torch.as_tensor(raw_command, dtype=torch.float32),
        mode.minimum_rolling_speed,
    )
    projected = project_velocity_commands(
        adjusted,
        maximum_forward_speed,
        maximum_yaw_rate,
        minimum_turn_radius,
        envelope_fraction,
        stationary_threshold,
        turn_authority_start_speed,
        turn_authority_full_speed,
    )
    return adjusted, projected


def summarize_command_transitions(projected_v, projected_w):
    """Summarize adjacent 5 Hz command deltas, never 50 Hz action deltas."""
    v = torch.as_tensor(projected_v, dtype=torch.float32).reshape(-1)
    w = torch.as_tensor(projected_w, dtype=torch.float32).reshape(-1)
    if v.numel() != w.numel():
        raise ValueError("projected_v and projected_w must have equal length")
    if v.numel() < 2:
        deltas_v = torch.zeros(0)
        deltas_w = torch.zeros(0)
    else:
        deltas_v = torch.abs(v[1:] - v[:-1])
        deltas_w = torch.abs(w[1:] - w[:-1])

    def _summary(values, first_threshold, second_threshold, prefix):
        if values.numel() == 0:
            return {
                "mean_abs_delta_%s" % prefix: 0.0,
                "p95_abs_delta_%s" % prefix: 0.0,
                "max_abs_delta_%s" % prefix: 0.0,
                "fraction_abs_delta_%s_gt_%s" % (prefix, first_threshold): 0.0,
                "fraction_abs_delta_%s_gt_%s" % (prefix, second_threshold): 0.0,
            }
        return {
            "mean_abs_delta_%s" % prefix: float(values.mean()),
            "p95_abs_delta_%s" % prefix: float(torch.quantile(values, 0.95)),
            "max_abs_delta_%s" % prefix: float(values.max()),
            "fraction_abs_delta_%s_gt_%s" % (prefix, first_threshold): float(
                (values > first_threshold).float().mean()
            ),
            "fraction_abs_delta_%s_gt_%s" % (prefix, second_threshold): float(
                (values > second_threshold).float().mean()
            ),
        }

    result = _summary(deltas_v, 0.008, 0.016, "v")
    result.update(_summary(deltas_w, 0.004, 0.008, "w"))
    return result


def yaw_sign_reversal_count(previous_w, current_w, meaningful_threshold=0.01):
    """Count meaningful sign reversals between adjacent 5 Hz commands."""
    previous = torch.as_tensor(previous_w, dtype=torch.float32).reshape(-1)
    current = torch.as_tensor(current_w, dtype=torch.float32).reshape(-1)
    if previous.numel() != current.numel():
        raise ValueError("previous_w and current_w must have equal length")
    reversal = (
        (previous * current < 0.0)
        & (torch.abs(previous) >= meaningful_threshold)
        & (torch.abs(current) >= meaningful_threshold)
    )
    return int(reversal.sum().item())


def rate_feedforward_active_ratio(action_0, action_1, epsilon=1.0e-6):
    """Return the fraction of ticks with a non-negligible rate-FF component."""
    first = torch.as_tensor(action_0, dtype=torch.float32).reshape(-1)
    second = torch.as_tensor(action_1, dtype=torch.float32).reshape(-1)
    if first.numel() != second.numel():
        raise ValueError("rate feedforward action channels must have equal length")
    if first.numel() == 0:
        return 0.0
    active = (torch.abs(first) > epsilon) | (torch.abs(second) > epsilon)
    return float(active.float().mean())


def dynamic_transition_severity(delta_v, delta_w):
    """Normalize a 5 Hz transition against V49 random-walk increments."""
    return max(abs(float(delta_v)) / 0.008, abs(float(delta_w)) / 0.004)


def detect_low_speed_yaw_collapse(ticks, tick_period_s=0.2):
    """Detect >=0.4 s below 0.08 m/s with >=10 degree bearing not reducing 20%."""
    minimum_ticks = int(math.ceil(0.4 / float(tick_period_s)))
    segment = []
    for tick in list(ticks) + [None]:
        valid = tick is not None and abs(float(tick["measured_v"])) < 0.08
        valid = valid and abs(float(tick["bearing_error"])) >= math.radians(10.0)
        if valid:
            segment.append(abs(float(tick["bearing_error"])))
            continue
        if len(segment) >= minimum_ticks:
            if min(segment) >= segment[0] * 0.80:
                return True
        segment = []
    return False


def summarize_terminal_results(results):
    """Count settling only for completed routes; incomplete routes stay separate."""
    completed = [result for result in results if bool(result["route_complete"])]
    incomplete_count = len(results) - len(completed)
    safe_count = sum(bool(result.get("terminal_speed_safe", False)) for result in completed)
    distances = sorted(
        float(result["stop_distance"])
        for result in completed
        if result.get("stop_distance") is not None
    )

    def percentile(values, fraction):
        if not values:
            return None
        position = (len(values) - 1) * fraction
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return {
        "route_complete_count": len(completed),
        "route_incomplete_count": incomplete_count,
        "completed_terminal_speed_safe_count": safe_count,
        "completed_terminal_speed_failure_count": len(completed) - safe_count,
        "mean_stop_distance": (
            sum(distances) / len(distances) if distances else None
        ),
        "p95_stop_distance": percentile(distances, 0.95),
    }
