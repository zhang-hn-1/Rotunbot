"""Scenario and metric helpers for the Stage1.4 matched-mode experiment."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GovernorScenario:
    name: str
    group: str
    commands: tuple


def _commands(*pairs):
    return tuple((float(v), float(w)) for v, w in pairs)


# The request sequences are deliberately fixed before looking at any result.
# Each request is held for ten 50 Hz policy steps by the GPU evaluator.
STAGE14_SCENARIOS = (
    GovernorScenario("low_speed_high_yaw_positive", "low_speed", _commands((.04, .10), (.04, .10), (.06, .08))),
    GovernorScenario("low_speed_high_yaw_negative", "low_speed", _commands((-.04, -.10), (-.04, -.10), (-.06, -.08))),
    GovernorScenario("near_low_speed_boundary", "low_speed", _commands((.06, .06), (.08, .06), (.10, .06))),
    GovernorScenario("low_speed_forward_ramp", "low_speed", _commands((0.0, 0.0), (.04, 0.0), (.08, .02), (.10, .02))),
    GovernorScenario("low_speed_yaw_reversal", "low_speed", _commands((.10, .05), (.10, -.05), (.10, .05))),
    GovernorScenario("high_speed_curvature_positive", "high_speed", _commands((.10, .06), (.13, .10), (.10, .06))),
    GovernorScenario("high_speed_curvature_negative", "high_speed", _commands((-.10, -.06), (-.13, -.10), (-.10, -.06))),
    GovernorScenario("high_speed_forward_reversal", "high_speed", _commands((.13, .04), (-.13, -.04), (.13, .04))),
    GovernorScenario("mixed_combined_reversal", "mixed", _commands((.04, .02), (-.10, -.08), (.12, .06), (-.06, -.04))),
    GovernorScenario("mixed_continuous_curvature", "mixed", _commands((.04, .02), (.08, .04), (.12, .06), (.08, .04), (.04, .02))),
)


def count_command_oscillations(values, epsilon=1.0e-8):
    signs = []
    for value in values:
        value = float(value)
        if abs(value) > float(epsilon):
            sign = 1 if value > 0.0 else -1
            if not signs or signs[-1] != sign:
                signs.append(sign)
    return max(0, len(signs) - 1)


def _percentile(values, fraction):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    position = (len(values) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    ratio = position - lower
    return values[lower] + ratio * (values[upper] - values[lower])


def aggregate_governor_rows(rows):
    """Return mode-level error distributions and governor counters."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["mode"], []).append(row)
    result = {}
    for mode, samples in grouped.items():
        v_errors = [abs(float(row["v_error"])) for row in samples]
        w_errors = [abs(float(row["w_error"])) for row in samples]
        selected_yaw = [
            float(row.get("selected_yaw", row.get("selected_w", 0.0)))
            for row in samples
        ]
        result[mode] = {
            "sample_count": len(samples),
            "v_error_mean": sum(v_errors) / len(v_errors) if v_errors else 0.0,
            "v_error_median": _percentile(v_errors, 0.50),
            "v_error_p90": _percentile(v_errors, 0.90),
            "v_error_max": max(v_errors) if v_errors else 0.0,
            "w_error_mean": sum(w_errors) / len(w_errors) if w_errors else 0.0,
            "w_error_median": _percentile(w_errors, 0.50),
            "w_error_p90": _percentile(w_errors, 0.90),
            "w_error_max": max(w_errors) if w_errors else 0.0,
            "command_modification_count": sum(bool(row.get("command_modified", False)) for row in samples),
            "static_saturation_count": sum(bool(row.get("static_saturated", False)) for row in samples),
            "fallback_count": sum(bool(row.get("fallback", False)) for row in samples),
            "yaw_sign_error_count": sum(bool(row.get("yaw_sign_error", False)) for row in samples),
            "oscillation_count": sum(bool(row.get("oscillation", False)) for row in samples),
            "selected_yaw_oscillations": count_command_oscillations(selected_yaw),
        }
        if any("requested_v_error" in row for row in samples):
            requested_v = [abs(float(row["requested_v_error"])) for row in samples]
            requested_w = [abs(float(row["requested_w_error"])) for row in samples]
            result[mode].update({
                "requested_v_error_mean": sum(requested_v) / len(requested_v),
                "requested_v_error_p90": _percentile(requested_v, 0.90),
                "requested_v_error_max": max(requested_v),
                "requested_w_error_mean": sum(requested_w) / len(requested_w),
                "requested_w_error_p90": _percentile(requested_w, 0.90),
                "requested_w_error_max": max(requested_w),
            })
    return result
