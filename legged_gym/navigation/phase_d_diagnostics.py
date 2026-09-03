"""Pure diagnostics for the Phase-D execution-chain gates."""

import math
from collections import Counter

import numpy as np


def _mean(rows, key):
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else None


def command_loss_breakdown(rows):
    """Compute projection, transition, and plant tracking loss ratios."""
    rows = list(rows)
    if not rows:
        return {
            "sample_count": 0,
            "mean_teacher_raw_v_mps": None,
            "mean_projected_v_mps": None,
            "mean_command_target_v_mps": None,
            "mean_applied_v_mps": None,
            "mean_actual_v_mps": None,
            "r_projection": None,
            "r_target_projection": None,
            "r_transition": None,
            "r_tracking": None,
            "fraction_projected_loss": None,
            "fraction_transition_loss": None,
            "fraction_tracking_loss": None,
            "transition_state_counts": {},
        }
    raw = np.asarray([abs(float(row.get("desired_v_raw_mps", row.get("teacher_raw_v_mps", 0.0)))) for row in rows])
    projected = np.asarray([abs(float(row.get("desired_v_projected_mps", row.get("teacher_projected_v_mps", 0.0)))) for row in rows])
    target = np.asarray([abs(float(row.get("command_target_v_mps", row.get("desired_v_scheduled_mps", 0.0)))) for row in rows])
    applied = np.asarray([abs(float(row.get("applied_v_mps", row.get("transition_applied_v_mps", 0.0)))) for row in rows])
    actual = np.asarray([abs(float(row.get("actual_v_mps", 0.0))) for row in rows])
    raw_mean = float(raw.mean())
    projected_mean = float(projected.mean())
    target_mean = float(target.mean())
    applied_mean = float(applied.mean())
    actual_mean = float(actual.mean())
    state_counts = Counter(str(row.get("transition_state", "UNKNOWN")) for row in rows)
    return {
        "sample_count": int(len(rows)),
        "mean_teacher_raw_v_mps": raw_mean,
        "mean_projected_v_mps": projected_mean,
        "mean_command_target_v_mps": target_mean,
        "mean_applied_v_mps": applied_mean,
        "mean_actual_v_mps": actual_mean,
        "r_projection": projected_mean / max(raw_mean, 1.0e-9),
        "r_target_projection": target_mean / max(projected_mean, 1.0e-9),
        "r_transition": applied_mean / max(target_mean, 1.0e-9),
        "r_tracking": actual_mean / max(applied_mean, 1.0e-9),
        "fraction_projected_loss": float(np.mean(projected < raw - 0.01)),
        "fraction_target_projection_loss": float(np.mean(target < projected - 0.01)),
        "fraction_transition_loss": float(np.mean(applied < target - 0.01)),
        "fraction_tracking_loss": float(np.mean(actual < applied - 0.01)),
        "transition_state_counts": dict(state_counts),
        "mean_abs_teacher_w_rps": _mean(rows, "desired_w_raw_rps"),
        "mean_abs_actual_w_rps": _mean(rows, "actual_w_rps"),
        "mean_goal_progress_m": (
            float(rows[0].get("global_goal_distance_m", 0.0))
            - float(rows[-1].get("global_goal_distance_m", 0.0))
        ),
    }


def classify_constant_command_gate(summary, *, minimum_ratio=0.60, maximum_p90_error=0.08):
    """Return a conservative D0-A verdict from calibration summaries."""
    if summary.get("error"):
        return "FAIL"
    ratio = summary.get("actual_over_applied")
    p90 = summary.get("p90_tracking_error_v_mps")
    if ratio is None or p90 is None:
        return "FAIL"
    return "PASS" if ratio >= float(minimum_ratio) and p90 <= float(maximum_p90_error) else "FAIL"
