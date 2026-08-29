"""Pure diagnostics for Stage 1.2 V49 state and reachability audits.

The helpers in this module intentionally have no simulator or controller
side-effects.  Runtime scripts may feed them snapshots captured from the
existing V49 environment, while unit tests can exercise the comparison and
metric contracts on CPU.
"""

import math


RAW_TRACE_FIELDS = (
    "episode_id", "trial_id", "time_s", "physics_step", "policy_step",
    "high_level_tick", "step_within_high_level_tick", "initial_v", "initial_w",
    "desired_v", "desired_w", "projected_v", "projected_w",
    "previous_command_v", "previous_command_w", "delta_command_v",
    "delta_command_w", "measured_v", "measured_w", "v_tracking_error",
    "w_tracking_error", "root_linear_velocity_world_0",
    "root_linear_velocity_world_1", "root_linear_velocity_world_2",
    "root_linear_velocity_body_0", "root_linear_velocity_body_1",
    "root_linear_velocity_body_2", "root_angular_velocity_world_0",
    "root_angular_velocity_world_1", "root_angular_velocity_world_2",
    "root_angular_velocity_body_0", "root_angular_velocity_body_1",
    "root_angular_velocity_body_2", "yaw", "yaw_rate", "joint1_position",
    "joint1_velocity", "joint2_position", "joint2_velocity",
    "nominal_action_0", "nominal_action_1", "feedback_action_0",
    "feedback_action_1", "derivative_action_0", "derivative_action_1",
    "feedforward_action_0", "feedforward_action_1", "rate_feedforward_action_0",
    "rate_feedforward_action_1", "residual_action_0", "residual_action_1",
    "combined_action_0", "combined_action_1", "final_action_0", "final_action_1",
    "smooth_reference_flag", "rate_feedforward_active", "contact_yaw_damping_factor",
    "low_speed_lt_010", "low_speed_lt_008", "direction_agreement_v",
    "direction_agreement_w",
)

REACHABILITY_GRID_FIELDS = (
    "initial_v_bin", "initial_w_bin", "target_v", "target_w", "projected_v",
    "projected_w", "repeat_count", "mean_actual_v_200ms", "std_actual_v_200ms",
    "mean_actual_w_200ms", "std_actual_w_200ms", "mean_v_error_200ms",
    "mean_w_error_200ms", "response_reachable_rate", "tracking_reachable_rate",
    "forward_sign_failure_rate", "yaw_sign_failure_rate",
)


def _flatten(value):
    """Convert scalars, sequences, and torch-like tensors to floats."""
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if hasattr(value, "reshape"):
            value = value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return [value]


def _difference(reference, candidate, abs_tol, rel_tol):
    left = _flatten(reference)
    right = _flatten(candidate)
    if len(left) == 1 and len(right) > 1:
        left = left * len(right)
    elif len(right) == 1 and len(left) > 1:
        right = right * len(left)
    if len(left) != len(right):
        return False, float("inf"), float("inf")
    max_abs = 0.0
    max_rel = 0.0
    equivalent = True
    for first, second in zip(left, right):
        try:
            absolute = abs(float(first) - float(second))
            scale = max(abs(float(first)), abs(float(second)), 1.0e-12)
        except (TypeError, ValueError):
            absolute = 0.0 if first == second else float("inf")
            scale = 1.0
        relative = absolute / scale
        max_abs = max(max_abs, absolute)
        max_rel = max(max_rel, relative)
        if absolute > abs_tol + rel_tol * scale:
            equivalent = False
    return equivalent, max_abs, max_rel


def compare_snapshot_sequences(
    reference_snapshots,
    candidate_snapshots,
    fields=None,
    abs_tol=1.0e-6,
    rel_tol=1.0e-5,
):
    """Find the first policy step/variable that diverges between traces."""
    reference = list(reference_snapshots)
    candidate = list(candidate_snapshots)
    if fields is None:
        fields = sorted(
            set().union(*(set(row) for row in reference + candidate))
        )
    for step, (left, right) in enumerate(zip(reference, candidate)):
        for name in fields:
            if name not in left or name not in right:
                return {
                    "equivalent": False,
                    "compared_policy_steps": step,
                    "reference_length": len(reference),
                    "candidate_length": len(candidate),
                    "first_divergence_policy_step": step,
                    "first_divergence_variable": name,
                    "absolute_difference": float("inf"),
                    "relative_difference": float("inf"),
                }
            equal, absolute, relative = _difference(
                left[name], right[name], abs_tol, rel_tol
            )
            if not equal:
                return {
                    "equivalent": False,
                    "compared_policy_steps": step + 1,
                    "reference_length": len(reference),
                    "candidate_length": len(candidate),
                    "first_divergence_policy_step": step,
                    "first_divergence_variable": name,
                    "absolute_difference": absolute,
                    "relative_difference": relative,
                }
    if len(reference) != len(candidate):
        step = min(len(reference), len(candidate))
        return {
            "equivalent": False,
            "compared_policy_steps": step,
            "reference_length": len(reference),
            "candidate_length": len(candidate),
            "first_divergence_policy_step": step,
            "first_divergence_variable": "sequence_length",
            "absolute_difference": float("inf"),
            "relative_difference": float("inf"),
        }
    return {
        "equivalent": True,
        "compared_policy_steps": len(reference),
        "reference_length": len(reference),
        "candidate_length": len(candidate),
        "first_divergence_policy_step": None,
        "first_divergence_variable": None,
        "absolute_difference": 0.0,
        "relative_difference": 0.0,
    }


def reset_audit_rows(specs, before, after, abs_tol=1.0e-6, rel_tol=1.0e-5):
    """Build an auditable PASS/FAIL/NOT_AVAILABLE row per runtime state."""
    rows = []
    for spec in specs:
        name = spec["name"]
        expected = spec.get("expected", 0.0)
        location = spec.get("location", "unknown")
        before_value = before.get(name)
        after_value = after.get(name)
        base = {
            "variable_name": name,
            "location": location,
            "expected_reset_value": expected,
            "before_reset": before_value,
            "after_reset": after_value,
            "absolute_difference": None,
            "relative_difference": None,
            "availability": "available",
            "status": "PASS",
            "notes": "",
        }
        if before_value is None or after_value is None:
            base["availability"] = "not_available"
            base["status"] = "NOT_AVAILABLE"
            rows.append(base)
            continue
        if isinstance(expected, str):
            base["status"] = "PASS"
            base["notes"] = expected
            rows.append(base)
            continue
        equal, absolute, relative = _difference(
            after_value, expected, abs_tol, rel_tol
        )
        base["absolute_difference"] = absolute
        base["relative_difference"] = relative
        if not equal:
            base["status"] = "FAIL"
        rows.append(base)
    return rows


def high_level_alignment(policy_step, policy_steps_per_tick=10, policy_dt=0.02):
    """Return `(high_level_tick, step_within_tick, time_s)` for a 50 Hz row."""
    step = int(policy_step)
    interval = int(policy_steps_per_tick)
    if interval <= 0:
        raise ValueError("policy_steps_per_tick must be positive")
    return step // interval, step % interval, step * float(policy_dt)


def dynamic_response_ratio(initial, target, actual, denominator_epsilon=1.0e-8):
    """Return response fraction, or None when the target delta is undefined."""
    denominator = float(target) - float(initial)
    if abs(denominator) <= float(denominator_epsilon):
        return None
    return (float(actual) - float(initial)) / denominator


def direction_agreement(target_delta, actual_delta, epsilon=1.0e-6):
    """Whether actual motion agrees with a requested delta direction."""
    target = float(target_delta)
    actual = float(actual_delta)
    if abs(target) <= epsilon:
        return abs(actual) <= epsilon
    return actual * target > 0.0


def response_reachable(target_delta, actual_delta, minimum_fraction=0.2):
    """Return whether a transition moves in the right direction sufficiently."""
    target = float(target_delta)
    actual = float(actual_delta)
    if abs(target) <= 1.0e-8:
        return abs(actual) <= 1.0e-3
    return direction_agreement(target, actual) and abs(actual) >= (
        float(minimum_fraction) * abs(target)
    )


def velocity_bin(speed):
    """Use the Stage1.1 low-speed boundaries for an absolute speed."""
    value = abs(float(speed))
    if value < 0.08:
        return "lt_0.08"
    if value < 0.10:
        return "0.08_to_0.10"
    return "ge_0.10"


def summarize_reachability_samples(samples):
    """Aggregate repeat rows that already contain 200 ms outcome fields."""
    rows = list(samples)
    if not rows:
        return {
            "repeat_count": 0,
            "mean_actual_v_200ms": None,
            "std_actual_v_200ms": None,
            "mean_actual_w_200ms": None,
            "std_actual_w_200ms": None,
            "mean_v_error_200ms": None,
            "mean_w_error_200ms": None,
            "response_reachable_rate": None,
            "tracking_reachable_rate": None,
            "forward_sign_failure_rate": None,
            "yaw_sign_failure_rate": None,
        }

    def values(name):
        return [float(row[name]) for row in rows]

    def mean(name):
        data = values(name)
        return sum(data) / len(data)

    def std(name):
        data = values(name)
        average = sum(data) / len(data)
        return math.sqrt(sum((item - average) ** 2 for item in data) / len(data))

    def rate(name, invert=False):
        data = [bool(row[name]) for row in rows]
        if invert:
            data = [not item for item in data]
        return sum(data) / float(len(data))

    def response_reachable_sample(row):
        return all((
            response_reachable(
                row["target_v"] - row["initial_v"],
                row["actual_v_200ms"] - row["initial_v"],
            ),
            response_reachable(
                row["target_w"] - row["initial_w"],
                row["actual_w_200ms"] - row["initial_w"],
            ),
        ))

    def tracking_reachable(row):
        return (
            abs(row["actual_v_200ms"] - row["target_v"])
            <= float(row["v_tracking_tolerance"])
            and abs(row["actual_w_200ms"] - row["target_w"])
            <= float(row["w_tracking_tolerance"])
        )

    response_values = [
        bool(row.get("response_reachable", response_reachable_sample(row)))
        for row in rows
    ]
    tracking_values = [
        bool(row.get("tracking_reachable", tracking_reachable(row)))
        for row in rows
    ]
    forward_values = [
        bool(row.get("forward_sign_correct", direction_agreement(
            row["target_v"] - row["initial_v"],
            row["actual_v_200ms"] - row["initial_v"],
        )))
        for row in rows
    ]
    yaw_values = [
        bool(row.get("yaw_sign_correct", direction_agreement(
            row["target_w"] - row["initial_w"],
            row["actual_w_200ms"] - row["initial_w"],
        )))
        for row in rows
    ]

    return {
        "repeat_count": len(rows),
        "mean_actual_v_200ms": mean("actual_v_200ms"),
        "std_actual_v_200ms": std("actual_v_200ms"),
        "mean_actual_w_200ms": mean("actual_w_200ms"),
        "std_actual_w_200ms": std("actual_w_200ms"),
        "mean_v_error_200ms": mean("actual_v_200ms") - mean("target_v"),
        "mean_w_error_200ms": mean("actual_w_200ms") - mean("target_w"),
        "response_reachable_rate": sum(response_values) / float(len(rows)),
        "tracking_reachable_rate": sum(tracking_values) / float(len(rows)),
        "forward_sign_failure_rate": 1.0 - sum(forward_values) / float(len(rows)),
        "yaw_sign_failure_rate": 1.0 - sum(yaw_values) / float(len(rows)),
    }
