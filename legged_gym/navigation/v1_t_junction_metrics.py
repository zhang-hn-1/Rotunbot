"""Auditable episode, pair, and ablation metrics for T-junction evaluation."""

import math


def _rate(records, key):
    return sum(bool(row.get(key, False)) for row in records) / max(len(records), 1)


def _branch_accuracy(records):
    usable = [row for row in records if "branch_prediction" in row and "expected_branch" in row]
    return sum(str(row["branch_prediction"]).upper() == str(row["expected_branch"]).upper() for row in usable) / max(len(usable), 1)


def _pair_consistency(records, pairs):
    by_id = {row.get("episode_id"): row for row in records}
    checks = []
    for pair in pairs:
        if isinstance(pair, dict):
            ids = (pair.get("left"), pair.get("right"))
        else:
            ids = tuple(pair)
        if len(ids) != 2 or ids[0] not in by_id or ids[1] not in by_id:
            raise ValueError("each pair must identify two records")
        first, second = by_id[ids[0]], by_id[ids[1]]
        checks.append(
            "branch_prediction" in first
            and "branch_prediction" in second
            and str(first["branch_prediction"]).upper()
            == str(first.get("expected_branch", "")).upper()
            and str(second["branch_prediction"]).upper()
            == str(second.get("expected_branch", "")).upper()
        )
    return sum(checks) / max(len(checks), 1)


def aggregate_t_gate(records, pairs, ablations):
    """Aggregate records and apply the fixed T-junction release gate."""
    records = list(records)
    if not records:
        raise ValueError("at least one T-junction record is required")
    for record in records:
        for value in record.values():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise ValueError("T-junction records must contain finite numeric values")
    values = {
        "episodes": len(records),
        "success_rate": _rate(records, "success"),
        "collision_rate": _rate(records, "collision"),
        "timeout_rate": _rate(records, "timeout"),
        "wrong_turn_rate": _rate(records, "wrong_turn"),
        "turn_completion_rate": _rate(records, "turn_completion"),
        "exit_rate": _rate(records, "exit"),
        "branch_accuracy": _branch_accuracy(records),
        "goal_consistency_rate": _pair_consistency(records, pairs),
        "ablations": dict(ablations),
    }
    backends = {row.get("depth_backend_actual", row.get("depth_backend")) for row in records}
    values["depth_backend_actual"] = sorted(str(item) for item in backends)
    values["checks"] = {
        "success_rate_ge_0.80": values["success_rate"] >= 0.80,
        "collision_rate_le_0.10": values["collision_rate"] <= 0.10,
        "timeout_rate_le_0.10": values["timeout_rate"] <= 0.10,
        "wrong_turn_rate_le_0.10": values["wrong_turn_rate"] <= 0.10,
        "turn_completion_rate_ge_0.80": values["turn_completion_rate"] >= 0.80,
        "exit_rate_ge_0.80": values["exit_rate"] >= 0.80,
        "branch_accuracy_ge_0.80": values["branch_accuracy"] >= 0.80,
        "goal_consistency_rate_ge_0.80": values["goal_consistency_rate"] >= 0.80,
        "real_depth_backend": backends == {"isaacgym"},
    }
    numeric = [item for item in values.values() if isinstance(item, (int, float))]
    values["finite"] = all(math.isfinite(float(item)) for item in numeric)
    values["pass"] = values["finite"] and all(values["checks"].values())
    return values
