"""Auditable episode, pair, and release metrics for T-junction evaluation."""

from collections.abc import Mapping
import math
import numbers

import numpy as np


_SCENARIOS = ("T_LEFT", "T_RIGHT")
_EXPECTED_BRANCH = {"T_LEFT": "LEFT", "T_RIGHT": "RIGHT"}
_PAIR_METADATA = ("seed", "initial_pose", "initial_yaw", "horizon")


def _assert_finite(value, path):
    """Reject non-finite Python, NumPy, and tensor-like numeric payloads."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_finite(nested, "%s.%s" % (path, key))
        return
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            for index, nested in np.ndenumerate(value):
                _assert_finite(nested, "%s%s" % (path, index))
        elif not bool(np.isfinite(value).all()):
            raise ValueError("%s must contain finite numeric values" % path)
        return
    if isinstance(value, numbers.Number):
        if not math.isfinite(float(value)):
            raise ValueError("%s must contain finite numeric values" % path)
        return
    finite = getattr(value, "isfinite", None)
    if callable(finite):
        result = finite()
        all_values = getattr(result, "all", None)
        result = all_values() if callable(all_values) else result
        item = getattr(result, "item", None)
        result = item() if callable(item) else result
        if not bool(result):
            raise ValueError("%s must contain finite numeric values" % path)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for index, nested in enumerate(value):
            _assert_finite(nested, "%s[%d]" % (path, index))


def _rate(records, key):
    return sum(bool(row.get(key, False)) for row in records) / len(records)


def _branch_accuracy(records):
    return sum(
        str(row.get("branch_prediction", "")).upper()
        == str(row.get("expected_branch", "")).upper()
        for row in records
    ) / len(records)


def _scenario(record):
    value = str(record.get("scenario", "")).upper()
    if value not in _SCENARIOS:
        raise ValueError("each record must declare scenario T_LEFT or T_RIGHT")
    return value


def _role(record):
    value = str(record.get("policy_role", record.get("role", "student"))).lower()
    if value not in ("student", "teacher"):
        raise ValueError("policy_role must be student or teacher")
    return value


def _same_value(first, second):
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return bool(np.array_equal(np.asarray(first), np.asarray(second)))
    try:
        equal = first == second
    except (TypeError, ValueError):
        return False
    all_values = getattr(equal, "all", None)
    equal = all_values() if callable(all_values) else equal
    item = getattr(equal, "item", None)
    equal = item() if callable(item) else equal
    return bool(equal)


def _pair_ids(pair):
    if isinstance(pair, Mapping):
        ids = (pair.get("left"), pair.get("right"))
    else:
        try:
            ids = tuple(pair)
        except TypeError as error:
            raise ValueError("each pair must identify two records") from error
    if len(ids) != 2 or ids[0] is None or ids[1] is None or ids[0] == ids[1]:
        raise ValueError("each pair must identify two distinct episodes")
    return ids


def _pair_consistency(records, pairs, require_coverage=False):
    by_id = {}
    for row in records:
        episode_id = row.get("episode_id")
        if episode_id is None:
            raise ValueError("each record must declare episode_id")
        if episode_id in by_id:
            raise ValueError("episode_id values must be unique")
        by_id[episode_id] = row

    checks = []
    paired_ids = set()
    for pair in pairs:
        first_id, second_id = _pair_ids(pair)
        if first_id not in by_id or second_id not in by_id:
            raise ValueError("each pair must identify existing records")
        if first_id in paired_ids or second_id in paired_ids:
            raise ValueError("episode_id values may appear in only one pair")
        paired_ids.update((first_id, second_id))
        first, second = by_id[first_id], by_id[second_id]
        if _scenario(first) == _scenario(second):
            raise ValueError("each pair must contain one T_LEFT and one T_RIGHT episode")
        if _role(first) != _role(second):
            raise ValueError("paired episodes must have the same policy_role")
        for field in _PAIR_METADATA:
            if field not in first or field not in second:
                raise ValueError("paired episodes must declare %s" % field)
            if not _same_value(first[field], second[field]):
                raise ValueError("paired episodes must share %s" % field)
        for row in (first, second):
            expected = _EXPECTED_BRANCH[_scenario(row)]
            if str(row.get("expected_branch", "")).upper() != expected:
                raise ValueError("expected_branch must match the T-junction side")
            if "branch_prediction" not in row:
                raise ValueError("paired episodes must declare branch_prediction")
        first_prediction = str(first["branch_prediction"]).upper()
        second_prediction = str(second["branch_prediction"]).upper()
        checks.append(
            first_prediction == _EXPECTED_BRANCH[_scenario(first)]
            and second_prediction == _EXPECTED_BRANCH[_scenario(second)]
            and first_prediction != second_prediction
        )
    if require_coverage and paired_ids != set(by_id):
        raise ValueError("student pairs must cover every record exactly once")
    return sum(checks) / len(checks) if checks else 0.0, len(checks)


def _metrics(records):
    return {
        "episodes": len(records),
        "success_rate": _rate(records, "success"),
        "collision_rate": _rate(records, "collision"),
        "timeout_rate": _rate(records, "timeout"),
        "wrong_turn_rate": _rate(records, "wrong_turn"),
        "turn_completion_rate": _rate(records, "turn_completion"),
        "exit_rate": _rate(records, "exit"),
        "branch_accuracy": _branch_accuracy(records),
    }


def _side_checks(metrics, role):
    checks = {
        "success_rate_ge_0.95": metrics["success_rate"] >= 0.95,
        "collision_rate_eq_0": metrics["collision_rate"] == 0.0,
        "wrong_turn_rate_eq_0": metrics["wrong_turn_rate"] == 0.0,
    }
    if role == "student":
        checks.update(
            {
                "timeout_rate_le_0.05": metrics["timeout_rate"] <= 0.05,
                "turn_completion_rate_ge_0.95": metrics["turn_completion_rate"] >= 0.95,
            }
        )
    return checks


def aggregate_t_gate(records, pairs, ablations):
    """Aggregate strict per-side T-junction release evidence.

    Student runs require a paired left/right counterfactual for goal
    consistency.  Teacher runs retain the shared zero-collision and
    zero-wrong-turn requirements without applying student-only timeout and
    turn-completion requirements.
    """
    records = list(records)
    pairs = list(pairs)
    if not records:
        raise ValueError("at least one T-junction record is required")
    _assert_finite(records, "records")
    _assert_finite(ablations, "ablations")

    grouped = {side: [] for side in _SCENARIOS}
    roles = set()
    backends = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("each T-junction record must be a mapping")
        side = _scenario(record)
        grouped[side].append(record)
        roles.add(_role(record))
        if record.get("depth_backend_actual") != "isaacgym":
            raise ValueError("depth_backend_actual must be isaacgym for every record")
        backends.add(record["depth_backend_actual"])
    if len(roles) != 1:
        raise ValueError("a release gate must contain one policy_role")
    if any(not grouped[side] for side in _SCENARIOS):
        raise ValueError("a release gate requires both T_LEFT and T_RIGHT records")

    role = roles.pop()
    goal_consistency_rate, pair_count = _pair_consistency(
        records, pairs, require_coverage=(role == "student")
    )
    if role == "student" and not pair_count:
        raise ValueError("student release gate requires left/right counterfactual pairs")

    by_scenario = {side: _metrics(grouped[side]) for side in _SCENARIOS}
    aggregate = _metrics(records)
    checks_by_scenario = {
        side: _side_checks(by_scenario[side], role) for side in _SCENARIOS
    }
    checks = {
        "real_depth_backend": backends == {"isaacgym"},
        "by_scenario": checks_by_scenario,
    }
    if role == "student":
        checks["goal_consistency_rate_ge_0.95"] = goal_consistency_rate >= 0.95

    passed = checks["real_depth_backend"] and all(
        all(side_checks.values()) for side_checks in checks_by_scenario.values()
    )
    if role == "student":
        passed = passed and checks["goal_consistency_rate_ge_0.95"]

    values = {
        **aggregate,
        "policy_role": role,
        "by_scenario": by_scenario,
        "goal_consistency_rate": goal_consistency_rate,
        "pair_count": pair_count,
        "ablations": dict(ablations),
        "depth_backend_actual": sorted(backends),
        "checks": checks,
        "finite": True,
        "pass": bool(passed),
    }
    _assert_finite(values, "gate")
    return values
