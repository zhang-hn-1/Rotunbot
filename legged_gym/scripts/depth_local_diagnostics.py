"""Pure metric helpers shared by the depth-local root-cause diagnostics."""

from collections import defaultdict
import math


def _sign(value, tolerance=1e-8):
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _pearson(xs, ys):
    if len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    return numerator / denominator if denominator > 0.0 else 0.0


def _ranks(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original_index, _ in ordered[index:end]:
            result[original_index] = rank
        index = end
    return result


def policy_gy_metrics(records, physical_sign=1):
    """Calculate lateral sensitivity, sign, correlation, and symmetry metrics."""
    records = list(records)
    if not records:
        raise ValueError("policy records must not be empty")
    gy = [float(record["gy"]) for record in records]
    a0 = [float(record["actor_mean_a0"]) for record in records]
    a1 = [float(record["actor_mean_a1"]) for record in records]
    lateral = [
        (g, action)
        for g, action in zip(gy, a1)
        if abs(g) >= 0.2
    ]
    sign_matches = [
        _sign(action) == physical_sign * _sign(g)
        for g, action in lateral
        if _sign(action) != 0
    ]
    by_key = {(round(float(r["gx"]), 8), round(float(r["gy"]), 8)): r for r in records}
    symmetry_errors = []
    for (gx, gy_value), record in by_key.items():
        if gy_value <= 0.0:
            continue
        opposite = by_key.get((gx, round(-gy_value, 8)))
        if opposite is not None:
            symmetry_errors.append(abs(
                float(record["actor_mean_a1"]) + float(opposite["actor_mean_a1"])
            ))
    return {
        "records": len(records),
        "a1_response_span": max(a1) - min(a1),
        "a0_response_span": max(a0) - min(a0),
        "sign_agreement_rate": sum(sign_matches) / len(sign_matches) if sign_matches else 0.0,
        "pearson_gy_a1": _pearson(gy, a1),
        "spearman_gy_a1": _pearson(_ranks(gy), _ranks(a1)),
        "symmetry_error": sum(symmetry_errors) / len(symmetry_errors) if symmetry_errors else 0.0,
        "lateral_probe_count": len(lateral),
    }


def action_mapping_decision(cases, braking):
    """Classify A as GOOD, WEAK, or FAIL from the prescribed action sweep."""
    endpoint = [
        case for case in cases
        if abs(float(case["action1"])) >= 0.75
        and abs(float(case["duration_s"]) - 5.0) < 1e-6
    ]
    positive = [case for case in endpoint if float(case["action1"]) > 0.0]
    negative = [case for case in endpoint if float(case["action1"]) < 0.0]
    if not positive or not negative:
        return "A-FAIL"
    positive_dy = sum(float(case["delta_body_y"]) for case in positive) / len(positive)
    negative_dy = sum(float(case["delta_body_y"]) for case in negative) / len(negative)
    magnitude = (abs(positive_dy) + abs(negative_dy)) / 2.0
    same_direction = positive_dy > 0.0 and negative_dy < 0.0
    if magnitude < 0.2 or not same_direction:
        return "A-FAIL"
    symmetric = min(abs(positive_dy), abs(negative_dy)) / max(abs(positive_dy), abs(negative_dy)) >= 0.5
    if magnitude >= 0.4 and magnitude <= 0.6 and symmetric:
        if braking and not all(bool(case.get("stopped", False)) for case in braking):
            return "A-WEAK"
        return "A-GOOD"
    return "A-WEAK"


def group_policy_metrics(records, physical_sign=1):
    """Return the same policy metrics independently for each gx probe."""
    grouped = defaultdict(list)
    for record in records:
        grouped[round(float(record["gx"]), 8)].append(record)
    return {
        str(gx): policy_gy_metrics(group, physical_sign=physical_sign)
        for gx, group in sorted(grouped.items())
    }
