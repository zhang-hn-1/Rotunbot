"""Pure-Python metrics for fixed Robot-frame Local P2P evaluation."""


def _mean(records, key):
    return sum(float(record[key]) for record in records) / len(records)


def _yaw_key(yaw_deg):
    value = float(yaw_deg)
    if abs(value - round(value)) < 1.0e-6:
        return str(int(round(value)))
    return f"{value:g}"


def _summarize(records):
    episodes = len(records)
    success = sum(bool(record["success"]) for record in records)
    timeout = sum(bool(record["timeout"]) for record in records)
    divergence = sum(bool(record["divergent"]) for record in records)
    near_miss = sum(bool(record["near_miss"]) for record in records)
    return {
        "episodes": episodes,
        "success": success,
        "timeout": timeout,
        "divergence": divergence,
        "near_miss": near_miss,
        "success_rate": success / episodes,
        "timeout_rate": timeout / episodes,
        "divergence_rate": divergence / episodes,
        "near_miss_rate": near_miss / episodes,
        "mean_min_distance": _mean(records, "min_distance"),
        "mean_final_distance": _mean(records, "final_distance"),
        "mean_steps": _mean(records, "steps"),
        "mean_clip_ratio": _mean(records, "clip_ratio"),
    }


def aggregate_local_goal_records(records):
    """Aggregate episode records and compute the world-yaw success gap."""
    records = list(records)
    if not records:
        raise ValueError("cannot aggregate an empty evaluation")
    summary = _summarize(records)
    grouped = {}
    for record in records:
        grouped.setdefault(_yaw_key(record["yaw_deg"]), []).append(record)
    yaw_groups = {key: _summarize(group) for key, group in sorted(grouped.items())}
    yaw_rates = [group["success_rate"] for group in yaw_groups.values()]
    summary["yaw_groups"] = yaw_groups
    summary["yaw_success_gap"] = max(yaw_rates) - min(yaw_rates)
    return summary
