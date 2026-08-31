"""Auditable episode specifications and gates for V1 evaluation."""

import math
import random
from statistics import median


def build_fixed_distance_specs(distance, episodes, seed):
    """Build deterministic straight-corridor goals at one exact distance."""
    distance = float(distance)
    episodes = int(episodes)
    seed = int(seed)
    if distance <= 0.0:
        raise ValueError("distance must be positive")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    # Instantiate the RNG even though the straight-corridor bearing is fixed.
    # This makes the independent seed part of the spec and leaves a stable
    # extension point for bounded start-pose sampling.
    rng = random.Random(seed)
    specs = []
    for episode_id in range(episodes):
        episode_seed = rng.randrange(0, 2**31)
        specs.append(
            {
                "episode_id": episode_id,
                "seed": episode_seed,
                "distance_m": distance,
                "bearing_rad": 0.0,
                "evaluation_distance_m": distance,
            }
        )
    return specs


def _mean(records, key):
    values = [float(record[key]) for record in records if key in record]
    return sum(values) / len(values) if values else None


def summarize_v1_episodes(records):
    """Aggregate episode-level records without claiming an undefined SPL."""
    records = list(records)
    if not records:
        raise ValueError("at least one episode record is required")
    count = float(len(records))
    successes = sum(bool(record.get("success", False)) for record in records)
    collisions = sum(bool(record.get("collision", False)) for record in records)
    timeouts = sum(bool(record.get("timeout", False)) for record in records)
    initial_distances = [
        float(record["initial_goal_distance_m"]) for record in records
    ]
    final_distances = [
        float(record["terminal_goal_distance_m"]) for record in records
    ]
    path_lengths = [float(record.get("path_length_m", 0.0)) for record in records]
    episode_lengths = [
        float(record.get("episode_length", record.get("steps", 0.0)))
        for record in records
    ]
    total_steps = sum(float(record.get("steps", 0.0)) for record in records)
    total_reverse_steps = sum(float(record.get("reverse_steps", 0.0)) for record in records)
    positive_path_efficiencies = [
        initial / path
        for initial, path in zip(initial_distances, path_lengths)
        if path > 0.0
    ]
    # This follows the repository's existing play.py definition: shortest
    # path divided by max(actual path, shortest path), and zero for failures.
    spl_values = [
        (initial / max(path, initial)) if bool(record.get("success", False)) else 0.0
        for record, initial, path in zip(records, initial_distances, path_lengths)
    ]
    summary = {
        "episodes": len(records),
        "success_count": successes,
        "success_rate": successes / count,
        "collision_count": collisions,
        "collision_rate": collisions / count,
        "timeout_count": timeouts,
        "timeout_rate": timeouts / count,
        "mean_final_goal_distance_m": sum(final_distances) / count,
        "median_final_goal_distance_m": median(final_distances),
        "mean_episode_length": sum(episode_lengths) / count,
        "mean_path_length_m": sum(path_lengths) / count,
        "mean_initial_goal_distance_m": sum(initial_distances) / count,
        "reverse_motion_ratio": (
            total_reverse_steps / total_steps if total_steps > 0.0 else 0.0
        ),
        "path_efficiency": (
            sum(positive_path_efficiencies) / len(positive_path_efficiencies)
            if positive_path_efficiencies
            else None
        ),
        "spl": sum(spl_values) / count,
        "mean_forward_velocity": _mean(records, "mean_forward_velocity"),
        "mean_absolute_yaw_velocity": _mean(records, "mean_absolute_yaw_velocity"),
    }
    for key in (
        "rate_violation_count",
        "feasible_domain_violation_count",
        "hidden_projection_jump_count",
        "raw_reverse_command_count",
        "requested_reverse_command_count",
        "applied_reverse_command_count",
    ):
        summary[key] = sum(int(record.get(key, 0)) for record in records)
    summary["mean_command_correction"] = _mean(records, "mean_command_correction")
    numeric_values = [value for value in summary.values() if isinstance(value, (int, float))]
    summary["finite"] = all(math.isfinite(float(value)) for value in numeric_values)
    return summary


def curriculum_gate(current_summary, next_summary):
    """Apply the proposed current/next-distance promotion gate."""
    checks = {
        "current_success_rate": float(current_summary["success_rate"]) >= 0.90,
        "next_success_rate": float(next_summary["success_rate"]) >= 0.80,
        "current_collision_rate": float(current_summary["collision_rate"]) <= 0.10,
        "next_collision_rate": float(next_summary["collision_rate"]) <= 0.10,
        "current_finite": bool(current_summary.get("finite", True)),
        "next_finite": bool(next_summary.get("finite", True)),
    }
    return {"pass": all(checks.values()), "checks": checks}
