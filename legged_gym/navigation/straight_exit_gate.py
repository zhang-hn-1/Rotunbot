"""Reverse-command diagnostics and the bounded Straight exit sanity gate."""

import math


def _nearest_rank(values, percentile):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    rank = max(1, int(math.ceil(float(percentile) * len(values))))
    return values[min(rank, len(values)) - 1]


def summarize_reverse_diagnostics(
    trajectory_rows,
    initial_goal_distance_by_episode,
    collision_episodes=(),
    timeout_episodes=(),
    dt=0.2,
    reverse_threshold=-1.0e-5,
):
    """Summarize one command sample per macro step, grouped by episode."""
    grouped = {}
    for row in trajectory_rows:
        episode_id = int(row["episode_id"])
        if "macro_step" in row:
            macro_step = int(row["macro_step"])
        else:
            macro_step = (int(row.get("step", 1)) - 1) // 10
        grouped.setdefault(episode_id, {})[macro_step] = row
    samples = []
    for episode_id, rows in grouped.items():
        previous_distance = None
        for macro_step in sorted(rows):
            row = rows[macro_step]
            # ``raw_v_cmd`` is the policy command before V62 projection.  It
            # preserves the sign the reverse diagnostic is intended to audit;
            # ``v_cmd`` is the post-projection command and may hide it.
            v_cmd = float(row.get("raw_v_cmd", row.get("v_cmd", row.get("applied_v_cmd", 0.0))))
            goal_distance = float(row.get("goal_distance", math.nan))
            reverse = v_cmd < float(reverse_threshold)
            samples.append(
                {
                    "episode_id": episode_id,
                    "macro_step": macro_step,
                    "v_cmd": v_cmd,
                    "goal_distance": goal_distance,
                    "negative_progress": (
                        previous_distance is not None
                        and math.isfinite(goal_distance)
                        and goal_distance > previous_distance + 1.0e-6
                    ),
                    "reverse": reverse,
                }
            )
            if math.isfinite(goal_distance):
                previous_distance = goal_distance
    reverse = [item for item in samples if item["reverse"]]
    reverse_values = [item["v_cmd"] for item in reverse]
    episode_ids = set(item["episode_id"] for item in samples)
    reverse_episode_ids = set(item["episode_id"] for item in reverse)
    runs = []
    current_episode = None
    current_run = 0
    for item in samples:
        if item["episode_id"] != current_episode:
            if current_run:
                runs.append(current_run)
            current_episode = item["episode_id"]
            current_run = 0
        if item["reverse"]:
            current_run += 1
        elif current_run:
            runs.append(current_run)
            current_run = 0
    if current_run:
        runs.append(current_run)
    initial = initial_goal_distance_by_episode
    near_start = near_mid = near_goal = 0
    for item in reverse:
        distance = item["goal_distance"]
        initial_distance = float(initial.get(item["episode_id"], distance))
        if not math.isfinite(distance) or initial_distance <= 0.0:
            near_mid += 1
        elif distance >= 0.8 * initial_distance:
            near_start += 1
        elif distance <= 0.75:
            near_goal += 1
        else:
            near_mid += 1
    reverse_count = len(reverse)
    total_count = len(samples)
    negative_p50 = _nearest_rank(reverse_values, 0.50)
    negative_p95 = _nearest_rank(reverse_values, 0.95)
    max_run = max(runs, default=0)
    mean_run = sum(runs) / max(len(runs), 1)
    negative_progress_count = sum(item["negative_progress"] for item in reverse)
    high_speed = (
        reverse_count > 0
        and abs(negative_p50) >= 0.03
        and max_run >= 3
        and negative_progress_count / max(reverse_count, 1) >= 0.5
    )
    collision_episodes = set(collision_episodes)
    timeout_episodes = set(timeout_episodes)
    return {
        "reverse_step_count": reverse_count,
        "reverse_step_ratio": reverse_count / max(total_count, 1),
        "reverse_episode_count": len(reverse_episode_ids),
        "reverse_episode_ratio": len(reverse_episode_ids) / max(len(episode_ids), 1),
        "negative_v_mean": sum(reverse_values) / max(reverse_count, 1),
        "negative_v_p50": negative_p50,
        "negative_v_p95": negative_p95,
        "negative_v_min": min(reverse_values, default=0.0),
        "max_consecutive_reverse_steps": max_run,
        "mean_reverse_run_length": mean_run,
        "max_reverse_duration_sec": max_run * float(dt),
        "reverse_near_start_ratio": near_start / max(reverse_count, 1),
        "reverse_mid_episode_ratio": near_mid / max(reverse_count, 1),
        "reverse_near_goal_ratio": near_goal / max(reverse_count, 1),
        "reverse_with_negative_progress_ratio": negative_progress_count / max(reverse_count, 1),
        "reverse_with_collision_count": len(reverse_episode_ids & collision_episodes),
        "reverse_with_timeout_count": len(reverse_episode_ids & timeout_episodes),
        "sustained_high_speed_reverse": bool(high_speed),
    }


def build_straight_exit_gate(per_distance, commit, checkpoint=None):
    """Apply the 80% overall sanity gate without reintroducing V1 curriculum."""
    summaries = list(per_distance.values())
    episodes = sum(int(item["episodes"]) for item in summaries)
    successes = sum(int(item.get("success_count", 0)) for item in summaries)
    collisions = sum(int(item.get("collision_count", 0)) for item in summaries)
    timeouts = sum(int(item.get("timeout_count", 0)) for item in summaries)
    systematic_reverse = any(
        bool(item.get("reverse_diagnostics", {}).get("sustained_high_speed_reverse", False))
        for item in summaries
    )
    backends = {
        item.get("depth_backend_actual", item.get("depth_backend"))
        for item in summaries
    }
    overall_success_rate = successes / max(episodes, 1)
    collision_rate = collisions / max(episodes, 1)
    timeout_rate = timeouts / max(episodes, 1)
    checks = {
        "overall_success_rate_ge_0.80": overall_success_rate >= 0.80,
        "collision_rate_not_systematic": collision_rate <= 0.20,
        "timeout_rate_not_systematic": timeout_rate <= 0.20,
        "no_sustained_high_speed_reverse": not systematic_reverse,
        "real_depth_backend": backends == {"isaacgym"},
    }
    reverse_diagnostics = aggregate_reverse_diagnostics(
        [item.get("reverse_diagnostics", {}) for item in summaries]
    )
    return {
        "stage": "STRAIGHT_EXIT",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "overall_success_rate": overall_success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "per_distance": per_distance,
        "checks": checks,
        "reverse_diagnostics": reverse_diagnostics,
        "checkpoint": checkpoint,
        "commit": str(commit),
    }


def aggregate_reverse_diagnostics(summaries):
    """Combine per-distance reverse diagnostics for the exit artifact."""
    summaries = list(summaries)
    reverse_count = sum(int(item.get("reverse_step_count", 0)) for item in summaries)
    reverse_episodes = sum(int(item.get("reverse_episode_count", 0)) for item in summaries)
    total_steps = sum(
        int(round(item.get("reverse_step_count", 0) / item.get("reverse_step_ratio", 1.0)))
        if item.get("reverse_step_ratio", 0.0) > 0.0 else 0
        for item in summaries
    )
    total_episodes = sum(
        int(round(item.get("reverse_episode_count", 0) / item.get("reverse_episode_ratio", 1.0)))
        if item.get("reverse_episode_ratio", 0.0) > 0.0 else 0
        for item in summaries
    )
    weighted_mean = sum(
        float(item.get("negative_v_mean", 0.0)) * int(item.get("reverse_step_count", 0))
        for item in summaries
    ) / max(reverse_count, 1)
    return {
        "reverse_step_count": reverse_count,
        "reverse_step_ratio": reverse_count / max(total_steps, 1),
        "reverse_episode_count": reverse_episodes,
        "reverse_episode_ratio": reverse_episodes / max(total_episodes, 1),
        "negative_v_mean": weighted_mean,
        "negative_v_p50": min((float(item.get("negative_v_p50", 0.0)) for item in summaries), default=0.0),
        "negative_v_p95": max((float(item.get("negative_v_p95", 0.0)) for item in summaries), default=0.0),
        "negative_v_min": min((float(item.get("negative_v_min", 0.0)) for item in summaries), default=0.0),
        "max_consecutive_reverse_steps": max((int(item.get("max_consecutive_reverse_steps", 0)) for item in summaries), default=0),
        "mean_reverse_run_length": sum(float(item.get("mean_reverse_run_length", 0.0)) for item in summaries) / max(len(summaries), 1),
        "max_reverse_duration_sec": max((float(item.get("max_reverse_duration_sec", 0.0)) for item in summaries), default=0.0),
        "reverse_near_start_ratio": sum(float(item.get("reverse_near_start_ratio", 0.0)) * int(item.get("reverse_step_count", 0)) for item in summaries) / max(reverse_count, 1),
        "reverse_mid_episode_ratio": sum(float(item.get("reverse_mid_episode_ratio", 0.0)) * int(item.get("reverse_step_count", 0)) for item in summaries) / max(reverse_count, 1),
        "reverse_near_goal_ratio": sum(float(item.get("reverse_near_goal_ratio", 0.0)) * int(item.get("reverse_step_count", 0)) for item in summaries) / max(reverse_count, 1),
        "reverse_with_negative_progress_ratio": sum(float(item.get("reverse_with_negative_progress_ratio", 0.0)) * int(item.get("reverse_step_count", 0)) for item in summaries) / max(reverse_count, 1),
        "reverse_with_collision_count": sum(int(item.get("reverse_with_collision_count", 0)) for item in summaries),
        "reverse_with_timeout_count": sum(int(item.get("reverse_with_timeout_count", 0)) for item in summaries),
        "sustained_high_speed_reverse": any(bool(item.get("sustained_high_speed_reverse", False)) for item in summaries),
    }
