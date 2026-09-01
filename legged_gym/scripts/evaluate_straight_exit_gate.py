"""Run the small real-depth Straight exit sanity gate."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

from legged_gym.scripts.eval_sru_visual_corridor_v1 import evaluate_distance
from legged_gym.navigation.straight_exit_gate import (
    build_straight_exit_gate,
    summarize_reverse_diagnostics,
)


def _commit_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def _read_existing_summary(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    records = []
    with (root / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(row)
    trajectories = []
    with (root / "trajectory.csv").open(newline="", encoding="utf-8") as handle:
        trajectories.extend(csv.DictReader(handle))
    reverse = summarize_reverse_diagnostics(
        trajectories,
        {int(row["episode_id"]): float(row["initial_goal_distance_m"]) for row in records},
        collision_episodes={int(row["episode_id"]) for row in records if row["collision"] == "True"},
        timeout_episodes={int(row["episode_id"]) for row in records if row["timeout"] == "True"},
        dt=0.2,
    )
    summary["depth_backend_actual"] = summary.get("depth_backend_actual", summary.get("depth_backend"))
    summary["reverse_diagnostics"] = reverse
    return summary


def _read_distance_artifact(root):
    """Reload an existing evaluator artifact and refresh reverse diagnostics."""
    return _read_existing_summary(root)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--existing-1m-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    args, framework_args = parser.parse_known_args(argv)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    per_distance = {"1.0": _read_existing_summary(args.existing_1m_dir)}
    for distance in (1.5, 2.0, 2.5):
        distance_root = root / ("distance_%0.1fm" % distance)
        if args.reuse_existing and (distance_root / "summary.json").is_file():
            per_distance[str(distance)] = _read_distance_artifact(distance_root)
        else:
            per_distance[str(distance)] = evaluate_distance(
                args.checkpoint,
                distance,
                args.episodes,
                args.seed,
                distance_root,
                num_envs=args.num_envs,
                max_steps=args.max_steps,
                depth_backend="isaacgym",
                framework_args=framework_args,
            )
    gate = build_straight_exit_gate(
        per_distance,
        _commit_sha(),
        checkpoint=str(Path(args.checkpoint).resolve()),
    )
    gate["date"] = "2026-09-01"
    (root / "straight_exit_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, sort_keys=True))
    return gate


if __name__ == "__main__":
    main()
