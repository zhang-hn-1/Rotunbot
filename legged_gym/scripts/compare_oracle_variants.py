"""Run and compare the four Raw Oracle variants on one episode manifest."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


VARIANTS = (
    ("A", "raw_v1", ()),
    ("B", "raw_v1_reachability", ("--reachability-envelope",)),
    ("C", "raw_v1_turn_aware", ("--turn-aware",)),
    ("D", "raw_v1_reachability_turn_aware", ("--reachability-envelope", "--turn-aware")),
)


def build_comparison_table(summaries):
    fields = (
        "global_sr",
        "collision_rate",
        "timeout_rate",
        "waypoint_failure_rate",
        "unstable_rate",
        "out_of_bounds_rate",
        "waypoint_count",
        "local_waypoint_reach_rate",
        "completion_time_s",
        "actual_path_length_m",
        "bfs_shortest_path_length_m",
        "maze_spl",
        "final_approach_entry_count",
        "final_approach_success_count",
        "final_approach_timeout_count",
        "final_approach_escape_count",
        "planner_error_count",
        "goal_switch_error_count",
    )
    table = []
    for label, _name, _flags in VARIANTS:
        summary = summaries[label]
        reason_counts = summary.get("failure_reason_counts", {})
        row = {"variant": label}
        for field in fields:
            if field in summary:
                row[field] = summary[field]
            elif field.endswith("_rate"):
                reason = field[:-5]
                denominator = max(int(summary.get("episodes", 1)), 1)
                row[field] = float(reason_counts.get(reason, 0) / denominator)
            else:
                row[field] = 0.0
        table.append(row)
    return table


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reachability-envelope", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reuse-a-summary")
    return parser.parse_args()


def run(args):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for label, name, flag_template in VARIANTS:
        if label == "A" and args.reuse_a_summary:
            summaries[label] = json.loads(
                Path(args.reuse_a_summary).read_text(encoding="utf-8")
            )
            continue
        flags = []
        if "--reachability-envelope" in flag_template:
            flags.extend(["--reachability-envelope", args.reachability_envelope])
        if "--turn-aware" in flag_template:
            flags.append("--turn-aware")
        output_dir = output_root / name
        command = [
            sys.executable,
            str(Path(__file__).with_name("evaluate_oracle_maze.py")),
            "--episodes", "100",
            "--checkpoint", args.checkpoint,
            "--output-dir", str(output_dir),
            "--episode-manifest", args.manifest,
            *flags,
        ]
        with (output_dir.with_suffix(".log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        summaries[label] = json.loads(
            (output_dir / "summary.json").read_text(encoding="utf-8")
        )
    table = build_comparison_table(summaries)
    payload = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "variants": table,
    }
    (output_root / "comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    headers = ["variant"] + [key for key in table[0] if key != "variant"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in table:
        lines.append("| " + " | ".join(str(row[key]) for key in headers) + " |")
    (output_root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    run(_parse_args())
