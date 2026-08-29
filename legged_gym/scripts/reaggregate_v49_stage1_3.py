"""Rebuild Stage1.3 aggregate/table CSVs without rerunning Isaac Gym.

This is useful after changing an offline aggregation rule: the raw traces and
per-transition summaries remain the source of truth, while repeated trials are
grouped by the requested experimental initial state and projected command.
"""

import argparse
import csv
import json
import os

from legged_gym.navigation.v49_stage1_3_diagnostics import (
    STAGE13_HORIZONS_MS,
    aggregate_stage13_trials,
)


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path, rows):
    fields = tuple(sorted(set().union(*(set(row) for row in rows)))) if rows else ()
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage13_output_dir", default="logs/stage1_3_dynamic_reachability")
    args = parser.parse_args()
    output_dir = os.path.abspath(args.stage13_output_dir)
    trials = [
        row for row in _read(os.path.join(output_dir, "dynamic_response_trials.csv"))
        if row.get("status") == "complete"
    ]
    grouped = {}
    for row in trials:
        key = (
            float(row["requested_initial_forward_velocity"]),
            float(row["requested_initial_yaw_rate"]),
            float(row["projected_forward_velocity"]),
            float(row["projected_yaw_rate"]),
        )
        grouped.setdefault(key, []).append(row)
    aggregate_rows = []
    table_rows = []
    for key, samples in sorted(grouped.items()):
        aggregate = aggregate_stage13_trials(samples)
        aggregate.update({
            "requested_initial_forward_velocity": key[0],
            "requested_initial_yaw_rate": key[1],
            "projected_forward_velocity": key[2],
            "projected_yaw_rate": key[3],
        })
        aggregate_rows.append(aggregate)
        table = {
            "current_v": key[0],
            "projected_v": key[2],
            "projected_w": key[3],
            "repeat_count": len(samples),
        }
        for horizon in STAGE13_HORIZONS_MS:
            table["mean_actual_v_%dms" % horizon] = aggregate["mean_actual_v_%dms" % horizon]
            table["mean_actual_w_%dms" % horizon] = aggregate["mean_actual_w_%dms" % horizon]
        table_rows.append(table)
    _write(os.path.join(output_dir, "dynamic_response_aggregated.csv"), aggregate_rows)
    _write(os.path.join(output_dir, "dynamic_reachability_table.csv"), table_rows)
    summary_path = os.path.join(output_dir, "stage1_3_summary.json")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    summary["table_row_count"] = len(table_rows)
    summary["aggregation_key"] = [
        "requested_initial_forward_velocity", "requested_initial_yaw_rate",
        "projected_forward_velocity", "projected_yaw_rate",
    ]
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"completed_trials": len(trials), "table_rows": len(table_rows)}, indent=2))


if __name__ == "__main__":
    main()
