"""Aggregate independent Phase-D map results without recreating IsaacGym."""

import argparse
import csv
import json
from pathlib import Path


STAGE_RULES = {
    "d0": {"maps": 10, "success": 10, "collision_max": 0},
    "d1.1-smoke": {"maps": 5, "success": 5, "collision_max": 0},
    "d1.1": {"maps": 20, "success": 19, "collision_max": 1},
    "d1.2-smoke": {"maps": 10, "success": 9, "collision_max": 1},
    "d1.2": {"maps": 40, "success": 38, "collision_max": 2},
}


def aggregate_phase_d_gate(root, stage="d0"):
    root = Path(root).resolve()
    stage = str(stage).lower()
    if stage not in STAGE_RULES:
        raise ValueError("unknown Phase-D stage: %s" % stage)
    rule = STAGE_RULES[stage]
    summaries = []
    for path in sorted(root.glob("map_*/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    success_count = sum(bool(item.get("success", False)) for item in summaries)
    collision_count = sum(bool(item.get("collision", False)) for item in summaries)
    process_failure_count = sum(item.get("failure_reason") == "PROCESS_FAILURE" for item in summaries)
    actual_backends = sorted({item.get("depth_backend_actual") for item in summaries})
    requested_backends = sorted({item.get("depth_backend_requested") for item in summaries})
    finite_ratios = [
        float(row.get("depth_finite_ratio", 0.0))
        for path in sorted(root.glob("map_*/trajectory.csv"))
        for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
    ]
    checks = {
        "map_count": len(summaries) == rule["maps"],
        "success_count": success_count >= rule["success"],
        "collision_count": collision_count <= rule["collision_max"],
        "process_failure_count": process_failure_count == 0,
        "requested_backend_isaacgym": requested_backends == ["isaacgym"],
        "actual_backend_isaacgym": actual_backends == ["isaacgym"],
        "depth_finite": bool(finite_ratios) and min(finite_ratios) >= 1.0,
    }
    result = {
        "stage": stage,
        "required": rule,
        "episodes": summaries,
        "success_count": success_count,
        "collision_count": collision_count,
        "process_failure_count": process_failure_count,
        "depth_backend_requested": requested_backends,
        "depth_backend_actual": actual_backends,
        "depth_finite_min_ratio": min(finite_ratios) if finite_ratios else None,
        "checks": checks,
        "pass": all(checks.values()),
        "verdict": "PASS" if all(checks.values()) else stage.upper() + "_FAIL",
    }
    (root / ("%s_summary.json" % stage)).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    markdown = [
        "# Phase-D %s Gate" % stage,
        "",
        "`%s`" % result["verdict"],
        "",
        "```json",
        json.dumps({"checks": checks, "counts": {"maps": len(summaries), "success": success_count, "collision": collision_count, "process_failure": process_failure_count}}, indent=2, sort_keys=True),
        "```",
    ]
    (root / ("%s_gate.md" % stage)).write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--stage", default="d0", choices=tuple(STAGE_RULES))
    args = parser.parse_args(argv)
    print(json.dumps(aggregate_phase_d_gate(args.root, args.stage), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
