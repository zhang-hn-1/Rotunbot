"""Build auditable D0-A/B/C comparisons from independent run artifacts."""

import argparse
import csv
import json
from pathlib import Path


SEEDS = tuple(range(20000, 20010))


def _load(root, seed):
    path = Path(root) / ("map_%d" % seed) / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _aggregate(root):
    rows = [_load(root, seed) for seed in SEEDS]
    rows = [row for row in rows if row is not None]
    return {
        "maps": len(rows),
        "success": sum(bool(row.get("success")) for row in rows),
        "collision": sum(bool(row.get("collision")) for row in rows),
        "process_failure": sum(row.get("failure_reason") == "PROCESS_FAILURE" for row in rows),
        "timeout": sum(bool(row.get("timeout")) for row in rows),
        "mean_final_distance_m": sum(float(row.get("final_goal_distance_m", 0.0)) for row in rows) / max(len(rows), 1),
        "rows": rows,
    }


def _gate(name, aggregate, required_success=10):
    checks = {
        "map_count": aggregate["maps"] == 10,
        "success_count": aggregate["success"] >= required_success,
        "collision_count": aggregate["collision"] == 0,
        "process_failure_count": aggregate["process_failure"] == 0,
        "real_depth_all_maps": all(
            row.get("depth_backend_requested") == "isaacgym"
            and row.get("depth_backend_actual") == "isaacgym"
            for row in aggregate["rows"]
        ),
    }
    return {
        "name": name,
        "checks": checks,
        "pass": all(checks.values()),
        "success": aggregate["success"],
        "collision": aggregate["collision"],
        "timeout": aggregate["timeout"],
        "process_failure": aggregate["process_failure"],
    }


def report(output_root, constant_root, direct_root, current_root, sweep_roots):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    a = json.loads((Path(constant_root).resolve() / "v62_constant_command_audit.json").read_text())
    a_summaries = [case.get("summary", {}) for case in a.get("cases", []) if case.get("summary")]
    a_pass = a.get("D0_A") == "PASS" and len(a_summaries) == 18
    gates = {
        "D0_A": {"name": "D0-A Constant Frozen V62", "pass": a_pass, "case_count": len(a_summaries), "cases": a_summaries},
        "D0_B": _gate("D0-B Direct Global Goal", _aggregate(direct_root)),
        "D0_C": _gate("D0-C Oracle Path Following 0.8m", _aggregate(current_root)),
    }
    for name, gate in gates.items():
        text = ["# %s" % gate["name"], "", "`%s = %s`" % (name, "PASS" if gate["pass"] else "FAIL"), "", "```json", json.dumps(gate, indent=2, sort_keys=True), "```", ""]
        (output_root / (name.lower().replace("_", "_") + "_gate.md")).write_text("\n".join(text), encoding="utf-8")
    comparison = []
    for seed in SEEDS:
        direct = _load(direct_root, seed) or {}
        current = _load(current_root, seed) or {}
        row = {
            "seed": seed,
            "direct_goal_success": bool(direct.get("success", False)),
            "lookahead_0.8_success": bool(current.get("success", False)),
            "direct_final_distance_m": direct.get("final_goal_distance_m"),
            "lookahead_0.8_final_distance_m": current.get("final_goal_distance_m"),
            "direct_mean_teacher_v_mps": direct.get("command_loss_breakdown", {}).get("mean_teacher_raw_v_mps"),
            "direct_mean_projected_v_mps": direct.get("command_loss_breakdown", {}).get("mean_projected_v_mps"),
            "direct_mean_applied_v_mps": direct.get("command_loss_breakdown", {}).get("mean_applied_v_mps"),
            "direct_mean_actual_v_mps": direct.get("command_loss_breakdown", {}).get("mean_actual_v_mps"),
            "lookahead_0.8_mean_teacher_v_mps": current.get("command_loss_breakdown", {}).get("mean_teacher_raw_v_mps"),
            "lookahead_0.8_mean_projected_v_mps": current.get("command_loss_breakdown", {}).get("mean_projected_v_mps"),
            "lookahead_0.8_mean_applied_v_mps": current.get("command_loss_breakdown", {}).get("mean_applied_v_mps"),
            "lookahead_0.8_mean_actual_v_mps": current.get("command_loss_breakdown", {}).get("mean_actual_v_mps"),
            "failure_direct": direct.get("failure_reason"),
            "failure_lookahead_0.8": current.get("failure_reason"),
        }
        comparison.append(row)
    with (output_root / "d0_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    result = {
        "gates": {name: {key: value for key, value in gate.items() if key != "cases"} for name, gate in gates.items()},
        "sweep": {},
        "comparison_csv": str(output_root / "d0_comparison.csv"),
    }
    for label, root in sweep_roots.items():
        result["sweep"][label] = _aggregate(root)
    (output_root / "d0_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--constant-root", required=True)
    parser.add_argument("--direct-root", required=True)
    parser.add_argument("--current-root", required=True)
    parser.add_argument("--sweep", action="append", default=[], metavar="LABEL=ROOT")
    args = parser.parse_args(argv)
    sweep = dict(item.split("=", 1) for item in args.sweep)
    print(json.dumps(report(args.output_root, args.constant_root, args.direct_root, args.current_root, sweep), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
