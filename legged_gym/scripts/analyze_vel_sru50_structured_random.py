"""Aggregate and plot V62 structured-random velocity tracking results."""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILY_LABELS = {
    "straight_v_reversal": "straight v reversal",
    "fixed_w_v_reversal": "fixed w, v reversal",
    "constant_curvature_reversal": "constant-curvature reversal",
    "fixed_v_w_reversal": "fixed v, w reversal",
    "fixed_w_speed_change": "fixed w, speed change",
    "fixed_v_yaw_magnitude_change": "fixed v, yaw magnitude",
    "straight_stop_or_restart": "straight stop/restart",
    "turn_stop_or_restart": "turn stop/restart",
    "infeasible_low_speed_high_yaw": "infeasible low-v/high-w",
    "boundary_curvature_jump": "boundary curvature jump",
    "all_quadrant_jump": "all-quadrant jump",
    "independent_feasible_jump": "independent feasible jump",
}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def _find_runs(root):
    runs = []
    for entry in sorted(os.listdir(root)):
        run_dir = os.path.join(root, entry)
        summary_path = os.path.join(run_dir, "structured_random_summary.json")
        metrics_path = os.path.join(
            run_dir, "structured_transition_environment_metrics.csv"
        )
        if not os.path.isfile(summary_path) or not os.path.isfile(metrics_path):
            continue
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        with open(metrics_path, "r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        runs.append(
            {
                "name": entry,
                "directory": run_dir,
                "summary": summary,
                "rows": rows,
            }
        )
    if not runs:
        raise RuntimeError("No structured-random runs found under %s" % root)
    return runs


def _float(row, key):
    return float(row[key])


def _mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _p95(values):
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def _aggregate(runs):
    profiles = sorted({run["summary"]["noise_profile"] for run in runs})
    family_order = runs[0]["summary"]["families"]
    family_rows = []
    profile_overall = {}
    for profile in profiles:
        selected = [run for run in runs if run["summary"]["noise_profile"] == profile]
        all_rows = [row for run in selected for row in run["rows"]]
        profile_overall[profile] = {
            "runs": len(selected),
            "transitions": len(all_rows),
            "applied_v_mae_mps": _mean(
                [_float(row, "applied_v_mae_mps") for row in all_rows]
            ),
            "applied_w_mae_radps": _mean(
                [_float(row, "applied_w_mae_radps") for row in all_rows]
            ),
            "p95_environment_applied_v_mae_mps": _p95(
                [_float(row, "applied_v_mae_mps") for row in all_rows]
            ),
            "p95_environment_applied_w_mae_radps": _p95(
                [_float(row, "applied_w_mae_radps") for row in all_rows]
            ),
            "maximum_environment_applied_w_p95_radps": max(
                _float(row, "applied_w_p95_radps") for row in all_rows
            ),
            "mean_abs_requested_yaw_integral_gap_rad": _mean(
                [abs(_float(row, "requested_yaw_integral_gap_rad")) for row in all_rows]
            ),
            "minimum_applied_v_sign_correct_ratio": min(
                _float(row, "applied_v_sign_correct_ratio") for row in all_rows
            ),
            "minimum_applied_w_sign_correct_ratio": min(
                _float(row, "applied_w_sign_correct_ratio") for row in all_rows
            ),
        }
        for family in family_order:
            family_run_metrics = [
                run["summary"]["family_metrics"][family] for run in selected
            ]
            rows = [
                row
                for run in selected
                for row in run["rows"]
                if row["family"] == family
            ]
            aggregate = {
                "noise_profile": profile,
                "family": family,
                "transitions": len(rows),
                "applied_v_mae_mps": _mean(
                    [item["applied_tracking_v"]["mae"] for item in family_run_metrics]
                ),
                "applied_w_mae_radps": _mean(
                    [item["applied_tracking_w"]["mae"] for item in family_run_metrics]
                ),
                "applied_v_p95_mps": _mean(
                    [item["applied_tracking_v"]["p95"] for item in family_run_metrics]
                ),
                "applied_w_p95_radps": _mean(
                    [item["applied_tracking_w"]["p95"] for item in family_run_metrics]
                ),
                "applied_w_p99_radps": _mean(
                    [item["applied_tracking_w"]["p99"] for item in family_run_metrics]
                ),
                "applied_w_max_radps": max(
                    item["applied_tracking_w"]["max"] for item in family_run_metrics
                ),
                "settled_applied_v_mae_mps": _mean(
                    [
                        item["settled_applied_tracking_v"]["mae"]
                        for item in family_run_metrics
                    ]
                ),
                "settled_applied_w_mae_radps": _mean(
                    [
                        item["settled_applied_tracking_w"]["mae"]
                        for item in family_run_metrics
                    ]
                ),
                "request_projection_v_mae_mps": _mean(
                    [item["request_projection_v"]["mae"] for item in family_run_metrics]
                ),
                "request_projection_w_mae_radps": _mean(
                    [item["request_projection_w"]["mae"] for item in family_run_metrics]
                ),
                "projection_fraction": _mean(
                    [item["projection_fraction"] for item in family_run_metrics]
                ),
                "mean_abs_requested_yaw_integral_gap_rad": _mean(
                    [
                        item["mean_abs_requested_yaw_integral_gap_rad"]
                        for item in family_run_metrics
                    ]
                ),
                "applied_v_sign_correct_ratio": _mean(
                    [item["applied_v_sign_correct_ratio"] for item in family_run_metrics]
                ),
                "applied_w_sign_correct_ratio": _mean(
                    [item["applied_w_sign_correct_ratio"] for item in family_run_metrics]
                ),
                "minimum_environment_applied_v_sign_correct_ratio": min(
                    _float(row, "applied_v_sign_correct_ratio") for row in rows
                ),
                "minimum_environment_applied_w_sign_correct_ratio": min(
                    _float(row, "applied_w_sign_correct_ratio") for row in rows
                ),
                "maximum_environment_applied_w_p95_radps": max(
                    _float(row, "applied_w_p95_radps") for row in rows
                ),
            }
            aggregate["tracking_gate_pass"] = bool(
                aggregate["applied_v_mae_mps"] <= 0.015
                and aggregate["applied_w_mae_radps"] <= 0.008
                and aggregate["applied_v_p95_mps"] <= 0.030
                and aggregate["applied_w_p95_radps"] <= 0.015
                and aggregate["applied_v_sign_correct_ratio"] >= 0.98
                and aggregate["applied_w_sign_correct_ratio"] >= 0.95
            )
            family_rows.append(aggregate)
    return profiles, family_order, profile_overall, family_rows


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _bar_plot(output_dir, profiles, family_order, family_rows):
    x = np.arange(len(family_order))
    width = 0.36
    colors = {"nominal": "#2563eb", "standard": "#ea580c"}
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    for profile_index, profile in enumerate(profiles):
        rows = [
            next(
                row
                for row in family_rows
                if row["noise_profile"] == profile and row["family"] == family
            )
            for family in family_order
        ]
        offset = (profile_index - (len(profiles) - 1) / 2.0) * width
        axes[0].bar(
            x + offset,
            [row["applied_v_mae_mps"] for row in rows],
            width,
            label=profile,
            color=colors.get(profile),
            alpha=0.88,
        )
        axes[1].bar(
            x + offset,
            [row["applied_w_mae_radps"] for row in rows],
            width,
            label=profile,
            color=colors.get(profile),
            alpha=0.88,
        )
    axes[0].axhline(0.015, color="black", linestyle="--", linewidth=1, label="gate")
    axes[1].axhline(0.008, color="black", linestyle="--", linewidth=1, label="gate")
    axes[0].set_ylabel("v MAE (m/s)")
    axes[1].set_ylabel("w MAE (rad/s)")
    axes[0].set_title("V62 structured-random tracking against applied feasible reference")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [FAMILY_LABELS.get(family, family) for family in family_order],
        rotation=28,
        ha="right",
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "structured_family_tracking_mae.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _projection_plot(output_dir, profiles, family_order, family_rows):
    x = np.arange(len(family_order))
    width = 0.36
    colors = {"nominal": "#2563eb", "standard": "#ea580c"}
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    for profile_index, profile in enumerate(profiles):
        rows = [
            next(
                row
                for row in family_rows
                if row["noise_profile"] == profile and row["family"] == family
            )
            for family in family_order
        ]
        offset = (profile_index - (len(profiles) - 1) / 2.0) * width
        axes[0].bar(
            x + offset,
            [row["request_projection_w_mae_radps"] for row in rows],
            width,
            label=profile,
            color=colors.get(profile),
            alpha=0.88,
        )
        axes[1].bar(
            x + offset,
            [row["mean_abs_requested_yaw_integral_gap_rad"] for row in rows],
            width,
            label=profile,
            color=colors.get(profile),
            alpha=0.88,
        )
    axes[0].set_ylabel("mean |applied w - requested w| (rad/s)")
    axes[1].set_ylabel("mean |integrated yaw request gap| (rad)")
    axes[0].set_title("Raw SRU request distortion caused by feasibility/governor")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [FAMILY_LABELS.get(family, family) for family in family_order],
        rotation=28,
        ha="right",
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "structured_request_projection_gap.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _find_trace_case(runs, profile, family, criterion):
    candidates = []
    for run in runs:
        if run["summary"]["noise_profile"] != profile:
            continue
        rows = [row for row in run["rows"] if row["family"] == family]
        row = max(rows, key=lambda item: _float(item, criterion))
        candidates.append((run, row))
    return max(candidates, key=lambda item: _float(item[1], criterion))


def _trace_plot(output_dir, runs, profile, family, criterion):
    run, row = _find_trace_case(runs, profile, family, criterion)
    path = os.path.join(run["directory"], family + "_trace.npz")
    data = np.load(path)
    env_id = int(row["environment"])
    dt = float(data["dt_s"])
    time = np.arange(data["requested_v"].shape[0]) * dt
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(time, data["requested_v"][:, env_id], "--", color="#64748b", label="SRU request")
    axes[0].plot(time, data["applied_v"][:, env_id], color="#2563eb", label="feasible reference")
    axes[0].plot(time, data["measured_v"][:, env_id], color="#ea580c", label="measured")
    axes[1].plot(time, data["requested_w"][:, env_id], "--", color="#64748b", label="SRU request")
    axes[1].plot(time, data["applied_w"][:, env_id], color="#2563eb", label="feasible reference")
    axes[1].plot(time, data["measured_w"][:, env_id], color="#ea580c", label="measured")
    axes[2].plot(time, data["residual_yaw_gate"][:, env_id], color="#7c3aed", label="V62 yaw gate")
    axes[2].plot(time, data["action1"][:, env_id], color="#059669", alpha=0.8, label="executed action1")
    axes[0].set_ylabel("v (m/s)")
    axes[1].set_ylabel("w (rad/s)")
    axes[2].set_ylabel("gate / action")
    axes[2].set_xlabel("time after request change (s)")
    title = "%s | %s | seed=%s env=%d | %s=%.5f" % (
        profile,
        FAMILY_LABELS.get(family, family),
        run["summary"]["structured_seed"],
        env_id,
        criterion,
        _float(row, criterion),
    )
    axes[0].set_title(title)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best", ncol=3)
    fig.tight_layout()
    output = os.path.join(output_dir, "%s_%s_worst.png" % (profile, family))
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    args = _parse_args()
    root = os.path.abspath(args.root)
    output_dir = os.path.abspath(args.output_dir or os.path.join(root, "aggregate"))
    os.makedirs(output_dir, exist_ok=True)
    runs = _find_runs(root)
    profiles, family_order, profile_overall, family_rows = _aggregate(runs)
    csv_path = os.path.join(output_dir, "structured_family_aggregate.csv")
    _write_csv(csv_path, family_rows)
    plots = [
        _bar_plot(output_dir, profiles, family_order, family_rows),
        _projection_plot(output_dir, profiles, family_order, family_rows),
    ]
    for profile in profiles:
        plots.append(
            _trace_plot(
                output_dir,
                runs,
                profile,
                "fixed_w_v_reversal",
                "applied_w_p95_radps",
            )
        )
        plots.append(
            _trace_plot(
                output_dir,
                runs,
                profile,
                "infeasible_low_speed_high_yaw",
                "request_projection_w_mae_radps",
            )
        )
    summary = {
        "runs": len(runs),
        "profiles": profiles,
        "transitions": sum(len(run["rows"]) for run in runs),
        "transitions_per_profile": {
            profile: sum(
                len(run["rows"])
                for run in runs
                if run["summary"]["noise_profile"] == profile
            )
            for profile in profiles
        },
        "profile_overall": profile_overall,
        "family_aggregate": family_rows,
        "plots": plots,
        "tracking_gate_definition": {
            "v_mae_mps_max": 0.015,
            "w_mae_radps_max": 0.008,
            "v_p95_mps_max": 0.030,
            "w_p95_radps_max": 0.015,
            "v_sign_correct_ratio_min": 0.98,
            "w_sign_correct_ratio_min": 0.95,
        },
    }
    summary_path = os.path.join(output_dir, "structured_aggregate_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(profile_overall, indent=2, sort_keys=True))
    print("CSV:", csv_path)
    print("JSON:", summary_path)
    for plot in plots:
        print("PLOT:", plot)


if __name__ == "__main__":
    main()
