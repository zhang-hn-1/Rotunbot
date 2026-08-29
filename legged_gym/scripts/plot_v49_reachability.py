"""Offline plots for the Stage 1.2 V49 dynamic reachability sweep."""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row, name):
    value = row.get(name, "")
    return None if value in ("", "None", "nan") else float(value)


def _bool(row, name):
    return str(row.get(name, "False")).lower() == "true"


def _save(fig, output_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, name), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reach_output_dir", default="logs/stage1_2_reachability")
    args = parser.parse_args()
    output_dir = os.path.abspath(args.reach_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    grid = _read(os.path.join(output_dir, "reachability_grid.csv"))
    transitions = _read(os.path.join(output_dir, "transition_summary.csv"))
    trace = _read(os.path.join(output_dir, "raw_50hz_trace.csv"))
    completed = [row for row in transitions if row.get("status") == "complete"]
    trace_by_trial = {}
    for row in trace:
        trace_by_trial.setdefault(row.get("trial_id"), []).append(row)

    fig, ax = plt.subplots(figsize=(7, 5))
    static_v = [_float(row, "projected_v") for row in grid]
    static_w = [_float(row, "projected_w") for row in grid]
    ax.scatter(static_v, static_w, marker="s", facecolors="none",
               edgecolors="tab:blue", label="static projected targets")
    dynamic = [row for row in completed if _bool(row, "response_reachable")]
    ax.scatter([_float(row, "actual_v_200ms") for row in dynamic],
               [_float(row, "actual_w_200ms") for row in dynamic],
               c=[_float(row, "initial_v") for row in dynamic], cmap="viridis",
               label="observed dynamic response")
    ax.set(xlabel="v (m/s)", ylabel="w (rad/s)",
           title="Static feasible targets vs 0.2 s dynamic responses")
    ax.legend()
    _save(fig, output_dir, "static_vs_dynamic_envelope.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for v0 in sorted(set(_float(row, "initial_v") for row in completed)):
        rows = [row for row in completed if _float(row, "initial_v") == v0]
        rows.sort(key=lambda row: _float(row, "projected_w"))
        ax.plot([_float(row, "projected_w") for row in rows],
                [_float(row, "actual_w_200ms") for row in rows], "o-",
                label="v0=%.2f" % v0)
    ax.plot([-0.03, 0.03], [-0.03, 0.03], "k--", linewidth=1)
    ax.set(xlabel="target projected w (rad/s)", ylabel="actual w at 0.2 s (rad/s)",
           title="Yaw response by initial forward speed")
    ax.legend()
    _save(fig, output_dir, "target_w_vs_actual_w_by_initial_v.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    grouped = {}
    for row in completed:
        if not _bool(row, "response_reachable"):
            continue
        v0 = _float(row, "initial_v")
        grouped.setdefault(v0, []).append(abs(_float(row, "actual_w_200ms")))
    xs = sorted(grouped)
    ax.plot(xs, [max(grouped[x]) for x in xs], "o-")
    ax.axvline(0.08, color="tab:orange", linestyle="--", linewidth=1)
    ax.axvline(0.10, color="tab:red", linestyle="--", linewidth=1)
    ax.set(xlabel="initial v (m/s)", ylabel="maximum reachable |w| at 0.2 s (rad/s)",
           title="Dynamic yaw envelope and low-speed boundaries")
    _save(fig, output_dir, "initial_v_vs_max_dynamic_abs_w.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    delta_w = [
        _float(trace_by_trial[row.get("trial_id")][0], "delta_command_w")
        for row in completed
    ]
    ax.scatter(delta_w,
               [_float(row, "actual_w_200ms") - _float(row, "projected_w")
                for row in completed],
               alpha=0.55)
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set(xlabel="delta command w (rad/s)", ylabel="w tracking error at 0.2 s (rad/s)",
           title="Yaw command transition vs terminal tracking error")
    _save(fig, output_dir, "delta_w_vs_terminal_w_error.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    selected = []
    for row in completed:
        key = (_float(row, "requested_initial_v"), _float(row, "target_w"))
        if key in ((0.08, 0.02), (0.12, 0.02), (0.06, -0.02)):
            selected.append(row)
    selected_ids = {row.get("trial_id") for row in selected}
    for trial_id in sorted(selected_ids):
        rows = [row for row in trace if row.get("trial_id") == trial_id]
        rows.sort(key=lambda row: int(row["policy_step"]))
        if not rows:
            continue
        label = "v0=%.2f,w*=%.2f" % (_float(rows[0], "initial_v"), _float(rows[0], "desired_w"))
        ax.plot([_float(row, "measured_v") for row in rows],
                [_float(row, "measured_w") for row in rows], "o-", label=label)
    ax.set(xlabel="measured v (m/s)", ylabel="measured w (rad/s)",
           title="Representative 50 Hz velocity-space trajectories")
    ax.legend()
    _save(fig, output_dir, "velocity_space_trajectories.png")


if __name__ == "__main__":
    main()
