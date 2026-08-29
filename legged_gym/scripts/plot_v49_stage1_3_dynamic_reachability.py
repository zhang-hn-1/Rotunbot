"""Offline plots for Stage1.3 dynamic response data."""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row, name):
    return float(row[name])


def _save(fig, output_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, name), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage13_output_dir", default="logs/stage1_3_dynamic_reachability")
    args = parser.parse_args()
    output_dir = os.path.abspath(args.stage13_output_dir)
    rows = _read(os.path.join(output_dir, "dynamic_response_aggregated.csv"))

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        [_f(row, "initial_forward_velocity") for row in rows],
        [_f(row, "projected_yaw_rate") for row in rows],
        c=[_f(row, "mean_actual_w_200ms") for row in rows], cmap="coolwarm",
    )
    fig.colorbar(scatter, ax=ax, label="mean actual w at 200 ms (rad/s)")
    ax.set(xlabel="current v (m/s)", ylabel="projected yaw command (rad/s)",
           title="Stage1.3 projected command coverage")
    _save(fig, output_dir, "command_coverage_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for current_v in sorted(set(round(_f(row, "initial_forward_velocity"), 3) for row in rows)):
        subset = [row for row in rows if round(_f(row, "initial_forward_velocity"), 3) == current_v]
        by_w = {}
        for row in subset:
            by_w.setdefault(round(_f(row, "projected_yaw_rate"), 5), []).append(_f(row, "mean_actual_w_200ms"))
        xs = sorted(by_w)
        ax.plot(xs, [sum(by_w[x]) / len(by_w[x]) for x in xs], "o-", label="v0=%.2f" % current_v)
    ax.plot([-0.04, 0.04], [-0.04, 0.04], "k--", linewidth=1)
    ax.set(xlabel="projected yaw command (rad/s)", ylabel="mean actual yaw at 200 ms (rad/s)",
           title="Command-to-response curves")
    ax.legend(ncol=2)
    _save(fig, output_dir, "command_to_response_curves.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for current_v in sorted(set(round(_f(row, "initial_forward_velocity"), 3) for row in rows)):
        subset = [row for row in rows if round(_f(row, "initial_forward_velocity"), 3) == current_v]
        ax.scatter(
            [_f(row, "projected_yaw_rate") for row in subset],
            [_f(row, "mean_actual_w_200ms") for row in subset],
            label="v0=%.2f" % current_v,
        )
    ax.set(xlabel="projected yaw command (rad/s)", ylabel="actual yaw response at 200 ms (rad/s)",
           title="Observed yaw response envelope")
    ax.legend(ncol=2)
    _save(fig, output_dir, "yaw_response_envelope.png")


if __name__ == "__main__":
    main()
