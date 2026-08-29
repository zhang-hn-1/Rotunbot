"""Plots for the Stage1.4 matched-mode evaluation."""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig, directory, name):
    fig.tight_layout()
    fig.savefig(os.path.join(directory, name), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage14_output_dir", default="logs/stage1_4_dynamic_governor")
    args = parser.parse_args()
    directory = os.path.abspath(args.stage14_output_dir)
    rows = _read(os.path.join(directory, "stage1_4_trials.csv"))
    modes = ("Baseline", "Static", "Dynamic")
    colors = {"Baseline": "#777777", "Static": "#1f77b4", "Dynamic": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for group in ("low_speed", "high_speed"):
        for mode in modes:
            subset = [r for r in rows if r["group"] == group and r["mode"] == mode]
            axes[0].bar(
                "%s\n%s" % (group.replace("_", " "), mode),
                sum(abs(float(r["w_error"])) for r in subset) / len(subset),
                color=colors[mode],
            )
            axes[1].bar(
                "%s\n%s" % (group.replace("_", " "), mode),
                sum(abs(float(r["requested_w_error"])) for r in subset) / len(subset),
                color=colors[mode],
            )
    axes[0].set_ylabel("mean selected-command yaw error (rad/s)")
    axes[1].set_ylabel("mean original-request yaw error (rad/s)")
    axes[0].set_title("Error to applied command")
    axes[1].set_title("Error to original request")
    for axis in axes:
        axis.tick_params(axis="x", labelsize=8)
    _save(fig, directory, "mode_yaw_error_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = ("command_modification_count", "static_saturation_count", "fallback_count", "yaw_sign_error_count")
    labels = ("dynamic modifications", "static saturation", "fallback", "yaw sign errors")
    width = 0.24
    x = list(range(len(metrics)))
    for index, mode in enumerate(modes):
        with open(os.path.join(directory, "stage1_4_aggregate.json"), encoding="utf-8") as handle:
            aggregate = json.load(handle)
        ax.bar([value + (index - 1) * width for value in x], [aggregate[mode][metric] for metric in metrics], width, label=mode, color=colors[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("sample count")
    ax.set_title("Governor and safety counters")
    ax.legend()
    _save(fig, directory, "governor_counters.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode and r["scenario"] == "low_speed_high_yaw_positive" and int(r["policy_step"]) == 0]
        ax.plot(range(len(subset)), [float(r["selected_w"]) for r in subset], "o-", label=mode, color=colors[mode])
    ax.set_xlabel("matched transition sample")
    ax.set_ylabel("selected yaw command (rad/s)")
    ax.set_title("Low-speed high-yaw command selection")
    ax.legend()
    _save(fig, directory, "low_speed_command_selection.png")


if __name__ == "__main__":
    main()
