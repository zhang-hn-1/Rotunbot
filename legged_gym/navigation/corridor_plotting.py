"""Non-interactive plots shared by corridor evaluators."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _series(rows, key):
    return [float(row[key]) for row in rows]


def plot_corridor_artifacts(trajectory_csv, output_dir):
    rows = _read_rows(trajectory_csv)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []

    figure, axis = plt.subplots()
    axis.plot(_series(rows, "x"), _series(rows, "y"))
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title("XY trajectory")
    path = output / "xy_trajectory.png"
    figure.savefig(path)
    plt.close(figure)
    paths.append(str(path))

    figure, axis = plt.subplots()
    time = _series(rows, "time_s")
    axis.plot(time, _series(rows, "v_cmd"), label="v cmd")
    axis.plot(time, _series(rows, "v_actual"), label="v actual")
    axis.plot(time, _series(rows, "w_cmd"), label="w cmd")
    axis.plot(time, _series(rows, "w_actual"), label="w actual")
    axis.legend()
    axis.set_title("Velocity tracking")
    path = output / "velocity_tracking.png"
    figure.savefig(path)
    plt.close(figure)
    paths.append(str(path))

    figure, axis = plt.subplots()
    axis.plot(time, _series(rows, "goal_distance"))
    axis.set_xlabel("time (s)")
    axis.set_ylabel("distance (m)")
    axis.set_title("Goal distance")
    path = output / "goal_distance.png"
    figure.savefig(path)
    plt.close(figure)
    paths.append(str(path))
    return tuple(paths)
