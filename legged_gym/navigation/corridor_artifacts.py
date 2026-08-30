"""Stable JSON/CSV artifacts and gate calculations for corridor experiments."""

import csv
import hashlib
import json
from pathlib import Path


def _compare(actual, operator, expected):
    if operator == ">=":
        return actual >= expected
    if operator == "<=":
        return actual <= expected
    if operator == ">":
        return actual > expected
    if operator == "<":
        return actual < expected
    if operator == "==":
        return actual == expected
    raise ValueError("unsupported gate operator: %s" % operator)


class GateResult:
    @staticmethod
    def evaluate(summary, current_rules, regression_rules):
        failures = []

        def evaluate_rules(label, rules):
            passed = True
            for metric, (operator, expected) in rules.items():
                if metric not in summary:
                    failures.append("%s missing metric %s" % (label, metric))
                    passed = False
                    continue
                actual = summary[metric]
                if not _compare(actual, operator, expected):
                    failures.append(
                        "%s %s=%s failed %s %s" % (
                            label, metric, actual, operator, expected
                        )
                    )
                    passed = False
            return passed

        current_pass = evaluate_rules("current", current_rules)
        regression_pass = evaluate_rules("regression", regression_rules)
        return {
            "current_pass": current_pass,
            "regression_pass": regression_pass,
            "pass": current_pass and regression_pass,
            "failures": failures,
        }


class CheckpointMetadata:
    @staticmethod
    def from_path(path, parent, stage, seed, iterations):
        checkpoint = Path(path).expanduser().resolve()
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        return {
            "checkpoint": str(checkpoint),
            "parent_checkpoint": str(parent),
            "stage": str(stage),
            "seed": int(seed),
            "iterations": int(iterations),
            "sha256": digest,
        }


class EpisodeLogger:
    def __init__(self, root, append=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._episodes = []
        self._trajectory = []
        if append:
            episodes_path = self.root / "episodes.csv"
            if episodes_path.is_file():
                with episodes_path.open(newline="") as handle:
                    self._episodes = list(csv.DictReader(handle))
                for row in self._episodes:
                    if "episode_id" in row:
                        row["episode_id"] = int(row["episode_id"])
            trajectory_path = self.root / "trajectory.csv"
            if trajectory_path.is_file():
                with trajectory_path.open(newline="") as handle:
                    self._trajectory = list(csv.DictReader(handle))

    @property
    def episodes(self):
        return self._episodes

    @property
    def trajectory(self):
        return self._trajectory

    def write_episode(self, record):
        self._episodes.append(dict(record))
        fields = sorted({key for row in self._episodes for key in row})
        with (self.root / "episodes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in self._episodes:
                writer.writerow(
                    {
                        key: json.dumps(value) if isinstance(value, (dict, list)) else value
                        for key, value in row.items()
                    }
                )

    def write_trajectory(self, rows):
        self._trajectory.extend(dict(row) for row in rows)
        if not self._trajectory:
            return
        fields = sorted({key for row in self._trajectory for key in row})
        with (self.root / "trajectory.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._trajectory)

    def write_summary(self, summary):
        (self.root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def replay_episode(root, episode_id):
    root = Path(root)
    with (root / "episodes.csv").open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    selected = next(row for row in episodes if int(row["episode_id"]) == int(episode_id))
    for key, value in list(selected.items()):
        if key == "scenario_parameters":
            selected[key] = json.loads(value)
    trajectory = []
    with (root / "trajectory.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["episode_id"]) == int(episode_id):
                trajectory.append(row)
    selected["episode_id"] = int(selected["episode_id"])
    selected["seed"] = int(selected["seed"])
    selected["trajectory"] = trajectory
    return selected
