"""Small JSON/CSV logger shared by non-learning navigation gates."""

import csv
import json
from pathlib import Path

import numpy as np


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class EpisodeLogger:
    """Collect one episode and write a stable machine-readable record."""

    def __init__(self, metadata=None):
        self.metadata = dict(metadata or {})
        self.trajectory = []
        self.summary = None

    def record_step(self, **fields):
        self.trajectory.append(_json_value(dict(fields)))

    def finish(self, **summary):
        self.summary = _json_value(dict(summary))

    def _payload(self):
        if self.summary is None:
            raise RuntimeError("finish() must be called before writing an episode")
        return {
            "metadata": _json_value(self.metadata),
            "summary": self.summary,
            "trajectory": self.trajectory,
        }

    def write_json(self, path):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self._payload(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_csv(self, path):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in self.trajectory for key in row})
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.trajectory)
