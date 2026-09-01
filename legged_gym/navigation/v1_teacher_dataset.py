"""Ordered, episode-bounded storage for V1 teacher rollouts."""

from pathlib import Path

import torch


REQUIRED_STEP_FIELDS = (
    "episode_id",
    "step_id",
    "depth",
    "goal_xy_robot",
    "proprioception",
    "previous_command",
    "previous_actual_velocity",
    "teacher_command",
    "actual_velocity",
    "governor_command",
    "projection_command",
    "done",
    "success",
    "collision",
    "goal_distance",
)

_TENSOR_FIELDS = {
    "depth",
    "goal_xy_robot",
    "proprioception",
    "previous_command",
    "previous_actual_velocity",
    "teacher_command",
    "actual_velocity",
    "governor_command",
    "projection_command",
    "goal_distance",
}
_BOOL_FIELDS = {"done", "success", "collision"}


class TeacherSequenceWriter:
    """Collect macro-step rows without ever concatenating episodes."""

    def __init__(self, sequence_length=16):
        self.sequence_length = int(sequence_length)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self._current_episode = None
        self._episodes = []

    def append(self, row):
        missing = [field for field in REQUIRED_STEP_FIELDS if field not in row]
        if missing:
            raise ValueError("missing teacher dataset fields: %s" % ", ".join(missing))
        episode_id = int(row["episode_id"])
        step_id = int(row["step_id"])
        if self._current_episode is None:
            self._current_episode = {"episode_id": episode_id, "rows": []}
        elif episode_id != self._current_episode["episode_id"]:
            self._close_current_episode()
            if episode_id <= self._episodes[-1]["episode_id"]:
                raise ValueError("episode_id must increase monotonically")
            self._current_episode = {"episode_id": episode_id, "rows": []}
        rows = self._current_episode["rows"]
        if step_id != len(rows):
            raise ValueError("step_id must be chronological from zero within each episode")
        normalized = {}
        for field in REQUIRED_STEP_FIELDS:
            value = row[field]
            if field in _TENSOR_FIELDS:
                value = torch.as_tensor(value).detach().cpu()
                if value.dtype.is_floating_point and not torch.isfinite(value).all():
                    raise ValueError("non-finite teacher dataset field: %s" % field)
            elif field in _BOOL_FIELDS:
                value = bool(value)
            elif field in ("episode_id", "step_id"):
                value = int(value)
            normalized[field] = value
        normalized["done"] = bool(normalized["done"])
        if normalized["done"] and rows and bool(rows[-1]["done"]):
            raise ValueError("duplicate done row in one episode")
        rows.append(normalized)

    def _close_current_episode(self):
        if self._current_episode is None:
            return
        rows = self._current_episode["rows"]
        if not rows:
            raise ValueError("cannot close an empty teacher episode")
        if not bool(rows[-1]["done"]):
            raise ValueError("episode must end with done=True")
        self._episodes.append(
            self._materialize(self._current_episode, self.sequence_length)
        )
        self._current_episode = None

    @staticmethod
    def _materialize(episode, sequence_length):
        output = {"episode_id": int(episode["episode_id"])}
        rows = episode["rows"]
        for field in REQUIRED_STEP_FIELDS:
            if field == "step_id":
                dtype = torch.long
                values = [row[field] for row in rows]
                output[field] = torch.as_tensor(values, dtype=dtype)
            elif field == "episode_id":
                # ``episode_id`` is kept as the episode-level key; the
                # repeated per-step value is redundant inside one episode.
                output["episode_ids"] = torch.full(
                    (len(rows),), int(episode["episode_id"]), dtype=torch.long
                )
            elif field in _BOOL_FIELDS:
                output[field] = torch.as_tensor(
                    [row[field] for row in rows], dtype=torch.bool
                )
            else:
                output[field] = torch.stack([row[field] for row in rows])
        output["sequence_length"] = int(len(rows))
        output["num_sequences"] = int(
            (len(rows) + sequence_length - 1) // sequence_length
        )
        return output

    def finalize(self):
        self._close_current_episode()
        if not self._episodes:
            raise ValueError("teacher dataset must contain at least one episode")
        result = {
            "schema_version": 1,
            "step_fields": list(REQUIRED_STEP_FIELDS),
            "sequence_length": self.sequence_length,
            "episodes": list(self._episodes),
        }
        return result

    def save(self, path, metadata=None):
        result = self.finalize()
        result["metadata"] = dict(metadata or {})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, path)
        return path


def load_teacher_dataset(path):
    """Load a CPU teacher dataset and validate its top-level schema."""
    dataset = torch.load(Path(path), map_location="cpu")
    if dataset.get("schema_version") != 1:
        raise ValueError("unsupported teacher dataset schema")
    if tuple(dataset.get("step_fields", ())) != REQUIRED_STEP_FIELDS:
        raise ValueError("teacher dataset step schema mismatch")
    if not dataset.get("episodes"):
        raise ValueError("teacher dataset contains no episodes")
    return dataset


def merge_teacher_datasets(datasets):
    """Merge compatible CPU datasets and assign fresh episode ids."""
    datasets = list(datasets)
    if not datasets:
        raise ValueError("at least one teacher dataset is required")
    sequence_length = int(datasets[0].get("sequence_length", 0))
    if sequence_length <= 0:
        raise ValueError("teacher dataset sequence length must be positive")
    episodes = []
    sources = []
    for source_index, dataset in enumerate(datasets):
        if dataset.get("schema_version") != 1:
            raise ValueError("unsupported teacher dataset schema")
        if tuple(dataset.get("step_fields", ())) != REQUIRED_STEP_FIELDS:
            raise ValueError("teacher dataset step schema mismatch")
        if int(dataset.get("sequence_length", 0)) != sequence_length:
            raise ValueError("teacher datasets must use the same sequence length")
        start = len(episodes)
        for episode in dataset.get("episodes", ()):
            copied = {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in episode.items()
            }
            copied["episode_id"] = len(episodes)
            if "episode_ids" in copied:
                copied["episode_ids"] = torch.full_like(
                    copied["episode_ids"], len(episodes)
                )
            episodes.append(copied)
        sources.append(
            {
                "source_index": source_index,
                "metadata": dict(dataset.get("metadata", {})),
                "episode_start": start,
                "episode_end": len(episodes),
            }
        )
    if not episodes:
        raise ValueError("teacher datasets contain no episodes")
    return {
        "schema_version": 1,
        "step_fields": list(REQUIRED_STEP_FIELDS),
        "sequence_length": sequence_length,
        "episodes": episodes,
        "metadata": {"merged_sources": sources},
    }
