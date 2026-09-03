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
# Optional audit fields are preserved without changing the frozen v1 training ABI.
OPTIONAL_STEP_FIELDS = (
    "depth_raw",
    "timestamp_s",
    "robot_pose",
    "global_goal_distance",
    "waypoint",
    "remaining_path",
    "teacher_raw_command",
    "teacher_projected_command",
    "applied_feasible_command",
    "transition_state",
    "transition_active",
    "failure_reason",
)
_OPTIONAL_TENSOR_FIELDS = {
    "depth_raw",
    "timestamp_s",
    "robot_pose",
    "global_goal_distance",
    "waypoint",
    "remaining_path",
    "teacher_raw_command",
    "teacher_projected_command",
    "applied_feasible_command",
    "transition_state",
}
_OPTIONAL_BOOL_FIELDS = {"transition_active"}


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
        fields = list(REQUIRED_STEP_FIELDS) + [
            field for field in OPTIONAL_STEP_FIELDS if field in row
        ]
        for field in fields:
            value = row[field]
            if field in _TENSOR_FIELDS or field in _OPTIONAL_TENSOR_FIELDS:
                value = torch.as_tensor(value).detach().cpu()
                if value.dtype.is_floating_point and not torch.isfinite(value).all():
                    raise ValueError("non-finite teacher dataset field: %s" % field)
            elif field in _BOOL_FIELDS or field in _OPTIONAL_BOOL_FIELDS:
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
        optional_fields = [
            field for field in OPTIONAL_STEP_FIELDS if field in rows[0]
        ]
        if any(set(optional_fields) != {field for field in OPTIONAL_STEP_FIELDS if field in row} for row in rows):
            raise ValueError("optional teacher dataset fields must be present on every row")
        fields = list(REQUIRED_STEP_FIELDS) + optional_fields
        for field in fields:
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
            elif field in _BOOL_FIELDS or field in _OPTIONAL_BOOL_FIELDS:
                output[field] = torch.as_tensor(
                    [row[field] for row in rows], dtype=torch.bool
                )
            elif field == "failure_reason":
                output[field] = [row[field] for row in rows]
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


def validate_teacher_dataset(dataset):
    """Validate a complete CPU dataset, including every episode payload."""
    if not isinstance(dataset, dict) or dataset.get("schema_version") != 1:
        raise ValueError("unsupported teacher dataset schema")
    if tuple(dataset.get("step_fields", ())) != REQUIRED_STEP_FIELDS:
        raise ValueError("teacher dataset step schema mismatch")
    episodes = dataset.get("episodes")
    if not isinstance(episodes, (list, tuple)) or not episodes:
        raise ValueError("teacher dataset contains no episodes")
    previous_episode_id = -1
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("teacher episode must be a mapping")
        episode_id = int(episode.get("episode_id", -1))
        if episode_id <= previous_episode_id:
            raise ValueError("episode ids must increase monotonically")
        previous_episode_id = episode_id
        lengths = []
        for field in REQUIRED_STEP_FIELDS:
            key = "episode_ids" if field == "episode_id" else field
            if key not in episode:
                raise ValueError("episode missing field: %s" % key)
            value = episode[key]
            if field in _TENSOR_FIELDS or field in _BOOL_FIELDS or field == "step_id" or field == "episode_id":
                if not isinstance(value, torch.Tensor) or value.ndim == 0:
                    raise ValueError("episode field must be a tensor: %s" % key)
                if value.dtype.is_floating_point and not torch.isfinite(value).all():
                    raise ValueError("non-finite teacher dataset field: %s" % key)
                lengths.append(int(value.shape[0]))
        if len(set(lengths)) != 1 or lengths[0] <= 0:
            raise ValueError("episode field lengths are inconsistent")
        optional_present = {field for field in OPTIONAL_STEP_FIELDS if field in episode}
        for field in optional_present:
            value = episode[field]
            if field == "failure_reason":
                if not isinstance(value, (list, tuple)) or len(value) != lengths[0]:
                    raise ValueError("optional failure_reason field has invalid length")
            else:
                if not isinstance(value, torch.Tensor) or value.ndim == 0 or int(value.shape[0]) != lengths[0]:
                    raise ValueError("optional field has invalid shape: %s" % field)
                if value.dtype.is_floating_point and not torch.isfinite(value).all():
                    raise ValueError("non-finite optional teacher dataset field: %s" % field)
        expected_steps = torch.arange(lengths[0], dtype=torch.long)
        if not torch.equal(episode["step_id"].cpu(), expected_steps):
            raise ValueError("step_id must be chronological from zero")
        if not torch.equal(episode["episode_ids"].cpu(), torch.full((lengths[0],), episode_id, dtype=torch.long)):
            raise ValueError("episode_ids do not match episode key")
        if not bool(episode["done"][-1].item()) or bool(episode["done"][:-1].any().item()):
            raise ValueError("done must occur exactly at the episode boundary")
        if int(episode.get("sequence_length", lengths[0])) <= 0:
            raise ValueError("episode sequence_length must be positive")
    return dataset


def load_teacher_dataset(path):
    """Load a CPU teacher dataset and validate every episode payload."""
    dataset = torch.load(Path(path), map_location="cpu")
    return validate_teacher_dataset(dataset)


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
        validate_teacher_dataset(dataset)
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
