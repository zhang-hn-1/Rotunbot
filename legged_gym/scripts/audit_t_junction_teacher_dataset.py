"""Audit a V1-compatible real-depth T-junction teacher dataset."""

import argparse
import json
import math
import numbers
import sys
from collections.abc import Mapping
from pathlib import Path


_V1_STEP_FIELDS = (
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
_SCENARIOS = ("T_LEFT", "T_RIGHT")
_METADATA_FIELDS = (
    "depth_backend_requested",
    "depth_backend_actual",
    "scenarios",
    "episode_scenarios",
    "seed",
    "geometry",
    "command_ranges",
)


def _to_list(value):
    """Convert tensor-like vectors to Python values without requiring torch."""
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else list(value)


def _finite(value, path="dataset"):
    """Reject every non-finite scalar nested inside lists, mappings, or tensors."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite(item, "%s.%s" % (path, key))
        return
    if isinstance(value, numbers.Number):
        if not math.isfinite(float(value)):
            raise ValueError("%s contains a non-finite numeric value" % path)
        return
    if isinstance(value, (str, bytes, bool)) or value is None:
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, "%s[%d]" % (path, index))
        return
    try:
        import torch

        if torch.is_tensor(value):
            if value.dtype.is_floating_point and not bool(torch.isfinite(value).all().item()):
                raise ValueError("%s contains a non-finite tensor value" % path)
            return
    except ImportError:
        pass
    isfinite = getattr(value, "isfinite", None)
    if callable(isfinite):
        result = isfinite()
        all_values = getattr(result, "all", None)
        result = all_values() if callable(all_values) else result
        item = getattr(result, "item", None)
        result = item() if callable(item) else result
        if not bool(result):
            raise ValueError("%s contains a non-finite tensor value" % path)


def _terminal_done(value):
    values = _to_list(value)
    if not values:
        raise ValueError("episode done field must not be empty")
    if not bool(values[-1]) or any(bool(item) for item in values[:-1]):
        raise ValueError("episode done field must be true only on its terminal row")


def audit_t_teacher_dataset(dataset):
    """Validate schema, provenance, chronology, terminal rows, and T coverage."""
    if not isinstance(dataset, Mapping):
        raise ValueError("teacher dataset must be a mapping")
    if dataset.get("schema_version") != 1:
        raise ValueError("unsupported teacher dataset schema")
    if tuple(dataset.get("step_fields", ())) != _V1_STEP_FIELDS:
        raise ValueError("teacher dataset step schema mismatch")
    metadata = dataset.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("teacher dataset metadata is required")
    missing_metadata = [field for field in _METADATA_FIELDS if field not in metadata]
    if missing_metadata:
        raise ValueError("missing T teacher metadata: %s" % ", ".join(missing_metadata))
    if metadata["depth_backend_requested"] != "isaacgym":
        raise ValueError("depth_backend_requested must be isaacgym")
    if metadata["depth_backend_actual"] != "isaacgym":
        raise ValueError("depth_backend_actual must be isaacgym")
    if tuple(metadata["scenarios"]) != _SCENARIOS:
        raise ValueError("metadata scenarios must be T_LEFT and T_RIGHT")

    episodes = list(dataset.get("episodes", ()))
    if not episodes:
        raise ValueError("teacher dataset contains no episodes")
    episode_scenarios = list(metadata["episode_scenarios"])
    if len(episode_scenarios) != len(episodes):
        raise ValueError("episode_scenarios must align with dataset episodes")
    if any(side not in _SCENARIOS for side in episode_scenarios):
        raise ValueError("episode_scenarios must only contain T_LEFT or T_RIGHT")

    macro_steps = 0
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise ValueError("dataset episode %d must be a mapping" % index)
        if "episode_id" not in episode:
            raise ValueError("dataset episode %d is missing episode_id" % index)
        missing_fields = [field for field in _V1_STEP_FIELDS if field not in ("episode_id",) and field not in episode]
        if missing_fields:
            raise ValueError(
                "dataset episode %d missing fields: %s" % (index, ", ".join(missing_fields))
            )
        step_ids = _to_list(episode["step_id"])
        if step_ids != list(range(len(step_ids))):
            raise ValueError("dataset episode %d step ids are not chronological" % index)
        _terminal_done(episode["done"])
        if int(episode.get("sequence_length", len(step_ids))) != len(step_ids):
            raise ValueError("dataset episode %d sequence length mismatch" % index)
        macro_steps += len(step_ids)
        for field in _V1_STEP_FIELDS:
            if field != "episode_id":
                _finite(episode[field], "episodes[%d].%s" % (index, field))

    _finite(metadata, "metadata")
    scenario_counts = {side: episode_scenarios.count(side) for side in _SCENARIOS}
    if not all(scenario_counts.values()):
        raise ValueError("dataset must contain T_LEFT and T_RIGHT episodes")
    return {
        "schema_version": 1,
        "depth_backend_requested": metadata["depth_backend_requested"],
        "depth_backend_actual": metadata["depth_backend_actual"],
        "finite": True,
        "terminal_done": True,
        "chronological_step_ids": True,
        "episode_count": len(episodes),
        "scenario_counts": scenario_counts,
        "macro_steps": int(macro_steps),
        "sequence_length": int(dataset["sequence_length"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    stage_args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    # Keep Isaac Gym ahead of torch for this repository's pinned runtime.
    import isaacgym  # noqa: F401
    import torch  # noqa: F401
    from legged_gym.navigation.v1_teacher_dataset import load_teacher_dataset

    dataset_path = Path(stage_args.dataset).resolve()
    result = audit_t_teacher_dataset(load_teacher_dataset(dataset_path))
    result["dataset"] = str(dataset_path)
    output = Path(stage_args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
