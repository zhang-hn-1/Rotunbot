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
    "episodes_per_scene",
    "episode_provenance",
)
_VECTOR_FIELDS = (
    "goal_xy_robot",
    "previous_command",
    "previous_actual_velocity",
    "teacher_command",
    "actual_velocity",
    "governor_command",
    "projection_command",
)
_BOOLEAN_FIELDS = ("done", "success", "collision")


def _to_list(value, path):
    """Convert tensor-like vectors to Python values without requiring torch."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (str, bytes, Mapping)) or value is None:
        raise ValueError("%s must be an array" % path)
    try:
        return list(value)
    except TypeError as error:
        raise ValueError("%s must be an array" % path) from error


def _finite_number(value, path):
    if isinstance(value, bool) or not isinstance(value, numbers.Number):
        raise ValueError("%s must contain only finite numeric values" % path)
    if not math.isfinite(float(value)):
        raise ValueError("%s contains a non-finite numeric value" % path)


def _nested_shape_and_finite(value, path):
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        child_shapes = [
            _nested_shape_and_finite(item, "%s[%d]" % (path, index))
            for index, item in enumerate(value)
        ]
        if any(shape != child_shapes[0] for shape in child_shapes[1:]):
            raise ValueError("%s must not be ragged" % path)
        return (len(value),) + child_shapes[0]
    _finite_number(value, path)
    return ()


def _require_integral(value, path):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("%s must be an integer" % path)
    return int(value)


def _require_shape(value, expected, path):
    shape = _nested_shape_and_finite(value, path)
    if len(shape) != len(expected) or any(
        actual != wanted for actual, wanted in zip(shape, expected) if wanted is not None
    ):
        raise ValueError("%s must have shape %s; got %s" % (path, expected, shape))


def _numeric_vector(value, width, path):
    values = _to_list(value, path)
    if len(values) != width:
        raise ValueError("%s must contain %d values" % (path, width))
    for index, item in enumerate(values):
        _finite_number(item, "%s[%d]" % (path, index))
    return tuple(float(item) for item in values)


def _terminal_done(value):
    values = _to_list(value, "episode done field")
    if not values:
        raise ValueError("episode done field must not be empty")
    if any(not isinstance(item, bool) for item in values):
        raise ValueError("episode done field must contain booleans")
    if not values[-1] or any(values[:-1]):
        raise ValueError("episode done field must be true only on its terminal row")


def _validate_provenance(metadata, episodes, episode_scenarios):
    episodes_per_scene = _require_integral(
        metadata["episodes_per_scene"], "metadata.episodes_per_scene"
    )
    if episodes_per_scene <= 0 or len(episodes) != 2 * episodes_per_scene:
        raise ValueError("episodes must contain the declared number of both T sides")
    expected_scenarios = ["T_LEFT"] * episodes_per_scene + ["T_RIGHT"] * episodes_per_scene
    if episode_scenarios != expected_scenarios:
        raise ValueError("episode_scenarios must preserve T_LEFT then T_RIGHT collection order")
    geometry = metadata["geometry"]
    if not isinstance(geometry, Mapping) or set(geometry) != set(_SCENARIOS):
        raise ValueError("metadata geometry must contain exactly both T scenarios")
    goals = {}
    for scenario in _SCENARIOS:
        scene_geometry = geometry[scenario]
        if not isinstance(scene_geometry, Mapping) or "goal_xy" not in scene_geometry:
            raise ValueError("metadata geometry for %s must contain goal_xy" % scenario)
        goals[scenario] = _numeric_vector(
            scene_geometry["goal_xy"], 2, "metadata.geometry.%s.goal_xy" % scenario
        )
    provenance = metadata["episode_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("episode_provenance must map episode ids to evidence")
    episode_ids = [_require_integral(episode["episode_id"], "episode_id") for episode in episodes]
    if len(provenance) != len(episode_ids) or any(
        str(episode_id) not in provenance for episode_id in episode_ids
    ):
        raise ValueError("episode_provenance must cover each dataset episode exactly once")
    for index, episode in enumerate(episodes):
        episode_id = _require_integral(episode["episode_id"], "episode_id")
        entry = provenance[str(episode_id)]
        if not isinstance(entry, Mapping):
            raise ValueError("episode_provenance[%d] must be a mapping" % episode_id)
        required = {"scenario", "goal", "initial_pose", "initial_yaw", "horizon"}
        missing = required.difference(entry)
        if missing:
            raise ValueError("episode_provenance[%d] is missing %s" % (episode_id, ", ".join(sorted(missing))))
        scenario = entry["scenario"]
        if scenario not in _SCENARIOS or scenario != episode_scenarios[index]:
            raise ValueError("episode_provenance[%d] scenario mismatch" % episode_id)
        if _numeric_vector(entry["goal"], 2, "episode_provenance[%d].goal" % episode_id) != goals[scenario]:
            raise ValueError("episode_provenance[%d] goal does not match its scenario" % episode_id)
        _numeric_vector(entry["initial_pose"], 3, "episode_provenance[%d].initial_pose" % episode_id)
        _finite_number(entry["initial_yaw"], "episode_provenance[%d].initial_yaw" % episode_id)
        if _require_integral(entry["horizon"], "episode_provenance[%d].horizon" % episode_id) <= 0:
            raise ValueError("episode_provenance[%d].horizon must be positive" % episode_id)


def audit_t_teacher_dataset(dataset):
    """Validate schema, provenance, chronology, terminal rows, and T coverage."""
    if not isinstance(dataset, Mapping):
        raise ValueError("teacher dataset must be a mapping")
    if dataset.get("schema_version") != 1:
        raise ValueError("unsupported teacher dataset schema")
    if tuple(dataset.get("step_fields", ())) != _V1_STEP_FIELDS:
        raise ValueError("teacher dataset step schema mismatch")
    if _require_integral(dataset.get("sequence_length"), "dataset.sequence_length") != 16:
        raise ValueError("T teacher dataset sequence_length must be exactly 16")
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
    episode_scenarios = _to_list(metadata["episode_scenarios"], "metadata.episode_scenarios")
    if len(episode_scenarios) != len(episodes):
        raise ValueError("episode_scenarios must align with dataset episodes")
    if any(side not in _SCENARIOS for side in episode_scenarios):
        raise ValueError("episode_scenarios must only contain T_LEFT or T_RIGHT")

    macro_steps = 0
    previous_episode_id = None
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise ValueError("dataset episode %d must be a mapping" % index)
        if "episode_id" not in episode:
            raise ValueError("dataset episode %d is missing episode_id" % index)
        episode_id = _require_integral(episode["episode_id"], "episodes[%d].episode_id" % index)
        if previous_episode_id is not None and episode_id <= previous_episode_id:
            raise ValueError("dataset episode ids must increase strictly")
        previous_episode_id = episode_id
        missing_fields = [field for field in _V1_STEP_FIELDS if field not in ("episode_id",) and field not in episode]
        if missing_fields:
            raise ValueError(
                "dataset episode %d missing fields: %s" % (index, ", ".join(missing_fields))
            )
        step_ids = _to_list(episode["step_id"], "episodes[%d].step_id" % index)
        if any(_require_integral(value, "episodes[%d].step_id[%d]" % (index, step)) != step for step, value in enumerate(step_ids)):
            raise ValueError("dataset episode %d step ids are not chronological" % index)
        if not step_ids:
            raise ValueError("dataset episode %d must contain at least one step" % index)
        _terminal_done(episode["done"])
        if _require_integral(
            episode.get("sequence_length", len(step_ids)), "episodes[%d].sequence_length" % index
        ) != len(step_ids):
            raise ValueError("dataset episode %d sequence length mismatch" % index)
        expected_length = len(step_ids)
        if "num_sequences" not in episode:
            raise ValueError("dataset episode %d is missing writer num_sequences" % index)
        expected_num_sequences = int(math.ceil(expected_length / 16))
        if _require_integral(
            episode["num_sequences"], "episodes[%d].num_sequences" % index
        ) != expected_num_sequences:
            raise ValueError("dataset episode %d num_sequences mismatch" % index)
        if "episode_ids" not in episode:
            raise ValueError("dataset episode %d is missing writer episode_ids" % index)
        episode_ids = _to_list(episode["episode_ids"], "episodes[%d].episode_ids" % index)
        if len(episode_ids) != expected_length or any(
            _require_integral(value, "episodes[%d].episode_ids[%d]" % (index, step)) != episode_id
            for step, value in enumerate(episode_ids)
        ):
            raise ValueError("dataset episode %d episode_ids mismatch" % index)
        _require_shape(episode["depth"], (expected_length, 8, 32), "episodes[%d].depth" % index)
        _require_shape(episode["proprioception"], (expected_length, None), "episodes[%d].proprioception" % index)
        if _nested_shape_and_finite(episode["proprioception"], "episodes[%d].proprioception" % index)[1] <= 0:
            raise ValueError("episodes[%d].proprioception must have a positive width" % index)
        for field in _VECTOR_FIELDS:
            _require_shape(episode[field], (expected_length, 2), "episodes[%d].%s" % (index, field))
        _require_shape(episode["goal_distance"], (expected_length,), "episodes[%d].goal_distance" % index)
        for field in _BOOLEAN_FIELDS:
            values = _to_list(episode[field], "episodes[%d].%s" % (index, field))
            if len(values) != expected_length or any(not isinstance(value, bool) for value in values):
                raise ValueError("episodes[%d].%s must be boolean [N]" % (index, field))
        macro_steps += len(step_ids)

    scenario_counts = {side: episode_scenarios.count(side) for side in _SCENARIOS}
    if not all(scenario_counts.values()):
        raise ValueError("dataset must contain T_LEFT and T_RIGHT episodes")
    _validate_provenance(metadata, episodes, episode_scenarios)
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
