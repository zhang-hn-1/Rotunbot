"""Train the frozen V1 recurrent visual policy on 1:3:5 Straight/L/T data."""

import argparse
import json
import random
from pathlib import Path

import torch

from legged_gym.dwl.actor_critic_direct_velocity import load_direct_velocity_warm_start
from legged_gym.navigation.v1_teacher_dataset import (
    REQUIRED_STEP_FIELDS,
    load_teacher_dataset,
    merge_teacher_datasets,
)
from legged_gym.navigation.v1_velocity_imitation import (
    collate_imitation_sequences,
    imitation_loss,
    iter_imitation_sequences,
    train_imitation_epoch,
)
from legged_gym.scripts.train_sru_visual_corridor_v1_imitation import (
    build_v1_imitation_policy,
)


SOURCE_NAMES = ("straight", "l_turn", "t_junction")
DEFAULT_WEIGHTS = {"straight": 1, "l_turn": 3, "t_junction": 5}
DEFAULT_PARENT_CHECKPOINT = (
    "/home/jason/.codex/worktrees/codex-corridor-curriculum-navigation/"
    "logs/phase_c/v1_imitation_straight_l_balanced_20260901.pt"
)


def _dataset_sequence_length(dataset):
    try:
        return int(dataset["sequence_length"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("dataset must declare an integer sequence_length") from error


def _basic_dataset_shape(dataset, name):
    """Validate the fields needed by the weighted sampler without loading tensors."""
    if not isinstance(dataset, dict):
        raise ValueError("%s dataset must be a mapping" % name)
    if _dataset_sequence_length(dataset) != 16:
        raise ValueError("%s dataset must use sequence_length=16" % name)
    episodes = dataset.get("episodes")
    if not isinstance(episodes, (list, tuple)) or not episodes:
        raise ValueError("%s dataset must contain episodes" % name)
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("%s dataset episodes must be mappings" % name)
        if "episode_id" not in episode:
            raise ValueError("%s dataset episode is missing episode_id" % name)
    return dataset


def validate_imitation_dataset(dataset, name):
    """Fail closed on schema, temporal length, and real-depth provenance."""
    _basic_dataset_shape(dataset, name)
    if dataset.get("schema_version") != 1:
        raise ValueError("%s dataset must use schema_version=1" % name)
    if tuple(dataset.get("step_fields", ())) != REQUIRED_STEP_FIELDS:
        raise ValueError("%s dataset step schema mismatch" % name)
    metadata = dataset.get("metadata", {})
    if metadata.get("depth_backend_actual") != "isaacgym":
        raise RuntimeError("%s dataset is not real Isaac Gym IMAGE_DEPTH" % name)
    return dataset


def build_weighted_dataset(datasets, weights=None):
    """Build deterministic integer episode repetitions for the 1:3:5 mixture.

    The returned ``samples`` are lightweight episode references so this helper is
    usable in CPU-only tests.  The optional ``dataset`` member is populated when
    all inputs are complete V1 datasets and is what the trainer consumes.
    """
    if not hasattr(datasets, "keys"):
        raise ValueError("datasets must be a mapping keyed by source name")
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if tuple(weights.keys()) != SOURCE_NAMES:
        raise ValueError("weights must contain straight, l_turn, and t_junction")
    for source in SOURCE_NAMES:
        try:
            weight = int(weights[source])
        except (TypeError, ValueError) as error:
            raise ValueError("weights must be positive integers") from error
        if weight <= 0 or float(weights[source]) != weight:
            raise ValueError("weights must be positive integers")
        _basic_dataset_shape(datasets[source], source)

    samples = []
    repeated = []
    for source in SOURCE_NAMES:
        dataset = datasets[source]
        weight = int(weights[source])
        repeated.extend([dataset] * weight)
        # Preserve source ordering within each integer repetition.  This makes
        # the effective distribution inspectable and the epoch reproducible.
        for _ in range(weight):
            for episode in dataset["episodes"]:
                samples.append({"source": source, "episode": episode})

    result = {
        "samples": samples,
        "effective_sample_distribution": {
            source: int(weights[source]) for source in SOURCE_NAMES
        },
        "effective_episode_counts": {
            source: int(len(datasets[source]["episodes"]) * int(weights[source]))
            for source in SOURCE_NAMES
        },
    }
    try:
        result["dataset"] = merge_teacher_datasets(repeated)
    except (KeyError, TypeError, ValueError):
        # Small contract tests may intentionally use abbreviated episode maps.
        # The full trainer calls validate_imitation_dataset first, so real runs
        # always take the merged-dataset path.
        pass
    return result


def _weighted_training_dataset(datasets):
    sampled = build_weighted_dataset(datasets)
    if "dataset" not in sampled:
        raise ValueError("complete V1 datasets are required for training")
    return sampled["dataset"], sampled


def _evaluate_detailed(model, dataset, batch_size, device, max_forward_speed, max_yaw_rate):
    """Compute validation loss and normalized/physical command MAE."""
    model.eval()
    losses = []
    absolute_error = torch.zeros(2, dtype=torch.float64)
    valid_steps = 0
    with torch.no_grad():
        sequences = list(iter_imitation_sequences(dataset))
        for start in range(0, len(sequences), max(1, int(batch_size))):
            batch = collate_imitation_sequences(
                sequences[start:start + max(1, int(batch_size))],
                hidden_dim=model.memory.hidden_dim,
                device=device,
            )
            prediction = model._mean(
                batch["observations"],
                hidden_states=batch["initial_hidden"],
                masks=batch["recurrent_masks"],
                update_state=False,
            )
            losses.append(float(imitation_loss(model, batch).detach().cpu()))
            valid = batch["valid_mask"].unsqueeze(-1).to(dtype=prediction.dtype)
            error = (prediction - batch["targets"]).abs() * valid
            absolute_error += error.sum(dim=(0, 1)).detach().cpu().to(torch.float64)
            valid_steps += int(batch["valid_mask"].sum().item())
    if not losses or valid_steps <= 0:
        raise ValueError("teacher dataset produced no validation sequences")
    normalized = absolute_error / float(valid_steps)
    return {
        "validation_masked_huber_loss": sum(losses) / len(losses),
        "mean_normalized_command_mae": float(normalized.mean().item()),
        "normalized_v_mae": float(normalized[0].item()),
        "normalized_w_mae": float(normalized[1].item()),
        "v_mae_mps": float((normalized[0] * float(max_forward_speed)).item()),
        "w_mae_rps": float((normalized[1] * float(max_yaw_rate)).item()),
        "validation_sequences": int(len(sequences)),
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--straight-dataset", required=True)
    parser.add_argument("--l-dataset", required=True)
    parser.add_argument("--t-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-checkpoint", default=DEFAULT_PARENT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if int(args.epochs) <= 0 or int(args.batch_size) <= 0:
        raise ValueError("epochs and batch-size must be positive")
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)

    loaded = {
        "straight": load_teacher_dataset(args.straight_dataset),
        "l_turn": load_teacher_dataset(args.l_dataset),
        "t_junction": load_teacher_dataset(args.t_dataset),
    }
    for name, dataset in loaded.items():
        validate_imitation_dataset(dataset, name)
    dataset, sampled = _weighted_training_dataset(loaded)
    model = build_v1_imitation_policy(device)
    warm_start = load_direct_velocity_warm_start(
        model, args.parent_checkpoint, map_location=device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    max_forward_speed = float(loaded["straight"]["metadata"].get(
        "command_ranges", {}
    ).get("v_cmd", [0.25, 0.25])[-1])
    max_yaw_rate = float(loaded["straight"]["metadata"].get(
        "command_ranges", {}
    ).get("w_cmd", [0.10, 0.10])[-1])

    history = []
    best_epoch = 0
    best_loss = float("inf")
    best_state = None
    for epoch in range(int(args.epochs)):
        train_loss = train_imitation_epoch(
            model,
            dataset,
            optimizer,
            batch_size=int(args.batch_size),
            seed=int(args.seed) + epoch,
            device=device,
        )
        validation = _evaluate_detailed(
            model, dataset, args.batch_size, device,
            max_forward_speed, max_yaw_rate,
        )
        row = {
            "epoch": epoch + 1,
            "masked_huber_loss": float(train_loss),
            **validation,
        }
        history.append(row)
        if row["validation_masked_huber_loss"] < best_loss:
            best_loss = row["validation_masked_huber_loss"]
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(json.dumps(row, sort_keys=True), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    final_audit = _evaluate_detailed(
        model, dataset, args.batch_size, device,
        max_forward_speed, max_yaw_rate,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "training_stage": "T_JUNCTION_MIXED_BC",
        "datasets": {
            name: str(Path(path).resolve())
            for name, path in {
                "straight": args.straight_dataset,
                "l_turn": args.l_dataset,
                "t_junction": args.t_dataset,
            }.items()
        },
        "weights": dict(DEFAULT_WEIGHTS),
        "effective_sample_distribution": sampled["effective_sample_distribution"],
        "effective_episode_counts": sampled["effective_episode_counts"],
        "merged_episode_count": int(len(dataset["episodes"])),
        "sequence_length": int(dataset["sequence_length"]),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "parent_checkpoint": str(Path(args.parent_checkpoint).resolve()),
        "warm_start": warm_start,
        "best_epoch": int(best_epoch),
        "history": history,
        "audit": final_audit,
        "abi": {
            "actor_observation_dim": 275,
            "depth_shape": [8, 32],
            "recurrent_hidden_dim": 128,
            "action_dim": 2,
            "sequence_length": 16,
        },
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": int(best_epoch),
            "imitation": metadata,
        },
        output,
    )
    with open(str(output) + ".json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps({"output": str(output), "audit": final_audit}, indent=2, sort_keys=True))
    return metadata


if __name__ == "__main__":
    main()
