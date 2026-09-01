"""Train the recurrent V1 SRU actor from the auditable teacher dataset."""

import argparse
import json
import os
import random
from pathlib import Path

import torch

from legged_gym.dwl.actor_critic_direct_velocity import (
    ActorCriticDirectVelocity,
    load_direct_velocity_warm_start,
)
from legged_gym.navigation.v1_teacher_dataset import load_teacher_dataset
from legged_gym.navigation.v1_velocity_imitation import (
    collate_imitation_sequences,
    imitation_loss,
    iter_imitation_sequences,
    make_imitation_batches,
    train_imitation_epoch,
)


def build_v1_imitation_policy(device):
    """Construct the exact current V1 recurrent actor ABI."""
    return ActorCriticDirectVelocity(
        num_short_obs=275,
        num_proprio_obs=275,
        num_critic_obs=21,
        num_actions=2,
        depth_height=8,
        depth_width=32,
        proprio_dim=12,
        goal_dim=2,
        previous_command_dim=2,
        previous_actual_velocity_dim=2,
        encoder_dim=64,
        attention_heads=4,
        hidden_dim=128,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        init_noise_std=0.20,
        min_noise_std=0.05,
        max_noise_std=0.80,
    ).to(device)


def evaluate_imitation(model, dataset, batch_size, device):
    """Return teacher-vs-student command audit metrics on all stored sequences."""
    model.eval()
    losses = []
    absolute_error = []
    maximum_action = 0.0
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
            valid = batch["valid_mask"].unsqueeze(-1)
            absolute_error.append(
                ((prediction - batch["targets"]).abs() * valid).sum().item()
                / max(float(valid.sum().item()), 1.0)
            )
            maximum_action = max(maximum_action, float(prediction.abs().max().item()))
    episode_count = len(dataset["episodes"])
    sequence_count = len(sequences)
    return {
        "episodes": int(episode_count),
        "sequences": int(sequence_count),
        "mean_masked_huber_loss": sum(losses) / max(len(losses), 1),
        "mean_normalized_command_abs_error": sum(absolute_error) / max(len(absolute_error), 1),
        "maximum_absolute_action": maximum_action,
        "finite": bool(maximum_action == maximum_action and maximum_action < float("inf")),
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)
    dataset = load_teacher_dataset(args.dataset)
    metadata = dataset.get("metadata", {})
    if metadata.get("depth_backend_actual") != "isaacgym":
        raise RuntimeError("imitation requires a real Isaac Gym IMAGE_DEPTH dataset")
    model = build_v1_imitation_policy(device)
    warm_start = None
    if args.parent_checkpoint:
        warm_start = load_direct_velocity_warm_start(
            model, args.parent_checkpoint, map_location=device
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    history = []
    for epoch in range(int(args.epochs)):
        loss = train_imitation_epoch(
            model,
            dataset,
            optimizer,
            batch_size=int(args.batch_size),
            seed=int(args.seed) + epoch,
            device=device,
        )
        row = {"epoch": epoch + 1, "masked_huber_loss": loss}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    audit = evaluate_imitation(model, dataset, args.batch_size, device)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": int(args.epochs),
        "imitation": {
            "schema_version": 1,
            "dataset": str(Path(args.dataset).resolve()),
            "dataset_metadata": metadata,
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "warm_start": warm_start,
            "history": history,
            "audit": audit,
        },
    }
    torch.save(payload, output)
    with open(str(output) + ".json", "w", encoding="utf-8") as handle:
        json.dump(payload["imitation"], handle, indent=2, sort_keys=True)
    print(json.dumps({"output": str(output), "audit": audit}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
