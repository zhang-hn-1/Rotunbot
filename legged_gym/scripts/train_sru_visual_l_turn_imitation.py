"""Train the V1 recurrent visual policy on mixed Straight and L-turn data."""

import argparse
import json
import random
from pathlib import Path

import torch

from legged_gym.navigation.v1_teacher_dataset import (
    load_teacher_dataset,
    merge_teacher_datasets,
)
from legged_gym.navigation.v1_velocity_imitation import train_imitation_epoch
from legged_gym.scripts.train_sru_visual_corridor_v1_imitation import (
    build_v1_imitation_policy,
    evaluate_imitation,
)
from legged_gym.dwl.actor_critic_direct_velocity import load_direct_velocity_warm_start


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--straight-dataset", required=True)
    parser.add_argument("--l-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
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
    straight = load_teacher_dataset(args.straight_dataset)
    l_turn = load_teacher_dataset(args.l_dataset)
    for name, dataset in (("straight", straight), ("l_turn", l_turn)):
        if dataset.get("metadata", {}).get("depth_backend_actual") != "isaacgym":
            raise RuntimeError("%s dataset is not real Isaac Gym IMAGE_DEPTH" % name)
    dataset = merge_teacher_datasets([straight, l_turn])
    model = build_v1_imitation_policy(device)
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
    metadata = {
        "schema_version": 1,
        "datasets": {
            "straight": str(Path(args.straight_dataset).resolve()),
            "l_turn": str(Path(args.l_dataset).resolve()),
        },
        "merged_dataset_metadata": dataset["metadata"],
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "warm_start": warm_start,
        "history": history,
        "audit": audit,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": int(args.epochs),
            "imitation": metadata,
        },
        output,
    )
    with open(str(output) + ".json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps({"output": str(output), "audit": audit}, indent=2))
    return metadata


if __name__ == "__main__":
    main()
