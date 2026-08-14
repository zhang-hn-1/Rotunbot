#!/usr/bin/env python3
"""Average fine-tuning deltas from checkpoints sharing one base model."""

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--branches", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--delta-scale", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    base = torch.load(args.base, map_location="cpu")
    branches = [torch.load(path, map_location="cpu") for path in args.branches]
    base_state = base["model_state_dict"]

    for path, branch in zip(args.branches, branches):
        if branch["model_state_dict"].keys() != base_state.keys():
            raise ValueError(f"incompatible model keys: {path}")

    averaged = {}
    for key, base_value in base_state.items():
        if torch.is_floating_point(base_value):
            deltas = [
                branch["model_state_dict"][key].to(torch.float64)
                - base_value.to(torch.float64)
                for branch in branches
            ]
            value = base_value.to(torch.float64) + float(args.delta_scale) * torch.stack(
                deltas, dim=0
            ).mean(dim=0)
            averaged[key] = value.to(dtype=base_value.dtype)
        else:
            averaged[key] = base_value.clone()

    output = dict(base)
    output["model_state_dict"] = averaged
    output["iter"] = max(int(branch.get("iter", 0)) for branch in branches)
    output["optimizer_state_dict"] = {}
    output["env_state"] = branches[0].get("env_state")
    output["infos"] = {
        "checkpoint_average": {
            "base": str(Path(args.base).resolve()),
            "branches": [str(Path(path).resolve()) for path in args.branches],
            "delta_scale": float(args.delta_scale),
        }
    }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)
    print(f"Wrote averaged checkpoint: {destination}")


if __name__ == "__main__":
    main()
