"""Small GPU smoke test for the frozen V62 corridor evaluator."""

import argparse
import os

import isaacgym  # noqa: F401

from evaluate_v62_corridor import DEFAULT_CHECKPOINT, _scenario_for_family, run_corridor
from legged_gym.utils import get_args


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke_output_dir", default="logs/corridor_smoke")
    parser.add_argument("--smoke_checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--smoke_family", choices=("A0", "A1"), default="A0")
    original = list(os.sys.argv)
    diagnostic, remaining = parser.parse_known_args()
    os.sys.argv = [original[0]] + remaining
    try:
        args = get_args()
    finally:
        os.sys.argv = original
    args.task = "rotunbot_vel_sru50_v62_corridor_eval"
    args.corridor_checkpoint = diagnostic.smoke_checkpoint
    args.corridor_output_dir = diagnostic.smoke_output_dir
    args.num_envs = 1
    family = diagnostic.smoke_family
    output = os.path.join(diagnostic.smoke_output_dir, family.lower())
    run_corridor(
        args,
        _scenario_for_family(family, 20260829),
        episodes=1,
        output_dir=__import__("pathlib").Path(output),
        enforce_gate=False,
        max_steps=250,
    )
    print("V62 corridor smoke PASS", flush=True)


if __name__ == "__main__":
    main()
