"""Full-maze evaluator entry point using planner-produced record data."""

import argparse
import json
import sys

from legged_gym.scripts.evaluate_depth_local import aggregate_records


def evaluate_maze_records(records, formal_camera=False):
    """Aggregate fixed-maze records; BFS remains outside the actor/environment."""
    return aggregate_records(records, formal_camera=formal_camera)


def _main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--formal-camera", action="store_true")
    args = parser.parse_args(argv)
    with open(args.records, "r", encoding="utf-8") as stream:
        records = json.load(stream)
    print(json.dumps(evaluate_maze_records(records, args.formal_camera), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
