"""Fixed local-depth evaluation helpers and a JSON-record CLI."""

import argparse
import json
import sys


def validate_backend(records, formal_camera=False):
    """Reject fallback records when the formal IMAGE_DEPTH path is requested."""
    if not formal_camera:
        return
    invalid = [
        record for record in records
        if record.get("depth_backend_requested") != "isaacgym"
        or record.get("depth_backend_actual") != "isaacgym"
    ]
    if invalid:
        raise ValueError("formal camera evaluation requires actual Isaac Gym IMAGE_DEPTH")


def aggregate_records(records, formal_camera=False):
    records = list(records)
    if not records:
        raise ValueError("at least one evaluation record is required")
    validate_backend(records, formal_camera=formal_camera)

    def mean(key, default=0.0):
        return sum(float(record.get(key, default)) for record in records) / len(records)

    return {
        "episodes": len(records),
        "local_success_rate": mean("local_success"),
        "global_success_rate": mean("global_success"),
        "collision_rate": mean("collision"),
        "timeout_rate": mean("timeout"),
        "waypoint_reach_count": sum(int(record.get("waypoint_reach_count", 0)) for record in records),
        "final_distance_mean": mean("final_distance"),
        "path_length_mean": mean("path_length"),
        "completion_time_mean": mean("completion_time"),
        "depth_backend_requested": sorted({record.get("depth_backend_requested") for record in records}),
        "depth_backend_actual": sorted({record.get("depth_backend_actual") for record in records}),
    }


def side_obstacle_observability(depth, edge_fraction=0.2, far_threshold=0.95):
    """Return the fraction of edge pixels that are not far/open-space values."""
    if depth.ndim != 3 or depth.shape[1] != 8:
        raise ValueError("depth must have shape [N, 8, W]")
    width = depth.shape[2]
    edge = max(1, int(width * float(edge_fraction)))
    side = depth[:, :, :edge].reshape(-1)
    return float((side < float(far_threshold)).float().mean().item())


def _main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, help="JSON list of per-episode records")
    parser.add_argument("--formal-camera", action="store_true")
    args = parser.parse_args(argv)
    with open(args.records, "r", encoding="utf-8") as stream:
        records = json.load(stream)
    print(json.dumps(aggregate_records(records, formal_camera=args.formal_camera), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
