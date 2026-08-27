"""Fixed local-depth evaluation helpers and a JSON-record CLI."""

import argparse
import json
import os
from pathlib import Path
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


def goal_y_bin(abs_goal_y):
    """Return the fixed lateral-goal bucket used by the Stage0 report."""
    value = abs(float(abs_goal_y))
    if value < 0.2:
        return "<0.2"
    if value < 0.4:
        return "0.2-0.4"
    if value <= 0.6:
        return "0.4-0.6"
    return ">0.6"


def aggregate_records(records, formal_camera=False):
    records = list(records)
    if not records:
        raise ValueError("at least one evaluation record is required")
    validate_backend(records, formal_camera=formal_camera)

    def mean(key, default=0.0):
        return sum(float(record.get(key, default)) for record in records) / len(records)

    summary = {
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
    buckets = ("<0.2", "0.2-0.4", "0.4-0.6", ">0.6")
    summary["local_goal_y_bins"] = {}
    for bucket in buckets:
        bucket_records = [
            record for record in records
            if "abs_goal_y" in record and goal_y_bin(record["abs_goal_y"]) == bucket
        ]
        if not bucket_records:
            summary["local_goal_y_bins"][bucket] = {
                "episodes": 0,
                "local_success_rate": 0.0,
                "final_distance_mean": 0.0,
                "completion_time_mean": 0.0,
            }
            continue
        summary["local_goal_y_bins"][bucket] = {
            "episodes": len(bucket_records),
            "local_success_rate": sum(
                float(record.get("local_success", 0)) for record in bucket_records
            ) / len(bucket_records),
            "final_distance_mean": sum(
                float(record.get("final_distance", 0.0)) for record in bucket_records
            ) / len(bucket_records),
            "completion_time_mean": sum(
                float(record.get("completion_time", 0.0)) for record in bucket_records
            ) / len(bucket_records),
        }
    return summary


def side_obstacle_observability(depth, edge_fraction=0.2, far_threshold=0.95):
    """Return the fraction of edge pixels that are not far/open-space values."""
    if depth.ndim != 3 or depth.shape[1] != 8:
        raise ValueError("depth must have shape [N, 8, W]")
    width = depth.shape[2]
    edge = max(1, int(width * float(edge_fraction)))
    side = depth[:, :, :edge].reshape(-1)
    return float((side < float(far_threshold)).float().mean().item())


def evaluate_checkpoint(checkpoint, episodes=20, depth_backend="fallback", stage=0,
                        report_path=None, headless=True):
    """Run policy rollouts and save per-episode plus aggregate Stage metrics."""
    import isaacgym  # noqa: F401 - must precede torch in Isaac Gym Preview 4
    import numpy as np

    if not hasattr(np, "float"):
        np.float = float
    import torch

    import legged_gym.envs  # noqa: F401 - registration side effects
    from legged_gym.utils import get_args, task_registry

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if depth_backend not in ("fallback", "isaacgym"):
        raise ValueError("depth_backend must be fallback or isaacgym")

    old_argv = sys.argv
    sys.argv = [old_argv[0]] + (["--headless"] if headless else [])
    try:
        args = get_args()
    finally:
        sys.argv = old_argv
    args.task = "rotunbot_maze_local_depth"
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.camera.depth_backend = depth_backend
    env_cfg.camera.add_noise = False
    env_cfg.camera.dropout_probability = 0.0
    env_cfg.camera.quantization = 0.0
    env_cfg.commands.local_curriculum_stage = int(stage)
    if int(stage) == 1:
        env_cfg.maze.scene_mode = "corridor"
        env_cfg.maze.enabled = False
    env_cfg.enable_camera_sensors_in_headless = depth_backend == "isaacgym"
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    train_cfg.runner.resume = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.data_print = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.device)
    records = []
    obs, _ = env.reset()
    print(
        f"Depth-local evaluation checkpoint={checkpoint} episodes={episodes} "
        f"backend={depth_backend} stage={stage}", flush=True,
    )
    try:
        with torch.no_grad():
            for episode in range(1, int(episodes) + 1):
                abs_goal_y = abs(float(env.active_local_goal_xy_robot[0, 1].item()))
                path_length = 0.0
                steps = 0
                old_position = env.root_states[0, :2].detach().clone()
                while True:
                    actions = policy(obs)
                    obs, _, _, dones, _ = env.step(actions)
                    steps += 1
                    done = bool(dones[0].item())
                    if done:
                        terminal_position = env.terminal_position[0]
                        path_length += float(torch.linalg.vector_norm(
                            terminal_position - old_position
                        ).item())
                        record = {
                            "episode": episode,
                            "local_success": int(env.terminal_local_success[0].item()),
                            "global_success": int(env.terminal_global_success[0].item()),
                            "collision": int(env.terminal_collision[0].item()),
                            "timeout": int(env.terminal_timeout[0].item()),
                            "waypoint_reach_count": int(env.terminal_waypoint_reach_count[0].item()),
                            "final_distance": float(env.terminal_goal_distance[0].item()),
                            "final_local_distance": float(env.terminal_local_goal_distance[0].item()),
                            "path_length": path_length,
                            "completion_time": steps * float(env.dt),
                            "steps": steps,
                            "abs_goal_y": abs_goal_y,
                            "depth_backend_requested": env.depth_backend_requested,
                            "depth_backend_actual": env.depth_backend_actual,
                        }
                        records.append(record)
                        if episode <= 3 or episode % 10 == 0:
                            print(
                                f"episode={episode} local_success={record['local_success']} "
                                f"global_success={record['global_success']} "
                                f"final_distance={record['final_distance']:.3f} "
                                f"|goal_y|={abs_goal_y:.3f}", flush=True,
                            )
                        break
                    new_position = env.root_states[0, :2].detach().clone()
                    path_length += float(torch.linalg.vector_norm(new_position - old_position).item())
                    old_position = new_position
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)

    report = {
        "checkpoint": str(checkpoint),
        "stage": int(stage),
        "depth_backend_requested": depth_backend,
        "records": records,
        "summary": aggregate_records(records, formal_camera=depth_backend == "isaacgym"),
    }
    if report_path is None:
        report_path = os.environ.get(
            "DEPTH_LOCAL_EVAL_REPORT", "logs/rotunbot_maze_local_depth_eval.json"
        )
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", help="JSON list of per-episode records")
    parser.add_argument("--checkpoint", help="run checkpoint rollouts instead of aggregating records")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--depth-backend", choices=("fallback", "isaacgym"), default="fallback")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--report", default=None)
    parser.add_argument("--formal-camera", action="store_true")
    args = parser.parse_args(argv)
    if args.checkpoint:
        result = evaluate_checkpoint(
            args.checkpoint,
            episodes=args.episodes,
            depth_backend=args.depth_backend,
            stage=args.stage,
            report_path=args.report,
        )
        print(json.dumps(result["summary"], indent=2))
        return 0
    if not args.records:
        parser.error("one of --records or --checkpoint is required")
    with open(args.records, "r", encoding="utf-8") as stream:
        records = json.load(stream)
    print(json.dumps(aggregate_records(records, formal_camera=args.formal_camera), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
