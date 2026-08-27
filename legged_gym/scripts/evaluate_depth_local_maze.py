"""Full-maze evaluator entry point using planner-produced record data."""

import argparse
import json
import os
from pathlib import Path
import sys

from legged_gym.scripts.evaluate_depth_local import aggregate_records


def evaluate_maze_records(records, formal_camera=False):
    """Aggregate fixed-maze records; BFS remains outside the actor/environment."""
    return aggregate_records(records, formal_camera=formal_camera)


def evaluate_maze_checkpoint(checkpoint, episodes=3, depth_backend="fallback", report_path=None):
    """Run the real Stage4 BFS -> local waypoint -> actor rollout chain."""
    import isaacgym  # noqa: F401 - must precede torch in Isaac Gym Preview 4
    import numpy as np

    if not hasattr(np, "float"):
        np.float = float
    import torch

    import legged_gym.envs  # noqa: F401 - registration side effects
    from legged_gym.planners import OracleLocalSubgoalPlanner
    from legged_gym.utils import get_args, task_registry

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if depth_backend not in ("fallback", "isaacgym"):
        raise ValueError("depth_backend must be fallback or isaacgym")

    old_argv = sys.argv
    sys.argv = [old_argv[0], "--headless"]
    try:
        args = get_args()
    finally:
        sys.argv = old_argv
    args.task = "rotunbot_maze_local_depth"
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.maze.enabled = True
    env_cfg.maze.scene_mode = "none"
    env_cfg.commands.local_curriculum_stage = 4
    env_cfg.commands.random_start_yaw = False
    env_cfg.camera.depth_backend = depth_backend
    env_cfg.camera.add_noise = False
    env_cfg.camera.dropout_probability = 0.0
    env_cfg.camera.quantization = 0.0
    env_cfg.enable_camera_sensors_in_headless = depth_backend == "isaacgym"
    env_cfg.noise.add_noise = False
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
    obs, _ = env.reset()
    records = []
    planner = OracleLocalSubgoalPlanner(
        env.maze_layout, cell_size=env.cfg.maze.cell_size, lookahead_cells=1
    )

    def latch_next_waypoint():
        position = (env.root_states[0, :2] - env.env_origins[0, :2]).detach().cpu().numpy()
        goal = (env.global_goal_xy_world[0] - env.env_origins[0, :2]).detach().cpu().numpy()
        waypoint_local, path = planner.plan(position, goal)
        waypoint_world = torch.as_tensor(
            waypoint_local, dtype=torch.float32, device=env.device
        ).view(1, 2) + env.env_origins[0:1, :2]
        env.set_active_waypoint(waypoint_world)
        return path

    try:
        print(
            f"Stage4 maze evaluation checkpoint={checkpoint} episodes={episodes} "
            f"backend={depth_backend}", flush=True,
        )
        with torch.no_grad():
            for episode in range(1, int(episodes) + 1):
                path_cells = latch_next_waypoint()
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
                        records.append({
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
                            "planned_path_cells": len(path_cells),
                            "depth_backend_requested": env.depth_backend_requested,
                            "depth_backend_actual": env.depth_backend_actual,
                        })
                        print(
                            f"episode={episode} global_success={records[-1]['global_success']} "
                            f"waypoints={records[-1]['waypoint_reach_count']} "
                            f"collision={records[-1]['collision']}", flush=True,
                        )
                        break
                    new_position = env.root_states[0, :2].detach().clone()
                    path_length += float(torch.linalg.vector_norm(new_position - old_position).item())
                    old_position = new_position
                    if bool(env.needs_new_waypoint[0].item()):
                        path_cells = latch_next_waypoint()
                        obs = env.obs_buf
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)

    report = {
        "checkpoint": str(checkpoint),
        "stage": 4,
        "depth_backend_requested": depth_backend,
        "records": records,
        "summary": evaluate_maze_records(
            records, formal_camera=depth_backend == "isaacgym"
        ),
    }
    if report_path is None:
        report_path = os.environ.get(
            "DEPTH_LOCAL_MAZE_EVAL_REPORT",
            "logs/rotunbot_maze_local_depth_stage4_eval.json",
        )
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--records")
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--depth-backend", choices=("fallback", "isaacgym"), default="fallback")
    parser.add_argument("--report", default=None)
    parser.add_argument("--formal-camera", action="store_true")
    args = parser.parse_args(argv)
    if args.checkpoint:
        result = evaluate_maze_checkpoint(
            args.checkpoint,
            episodes=args.episodes,
            depth_backend=args.depth_backend,
            report_path=args.report,
        )
        print(json.dumps(result["summary"], indent=2))
        return 0
    if not args.records:
        parser.error("one of --records or --checkpoint is required")
    with open(args.records, "r", encoding="utf-8") as stream:
        records = json.load(stream)
    print(json.dumps(evaluate_maze_records(records, args.formal_camera), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
