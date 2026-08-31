"""Train V1 Depth Straight Corridor with direct SRU velocity actions."""

import argparse
import csv
import json
import os
import sys

import isaacgym  # noqa: F401 - must precede torch
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.dwl.actor_critic_direct_velocity import (
    load_direct_velocity_warm_start,
)
from legged_gym.navigation.corridor_artifacts import CheckpointMetadata
from legged_gym.utils import get_args, task_registry


def curriculum_history_row(
    iteration,
    level,
    current_distance,
    next_distance,
    current_summary,
    next_summary,
    gate,
):
    """Flatten one independent current/next evaluation into CSV columns."""
    return {
        "iteration": int(iteration),
        "current_level": int(level),
        "current_distance_m": float(current_distance),
        "next_distance_m": float(next_distance),
        "current_eval_success_rate": float(current_summary["success_rate"]),
        "next_eval_success_rate": float(next_summary["success_rate"]),
        "current_collision_rate": float(current_summary["collision_rate"]),
        "next_collision_rate": float(next_summary["collision_rate"]),
        "current_timeout_rate": float(current_summary["timeout_rate"]),
        "next_timeout_rate": float(next_summary["timeout_rate"]),
        "current_reverse_motion_ratio": float(
            current_summary["reverse_motion_ratio"]
        ),
        "next_reverse_motion_ratio": float(next_summary["reverse_motion_ratio"]),
        "current_final_distance_m": float(
            current_summary.get("mean_final_goal_distance_m", 0.0)
        ),
        "next_final_distance_m": float(
            next_summary.get("mean_final_goal_distance_m", 0.0)
        ),
        "gate_pass": bool(gate["pass"]),
    }


def _write_history_row(path, row):
    fields = list(row)
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def curriculum_state_payload(payload):
    """Normalize a plain curriculum JSON object for the environment API."""
    if not isinstance(payload, dict):
        raise ValueError("curriculum state must be a JSON object")
    if "v1_performance_curriculum" in payload:
        return payload
    return {"v1_performance_curriculum": payload}


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--resume_path", required=True)
    parser.add_argument("--parent_checkpoint", default=None)
    parser.add_argument("--disable_camera_noise", action="store_true")
    parser.add_argument("--curriculum-state", default=None)
    stage_args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + remaining
    try:
        args = get_args()
    finally:
        os.sys.argv = original
    if bool(getattr(args, "resume", False)):
        raise ValueError("V1 accepts only --resume_path model-only warm starts")

    args.task = "rotunbot_sru_visual_corridor_v1"
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    train_cfg.runner.resume = False
    if stage_args.iterations is not None:
        train_cfg.runner.max_iterations = int(stage_args.iterations)
    if stage_args.num_envs is not None:
        env_cfg.env.num_envs = int(stage_args.num_envs)
    if stage_args.disable_camera_noise:
        env_cfg.camera.add_noise = False
    env_cfg.commands.v1_goal_curriculum_enabled = False
    env_cfg.commands.v1_performance_curriculum_enabled = True

    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root="default",
    )
    warm_start = load_direct_velocity_warm_start(
        runner.alg.actor_critic,
        stage_args.resume_path,
        map_location=runner.device,
    )
    checkpoint_payload = torch.load(stage_args.resume_path, map_location="cpu")
    if isinstance(checkpoint_payload, dict):
        env.set_checkpoint_state(checkpoint_payload.get("env_state"))
    if stage_args.curriculum_state:
        with open(stage_args.curriculum_state, encoding="utf-8") as handle:
            env.set_checkpoint_state(curriculum_state_payload(json.load(handle)))
    print(
        "V1 model-only warm start loaded: {checkpoint} "
        "(migrated={migrated}, source_iteration={source_iteration})".format(**warm_start),
        flush=True,
    )
    target_iterations = int(train_cfg.runner.max_iterations)
    runner.learn(
        num_learning_iterations=target_iterations,
        init_at_random_ep_len=True,
    )
    parent_checkpoint = stage_args.parent_checkpoint or stage_args.resume_path
    if runner.log_dir is not None:
        curriculum_state = getattr(env, "v1_curriculum", None)
        if curriculum_state is not None:
            with open(
                os.path.join(runner.log_dir, "curriculum_state.json"), "w"
            ) as handle:
                json.dump(
                    curriculum_state.to_dict(), handle, indent=2, sort_keys=True
                )
        checkpoint = os.path.join(
            runner.log_dir, "model_{}.pt".format(runner.current_learning_iteration)
        )
        metadata = CheckpointMetadata.from_path(
            checkpoint,
            parent=parent_checkpoint,
            stage="V1",
            seed=train_cfg.seed,
            iterations=runner.current_learning_iteration,
        )
        with open(os.path.join(runner.log_dir, "checkpoint_metadata.json"), "w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        print("Checkpoint metadata: {}".format(os.path.join(runner.log_dir, "checkpoint_metadata.json")))


if __name__ == "__main__":
    main()
