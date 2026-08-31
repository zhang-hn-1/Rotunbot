"""Strict GPU smoke for the 2 Hz corridor Oracle -> Local Goal -> V62 stack."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import isaacgym  # noqa: F401 - must precede torch
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.navigation.corridor_scenarios import make_l_scenario
from legged_gym.navigation.corridor_waypoint_oracle import CorridorWaypointOracle
from legged_gym.navigation.direct_velocity_curriculum import configure_direct_velocity_stage
from legged_gym.navigation.v62_corridor_task import make_wall_segments
from legged_gym.planners.oracle_local_subgoal import CorridorWaypointAdapter
from legged_gym.utils import get_args, task_registry


PLANNER_HZ = 2.0
B3_DISTANCE_LIMIT_M = 2.0
B3_BEARING_LIMIT_DEG = 45.0
_SAFETY_COUNT_FIELDS = (
    "collision_count",
    "divergence_count",
    "feasible_domain_violation_count",
    "hidden_projection_jump_count",
    "rate_violation_count",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adjacent_gate_summary(checkpoint):
    candidate = checkpoint.parent / "summary.json"
    return candidate if candidate.is_file() else None


def validate_approved_s2b_checkpoint(checkpoint, gate_summary=None):
    """Return a verified gate artifact or reject every unapproved checkpoint."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise RuntimeError("GPU smoke requires a checkpoint file: %s" % checkpoint)
    summary = (
        Path(gate_summary).expanduser().resolve()
        if gate_summary is not None
        else _adjacent_gate_summary(checkpoint)
    )
    if summary is None or not summary.is_file():
        raise RuntimeError(
            "GPU smoke requires --smoke_gate_summary or an adjacent approved summary.json"
        )
    try:
        payload = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid S2B gate summary: %s" % summary) from error
    if payload.get("stage") != "S2B" or payload.get("gate", {}).get("stage") != "S2B":
        raise RuntimeError("gate summary is not an S2B artifact: %s" % summary)
    if payload.get("gate", {}).get("pass") is not True:
        raise RuntimeError("S2B gate did not pass: %s" % summary)
    try:
        artifact_checkpoint = Path(payload["checkpoint"]).expanduser().resolve()
    except (KeyError, TypeError) as error:
        raise RuntimeError("S2B gate summary lacks a valid checkpoint") from error
    if artifact_checkpoint != checkpoint:
        raise RuntimeError("gate checkpoint does not resolve to requested checkpoint")
    if payload.get("checkpoint_sha256") != _sha256(checkpoint):
        raise RuntimeError("gate checkpoint SHA256 does not match requested checkpoint")
    for field in _SAFETY_COUNT_FIELDS:
        if payload.get(field) != 0:
            raise RuntimeError("S2B gate safety count %s is not zero" % field)
    return summary


def _yaw_from_quaternion(quaternion):
    qx, qy, qz, qw = quaternion.unbind(dim=0)
    return float(torch.atan2(
        2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy.square() + qz.square())
    ).item())


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke_checkpoint", default=None)
    parser.add_argument("--smoke_gate_summary", default=None)
    parser.add_argument("--smoke_output_dir", default="logs/oracle_velocity_smoke")
    parser.add_argument("--smoke_max_steps", type=int, default=1200)
    original = list(os.sys.argv)
    diagnostic, remaining = parser.parse_known_args()
    os.sys.argv = [original[0]] + remaining
    try:
        args = get_args()
    finally:
        os.sys.argv = original
    if diagnostic.smoke_checkpoint is None:
        raise RuntimeError(
            "GPU smoke is blocked: no approved frozen B3 checkpoint is available; "
            "pass --smoke_checkpoint and its approved gate summary after B3 passes"
        )
    checkpoint = Path(diagnostic.smoke_checkpoint).expanduser().resolve()
    gate_summary = validate_approved_s2b_checkpoint(
        checkpoint, diagnostic.smoke_gate_summary
    )
    if diagnostic.smoke_max_steps < 1:
        raise ValueError("--smoke_max_steps must be positive")
    args.smoke_checkpoint = checkpoint
    args.smoke_gate_summary = gate_summary
    args.smoke_output_dir = Path(diagnostic.smoke_output_dir).expanduser().resolve()
    args.smoke_max_steps = int(diagnostic.smoke_max_steps)
    args.task = "rotunbot_sru_direct_velocity"
    args.num_envs = 1
    return args


def _configure_env(env_cfg, scenario, max_steps):
    configure_direct_velocity_stage(env_cfg, "S2B")
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = max_steps * 0.005
    env_cfg.noise.add_noise = False
    env_cfg.camera.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.commands.random_start_yaw = False
    env_cfg.corridor_wall_width_m = scenario.width_m
    env_cfg.corridor_wall_segments = make_wall_segments(scenario.centerline)


def _validate_waypoint(pose, waypoint):
    if waypoint.shape != (2,) or not np.isfinite(waypoint).all():
        raise AssertionError("Oracle did not produce finite world XY waypoint")
    delta = waypoint - pose[:2]
    distance = float(np.linalg.norm(delta))
    bearing = math.atan2(float(delta[1]), float(delta[0])) - float(pose[2])
    bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
    if distance > B3_DISTANCE_LIMIT_M + 1.0e-6:
        raise AssertionError("Oracle waypoint exceeds B3 local distance capability")
    if abs(bearing) > math.radians(B3_BEARING_LIMIT_DEG) + 1.0e-6:
        raise AssertionError("Oracle waypoint exceeds B3 bearing capability")


def main():
    args = _parse_args()
    scenario = make_l_scenario(2.0, 3.0, 2.0, seed=20260831)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _configure_env(env_cfg, scenario, args.smoke_max_steps)
    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    runner.load(str(args.smoke_checkpoint))
    policy = runner.get_inference_policy(device=env.device)
    adapter = CorridorWaypointAdapter(CorridorWaypointOracle(
        scenario,
        local_distance_limit=B3_DISTANCE_LIMIT_M,
        bearing_limit_deg=B3_BEARING_LIMIT_DEG,
    ))
    if hasattr(adapter, "command") or hasattr(adapter, "actuator_command"):
        raise AssertionError("corridor adapter must not expose actuator commands")

    planner_interval = int(round((1.0 / PLANNER_HZ) / float(env.dt)))
    if planner_interval < 1:
        raise AssertionError("2 Hz planner interval is invalid")
    obs, _ = env.reset()
    completed = False
    planner_ticks = 0
    try:
        with torch.no_grad():
            for step in range(args.smoke_max_steps):
                root = env.root_states[0]
                origin = env.env_origins[0, :2]
                pose = np.asarray((
                    float((root[0] - origin[0]).item()),
                    float((root[1] - origin[1]).item()),
                    _yaw_from_quaternion(env.base_quat[0]),
                ))
                if not np.isfinite(pose).all():
                    raise AssertionError("V62/local-goal state became non-finite")
                if step % planner_interval == 0:
                    waypoint = adapter.next_waypoint(pose)
                    _validate_waypoint(pose, waypoint)
                    env.global_goal_xy_world[0] = torch.as_tensor(
                        waypoint, dtype=env.root_states.dtype, device=env.device
                    ) + origin
                    env.compute_observations()
                    planner_ticks += 1
                actions = policy(obs)
                if actions.shape[-1] != 2:
                    raise AssertionError("Local Goal policy must emit two velocity-port actions")
                obs, _, _, dones, _ = env.step(actions)
                if not torch.isfinite(env.root_states).all():
                    raise AssertionError("V62 state became non-finite")
                if bool(dones[0].item()):
                    completed = True
                    break
    finally:
        env.gym.destroy_sim(env.sim)

    if not completed:
        raise AssertionError("smoke episode did not complete")
    if planner_ticks < 1:
        raise AssertionError("2 Hz planner did not tick")
    args.smoke_output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "completed_episode": completed,
        "planner_hz": PLANNER_HZ,
        "planner_ticks": planner_ticks,
        "local_distance_limit_m": B3_DISTANCE_LIMIT_M,
        "bearing_limit_deg": B3_BEARING_LIMIT_DEG,
        "actuator_output": False,
    }
    (args.smoke_output_dir / "oracle_velocity_smoke.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    print("Oracle -> Local Goal -> V62 smoke PASS", flush=True)


if __name__ == "__main__":
    main()
