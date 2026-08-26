"""Measure fixed-action motion of the frozen P2P simulator without training."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from legged_gym.navigation.baseline import CHECKPOINT_RELATIVE_PATH
from legged_gym.navigation.frozen_p2p import load_frozen_p2p, robot_pose, robot_speed
from legged_gym.navigation.reachability import (
    ReachabilityEnvelope,
    ReachabilitySample,
    save_envelope,
    save_samples,
)


BEARING_GROUPS_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0)
LOCAL_REACH_DISTANCE_M = 0.35


def _bearing_error_deg(first, second):
    return abs(float(np.degrees(np.angle(
        np.exp(1j * np.deg2rad(float(first) - float(second)))
    ))))


def summarize_bearing_dependence(samples, coverage_summary=None):
    """Combine measured displacement bearings with optional goal coverage."""
    rows = {}
    coverage_rows = {
        (float(row["distance_m"]), float(row["bearing_deg"])): row
        for row in (coverage_summary or {}).get("case_summaries", [])
    }
    for bearing in BEARING_GROUPS_DEG:
        selected = []
        for sample in samples:
            displacement = np.asarray(sample.displacement_body_xy, dtype=np.float64)
            radius = float(np.linalg.norm(displacement))
            if radius <= 1.0e-12:
                continue
            sample_bearing = float(np.degrees(np.arctan2(displacement[1], displacement[0])))
            nearest = min(
                BEARING_GROUPS_DEG,
                key=lambda target: _bearing_error_deg(sample_bearing, target),
            )
            if nearest == bearing:
                selected.append(radius)
        matching_coverage = [
            row for (distance, row_bearing), row in coverage_rows.items()
            if _bearing_error_deg(row_bearing, bearing) <= 1.0e-9
        ]
        coverage_episodes = sum(int(row["episodes"]) for row in matching_coverage)
        coverage_successes = sum(
            float(row["success_rate"]) * int(row["episodes"])
            for row in matching_coverage
        )
        coverage_timeouts = sum(int(row["timeout_count"]) for row in matching_coverage)
        rows[str(int(bearing))] = {
            "bearing_deg": bearing,
            "measured_action_count": len(selected),
            "reachable_action_count": int(sum(radius >= LOCAL_REACH_DISTANCE_M for radius in selected)),
            "reachable_action_rate": (
                float(np.mean([radius >= LOCAL_REACH_DISTANCE_M for radius in selected]))
                if selected else 0.0
            ),
            "max_displacement_m": max(selected, default=0.0),
            "mean_displacement_m": float(np.mean(selected)) if selected else 0.0,
            "coverage_episode_count": coverage_episodes,
            "coverage_success_rate": coverage_successes / coverage_episodes if coverage_episodes else 0.0,
            "coverage_timeout_rate": coverage_timeouts / coverage_episodes if coverage_episodes else 0.0,
        }
    return rows


def _parse_script_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT_RELATIVE_PATH)
    parser.add_argument("--output-dir", default="logs/hierarchical_navigation/reachability")
    parser.add_argument("--duration-steps", type=int, default=250)
    parser.add_argument("--angular-bins", type=int, default=16)
    parser.add_argument("--coverage-summary")
    return parser.parse_args()


def _isaac_args():
    from legged_gym.navigation.isaac_compat import install_isaac_gym_compat

    install_isaac_gym_compat()
    from legged_gym import envs  # noqa: F401
    from legged_gym.utils import get_args

    saved = sys.argv
    sys.argv = [saved[0], "--headless", "--rl_device=cuda:0", "--sim_device=cuda:0"]
    try:
        return get_args()
    finally:
        sys.argv = saved


def _measure_one(env, action0, action1, duration_steps):
    import torch

    env.reset()
    start_xy, start_yaw = robot_pose(env)
    positions = []
    speeds = []
    body_velocities = []
    action = torch.tensor([[action0, action1]], dtype=torch.float32, device=env.device)
    clipped = False
    for _ in range(int(duration_steps)):
        _obs, _privileged, _reward, dones, _infos = env.step(action)
        position, _ = robot_pose(env)
        positions.append(position)
        speeds.append(robot_speed(env))
        if hasattr(env, "base_lin_vel"):
            body_velocity = env.base_lin_vel[0, :2].detach().cpu().numpy()
        else:
            _, yaw = robot_pose(env)
            world_velocity = env.root_states[0, 7:9].detach().cpu().numpy()
            cosine, sine = np.cos(yaw), np.sin(yaw)
            body_velocity = np.array([
                cosine * world_velocity[0] + sine * world_velocity[1],
                -sine * world_velocity[0] + cosine * world_velocity[1],
            ])
        body_velocities.append(body_velocity)
        if hasattr(env, "actions"):
            applied = env.actions[0].detach().cpu().numpy()
            clipped = clipped or bool(np.any(np.abs(applied - action[0].cpu().numpy()) > 1.0e-6))
        if bool(dones[0].item()):
            break
    displacement_world = np.asarray(positions[-1]) - start_xy if positions else np.zeros(2)
    cosine, sine = np.cos(start_yaw), np.sin(start_yaw)
    inverse_rotation = np.array([[cosine, sine], [-sine, cosine]])
    displacement_body = inverse_rotation.dot(displacement_world)
    tail = np.asarray(speeds[-min(20, len(speeds)):]) if speeds else np.zeros(1)
    steady_speed = float(np.mean(tail))
    velocity_tail = (
        np.asarray(body_velocities[-min(20, len(body_velocities)):])
        if body_velocities else np.zeros((1, 2))
    )
    steady_velocity_body = np.mean(velocity_tail, axis=0)
    forward = abs(float(displacement_body[0]))
    lateral = abs(float(displacement_body[1]))
    coupling = lateral / max(forward, 1.0e-6)
    rise_time = float(next((index * float(env.dt) for index, value in enumerate(speeds) if value >= 0.9 * steady_speed), 0.0))
    joint_response = env.dof_pos[0].detach().cpu().numpy()[:2] if hasattr(env, "dof_pos") else np.zeros(2)
    return ReachabilitySample(
        action0=action0,
        action1=action1,
        displacement_body_xy=displacement_body,
        steady_state_velocity_body_xy=steady_velocity_body,
        rise_time_s=rise_time,
        cross_axis_coupling=coupling,
        action_clipping=clipped,
        joint_response=joint_response,
    )


def run(args, script_args):
    values = (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)
    env, _runner, _policy = load_frozen_p2p(args, script_args.checkpoint)
    output_dir = Path(script_args.output_dir)
    samples = []
    try:
        for action0 in values:
            for action1 in values:
                samples.append(_measure_one(env, action0, action1, script_args.duration_steps))
    finally:
        env.gym.destroy_sim(env.sim)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_samples(output_dir / "raw_samples.json", samples)
    envelope = ReachabilityEnvelope.from_samples(samples, script_args.angular_bins)
    save_envelope(output_dir / "envelope.json", envelope)
    coverage_summary = (
        json.loads(Path(script_args.coverage_summary).read_text(encoding="utf-8"))
        if script_args.coverage_summary else None
    )
    bearing_summary = summarize_bearing_dependence(samples, coverage_summary)
    (output_dir / "bearing_summary.json").write_text(
        json.dumps(bearing_summary, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(json.dumps({
        "sample_count": len(samples),
        "coverage_summary": script_args.coverage_summary,
        "bearing_summary": bearing_summary,
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(samples)} raw reachability samples to {output_dir}", flush=True)
    return envelope


if __name__ == "__main__":
    script_args = _parse_script_args()
    run(_isaac_args(), script_args)
