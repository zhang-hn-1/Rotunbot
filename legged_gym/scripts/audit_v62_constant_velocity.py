"""D0-A calibration of Frozen V62 under constant velocity requests."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

from legged_gym.navigation.phase_d_contracts import require_isaacgym_depth, resolve_phase_d_timing
from legged_gym.navigation.phase_d_diagnostics import command_loss_breakdown, classify_constant_command_gate


COMMANDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
YAWS_DEG = (-20.0, 0.0, 20.0)


def summarize_command_trace(rows):
    """Compute command-domain and tracking ratios from per-step rows."""
    rows = list(rows)
    if not rows:
        return {
            "sample_count": 0,
            "mean_requested_v_mps": None,
            "mean_projected_v_mps": None,
            "mean_applied_v_mps": None,
            "mean_actual_v_mps": None,
            "p50_tracking_error_v_mps": None,
            "p90_tracking_error_v_mps": None,
            "actual_over_requested": None,
            "actual_over_applied": None,
        }
    requested = np.asarray([abs(float(r["requested_v_mps"])) for r in rows], dtype=np.float64)
    projected = np.asarray([abs(float(r["projected_v_mps"])) for r in rows], dtype=np.float64)
    applied = np.asarray([abs(float(r["applied_v_mps"])) for r in rows], dtype=np.float64)
    actual = np.asarray([abs(float(r["actual_v_mps"])) for r in rows], dtype=np.float64)
    error = np.asarray([abs(float(r["tracking_error_v_mps"])) for r in rows], dtype=np.float64)
    return {
        "sample_count": int(len(rows)),
        "mean_requested_v_mps": float(requested.mean()),
        "mean_projected_v_mps": float(projected.mean()),
        "mean_applied_v_mps": float(applied.mean()),
        "mean_actual_v_mps": float(actual.mean()),
        "p50_tracking_error_v_mps": float(np.percentile(error, 50.0)),
        "p90_tracking_error_v_mps": float(np.percentile(error, 90.0)),
        "actual_over_requested": float(actual.mean() / max(requested.mean(), 1.0e-9)),
        "actual_over_applied": float(actual.mean() / max(applied.mean(), 1.0e-9)),
    }


def _framework_args(values):
    from legged_gym.utils import get_args
    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + list(values)
    try:
        return get_args()
    finally:
        os.sys.argv = original


def _set_pose(env, yaw_rad, torch):
    state = env.root_states[0]
    state[0] = env.env_origins[0, 0] + 1.5
    state[1] = env.env_origins[0, 1]
    state[3:7] = torch.as_tensor(
        (0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)),
        dtype=state.dtype, device=state.device,
    )
    state[7:13] = 0.0
    from isaacgym import gymtorch
    actor_id = env._robot_actor_ids(torch.zeros(1, dtype=torch.long, device=env.device))
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim, gymtorch.unwrap_tensor(env._all_root_states),
        gymtorch.unwrap_tensor(actor_id.to(dtype=torch.int32)), 1,
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.base_quat[0] = env.root_states[0, 3:7]
    env.base_lin_vel[0] = env.root_states[0, 7:10]
    env.base_ang_vel[0] = env.root_states[0, 10:13]
    if hasattr(env, "tracking_heading"):
        env.tracking_heading[0] = float(yaw_rad)
        env._update_tracking_motion(integrate_heading=False)


def _make_env(framework_args, yaw_rad):
    import torch
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils import task_registry
    args = _framework_args(framework_args)
    args.task = "rotunbot_sru_visual_corridor_v1"
    cfg, _ = task_registry.get_cfgs(args.task)
    cfg.env.num_envs = 1
    cfg.env.episode_length_s = 30.0
    cfg.enable_camera_sensors_in_headless = True
    cfg.camera.depth_backend = "isaacgym"
    cfg.camera.add_noise = False
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    cfg.init_state.randomize_initial_velocity = False
    cfg.init_state.random_start_lateral = 0.0
    cfg.init_state.random_start_yaw = 0.0
    cfg.commands.v1_goal_curriculum_enabled = False
    cfg.commands.v1_performance_curriculum_enabled = False
    cfg.commands.resample_commands = False
    cfg.commands.smooth_profile_fraction = 0.0
    cfg.commands.random_walk_profile_fraction = 0.0
    cfg.commands.independent_smooth_profile_fraction = 0.0
    cfg.corridor_wall_segments = ()
    cfg.corridor_explicit_wall_segments = ()
    cfg.direct_obstacle_aabbs = ()
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    # BaseTask starts with reset_buf=1.  Keep reset, pose/goal setup and the
    # first observation in one inference scope: the legacy Torch 1.10 runtime
    # can otherwise mark buffers created during reset as inference tensors and
    # reject the next ordinary in-place observation update.
    with torch.inference_mode():
        env.reset()
        _set_pose(env, yaw_rad, torch=torch)
        env.reset_buf.zero_()
        env.time_out_buf.zero_()
        env.episode_length_buf.zero_()
        goal = env.root_states[0, :2].clone()
        goal[0] += 20.0
        env.global_goal_xy_world[0] = goal
        env.goal_dist[0] = torch.linalg.vector_norm(goal - env.root_states[0, :2])
        env.previous_goal_distance[0] = env.goal_dist[0]
        env.compute_observations()
        require_isaacgym_depth(env.depth_backend_requested, env.depth_backend_actual)
    return env, cfg


def _close(env):
    if env is None:
        return
    if getattr(env, "viewer", None) is not None:
        env.gym.destroy_viewer(env.viewer)
    if getattr(env, "sim", None) is not None:
        env.gym.destroy_sim(env.sim)


def run_case(requested_v, yaw_deg, duration_s=10.0, framework_args=()):
    import torch
    from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import RotunbotVel
    env = None
    try:
        env, cfg = _make_env(framework_args, math.radians(float(yaw_deg)))
        timing = resolve_phase_d_timing(
            env.sim_params.dt, env.cfg.control.decimation,
            env.cfg.commands.upper_level_command_frequency_hz,
        )
        requested = torch.tensor([[float(requested_v), 0.0]], dtype=env.command_targets.dtype, device=env.device)
        with torch.inference_mode():
            env.set_command_targets(requested)
        projected = env.command_targets[0].detach().cpu().numpy().copy()
        rows = []
        start = env.root_states[0, :2].detach().cpu().numpy().copy()
        previous = start.copy()
        steps = int(round(float(duration_s) / timing.policy_dt_s))
        with torch.inference_mode():
            for step in range(steps):
                _, _, _, dones, _ = RotunbotVel.step(env, torch.zeros_like(requested))
                position = (env.root_states[0, :2] - env.env_origins[0, :2]).detach().cpu().numpy().copy()
                applied = env.applied_feasible_command[0]
                actual_v = float(env.tracking_lin_vel[0, 0].item())
                actual_w = float(env.tracking_ang_vel[0, 2].item())
                rows.append({
                    "step": step + 1,
                    "time_s": float((step + 1) * timing.policy_dt_s),
                    "requested_v_mps": float(requested_v),
                    "requested_w_rps": 0.0,
                    "projected_v_mps": float(projected[0]),
                    "projected_w_rps": float(projected[1]),
                    "command_target_v_mps": float(env.command_targets[0, 0].item()),
                    "command_target_w_rps": float(env.command_targets[0, 1].item()),
                    "applied_v_mps": float(applied[0].item()),
                    "applied_w_rps": float(applied[1].item()),
                    "actual_v_mps": actual_v,
                    "actual_w_rps": actual_w,
                    "tracking_error_v_mps": float(applied[0].item()) - actual_v,
                    "tracking_error_w_rps": float(applied[1].item()) - actual_w,
                    "transition_state": int(env.transition_state[0].item()),
                    "transition_active": int(env.transition_active[0].item()),
                    "nominal_action_1": float(env.nominal_policy_actions[0, 0].item()),
                    "nominal_action_2": float(env.nominal_policy_actions[0, 1].item()),
                    "feedback_action_1": float(env.feedback_policy_actions[0, 0].item()),
                    "feedback_action_2": float(env.feedback_policy_actions[0, 1].item()),
                    "combined_action_1": float(env.combined_policy_actions[0, 0].item()),
                    "combined_action_2": float(env.combined_policy_actions[0, 1].item()),
                    "output_action_1": float(env.output_actions[0, 0].item()),
                    "output_action_2": float(env.output_actions[0, 1].item()),
                    "torque_norm": float(torch.linalg.vector_norm(env.torques[0]).item()),
                })
                previous = position
                if bool(dones.flatten()[0].item()):
                    break
        summary = summarize_command_trace(rows)
        summary.update({
            "requested_v_mps": float(requested_v),
            "requested_w_rps": 0.0,
            "yaw_deg": float(yaw_deg),
            "projected_v_mps": float(projected[0]),
            "projected_w_rps": float(projected[1]),
            "distance_travelled_m": float(np.linalg.norm((env.root_states[0, :2] - env.env_origins[0, :2]).detach().cpu().numpy() - start)),
            "transition_state_counts": {str(state): sum(int(row["transition_state"] == state) for row in rows) for state in range(4)},
            "timing": timing.to_dict(),
            "depth_backend_requested": env.depth_backend_requested,
            "depth_backend_actual": env.depth_backend_actual,
        })
        return summary, rows
    finally:
        _close(env)


def _worker(output, requested_v, yaw_deg, duration_s, framework_args):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {"requested_v_mps": requested_v, "yaw_deg": yaw_deg}
    try:
        summary, rows = run_case(requested_v, yaw_deg, duration_s, framework_args)
        result["summary"] = summary
        with (Path(output) / "trace.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    except Exception as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def run_audit(output, duration_s=10.0, framework_args=()):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    cases = []
    script = str(Path(__file__).resolve())
    for requested_v in COMMANDS:
        for yaw_deg in YAWS_DEG:
            case_dir = output / ("v_%0.2f_yaw_%+03d" % (requested_v, int(yaw_deg)))
            case_dir.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, script, "--worker", str(case_dir), str(requested_v), str(yaw_deg), str(duration_s)] + list(framework_args)
            completed = subprocess.run(command, cwd=str(Path(__file__).resolve().parents[2]), text=True, capture_output=True)
            result_path = case_dir / "result.json"
            result = json.loads(result_path.read_text()) if result_path.is_file() else {"error": completed.stderr[-4000:]}
            cases.append(result)
    failures = [case for case in cases if "error" in case]
    summaries = [case.get("summary", {}) for case in cases if "summary" in case]
    evidence_failures = []
    for summary in summaries:
        case_name = "v_%0.2f_yaw_%+03d" % (summary.get("requested_v_mps", 0.0), int(summary.get("yaw_deg", 0.0)))
        if summary.get("sample_count", 0) < int(round(duration_s / summary.get("timing", {}).get("policy_dt_s", 0.02))) * 0.95:
            evidence_failures.append(case_name + ":short_trace")
        if summary.get("depth_backend_requested") != "isaacgym" or summary.get("depth_backend_actual") != "isaacgym":
            evidence_failures.append(case_name + ":depth_backend")
        if summary.get("actual_over_applied") is None or not (0.60 <= summary["actual_over_applied"] <= 1.20):
            evidence_failures.append(case_name + ":tracking_ratio")
    report = {"D0_A": "PASS" if not failures and not evidence_failures else "FAIL", "duration_s": duration_s, "cases": cases, "errors": failures, "evidence_failures": evidence_failures}
    (output / "v62_constant_command_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with (output / "v62_constant_command_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [case.get("summary", {}) for case in cases if "summary" in case]
        if rows:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (output / "v62_constant_command_audit.md").write_text(
        "# D0-A Frozen V62 Constant Command Audit\n\n`D0_A = %s`\n\nCases: %d; errors: %d\n" % (report["D0_A"], len(cases), len(failures)), encoding="utf-8"
    )
    return report


def main(argv=None):
    values = sys.argv[1:] if argv is None else list(argv)
    if "--worker" in values:
        index = values.index("--worker")
        _worker(values[index + 1], float(values[index + 2]), float(values[index + 3]), float(values[index + 4]), values[index + 5:])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="logs/phase_d/d0_a_20260903")
    parser.add_argument("--duration-s", type=float, default=10.0)
    stage, remaining = parser.parse_known_args(values)
    report = run_audit(stage.output, stage.duration_s, remaining)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
