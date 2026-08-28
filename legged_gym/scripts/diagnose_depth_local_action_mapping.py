"""Measure the actual two-action Rotunbot low-level mapping in isolation.

This diagnostic deliberately uses the task unchanged.  It disables scene,
noise, randomization, and random yaw, then resets one robot to a canonical
state before each open-loop action sequence.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import isaacgym  # noqa: F401 - must precede torch in Isaac Gym Preview 4
import numpy as np

if not hasattr(np, "float"):
    np.float = float

import torch
from isaacgym import gymtorch

import legged_gym.envs  # noqa: F401 - registration side effects
from legged_gym.scripts.depth_local_diagnostics import action_mapping_decision
from legged_gym.utils import get_args, task_registry


TASK = "rotunbot_maze_local_depth"


def _make_env():
    old_argv = sys.argv
    sys.argv = [old_argv[0], "--headless"]
    try:
        args = get_args()
    finally:
        sys.argv = old_argv
    args.task = TASK
    env_cfg, _ = task_registry.get_cfgs(TASK)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 10.0
    env_cfg.maze.enabled = False
    env_cfg.maze.scene_mode = "none"
    env_cfg.camera.depth_backend = "fallback"
    env_cfg.camera.add_noise = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.random_start_yaw = False
    env_cfg.commands.resample_commands = False
    env, _ = task_registry.make_env(name=TASK, args=args, env_cfg=env_cfg)
    env.data_print = False
    return env


def _set_canonical_state(env):
    """Reset state after BaseTask.reset's mandatory zero-action step."""
    env.reset()
    env.root_states[0] = env.base_init_state
    env.root_states[0, :3] += env.env_origins[0]
    env.root_states[0, :2] = env.env_origins[0, :2]
    env.root_states[0, 3:7] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device=env.device
    )
    env.root_states[0, 7:13] = 0.0
    actor_indices = env.robot_actor_indices[:1].to(dtype=torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.actor_root_state),
        gymtorch.unwrap_tensor(actor_indices),
        1,
    )
    env.dof_pos[0] = 0.0
    env.dof_vel[0] = 0.0
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(actor_indices),
        1,
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel.zero_()
    env.base_ang_vel.zero_()
    env.projected_gravity[:] = torch.tensor(
        [[0.0, 0.0, -1.0]], device=env.device
    )
    env.last_actions.zero_()
    env.actions.zero_()
    env.episode_length_buf.zero_()
    env.reset_buf.zero_()
    env.time_out_buf.zero_()
    # Keep the target outside every sweep so success cannot auto-reset the env.
    far_goal = env.env_origins[0, :2] + torch.tensor(
        [10.0, 10.0], device=env.device
    )
    env.global_goal_xy_world[0] = far_goal
    env.active_local_goal_xy_world[0] = far_goal
    env.commands[0, :2] = far_goal
    env._update_base_euler()
    env._update_local_goal()
    env.obstacle_clearance = env._wall_distance()
    env.maze_collision_buf.zero_()
    env.prev_local_goal_dist[0] = torch.linalg.vector_norm(
        env.active_local_goal_xy_robot[0]
    )
    env.compute_observations()


def _float(value):
    return float(value.detach().cpu().item())


def _run_constant(env, action0, action1, duration):
    _set_canonical_state(env)
    start_xy = env.root_states[0, :2].detach().clone()
    start_yaw = _float(env.base_euler_tensor[0, 2])
    q2_values, torque0_values, torque1_values, vx_values, vy_values = [], [], [], [], []
    steps = max(1, int(round(float(duration) / float(env.dt))))
    for _ in range(steps):
        action = torch.tensor([[float(action0), float(action1)]], device=env.device)
        _, _, _, dones, _ = env.step(action)
        q2_values.append(_float(env.dof_pos[0, 1]))
        torque0_values.append(_float(env.torques[0, 0]))
        torque1_values.append(_float(env.torques[0, 1]))
        vx_values.append(_float(env.base_lin_vel[0, 0]))
        vy_values.append(_float(env.base_lin_vel[0, 1]))
        if bool(dones[0].item()):
            break
    end_xy = env.root_states[0, :2].detach().clone()
    dx, dy = _float(end_xy[0] - start_xy[0]), _float(end_xy[1] - start_xy[1])
    c, s = math.cos(start_yaw), math.sin(start_yaw)
    end_yaw = _float(env.base_euler_tensor[0, 2])
    return {
        "action0": float(action0), "action1": float(action1), "duration_s": float(duration),
        "start_xy": [_float(start_xy[0]), _float(start_xy[1])],
        "end_xy": [_float(end_xy[0]), _float(end_xy[1])],
        "delta_world_x": dx, "delta_world_y": dy,
        "delta_body_x": c * dx + s * dy, "delta_body_y": -s * dx + c * dy,
        "start_yaw": start_yaw, "end_yaw": end_yaw,
        "delta_yaw": math.atan2(math.sin(end_yaw - start_yaw), math.cos(end_yaw - start_yaw)),
        "final_body_vx": vx_values[-1], "final_body_vy": vy_values[-1],
        "max_abs_body_vx": max(map(abs, vx_values), default=0.0),
        "max_abs_body_vy": max(map(abs, vy_values), default=0.0),
        "joint1_pos": _float(env.dof_pos[0, 0]), "joint1_vel": _float(env.dof_vel[0, 0]),
        "joint2_pos": q2_values[-1], "joint2_vel": _float(env.dof_vel[0, 1]),
        "max_joint2_pos": max(q2_values), "min_joint2_pos": min(q2_values),
        "mean_torque0": sum(torque0_values) / len(torque0_values),
        "mean_torque1": sum(torque1_values) / len(torque1_values),
        "max_abs_torque0": max(map(abs, torque0_values), default=0.0),
        "max_abs_torque1": max(map(abs, torque1_values), default=0.0),
        "steps": len(vx_values), "terminated": bool(dones[0].item()),
    }


def _run_braking(env, first_action1, second_action1):
    _set_canonical_state(env)
    ys, vys = [], []
    phase_values = (first_action1, second_action1)
    for phase, action1 in enumerate(phase_values):
        for _ in range(max(1, int(round(2.0 / float(env.dt))))):
            action = torch.tensor([[0.0, float(action1)]], device=env.device)
            _, _, _, dones, _ = env.step(action)
            ys.append(_float(env.root_states[0, 1]))
            vys.append(_float(env.base_lin_vel[0, 1]))
            if bool(dones[0].item()):
                break
        if phase == 0:
            first_y, first_vy = ys[-1], vys[-1]
    final_vy, final_y = vys[-1], ys[-1]
    return {
        "first_action1": float(first_action1), "second_action1": float(second_action1),
        "vy_after_first": first_vy, "final_vy": final_vy,
        "y_after_first": first_y, "final_y": final_y,
        "peak_abs_body_vy": max(map(abs, vys), default=0.0),
        "reversal": bool(first_vy * final_vy < -0.01),
        "stopped": bool(abs(final_vy) <= 0.05),
        "braking_effective": bool(abs(final_vy) < abs(first_vy)),
    }


def _slope(cases, x_key, y_key, duration=None):
    rows = [case for case in cases if duration is None or case["duration_s"] == duration]
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    if len(xs) < 2:
        return 0.0
    xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom if denom else 0.0


def _write_summary(report, path):
    cases = report["constant_action"]
    lines = [
        "# Depth-local action mapping diagnostic (Experiment A)", "",
        "Canonical one-robot, no-maze, no-noise, no-randomization state; body frame uses initial yaw.", "",
        "Decision: **{}**".format(report["decision"]), "",
        "| a1 | duration (s) | ΔX_body | ΔY_body | final Vy | Δyaw | q2 final |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cases:
        lines.append(
            "| {action1:.2f} | {duration_s:.1f} | {delta_body_x:.4f} | {delta_body_y:.4f} | "
            "{final_body_vy:.4f} | {delta_yaw:.4f} | {joint2_pos:.4f} |".format(**row)
        )
    lines.extend([
        "", "Numeric relationships:", "",
        "- 5 s action1 → ΔY_body slope: `{:.6f} m/action`".format(
            _slope(cases, "action1", "delta_body_y", duration=5.0)
        ),
        "- 5 s action1 → final Vy slope: `{:.6f} m/s/action`".format(
            _slope(cases, "action1", "final_body_vy", duration=5.0)
        ),
        "", "Braking sequences:", "",
        "| first a1 | second a1 | Vy after first | final Vy | final Y | reversal | stopped |",
        "|---:|---:|---:|---:|---:|:---:|:---:|",
    ])
    for row in report["braking"]:
        lines.append(
            "| {first_action1:.2f} | {second_action1:.2f} | {vy_after_first:.4f} | "
            "{final_vy:.4f} | {final_y:.4f} | {reversal} | {stopped} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output):
    env = _make_env()
    a1_values = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
    cases = [
        _run_constant(env, 0.0, action1, duration)
        for duration in (2.0, 5.0)
        for action1 in a1_values
    ]
    cases.extend(
        _run_constant(env, action0, 0.0, 2.0)
        for action0 in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)
    )
    braking = [
        _run_braking(env, first, second)
        for first, second in ((0.5, 0.0), (0.5, -0.5), (1.0, 0.0), (1.0, -1.0),
                              (-0.5, 0.0), (-0.5, 0.5), (-1.0, 0.0), (-1.0, 1.0))
    ]
    try:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)
    except Exception:
        pass
    report = {
        "task": TASK,
        "experiment": "A_action_mapping",
        "settings": {
            "num_envs": 1, "maze_enabled": False, "scene_mode": "none",
            "depth_backend": "fallback", "add_noise": False,
            "randomize_friction": False, "randomize_base_mass": False,
            "push_robots": False, "random_start_yaw": False,
            "resample_commands": False, "start_yaw": 0.0,
        },
        "constant_action": cases,
        "action0_reference": [case for case in cases if case["action1"] == 0.0 and case["duration_s"] == 2.0 and case["action0"] != 0.0],
        "braking": braking,
    }
    endpoint = [case for case in cases if case["action0"] == 0.0]
    report["decision"] = action_mapping_decision(endpoint, braking)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_summary(report, output.with_name("action_mapping_summary.md"))
    print("Experiment A decision:", report["decision"])
    print("Saved:", output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="logs/depth_local_diagnostics/action_mapping_sweep.json")
    args = parser.parse_args(argv)
    run(Path(args.output))


if __name__ == "__main__":
    main()
