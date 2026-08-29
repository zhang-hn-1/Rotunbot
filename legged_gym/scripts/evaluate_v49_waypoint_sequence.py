"""Evaluate the frozen V49 policy on empty-map multi-waypoint sequences.

The evaluator owns only the world-frame waypoint layer.  It sends a projected
``[v, w]`` command to the existing V49 task, holds that command for ten 50 Hz
policy steps, and never resets the environment between waypoints.
"""

import argparse
import csv
import json
import math
import os
import sys

import isaacgym  # noqa: F401 - must precede torch/task imports
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403 - task registration
from legged_gym.envs.rotunbot.vel_tracking.rotunbot_vel import (
    command_update_interval_steps,
    feasible_yaw_rate_limit,
    yaw_from_quaternion,
)
from legged_gym.navigation.v49_waypoint_diagnostics import (
    DiagnosticMode,
    apply_diagnostic_command,
    detect_low_speed_yaw_collapse,
    summarize_command_transitions,
    summarize_terminal_results,
    rate_feedforward_active_ratio,
    dynamic_transition_severity,
    yaw_sign_reversal_count,
)
from legged_gym.navigation.v49_waypoint_controller import (
    V49WaypointConfig,
    WaypointSequenceController,
)
from legged_gym.utils import get_args, task_registry


TRAJECTORIES = {
    "A": ((1.0, 0.0), (2.0, 0.0), (3.0, 0.0)),
    "B": ((1.0, 0.0), (2.0, 0.25), (3.0, 0.0)),
}
INITIAL_YAWS_DEG = (-15, -10, -5, 0, 5, 10, 15)
PROFILE_ASSETS = {
    "maze": "Rotunbot.urdf",
    "v49_reference": "Rotunbot_test2.urdf",
}
LOG_FIELDS = (
    "phase", "time_s", "episode_id", "policy_step", "active_waypoint_index",
    "pose_x", "pose_y", "pose_yaw", "target_x", "target_y",
    "distance", "distance_to_waypoint_m", "bearing_error", "bearing_error_rad",
    "raw_v", "raw_v_mps",
    "raw_w", "raw_w_radps", "projected_v", "projected_v_mps", "projected_w",
    "projected_w_radps", "delta_projected_v_mps", "delta_projected_w_radps",
    "measured_v", "measured_v_mps", "measured_w", "measured_w_radps",
    "v_tracking_error_mps", "w_tracking_error_radps",
    "nominal_action_0", "nominal_action_1", "feedback_action_0", "feedback_action_1",
    "derivative_feedback_action_0", "derivative_feedback_action_1",
    "rate_feedforward_action_0", "rate_feedforward_action_1",
    "residual_action_0", "residual_action_1", "combined_action_0", "combined_action_1",
    "output_action_0", "output_action_1", "command_reference_is_smooth",
    "command_rate_v", "command_rate_w", "rate_feedforward_active",
    "low_speed_below_0_10", "low_speed_below_0_08", "static_yaw_limit_radps",
    "measured_speed_yaw_limit_radps", "static_feasible", "dynamic_transition_severity",
    "contact_yaw_damping_active", "contact_yaw_damping_torque",
    "contact_yaw_damping_speed_factor", "contact_yaw_damping_spin_rate",
    "contact_yaw_damping_planar_speed", "joint1_pos", "joint1_vel", "joint2_pos",
    "joint2_vel", "waypoint_reached", "waypoint_switched", "sequence_complete",
    "reset_count", "timeout", "nan_inf",
)
SETTLING_FIELDS = (
    "time_since_stop_s", "episode_id", "pose_x", "pose_y", "distance_to_final_goal_m",
    "measured_v_mps", "measured_w_radps", "joint1_vel", "joint2_pos", "joint2_vel",
)


def trajectory_waypoints(name):
    """Return the immutable Stage 1 waypoint tuple for trajectory A or B."""
    key = str(name).upper()
    if key not in TRAJECTORIES:
        raise ValueError("trajectory must be A or B")
    return TRAJECTORIES[key]


def initial_pose_for_episode(seed, episode_id):
    """Return deterministic ``(x, y, yaw)`` initialization for one episode."""
    generator = np.random.RandomState(int(seed) + int(episode_id))
    x, y = generator.uniform(-0.05, 0.05, size=2)
    yaw_deg = int(generator.choice(np.asarray(INITIAL_YAWS_DEG)))
    return float(x), float(y), float(yaw_deg) * math.pi / 180.0


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--waypoint_episodes", type=int, default=30)
    parser.add_argument("--waypoint_seed", type=int, default=20260828)
    parser.add_argument("--waypoint_output_dir", type=str, default=None)
    parser.add_argument(
        "--waypoint_asset_profile",
        choices=tuple(PROFILE_ASSETS),
        default="maze",
    )
    parser.add_argument(
        "--waypoint_trajectory", choices=("A", "B", "both"), default="both"
    )
    parser.add_argument("--waypoint_max_route_steps", type=int, default=2500)
    parser.add_argument("--waypoint_settle_s", type=float, default=2.0)
    parser.add_argument(
        "--diagnostic_smooth_reference",
        choices=("baseline", "true"),
        default="baseline",
    )
    parser.add_argument(
        "--diagnostic_minimum_rolling_speed", type=float, default=None
    )
    original_argv = list(sys.argv)
    diagnostic, remaining = parser.parse_known_args()
    sys.argv = [original_argv[0]] + remaining
    try:
        args = get_args()
    finally:
        sys.argv = original_argv
    if args.task != "rotunbot_vel_sru50_v49_integration":
        raise ValueError(
            "Stage 1 requires --task rotunbot_vel_sru50_v49_integration"
        )
    if not args.load_run or args.checkpoint is None or int(args.checkpoint) < 0:
        raise ValueError("--load_run and an explicit --checkpoint are required")
    if diagnostic.waypoint_episodes < 1:
        raise ValueError("--waypoint_episodes must be positive")
    if diagnostic.waypoint_max_route_steps < 1:
        raise ValueError("--waypoint_max_route_steps must be positive")
    if (
        diagnostic.waypoint_settle_s < 0.0
        or diagnostic.diagnostic_minimum_rolling_speed is not None
        and diagnostic.diagnostic_minimum_rolling_speed <= 0.0
    ):
        raise ValueError("settling time and rolling floor must be nonnegative/positive")
    args.load_run = os.path.abspath(args.load_run)
    args.checkpoint = int(args.checkpoint)
    args.num_envs = 1
    for name, value in vars(diagnostic).items():
        setattr(args, name, value)
    return args


def _configure(env_cfg, args):
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 90.0
    env_cfg.commands.resampling_time = 10000.0
    env_cfg.commands.direct_command_tracking = True
    env_cfg.commands.hold_upper_command_rate = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.asset.file = (
        "{LEGGED_GYM_ROOT_DIR}/resources/robots/Rotunbot/urdf/"
        + PROFILE_ASSETS[args.waypoint_asset_profile]
    )


def _close_env(env):
    try:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
    finally:
        if env.sim is not None:
            env.gym.destroy_sim(env.sim)


def _set_command(env, command, smooth_reference=False):
    env.command_reference_is_smooth.fill_(bool(smooth_reference))
    env.set_command_targets(command)
    env.compute_observations()


def _set_initial_pose(env, pose):
    x, y, yaw = pose
    env.root_states[0, :3] = torch.as_tensor(
        [x, y, float(env.cfg.init_state.pos[2])], device=env.device
    ) + env.env_origins[0]
    env.root_states[0, 3:7] = torch.as_tensor(
        [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
        device=env.device,
    )
    env.root_states[0, 7:13] = 0.0
    env.dof_pos[0] = env.default_dof_pos
    env.dof_vel[0] = 0.0
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(torch.zeros(1, dtype=torch.int32, device=env.device)),
        1,
    )
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.root_states)
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel.zero_()
    env.base_ang_vel.zero_()
    env.tracking_heading[:] = yaw
    env.projected_gravity[:] = quat_rotate_inverse(
        env.base_quat, env.gravity_vec
    )
    # BaseTask.reset() performs one zero-action step after reset_idx().  That
    # compatibility step can repopulate command/profile state from the base
    # sampler before this evaluator installs its first explicit target.  Clear
    # those runtime histories again at the episode boundary so the first
    # observation is independent of the previous episode's run order.
    for name in (
        "command_targets", "commands", "command_rates", "held_upper_command_rates",
        "tracking_error_integral", "last_tracking_error", "tracking_error_derivative",
        "requested_output_actions", "output_actions", "last_output_actions", "actions",
        "last_actions", "nominal_policy_actions", "feedback_policy_actions",
        "derivative_feedback_policy_actions", "rate_feedforward_policy_actions",
        "combined_policy_actions", "applied_residual_actions", "tracking_lin_vel",
        "tracking_ang_vel", "command_profile_phase", "command_profile_speed_amplitude",
        "command_profile_signed_curvature", "command_profile_velocity_offset",
        "command_profile_velocity_amplitude", "command_profile_yaw_amplitude",
        "command_profile_yaw_phase_offset",
    ):
        value = getattr(env, name, None)
        if value is not None:
            value.zero_()
    if hasattr(env, "command_profile_period"):
        env.command_profile_period.fill_(1.0)
    if hasattr(env, "command_profile_yaw_frequency_ratio"):
        env.command_profile_yaw_frequency_ratio.fill_(1.0)
    for name in (
        "command_brake_pending", "command_yaw_brake_pending",
        "command_profile_is_smooth", "command_profile_is_random_walk",
        "command_profile_is_independent", "command_reference_is_smooth",
    ):
        value = getattr(env, name, None)
        if value is not None:
            value.zero_()
    env.compute_observations()


def _pose(env):
    xy = env.root_states[:, :2]
    yaw = yaw_from_quaternion(env.root_states[:, 3:7])
    return xy, yaw


def _finite_tensor(*tensors):
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


def _buffer_pair(env, name):
    value = getattr(env, name, None)
    if value is None:
        return 0.0, 0.0
    return float(value[0, 0].item()), float(value[0, 1].item())


def _tick_control_snapshot(env):
    names = {
        "nominal_action": "nominal_policy_actions",
        "feedback_action": "feedback_policy_actions",
        "derivative_feedback_action": "derivative_feedback_policy_actions",
        "rate_feedforward_action": "rate_feedforward_policy_actions",
        "residual_action": "applied_residual_actions",
        "combined_action": "combined_policy_actions",
        "output_action": "output_actions",
    }
    snapshot = {}
    for prefix, buffer_name in names.items():
        first, second = _buffer_pair(env, buffer_name)
        snapshot[prefix + "_0"] = first
        snapshot[prefix + "_1"] = second
    snapshot["rate_feedforward_active"] = (
        abs(snapshot["rate_feedforward_action_0"]) > 1.0e-6
        or abs(snapshot["rate_feedforward_action_1"]) > 1.0e-6
    )
    return snapshot


def _tick_row(
    env, episode_id, policy_step, tick, target, reset_count, timeout,
    raw_command, projected_command, delta_v, delta_w, smooth_reference,
):
    xy, yaw = _pose(env)
    measured_v = env.tracking_lin_vel[:, 0]
    measured_w = env.tracking_ang_vel[:, 2]
    static_yaw_limit = feasible_yaw_rate_limit(
        projected_command[:, 0], 0.10, 3.148148148148148, 0.85, 0.08, 0.10
    )
    measured_speed_yaw_limit = feasible_yaw_rate_limit(
        measured_v, 0.10, 3.148148148148148, 0.85, 0.08, 0.10
    )
    rate_v = float(env.command_rates[0, 0].item())
    rate_w = float(env.command_rates[0, 1].item())
    control = _tick_control_snapshot(env)
    finite = _finite_tensor(
        xy, yaw, target, tick.raw_command, tick.projected_command,
        raw_command, projected_command, tick.distance, tick.bearing_error,
        measured_v, measured_w,
    )
    row = {
        "phase": "route",
        "time_s": float(policy_step * env.dt),
        "episode_id": int(episode_id),
        "policy_step": int(policy_step),
        "active_waypoint_index": int(tick.active_waypoint_index),
        "pose_x": float(xy[0, 0].item()),
        "pose_y": float(xy[0, 1].item()),
        "pose_yaw": float(yaw[0].item()),
        "target_x": float(target[0, 0].item()),
        "target_y": float(target[0, 1].item()),
        "distance": float(tick.distance[0].item()),
        "distance_to_waypoint_m": float(tick.distance[0].item()),
        "bearing_error": float(tick.bearing_error[0].item()),
        "bearing_error_rad": float(tick.bearing_error[0].item()),
        "raw_v": float(raw_command[0, 0].item()),
        "raw_v_mps": float(raw_command[0, 0].item()),
        "raw_w": float(raw_command[0, 1].item()),
        "raw_w_radps": float(raw_command[0, 1].item()),
        "projected_v": float(projected_command[0, 0].item()),
        "projected_v_mps": float(projected_command[0, 0].item()),
        "projected_w": float(projected_command[0, 1].item()),
        "projected_w_radps": float(projected_command[0, 1].item()),
        "delta_projected_v_mps": float(delta_v),
        "delta_projected_w_radps": float(delta_w),
        "measured_v": float(measured_v[0].item()),
        "measured_v_mps": float(measured_v[0].item()),
        "measured_w": float(measured_w[0].item()),
        "measured_w_radps": float(measured_w[0].item()),
        "v_tracking_error_mps": float(projected_command[0, 0] - measured_v[0]),
        "w_tracking_error_radps": float(projected_command[0, 1] - measured_w[0]),
        "command_reference_is_smooth": bool(smooth_reference),
        "command_rate_v": rate_v,
        "command_rate_w": rate_w,
        "low_speed_below_0_10": abs(float(measured_v[0])) < 0.10,
        "low_speed_below_0_08": abs(float(measured_v[0])) < 0.08,
        "static_yaw_limit_radps": float(static_yaw_limit[0].item()),
        "measured_speed_yaw_limit_radps": float(measured_speed_yaw_limit[0].item()),
        # A raw command is statically feasible exactly when the unchanged
        # V49 projection leaves it untouched.  The diagnostic rolling floor
        # is intentionally included in this comparison: it can create a
        # request that is subsequently clipped by the same static set.
        "static_feasible": bool(
            torch.allclose(raw_command, projected_command, atol=1.0e-6, rtol=0.0)
        ),
        "dynamic_transition_severity": dynamic_transition_severity(delta_v, delta_w),
        "contact_yaw_damping_active": bool(
            getattr(env, "contact_yaw_damping_active", torch.zeros(1))[0].item()
        ),
        "contact_yaw_damping_torque": float(
            getattr(env, "contact_yaw_damping_torque", torch.zeros(1))[0].item()
        ),
        "contact_yaw_damping_speed_factor": float(
            getattr(env, "contact_yaw_damping_speed_factor", torch.zeros(1))[0].item()
        ),
        "contact_yaw_damping_spin_rate": float(
            getattr(env, "contact_yaw_damping_spin_rate", torch.zeros(1))[0].item()
        ),
        "contact_yaw_damping_planar_speed": float(
            getattr(env, "contact_yaw_damping_planar_speed", torch.zeros(1))[0].item()
        ),
        "joint1_pos": float(env.dof_pos[0, 0].item()),
        "joint1_vel": float(env.dof_vel[0, 0].item()),
        "joint2_pos": float(env.dof_pos[0, 1].item()),
        "joint2_vel": float(env.dof_vel[0, 1].item()),
        "waypoint_reached": bool(tick.waypoint_reached),
        "waypoint_switched": bool(tick.waypoint_switched),
        "sequence_complete": bool(tick.sequence_complete),
        "reset_count": int(reset_count),
        "timeout": bool(timeout),
        "nan_inf": not finite,
    }
    row.update(control)
    row["rate_feedforward_active"] = bool(control["rate_feedforward_active"])
    return row


def _step_policy(env, policy):
    with torch.no_grad():
        actions = policy(env.get_observations())
        return env.step(actions)


def _refresh_runtime_row(env, row):
    measured_v = float(env.tracking_lin_vel[0, 0].item())
    measured_w = float(env.tracking_ang_vel[0, 2].item())
    row["measured_v"] = measured_v
    row["measured_v_mps"] = measured_v
    row["measured_w"] = measured_w
    row["measured_w_radps"] = measured_w
    row["v_tracking_error_mps"] = row["projected_v_mps"] - measured_v
    row["w_tracking_error_radps"] = row["projected_w_radps"] - measured_w
    row["command_rate_v"] = float(env.command_rates[0, 0].item())
    row["command_rate_w"] = float(env.command_rates[0, 1].item())
    row["low_speed_below_0_10"] = abs(measured_v) < 0.10
    row["low_speed_below_0_08"] = abs(measured_v) < 0.08
    row["joint1_pos"] = float(env.dof_pos[0, 0].item())
    row["joint1_vel"] = float(env.dof_vel[0, 0].item())
    row["joint2_pos"] = float(env.dof_pos[0, 1].item())
    row["joint2_vel"] = float(env.dof_vel[0, 1].item())
    damping = {
        "contact_yaw_damping_active": bool(
            getattr(env, "contact_yaw_damping_active", torch.zeros(1))[0].item()
        ),
        "contact_yaw_damping_torque": float(
            getattr(env, "contact_yaw_damping_torque", torch.zeros(1))[0].item()
        ),
        "contact_yaw_damping_speed_factor": float(
            getattr(env, "contact_yaw_damping_speed_factor", torch.zeros(1))[0].item()
        ),
        "contact_yaw_damping_spin_rate": float(
            getattr(env, "contact_yaw_damping_spin_rate", torch.zeros(1))[0].item()
        ),
        "contact_yaw_damping_planar_speed": float(
            getattr(env, "contact_yaw_damping_planar_speed", torch.zeros(1))[0].item()
        ),
    }
    row.update(damping)
    row.update(_tick_control_snapshot(env))
    return row


def _settling_row(env, episode_id, time_since_stop_s, final_target):
    xy, _ = _pose(env)
    return {
        "time_since_stop_s": float(time_since_stop_s),
        "episode_id": int(episode_id),
        "pose_x": float(xy[0, 0].item()),
        "pose_y": float(xy[0, 1].item()),
        "distance_to_final_goal_m": float(
            torch.linalg.vector_norm(xy - final_target).item()
        ),
        "measured_v_mps": float(env.tracking_lin_vel[0, 0].item()),
        "measured_w_radps": float(env.tracking_ang_vel[0, 2].item()),
        "joint1_vel": float(env.dof_vel[0, 0].item()),
        "joint2_pos": float(env.dof_pos[0, 1].item()),
        "joint2_vel": float(env.dof_vel[0, 1].item()),
    }


def _failure_signature(result):
    if result["nan_inf"]:
        return "UNCLASSIFIED"
    if result["low_speed_yaw_collapse_detected"]:
        return "LOW_SPEED_YAW_COLLAPSE"
    if result["max_abs_delta_projected_v"] > 0.016 or result["max_abs_delta_projected_w"] > 0.008:
        return "LARGE_COMMAND_TRANSIENT"
    if not result["route_complete"]:
        if result["max_abs_bearing_error"] >= math.radians(10.0):
            return "YAW_TRACKING_DEFICIT"
        return "FORWARD_TRACKING_DEFICIT"
    if result["terminal_speed_failure"]:
        return "TERMINAL_SETTLING"
    if result["final_position_error_m"] > 0.25:
        return "WAYPOINT_GEOMETRY"
    return "UNCLASSIFIED"


def _run_episode(env, policy, args, trajectory_name, episode_id):
    env.reset()
    _set_initial_pose(env, initial_pose_for_episode(args.waypoint_seed, episode_id))
    diagnostic_mode = DiagnosticMode(
        smooth_reference=args.diagnostic_smooth_reference == "true",
        minimum_rolling_speed=args.diagnostic_minimum_rolling_speed,
    )
    controller = WaypointSequenceController(
        torch.as_tensor(trajectory_waypoints(trajectory_name)),
        config=V49WaypointConfig(),
        policy_steps_per_tick=command_update_interval_steps(env.dt, 5.0),
    )
    rows = []
    settling_rows = []
    reset_count = 0
    timeout = False
    hold_violations = 0
    projection_violations = 0
    previous_command = None
    previous_projected_v = []
    previous_projected_w = []
    yaw_reversal_events = []
    previous_index = 0
    reached_count = 0
    entered_e1 = False
    entered_e2 = False
    bearing_at_e1 = None
    bearing_at_e2 = None
    measured_v_at_e1 = None
    measured_v_at_e2 = None
    measured_w_at_e1 = None
    measured_w_at_e2 = None
    e2_bearings = []
    last_tick = None
    last_route_row = None
    policy_step = 0

    while policy_step < args.waypoint_max_route_steps:
        if policy_step % controller.policy_steps_per_tick == 0:
            xy, yaw = _pose(env)
            last_tick = controller.tick(xy, yaw)
            target = controller.waypoints[last_tick.active_waypoint_index].to(
                device=env.device
            ).unsqueeze(0)
            if last_tick.sequence_complete:
                raw_command = last_tick.raw_command
                projected_command = last_tick.projected_command
            else:
                raw_command, projected_command = apply_diagnostic_command(
                    last_tick.raw_command,
                    diagnostic_mode,
                    maximum_forward_speed=0.13,
                    maximum_yaw_rate=0.10,
                    minimum_turn_radius=3.148148148148148,
                    envelope_fraction=0.85,
                    stationary_threshold=0.02,
                    turn_authority_start_speed=0.08,
                    turn_authority_full_speed=0.10,
                )
            _set_command(
                env, projected_command,
                smooth_reference=diagnostic_mode.smooth_reference,
            )
            previous_command = projected_command.detach().clone()
            if last_tick.waypoint_reached:
                reached_count += 1
            if last_tick.active_waypoint_index - previous_index > 1:
                projection_violations += 1
            previous_index = last_tick.active_waypoint_index
            delta_v = 0.0 if previous_projected_v == [] else float(
                projected_command[0, 0] - previous_projected_v[-1]
            )
            delta_w = 0.0 if previous_projected_w == [] else float(
                projected_command[0, 1] - previous_projected_w[-1]
            )
            if previous_projected_w and yaw_sign_reversal_count(
                torch.as_tensor(previous_projected_w[-1:]),
                projected_command[:, 1].detach().cpu(),
            ):
                yaw_reversal_events.append({
                    "policy_step": policy_step,
                    "measured_v": float(env.tracking_lin_vel[0, 0].item()),
                    "measured_w": float(env.tracking_ang_vel[0, 2].item()),
                    "delta_w": delta_w,
                })
            previous_projected_v.append(float(projected_command[0, 0].item()))
            previous_projected_w.append(float(projected_command[0, 1].item()))
            if not entered_e1 and float(last_tick.distance[0]) <= 0.58:
                entered_e1 = True
                bearing_at_e1 = float(last_tick.bearing_error[0].item())
                measured_v_at_e1 = float(env.tracking_lin_vel[0, 0].item())
                measured_w_at_e1 = float(env.tracking_ang_vel[0, 2].item())
            if not entered_e2 and float(last_tick.distance[0]) <= 0.46:
                entered_e2 = True
                bearing_at_e2 = float(last_tick.bearing_error[0].item())
                measured_v_at_e2 = float(env.tracking_lin_vel[0, 0].item())
                measured_w_at_e2 = float(env.tracking_ang_vel[0, 2].item())
            if entered_e2 and not last_tick.waypoint_switched:
                e2_bearings.append(abs(float(last_tick.bearing_error[0].item())))
            row = _tick_row(
                env, episode_id, policy_step, last_tick, target, reset_count,
                timeout, raw_command, projected_command, delta_v, delta_w,
                diagnostic_mode.smooth_reference,
            )
            last_route_row = row
            if last_tick.sequence_complete:
                rows.append(row)
                break

        _, _, _, dones, _ = _step_policy(env, policy)
        # The upper-layer request is assigned only in the tick branch above;
        # this explicit schedule check documents and audits the 10-step hold
        # without confusing V49's internal measured-state buffers with the
        # external command port.
        if policy_step % controller.policy_steps_per_tick:
            if previous_command is None:
                hold_violations += 1
        if bool(torch.any(dones)):
            reset_count += int(torch.sum(dones).item())
            timeout = bool(torch.any(env.time_out_buf))
            break
        if last_route_row is not None and policy_step % controller.policy_steps_per_tick == 0:
            _refresh_runtime_row(env, last_route_row)
            rows.append(last_route_row)
            last_route_row = None
        policy_step += 1

    route_complete = bool(last_tick is not None and last_tick.sequence_complete)
    final_xy, _ = _pose(env)
    final_target = torch.as_tensor(
        trajectory_waypoints(trajectory_name)[-1],
        dtype=final_xy.dtype, device=final_xy.device,
    ).unsqueeze(0)
    arrival_error = float(torch.linalg.vector_norm(final_xy - final_target).item())
    settled_error = arrival_error
    last_measured_v = float(env.tracking_lin_vel[0, 0].item())
    last_measured_w = float(env.tracking_ang_vel[0, 2].item())
    settle_steps = int(round(args.waypoint_settle_s / env.dt))
    terminal_v = last_measured_v
    stop_distance = None
    peak_post_stop_error = None
    time_to_v10 = None
    time_to_v05 = None
    if route_complete and reset_count == 0:
        zero_command = torch.zeros(1, 2, device=env.device)
        _set_command(env, zero_command, smooth_reference=diagnostic_mode.smooth_reference)
        stop_xy, _ = _pose(env)
        previous_xy = stop_xy.clone()
        travelled = 0.0
        for settle_step in range(settle_steps):
            _, _, _, dones, _ = _step_policy(env, policy)
            current_xy, _ = _pose(env)
            travelled += float(torch.linalg.vector_norm(current_xy - previous_xy).item())
            previous_xy = current_xy.clone()
            settling_rows.append(_settling_row(
                env, episode_id, (settle_step + 1) * env.dt, final_target
            ))
            speed = abs(settling_rows[-1]["measured_v_mps"])
            if time_to_v10 is None and speed <= 0.10:
                time_to_v10 = settling_rows[-1]["time_since_stop_s"]
            if time_to_v05 is None and speed <= 0.05:
                time_to_v05 = settling_rows[-1]["time_since_stop_s"]
            if bool(torch.any(dones)):
                reset_count += int(torch.sum(dones).item())
                timeout = bool(torch.any(env.time_out_buf))
                break
        final_xy, _ = _pose(env)
        settled_error = float(torch.linalg.vector_norm(final_xy - final_target).item())
        terminal_v = float(env.tracking_lin_vel[0, 0].item())
        stop_distance = travelled
        peak_post_stop_error = max(
            row["distance_to_final_goal_m"] for row in settling_rows
        ) if settling_rows else arrival_error
    transition_stats = summarize_command_transitions(
        previous_projected_v, previous_projected_w
    )
    low_speed_collapse = detect_low_speed_yaw_collapse(
        rows, tick_period_s=controller.policy_steps_per_tick * env.dt
    )
    rate_ff_ratio = rate_feedforward_active_ratio(
        [row["rate_feedforward_action_0"] for row in rows],
        [row["rate_feedforward_action_1"] for row in rows],
    )
    max_bearing = max((abs(row["bearing_error"]) for row in rows), default=0.0)
    final_abs_bearing = abs(rows[-1]["bearing_error"]) if rows else 0.0
    failed_waypoint = None if route_complete else int(controller.active_waypoint_index)
    failed_rows = [
        row for row in rows
        if failed_waypoint is not None
        and row["active_waypoint_index"] == failed_waypoint
    ]
    terminal_speed_failure = route_complete and abs(terminal_v) > 0.10

    if any(row["nan_inf"] for row in rows):
        failure_classes = ["UNCLASSIFIED"]
    else:
        failure_classes = []
        if hold_violations or projection_violations:
            failure_classes.append("LARGE_COMMAND_TRANSIENT")
        if reached_count < len(trajectory_waypoints(trajectory_name)):
            failure_classes.append("WAYPOINT_GEOMETRY")
        if reset_count or timeout or (
            route_complete
            and controller.switch_count
            != len(trajectory_waypoints(trajectory_name)) - 1
        ):
            failure_classes.append("UNCLASSIFIED")
        if not route_complete or arrival_error > 0.25 or terminal_speed_failure:
            failure_classes.append("TERMINAL_SETTLING" if terminal_speed_failure else "FORWARD_TRACKING_DEFICIT")
    result = {
        "trajectory": trajectory_name,
        "episode_id": int(episode_id),
        "initial_pose": initial_pose_for_episode(args.waypoint_seed, episode_id),
        "route_complete": route_complete,
        "waypoints_reached": int(reached_count),
        "waypoints_total": len(trajectory_waypoints(trajectory_name)),
        "switch_count": int(controller.switch_count),
        "reset_count": int(reset_count),
        "timeout": bool(timeout),
        "hold_violations": int(hold_violations),
        "projection_or_skip_violations": int(projection_violations),
        "final_position_error_m": arrival_error,
        "settled_position_error_m": settled_error,
        "terminal_measured_v_mps": terminal_v,
        "last_measured_v_mps": last_measured_v,
        "last_measured_w_radps": last_measured_w,
        "terminal_speed_safe": bool(route_complete and not terminal_speed_failure),
        "terminal_speed_failure": bool(terminal_speed_failure),
        "stop_distance": stop_distance,
        "peak_post_stop_position_error_m": peak_post_stop_error,
        "time_to_abs_v_below_0.10_s": time_to_v10,
        "time_to_abs_v_below_0.05_s": time_to_v05,
        "entered_E1": entered_e1,
        "entered_E2": entered_e2,
        "bearing_error_at_E1": bearing_at_e1,
        "bearing_error_at_E2": bearing_at_e2,
        "measured_v_at_E1": measured_v_at_e1,
        "measured_v_at_E2": measured_v_at_e2,
        "measured_w_at_E1": measured_w_at_e1,
        "measured_w_at_E2": measured_w_at_e2,
        "bearing_error_reduction_after_E2": (
            abs(bearing_at_e2) - min(e2_bearings)
            if bearing_at_e2 is not None and e2_bearings else None
        ),
        "low_speed_yaw_collapse_detected": bool(low_speed_collapse),
        "rate_feedforward_active_tick_ratio": rate_ff_ratio,
        "max_abs_bearing_error": max_bearing,
        "final_abs_bearing_error": final_abs_bearing,
        "failed_waypoint_index": failed_waypoint,
        "minimum_distance_to_failed_waypoint_m": min(
            (row["distance_to_waypoint_m"] for row in failed_rows),
            default=None,
        ),
        "final_distance_to_failed_waypoint_m": (
            failed_rows[-1]["distance_to_waypoint_m"] if failed_rows else None
        ),
        "min_abs_measured_v_mps": min(
            (abs(row["measured_v_mps"]) for row in rows), default=None
        ),
        "yaw_sign_reversal_count": len(yaw_reversal_events),
        "yaw_sign_reversal_events": yaw_reversal_events,
        "command_transition": transition_stats,
        "max_abs_delta_projected_v": transition_stats["max_abs_delta_v"],
        "max_abs_delta_projected_w": transition_stats["max_abs_delta_w"],
        "time_fraction_measured_v_below_0_10": float(
            sum(row["low_speed_below_0_10"] for row in rows)
        ) / max(len(rows), 1),
        "time_fraction_measured_v_below_0_08": float(
            sum(row["low_speed_below_0_08"] for row in rows)
        ) / max(len(rows), 1),
        "max_abs_contact_yaw_damping_torque": max(
            (abs(row["contact_yaw_damping_torque"]) for row in rows),
            default=0.0,
        ),
        "final_joint1_vel": float(env.dof_vel[0, 0].item()),
        "final_joint2_pos": float(env.dof_pos[0, 1].item()),
        "final_joint2_vel": float(env.dof_vel[0, 1].item()),
        "nan_inf": bool(any(row["nan_inf"] for row in rows)),
        "failure_classes": failure_classes,
        "rows": rows,
        "settling_rows": settling_rows,
    }
    result["primary_observed_failure_signature"] = _failure_signature(result)
    result["failure_type"] = "route_complete" if route_complete else "route_incomplete"
    return result


def _write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_settling_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SETTLING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _failure_diagnostic(result):
    keys = (
        "trajectory", "episode_id", "initial_pose", "failed_waypoint_index",
        "minimum_distance_to_failed_waypoint_m", "final_distance_to_failed_waypoint_m",
        "max_abs_bearing_error", "final_abs_bearing_error", "time_fraction_measured_v_below_0_10",
        "time_fraction_measured_v_below_0_08", "max_abs_delta_projected_v",
        "max_abs_delta_projected_w", "yaw_sign_reversal_count",
        "rate_feedforward_active_tick_ratio", "max_abs_contact_yaw_damping_torque",
        "final_joint1_vel", "final_joint2_pos", "final_joint2_vel",
        "primary_observed_failure_signature", "failure_type", "route_complete",
        "last_measured_v_mps", "last_measured_w_radps", "min_abs_measured_v_mps",
    )
    diagnostic = {key: result.get(key) for key in keys}
    initial_x, initial_y, initial_yaw = result["initial_pose"]
    diagnostic.update({
        "initial_x": initial_x,
        "initial_y": initial_y,
        "initial_yaw": initial_yaw,
        "failed_waypoint": result.get("failed_waypoint_index"),
        "min_distance_to_failed_waypoint": result.get(
            "minimum_distance_to_failed_waypoint_m"
        ),
        "final_distance_to_failed_waypoint": result.get(
            "final_distance_to_failed_waypoint_m"
        ),
        "min_abs_measured_v": result.get("min_abs_measured_v_mps"),
    })
    return diagnostic


def _p95(values):
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95)) if values else None


def _group_summary(results):
    episodes = len(results)
    complete = sum(bool(result["route_complete"]) for result in results)
    reached = sum(result["waypoints_reached"] for result in results)
    total_waypoints = sum(result["waypoints_total"] for result in results)
    final_errors = [result["final_position_error_m"] for result in results]
    return {
        "episodes": episodes,
        "success": int(complete),
        "sequence_success_ratio": float(complete) / max(episodes, 1),
        "waypoints_reached": int(reached),
        "waypoints_total": int(total_waypoints),
        "waypoint_reach_ratio": float(reached) / max(total_waypoints, 1),
        "timeout": int(sum(bool(result["timeout"]) for result in results)),
        "final_error_mean_m": float(np.mean(final_errors)) if final_errors else None,
        "final_error_p95_m": _p95(final_errors),
    }


def _episode_detail(result):
    return {
        key: value for key, value in result.items()
        if key not in ("rows", "settling_rows")
    }


def _summarize(results, args, model_path):
    episodes = len(results)
    complete = sum(bool(result["route_complete"]) for result in results)
    reached = sum(result["waypoints_reached"] for result in results)
    total_waypoints = sum(result["waypoints_total"] for result in results)
    intermediate_resets = sum(result["reset_count"] for result in results)
    skips = sum(result["projection_or_skip_violations"] for result in results)
    nan_inf = sum(result["nan_inf"] for result in results)
    completed_terminal = summarize_terminal_results(results)
    terminal_safe = completed_terminal["completed_terminal_speed_safe_count"]
    final_error_ok = all(
        result["final_position_error_m"] <= 0.25 for result in results
    )
    transition_v = []
    transition_w = []
    tracking_v = []
    tracking_w = []
    all_rows = []
    for result in results:
        rows = result.get("rows", [])
        all_rows.extend(rows)
        transition_v.extend(abs(row["delta_projected_v_mps"]) for row in rows[1:])
        transition_w.extend(abs(row["delta_projected_w_radps"]) for row in rows[1:])
        tracking_v.extend(abs(row["v_tracking_error_mps"]) for row in rows)
        tracking_w.extend(abs(row["w_tracking_error_radps"]) for row in rows)
    transition = {
        "mean_abs_delta_v": float(np.mean(transition_v)) if transition_v else 0.0,
        "p95_abs_delta_v": _p95(transition_v) or 0.0,
        "max_abs_delta_v": max(transition_v, default=0.0),
        "mean_abs_delta_w": float(np.mean(transition_w)) if transition_w else 0.0,
        "p95_abs_delta_w": _p95(transition_w) or 0.0,
        "max_abs_delta_w": max(transition_w, default=0.0),
    }
    transition["fraction_abs_delta_v_gt_0.008"] = float(
        np.mean(np.asarray(transition_v) > 0.008)
    ) if transition_v else 0.0
    transition["fraction_abs_delta_w_gt_0.004"] = float(
        np.mean(np.asarray(transition_w) > 0.004)
    ) if transition_w else 0.0
    transition["fraction_abs_delta_v_gt_0.016"] = float(
        np.mean(np.asarray(transition_v) > 0.016)
    ) if transition_v else 0.0
    transition["fraction_abs_delta_w_gt_0.008"] = float(
        np.mean(np.asarray(transition_w) > 0.008)
    ) if transition_w else 0.0
    rate_ff_ratio = rate_feedforward_active_ratio(
        [row["rate_feedforward_action_0"] for row in all_rows],
        [row["rate_feedforward_action_1"] for row in all_rows],
    )
    checks = {
        "sequence_success_ge_90pct": float(complete) / max(episodes, 1) >= 0.90,
        "waypoint_reach_ge_95pct": float(reached) / max(total_waypoints, 1) >= 0.95,
        "final_error_le_0.25m": final_error_ok,
        "intermediate_resets_zero": intermediate_resets == 0,
        "skip_zero": skips == 0,
        "nan_inf_zero": nan_inf == 0,
        "completed_terminal_speed_le_0.10mps": (
            completed_terminal["completed_terminal_speed_failure_count"] == 0
        ),
    }
    failure_counts = {}
    for result in results:
        signature = result["primary_observed_failure_signature"]
        if not result["route_complete"] or not result["terminal_speed_safe"]:
            failure_counts[signature] = failure_counts.get(signature, 0) + 1
    by_trajectory = {
        name: _group_summary(
            [result for result in results if result["trajectory"] == name]
        ) for name in ("A", "B")
    }
    by_yaw = {}
    for yaw_deg in INITIAL_YAWS_DEG:
        matching = [
            result for result in results
            if round(result["initial_pose"][2] * 180.0 / math.pi) == yaw_deg
        ]
        if matching:
            by_yaw[str(yaw_deg)] = _group_summary(matching)
    failed_waypoints = {
        "fail_at_P1": sum(result["failed_waypoint_index"] == 0 for result in results),
        "fail_at_P2": sum(result["failed_waypoint_index"] == 1 for result in results),
        "fail_at_P3": sum(result["failed_waypoint_index"] == 2 for result in results),
    }
    return {
        "task": args.task,
        "asset_profile": args.waypoint_asset_profile,
        "asset": PROFILE_ASSETS[args.waypoint_asset_profile],
        "checkpoint": args.checkpoint,
        "checkpoint_path": model_path,
        "physics_hz": 1.0 / float(args._sim_dt),
        "low_level_hz": 1.0 / float(args._policy_dt),
        "upper_command_hz": 5.0,
        "policy_steps_per_upper_tick": 10,
        "diagnostic_smooth_reference": args.diagnostic_smooth_reference,
        "diagnostic_minimum_rolling_speed": args.diagnostic_minimum_rolling_speed,
        "episodes": episodes,
        "route_complete_count": int(complete),
        "route_incomplete_count": int(episodes - complete),
        "sequence_success_count": int(complete),
        "sequence_success_ratio": float(complete) / max(episodes, 1),
        "waypoints_reached": int(reached),
        "waypoints_total": int(total_waypoints),
        "waypoint_reach_ratio": float(reached) / max(total_waypoints, 1),
        "intermediate_reset_count": int(intermediate_resets),
        "skip_count": int(skips),
        "nan_inf_count": int(nan_inf),
        "completed_terminal_speed_safe_count": int(terminal_safe),
        "completed_terminal_speed_failure_count": int(
            completed_terminal["completed_terminal_speed_failure_count"]
        ),
        "terminal": completed_terminal,
        "trajectory": by_trajectory,
        "initial_yaw": by_yaw,
        "failed_waypoint": failed_waypoints,
        "rate_feedforward_active_tick_ratio": rate_ff_ratio,
        # This run-local field only confirms that the baseline has no active
        # rate-FF.  Cross-run mismatch is reported as detected only when a
        # separate smooth AB shows a nonzero ratio; that comparison belongs in
        # the audit report, not in an isolated evaluator invocation.
        "smooth_control_disabled_detected": False,
        "smooth_control_baseline_rate_ff_zero": bool(
            args.diagnostic_smooth_reference == "baseline" and rate_ff_ratio <= 1.0e-6
        ),
        "tracking": {
            "v_mae_mps": float(np.mean(tracking_v)) if tracking_v else 0.0,
            "w_mae_radps": float(np.mean(tracking_w)) if tracking_w else 0.0,
            "v_p95_abs_error_mps": _p95(tracking_v),
            "w_p95_abs_error_radps": _p95(tracking_w),
        },
        "low_speed": {
            "failure_episodes_entering_below_0.10": sum(
                bool(result["entered_E1"]) and not result["route_complete"]
                for result in results
            ),
            "failure_episodes_entering_below_0.08": sum(
                bool(result["entered_E2"]) and not result["route_complete"]
                for result in results
            ),
            "low_speed_yaw_collapse_detected": sum(
                bool(result["low_speed_yaw_collapse_detected"])
                for result in results
            ),
        },
        "command_transition": transition,
        "failure_class_counts": failure_counts,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "episodes_detail": [_episode_detail(result) for result in results],
    }


def main():
    args = _parse_args()
    model_path = os.path.join(args.load_run, "model_%d.pt" % args.checkpoint)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _configure(env_cfg, args)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    args._sim_dt = float(env.sim_params.dt)
    args._policy_dt = float(env.dt)
    try:
        runner, _ = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
        )
        policy = runner.get_inference_policy(device=env.device)
        trajectory_names = (
            ("A", "B") if args.waypoint_trajectory == "both"
            else (args.waypoint_trajectory,)
        )
        results = []
        output_dir = args.waypoint_output_dir or os.path.join(
            args.load_run, "v49_waypoint_sequence", args.waypoint_asset_profile
        )
        os.makedirs(output_dir, exist_ok=True)
        for trajectory_name in trajectory_names:
            all_rows = []
            all_settling_rows = []
            for episode_id in range(args.waypoint_episodes):
                result = _run_episode(
                    env, policy, args, trajectory_name, episode_id
                )
                results.append(result)
                all_rows.extend(result["rows"])
                all_settling_rows.extend(result["settling_rows"])
            _write_rows(
                os.path.join(
                    output_dir,
                    "trajectory_%s.csv" % trajectory_name,
                ),
                all_rows,
            )
            _write_settling_rows(
                os.path.join(
                    output_dir,
                    "trajectory_%s_settling.csv" % trajectory_name,
                ),
                all_settling_rows,
            )
        summary = _summarize(results, args, model_path)
        failure_path = os.path.join(output_dir, "failure_diagnostics.json")
        with open(failure_path, "w", encoding="utf-8") as handle:
            json.dump(
                [
                    _failure_diagnostic(result) for result in results
                    if not result["route_complete"] or not result["terminal_speed_safe"]
                ],
                handle,
                indent=2,
                ensure_ascii=False,
            )
        json_path = os.path.join(
            output_dir,
            "stage1_v49_waypoint_summary_%s.json" % args.waypoint_asset_profile,
        )
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("JSON: %s" % json_path)
        if summary["verdict"] != "PASS":
            raise SystemExit(1)
    finally:
        _close_env(env)


if __name__ == "__main__":
    main()
