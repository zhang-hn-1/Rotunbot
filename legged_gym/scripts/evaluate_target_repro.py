#!/usr/bin/env python3
"""Strict, paired evaluation and failure diagnostics for rotunbot_target_repro.

This module is intentionally evaluation-only.  It does not edit the task,
reward, PPO, policy, history encoder, curriculum, robot model, controller, or
target sampler.  The only runtime changes are evaluation flags and the local
Isaac Gym/PyTorch compatibility shim required by RTX 40-series GPUs.

The executable modes are:

  generate-scenarios  Create a deterministic scenario manifest for one seed.
  worker              Evaluate one checkpoint/seed and write raw artifacts.
  analyze-screening   Aggregate screening workers and select the five models.
  analyze-final       Aggregate the 200-episode runs and write figures/report.

The parent process should invoke one fresh ``worker`` process per checkpoint
and seed.  This avoids Isaac Gym simulator/GPU state accumulating between
models.
"""

import argparse
import csv
from collections import deque
import distutils
import distutils.version
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

# Isaac Gym Preview 4 still imports the removed NumPy alias.  Keep this local
# to the evaluation process; do not change the project's dependency versions.
np.float = float
distutils.version = distutils.version

import isaacgym  # noqa: E402
import torch  # noqa: E402
import isaacgym.torch_utils as torch_utils  # noqa: E402


def _quat_rotate_inverse(q, v):
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(
        q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)
    ).squeeze(-1) * 2.0
    return a - b + c


def _quat_apply(a, b):
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2.0
    return b + a[:, 3:] * t + xyz.cross(t, dim=-1)


def _normalize(x, eps=1.0e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def _torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


# PyTorch 1.10/cu113 cannot JIT-compile these helpers for Ada GPUs.  Replacing
# only these tiny helpers with eager equivalents leaves the algorithm and task
# code untouched and makes the compatibility boundary explicit.
torch_utils.quat_rotate_inverse = _quat_rotate_inverse
torch_utils.quat_apply = _quat_apply
torch_utils.normalize = _normalize
torch_utils.torch_rand_float = _torch_rand_float

from isaacgym import gymtorch  # noqa: E402
from legged_gym.envs import *  # noqa: F401,F403,E402
from legged_gym.utils import get_args, task_registry  # noqa: E402


TASK = "rotunbot_target_repro"
DISTANCE_THRESHOLD = 0.20
SPEED_THRESHOLD = 0.10
EPISODES_SCREENING = 40
EPISODES_FINAL = 40
EPISODE_LENGTH_SECONDS = 60.0
MANDATORY_CHECKPOINTS = (2050, 2150, 3100)

DETAIL_FIELDS = [
    "phase", "seed", "checkpoint", "episode_id", "start_x", "start_y",
    "target_x", "target_y", "initial_yaw", "success", "termination_reason",
    "failure_mode", "failure_mode_detail", "episode_time", "path_length",
    "shortest_path", "spl", "min_distance", "final_distance", "final_speed",
    "ever_enter_020", "min_speed_inside_020", "time_inside_020",
    "overshoot_count", "reentry_count", "max_speed", "mean_abs_roll",
    "balance_metric", "terminal_x", "terminal_y", "steps",
]


def _parse_user_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--checkpoint", type=int)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--scenario-file")
    parser.add_argument("--phase", default="screening")
    parser.add_argument(
        "--control-type",
        choices=("R", "DIRECT_VP", "DIRECT_VP_TORQUE"),
        default="DIRECT_VP_TORQUE",
        help="Controller executor used only for this evaluation process.",
    )
    parser.add_argument(
        "--perturbation",
        choices=("nominal", "physical", "noise", "delay", "push", "combined", "paper"),
        default="nominal",
        help=(
            "Evaluation stress profile. 'combined' is the legacy physical "
            "profile; 'paper' combines physical randomization, observation "
            "noise, random 0-0.1 s sensor/action delay, and pushes."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--uniform-targets",
        action="store_true",
        help="generate-scenarios: disable hard-side sampling so the manifest "
        "follows the paper's uniform random-target protocol.",
    )
    known, _ = parser.parse_known_args()
    return known


def _gym_args():
    # Keep Isaac Gym's own argument parser out of the evaluation-specific CLI.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sys.argv = [
        sys.argv[0],
        "--headless",
        f"--sim_device={device}",
        f"--rl_device={device}",
    ]
    return get_args()


def _configure(seed, control_type="DIRECT_VP_TORQUE", perturbation="nominal"):
    args = _gym_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK)
    train_cfg.seed = int(seed)
    env_cfg.seed = int(seed)

    # Evaluation-only overrides.  No reward, PPO, actor/critic, history,
    # curriculum implementation, URDF, controller, or sampler code is edited.
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.commands.target_curriculum = False
    env_cfg.evaluation.target_error_threshold = DISTANCE_THRESHOLD
    env_cfg.evaluation.stop_velocity_threshold = SPEED_THRESHOLD
    env_cfg.control.control_type = control_type
    # Evaluation always uses the fixed configured executor gains: the
    # training-only per-episode gain randomization must not leak into the
    # measured protocol.
    env_cfg.control.direct_velocity_gain_randomize = False
    # All three executors receive the same action targets and slew limits.
    env_cfg.control.decimation = 1
    env_cfg.control.rate_limit_1 = 0.02
    env_cfg.control.rate_limit_2 = 0.04
    env_cfg.control.set_a_rate_limit = True
    # Delay is applied by the evaluation wrapper below, so keep the training
    # environment's latency augmentation disabled to avoid applying it twice.
    if hasattr(env_cfg, "latency"):
        env_cfg.latency.enabled = False

    if perturbation in ("physical", "combined", "paper"):
        # A deliberately visible but bounded stress profile: low contact
        # friction and +/-5 kg base-mass variation.  task_registry seeds
        # NumPy/Torch from `seed`, so each controller sees the same sampled
        # physical parameters for a paired seed.
        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.friction_range = [0.5, 0.7]
        env_cfg.domain_rand.randomize_base_mass = True
        env_cfg.domain_rand.added_mass_range = [-5.0, 5.0]
    else:
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_base_mass = False

    if perturbation in ("noise", "paper"):
        env_cfg.noise.add_noise = True

    if perturbation in ("push", "paper"):
        env_cfg.domain_rand.push_robots = True
        env_cfg.domain_rand.push_interval_s = 15.0
        env_cfg.domain_rand.max_push_vel_xy = 1.0

    return args, env_cfg, train_cfg


def _close_env(env):
    try:
        if getattr(env, "viewer", None) is not None:
            env.gym.destroy_viewer(env.viewer)
    finally:
        env.gym.destroy_sim(env.sim)


def _make_env(seed, control_type="DIRECT_VP_TORQUE", perturbation="nominal"):
    args, env_cfg, train_cfg = _configure(seed, control_type, perturbation)
    env, _ = task_registry.make_env(name=TASK, args=args, env_cfg=env_cfg)
    return args, env, train_cfg


def _yaw_from_quat(quat):
    qx, qy, qz, qw = [float(v) for v in quat]
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _capture_scenario(env):
    return {
        "root_states": env.root_states[0].detach().cpu().numpy().astype(np.float32),
        "dof_pos": env.dof_pos[0].detach().cpu().numpy().astype(np.float32),
        "dof_vel": env.dof_vel[0].detach().cpu().numpy().astype(np.float32),
        "target_xy": env.commands[0, :2].detach().cpu().numpy().astype(np.float32),
        "initial_yaw": _yaw_from_quat(env.root_states[0, 3:7].detach().cpu().numpy()),
    }


def generate_scenarios(seed, episodes, output_path, uniform_targets=False):
    """Generate the exact reset states used by every paired evaluation.

    With ``uniform_targets`` the hard-side sampling is disabled so the
    manifest matches the paper's "random targets" protocol (uniform full-map
    sampling) instead of the training mixture.
    """
    args, env_cfg, train_cfg = _configure(seed)
    if uniform_targets:
        env_cfg.commands.hard_side_target_probability = 0.0
        env_cfg.commands.target_curriculum = False
    env, _ = task_registry.make_env(name=TASK, args=args, env_cfg=env_cfg)
    try:
        # BaseTask.reset performs the same initial reset/zero-action transition
        # used by the policy runner.  Subsequent direct reset_idx calls consume
        # exactly one reset's random draws per scenario and do not depend on a
        # policy's episode duration.
        env.reset()
        scenarios = [_capture_scenario(env)]
        env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
        for _ in range(1, int(episodes)):
            env.reset_idx(env_ids)
            scenarios.append(_capture_scenario(env))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        root_states = np.stack([x["root_states"] for x in scenarios])
        dof_pos = np.stack([x["dof_pos"] for x in scenarios])
        dof_vel = np.stack([x["dof_vel"] for x in scenarios])
        target_xy = np.stack([x["target_xy"] for x in scenarios])
        initial_yaw = np.asarray([x["initial_yaw"] for x in scenarios], dtype=np.float32)
        np.savez_compressed(
            output_path,
            seed=np.asarray([seed], dtype=np.int64),
            episodes=np.asarray([episodes], dtype=np.int64),
            root_states=root_states,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            target_xy=target_xy,
            initial_yaw=initial_yaw,
        )
        print("Wrote scenario manifest:", output_path)
    finally:
        _close_env(env)


def _load_scenarios(path):
    data = np.load(path, allow_pickle=False)
    return {
        "root_states": data["root_states"],
        "dof_pos": data["dof_pos"],
        "dof_vel": data["dof_vel"],
        "target_xy": data["target_xy"],
        "initial_yaw": data["initial_yaw"],
    }


def _reset_history(env):
    env.actions.zero_()
    env.last_actions.zero_()
    env.output_actions.zero_()
    env.last_output_actions.zero_()
    env.last_dof_vel.zero_()
    env.last_root_vel.zero_()
    for history in env.obs_history:
        history.zero_()
    for history in env.critic_history:
        history.zero_()


def _apply_scenario(env, scenario):
    """Apply a manifest state after the environment's normal auto-reset."""
    env_ids = torch.tensor([0], dtype=torch.int32, device=env.device)
    root = torch.as_tensor(scenario["root_states"], device=env.device)
    dof_pos = torch.as_tensor(scenario["dof_pos"], device=env.device)
    dof_vel = torch.as_tensor(scenario["dof_vel"], device=env.device)
    target = torch.as_tensor(scenario["target_xy"], device=env.device)

    env.root_states[0] = root
    env.dof_pos[0] = dof_pos
    env.dof_vel[0] = dof_vel
    env.commands[0, :2] = target
    if env.commands.shape[1] > 2:
        env.commands[0, 2:] = 0.0
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim, gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids), 1,
    )
    env.gym.set_dof_state_tensor_indexed(
        env.sim, gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids), 1,
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)

    env.episode_length_buf[0] = 0
    env.reset_buf[0] = 1
    env.time_out_buf[0] = False
    env.success_buf[0] = 0.0
    env.arrived_target_buf[0] = 0.0
    env.stop_buf[0] = False
    env.base_quat[0] = env.root_states[0, 3:7]
    env.base_lin_vel[0] = _quat_rotate_inverse(
        env.base_quat[0:1], env.root_states[0:1, 7:10]
    )[0]
    env.base_ang_vel[0] = _quat_rotate_inverse(
        env.base_quat[0:1], env.root_states[0:1, 10:13]
    )[0]
    env.projected_gravity[0] = _quat_rotate_inverse(
        env.base_quat[0:1], env.gravity_vec[0:1]
    )[0]
    env._update_base_euler()
    env.goal_dist[0] = torch.linalg.norm(target - env.root_states[0, :2])
    env.last_goal_dist[0] = env.goal_dist[0]
    env.orientation_error[0] = 0.0
    env.last_orientation_error[0] = 0.0
    _reset_history(env)
    env.compute_observations()
    return env.get_observations()


def _install_terminal_probe(env):
    """Cache quantities before repro.check_termination auto-resets the env."""
    original = env.check_termination

    def probe():
        original()
        env._strict_terminal_dof_vel = env.dof_vel[0].detach().clone()
        env._strict_terminal_roll = env.base_euler_tensor[0, 0].detach().clone()
        env._strict_terminal_ang_vel = env.base_ang_vel[0].detach().clone()
        env._strict_terminal_position = env.root_states[0, :2].detach().clone()

    env.check_termination = probe


def _load_policy(seed, checkpoint, run_dir, control_type, perturbation):
    args, env, train_cfg = _make_env(seed, control_type, perturbation)
    _install_terminal_probe(env)
    args.task = TASK
    args.load_run = str(run_dir)
    args.checkpoint = int(checkpoint)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = str(run_dir)
    train_cfg.runner.checkpoint = int(checkpoint)
    # log_root is only used to resolve an absolute load_run; no TensorBoard
    # writer is created during evaluation.
    runner, _ = task_registry.make_alg_runner(
        env=env, args=args, train_cfg=train_cfg,
        log_root=str(Path(run_dir).parent),
    )
    policy = runner.get_inference_policy(device=env.device)
    return env, policy


def _episode_delay_steps(perturbation, seed, episode_id):
    """Return deterministic per-episode delays matching the paper's 0-0.1 s range."""
    if perturbation not in ("delay", "paper"):
        return 0, 0
    # One local RNG keeps delay sampling reproducible without changing the
    # simulator's NumPy/Torch random streams.  At 50 Hz, 0..5 steps is 0..0.1 s.
    rng = np.random.RandomState(int(seed) * 100003 + int(episode_id))
    observation_steps = int(rng.randint(0, 6))
    action_steps = int(rng.randint(0, 6))
    return observation_steps, action_steps


def _trace_state(env, action, done):
    if done:
        position = env._strict_terminal_position.detach().cpu().numpy()
        speed = float(env.terminal_speed[0].item())
        dof_vel = env._strict_terminal_dof_vel.detach().cpu().numpy()
        roll = float(env._strict_terminal_roll.item())
        ang_vel = env._strict_terminal_ang_vel.detach().cpu().numpy()
    else:
        position = env.root_states[0, :2].detach().cpu().numpy().copy()
        speed = float(torch.linalg.norm(env.base_lin_vel[0]).item())
        dof_vel = env.dof_vel[0].detach().cpu().numpy().copy()
        roll = float(env.base_euler_tensor[0, 0].item())
        ang_vel = env.base_ang_vel[0].detach().cpu().numpy().copy()
    distance = float(env.terminal_goal_dist[0].item()) if done else float(env.goal_dist[0].item())
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "distance": distance,
        "speed": speed,
        "action": action.detach().cpu().numpy().astype(np.float32).copy(),
        "joint_velocity": np.asarray(dof_vel, dtype=np.float32).copy(),
        "roll": roll,
        # Body-frame base angular velocity (x/y/z), stored so the balance
        # metric can be recomputed offline under any axis convention.
        "ang_vel": np.asarray(ang_vel, dtype=np.float32).copy(),
    }


def _failure_mode(trace, termination_reason, initial_distance):
    inside = np.asarray(trace["distance"]) <= DISTANCE_THRESHOLD + 1.0e-7
    ever_enter = bool(np.any(inside))
    overshoots = int(trace["overshoot_count"])
    reentries = int(trace["reentry_count"])

    if termination_reason in ("unstable", "out_of_bounds", "other"):
        return "F5 Instability", "environment terminated before a formal success"
    if ever_enter and overshoots > 0 and (reentries > 0 or termination_reason == "timeout"):
        return "F3 Overshoot", "entered 0.20 m, then exited the success region"
    if ever_enter and termination_reason == "timeout":
        return "F2 Reached But Too Fast", "entered 0.20 m but never met speed <= 0.10 m/s"
    if not ever_enter and termination_reason == "timeout":
        distances = np.asarray(trace["distance"], dtype=np.float64)
        if len(distances) > 2:
            delta = np.diff(distances)
            approaching_fraction = float(np.mean(delta < 0.0))
        else:
            approaching_fraction = 0.0
        stable = float(np.mean(np.abs(trace["roll"]))) < 0.5
        made_progress = float(np.min(distances)) < max(0.5 * initial_distance, initial_distance - 1.0)
        if stable and made_progress and approaching_fraction >= 0.55:
            return "F4 Slow Timeout", "stable and approaching, but did not complete in 60 s"
        return "F1 Never Reached", "minimum distance remained above 0.20 m"
    return "F1 Never Reached", "minimum distance remained above 0.20 m"


def _write_trace(path, trace, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=np.asarray(trace["x"], dtype=np.float32),
        y=np.asarray(trace["y"], dtype=np.float32),
        distance=np.asarray(trace["distance"], dtype=np.float32),
        speed=np.asarray(trace["speed"], dtype=np.float32),
        action=np.asarray(trace["action"], dtype=np.float32),
        joint_velocity=np.asarray(trace["joint_velocity"], dtype=np.float32),
        roll=np.asarray(trace["roll"], dtype=np.float32),
        ang_vel=np.asarray(trace["ang_vel"], dtype=np.float32),
        dt=np.asarray([0.02], dtype=np.float32),
        metadata=np.asarray([json.dumps(row, ensure_ascii=False)], dtype=object),
    )


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_worker(args, checkpoint, seed, episodes, scenario_file, phase, output_dir):
    scenarios = _load_scenarios(scenario_file)
    if len(scenarios["target_xy"]) < episodes:
        raise ValueError("scenario manifest has fewer episodes than requested")

    raw_dir = Path(output_dir) / "raw" / phase / f"seed_{seed}" / f"checkpoint_{checkpoint:05d}"
    details_path = raw_dir / "episode_details.csv"
    summary_path = raw_dir / "summary.json"
    trace_dir = raw_dir / "traces"
    if details_path.exists() and summary_path.exists() and not args.force:
        print("Skipping existing worker:", raw_dir)
        return

    env, policy = _load_policy(
        seed, checkpoint, args.run_dir, args.control_type, args.perturbation
    )
    rows = []
    traces = []
    try:
        obs = _apply_scenario(env, {k: v[0] for k, v in scenarios.items()})
        for episode_id in range(int(episodes)):
            if episode_id > 0:
                scenario = {k: v[episode_id] for k, v in scenarios.items()}
                obs = _apply_scenario(env, scenario)
            scenario = {k: v[episode_id] for k, v in scenarios.items()}
            observation_delay_steps, action_delay_steps = _episode_delay_steps(
                args.perturbation, seed, episode_id
            )
            observation_queue = deque(
                [obs.detach().clone() for _ in range(observation_delay_steps + 1)],
                maxlen=observation_delay_steps + 1,
            )
            zero_action = torch.zeros(
                (1, env.num_actions), dtype=obs.dtype, device=env.device
            )
            action_queue = deque(
                [zero_action.clone() for _ in range(action_delay_steps + 1)],
                maxlen=action_delay_steps + 1,
            )
            target = scenario["target_xy"]
            start = scenario["root_states"][:2]
            initial_distance = float(np.linalg.norm(target - start))
            trace = {"x": [float(start[0])], "y": [float(start[1])],
                     "distance": [initial_distance], "speed": [0.0],
                     "action": [np.zeros(env.num_actions, dtype=np.float32)],
                     "joint_velocity": [scenario["dof_vel"]], "roll": [0.0],
                     "ang_vel": [np.zeros(3, dtype=np.float32)],
                     "overshoot_count": 0, "reentry_count": 0}
            inside_previous = initial_distance <= DISTANCE_THRESHOLD
            ever_enter = inside_previous
            time_inside = 0.0
            min_speed_inside = math.inf
            balance_sum = 0.0
            steps = 0
            done = False
            termination_reason = "other"
            # Use no_grad rather than inference_mode: the task rebinds a few
            # state buffers during check_termination, and those buffers must
            # remain mutable when the next paired scenario is applied.
            with torch.no_grad():
                while not done and steps < int(env.max_episode_length) + 2:
                    commanded_action = policy(observation_queue[0])
                    action_queue.append(commanded_action.detach().clone())
                    action = action_queue[0]
                    obs, _, _, dones, _ = env.step(action)
                    observation_queue.append(obs.detach().clone())
                    done = bool(dones[0].item())
                    steps += 1
                    sample = _trace_state(env, action[0], done)
                    current_inside = sample["distance"] <= DISTANCE_THRESHOLD + 1.0e-7
                    had_entered_before = ever_enter
                    if current_inside:
                        time_inside += float(env.dt)
                        min_speed_inside = min(min_speed_inside, sample["speed"])
                        ever_enter = True
                    if inside_previous and not current_inside:
                        trace["overshoot_count"] += 1
                    if (not inside_previous) and current_inside and had_entered_before:
                        trace["reentry_count"] += 1
                    inside_previous = current_inside
                    balance_reward = math.exp(
                        -float(np.sum(np.square(env._strict_terminal_ang_vel.detach().cpu().numpy()[:2])))
                    ) if done else math.exp(
                        -float(torch.sum(torch.square(env.base_ang_vel[0, :2])).item())
                    )
                    balance_sum += balance_reward
                    for key in ("x", "y", "distance", "speed", "action", "joint_velocity", "roll", "ang_vel"):
                        trace[key].append(sample[key])

                    if done:
                        success = bool(env.success_buf[0].item())
                        timeout = bool(env.terminal_timeout[0].item())
                        unstable = bool(env.terminal_unstable[0].item())
                        out_of_bounds = bool(env.terminal_out_of_bounds[0].item())
                        if success:
                            termination_reason = "success"
                        elif unstable:
                            termination_reason = "unstable"
                        elif out_of_bounds:
                            termination_reason = "out_of_bounds"
                        elif timeout:
                            termination_reason = "timeout"
                        else:
                            termination_reason = "other"
            if not done:
                raise RuntimeError(f"episode {episode_id} exceeded evaluation guard")

            trace_arrays = {k: np.asarray(trace[k]) for k in
                            ("x", "y", "distance", "speed", "action", "joint_velocity", "roll", "ang_vel")}
            min_distance = float(np.min(trace_arrays["distance"]))
            final_distance = float(env.terminal_goal_dist[0].item())
            final_speed = float(env.terminal_speed[0].item())
            path_length = float(np.sum(np.hypot(np.diff(trace_arrays["x"]), np.diff(trace_arrays["y"]))))
            shortest_path = max(initial_distance, 1.0e-6)
            path_length = max(path_length, shortest_path)
            success = termination_reason == "success"
            spl = shortest_path / path_length if success else 0.0
            min_speed_value = min_speed_inside if math.isfinite(min_speed_inside) else ""
            row = {
                "phase": phase, "seed": seed, "checkpoint": checkpoint,
                "episode_id": episode_id, "start_x": float(start[0]), "start_y": float(start[1]),
                "target_x": float(target[0]), "target_y": float(target[1]),
                "initial_yaw": float(scenario["initial_yaw"]), "success": int(success),
                "termination_reason": termination_reason, "failure_mode": "", "failure_mode_detail": "",
                "episode_time": steps * float(env.dt), "path_length": path_length,
                "shortest_path": shortest_path, "spl": spl, "min_distance": min_distance,
                "final_distance": final_distance, "final_speed": final_speed,
                "ever_enter_020": int(ever_enter), "min_speed_inside_020": min_speed_value,
                "time_inside_020": time_inside, "overshoot_count": int(trace["overshoot_count"]),
                "reentry_count": int(trace["reentry_count"]), "max_speed": float(np.max(trace_arrays["speed"])),
                "mean_abs_roll": float(np.mean(np.abs(trace_arrays["roll"]))),
                "balance_metric": balance_sum / max(steps, 1) * 100.0,
                "terminal_x": float(trace_arrays["x"][-1]), "terminal_y": float(trace_arrays["y"][-1]),
                "steps": steps,
            }
            if not success:
                mode, detail = _failure_mode({**trace_arrays, "overshoot_count": trace["overshoot_count"],
                                               "reentry_count": trace["reentry_count"]},
                                              termination_reason, initial_distance)
                row["failure_mode"] = mode
                row["failure_mode_detail"] = detail
            rows.append(row)
            traces.append((row, trace_arrays))
            if (episode_id + 1) % 10 == 0 or episode_id + 1 == episodes:
                print(
                    f"{phase} checkpoint={checkpoint} seed={seed} "
                    f"episode={episode_id + 1}/{episodes}",
                    flush=True,
                )

        # Keep the complete raw trace bundle so post-processing can select
        # failures, overshoots, low-SPL successes, and paired exemplars later.
        raw_dir.mkdir(parents=True, exist_ok=True)
        trace_blob = raw_dir / "traces.npz"
        np.savez_compressed(
            trace_blob,
            traces=np.asarray([t[1] for t in traces], dtype=object),
        )
        _write_csv(details_path, rows, DETAIL_FIELDS)
        success_rows = [r for r in rows if int(r["success"]) == 1]
        failure_counts = {}
        for row in rows:
            key = row["failure_mode"] or "success"
            failure_counts[key] = failure_counts.get(key, 0) + 1
        summary = {
            "phase": phase, "seed": seed, "checkpoint": checkpoint,
            "control_type": args.control_type,
            "perturbation": args.perturbation,
            "episodes": len(rows), "success_count": sum(int(r["success"]) for r in rows),
            "success_rate": float(np.mean([int(r["success"]) for r in rows])),
            "spl": float(np.mean([float(r["spl"]) for r in rows])),
            "cls_m": float(np.mean([float(r["min_distance"]) for r in rows])),
            "balance_metric_percent": float(np.mean([float(r["balance_metric"]) for r in rows])),
            "average_path_length_m": float(np.mean([float(r["path_length"]) for r in rows])),
            "average_final_distance_m": float(np.mean([float(r["final_distance"]) for r in rows])),
            "average_final_speed_mps": float(np.mean([float(r["final_speed"]) for r in rows])),
            "average_episode_time_s": float(np.mean([float(r["episode_time"]) for r in rows])),
            "failure_counts": failure_counts,
            "termination_counts": _counts(rows, "termination_reason"),
            "scenario_file": str(scenario_file),
            "distance_threshold_m": DISTANCE_THRESHOLD,
            "speed_threshold_mps": SPEED_THRESHOLD,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Wrote worker:", raw_dir, flush=True)
    finally:
        _close_env(env)


def _counts(rows, field):
    result = {}
    for row in rows:
        value = row[field]
        result[value] = result.get(value, 0) + 1
    return result


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(row, key):
    value = row.get(key, "")
    return float(value) if value not in (None, "") else math.nan


def _aggregate_workers(output_dir, phase):
    root = Path(output_dir) / "raw" / phase
    rows = []
    summaries = []
    for summary_path in sorted(root.glob("seed_*/checkpoint_*/summary.json")):
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        rows.extend(_read_csv(summary_path.parent / "episode_details.csv"))
    return rows, summaries


def _summary_from_rows(rows, phase):
    grouped = {}
    for row in rows:
        grouped.setdefault((int(row["seed"]), int(row["checkpoint"])), []).append(row)
    output = []
    for (seed, checkpoint), group in sorted(grouped.items()):
        output.append({
            "phase": phase, "seed": seed, "checkpoint": checkpoint,
            "episodes": len(group),
            "success_count": sum(int(r["success"]) for r in group),
            "success_rate": np.mean([int(r["success"]) for r in group]),
            "spl": np.mean([_to_float(r, "spl") for r in group]),
            "cls_m": np.mean([_to_float(r, "min_distance") for r in group]),
            "balance_metric_percent": np.mean([_to_float(r, "balance_metric") for r in group]),
            "average_path_length_m": np.mean([_to_float(r, "path_length") for r in group]),
            "average_final_distance_m": np.mean([_to_float(r, "final_distance") for r in group]),
            "average_final_speed_mps": np.mean([_to_float(r, "final_speed") for r in group]),
            "average_episode_time_s": np.mean([_to_float(r, "episode_time") for r in group]),
            "failure_counts": json.dumps(_counts(group, "failure_mode"), ensure_ascii=False),
            "termination_counts": json.dumps(_counts(group, "termination_reason"), ensure_ascii=False),
        })
    return output


def _write_summary_csv(path, rows):
    fields = [
        "phase", "seed", "checkpoint", "episodes", "success_count", "success_rate",
        "spl", "cls_m", "balance_metric_percent", "average_path_length_m",
        "average_final_distance_m", "average_final_speed_mps", "average_episode_time_s",
        "failure_counts", "termination_counts",
    ]
    normalized = []
    for row in rows:
        normalized.append({k: row.get(k, "") for k in fields})
    _write_csv(path, normalized, fields)


def _percentile_rank(values, value, higher=True):
    values = np.asarray(values, dtype=float)
    if not higher:
        values = -values
        value = -value
    return float(np.mean(values <= value))


def _rank_checkpoints(summary_rows):
    # Screening is one seed per checkpoint.  Rank on several diagnostics rather
    # than success rate alone, then retain the three requested comparison models.
    groups = {}
    for row in summary_rows:
        groups[int(row["checkpoint"])] = row
    candidates = list(groups.values())
    fields = ["success_rate", "spl", "cls_m", "balance_metric_percent", "average_path_length_m"]
    higher = {"success_rate": True, "spl": True, "cls_m": False,
              "balance_metric_percent": True, "average_path_length_m": False}
    values = {field: [_to_float(r, field) for r in candidates] for field in fields}
    for row in candidates:
        score = 0.40 * _percentile_rank(values["success_rate"], _to_float(row, "success_rate"), True)
        score += 0.25 * _percentile_rank(values["spl"], _to_float(row, "spl"), True)
        score += 0.15 * _percentile_rank(values["cls_m"], _to_float(row, "cls_m"), False)
        score += 0.10 * _percentile_rank(values["balance_metric_percent"], _to_float(row, "balance_metric_percent"), True)
        score += 0.10 * _percentile_rank(values["average_path_length_m"], _to_float(row, "average_path_length_m"), False)
        row["composite_score"] = score
    ranked = sorted(candidates, key=lambda r: r["composite_score"], reverse=True)
    mandatory = [groups[ck] for ck in MANDATORY_CHECKPOINTS if ck in groups]
    chosen = []
    for row in mandatory:
        if row not in chosen:
            chosen.append(row)
    for row in ranked:
        if row not in chosen:
            chosen.append(row)
        if len(chosen) >= 5:
            break
    return chosen[:5], ranked


def analyze_screening(output_dir):
    rows, _ = _aggregate_workers(output_dir, "screening_40")
    if not rows:
        raise ValueError("No screening workers found")
    summary_rows = _summary_from_rows(rows, "screening_40")
    out = Path(output_dir) / "screening_40"
    out.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(out / "checkpoint_summary.csv", summary_rows)
    _write_csv(out / "episode_details.csv", rows, DETAIL_FIELDS)
    chosen, ranked = _rank_checkpoints(summary_rows)
    selected = [int(r["checkpoint"]) for r in chosen]
    (Path(output_dir) / "screening_40" / "selected_top5.json").write_text(
        json.dumps({"top5": selected, "mandatory": list(MANDATORY_CHECKPOINTS),
                    "ranking": [int(r["checkpoint"]) for r in ranked]}, indent=2),
        encoding="utf-8",
    )
    print("Selected strict Top5:", selected)
    return selected


def _load_all_details(output_dir, phase):
    rows, _ = _aggregate_workers(output_dir, phase)
    return rows


def _find_trace(output_dir, phase, seed, checkpoint, episode_id):
    path = Path(output_dir) / "raw" / phase / f"seed_{seed}" / f"checkpoint_{checkpoint:05d}" / "traces.npz"
    bundle = np.load(path, allow_pickle=True)["traces"]
    return bundle[int(episode_id)]


def _materialize_selected_traces(output_dir, phase, rows, limit_low_spl=10):
    trajectories = Path(output_dir) / "trajectories"
    figures = Path(output_dir) / "figures"
    trajectories.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    selected = []
    for row in rows:
        if int(row["success"]) == 0 or int(row["overshoot_count"]) > 0:
            selected.append(row)
    successful = [r for r in rows if int(r["success"]) == 1]
    successful.sort(key=lambda r: _to_float(r, "spl"))
    selected.extend(successful[:limit_low_spl])
    # Keep one copy per episode key.
    unique = {}
    for row in selected:
        key = (row["phase"], row["seed"], row["checkpoint"], row["episode_id"])
        unique[key] = row
    figure_paths = []
    for row in unique.values():
        trace = _find_trace(output_dir, phase, int(row["seed"]), int(row["checkpoint"]), int(row["episode_id"]))
        name = f"{phase}_seed{row['seed']}_ck{int(row['checkpoint']):05d}_ep{int(row['episode_id']):03d}"
        path = trajectories / f"{name}.npz"
        _write_trace(path, trace, row)
        figure_paths.append(path)
    _make_figures(figures, figure_paths)


def _materialize_pair_difference_traces(output_dir, rows, limit=8):
    """Save representative paired episodes with the largest model differences."""
    trajectories = Path(output_dir) / "trajectories"
    figures = Path(output_dir) / "figures"
    trajectories.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for row in rows:
        if int(row["checkpoint"]) in MANDATORY_CHECKPOINTS:
            grouped.setdefault((int(row["seed"]), int(row["episode_id"])), {})[
                int(row["checkpoint"])] = row

    scored = []
    for key, by_ck in grouped.items():
        if not all(ck in by_ck for ck in MANDATORY_CHECKPOINTS):
            continue
        path_values = [_to_float(by_ck[ck], "path_length") for ck in MANDATORY_CHECKPOINTS]
        dist_values = [_to_float(by_ck[ck], "min_distance") for ck in MANDATORY_CHECKPOINTS]
        time_values = [_to_float(by_ck[ck], "episode_time") for ck in MANDATORY_CHECKPOINTS]
        over_values = [_to_float(by_ck[ck], "overshoot_count") for ck in MANDATORY_CHECKPOINTS]
        success_values = [int(by_ck[ck]["success"]) for ck in MANDATORY_CHECKPOINTS]
        score = (
            10.0 * (max(success_values) - min(success_values))
            + (max(path_values) - min(path_values)) / (np.mean(path_values) + 1.0e-6)
            + (max(dist_values) - min(dist_values)) / (np.mean(dist_values) + 1.0e-6)
            + (max(time_values) - min(time_values)) / (np.mean(time_values) + 1.0e-6)
            + (max(over_values) - min(over_values)) / (max(over_values) + 1.0)
        )
        scored.append((score, key, by_ck))

    figure_paths = []
    for _, (seed, episode_id), by_ck in sorted(scored, reverse=True)[:limit]:
        for checkpoint in MANDATORY_CHECKPOINTS:
            row = by_ck[checkpoint]
            trace = _find_trace(output_dir, "top5_200", seed, checkpoint, episode_id)
            name = f"paired_diff_seed{seed}_ep{episode_id:03d}_ck{checkpoint:05d}"
            path = trajectories / f"{name}.npz"
            _write_trace(path, trace, row)
            figure_paths.append(path)
    _make_figures(figures, figure_paths)


def _make_figures(figures, trajectories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for path in sorted(trajectories):
        if not path.exists() or path.suffix != ".npz":
            continue
        output_path = figures / f"{path.stem}.png"
        if output_path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        x, y = data["x"], data["y"]
        distance, speed = data["distance"], data["speed"]
        actions = data["action"]
        joint_velocity = data["joint_velocity"]
        dt = float(data["dt"][0])
        t = np.arange(len(x)) * dt
        stem = path.stem

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(x, y, color="tab:blue", linewidth=1.2)
        axes[0].scatter([x[0]], [y[0]], color="green", label="start")
        axes[0].scatter([x[-1]], [y[-1]], color="black", label="end")
        theta = np.linspace(0, 2 * np.pi, 200)
        meta = json.loads(str(data["metadata"][0]))
        tx, ty = float(meta["target_x"]), float(meta["target_y"])
        axes[0].plot(tx + DISTANCE_THRESHOLD * np.cos(theta), ty + DISTANCE_THRESHOLD * np.sin(theta), "r--", label="0.20 m")
        axes[0].scatter([tx], [ty], color="red", label="target")
        axes[0].set_title("XY trajectory")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.25)

        axes[1].plot(t, distance, color="tab:purple")
        axes[1].axhline(DISTANCE_THRESHOLD, color="red", linestyle="--")
        axes[1].set_title("Distance vs time")
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("distance (m)")
        axes[1].grid(alpha=0.25)

        axes[2].plot(t, speed, color="tab:orange", label="base speed")
        axes[2].plot(t, np.linalg.norm(joint_velocity, axis=1), color="tab:green", alpha=0.65, label="joint velocity norm")
        axes[2].axhline(SPEED_THRESHOLD, color="red", linestyle="--")
        axes[2].set_title("Speed vs time")
        axes[2].set_xlabel("time (s)")
        axes[2].set_ylabel("speed")
        axes[2].legend(fontsize=7)
        axes[2].grid(alpha=0.25)
        fig.suptitle(stem, fontsize=9)
        fig.tight_layout()
        fig.savefig(output_path, dpi=130)
        plt.close(fig)


def _write_pairwise(output_dir, rows, phase):
    mandatory_rows = [r for r in rows if int(r["checkpoint"]) in MANDATORY_CHECKPOINTS]
    grouped = {}
    for row in mandatory_rows:
        grouped.setdefault((int(row["seed"]), int(row["episode_id"])), []).append(row)
    fields = ["phase", "seed", "episode_id", "checkpoint", "success", "failure_mode",
              "path_length", "min_distance", "spl", "episode_time", "overshoot_count",
              "final_distance", "final_speed"]
    pair_rows = []
    for (seed, episode_id), group in sorted(grouped.items()):
        for row in sorted(group, key=lambda x: int(x["checkpoint"])):
            pair_rows.append({k: row.get(k, "") for k in fields})
    out = Path(output_dir) / "top5_200" / "paired_comparison.csv"
    _write_csv(out, pair_rows, fields)

    # Produce a compact win table for the report.
    win_rows = []
    for key, group in sorted(grouped.items()):
        by_ck = {int(r["checkpoint"]): r for r in group}
        for metric, lower_is_better in [("success", False), ("path_length", True),
                                        ("min_distance", True), ("overshoot_count", True),
                                        ("episode_time", True)]:
            values = {ck: _to_float(by_ck[ck], metric) if metric != "success" else float(by_ck[ck][metric])
                      for ck in by_ck}
            best = (min if lower_is_better else max)(values, key=values.get)
            ties = [ck for ck, value in values.items() if value == values[best]]
            win_rows.append({"seed": key[0], "episode_id": key[1], "metric": metric,
                             "winner_checkpoint": best, "ties": ";".join(map(str, sorted(ties)))})
    _write_csv(Path(output_dir) / "top5_200" / "paired_winners.csv", win_rows,
               ["seed", "episode_id", "metric", "winner_checkpoint", "ties"])


def _write_report(output_dir, screening_rows, final_rows, top5, ranked):
    out = Path(output_dir) / "evaluation_report.md"
    def agg(rows, checkpoint=None):
        subset = [r for r in rows if checkpoint is None or int(r["checkpoint"]) == checkpoint]
        if not subset:
            return None
        return {
            "n": len(subset), "sr": np.mean([int(r["success"]) for r in subset]),
            "spl": np.mean([_to_float(r, "spl") for r in subset]),
            "cls": np.mean([_to_float(r, "min_distance") for r in subset]),
            "bal": np.mean([_to_float(r, "balance_metric") for r in subset]),
            "path": np.mean([_to_float(r, "path_length") for r in subset]),
        }
    strict_best = ranked[0]
    # Aggregate the five evaluation seeds before applying the composite rank.
    # `_summary_from_rows` is intentionally seed-level for the CSV artifact, so
    # it must not be fed directly into a final checkpoint ranking.
    final_aggregate = []
    for checkpoint in sorted({int(r["checkpoint"]) for r in final_rows}):
        a = agg(final_rows, checkpoint)
        final_aggregate.append({
            "phase": "top5_200", "seed": "all", "checkpoint": checkpoint,
            "episodes": a["n"], "success_rate": a["sr"], "spl": a["spl"],
            "cls_m": a["cls"], "balance_metric_percent": a["bal"],
            "average_path_length_m": a["path"],
        })
    _, final_ranked = _rank_checkpoints(final_aggregate)
    final_best = final_ranked[0] if final_ranked else strict_best
    failure_counts = _counts([r for r in final_rows if int(r["success"]) == 0], "failure_mode")
    dominant_failure = max(failure_counts, key=failure_counts.get) if failure_counts else "none"
    lines = [
        "# Strict Point-to-Point Evaluation Report",
        "",
        "## Scope and protocol",
        "",
        f"- Task: `{TASK}`; checkpoint directory: `{Path(output_dir).parent}`",
        f"- Formal success: distance <= {DISTANCE_THRESHOLD:.2f} m AND speed <= {SPEED_THRESHOLD:.2f} m/s.",
        "- Evaluation only: no training and no reward/PPO/policy/history/curriculum/URDF/controller/target-sampler changes.",
        "- Observation noise, friction randomization, push disturbance, initial velocity randomization, and target curriculum were disabled for evaluation.",
        "- Actor inference used deterministic actor mean in eval mode.",
        "- Each seed has one scenario manifest replayed across checkpoints, so target and initial state are paired.",
        "",
        "## 1. Best checkpoint under strict 0.20 m",
        "",
        f"The screening winner by the composite of strict SR, SPL, CLS, balance, and path length is **checkpoint {int(strict_best['checkpoint'])}**: SR={float(strict_best['success_rate']):.1%}, SPL={float(strict_best['spl']):.4f}, CLS={float(strict_best['cls_m']):.4f} m, balance={float(strict_best['balance_metric_percent']):.2f}%, path={float(strict_best['average_path_length_m']):.3f} m.",
        "",
        "## 2. Is model_2050 still the best综合 model?",
        "",
        f"Across the 200-episode Top5 evaluation, the same composite ranking places **checkpoint {int(final_best['checkpoint'])}** first (SR={float(final_best['success_rate']):.1%}, SPL={float(final_best['spl']):.4f}, CLS={float(final_best['cls_m']):.4f} m). Therefore model_2050 is {'still' if int(final_best['checkpoint']) == 2050 else 'not'} the best综合 model under this strict protocol. `2050`, `2150`, and `3100` were retained regardless of ranking for paired diagnosis.",
        "",
        "## 3. Failure-mode diagnosis",
        "",
        "Failure modes are mutually prioritized as instability/out-of-bounds, overshoot, reached-but-too-fast, slow-timeout refinement, then never-reached. F4 is a diagnostic refinement of timeout episodes that did not enter 0.20 m but showed stable, sustained progress; it is kept separate from the residual F1 count.",
        "",
        "| Failure mode | Final Top5 count |",
        "|---|---:|",
    ]
    for key in ["F1 Never Reached", "F2 Reached But Too Fast", "F3 Overshoot", "F4 Slow Timeout", "F5 Instability"]:
        lines.append(f"| {key} | {failure_counts.get(key, 0)} |")
    lines += [
        "",
        "## 4. 2050 / 2150 / 3100 behavior comparison",
        "",
        "See `top5_200/paired_comparison.csv` and `paired_winners.csv` for episode-level paired records and winner counts. The comparison uses the same targets and initial states within each seed.",
        "",
        "| Checkpoint | Episodes | Strict SR | SPL | CLS (m) | Balance (%) | Path (m) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ck in MANDATORY_CHECKPOINTS:
        a = agg(final_rows, ck)
        if a:
            lines.append(f"| {ck} | {a['n']} | {a['sr']:.1%} | {a['spl']:.4f} | {a['cls']:.4f} | {a['bal']:.2f} | {a['path']:.3f} |")
    paired_win_counts = {ck: {metric: 0 for metric in ["success", "path_length", "min_distance", "overshoot_count", "episode_time"]} for ck in MANDATORY_CHECKPOINTS}
    paired_groups = {}
    for row in final_rows:
        ck = int(row["checkpoint"])
        if ck in MANDATORY_CHECKPOINTS:
            paired_groups.setdefault((int(row["seed"]), int(row["episode_id"])), {})[ck] = row
    for by_ck in paired_groups.values():
        if not all(ck in by_ck for ck in MANDATORY_CHECKPOINTS):
            continue
        for metric, lower_is_better in [("success", False), ("path_length", True), ("min_distance", True), ("overshoot_count", True), ("episode_time", True)]:
            values = {ck: (float(int(by_ck[ck][metric])) if metric == "success" else _to_float(by_ck[ck], metric)) for ck in MANDATORY_CHECKPOINTS}
            best_value = (min if lower_is_better else max)(values.values())
            for ck, value in values.items():
                if value == best_value:
                    paired_win_counts[ck][metric] += 1
    lines += [
        "",
        "Paired winner counts across the 200 matched episodes (ties count for every tied checkpoint):",
        "",
        "| Checkpoint | Success | Shorter path | Smaller min distance | Fewer overshoots | Shorter episode |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for ck in MANDATORY_CHECKPOINTS:
        w = paired_win_counts[ck]
        lines.append(f"| {ck} | {w['success']} | {w['path_length']} | {w['min_distance']} | {w['overshoot_count']} | {w['episode_time']} |")
    lines += [
        "",
        f"The dominant final failure mode is **{dominant_failure}** ({failure_counts.get(dominant_failure, 0)} episodes).",
        "",
        "## 5. Next factor worth testing",
        "",
        ("The dominant pattern supports testing braking/near-goal stopping first, because the failure is speed control after entering or crossing the strict region."
         if dominant_failure in ("F2 Reached But Too Fast", "F3 Overshoot") else
         "The dominant pattern supports testing curriculum/approach behavior first, because the failure is reaching the strict region within the fixed timeout."
         if dominant_failure in ("F1 Never Reached", "F4 Slow Timeout") else
         "The dominant pattern supports a stability/controller diagnosis first; no training-factor conclusion should be drawn until this is separated from runtime instability."
         if dominant_failure == "F5 Instability" else
         "No failures were observed in the final set; retain the current training setup and expand robustness evaluation before changing a factor."),
        "Reward changes remain downstream of this failure-mode check; this evaluation alone does not justify changing reward, PPO, or network structure.",
        "",
        "## Artifacts",
        "",
        "- `screening_40/checkpoint_summary.csv` and `episode_details.csv`",
        "- `top5_200/checkpoint_summary.csv`, `episode_details.csv`, and paired comparison files",
        "- `trajectories/` and `figures/` contain selected failure, overshoot, low-SPL, and paired diagnostic episodes",
        "- Scenario manifests are under `scenarios/`",
        "",
        "## Caveats",
        "",
        "- The 40-episode screening stage is for ranking only; the multi-seed 200-episode stage is the stronger comparison.",
        "- Isaac Gym Preview 4/PyTorch 1.10 compatibility was handled by an evaluation-process-only runtime shim for RTX 4070; Python, PyTorch, CUDA, and Isaac Gym were not upgraded.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote report:", out)


def analyze_final(output_dir):
    screening_rows = _load_all_details(output_dir, "screening_40")
    final_rows = _load_all_details(output_dir, "top5_200")
    if not final_rows:
        raise ValueError("No final workers found")
    summary_rows = _summary_from_rows(final_rows, "top5_200")
    out = Path(output_dir) / "top5_200"
    out.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(out / "checkpoint_summary.csv", summary_rows)
    _write_csv(out / "episode_details.csv", final_rows, DETAIL_FIELDS)
    _write_pairwise(output_dir, final_rows, "top5_200")
    _materialize_selected_traces(output_dir, "screening_40", screening_rows)
    _materialize_selected_traces(output_dir, "top5_200", final_rows)
    _materialize_pair_difference_traces(output_dir, final_rows)
    _, ranked = _rank_checkpoints(_summary_from_rows(screening_rows, "screening_40"))
    top5 = [int(r["checkpoint"]) for r in ranked[:5]]
    _write_report(output_dir, screening_rows, final_rows, top5, ranked)


def main():
    args = _parse_user_args()
    args.run_dir = str(Path(args.run_dir).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    if args.scenario_file:
        args.scenario_file = str(Path(args.scenario_file).resolve())
    output_dir = Path(args.output_dir)
    if args.mode == "generate-scenarios":
        if not args.scenario_file:
            raise ValueError("--scenario-file is required")
        generate_scenarios(
            args.seed, args.episodes, args.scenario_file,
            uniform_targets=args.uniform_targets,
        )
    elif args.mode == "worker":
        if args.checkpoint is None or not args.scenario_file:
            raise ValueError("worker requires --checkpoint and --scenario-file")
        evaluate_worker(args, args.checkpoint, args.seed, args.episodes,
                        args.scenario_file, args.phase, output_dir)
    elif args.mode == "analyze-screening":
        analyze_screening(output_dir)
    elif args.mode == "analyze-final":
        analyze_final(output_dir)
    else:
        raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
