"""Record the complete rotunbot_target_depth dynamics for one checkpoint.

This diagnostic runner deliberately keeps the policy, reward, fixed maze, and
depth-fallback observation path unchanged.  It records every policy step of a
complete episode to NPZ/CSV and creates plots for the world trajectory,
body-frame velocity, yaw, DOFs, actions, torques, clearance, and contacts.
"""

import argparse
import csv
import json
import os
import sys

import isaacgym  # noqa: F401 (must be imported before task registration)
from isaacgym import gymapi, gymtorch
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


TASK_NAME = "rotunbot_target_depth"
BODY_NAMES = ("base_link", "link1", "link2")


def _parse_args():
    diagnostic_parser = argparse.ArgumentParser(add_help=False)
    diagnostic_parser.add_argument("--diag_checkpoint", type=int, required=True)
    diagnostic_parser.add_argument("--diag_episodes", type=int, default=1)
    diagnostic_parser.add_argument("--diag_output_dir", type=str, default=None)
    diagnostic_parser.add_argument(
        "--diag_max_steps",
        type=int,
        default=0,
        help="Override the episode step limit; 0 uses the task episode length.",
    )

    original_argv = list(sys.argv)
    diagnostic_args, remaining_argv = diagnostic_parser.parse_known_args()
    # Do not inherit get_args()'s generic default (anymal_c_flat).  This
    # diagnostic is only valid for the fixed rotunbot_target_depth task.
    has_task_argument = any(
        token == "--task" or token.startswith("--task=")
        for token in remaining_argv
    )
    if not has_task_argument:
        remaining_argv.extend(("--task", TASK_NAME))
    sys.argv = [original_argv[0]] + remaining_argv
    try:
        args = get_args()
    finally:
        sys.argv = original_argv

    if not args.load_run:
        raise ValueError("Diagnostics require --load_run RUN_DIR.")
    if diagnostic_args.diag_checkpoint < 0:
        raise ValueError("diag_checkpoint must be non-negative.")
    if diagnostic_args.diag_episodes <= 0:
        raise ValueError("diag_episodes must be positive.")
    if diagnostic_args.diag_max_steps < 0:
        raise ValueError("diag_max_steps must be non-negative.")

    if args.task != TASK_NAME:
        raise ValueError(
            "This diagnostic only supports --task rotunbot_target_depth; "
            f"got {args.task!r}."
        )
    args.num_envs = 1
    args.checkpoint = diagnostic_args.diag_checkpoint
    args.diag_episodes = diagnostic_args.diag_episodes
    args.diag_output_dir = diagnostic_args.diag_output_dir
    args.diag_max_steps = diagnostic_args.diag_max_steps
    return args


def _as_numpy(value, dtype=np.float32):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype).copy()


def _setup_rigid_body_reader(env):
    """Acquire robot rigid-body states and map the three URDF bodies."""
    try:
        body_names = list(
            env.gym.get_actor_rigid_body_names(
                env.envs[0], env.actor_handles[0]
            )
        )
    except Exception:
        body_names = list(BODY_NAMES)

    body_indices = []
    for body_name in BODY_NAMES:
        try:
            body_handle = env.gym.find_actor_rigid_body_handle(
                env.envs[0], env.actor_handles[0], body_name
            )
            body_indices.append(
                env.gym.get_actor_rigid_body_index(
                    env.envs[0],
                    env.actor_handles[0],
                    body_handle,
                    gymapi.DOMAIN_SIM,
                )
            )
        except Exception as exc:
            print(f"Warning: rigid-body diagnostics unavailable ({exc})")
            return None, body_names, None

    try:
        tensor = gymtorch.wrap_tensor(
            env.gym.acquire_rigid_body_state_tensor(env.sim)
        )
    except Exception as exc:
        print(f"Warning: rigid-body diagnostics unavailable ({exc})")
        return None, body_names, body_indices
    return tensor, body_names, body_indices


def _read_rigid_bodies(env, tensor, body_indices):
    if tensor is None or body_indices is None:
        return np.full((len(BODY_NAMES), 13), np.nan, dtype=np.float32)
    try:
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        return _as_numpy(tensor[body_indices])
    except Exception:
        return np.full((len(BODY_NAMES), 13), np.nan, dtype=np.float32)


def _empty_terminal_value(size=1):
    return np.full((size,), np.nan, dtype=np.float32)


def _capture_record(
    env,
    step,
    time_s,
    policy_action,
    rigid_body_tensor,
    body_indices,
    done,
):
    """Capture one policy-step sample without changing environment state."""
    root_states = _as_numpy(env.root_states[0])
    base_lin_vel = _as_numpy(env.base_lin_vel[0])
    base_ang_vel = _as_numpy(env.base_ang_vel[0])
    base_euler = _as_numpy(env.base_euler_tensor[0])
    dof_pos = _as_numpy(env.dof_pos[0])
    dof_vel = _as_numpy(env.dof_vel[0])
    contact_forces = _as_numpy(env.contact_forces[0])
    rigid_bodies = _read_rigid_bodies(env, rigid_body_tensor, body_indices)

    terminal_position = _empty_terminal_value(2)
    terminal_goal_distance = _empty_terminal_value()
    terminal_speed = _empty_terminal_value()
    terminal_clearance = _empty_terminal_value()
    success = np.array([0], dtype=np.int8)
    collision = np.array([0], dtype=np.int8)
    timeout = np.array([0], dtype=np.int8)
    unstable = np.array([0], dtype=np.int8)
    out_of_bounds = np.array([0], dtype=np.int8)

    if done:
        # env.step() automatically resets a terminated environment.  Preserve
        # the terminal position/metrics and mark reset-time state fields NaN so
        # the following plots never mistake the new episode start for the
        # previous episode's terminal pose.
        terminal_position = _as_numpy(env.terminal_position[0])
        terminal_goal_distance[:] = float(env.terminal_goal_dist[0].item())
        terminal_speed[:] = float(env.terminal_speed[0].item())
        terminal_clearance[:] = float(env.terminal_obstacle_clearance[0].item())
        success[:] = int(env.success_buf[0].item())
        collision[:] = int(env.step_collision_buf[0].item())
        timeout[:] = int(env.terminal_timeout[0].item())
        unstable[:] = int(env.terminal_unstable[0].item())
        out_of_bounds[:] = int(env.terminal_out_of_bounds[0].item())

        root_states[0:2] = terminal_position
        root_states[3:] = np.nan
        base_lin_vel[:] = np.nan
        base_ang_vel[:] = np.nan
        base_euler[:] = np.nan
        dof_pos[:] = np.nan
        dof_vel[:] = np.nan
        contact_forces[:] = np.nan
        rigid_bodies[:] = np.nan

    return {
        "step": np.int64(step),
        "time_s": np.float64(time_s),
        "root_states": root_states,
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
        "base_euler": base_euler,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "policy_action": _as_numpy(policy_action),
        "output_action": _as_numpy(env.output_actions[0]),
        "torques": _as_numpy(env.torques[0]),
        "commands": _as_numpy(env.commands[0]),
        "goal_dist": np.float32(
            terminal_goal_distance[0]
            if done
            else float(env.goal_dist[0].item())
        ),
        "obstacle_clearance": np.float32(
            terminal_clearance[0]
            if done
            else float(env.obstacle_clearance[0].item())
        ),
        "contact_forces": contact_forces,
        "rigid_body_states": rigid_bodies,
        "done": np.int8(done),
        "success": success[0],
        "collision": collision[0],
        "timeout": timeout[0],
        "unstable": unstable[0],
        "out_of_bounds": out_of_bounds[0],
        "terminal_position": terminal_position,
        "terminal_goal_distance": terminal_goal_distance[0],
        "terminal_speed": terminal_speed[0],
        "terminal_clearance": terminal_clearance[0],
    }


def _stack_records(records):
    return {
        key: np.stack([record[key] for record in records], axis=0)
        for key in records[0]
    }


def _write_csv(path, data):
    root = data["root_states"]
    base_lin = data["base_lin_vel"]
    base_ang = data["base_ang_vel"]
    euler = data["base_euler"]
    dof_pos = data["dof_pos"]
    dof_vel = data["dof_vel"]
    policy_action = data["policy_action"]
    output_action = data["output_action"]
    torques = data["torques"]
    commands = data["commands"]
    contact = data["contact_forces"]

    fields = ["step", "time_s"]
    fields += [f"root_{name}" for name in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    fields += [f"root_vel_world_{name}" for name in ("x", "y", "z")]
    fields += [f"root_ang_vel_world_{name}" for name in ("x", "y", "z")]
    fields += [f"base_lin_vel_body_{name}" for name in ("x", "y", "z")]
    fields += [f"base_ang_vel_body_{name}" for name in ("x", "y", "z")]
    fields += [f"base_euler_{name}" for name in ("roll", "pitch", "yaw")]
    fields += ["dof_pos_1", "dof_pos_2", "dof_vel_1", "dof_vel_2"]
    fields += ["policy_action_1", "policy_action_2"]
    fields += ["output_action_1", "output_action_2", "torque_1", "torque_2"]
    fields += ["command_1", "command_2", "command_3", "goal_dist", "obstacle_clearance"]
    fields += ["contact_norm_base_link", "contact_norm_link1", "contact_norm_link2"]
    fields += [
        "done",
        "success",
        "collision",
        "timeout",
        "unstable",
        "out_of_bounds",
        "terminal_x",
        "terminal_y",
        "terminal_goal_distance",
        "terminal_speed",
        "terminal_clearance",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index in range(len(data["step"])):
            row = {
                "step": int(data["step"][index]),
                "time_s": float(data["time_s"][index]),
            }
            row.update(
                dict(zip(fields[2:9], root[index, 0:7])),
                **dict(zip(fields[9:12], root[index, 7:10])),
                **dict(zip(fields[12:15], root[index, 10:13])),
                **dict(zip(fields[15:18], base_lin[index])),
                **dict(zip(fields[18:21], base_ang[index])),
                **dict(zip(fields[21:24], euler[index])),
            )
            row.update(
                dict(zip(fields[24:28], np.r_[dof_pos[index], dof_vel[index]])),
                **dict(zip(fields[28:30], policy_action[index])),
                **dict(zip(fields[30:34], np.r_[output_action[index], torques[index]])),
                **dict(zip(fields[34:37], commands[index, :3])),
            )
            contact_norm = np.linalg.norm(contact[index, :3], axis=1)
            row.update(
                {
                    "goal_dist": float(data["goal_dist"][index]),
                    "obstacle_clearance": float(data["obstacle_clearance"][index]),
                    "contact_norm_base_link": float(contact_norm[0]),
                    "contact_norm_link1": float(contact_norm[1]),
                    "contact_norm_link2": float(contact_norm[2]),
                    "done": int(data["done"][index]),
                    "success": int(data["success"][index]),
                    "collision": int(data["collision"][index]),
                    "timeout": int(data["timeout"][index]),
                    "unstable": int(data["unstable"][index]),
                    "out_of_bounds": int(data["out_of_bounds"][index]),
                    "terminal_x": float(data["terminal_position"][index, 0]),
                    "terminal_y": float(data["terminal_position"][index, 1]),
                    "terminal_goal_distance": float(
                        data["terminal_goal_distance"][index]
                    ),
                    "terminal_speed": float(data["terminal_speed"][index]),
                    "terminal_clearance": float(data["terminal_clearance"][index]),
                }
            )
            writer.writerow(row)


def _write_plot(path, data, goal_xy, checkpoint, episode_index):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: plot unavailable ({exc})")
        return

    time_s = data["time_s"]
    root = data["root_states"]
    base_lin = data["base_lin_vel"]
    base_ang = data["base_ang_vel"]
    euler = data["base_euler"]
    dof_pos = data["dof_pos"]
    dof_vel = data["dof_vel"]
    output_action = data["output_action"]
    torques = data["torques"]
    contact_norm = np.linalg.norm(data["contact_forces"][:, :3], axis=2)

    figure, axes = plt.subplots(3, 2, figsize=(16, 15))
    axes[0, 0].plot(root[:, 0], root[:, 1], linewidth=1.8, label="base_link trajectory")
    axes[0, 0].scatter(root[0, 0], root[0, 1], c="green", label="start")
    axes[0, 0].scatter(goal_xy[0], goal_xy[1], c="red", label="goal")
    axes[0, 0].set_title("World XY trajectory")
    axes[0, 0].set_xlabel("world x [m]")
    axes[0, 0].set_ylabel("world y [m]")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(time_s, base_lin[:, 0], label="body vx")
    axes[0, 1].plot(time_s, base_lin[:, 1], label="body vy")
    axes[0, 1].plot(time_s, root[:, 7], "--", label="world vx")
    axes[0, 1].plot(time_s, root[:, 8], "--", label="world vy")
    axes[0, 1].set_title("World/body velocity")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("velocity [m/s]")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(time_s, euler[:, 2], label="base yaw")
    axes[1, 0].plot(time_s, base_ang[:, 2], label="body yaw rate")
    axes[1, 0].set_title("Base orientation and yaw rate")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(time_s, dof_pos[:, 0], label="joint1 pos")
    axes[1, 1].plot(time_s, dof_pos[:, 1], label="joint2 pos")
    axes[1, 1].plot(time_s, dof_vel[:, 0], "--", label="joint1 vel")
    axes[1, 1].plot(time_s, dof_vel[:, 1], "--", label="joint2 vel")
    axes[1, 1].set_title("Joint state")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    axes[2, 0].plot(time_s, output_action[:, 0], label="output action 1")
    axes[2, 0].plot(time_s, output_action[:, 1], label="output action 2")
    axes[2, 0].plot(time_s, torques[:, 0], "--", label="torque 1")
    axes[2, 0].plot(time_s, torques[:, 1], "--", label="torque 2")
    axes[2, 0].set_title("Controller targets and torques")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].grid(True)
    axes[2, 0].legend()

    axes[2, 1].plot(time_s, data["obstacle_clearance"], label="clearance")
    axes[2, 1].plot(time_s, contact_norm[:, 0], label="contact base_link")
    axes[2, 1].plot(time_s, contact_norm[:, 1], label="contact link1")
    axes[2, 1].plot(time_s, contact_norm[:, 2], label="contact link2")
    axes[2, 1].set_title("Clearance and contact force")
    axes[2, 1].set_xlabel("time [s]")
    axes[2, 1].grid(True)
    axes[2, 1].legend()

    figure.suptitle(
        f"rotunbot_target_depth | checkpoint={checkpoint} | episode={episode_index}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _close_env(env):
    if env is None:
        return
    try:
        if getattr(env, "viewer", None) is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)
    except Exception as exc:
        print(f"Warning while closing Isaac Gym environment: {exc}")


def _run_episode(policy, env, episode_index, checkpoint, output_dir, args, rigid_body_tensor, body_indices, goal_xy):
    obs, _ = env.reset()
    policy_dt = float(env.sim_params.dt * env.cfg.control.decimation)
    max_steps = int(args.diag_max_steps or (env.max_episode_length + 1))
    records = []
    zero_action = torch.zeros(env.num_actions, device=env.device)
    records.append(
        _capture_record(
            env,
            step=0,
            time_s=0.0,
            policy_action=zero_action,
            rigid_body_tensor=rigid_body_tensor,
            body_indices=body_indices,
            done=False,
        )
    )

    for step in range(1, max_steps + 1):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, dones, _ = env.step(actions)
        done = bool(dones[0].item())
        records.append(
            _capture_record(
                env,
                step=step,
                time_s=step * policy_dt,
                policy_action=actions[0],
                rigid_body_tensor=rigid_body_tensor,
                body_indices=body_indices,
                done=done,
            )
        )
        if done:
            break

    data = _stack_records(records)
    stem = f"checkpoint_{checkpoint}_episode_{episode_index:02d}"
    np.savez_compressed(
        os.path.join(output_dir, stem + ".npz"),
        **data,
        goal_xy=np.asarray(goal_xy, dtype=np.float32),
    )
    _write_csv(os.path.join(output_dir, stem + ".csv"), data)
    _write_plot(
        os.path.join(output_dir, stem + ".png"),
        data,
        goal_xy,
        checkpoint,
        episode_index,
    )

    terminal_index = np.flatnonzero(data["done"])
    if len(terminal_index):
        index = int(terminal_index[-1])
        summary = {
            "checkpoint": int(checkpoint),
            "episode": int(episode_index),
            "steps_recorded": int(len(data["step"])),
            "success": int(data["success"][index]),
            "collision": int(data["collision"][index]),
            "timeout": int(data["timeout"][index]),
            "terminal_x": float(data["terminal_position"][index, 0]),
            "terminal_y": float(data["terminal_position"][index, 1]),
            "terminal_goal_distance": float(data["terminal_goal_distance"][index]),
            "terminal_speed": float(data["terminal_speed"][index]),
        }
    else:
        summary = {
            "checkpoint": int(checkpoint),
            "episode": int(episode_index),
            "steps_recorded": int(len(data["step"])),
            "terminated": False,
        }
    return summary


def main():
    args = _parse_args()
    run_dir = os.path.abspath(args.load_run)
    # task_registry.get_load_path() joins relative --load_run values with the
    # task experiment root.  Pass the already-resolved directory so a command
    # such as logs/rotunbot_target_depth/<run> is not prefixed twice.
    args.load_run = run_dir
    model_path = os.path.join(run_dir, f"model_{args.checkpoint}.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    output_dir = os.path.abspath(
        args.diag_output_dir
        or os.path.join(run_dir, "dynamics_diagnostics", f"checkpoint_{args.checkpoint}")
    )
    os.makedirs(output_dir, exist_ok=True)

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    env_cfg.env.num_envs = 1
    # Preserve the same deterministic policy input used by headless training.
    env_cfg.camera.enable = False
    env_cfg.enable_camera_sensors_in_headless = False
    env_cfg.camera.add_noise = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.commands.target_curriculum = False
    env_cfg.commands.random_start_yaw = False

    args.checkpoint = int(args.checkpoint)
    train_cfg.runner.resume = True
    env = None
    runner = None
    summaries = []
    try:
        env, _ = task_registry.make_env(
            name=TASK_NAME,
            args=args,
            env_cfg=env_cfg,
        )
        runner, _ = task_registry.make_alg_runner(
            env=env,
            name=TASK_NAME,
            args=args,
            train_cfg=train_cfg,
            log_root=None,
        )
        policy = runner.get_inference_policy(device=env.device)
        rigid_body_tensor, body_names, body_indices = _setup_rigid_body_reader(env)

        env.reset()
        goal_xy = _as_numpy(env.commands[0, :2])
        metadata = {
            "task": TASK_NAME,
            "checkpoint": int(args.checkpoint),
            "load_run": run_dir,
            "policy_dt_s": float(env.sim_params.dt * env.cfg.control.decimation),
            "sim_dt_s": float(env.sim_params.dt),
            "control_decimation": int(env.cfg.control.decimation),
            "goal_xy": goal_xy.tolist(),
            "body_names_from_sim": body_names,
            "robot_body_names_recorded": list(BODY_NAMES),
            "robot_body_indices_domain_sim": body_indices,
            "noise_enabled": bool(env_cfg.noise.add_noise),
            "camera_enabled": bool(env_cfg.camera.enable),
        }
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

        for episode_index in range(1, args.diag_episodes + 1):
            summary = _run_episode(
                policy,
                env,
                episode_index,
                args.checkpoint,
                output_dir,
                args,
                rigid_body_tensor,
                body_indices,
                goal_xy,
            )
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False))

        with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as file:
            json.dump(summaries, file, indent=2, ensure_ascii=False)
        print(f"Diagnostics written to: {output_dir}")
    finally:
        _close_env(env)
        if runner is not None:
            del runner


if __name__ == "__main__":
    main()
