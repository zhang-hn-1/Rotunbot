"""Evaluate rotunbot_target_depth checkpoints in a selected order.

This is an evaluation utility, not a new Isaac Gym task.  It creates one
fixed-maze environment per checkpoint, evaluates a fixed number of complete
episodes, and writes CSV/JSON/TensorBoard summaries.
"""

import argparse
import csv
import gc
import json
import os
import sys

import isaacgym  # noqa: F401 (must be imported before task registration)
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def _patch_tensorboard_pillow_compat():
    """Keep torch 1.10 TensorBoard compatible with Pillow >= 10.

    The TensorBoard bundled with the project's old PyTorch calls
    ``PIL.Image.ANTIALIAS``, which was removed in newer Pillow releases.
    This only affects image-summary encoding and does not change the image
    tensors or the policy observation path.
    """
    try:
        from PIL import Image

        if not hasattr(Image, "ANTIALIAS"):
            resampling = getattr(Image, "Resampling", None)
            Image.ANTIALIAS = (
                resampling.LANCZOS
                if resampling is not None
                else Image.LANCZOS
            )
    except Exception:
        # If Pillow is unavailable, TensorBoard will report its own image
        # dependency error when image summaries are requested.
        pass


_patch_tensorboard_pillow_compat()

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


TASK_NAME = "rotunbot_target_depth"


def parse_args():
    """Parse evaluator-only arguments together with the project arguments."""
    evaluator_parser = argparse.ArgumentParser(add_help=False)
    evaluator_parser.add_argument("--eval_start_checkpoint", type=int, required=True)
    evaluator_parser.add_argument("--eval_end_checkpoint", type=int, default=0)
    evaluator_parser.add_argument("--eval_step", type=int, default=50)
    evaluator_parser.add_argument("--eval_episodes", type=int, default=10)
    evaluator_parser.add_argument(
        "--eval_seed",
        type=int,
        default=None,
        help=(
            "Base seed for the evaluation scenario set. Fresh episode i uses "
            "base_seed+i-1; the same schedule is reused for every checkpoint."
        ),
    )
    evaluator_parser.add_argument("--eval_log_dir", type=str, default=None)
    evaluator_parser.add_argument("--eval_with_noise", action="store_true")
    evaluator_parser.add_argument(
        "--eval_use_camera_sensor",
        action="store_true",
        help=(
            "Compatibility alias for --eval_depth_source camera. Camera is "
            "already the default for the current depth-camera task."
        ),
    )
    evaluator_parser.add_argument(
        "--eval_depth_source",
        choices=("fallback", "camera"),
        default="camera",
        help=(
            "Depth source used by the policy. The default camera matches the "
            "current rotunbot_target_depth training configuration; pass "
            "--eval_depth_source fallback only for legacy fallback-trained runs."
        ),
    )
    evaluator_parser.add_argument(
        "--eval_compare_depth",
        action="store_true",
        help=(
            "Run the policy with the real GPU camera and log same-step "
            "fallback-vs-camera depth images and difference statistics."
        ),
    )
    evaluator_parser.add_argument(
        "--eval_compare_interval",
        type=int,
        default=10,
        help="Log one depth comparison every N policy steps (default: 10).",
    )
    evaluator_parser.add_argument(
        "--eval_compare_max_frames",
        type=int,
        default=100,
        help="Maximum comparison frames per checkpoint (default: 100).",
    )
    evaluator_parser.add_argument(
        "--eval_live_depth",
        action="store_true",
        help="Show fallback and GPU-camera depth images in a live OpenCV window.",
    )
    evaluator_parser.add_argument(
        "--eval_fresh_env_per_episode",
        action="store_true",
        help="Recreate Isaac Gym for every episode to remove reset/contact-cache state.",
    )
    evaluator_parser.add_argument(
        "--eval_reuse_env",
        action="store_true",
        help="Reuse one Isaac Gym environment and explicitly reset it before every episode.",
    )
    evaluator_parser.add_argument("--eval_overwrite", action="store_true")

    original_argv = list(sys.argv)
    evaluator_args, remaining_argv = evaluator_parser.parse_known_args()
    sys.argv = [original_argv[0]] + remaining_argv
    try:
        args = get_args()
    finally:
        sys.argv = original_argv

    if not args.load_run:
        raise ValueError("Evaluation requires --load_run RUN_DIR.")
    if evaluator_args.eval_start_checkpoint < 0:
        raise ValueError("eval_start_checkpoint must be non-negative.")
    if evaluator_args.eval_end_checkpoint < 0:
        raise ValueError("eval_end_checkpoint must be non-negative.")
    if evaluator_args.eval_step <= 0:
        raise ValueError("eval_step must be positive.")
    if evaluator_args.eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive.")
    if evaluator_args.eval_compare_interval <= 0:
        raise ValueError("eval_compare_interval must be positive.")
    if evaluator_args.eval_compare_max_frames <= 0:
        raise ValueError("eval_compare_max_frames must be positive.")

    args.task = TASK_NAME
    args.num_envs = 1
    args.eval_start_checkpoint = evaluator_args.eval_start_checkpoint
    args.eval_end_checkpoint = evaluator_args.eval_end_checkpoint
    args.eval_step = evaluator_args.eval_step
    args.eval_episodes = evaluator_args.eval_episodes
    args.eval_seed = evaluator_args.eval_seed
    args.eval_log_dir = evaluator_args.eval_log_dir
    args.eval_with_noise = evaluator_args.eval_with_noise
    # Live display needs both tensors as well, so make comparison capture
    # implicit when the user asks for the OpenCV window.
    args.eval_compare_depth = (
        evaluator_args.eval_compare_depth or evaluator_args.eval_live_depth
    )
    # ``--eval_use_camera_sensor`` is retained as a compatibility alias for
    # selecting the real camera as the policy input.  Comparing two sensors
    # must not silently change the policy input from fallback to camera.
    args.eval_depth_source = (
        "camera"
        if evaluator_args.eval_use_camera_sensor
        else evaluator_args.eval_depth_source
    )
    args.eval_use_camera_sensor = args.eval_depth_source == "camera"
    args.eval_compare_interval = evaluator_args.eval_compare_interval
    args.eval_compare_max_frames = evaluator_args.eval_compare_max_frames
    args.eval_live_depth = evaluator_args.eval_live_depth
    # Fixed-maze evaluation uses a newly reset simulator for each trial by
    # default. Reusing an environment remains available for fast diagnostics.
    args.eval_fresh_env_per_episode = (
        not evaluator_args.eval_reuse_env
        or evaluator_args.eval_fresh_env_per_episode
    )
    args.eval_overwrite = evaluator_args.eval_overwrite
    return args


def _close_env(env):
    if env is None:
        return
    try:
        if getattr(env, "viewer", None) is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)
    except Exception as exc:
        print(f"Warning while closing Isaac Gym environment: {exc}")


def _terminal_metrics(env, dones):
    """Read terminal buffers before starting the next episode."""
    done = bool(dones[0].item())
    if not done:
        position = env.root_states[0, :2].detach().clone()
        goal_distance = float(env.goal_dist[0].item())
        speed = float(torch.linalg.vector_norm(env.base_lin_vel[0]).item())
        return {
            "done": False,
            "position": position,
            "goal_distance": goal_distance,
            "speed": speed,
            "obstacle_clearance": float(env.obstacle_clearance[0].item()),
        }

    return {
        "done": True,
        "position": env.terminal_position[0].detach().clone(),
        "goal_distance": float(env.terminal_goal_dist[0].item()),
        "speed": float(env.terminal_speed[0].item()),
        "obstacle_clearance": float(
            env.terminal_obstacle_clearance[0].item()
        ),
        "success": bool(env.success_buf[0].item()),
        "collision": bool(env.step_collision_buf[0].item()),
        "timeout": bool(env.terminal_timeout[0].item()),
        "unstable": bool(env.terminal_unstable[0].item()),
        "out_of_bounds": bool(env.terminal_out_of_bounds[0].item()),
    }


def _show_live_depth(env, live_state):
    """Display fallback and real-camera depth side by side when possible."""
    if live_state is None or live_state.get("disabled", False):
        return
    try:
        import cv2

        fallback = env.depth_fallback_observation[0].detach().clamp(0.0, 1.0)
        camera = env.depth_camera_observation[0].detach().clamp(0.0, 1.0)
        fallback = (fallback.mul(255.0).round().byte().cpu().numpy())
        camera = (camera.mul(255.0).round().byte().cpu().numpy())

        colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        fallback = cv2.applyColorMap(fallback, colormap)
        camera = cv2.applyColorMap(camera, colormap)
        comparison = np.concatenate((fallback, camera), axis=1)
        # Upscale the low-resolution 32x64 policy image first.  Previously
        # labels were drawn before upscaling, so the text became several
        # times larger than the actual depth image.
        scale = 8
        comparison = cv2.resize(
            comparison,
            (comparison.shape[1] * scale, comparison.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        split_x = int(fallback.shape[1] * scale)
        label_height = max(30, int(34 * scale / 8))
        cv2.rectangle(comparison, (0, 0), (split_x, label_height), (0, 0, 0), -1)
        cv2.rectangle(
            comparison,
            (split_x, 0),
            (comparison.shape[1], label_height),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            comparison,
            "ray/AABB fallback",
            (10, int(label_height * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            comparison,
            "Isaac GPU camera",
            (split_x + 10, int(label_height * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        window_name = "Depth comparison: fallback | Isaac GPU camera"
        if not live_state.get("window_created", False):
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, comparison.shape[1], comparison.shape[0])
            live_state["window_created"] = True
        cv2.imshow(window_name, comparison)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            live_state["quit"] = True
            cv2.destroyWindow(window_name)
    except Exception as exc:
        if not live_state.get("warned", False):
            print(
                "Live depth window unavailable; TensorBoard comparison remains "
                f"enabled ({exc})."
            )
            live_state["warned"] = True
        live_state["disabled"] = True


def _run_single_episode(
    policy,
    env,
    checkpoint,
    episode_index,
    total_episodes,
    comparison_writer=None,
    comparison_state=None,
    comparison_interval=10,
    comparison_max_frames=100,
    live_depth=False,
    live_state=None,
):
    """Run one complete episode and return its diagnostic metrics."""
    obs = env.get_observations()
    episode_return = 0.0
    episode_length = 0
    path_length = 0.0
    min_goal_distance = float(env.goal_dist[0].item())
    initial_goal_distance = float(
        getattr(env, "maze_goal_distance", env.goal_dist)[0].item()
    )
    goal_bin = None
    if hasattr(env, "maze_goal_sampling_bin"):
        goal_bin = int(env.maze_goal_sampling_bin[0].item())
    sum_goal_distance = 0.0
    sum_speed = 0.0
    min_clearance = float(env.obstacle_clearance[0].item())
    max_speed = 0.0
    max_steps = int(env.max_episode_length) + 1

    for _ in range(max_steps):
        previous_position = env.root_states[0, :2].detach().clone()
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, rewards, dones, _ = env.step(actions)

        if live_depth:
            _show_live_depth(env, live_state)
            if live_state is not None and live_state.get("quit", False):
                raise KeyboardInterrupt("Depth comparison window closed by user.")

        if (
            comparison_writer is not None
            and comparison_state is not None
            and (episode_length % comparison_interval == 0)
            and comparison_state["frame"] < comparison_max_frames
        ):
            comparison_frame = int(comparison_state["frame"])
            fallback = env.depth_fallback_observation[0].detach().clamp(0.0, 1.0).cpu()
            camera = env.depth_camera_observation[0].detach().clamp(0.0, 1.0).cpu()
            valid = bool(env.depth_camera_valid[0].item())
            if not valid and not comparison_state["warned_no_camera"]:
                print(
                    "Warning: depth comparison requested, but the real camera "
                    "tensor is unavailable; comparison images are fallback copies."
                )
                comparison_state["warned_no_camera"] = True

            side_by_side = torch.cat((fallback, camera), dim=1).unsqueeze(0)
            difference = (fallback - camera).abs()
            comparison_writer.add_image(
                "DepthComparison/fallback_left_camera_right",
                side_by_side,
                comparison_frame,
            )
            comparison_writer.add_image(
                "DepthComparison/fallback",
                fallback.unsqueeze(0),
                comparison_frame,
            )
            comparison_writer.add_image(
                "DepthComparison/camera",
                camera.unsqueeze(0),
                comparison_frame,
            )
            comparison_writer.add_image(
                "DepthComparison/absolute_difference",
                difference.unsqueeze(0),
                comparison_frame,
            )
            comparison_writer.add_scalar(
                "DepthComparison/mean_absolute_difference",
                float(difference.mean().item()),
                comparison_frame,
            )
            comparison_writer.add_scalar(
                "DepthComparison/max_absolute_difference",
                float(difference.max().item()),
                comparison_frame,
            )
            comparison_writer.add_scalar(
                "DepthComparison/camera_tensor_valid",
                int(valid),
                comparison_frame,
            )
            comparison_writer.flush()
            print(
                f"depth_compare frame={comparison_frame} "
                f"episode={episode_index} valid={int(valid)} "
                f"mae={difference.mean().item():.5f} "
                f"max_abs={difference.max().item():.5f}"
            )
            comparison_state["frame"] = comparison_frame + 1

        episode_return += float(rewards[0].item())
        episode_length += 1
        terminal = _terminal_metrics(env, dones)
        path_length += float(
            torch.linalg.vector_norm(terminal["position"] - previous_position).item()
        )
        min_goal_distance = min(min_goal_distance, terminal["goal_distance"])
        sum_goal_distance += terminal["goal_distance"]
        sum_speed += terminal["speed"]
        min_clearance = min(min_clearance, terminal["obstacle_clearance"])
        max_speed = max(max_speed, terminal["speed"])

        if not terminal["done"]:
            continue

        episode = {
            "success": int(terminal["success"]),
            "collision": int(terminal["collision"]),
            "timeout": int(terminal["timeout"]),
            "unstable": int(terminal["unstable"]),
            "out_of_bounds": int(terminal["out_of_bounds"]),
            "terminal_goal_distance": terminal["goal_distance"],
            "terminal_speed": terminal["speed"],
            "terminal_obstacle_clearance": terminal["obstacle_clearance"],
            "terminal_x": float(terminal["position"][0].item()),
            "terminal_y": float(terminal["position"][1].item()),
            "episode_return": episode_return,
            "episode_length": episode_length,
            "path_length": path_length,
            "initial_goal_distance": initial_goal_distance,
            "goal_bin": goal_bin,
            "min_goal_distance": min_goal_distance,
            "mean_goal_distance": sum_goal_distance / max(episode_length, 1),
            "mean_speed": sum_speed / max(episode_length, 1),
            "min_obstacle_clearance": min_clearance,
            "max_speed": max_speed,
        }
        print(
            f"checkpoint={checkpoint} episode={episode_index}/{total_episodes} "
            f"goal_bin={episode['goal_bin']} "
            f"initial_distance={episode['initial_goal_distance']:.3f} "
            f"success={episode['success']} collision={episode['collision']} "
            f"steps={episode['episode_length']} "
            f"distance={episode['terminal_goal_distance']:.3f} "
            f"speed={episode['terminal_speed']:.3f} "
            f"min_clearance={episode['min_obstacle_clearance']:.3f} "
            f"max_speed={episode['max_speed']:.3f}"
        )
        return episode

    raise RuntimeError(
        f"Checkpoint {checkpoint} episode {episode_index} did not terminate "
        f"within {max_steps} steps."
    )


def _evaluate_checkpoint(args, checkpoint):
    run_dir = os.path.abspath(args.load_run)
    model_path = os.path.join(run_dir, f"model_{checkpoint}.pt")
    if not os.path.isfile(model_path):
        print(f"Skipping checkpoint {checkpoint}: {model_path} not found")
        return None

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    env_cfg.env.num_envs = 1
    evaluation_base_seed = int(
        getattr(args, "eval_seed", None)
        if getattr(args, "eval_seed", None) is not None
        else getattr(env_cfg, "seed", 3)
    )
    env_cfg.seed = evaluation_base_seed
    env_cfg.camera.add_noise = bool(args.eval_with_noise)
    # Apply the requested depth source to the CONFIG so the environment is
    # created with it (the env reads policy_source at construction; the
    # post-construction override is too late for the runner's initial reset).
    env_cfg.camera.policy_source = args.eval_depth_source
    # Keep evaluation on the same sensor distribution as training.  The
    # current task trains with the Isaac Gym GPU camera by default; fallback is
    # still available explicitly for legacy fallback-trained checkpoints.
    camera_required = bool(
        args.eval_compare_depth or args.eval_depth_source == "camera"
    )
    env_cfg.camera.enable = camera_required
    env_cfg.camera.capture_fallback = bool(args.eval_compare_depth)
    env_cfg.enable_camera_sensors_in_headless = camera_required
    env_cfg.noise.add_noise = bool(args.eval_with_noise)
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.commands.target_curriculum = False

    args.checkpoint = int(checkpoint)
    train_cfg.runner.resume = True
    env = None
    runner = None
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
        env.capture_depth_comparison = bool(args.eval_compare_depth)
        env.depth_policy_source = args.eval_depth_source
        obs = env.get_observations()

        comparison_writer = None
        comparison_state = {"frame": 0, "warned_no_camera": False}
        live_state = {"warned": False, "disabled": False, "quit": False}
        if args.eval_compare_depth:
            comparison_root = os.path.abspath(
                args.eval_log_dir
                or os.path.join(run_dir, "evaluation_depth_compare")
            )
            comparison_dir = os.path.join(
                comparison_root, "depth_comparison", f"checkpoint_{checkpoint}"
            )
            os.makedirs(comparison_dir, exist_ok=True)
            comparison_writer = SummaryWriter(log_dir=comparison_dir)

        episodes = []
        for episode_index in range(1, args.eval_episodes + 1):
            if episode_index > 1 and args.eval_fresh_env_per_episode:
                _close_env(env)
                env = None
                # Keep the maze layout fixed (maze.seed is unchanged), while
                # making each random-goal/yaw evaluation episode distinct and
                # reproducible across checkpoints.
                env_cfg.seed = evaluation_base_seed + episode_index - 1
                env, _ = task_registry.make_env(
                    name=TASK_NAME,
                    args=args,
                    env_cfg=env_cfg,
                )
            # Use the same complete reset protocol for both modes.  In reuse
            # mode this keeps one Isaac Gym instance alive while preventing
            # the next trial from relying only on step()'s implicit reset.
            # In fresh mode it also initializes the newly created instance.
            env.reset()
            episodes.append(
                _run_single_episode(
                    policy,
                    env,
                    checkpoint,
                    episode_index,
                    args.eval_episodes,
                    comparison_writer=comparison_writer,
                    comparison_state=comparison_state,
                    comparison_interval=args.eval_compare_interval,
                    comparison_max_frames=args.eval_compare_max_frames,
                    live_depth=args.eval_live_depth and not args.headless,
                    live_state=live_state,
                )
            )

        def avg(name):
            return float(np.mean([episode[name] for episode in episodes]))

        termination_counts = {
            name: int(sum(episode[name] for episode in episodes))
            for name in ("success", "collision", "timeout", "unstable", "out_of_bounds")
        }

        def group_avg(group, name):
            if not group:
                return float("nan")
            return float(np.mean([episode[name] for episode in group]))

        success_episodes = [episode for episode in episodes if episode["success"]]
        collision_episodes = [
            episode for episode in episodes if episode["collision"]
        ]
        goal_bin_names = ("near", "mid", "far")
        goal_bin_summaries = {}
        for bin_id, bin_name in enumerate(goal_bin_names):
            group = [episode for episode in episodes if episode["goal_bin"] == bin_id]
            goal_bin_summaries[f"{bin_name}_episodes"] = int(len(group))
            goal_bin_summaries[f"{bin_name}_success_rate"] = group_avg(
                group, "success"
            )

        summary = {
            "checkpoint": int(checkpoint),
            "evaluation_seed": int(evaluation_base_seed),
            "episodes": int(len(episodes)),
            "success_rate": avg("success"),
            "collision_rate": avg("collision"),
            "timeout_rate": avg("timeout"),
            "unstable_rate": avg("unstable"),
            "out_of_bounds_rate": avg("out_of_bounds"),
            "mean_terminal_goal_distance": avg("terminal_goal_distance"),
            "mean_terminal_speed": avg("terminal_speed"),
            "mean_terminal_obstacle_clearance": avg(
                "terminal_obstacle_clearance"
            ),
            "mean_terminal_x": avg("terminal_x"),
            "mean_terminal_y": avg("terminal_y"),
            "mean_episode_return": avg("episode_return"),
            "mean_episode_length": avg("episode_length"),
            "mean_path_length": avg("path_length"),
            "mean_min_goal_distance": avg("min_goal_distance"),
            "mean_goal_distance": avg("mean_goal_distance"),
            "mean_speed": avg("mean_speed"),
            "mean_min_obstacle_clearance": avg("min_obstacle_clearance"),
            "mean_max_speed": avg("max_speed"),
            # Keep success and collision behavior separate.  Overall means can
            # hide the relation between speed, wall clearance, and failure.
            "success_mean_max_speed": group_avg(success_episodes, "max_speed"),
            "collision_mean_max_speed": group_avg(collision_episodes, "max_speed"),
            "success_mean_min_obstacle_clearance": group_avg(
                success_episodes, "min_obstacle_clearance"
            ),
            "collision_mean_min_obstacle_clearance": group_avg(
                collision_episodes, "min_obstacle_clearance"
            ),
            "success_mean_episode_length": group_avg(
                success_episodes, "episode_length"
            ),
            "collision_mean_episode_length": group_avg(
                collision_episodes, "episode_length"
            ),
            "collision_mean_terminal_x": group_avg(collision_episodes, "terminal_x"),
            "collision_mean_terminal_y": group_avg(collision_episodes, "terminal_y"),
            "termination_counts": termination_counts,
        }
        summary.update(goal_bin_summaries)
        return summary
    finally:
        if "comparison_writer" in locals() and comparison_writer is not None:
            comparison_writer.close()
        del runner
        _close_env(env)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _write_outputs(
    summary,
    log_dir,
    writer,
    csv_writer,
    csv_file,
    tensorboard_step=None,
):
    checkpoint = summary["checkpoint"]
    # Checkpoints are evaluated in descending order (5350, 5300, ...), but
    # TensorBoard expects a monotonic global step.  Using the checkpoint as
    # global_step makes TensorBoard purge earlier points as out-of-order
    # restart events.  Keep checkpoint as a separate scalar and use the
    # monotonic evaluation index for all plotted metrics.
    plot_step = checkpoint if tensorboard_step is None else tensorboard_step
    json_path = os.path.join(log_dir, f"checkpoint_{checkpoint}.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    csv_row = dict(summary)
    csv_row["termination_counts"] = json.dumps(
        summary["termination_counts"], ensure_ascii=False
    )
    csv_writer.writerow(csv_row)
    csv_file.flush()

    for key, value in summary.items():
        if key == "termination_counts":
            for reason, count in value.items():
                writer.add_scalar(
                    f"Evaluation/termination_count/{reason}",
                    count,
                    plot_step,
                )
            continue
        writer.add_scalar(f"Evaluation/{key}", value, plot_step)
    writer.add_scalar("Evaluation/checkpoint", checkpoint, plot_step)
    writer.flush()
    print(
        f"checkpoint={checkpoint} "
        f"SR={summary['success_rate']:.2%} "
        f"collision={summary['collision_rate']:.2%} "
        f"timeout={summary['timeout_rate']:.2%} "
        f"distance={summary['mean_terminal_goal_distance']:.3f} "
        f"speed={summary['mean_terminal_speed']:.3f}"
    )


def main():
    args = parse_args()
    print(
        "Evaluation mode: "
        + (
            "fresh Isaac Gym environment per episode + explicit reset"
            if args.eval_fresh_env_per_episode
            else "reuse one Isaac Gym environment + explicit reset per episode"
        )
    )
    print(
        "Policy depth source: "
        + (
            "Isaac Gym GPU camera sensor"
            if args.eval_depth_source == "camera"
            else "deterministic maze fallback"
        )
    )
    if args.eval_compare_depth:
        print(
            "Depth comparison: fallback and Isaac GPU camera captured at the "
            "same state; policy input is not changed by --headless."
        )
    if args.eval_live_depth:
        print(
            "Live depth display: OpenCV window requested "
            "(press q or Esc to stop)."
        )
    run_dir = os.path.abspath(args.load_run)
    log_dir = os.path.abspath(
        args.eval_log_dir
        or os.path.join(run_dir, "evaluation_depth_reverse_nominal")
    )
    os.makedirs(log_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, "checkpoint_metrics.csv")
    fields = [
        "checkpoint",
        "evaluation_seed",
        "episodes",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "unstable_rate",
        "out_of_bounds_rate",
        "mean_terminal_goal_distance",
        "mean_terminal_speed",
        "mean_terminal_obstacle_clearance",
        "mean_terminal_x",
        "mean_terminal_y",
        "mean_episode_return",
        "mean_episode_length",
        "mean_path_length",
        "mean_min_goal_distance",
        "mean_goal_distance",
        "mean_speed",
        "mean_min_obstacle_clearance",
        "mean_max_speed",
        "success_mean_max_speed",
        "collision_mean_max_speed",
        "success_mean_min_obstacle_clearance",
        "collision_mean_min_obstacle_clearance",
        "success_mean_episode_length",
        "collision_mean_episode_length",
        "collision_mean_terminal_x",
        "collision_mean_terminal_y",
        "near_episodes",
        "near_success_rate",
        "mid_episodes",
        "mid_success_rate",
        "far_episodes",
        "far_success_rate",
        "termination_counts",
    ]
    csv_mode = "w" if args.eval_overwrite or not os.path.isfile(csv_path) else "a"
    existing = set()
    if csv_mode == "a":
        with open(csv_path, "r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                existing.add(int(row["checkpoint"]))

    writer = SummaryWriter(log_dir=log_dir)
    with open(csv_path, csv_mode, newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
        if csv_mode == "w":
            csv_writer.writeheader()
        tensorboard_step = 0
        if args.eval_start_checkpoint <= args.eval_end_checkpoint:
            checkpoint_range = range(
                args.eval_start_checkpoint,
                args.eval_end_checkpoint + 1,
                args.eval_step,
            )
        else:
            checkpoint_range = range(
                args.eval_start_checkpoint,
                args.eval_end_checkpoint - 1,
                -args.eval_step,
            )
        for checkpoint in checkpoint_range:
            if checkpoint in existing and not args.eval_overwrite:
                print(f"Skipping checkpoint {checkpoint}: already evaluated")
                continue
            summary = _evaluate_checkpoint(args, checkpoint)
            if summary is not None:
                _write_outputs(
                    summary,
                    log_dir,
                    writer,
                    csv_writer,
                    csv_file,
                    tensorboard_step=tensorboard_step,
                )
                tensorboard_step += 1
    writer.close()
    print(f"Evaluation results: {log_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
