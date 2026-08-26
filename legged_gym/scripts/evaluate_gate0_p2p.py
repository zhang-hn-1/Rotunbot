"""Gate0: evaluate an existing frozen P2P policy on an empty plane.

The protocol is intentionally separate from maze evaluation:

* one environment at a time;
* random target from the complete P2P target range;
* random initial yaw;
* no PPO updates and no scene obstacles;
* success is the task's terminal ``success_buf``;
* timeout is the task's terminal timeout buffer.
"""

import os
from pathlib import Path

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch
import torch

from legged_gym.envs import *  # noqa: F401,F403 - registers all tasks
from legged_gym.utils import get_args, task_registry


DEFAULT_CHECKPOINT = (
    "/home/jason/SphericalRobot_LeggedGym-master-new-map/logs/"
    "rotunbot_target_repro/Aug16_03-32-20_uniform_t1_long1000_from3809/model_4650.pt"
)


def _yaw_from_quaternion(quat):
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def evaluate(args):
    checkpoint_path = Path(os.environ.get("GATE0_P2P_CHECKPOINT", DEFAULT_CHECKPOINT)).resolve()
    episodes = int(os.environ.get("GATE0_P2P_EPISODES", "50"))
    if episodes <= 0:
        raise ValueError("GATE0_P2P_EPISODES must be positive")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Gate0 checkpoint does not exist: {checkpoint_path}")

    checkpoint = int(checkpoint_path.stem.split("_")[-1])
    run_dir = str(checkpoint_path.parent)

    args.task = "rotunbot_target_repro"
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.commands.random_start_yaw = True
    env_cfg.commands.target_curriculum = False

    # Build an inference runner without creating a log directory, then load
    # the explicit external checkpoint path.  This avoids the legacy loader's
    # assumption that every run lives below this workspace's logs/ directory.
    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    env.data_print = False
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    runner.load(str(checkpoint_path))
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.reset()
    max_steps = int(env.max_episode_length)
    successes = 0
    timeouts = 0
    other_failures = 0
    lengths = []
    yaws = []
    failure_categories = {}
    failure_records = []
    completed = 0
    episode_steps = 0
    episode_target = None
    episode_start = None
    episode_min_distance = None
    episode_start_yaw = None

    print(
        "Gate0 evaluation: "
        f"checkpoint={checkpoint_path}, episodes={episodes}, device={env.device}, "
        f"success_distance<={env_cfg.evaluation.target_error_threshold:.2f}m, "
        f"stop_speed<={env_cfg.evaluation.stop_velocity_threshold:.2f}m/s",
        flush=True,
    )

    with torch.no_grad():
        while completed < episodes:
            if episode_steps == 0:
                episode_start = env.root_states[0, :2].clone()
                episode_target = env.commands[0, :2].clone()
                episode_start_yaw = float(_yaw_from_quaternion(env.base_quat)[0].item())
                yaws.append(episode_start_yaw)
                episode_min_distance = float(
                    torch.linalg.norm(episode_target - episode_start).item()
                )
            pre_step_distance = float(
                torch.linalg.norm(env.commands[0, :2] - env.root_states[0, :2]).item()
            )
            episode_min_distance = min(episode_min_distance, pre_step_distance)
            actions = policy(obs)
            obs, _, _, dones, _ = env.step(actions)
            episode_steps += 1

            if bool(dones[0].item()) or episode_steps >= max_steps:
                success = bool(env.success_buf[0].item())
                timeout = bool(env.terminal_timeout[0].item())
                # A safety cap is counted as timeout even if the simulator did
                # not emit its terminal flag on the same step.
                timeout = timeout or episode_steps >= max_steps
                successes += int(success)
                timeouts += int(timeout and not success)
                other_failures += int(not success and not timeout)
                lengths.append(episode_steps)
                final_distance = float(env.terminal_goal_dist[0].item())
                final_speed = float(env.terminal_speed[0].item())
                episode_min_distance = min(episode_min_distance, final_distance)
                if success:
                    category = "success"
                elif final_distance <= 0.5:
                    category = "near_goal_timeout"
                elif episode_min_distance <= 0.5:
                    category = "oscillation"
                elif final_distance < 0.8 * float(
                    torch.linalg.norm(episode_target - episode_start).item()
                ):
                    category = "slow_convergence"
                else:
                    category = "control_failure"
                failure_categories[category] = failure_categories.get(category, 0) + 1
                if category != "success":
                    failure_records.append(
                        {
                            "episode": completed + 1,
                            "category": category,
                            "initial_yaw": episode_start_yaw,
                            "target_distance": float(
                                torch.linalg.norm(episode_target - episode_start).item()
                            ),
                            "min_distance": episode_min_distance,
                            "final_distance": final_distance,
                            "final_speed": final_speed,
                            "steps": episode_steps,
                            "timeout": timeout,
                        }
                    )
                completed += 1
                if category != "success":
                    print(
                        f"failure episode={completed:02d} category={category} "
                        f"min_dist={episode_min_distance:.3f} "
                        f"final_dist={final_distance:.3f} "
                        f"final_speed={final_speed:.3f} steps={episode_steps}",
                        flush=True,
                    )
                elif completed <= 5 or completed % 10 == 0 or completed == episodes:
                    print(
                        f"episode={completed:02d} success={int(success)} "
                        f"timeout={int(timeout and not success)} "
                        f"steps={episode_steps}",
                        flush=True,
                    )
                episode_steps = 0

    yaw_tensor = torch.tensor(yaws)
    success_rate = successes / episodes
    timeout_rate = timeouts / episodes
    print(
        "GATE0 SUMMARY "
        f"episodes={episodes} success_rate={success_rate:.2%} "
        f"timeout_rate={timeout_rate:.2%} "
        f"other_failure_rate={other_failures / episodes:.2%} "
        f"mean_steps={sum(lengths) / len(lengths):.1f} "
        f"yaw_min={yaw_tensor.min().item():.3f} "
        f"yaw_max={yaw_tensor.max().item():.3f} "
        f"yaw_std={yaw_tensor.std(unbiased=False).item():.3f}",
        flush=True,
    )
    print(f"failure_categories={failure_categories}", flush=True)
    if failure_records:
        print("failure_records:", flush=True)
        for record in failure_records:
            print(record, flush=True)

    if success_rate < 0.95 or timeout_rate > 0.05:
        raise RuntimeError(
            "Gate0 failed: required success_rate >= 95% and timeout_rate <= 5%"
        )
    print("Gate0 PASS", flush=True)


if __name__ == "__main__":
    evaluate(get_args())
