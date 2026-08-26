"""Fixed world-pose/yaw evaluation for Robot-frame Local P2P."""

import json
import math
import os
import random
from pathlib import Path

import isaacgym  # noqa: F401
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.local_goal_metrics import aggregate_local_goal_records
from legged_gym.utils import get_args, task_registry


DEFAULT_CHECKPOINT = ""
DISTANCES = (0.5, 1.0, 1.5, 2.0, 3.0)
BEARINGS_DEG = (0.0, 45.0, 90.0, -45.0, -90.0, 180.0)
WORLD_POSITIONS = ((0.0, 0.0), (4.0, -3.0), (-6.0, 5.0), (8.0, 7.0))
YAWS_DEG = (0.0, 90.0, 180.0, -90.0)
REACH_RADIUS = 0.35


def evaluation_grid(stage):
    """Return the deterministic distance/bearing grid for a curriculum stage."""
    stage = str(stage).upper()
    if stage == "A":
        return (0.5, 1.0, 1.5, 2.0), (0.0, 45.0, -45.0)
    if stage == "B":
        return (0.5, 1.0, 1.5, 2.0, 2.5), (0.0, 45.0, 90.0, -45.0, -90.0)
    if stage == "C":
        return DISTANCES, BEARINGS_DEG
    raise ValueError(f"unknown local-goal evaluation stage: {stage}")


def _yaw_quaternion(yaw, device):
    half = 0.5 * torch.as_tensor(yaw, device=device)
    quat = torch.zeros(4, device=device)
    quat[2] = torch.sin(half)
    quat[3] = torch.cos(half)
    return quat


def _set_pose(env, position, yaw_deg):
    env.root_states[0, :2] = torch.as_tensor(position, device=env.device)
    env.root_states[0, 3:7] = _yaw_quaternion(math.radians(yaw_deg), env.device)
    env.root_states[0, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
    env.episode_length_buf[0] = 0
    env.reset_buf[0] = 0
    env.time_out_buf[0] = False
    env.last_actions[0] = 0.0
    env.actions[0] = 0.0
    env.last_dof_vel[0] = 0.0


def _configure_episode(env, distance, bearing_deg, position, yaw_deg):
    env.reset()
    _set_pose(env, position, yaw_deg)
    bearing = math.radians(bearing_deg)
    local_delta = torch.tensor(
        (distance * math.cos(bearing), distance * math.sin(bearing)),
        device=env.device,
    )
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    world_delta = torch.tensor(
        (c * float(local_delta[0]) - s * float(local_delta[1]),
         s * float(local_delta[0]) + c * float(local_delta[1])),
        device=env.device,
    )
    env.world_goal[0] = env.root_states[0, :2] + world_delta
    env.commands[0, :2] = env.world_goal[0]
    env.compute_observations()
    return env.get_observations()


def _fixed_cases(count, seed, distances=DISTANCES, bearings=BEARINGS_DEG):
    rng = random.Random(seed)
    cases = []
    for index in range(count):
        distance = distances[index % len(distances)]
        bearing = bearings[(index // len(distances)) % len(bearings)]
        yaw = YAWS_DEG[(index // (len(distances) * len(bearings))) % len(YAWS_DEG)]
        position = WORLD_POSITIONS[
            (index // (len(distances) * len(bearings) * len(YAWS_DEG)))
            % len(WORLD_POSITIONS)
        ]
        # Add a deterministic sub-grid offset while keeping the pose bounded.
        offset = (rng.random() - 0.5, rng.random() - 0.5)
        cases.append(
            {
                "distance": distance,
                "bearing_deg": bearing,
                "yaw_deg": yaw,
                "position": (position[0] + offset[0], position[1] + offset[1]),
            }
        )
    return cases


def evaluate(args):
    checkpoint_value = os.environ.get("LOCAL_GOAL_CHECKPOINT", DEFAULT_CHECKPOINT)
    if not checkpoint_value:
        raise ValueError("set LOCAL_GOAL_CHECKPOINT to a new rotunbot_local_goal checkpoint")
    checkpoint = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Local Goal checkpoint does not exist: {checkpoint}")
    episodes = int(os.environ.get("LOCAL_GOAL_EVAL_EPISODES", "500"))
    seed = int(os.environ.get("LOCAL_GOAL_EVAL_SEED", "17"))
    stage = os.environ.get("LOCAL_GOAL_EVAL_STAGE", "C").upper()
    distances, bearings = evaluation_grid(stage)
    report_path = Path(
        os.environ.get("LOCAL_GOAL_EVAL_REPORT", "logs/local_goal_p2p_eval.json")
    )

    args.task = "rotunbot_local_goal"
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.local_curriculum_stage = stage
    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    env.data_print = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None
    )
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.device)
    cases = _fixed_cases(episodes, seed, distances=distances, bearings=bearings)
    records = []

    print(
        f"Local Goal evaluation checkpoint={checkpoint} episodes={episodes} "
        f"stage={stage} device={env.device} reach_radius={REACH_RADIUS:.2f}",
        flush=True,
    )
    with torch.no_grad():
        for index, case in enumerate(cases, start=1):
            obs = _configure_episode(env, **case)
            min_distance = float(env.goal_dist[0].item())
            steps = 0
            action_count_start = env.action_count.item()
            clip_count_start = env.clip_count.item()
            while True:
                min_distance = min(min_distance, float(env.goal_dist[0].item()))
                actions = policy(obs)
                obs, _, _, dones, _ = env.step(actions)
                steps += 1
                done = bool(dones[0].item())
                if done:
                    terminal_distance = float(env.terminal_goal_dist[0].item())
                    min_distance = min(min_distance, terminal_distance)
                    success = terminal_distance <= REACH_RADIUS
                    timeout = bool(env.terminal_timeout[0].item()) and not success
                    unstable = bool(
                        env.terminal_unstable[0].item()
                        or env.terminal_out_of_bounds[0].item()
                    )
                    divergent = (not success) and (unstable or min_distance >= 1.0)
                    near_miss = (not success) and (not divergent)
                    action_count = env.action_count.item() - action_count_start
                    clip_count = env.clip_count.item() - clip_count_start
                    records.append(
                        {
                            "episode": index,
                            "distance": case["distance"],
                            "bearing_deg": case["bearing_deg"],
                            "yaw_deg": case["yaw_deg"],
                            "world_position": case["position"],
                            "success": success,
                            "timeout": timeout,
                            "divergent": divergent,
                            "near_miss": near_miss,
                            "min_distance": min_distance,
                            "final_distance": terminal_distance,
                            "steps": steps,
                            "clip_ratio": clip_count / max(action_count, 1),
                        }
                    )
                    break
                min_distance = min(min_distance, float(env.goal_dist[0].item()))

            if index <= 5 or index % 50 == 0:
                print(
                    f"episode={index} success={success} yaw={case['yaw_deg']:+.0f} "
                    f"bearing={case['bearing_deg']:+.0f} dmin={min_distance:.3f}",
                    flush=True,
                )

    summary = aggregate_local_goal_records(records)
    report = {
        "checkpoint": str(checkpoint),
        "episodes": episodes,
        "stage": stage,
        "distances": distances,
        "bearings_deg": bearings,
        "yaws_deg": YAWS_DEG,
        "world_positions": WORLD_POSITIONS,
        "seed": seed,
        "reach_radius": REACH_RADIUS,
        "summary": summary,
        "records": records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(
        f"LOCAL_GOAL_EVAL episodes={summary['episodes']} "
        f"success_rate={summary['success_rate']:.2%} "
        f"divergence_rate={summary['divergence_rate']:.2%} "
        f"yaw_success_gap={summary['yaw_success_gap']:.2%} "
        f"clip_ratio={summary['mean_clip_ratio']:.2%}",
        flush=True,
    )
    if (
        summary["success_rate"] < 0.95
        or summary["divergence_rate"] > 0.02
        or summary["yaw_success_gap"] > 0.05
    ):
        raise RuntimeError("Local Goal Gate failed; see the JSON report for per-pose metrics")
    print("LOCAL_GOAL_GATE PASS", flush=True)


if __name__ == "__main__":
    evaluate(get_args())
