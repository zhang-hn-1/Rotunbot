"""Gate0C: fixed distance-by-bearing diagnosis for the frozen local P2P.

Each case creates one robot-frame waypoint at reset, converts it once to a
world-frame target, and then keeps that tensor unchanged until the episode
ends.  The policy receives the same world-frame target through ``commands``.
"""

import json
import math
import os
from pathlib import Path

import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


CHECKPOINT = (
    "/home/jason/SphericalRobot_LeggedGym-master-new-map/logs/"
    "rotunbot_target_repro/Aug11_16-44-07_/model_2050.pt"
)
DISTANCES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
BEARINGS_DEG = (0.0, 45.0, 90.0, -45.0, -90.0, 180.0)
REACH_RADIUS = 0.35


def _yaw_from_quaternion(quat):
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _empty_case(distance, bearing_deg):
    return {
        "distance": float(distance),
        "bearing_deg": float(bearing_deg),
        "episodes": 0,
        "success": 0,
        "timeout": 0,
        "divergence": 0,
        "near_miss": 0,
        "near_miss_bins": {"A_lt_035": 0, "B_035_045": 0, "C_045_060": 0, "D_ge_060": 0},
        "final_distance": [],
        "min_distance": [],
        "completion_steps": [],
        "path_length": [],
    }


def _classify_min_distance(min_distance):
    if min_distance < 0.35:
        return "A_lt_035"
    if min_distance < 0.45:
        return "B_035_045"
    if min_distance < 0.60:
        return "C_045_060"
    return "D_ge_060"


def _matrix(cases, key):
    return [
        [
            cases[(distance, bearing)][key]
            for bearing in BEARINGS_DEG
        ]
        for distance in DISTANCES
    ]


def evaluate(args):
    checkpoint_path = Path(os.environ.get("GATE0C_P2P_CHECKPOINT", CHECKPOINT)).resolve()
    episodes_per_case = int(os.environ.get("GATE0C_EPISODES_PER_CASE", "5"))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Gate0C checkpoint does not exist: {checkpoint_path}")
    if episodes_per_case <= 0:
        raise ValueError("GATE0C_EPISODES_PER_CASE must be positive")

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

    cases = {
        (distance, bearing): _empty_case(distance, bearing)
        for distance in DISTANCES
        for bearing in BEARINGS_DEG
    }

    def configure_case(distance, bearing_deg):
        position = env.root_states[0, :2].clone()
        yaw = _yaw_from_quaternion(env.base_quat)[0]
        bearing = torch.tensor(math.radians(bearing_deg), device=env.device)
        local_delta = torch.stack(
            (
                torch.tensor(distance, device=env.device) * torch.cos(bearing),
                torch.tensor(distance, device=env.device) * torch.sin(bearing),
            )
        )
        # This is the only Robot-frame -> World-frame conversion for the case.
        world_delta = torch.stack(
            (
                torch.cos(yaw) * local_delta[0] - torch.sin(yaw) * local_delta[1],
                torch.sin(yaw) * local_delta[0] + torch.cos(yaw) * local_delta[1],
            )
        )
        locked_world_waypoint = position + world_delta
        env.commands[0, :2] = locked_world_waypoint
        env.goal_dist[0] = float(distance)
        env.last_goal_dist[0] = float(distance)
        env.compute_observations()
        return env.get_observations(), locked_world_waypoint

    env.reset()
    total_cases = len(cases)
    completed_cases = 0
    print(
        "Gate0C evaluation: "
        f"checkpoint={checkpoint_path}, cases={total_cases}, "
        f"episodes_per_case={episodes_per_case}, device={env.device}, "
        f"reach_radius={REACH_RADIUS:.2f}m",
        flush=True,
    )

    with torch.no_grad():
        for distance in DISTANCES:
            for bearing_deg in BEARINGS_DEG:
                case = cases[(distance, bearing_deg)]
                for _ in range(episodes_per_case):
                    obs, locked_waypoint = configure_case(distance, bearing_deg)
                    episode_steps = 0
                    path_length = 0.0
                    min_distance = float(distance)
                    previous_position = env.root_states[0, :2].clone()

                    while True:
                        current_distance = float(
                            torch.linalg.norm(
                                locked_waypoint - env.root_states[0, :2]
                            ).item()
                        )
                        min_distance = min(min_distance, current_distance)
                        actions = policy(obs)
                        obs, _, _, dones, _ = env.step(actions)
                        episode_steps += 1

                        done = bool(dones[0].item())
                        current_position = env.root_states[0, :2].clone()
                        if not done:
                            path_length += float(
                                torch.linalg.norm(current_position - previous_position).item()
                            )
                            previous_position = current_position
                            current_distance = float(
                                torch.linalg.norm(locked_waypoint - current_position).item()
                            )
                            min_distance = min(min_distance, current_distance)

                        if done:
                            terminal_distance = float(env.terminal_goal_dist[0].item())
                            min_distance = min(min_distance, terminal_distance)
                            # The simulator has already reset this environment;
                            # use the cached pre-reset terminal distance only.
                            reached = terminal_distance < REACH_RADIUS
                        else:
                            reached = current_distance < REACH_RADIUS

                        if reached or done:
                            case["episodes"] += 1
                            case["success"] += int(reached)
                            timed_out = bool(env.terminal_timeout[0].item())
                            case["timeout"] += int((not reached) and timed_out)
                            unstable = bool(
                                env.terminal_unstable[0].item()
                                or env.terminal_out_of_bounds[0].item()
                            ) if done else False
                            divergent = (not reached) and (
                                unstable or min_distance >= 1.0
                            )
                            case["divergence"] += int(divergent)
                            case["near_miss"] += int(not reached and not divergent)
                            case["near_miss_bins"][_classify_min_distance(min_distance)] += int(
                                not reached
                            )
                            final_distance = (
                                float(env.terminal_goal_dist[0].item())
                                if done
                                else current_distance
                            )
                            case["final_distance"].append(final_distance)
                            case["min_distance"].append(min_distance)
                            case["completion_steps"].append(episode_steps)
                            case["path_length"].append(path_length)

                            if not done:
                                env.reset_idx(
                                    torch.tensor(
                                        [0], device=env.device, dtype=torch.long
                                    )
                                )
                            break

                    completed_cases += int(case["episodes"] == episodes_per_case)
                print(
                    "case distance={:.1f} bearing={:>5.0f}: "
                    "reach={:.0%} timeout={} divergence={} near_miss={} "
                    "dmin={:.3f}".format(
                        distance,
                        bearing_deg,
                        case["success"] / case["episodes"],
                        case["timeout"],
                        case["divergence"],
                        case["near_miss"],
                        sum(case["min_distance"]) / len(case["min_distance"]),
                    ),
                    flush=True,
                )

    for case in cases.values():
        for key in ("final_distance", "min_distance", "completion_steps", "path_length"):
            values = case[key]
            case[key + "_mean"] = sum(values) / len(values) if values else None
        del case["final_distance"]
        del case["min_distance"]
        del case["completion_steps"]
        del case["path_length"]

    output_path = Path(
        os.environ.get(
            "GATE0C_REPORT",
            "logs/gate0c_model2050_first_round.json",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "episodes_per_case": episodes_per_case,
                "distances": DISTANCES,
                "bearings_deg": BEARINGS_DEG,
                "reach_radius": REACH_RADIUS,
                "cases": {f"{d}_{b}": value for (d, b), value in cases.items()},
                "reach_rate_matrix": _matrix(cases, "success"),
                "divergence_matrix": _matrix(cases, "divergence"),
                "mean_min_distance_matrix": _matrix(cases, "min_distance_mean"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved Gate0C report: {output_path}", flush=True)

    print("Reach Rate Matrix (rows=distance, cols=bearing):", flush=True)
    for row in _matrix(cases, "success"):
        print(" ".join(f"{value / episodes_per_case:.0%}" for value in row), flush=True)
    print("Divergence Count Matrix:", flush=True)
    for row in _matrix(cases, "divergence"):
        print(" ".join(str(value) for value in row), flush=True)


if __name__ == "__main__":
    evaluate(get_args())
