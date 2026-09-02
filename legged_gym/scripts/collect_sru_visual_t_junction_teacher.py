"""Collect real-IMAGE_DEPTH T-junction teacher trajectories through Frozen V62."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path


_SCENARIOS = ("T_LEFT", "T_RIGHT")
_EXPECTED_BRANCH = {"T_LEFT": "LEFT", "T_RIGHT": "RIGHT"}


def _scenario(value):
    scenario = str(value).upper()
    if scenario not in _SCENARIOS:
        raise ValueError("scenario must be T_LEFT or T_RIGHT")
    return scenario


def expected_branch_yaw_sign(scenario):
    """Return the positive-left / negative-right yaw convention."""
    return 1 if _scenario(scenario) == "T_LEFT" else -1


def has_wrong_turn_command(yaw_rate, scenario, deadband_rps=1.0e-4):
    """Report a non-zero turning command that opposes the selected branch."""
    yaw_rate = float(yaw_rate)
    if not math.isfinite(yaw_rate):
        raise ValueError("yaw_rate must be finite")
    if abs(yaw_rate) <= float(deadband_rps):
        return False
    return expected_branch_yaw_sign(scenario) * yaw_rate < 0.0


def record_t_branch_occupancy(occupancies, local_xy):
    """Append one actual local branch occupancy without discarding prior steps."""
    from legged_gym.navigation.v1_t_junction import classify_t_branch

    return tuple(occupancies) + (classify_t_branch(local_xy),)


def classify_t_episode_progress(
    scenario,
    local_xy=None,
    waypoint_index=0,
    exit_reached=False,
    observed_branches=(),
    terminal_local_xy=None,
):
    """Classify selected branch, wrong branch, turn completion, and final exit."""
    from legged_gym.navigation.v1_t_junction import classify_t_branch

    scenario = _scenario(scenario)
    if terminal_local_xy is not None:
        local_xy = terminal_local_xy
    if local_xy is None:
        raise ValueError("local_xy or terminal_local_xy is required")
    terminal_branch = classify_t_branch(local_xy)
    branches = [str(branch) for branch in observed_branches if branch in ("LEFT", "RIGHT")]
    if terminal_branch in ("LEFT", "RIGHT"):
        branches.append(terminal_branch)
    branch_prediction = terminal_branch if terminal_branch in ("LEFT", "RIGHT") else (
        branches[-1] if branches else "UNDECIDED"
    )
    expected_branch = _EXPECTED_BRANCH[scenario]
    wrong_turn = any(branch != expected_branch for branch in branches)
    return {
        "expected_branch": expected_branch,
        "branch_prediction": branch_prediction,
        "wrong_turn": bool(wrong_turn),
        "turn_completed": bool(int(waypoint_index) >= 2 and expected_branch in branches),
        "exit_reached": bool(exit_reached),
    }


def require_isaacgym_depth_backend(actual):
    """Fail closed so fallback depth can never become teacher data."""
    if actual != "isaacgym":
        raise RuntimeError("T-junction teacher requires real Isaac Gym IMAGE_DEPTH; got %s" % actual)
    return actual


def make_t_episode_record(
    *,
    scenario,
    episode_id,
    seed,
    goal,
    initial_pose,
    initial_yaw,
    horizon,
    episode_steps,
    macro_steps,
    success,
    collision,
    timeout,
    progress,
    failure_trace=None,
    depth_backend_actual="isaacgym",
):
    """Create the release-gate record without changing the V1 sample schema."""
    scenario = _scenario(scenario)
    require_isaacgym_depth_backend(depth_backend_actual)
    missing = {"branch_prediction", "wrong_turn", "turn_completed", "exit_reached"}.difference(progress)
    if missing:
        raise ValueError("progress is missing: %s" % ", ".join(sorted(missing)))
    trace = list(failure_trace or ())
    if not bool(success) and not trace:
        trace.append(
            {
                "terminal": {
                    "collision": bool(collision),
                    "timeout": bool(timeout),
                    "wrong_turn": bool(progress["wrong_turn"]),
                    "exit_reached": bool(progress["exit_reached"]),
                }
            }
        )
    return {
        "episode_id": int(episode_id),
        "scenario": scenario,
        "policy_role": "teacher",
        "seed": int(seed),
        "goal": tuple(float(value) for value in goal),
        "initial_pose": tuple(float(value) for value in initial_pose),
        "initial_yaw": float(initial_yaw),
        "horizon": int(horizon),
        "episode_steps": int(episode_steps),
        "macro_steps": int(macro_steps),
        "success": bool(success),
        "collision": bool(collision),
        "timeout": bool(timeout),
        "expected_branch": _EXPECTED_BRANCH[scenario],
        "branch_prediction": str(progress["branch_prediction"]),
        "wrong_turn": bool(progress["wrong_turn"]),
        "turn_completed": bool(progress["turn_completed"]),
        "turn_completion": bool(progress["turn_completed"]),
        "exit_reached": bool(progress["exit_reached"]),
        "exit": bool(progress["exit_reached"]),
        "depth_backend_actual": depth_backend_actual,
        "failure_trace": trace,
    }


def _parse_framework_args(remaining):
    from legged_gym.utils import get_args

    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + list(remaining)
    try:
        return get_args()
    finally:
        os.sys.argv = original


def _close_environment(env):
    if env is None:
        return
    viewer = getattr(env, "viewer", None)
    if viewer is not None:
        env.gym.destroy_viewer(viewer)
    if getattr(env, "sim", None) is not None:
        env.gym.destroy_sim(env.sim)


def _commit_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _assign_final_goal(env, geometry, torch):
    goal = torch.as_tensor(
        geometry.scenario.goal_xy, dtype=env.root_states.dtype, device=env.device
    )
    env.global_goal_xy_world[0] = env.env_origins[0, :2] + goal
    distance = torch.linalg.vector_norm(env.global_goal_xy_world[0] - env.root_states[0, :2])
    env.goal_dist[0] = distance
    env.terminal_goal_distance[0] = distance
    env.previous_goal_distance[0] = distance
    env.goal_reached_buf[0] = False
    env.success_buf[0] = False


def _local_pose(env):
    position = env.root_states[0, :2] - env.env_origins[0, :2]
    yaw = env._yaw_from_quaternion(env.root_states[0:1, 3:7])[0]
    return position, yaw


def terminal_local_xy(env):
    """Return the terminal local position captured by reset_idx before auto-reset."""
    position = env.terminal_position[0, :2] - env.env_origins[0, :2]
    values = position.detach().cpu().tolist() if hasattr(position, "detach") else position.tolist()
    if len(values) != 2 or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("terminal_position must contain two finite XY values")
    return tuple(float(value) for value in values)


def install_observation_goal(env, local_waypoint_world):
    """Install the active waypoint into the actor observation before capture."""
    env.set_observation_goal_world(local_waypoint_world.reshape(1, 2))
    env.compute_observations()
    return env._goal_xy_robot()


def _new_state(scene, episode_id, env):
    position, yaw = _local_pose(env)
    return {
        "scene": scene,
        "episode_id": int(episode_id),
        "episode_steps": 0,
        "macro_steps": 0,
        "initial_pose": (
            float(position[0].item()),
            float(position[1].item()),
            float(env.root_states[0, 2].item()),
        ),
        "initial_yaw": float(yaw.item()),
        "observed_branches": [],
        "branch_occupancy": [],
        "wrong_turn": False,
        "failure_events": [],
    }


def _record_branch_occupancy(state, scenario, local_xy):
    """Persist physical branch occupancy and latch a wrong-side traversal."""
    occupancy = record_t_branch_occupancy(state["branch_occupancy"], local_xy)
    branch = occupancy[-1]
    state["branch_occupancy"] = list(occupancy)
    if branch in ("LEFT", "RIGHT"):
        state["observed_branches"].append(branch)
        if branch != _EXPECTED_BRANCH[_scenario(scenario)]:
            state["wrong_turn"] = True
    return branch


def _dataset_row(env, torch, episode_id, step_id, teacher, actual, goal_xy):
    return {
        "episode_id": int(episode_id),
        "step_id": int(step_id),
        "depth": env.depth_observation[0].detach().clone(),
        "goal_xy_robot": goal_xy[0].detach().clone(),
        "proprioception": env._proprioception()[0].detach().clone(),
        "previous_command": env.previous_velocity_command[0].detach().clone(),
        "previous_actual_velocity": env.previous_actual_velocity[0].detach().clone(),
        "teacher_command": teacher["applied_command"][0].detach().clone(),
        "actual_velocity": actual[0].detach().clone(),
        "governor_command": env.applied_feasible_command[0].detach().clone(),
        "projection_command": teacher["applied_command"][0].detach().clone(),
        "done": False,
        "success": False,
        "collision": False,
        "goal_distance": torch.linalg.vector_norm(goal_xy[0]).detach().cpu(),
    }


def evaluate_scene(
    env,
    scene,
    episodes,
    seed,
    max_steps,
    teacher_cfg,
    geometry,
    torch,
    waypoint_manager_cls,
    dataset_writer=None,
    episode_offset=0,
):
    """Run one deterministic T side, with macro commands held through env.step."""
    from legged_gym.navigation.v1_velocity_teacher import teacher_velocity_diagnostics

    with torch.inference_mode():
        env.reset()
    require_isaacgym_depth_backend(env.depth_backend_actual)
    _assign_final_goal(env, geometry, torch)
    manager = waypoint_manager_cls(geometry.waypoints, reach_radius=geometry.reach_radius_m)
    actions = torch.zeros(1, 2, device=env.device)
    records = []
    next_teacher_step = int(env.common_step_counter)
    pending = None
    episode_id = 0
    state = _new_state(scene, episode_id, env)

    with torch.inference_mode():
        while len(records) < int(episodes):
            if env.common_step_counter >= next_teacher_step:
                if pending is not None and dataset_writer is not None:
                    dataset_writer.append(pending)
                position, yaw = _local_pose(env)
                pose = (float(position[0].item()), float(position[1].item()), float(yaw.item()))
                manager.update(pose)
                waypoint_robot = manager.get_current_waypoint_robot(pose)
                waypoint_world = env.env_origins[0, :2] + torch.as_tensor(
                    manager.get_current_waypoint(), dtype=env.root_states.dtype, device=env.device
                ).reshape(1, 2)
                goal_xy = install_observation_goal(env, waypoint_world).detach().clone()
                actual = torch.stack((env.tracking_lin_vel[:, 0], env.tracking_ang_vel[:, 2]), dim=1)
                teacher = teacher_velocity_diagnostics(goal_xy, actual, env.obstacle_clearance, teacher_cfg)
                actions[:, 0] = teacher["applied_command"][:, 0] / teacher_cfg.max_forward_speed
                actions[:, 1] = teacher["applied_command"][:, 1] / teacher_cfg.max_yaw_rate
                actions.clamp_(-1.0, 1.0)
                state["macro_steps"] += 1
                _record_branch_occupancy(state, scene, (pose[0], pose[1]))
                state["failure_events"].append(
                    {
                        "macro_step": state["macro_steps"],
                        "position_xy": [float(position[0].item()), float(position[1].item())],
                        "waypoint_index": int(manager.current_index),
                        "teacher_command": [
                            float(teacher["applied_command"][0, 0].item()),
                            float(teacher["applied_command"][0, 1].item()),
                        ],
                        "clearance_m": float(env.obstacle_clearance[0].item()),
                    }
                )
                if dataset_writer is not None:
                    pending = _dataset_row(
                        env, torch, episode_offset + episode_id, state["macro_steps"] - 1,
                        teacher, actual, goal_xy,
                    )
                next_teacher_step = int(env.common_step_counter) + int(env.upper_level_command_interval_steps)

            _, _, _, dones, _ = env.step(actions)
            state["episode_steps"] += 1
            done = bool(dones.flatten()[0].item())
            if done:
                primitive_local_xy = terminal_local_xy(env)
            else:
                position, _ = _local_pose(env)
                primitive_local_xy = (float(position[0].item()), float(position[1].item()))
            _record_branch_occupancy(state, scene, primitive_local_xy)
            forced_timeout = state["episode_steps"] >= int(max_steps) and not done
            if not done and not forced_timeout:
                continue
            success = bool(env.terminal_success[0].item()) if done else False
            collision = bool(env.terminal_collision[0].item()) if done else False
            timeout = forced_timeout or (bool(env.terminal_timeout[0].item()) if done else False)
            if done:
                local_xy = terminal_local_xy(env)
            else:
                position, _ = _local_pose(env)
                local_xy = (float(position[0].item()), float(position[1].item()))
            progress = classify_t_episode_progress(
                scene,
                terminal_local_xy=local_xy,
                waypoint_index=manager.current_index,
                exit_reached=success,
                observed_branches=state["observed_branches"],
            )
            progress["wrong_turn"] = bool(state["wrong_turn"] or progress["wrong_turn"])
            if pending is not None and dataset_writer is not None:
                pending.update(
                    {
                        "done": True,
                        "success": success,
                        "collision": collision,
                        "goal_distance": torch.tensor(
                            float(env.terminal_goal_distance[0].item()) if done else float(env.goal_dist[0].item())
                        ),
                    }
                )
                dataset_writer.append(pending)
                pending = None
            record = make_t_episode_record(
                scenario=scene,
                episode_id=episode_offset + episode_id,
                seed=seed,
                goal=geometry.scenario.goal_xy,
                initial_pose=state["initial_pose"],
                initial_yaw=state["initial_yaw"],
                horizon=max_steps,
                episode_steps=state["episode_steps"],
                macro_steps=state["macro_steps"],
                success=success,
                collision=collision,
                timeout=timeout,
                progress=progress,
                failure_trace=[] if success else state["failure_events"],
            )
            records.append(record)
            if len(records) >= int(episodes):
                break
            env.reset_idx(torch.as_tensor([0], dtype=torch.long, device=env.device))
            episode_id += 1
            manager.reset()
            _assign_final_goal(env, geometry, torch)
            state = _new_state(scene, episode_id, env)
            next_teacher_step = int(env.common_step_counter)
    return records


def _write_scene_records(output, scene, records):
    scene_root = output / scene
    scene_root.mkdir(parents=True, exist_ok=True)
    with (scene_root / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    failures = [record for record in records if record["failure_trace"]]
    (scene_root / "failure_traces.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=2250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-output", default=None)
    stage_args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if int(stage_args.episodes) <= 0 or int(stage_args.max_steps) <= 0:
        raise ValueError("episodes and max-steps must be positive")

    # Isaac Gym must load before torch in the frozen V62 environment.
    import isaacgym  # noqa: F401
    import torch
    import legged_gym.envs  # noqa: F401
    from legged_gym.navigation.v1_t_junction import build_t_junction_geometry
    from legged_gym.navigation.v1_t_junction_metrics import aggregate_t_gate
    from legged_gym.navigation.v1_teacher_dataset import TeacherSequenceWriter
    from legged_gym.navigation.v1_velocity_teacher import V1VelocityTeacherConfig
    from legged_gym.navigation.v1_waypoint_manager import V1WaypointManager
    from legged_gym.utils import task_registry

    args = _parse_framework_args(remaining)
    args.task = "rotunbot_sru_visual_corridor_v1"
    args.seed = int(stage_args.seed)
    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg.seed = int(stage_args.seed)
    env_cfg.env.num_envs = 1
    env_cfg.enable_camera_sensors_in_headless = True
    env_cfg.camera.depth_backend = "isaacgym"
    env_cfg.camera.add_noise = False
    env_cfg.init_state.random_start_lateral = 0.0
    env_cfg.init_state.random_start_yaw = 0.0
    env_cfg.init_state.randomize_initial_velocity = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.v1_goal_curriculum_enabled = False
    env_cfg.commands.v1_performance_curriculum_enabled = False
    teacher_cfg = V1VelocityTeacherConfig(
        max_forward_speed=float(env_cfg.commands.max_forward_speed),
        max_yaw_rate=float(env_cfg.commands.max_yaw_rate),
        minimum_turn_radius=float(env_cfg.commands.minimum_turn_radius),
        feasible_envelope_fraction=float(env_cfg.commands.feasible_envelope_fraction),
        goal_radius=float(env_cfg.commands.goal_radius),
    )
    output = Path(stage_args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_writer = TeacherSequenceWriter(sequence_length=16) if stage_args.dataset_output else None
    records = []
    geometry_by_scene = {}
    for scene in _SCENARIOS:
        geometry = build_t_junction_geometry(scene)
        geometry_by_scene[scene] = {
            "goal_xy": [float(value) for value in geometry.scenario.goal_xy],
            "waypoints": geometry.waypoints.tolist(),
            "wall_segments": [[list(start), list(end)] for start, end in geometry.wall_segments],
            "width_m": float(geometry.scenario.width_m),
        }
        env_cfg.corridor_width_m = geometry.scenario.width_m
        env_cfg.corridor_wall_width_m = geometry.scenario.width_m
        env_cfg.corridor_wall_segments = ()
        env_cfg.corridor_explicit_wall_segments = geometry.wall_segments
        env_cfg.direct_obstacle_aabbs = geometry.obstacle_aabbs
        env = None
        try:
            env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
            require_isaacgym_depth_backend(env.depth_backend_actual)
            scene_records = evaluate_scene(
                env, scene, stage_args.episodes, stage_args.seed, stage_args.max_steps,
                teacher_cfg, geometry, torch, V1WaypointManager,
                dataset_writer=dataset_writer, episode_offset=len(records),
            )
        finally:
            _close_environment(env)
        _write_scene_records(output, scene, scene_records)
        records.extend(scene_records)

    gate = aggregate_t_gate(records, pairs=[], ablations={})
    payload = {
        "stage": "T_JUNCTION_TEACHER",
        "status": "PASS" if gate["pass"] else "FAIL",
        "commit": _commit_sha(),
        "seed": int(stage_args.seed),
        "episodes_per_scene": int(stage_args.episodes),
        "max_steps": int(stage_args.max_steps),
        "depth_backend_requested": "isaacgym",
        "depth_backend_actual": "isaacgym",
        "geometry": geometry_by_scene,
        "gate": gate,
    }
    if dataset_writer is not None:
        dataset_path = Path(stage_args.dataset_output).resolve()
        dataset_writer.save(
            dataset_path,
            metadata={
                "schema_name": "V1-compatible T-junction teacher dataset",
                "depth_backend_requested": "isaacgym",
                "depth_backend_actual": "isaacgym",
                "depth_representation": "normalized IMAGE_DEPTH, far-is-open",
                "scenarios": list(_SCENARIOS),
                "episode_scenarios": [record["scenario"] for record in records],
                "episodes_per_scene": int(stage_args.episodes),
                "sequence_length": 16,
                "seed": int(stage_args.seed),
                "geometry": geometry_by_scene,
                "episode_provenance": {
                    str(record["episode_id"]): {
                        "scenario": record["scenario"],
                        "goal": list(record["goal"]),
                        "initial_pose": list(record["initial_pose"]),
                        "initial_yaw": record["initial_yaw"],
                        "horizon": record["horizon"],
                    }
                    for record in records
                },
                "command_ranges": {
                    "v_cmd": [-teacher_cfg.max_forward_speed, teacher_cfg.max_forward_speed],
                    "w_cmd": [-teacher_cfg.max_yaw_rate, teacher_cfg.max_yaw_rate],
                    "normalized_action": [-1.0, 1.0],
                },
            },
        )
        payload["dataset"] = str(dataset_path)
    (output / "t_junction_teacher_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
