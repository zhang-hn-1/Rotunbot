# Hierarchical Visual Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement non-learning hierarchical navigation infrastructure around the frozen uniform-4150 point-to-point controller.

**Architecture:** Keep the existing P2P task and controller untouched. Add dependency-light navigation utilities for coordinate conversion, BFS, and data-derived reachability, then add evaluation scripts that update only the task's world-frame temporary command without resetting the robot, policy history, or episode. Add the dataset collector only after the closed-loop Oracle path is available.

**Tech Stack:** Python 3, NumPy, PyTorch/Isaac Gym for runtime evaluators, standard-library JSON/CSV/dataclasses, unittest/pytest-compatible tests.

**Spec:** `docs/superpowers/specs/2026-08-26-hierarchical-visual-navigation-design.md`

## Global Constraints

- Base branch is `hierarchical-visual-nav` from commit `0f6bde79e7f57e458d26907016368e6e4b7ee6b`.
- Frozen low-level policy is uniform 4150 at `Aug16_02-57-06_uniform_t1_long500_from3809/model_4150.pt`.
- Do not change the 19-D observation, 20-frame history, DWL-CNN, actor, 2-D action interface, DIRECT_VP_TORQUE, gains, reward, success rule, or original P2P evaluation protocol.
- Do not import depth-camera, SRU, ActorCriticDepth, CNN-training, DAgger, or PPO-update code.
- Local goals are converted to world-frame temporary goals before the frozen policy observes them.
- Gate 1 must precede Gate 2; Gate 2 must precede Oracle Maze; Dataset Collector is last.
- Missing checkpoint files are fatal and must never fall back to random initialization.
- Current GPU runtime, when used, is `/home/jason/legged_gym/.venv/bin/python`.

### Task 1: Freeze the baseline and add package boundaries

**Files:**
- Create: `legged_gym/navigation/__init__.py`
- Create: `legged_gym/navigation/baseline.py`
- Create: `docs/hierarchical_navigation_baseline.md`
- Test: `legged_gym/tests/test_navigation_baseline.py`

**Interfaces:**
- `baseline.py` exposes immutable constants for task name, checkpoint path, observation dimensions, action dimensions, frame stack, and accepted controller parameters.
- `require_checkpoint(path) -> Path` validates an explicit checkpoint path and raises `FileNotFoundError` for a missing file.

- [ ] Record the exact frozen policy contract and checkpoint path.
- [ ] Add tests for constants and missing/valid checkpoint validation.
- [ ] Run the baseline tests.
- [ ] Commit the baseline boundary.

### Task 2: Implement Local Goal Adapter

**Files:**
- Create: `legged_gym/navigation/local_goal_adapter.py`
- Create: `legged_gym/tests/test_local_goal_adapter.py`

**Interfaces:**
- `local_to_world(robot_xy, robot_yaw, local_goal_xy) -> np.ndarray`.
- `world_to_local(robot_xy, robot_yaw, world_goal_xy) -> np.ndarray`.
- Both functions validate finite two-element vectors and return float64 two-element vectors.

- [ ] Write tests for yaw `0, 45, 90, 135, 180, -45, -90, -135` degrees and all six requested goal directions.
- [ ] Write the round-trip test with error `< 1e-10`.
- [ ] Implement the rotation and inverse rotation.
- [ ] Run adapter tests.
- [ ] Commit the adapter.

### Task 3: Implement Oracle BFS and waypoint selection

**Files:**
- Create: `legged_gym/navigation/bfs_planner.py`
- Create: `legged_gym/tests/test_bfs_planner.py`
- Modify: `legged_gym/navigation/__init__.py`

**Interfaces:**
- `plan_cells(occupancy, start_cell, goal_cell) -> tuple[tuple[int, int], ...]`.
- `cell_center(cell, maze_shape, cell_size) -> np.ndarray`.
- `select_next_waypoint(path, current_index=0) -> tuple[int, int]`.
- `world_to_cell(world_xy, maze_shape, cell_size) -> tuple[int, int]`.

- [ ] Test shortest four-neighbor paths, wall rejection, unreachable goals, and invalid cells.
- [ ] Reuse `legged_gym.maps.rotunbot_maze` for map generation and conversion semantics.
- [ ] Ensure the planner returns waypoints only and never actions.
- [ ] Run map and BFS tests.
- [ ] Commit the planner.

### Task 4: Build explicit frozen-policy loading and Gate 1

**Files:**
- Create: `legged_gym/navigation/frozen_p2p.py`
- Create: `legged_gym/scripts/evaluate_single_local_goal.py`
- Create: `legged_gym/tests/test_gate_logging.py`

**Interfaces:**
- `FrozenP2PConfig` stores task name, checkpoint, seed, control type, and runtime device.
- `FrozenP2PConfig.load(...)` must validate the checkpoint before constructing the policy.
- `EpisodeLogger.write_json(path)` and `EpisodeLogger.write_csv(path)` write deterministic summaries and trajectories.

- [ ] Write a pure logger test before runtime code.
- [ ] Implement checkpoint validation and a loader that calls `runner.load(..., load_optimizer=False)` and prints the loaded checkpoint.
- [ ] Implement conservative Gate 1 goals at distances `0.5, 1.0, 1.5 m` and bearings `0, ±30, ±45°` with randomized position/yaw.
- [ ] Update only `env.commands[0, :2]` for a temporary world goal and call the existing observation path; never pass local coordinates to the actor.
- [ ] Log success, timeout, divergence, final/minimum distance, completion time, clipping, and trajectory.
- [ ] Run pure tests and GPU smoke only when Isaac Gym/checkpoint are present.
- [ ] Commit Gate 1.

### Task 5: Build Gate 2 without reset

**Files:**
- Create: `legged_gym/scripts/evaluate_goal_switch.py`
- Create: `legged_gym/navigation/goal_switch.py`
- Create: `legged_gym/tests/test_goal_switch.py`

**Interfaces:**
- `GoalSwitchController.update_world_goal(world_goal_xy) -> None` changes only the temporary command.
- `GoalSwitchController.step(action) -> StepRecord` preserves environment state and records action discontinuity.
- `run_waypoint_sequence(sequence, ...) -> dict` returns switch latency, stop duration, local/single waypoint success, and sequence success.

- [ ] Test that a goal update does not clear previous action, history, or episode counters in a fake environment.
- [ ] Implement straight, L, S, rectangle, forward-then-lateral, lateral-then-forward, and sharp-turn sequences.
- [ ] Detect and log full stops before switches without changing reward or controller behavior.
- [ ] Commit Gate 2.

### Task 6: Measure reachability and add data-derived filter

**Files:**
- Create: `legged_gym/navigation/reachability.py`
- Create: `legged_gym/scripts/measure_reachability.py`
- Create: `legged_gym/tests/test_reachability.py`

**Interfaces:**
- `ReachabilitySample` stores actions, displacement, velocity, rise time, coupling, clipping, and joint response.
- `ReachabilityEnvelope.from_samples(samples, angular_bins) -> ReachabilityEnvelope` derives radial limits from measured samples; it assumes no ellipse.
- `ReachabilityEnvelope.filter(local_goal_xy) -> np.ndarray` deterministically clips only outside measured limits while preserving bearing.
- `save_samples/load_samples` and `save_envelope/load_envelope` use JSON.

- [ ] Test raw sample round trips and deterministic inside/outside filtering.
- [ ] Implement the action sweep for `±0.25, ±0.5, ±0.75, ±1.0` without modifying PID/gains.
- [ ] Store raw data before deriving the envelope.
- [ ] Commit reachability tools.

### Task 7: Implement Oracle Maze closed-loop evaluator

**Files:**
- Create: `legged_gym/scripts/evaluate_oracle_maze.py`
- Create: `legged_gym/navigation/oracle_episode.py`
- Create: `legged_gym/tests/test_oracle_episode.py`

**Interfaces:**
- `OracleEpisodePlanner.next_local_waypoint(robot_xy, robot_yaw, global_goal_xy) -> LocalWaypoint`.
- `LocalWaypoint` contains cell, local goal, filtered local goal, and temporary world goal.
- `run_oracle_episode(...) -> dict` returns global metrics and categorized failure reason.

- [ ] Test replanning from the current cell and waypoint-only planner output.
- [ ] Use maze seed 0, center start, reachable random global goals, and 100 episodes in the GPU evaluator.
- [ ] Replan from the robot's current actual pose after every reached waypoint; never follow an ideal offline path blindly.
- [ ] Keep the global goal fixed and terminate only on global success.
- [ ] Save trajectory, waypoint sequence, goal, yaw, and failure category.
- [ ] Commit Oracle Maze.

### Task 8: Add closed-loop dataset collector last

**Files:**
- Create: `legged_gym/navigation/dataset.py`
- Create: `legged_gym/scripts/collect_oracle_depth_dataset.py`
- Create: `legged_gym/tests/test_dataset.py`

**Interfaces:**
- `DepthFrameProvider.get_frame() -> np.ndarray` is provider-neutral.
- `OracleSample` contains depth, pose, global/local/temporary goals, previous waypoint, collision state, timestamp, and episode id.
- `ClosedLoopDatasetWriter.append(sample)` and `.close()` write append-only episode records.

- [ ] Test serialization without importing Isaac Gym or a depth network.
- [ ] Require Oracle closed-loop mode and actual robot-state replanning before collection starts.
- [ ] Save labels only after the frozen controller executes and the next plan is computed from actual state.
- [ ] Commit the collector.

### Task 9: Verification and delivery

**Files:**
- Modify: `docs/hierarchical_navigation_baseline.md`
- Modify: `README.md` only if a concise navigation entry is needed

- [ ] Run all pure unit tests, existing maze map tests, compileall, and `git diff --check`.
- [ ] Run GPU Gate 1/Gate 2/Oracle/Reachability commands with `/home/jason/legged_gym/.venv/bin/python` when the checkpoint and Isaac Gym are available.
- [ ] Confirm no depth/SRU imports or PPO update calls are present in the new navigation modules.
- [ ] Commit and push `hierarchical-visual-nav` without force-pushing.
- [ ] Report exact validation results and any runtime-only gates blocked by missing local assets.
