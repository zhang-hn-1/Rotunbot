# Stage 1 V49 Waypoint Switching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an empty-map, multi-waypoint closed-loop evaluator that sends 5 Hz geometric `(v,w)` commands through frozen V49 without changing V49, URDFs, depth, maze, or training.

**Architecture:** `V49WaypointController` computes distance/bearing and raw commands from world-frame pose and a waypoint, then delegates feasibility to the existing `project_velocity_commands()`. `WaypointSequenceController` owns ordered waypoint state and only switches at 5 Hz ticks; the evaluator holds each projected command for exactly ten 50 Hz policy steps and records all gate evidence.

**Tech Stack:** Python 3.8, PyTorch 1.10.0+cu113, Isaac Gym, existing `legged_gym` task registry, unittest, CSV/JSON.

**Spec:** User-provided Stage 1 Empty-map V49 Multi-waypoint Closed Loop task.

## Global Constraints

- Use a new branch `codex/stage1-v49-waypoint-switching` from commit `27970c7`.
- Use `Rotunbot.urdf` for formal `maze` profile and `Rotunbot_test2.urdf` for `v49_reference`.
- Preserve 200 Hz physics, 50 Hz frozen V49 policy, 5 Hz high-level controller, and ten-policy-step command hold.
- Do not modify V49 checkpoint, release config/parameters, URDFs, depth/camera/maze files, or train any policy.
- Intermediate waypoint switches never reset the environment and happen at most once per 5 Hz tick.
- Do not commit CSV/JSON trajectory logs, checkpoints, TensorBoard logs, or local caches.

### Task 1: Pure controller and state-machine contracts

**Files:**
- Create: `legged_gym/navigation/v49_waypoint_controller.py`
- Modify: `legged_gym/navigation/__init__.py`
- Create: `legged_gym/tests/test_v49_waypoint_controller.py`

- [ ] Write failing tests for front/left/right geometry, ±pi wrapping, direct V49 projection, one-switch-per-tick behavior, no reset/done on intermediate switches, and ten-step command hold.
- [ ] Run `PATH=/home/jason/legged_gym/.venv/bin:$PATH PYTHONPATH=. /home/jason/legged_gym/.venv/bin/python legged_gym/tests/test_v49_waypoint_controller.py`; initial failure must identify missing controller symbols.
- [ ] Implement `V49WaypointConfig`, `V49WaypointController.command(robot_xy, robot_yaw, waypoint_xy)`, `WaypointSequenceState`, and `WaypointSequenceController.tick(...)` with fixed reach radius 0.25 m and at most one index increment per tick.
- [ ] Run the focused test and existing V49 55-test regression suite; no V49/depth regressions are acceptable.

### Task 2: V49 waypoint-sequence evaluator

**Files:**
- Create: `legged_gym/scripts/evaluate_v49_waypoint_sequence.py`
- Create: `legged_gym/tests/test_v49_waypoint_evaluator_contract.py`

- [ ] Write tests for profile-to-URDF mapping, A/B waypoint definitions, seed expansion, output schema, and no implicit reset after a switch.
- [ ] Implement evaluator using `task_registry.make_env()` and `make_alg_runner()`; do not implement a second policy inference path.
- [ ] Add `--asset-profile {maze,v49_reference}`, `--trajectory {A,B,both}`, `--episodes`, `--seed`, and `--output-dir`; load external frozen checkpoint through an explicit run-directory symlink when needed, never copy it into the repository.
- [ ] Update commands only at 5 Hz, hold each command for ten policy steps, send `[0,0]` after final reach, and observe up to 2 s settling.
- [ ] Log required tick fields and reset/timeout/NaN/Inf counters to local CSV/JSON.

### Task 3: Single-episode smoke and formal gate

**Files:**
- Create locally only: `logs/stage1_v49_waypoint_switching/`
- Create: `legged_gym/tests/test_v49_waypoint_artifacts.py`

- [ ] Run one A and one B smoke episode for `v49_reference`, then one A and one B smoke episode for `maze`; inspect no-reset, no-skip, and command-hold evidence.
- [ ] Run the 30-episode formal gate for both profiles, covering yaw `-15,-10,-5,0,5,10,15` degrees and position offsets in `[-.05,.05]` with a fixed seed.
- [ ] Enforce sequence success >= 90%, waypoint reach >= 95%, final error <= .25 m, intermediate resets = 0, skips = 0, NaN/Inf = 0, and terminal `|v| <= .10 m/s` as a separately reported settling metric.
- [ ] Classify failures as geometry, scheduling, V49 lag, URDF mismatch, or state-machine bug using newly logged evidence; never alter V49 to pass.

### Task 4: Commit and publish code only

- [ ] Verify `git diff` contains no `rotunbot_maze_local_depth*` or camera changes and no checkpoint/log artifacts.
- [ ] Run py_compile, focused tests, V49 55-test regression, and artifact schema tests with fresh output.
- [ ] Commit `feat: add V49 empty-map waypoint switching stage` and push `codex/stage1-v49-waypoint-switching` without merging.
