# Oracle Root-Cause Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible Oracle collision diagnostics and three non-training control gates while preserving the accepted Frozen-4150 behavior exactly.

**Architecture:** Keep the Oracle evaluator as the sole source of control actions and add pure geometry/classification helpers beside the existing navigation utilities. Extend episode JSON logging with step and collision diagnostics, then use a separate diagnostic runner and fixed control-gate scenarios so the Raw v1 evaluator remains behaviorally unchanged.

**Tech Stack:** Python 3.8, NumPy, PyTorch/Isaac Gym, existing `EpisodeLogger`, BFS planner, Frozen uniform-4150 policy, unittest.

**Spec:** `docs/superpowers/specs/2026-08-26-oracle-root-cause-diagnostics-design.md`

## Global Constraints

- Do not modify Frozen 4150, P2P observation/action definitions, Planner behavior, waypoint radius, turn-aware speed threshold, Reachability envelope, maze, or Raw100 manifest.
- Use checkpoint `logs/rotunbot_target_repro/Aug16_02-57-06_uniform_t1_long500_from3809/model_4150.pt`.
- `POST_SWITCH_COLLISION` primary window is `steps_since_goal_switch <= 10`; also report `<=5` and `<=20` sensitivity counts.
- Every collision has one primary class and overlapping boolean labels.
- Clearance is exterior robot clearance: wall surface distance minus `robot_collision_radius`.
- No training or Teacher Dataset collection.

### Task 1: Pure diagnostic geometry and collision classification

**Files:**
- Create: `legged_gym/navigation/oracle_diagnostics.py`
- Test: `legged_gym/tests/test_oracle_diagnostics.py`

**Interfaces:**
- `point_to_segment_distance(point_xy, segment_start_xy, segment_end_xy) -> float`
- `nearest_wall_clearance(robot_xy, wall_centers_xy, wall_size_xy, robot_collision_radius) -> tuple[float, float]`
- `classify_collision(*, phase, steps_since_goal_switch, delta_bearing_deg, waypoint_reached, current_cell, waypoint_cell, next_bfs_cell) -> dict`
- `summarize_collision_diagnostics(collision_records, episode_count) -> dict`

- [x] Write failing tests for point-to-segment distance, exterior clearance, overlapping labels, primary priority, and 5/10/20-step sensitivity.
- [x] Run `PYTHONPATH=. python3 -m unittest legged_gym.tests.test_oracle_diagnostics`; confirm expected missing-symbol failures.
- [x] Implement only the pure geometry and classification functions.
- [x] Re-run the focused tests and then the existing navigation test set.
- [x] Commit `feat: add oracle collision diagnostic primitives`.

### Task 2: Extend Oracle trajectory and collision logging

**Files:**
- Modify: `legged_gym/scripts/evaluate_oracle_maze.py`
- Modify: `legged_gym/navigation/oracle_metrics.py`
- Test: `legged_gym/tests/test_oracle_metrics.py`

**Interfaces:**
- Each trajectory row contains the requested robot, cell, goal, switch, turn-aware, reachability, clearance, and cross-track fields.
- Each collision episode summary contains collision step/XY/phase, post-switch steps, local goal geometry, and current/next BFS segment.
- Summary contains `collision_class_counts`, `collision_class_rates`, `collision_post_switch_window_counts`, and overlap-label counts.

- [x] Add failing metric tests for primary class/rate and overlap-label aggregation.
- [x] Run focused tests and verify they fail for absent diagnostic keys.
- [x] Add only observational state tracking to the evaluator; preserve action, waypoint, and switch operations byte-for-byte in behavior.
- [x] Record raw `steps_since_goal_switch` and derive `turn_aware_triggered`, `reachability_filtered`, clip ratio, wall clearance, and cross-track error from the active state.
- [x] Re-run focused and full CPU tests.
- [x] Commit `feat: log oracle collision root causes`.

### Task 3: Diagnostic Raw100 runner

**Files:**
- Create: `legged_gym/scripts/evaluate_oracle_diagnostics.py`
- Test: `legged_gym/tests/test_oracle_diagnostics_runner.py`

- [x] Write failing tests for manifest/checkpoint/config identity and diagnostic output naming.
- [x] Implement a runner that delegates to the existing Raw v1 control loop with diagnostics enabled, never changing evaluator inputs.
- [x] Run CPU checks and compileall.
- [x] Run Raw100 with the exact accepted manifest and Frozen checkpoint.
- [x] Verify all 100 episode files, summary counts, primary class counts, overlap labels, and threshold sensitivity counts.
- [x] Commit `feat: add oracle raw diagnostic runner`.

### Task 4: C1/C2/C3 non-training control gates

**Files:**
- Create: `legged_gym/scripts/evaluate_control_diagnostics.py`
- Test: `legged_gym/tests/test_control_diagnostics.py`

- [x] Write failing tests for fixed scenario definitions and C2 speed sweep values `(0.0, 0.2, 0.4, 0.6)`.
- [x] Implement fixed C1 straight corridor, C2 single 90-degree corner, and C3 wall-detour waypoint sequence using the existing maze and Frozen policy.
- [x] Record success/collision, cross-track error, minimum exterior clearance, and full trajectory for each case.
- [x] Run C1, C2, and C3 in order with no learning code path.
- [x] Commit `feat: add non-training collision control gates`.

### Task 5: Verification and root-cause report

- [x] Run all relevant unit tests, compileall, and `git diff --check`.
- [x] Confirm no training process or dataset collector was run.
- [x] Compare Raw and Reachability collision classes, post-switch windows, C1 stability, C2 corner failures, and C3 detour behavior.
- [x] State whether data supports the curved Frozen-4150 trajectory hypothesis; distinguish evidence from correlation.
- [x] Update navigation documentation with exact diagnostic definitions and output locations.
- [ ] Commit and push the completed diagnostic implementation to `hierarchical-visual-nav`.
