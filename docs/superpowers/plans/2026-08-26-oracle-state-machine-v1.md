# Oracle State Machine v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct Oracle waypoint scheduling at the goal cell, add an independently switchable turn-aware waypoint policy, and evaluate the four requested Raw/Reachability/Turn-aware combinations without training.

**Architecture:** Keep the frozen uniform-4150 controller and 19D×20 history unchanged. Extend the Oracle evaluator with an explicit NAVIGATE/FINAL_APPROACH state machine, keep reachability and turn-aware behavior as planner options, and aggregate all variants against one validated 100-episode manifest.

**Tech Stack:** Python, NumPy, unittest, Isaac Gym/PyTorch GPU evaluation.

**Spec:** User request in the active conversation dated 2026-08-26.

## Global Constraints

- Do not train Depth, CNN, SRU, PPO, or any other model.
- Do not modify the frozen P2P observation, history, actor, action interface, reward, success rule, or controller gains.
- Use `logs/rotunbot_target_repro/Aug16_02-57-06_uniform_t1_long500_from3809/model_4150.pt` and fail loudly if it is unavailable.
- Oracle outputs only local waypoints; it never emits joint/P2P actions or reads future robot state.
- Raw and all comparison variants use the exact same 100-episode manifest.
- Maze protocol is 120 s per episode; original uniform-4150 P2P protocol remains 60 s.

### Task 1: Lock the Oracle state machine in pure tests

**Files:**
- Create or modify: `legged_gym/tests/test_oracle_episode.py`
- Modify: `legged_gym/navigation/oracle_episode.py`

**Interfaces:**
- `OracleEpisodePlanner` exposes a deterministic state transition helper or equivalent planner state with `NAVIGATE` and `FINAL_APPROACH` states.
- The evaluator can identify final-approach entry, success, timeout, and escape without increasing ordinary waypoint count after entry.

- [ ] Step 1: Add a failing test where a path of one cell enters `FINAL_APPROACH` and does not return a repeated goal-cell waypoint.
- [ ] Step 2: Run the focused test and verify it fails for the current `select_next_waypoint(path, 0)` behavior.
- [ ] Step 3: Implement the smallest state-machine transition and final-goal selection needed for the test.
- [ ] Step 4: Add tests for final-approach success, timeout, and explicit escape accounting.
- [ ] Step 5: Run all pure Oracle/planner tests and keep them green.

### Task 2: Integrate state-machine metrics into Oracle Raw evaluation

**Files:**
- Modify: `legged_gym/scripts/evaluate_oracle_maze.py`
- Modify: `legged_gym/navigation/oracle_metrics.py`
- Test: `legged_gym/tests/test_oracle_metrics.py`

**Interfaces:**
- Episode summaries include `final_approach_entered`, `final_approach_success`, `final_approach_timeout`, `final_approach_escape`, `waypoint_count`, and local reach counts.
- Aggregate summaries include the corresponding entry/success/timeout/escape counts and preserve all required failure reasons.

- [ ] Step 1: Add failing aggregation tests for the four final-approach counters.
- [ ] Step 2: Run the focused metrics test and confirm the new counters are absent/failing.
- [ ] Step 3: Implement NAVIGATE→FINAL_APPROACH execution using exact global goal coordinates and no local reach/replan in final approach.
- [ ] Step 4: Record only explicit dynamic escape as `final_approach_escape`; do not silently resume ordinary waypoint scheduling.
- [ ] Step 5: Run CPU tests and a short GPU lifecycle smoke before Raw v1.

### Task 3: Run and validate Oracle Raw v1

**Files:**
- Runtime output: `/tmp/rotunbot_oracle_raw100_v1/`
- Reuse: `/tmp/rotunbot_oracle_raw100_ng/episode_manifest.json`

- [ ] Step 1: Verify the existing manifest is exactly 100 entries and matches the fixed seed-0 manifest used by accepted Raw 100.
- [ ] Step 2: Run Raw v1 with no reachability filter and no turn-aware policy.
- [ ] Step 3: Verify 100 episode summaries, no software integrity errors, and no repeated goal-cell waypoint loop.
- [ ] Step 4: Report Global SR, collision/timeout rates, waypoint count, local reach, completion time, actual/BFS path, SPL, and final-approach counters.

### Task 4: Add and test turn-aware waypoint switching

**Files:**
- Modify: `legged_gym/navigation/oracle_episode.py`
- Modify: `legged_gym/scripts/evaluate_oracle_maze.py`
- Modify: `legged_gym/navigation/oracle_metrics.py`
- Test: `legged_gym/tests/test_oracle_episode.py`

**Interfaces:**
- CLI switch `--turn-aware` is off by default.
- `abs(delta_bearing) >= 45°` requires waypoint distance `<=0.35 m` and speed `<=0.30 m/s`; smaller turns remain distance-only.
- Frozen 4150 control and success logic remain unchanged.

- [ ] Step 1: Add failing pure tests for small-turn distance-only and large-turn distance-plus-speed transitions.
- [ ] Step 2: Run focused tests and confirm the option is not implemented.
- [ ] Step 3: Implement segment-bearing calculation and deterministic switch criterion.
- [ ] Step 4: Run all pure tests and a short GPU turn-aware smoke.

### Task 5: Run Physical Reachability and four-way comparison

**Files:**
- Modify if needed: `legged_gym/scripts/measure_reachability.py`
- Modify: `legged_gym/scripts/evaluate_oracle_maze.py`
- Create or modify: `legged_gym/scripts/compare_oracle_variants.py`
- Test: `legged_gym/tests/test_oracle_metrics.py`

- [ ] Step 1: Run the existing physical action sweep and generate bearing-dependent success/reachability summaries for `0°, ±45°, ±90°, ±135°, 180°`.
- [ ] Step 2: Add a deterministic comparison runner for A Raw v1, B Raw v1 + Reachability, C Raw v1 + Turn-aware, and D both options, all using the same manifest.
- [ ] Step 3: Run the four 100-episode evaluations without model training.
- [ ] Step 4: Produce one comparison table and explicitly note the 3-repeat coverage sample-size limitation.
- [ ] Step 5: Stop before Dataset Collector and all Depth/CNN/SRU/PPO training unless a later user request changes scope.

### Task 6: Verification and delivery

- [ ] Step 1: Run compileall and all relevant CPU/unit tests.
- [ ] Step 2: Check no frozen P2P source files or prohibited learning modules changed.
- [ ] Step 3: Commit only implementation/tests/docs; preserve existing user artifacts.
- [ ] Step 4: Push `hierarchical-visual-nav` and report commit plus all Gate results.
