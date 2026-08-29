# Stage1.3/1.4 Dynamic Reachable Governor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, table-based V49 200 ms reachable-response model, then use it in an opt-in state-dependent command Governor and compare it fairly with baseline and static projection.

**Architecture:** Stage1.3 owns offline trace collection, aggregation, envelope extraction, interpolation, and prediction. Stage1.4 consumes the resulting table through a separate Governor that searches a finite candidate command grid and preserves the existing `project_velocity_commands()` hard guard. The default configuration keeps the Governor disabled and therefore preserves the Stage1.2 command path.

**Tech Stack:** Python 3.8, PyTorch 1.10, Isaac Gym GPU PhysX, NumPy, CSV/JSON, Matplotlib, unittest.

**Spec:** `/home/jason/.codex/attachments/1812524d-654a-40e2-b80c-ae06e1f88a47/pasted-text.txt`

## Global Constraints

- Base commit is `b572658` on `codex/stage1-2-v49-reachability-audit`.
- New branch is `codex/stage1-3-1-4-dynamic-reachable-governor`.
- V49 checkpoint, network, PPO, reward, URDF, contact model, planner, and static envelope constants are unchanged.
- Actual command interface is `[v, w]`: body-forward velocity in `m/s`, gravity-aligned world-Z yaw rate in `rad/s`.
- Runtime is 200 Hz physics (`0.005 s`), 50 Hz policy (`0.02 s`, decimation 4), and 5 Hz upper command (`10` policy steps per `0.2 s`).
- Formal runtime CSVs, PNGs, checkpoints, caches, and logs remain untracked; only source, tests, and Markdown reports are committed.
- Stage1.3 must be verified before Stage1.4 implementation begins.

---

### Task 1: Establish the branch and pure model API with failing tests

**Files:**
- Create: `legged_gym/navigation/v49_dynamic_reachability.py`
- Create: `legged_gym/tests/test_v49_dynamic_reachability.py`
- Modify: `legged_gym/navigation/__init__.py` only if the package requires an explicit export

**Interfaces:**
- Produces `ReachabilityState`, `ReachabilityPrediction`, `DynamicReachabilityTable`, and `DynamicReachabilityTable.from_rows(rows)`.
- Produces `predict_reachable_response(current_state, desired_command)` with deterministic coverage metadata, projected command, predicted v/w at 50/100/150/200 ms, and `out_of_coverage`.

- [ ] Step 1: Write failing tests for command units/state extraction, exact table lookup, multilinear/linear interpolation, clamp/out-of-coverage behavior, and symmetric yaw lookup.
- [ ] Step 2: Run `PATH=/home/jason/legged_gym/.venv/bin:$PATH PYTHONPATH=. python -m unittest legged_gym.tests.test_v49_dynamic_reachability` and confirm failure because the module/API is missing.
- [ ] Step 3: Implement the smallest table model: sorted current-v knots, projected forward-command knots, projected yaw-command knots, horizon fields, trilinear interpolation over current-v/projected-v/projected-w, deterministic nearest fallback for sparse cells, and explicit coverage flags.
- [ ] Step 4: Run the focused tests and confirm all pass; refactor only after green.

### Task 2: Implement Stage1.3 sweep and aggregation tests

**Files:**
- Create: `legged_gym/scripts/sweep_v49_stage1_3_dynamic_reachability.py`
- Create: `legged_gym/navigation/v49_stage1_3_diagnostics.py`
- Modify: `legged_gym/tests/test_v49_dynamic_reachability.py`

**Interfaces:**
- Consumes Stage1.2 `_make_runtime`, `_set_initial_pose`, `_set_command`, and `_projection` helpers.
- Produces raw trial rows, aggregated rows, and table-compatible rows with v0 `{0.00, .02, .04, .06, .08, .10, .12, .14}`, symmetric yaw grid, full allowed forward command grid, at least 3 repeats, 50/100/150/200 ms fields, cumulative yaw change, body displacement, sign/instability flags, seed, env id, trial id, simulation dt, and control dt.

- [ ] Step 1: Add pure failing tests for 50 Hz horizon indexing, five-horizon aggregation, symmetric command-grid generation, direction-reversal flags, and table-row schema.
- [ ] Step 2: Run the focused tests and confirm expected failures.
- [ ] Step 3: Implement diagnostics and a reusable sweep script that establishes initial state through V49 tracking, never root-velocity injection for formal transitions, records all required fields, and writes `logs/stage1_3_dynamic_reachability/`.
- [ ] Step 4: Run focused tests and a one-transition GPU smoke test; confirm exactly 10 policy steps per 200 ms transition and all required fields.

### Task 3: Run Stage1.3 formal sweep, extract envelope, plots, and report

**Files:**
- Create: `legged_gym/scripts/plot_v49_stage1_3_dynamic_reachability.py`
- Create: `docs/stage1_3_dynamic_reachability_report.md`

**Interfaces:**
- Consumes `logs/stage1_3_dynamic_reachability/` raw and aggregated CSVs.
- Produces a compact table artifact for the model, yaw envelope data, heatmap/response plots, and a report covering axes/units/dt/decimation, collapse boundaries, symmetry, time-window response, and whether forward-command changes restore low-speed yaw.

- [ ] Step 1: Run the required formal sweep with fixed seed and at least 3 repeats per combination.
- [ ] Step 2: Validate row count, stable initialization, finite values, symmetric grid, and all five horizons.
- [ ] Step 3: Generate plots and compact table data; check plots are non-empty and labels use physical units.
- [ ] Step 4: Write Stage1.3 report from measured values only, including explicit decision whether model input must retain forward command in addition to current v.
- [ ] Step 5: Run all Stage1.2 tests plus new Stage1.3 tests and mark Stage1.3 accepted only after every required check passes.

### Task 4: Add Governor configuration and pure candidate-search tests

**Files:**
- Create: `legged_gym/navigation/v49_dynamic_governor.py`
- Create: `legged_gym/tests/test_v49_dynamic_governor.py`
- Modify: `legged_gym/envs/rotunbot/vel_tracking/rotunbot_vel_config.py`

**Interfaces:**
- Produces `DynamicGovernorConfig` with all weights, candidate grid/rate limits, and `enable_dynamic_governor=False` defaults.
- Produces `StateDependentReachabilityGovernor.select_command(current_state, desired_command, previous_command)` returning command, prediction, fallback/coverage flags, and per-axis modification flags.

- [ ] Step 1: Write failing tests for preserving reachable commands, selecting a forward-adjusted candidate for low-speed high-yaw demand, hard projection bounds, no direction reversal, bounded command changes, deterministic tie-breaks, and out-of-coverage static fallback.
- [ ] Step 2: Run focused Governor tests and confirm failure because the module/config fields are missing.
- [ ] Step 3: Implement finite candidate enumeration, model prediction, configured objective `w_v*e_v² + w_yaw*e_yaw² + w_delta*||u-u_prev||²`, deterministic tie-break ordering, and static-projector fallback. Keep the V49 default path unchanged when disabled.
- [ ] Step 4: Run focused tests and then the full Stage1.3 test set.

### Task 5: Integrate the opt-in Governor without changing the downstream controller

**Files:**
- Modify: `legged_gym/envs/rotunbot/vel_tracking/rotunbot_vel.py`
- Modify: `legged_gym/scripts/evaluate_v49_waypoint_sequence.py` only if a shared command-application helper is needed
- Modify: `legged_gym/tests/test_v49_low_level_integration_contract.py` or add a focused integration test

**Interfaces:**
- Consumes `StateDependentReachabilityGovernor` through an explicitly attached/constructed runtime object.
- Preserves `set_command_targets()` semantics and applies the Governor before the existing hard projection only when the new config switch is enabled; disabled mode must byte-for-byte preserve the Stage1.2 command behavior at the interface level.

- [ ] Step 1: Add a failing integration test that compares disabled mode to the existing static command path and verifies enabled mode calls the Governor before downstream command application.
- [ ] Step 2: Run the focused integration test and confirm failure.
- [ ] Step 3: Add opt-in integration with no change to policy/action mapping, reward, gains, or physical simulation.
- [ ] Step 4: Run Stage1.2 regression, Stage1.3 model tests, and integration tests.

### Task 6: Implement fair Stage1.4 A/B/C experiment and report

**Files:**
- Create: `legged_gym/scripts/evaluate_v49_dynamic_governor.py`
- Create: `legged_gym/scripts/plot_v49_dynamic_governor.py`
- Create: `docs/stage1_4_dynamic_governor_report.md`

**Interfaces:**
- Consumes the Stage1.3 compact table and the same fixed seeds/initial-state protocol.
- Produces `logs/stage1_4_dynamic_governor/` Baseline/Static/Dynamic trial CSV, aggregate CSV, JSON summary, and plots for all 10 required scenarios.

- [ ] Step 1: Write pure failing tests for scenario schema, baseline/static/dynamic mode identity, percentile metrics (mean/median/P90/max), modification/saturation/fallback counters, and oscillation counting.
- [ ] Step 2: Run focused tests and confirm failure.
- [ ] Step 3: Implement matched-seed fresh-process or matched-reset evaluation for the three modes, preserving the same desired command sequence and simulation settings.
- [ ] Step 4: Generate metrics and plots, including separate `v_current<=.08` and `v_current>=.10` groups.
- [ ] Step 5: Write the Stage1.4 report without tuning heuristics after seeing results; mark PASS/PARTIAL/FAIL from measured comparisons.

### Task 7: Full verification, artifact hygiene, commit, and push

**Files:**
- Modify: any implementation/report files only for verified defects

- [ ] Step 1: Run all V49 regression, Stage1, Stage1.1, Stage1.2, Stage1.3, and Stage1.4 tests.
- [ ] Step 2: Run `py_compile`, `git diff --check`, CSV/JSON schema checks, and confirm no source code imports runtime logs as committed data.
- [ ] Step 3: Confirm `git status` shows only intentional untracked logs; do not stage `logs/`, PNGs, checkpoints, caches, or temporary files.
- [ ] Step 4: Commit with `feat: add V49 dynamic reachable governor`.
- [ ] Step 5: Push `codex/stage1-3-1-4-dynamic-reachable-governor` and verify the remote ref points to the final commit.
