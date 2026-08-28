# Stage1.1 Root-Cause Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add read-only runtime diagnostics and controlled A/B experiments that identify the causes of Stage 1 waypoint failures without changing V49 control performance.

**Architecture:** Keep the existing waypoint controller and V49 environment untouched. Extend the Stage 1 evaluator with an explicit diagnostic mode, tick-level snapshots, episode classifiers, and summary aggregation; isolate pure calculations in a navigation diagnostics module so they can be tested without a GPU.

**Tech Stack:** Python 3.8, PyTorch 1.10, Isaac Gym, NumPy, unittest, CSV/JSON/Markdown.

**Spec:** `/home/jason/.codex/attachments/43411a5c-6014-4111-b206-1876c032b179/pasted-text.txt`

## Global Constraints

- Base branch and commit: `codex/stage1-v49-waypoint-switching@60be1d8`.
- Do not modify V49 checkpoint, policy, PPO/reward/config limits, URDF, contact damping parameters, waypoint coordinates, or formal controller gains.
- Default diagnostic mode must preserve current Stage 1 behavior: `smooth=False`, rolling floor disabled.
- All rolling-floor requests must still pass the existing `project_velocity_commands()`.
- Raw CSV/JSON logs, checkpoints, TensorBoard output, and caches stay untracked.

### Task 1: Pure diagnostic contracts

**Files:**
- Create: `legged_gym/navigation/v49_waypoint_diagnostics.py`
- Test: `legged_gym/tests/test_stage1_root_cause_diagnostics.py`

- [x] Write failing tests for baseline defaults, rolling-floor behavior, transition metrics, yaw reversals, low-speed collapse, and completion-only settling metrics.
- [x] Run the focused test and verify the expected import/function failure.
- [x] Implement pure helpers with explicit inputs/outputs and no simulator access.
- [x] Run focused tests until all diagnostic contracts pass.

### Task 2: Runtime instrumentation and controlled modes

**Files:**
- Modify: `legged_gym/scripts/evaluate_v49_waypoint_sequence.py`

- [x] Add CLI flags `--diagnostic_smooth_reference` and `--diagnostic_minimum_rolling_speed` with baseline-safe defaults.
- [x] Add every required 5 Hz geometry, low-level tracking, V49 action-component, smooth-reference, authority, contact-damping, actuator, event, and terminal-settling field.
- [x] Apply rolling floor only to nonterminal raw requests and re-project through V49; do not modify controller configuration.
- [x] Separate incomplete-route last velocity from completed-route settling metrics.
- [x] Add per-trajectory, per-yaw, per-failed-waypoint, transition-distribution, and failure-signature summaries.
- [x] Run unit tests and a 2+2 GPU baseline smoke, verifying required fields and default equivalence.

### Task 3: Reference diagnostic experiment matrix

**Files:**
- Generate untracked: `logs/stage1_1_root_cause/{baseline,smooth_true,rolling_floor_010,combined}/`
- Generate: `docs/stage1_1_root_cause_report.md`

- [x] Run reference baseline A/B, 10 episodes each, with fixed shared initial conditions.
- [x] Run smooth-reference A/B, 10 episodes each, changing only `command_reference_is_smooth`.
- [x] Run rolling-floor A/B, 10 episodes each, changing only diagnostic raw-speed floor.
- [x] Run combined only if smooth or rolling floor meets the specified improvement threshold.
- [x] Produce detailed summaries for three baseline failures and three improved episodes when available.

### Task 4: Maze reproduction and handoff

**Files:**
- Modify: `docs/stage1_1_root_cause_report.md`

- [x] Reproduce only the strongest reference mechanism on `Rotunbot.urdf`.
- [x] Rank H1-H7 using only logged evidence and state primary/secondary causes.
- [x] Run V49 55-test and Stage 1 9-test regressions.
- [ ] Audit staged paths, commit `diagnostics: add Stage1 V49 root-cause audit`, and push `codex/stage1-1-root-cause-audit` without merging.
