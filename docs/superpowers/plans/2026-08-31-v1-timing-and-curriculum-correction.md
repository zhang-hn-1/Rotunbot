# V1 Timing and Curriculum Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align direct-velocity PPO transitions with the real 5 Hz high-level command interface, then establish a reproducible V1 visual-corridor training and evaluation path.

**Architecture:** Keep `Depth + goal_xy_robot + proprioception + previous command + approved state -> single-step SRU-block policy -> (v_cmd,w_cmd) -> frozen V62`. The environment will expose macro transitions at the configured high-level frequency while V62 continues its existing low-level execution inside each macro step. V1 training will use a performance-gated distance curriculum; formal evaluation will remain fixed at 6 m.

**Tech Stack:** Python 3.8, PyTorch 1.10, Isaac Gym, existing `DWLOnPolicyRunner`/`PPODWL`, unittest, CSV/JSON diagnostics, matplotlib where already used by the repository.

**Spec:** `/home/jason/.codex/attachments/96f1274c-53fa-451b-bcb4-0d661ca03e32/pasted-text.txt`

## Global Constraints

- Branch remains `codex/corridor-curriculum-navigation`; preserve the S2B formal result `80/100 FAIL` and `VISUAL_ENTRY_GATE PASS`.
- V62 velocity tracker, transition manager, governor, feasible projection, actuator mapping, camera resolution, and network architecture remain unchanged.
- Current network terminology is `single-step SRU block` or `stateless SRU-block baseline`; no persistent hidden state is introduced in V1/V2.
- Fallback depth is documented as 32 horizontal rays replicated across 8 rows; it is not called a true 2D depth image.
- Every behavior change follows failing test -> observed failure -> minimal implementation -> focused regression -> commit.
- Do not restore old optimizer state for new parents; model-only migration is the only warm-start path.
- Do not start V2, Maze, or stateful SRU before the V1 gates in this plan pass.

---

### Task 1: Capture deterministic V1 distance OOD evidence

**Files:**
- Create: `legged_gym/scripts/diagnose_v1_distance_action.py`
- Create: `legged_gym/tests/test_v1_distance_diagnostics.py`
- Create: `logs/diagnostics/v1_distance_action_scan.csv`
- Create: `logs/diagnostics/v1_distance_action_scan.png`
- Modify: `docs/sru_visual_corridor_curriculum_report.md`

**Interfaces:**
- Consumes the selected S2 parent checkpoint and the existing direct-velocity inference path.
- Produces a deterministic CSV with `distance_m`, normalized goal, raw action, mapped command, projected command, and explicit zero-crossing distances.

- [ ] Write tests for deterministic 0.25 m sampling from 0.50 m through 6.00 m, fixed inputs, and required CSV columns.
- [ ] Run the new tests and confirm they fail because the diagnostic entry point and output contract do not exist.
- [ ] Implement inference-only scanning with fixed pose/proprio/depth/previous-command/recovery state and no stochastic action sampling.
- [ ] Run the scan with `S2_best.pt`, generate the CSV and plot, and record the first raw and mapped forward zero crossings.
- [ ] Run the focused diagnostic tests and verify the output is reproducible on a second invocation.
- [ ] Commit as `test: add V1 distance-action OOD diagnostics` plus the diagnostic implementation if the repository convention keeps script and test together.

### Task 2: Run clipped-goal causal diagnostic

**Files:**
- Modify: `legged_gym/scripts/diagnose_v1_distance_action.py`
- Modify: `legged_gym/tests/test_v1_distance_diagnostics.py`
- Create: `logs/diagnostics/v1_clipped_goal_causal.csv`
- Modify: `docs/sru_visual_corridor_curriculum_report.md`

**Interfaces:**
- Consumes the same fixed actor input fixture as Task 1 and changes only actor-visible `goal_xy_robot`; physical goal metadata remains 6 m.
- Produces paired normal-6 m and clipped-2 m raw/mapped action rows and a causal conclusion without changing training configuration.

- [ ] Add a failing test that requires paired normal/clipped observations and preserves the physical 6 m target.
- [ ] Run the test to observe the missing paired diagnostic contract.
- [ ] Implement the temporary observation-only clipping branch inside the diagnostic script, never inside the environment or training config.
- [ ] Run the paired experiment and verify whether normal 6 m reverses while clipped 2 m drives forward.
- [ ] Record the result as a diagnostic hypothesis, not as the final root cause until timing alignment is corrected.
- [ ] Commit as `test: cover V1 clipped-goal causal diagnostic` and `diag: record V1 distance OOD evidence` as appropriate.

### Task 3: Specify high-level timing semantics

**Files:**
- Create: `docs/superpowers/specs/2026-08-31-high-level-action-timing-design.md`
- Inspect completely: `legged_gym/dwl/on_policy_runner_dwl.py`, `legged_gym/dwl/ppo_dwl.py`, rollout storage, `legged_gym/envs/rotunbot/direct_velocity/rotunbot_direct_velocity.py`, and V62 reset/termination paths.

**Interfaces:**
- Defines primitive env duration, high-level duration, dynamically derived integer repeat factor, discounted reward aggregation, macro gamma/lambda semantics, per-env done handling, and storage alignment.
- The design must state `repeat = round((1 / frequency) / env.dt)` and reject non-integral ratios beyond the configured tolerance.

- [ ] Write the design with the current measured values (`env.dt ~= 0.04 s`, high-level `5 Hz`, repeat `5`) and equations for discounted macro reward and `gamma_macro = gamma_primitive ** repeat`.
- [ ] Add a checklist mapping each design invariant to a concrete test before implementation.
- [ ] Review the design against all direct-velocity call paths and document which layer owns action holding and reward aggregation.
- [ ] Commit as `docs: specify high-level action timing semantics`.

### Task 4: Add macro-step and timing contract tests

**Files:**
- Create: `legged_gym/navigation/high_level_action_timing.py`
- Create: `legged_gym/tests/test_high_level_action_timing.py`
- Modify: `legged_gym/dwl/on_policy_runner_dwl.py`
- Modify: the existing rollout storage module identified in Task 3.

**Interfaces:**
- `derive_action_repeat(env_dt: float, high_level_frequency_hz: float, tolerance: float = 1e-6) -> int`.
- `aggregate_discounted_rewards(rewards, primitive_gamma) -> macro_reward`.
- A macro collector returns exactly one `(obs, action, log_prob, value, macro_reward, done, next_obs)` record per sampled high-level action.

- [ ] Write failing tests for one action per macro transition, exact repeat count, discounted reward aggregation, success/collision/timeout on repeat steps 2/3/4, per-env done isolation, invalid non-integral ratio rejection, and storage length.
- [ ] Run the tests and confirm failures are caused by absent timing utilities/collector behavior.
- [ ] Implement the smallest timing utility and collector changes without touching V62 internals or PPO optimization hyperparameters.
- [ ] Run all timing tests and verify macro gamma/lambda preserve primitive physical-time semantics.
- [ ] Commit tests and implementation separately where the repository allows: `test: cover macro transition timing` then `fix: align direct-velocity PPO with 5Hz command timing`.

### Task 5: Verify deterministic action holding in Isaac Gym

**Files:**
- Create: `legged_gym/scripts/diagnose_high_level_action_timing.py`
- Create: `logs/diagnostics/high_level_action_timing.csv`
- Modify: `docs/sru_visual_corridor_curriculum_report.md`

**Interfaces:**
- Consumes a deterministic policy output `a=[0.5,0.2]` and records policy sample id, primitive step, raw action, requested command, and applied command.
- Produces evidence that one policy decision owns one complete 0.2 s high-level period.

- [ ] Run the deterministic timing script against a 4-env V1 smoke setup.
- [ ] Verify each high-level action is held for the derived repeat count and that env0 reset cannot contaminate env1 or a new env0 episode.
- [ ] If evidence contradicts the design, stop and update the timing design/tests before any training.
- [ ] Commit as `diag: verify high-level command holding`.

### Task 6: Rebuild and regression-test a 5 Hz-aligned S2 parent

**Files:**
- Create or modify only the dedicated 5 Hz training entry point under `legged_gym/scripts/`.
- Create: `logs/sru_velocity/S2_5Hz_adapted_best.pt` and its metadata when the run passes.
- Modify: `docs/sru_visual_corridor_curriculum_report.md`.

**Interfaces:**
- Uses model-only warm start from `logs/sru_velocity/S2/S2_best.pt`, fresh optimizer, unchanged network/reward hyperparameters, and the corrected macro-transition collector.
- Produces an adapted checkpoint with S1/S2 regression artifacts and safety counters.

- [ ] Run a 100-300 iteration short adaptation, never a direct 1500-iteration V1 run.
- [ ] Evaluate S1 and S2 with fixed definitions and require S1 >= 93%, S2 >= 90%, and all safety/projection counters zero.
- [ ] If regression fails, stop at this gate and report evidence; do not change rewards/network/PPO hyperparameters.
- [ ] Commit only checkpoint metadata/report references; keep large logs untracked.

### Task 7: Correct V1 reward to current-goal progress

**Files:**
- Modify: `legged_gym/envs/rotunbot/visual_corridor_v1/rotunbot_visual_corridor_v1.py`
- Modify: `legged_gym/envs/rotunbot/visual_corridor_v1/rotunbot_visual_corridor_v1_config.py`
- Modify: `legged_gym/tests/test_visual_corridor_v1.py`

**Interfaces:**
- V1 uses `d_t = ||p_t - g_t||` and `r_progress = d_(t-1) - d_t` through `_reward_goal_progress`.
- V1 reward scales set `goal_progress=20.0` and `path_progress=0.0`; actor observation remains free of oracle path data.

- [ ] Add failing tests for forward progress, moving away, lateral correction, and use of the current curriculum goal.
- [ ] Run the tests and observe failure under the current path-progress implementation.
- [ ] Implement the minimal current-goal progress method and configuration change.
- [ ] Run V1 and direct-velocity regression tests.
- [ ] Commit as `fix: use current-goal progress for V1`.

### Task 8: Implement performance-gated V1 distance curriculum

**Files:**
- Create: `legged_gym/navigation/v1_curriculum.py`
- Create: `legged_gym/tests/test_v1_curriculum.py`
- Modify: `legged_gym/envs/rotunbot/visual_corridor_v1/rotunbot_visual_corridor_v1.py`
- Modify: `legged_gym/scripts/train_sru_visual_corridor_v1.py`
- Create: `curriculum_state.json` beside each V1 checkpoint.

**Interfaces:**
- Curriculum levels are `2.5, 3.0, 4.0, 5.0, 6.0 m`; sampling is 70% `Uniform(2.0,current_max)` and 30% frontier `Uniform(max(2.0,current_max-0.25),current_max)`.
- State fields are `current_level`, `current_max_distance`, `level_start_iteration`, `consecutive_pass_count`, and `internal_eval_history`.
- Promotion requires 30/30 frontier/replay evaluations at thresholds 26/30 and 27/30, collision <=1/30, all three safety counters zero, at least 50 iterations, and two consecutive passes.

- [ ] Write failing tests for mixed sampling bounds, promotion thresholds, failure freeze, state save/resume, and no reset to 2.5 m after resume.
- [ ] Run the tests and confirm missing curriculum state/sampling behavior.
- [ ] Implement stateful curriculum bookkeeping and fixed-seed internal evaluations every 50 PPO iterations.
- [ ] Verify failed levels freeze and emit diagnostics for raw action, mapped command, reverse ratio, timeout, terminal distance, progress, and correction.
- [ ] Commit as `feat: add performance-gated V1 distance curriculum`.

### Task 9: Add fixed-6 m V1 formal evaluator and Depth ablation

**Files:**
- Create: `legged_gym/scripts/evaluate_sru_visual_corridor_v1.py`
- Create: `legged_gym/tests/test_visual_corridor_v1_evaluator.py`
- Create: `logs/phase_b/v1_formal/summary.json`, CSV, and plots.
- Modify: `docs/sru_visual_corridor_curriculum_report.md`.

**Interfaces:**
- Evaluator supports fixed 6 m, 100 fixed seeds, normal depth, and masked depth using the same checkpoint and initial conditions.
- Outputs SR, collision/timeout/near-miss, initial reverse episodes, first-2 s mean command, path length/efficiency, centerline deviation, terminal metrics, safety counters, transition/governor/projection activation, mean/P95 correction.

- [ ] Write failing tests for fixed-distance episode generation, metric schema, normal-vs-masked pairing, and V1 gate thresholds.
- [ ] Run the tests and confirm the evaluator contract is absent.
- [ ] Implement the evaluator without retraining the masked variant.
- [ ] Run the fixed 6 m 100-episode evaluation and depth ablation; require V1 SR >= 90%, collision <= 5%, and zero rate/domain/hidden-jump violations before V2.
- [ ] Commit as `feat: add fixed-6m V1 formal evaluator`.

### Task 10: Calibrate fallback vs Isaac Gym IMAGE_DEPTH

**Files:**
- Create: `legged_gym/scripts/calibrate_v1_depth_backends.py`
- Create: `legged_gym/tests/test_v1_depth_backend_calibration.py`
- Create: `logs/depth_backend_calibration/` CSV, images, and `summary.json`.
- Modify: both navigation reports.

**Interfaces:**
- Samples centered, +/-0.25 m lateral, +/-10 degree yaw, and near-left/right-wall poses.
- Compares fallback horizontal-ray output to Isaac Gym `IMAGE_DEPTH` for middle row, nearest distance, wall distances, FOV edge, normalization, and camera frame orientation.

- [ ] Write failing tests for required calibration poses, output schema, and explicit backend labels.
- [ ] Run tests to confirm the calibration contract is missing.
- [ ] Implement calibration and inspect the real-image tensor path.
- [ ] Run fallback-trained V1 zero-shot on real depth for 100 episodes; fine-tune only if the zero-shot result materially drops, keeping V62/reward/interface fixed.
- [ ] Commit as `docs: record fallback and real-depth calibration` with code/test changes as needed.

### Task 11: Documentation correction and final regression

**Files:**
- Modify: `docs/sru_visual_corridor_curriculum_report.md`
- Modify: `docs/sru_direct_velocity_navigation_report.md`
- Modify: relevant checkpoint metadata and evaluation summaries.

**Interfaces:**
- Reports explicitly state `single-step SRU block`, fallback depth as replicated horizontal rays, `S2B formal FAIL`, and `VISUAL_ENTRY_GATE PASS`.
- Reports include timing semantics, OOD evidence, adapted parent identity, curriculum history, fixed-6 m V1 metrics, ablations, and remaining blockers.

- [ ] Update reports only from generated artifacts and verified test output.
- [ ] Run the full focused regression suite and `git diff --check`.
- [ ] Verify no V2/Maze artifacts are claimed before V1 gates pass.
- [ ] Commit as `docs: correct stateless SRU and fallback depth status`.
- [ ] Push only the reviewed commits to `origin/codex/corridor-curriculum-navigation` and report exact branch/commit/status.
