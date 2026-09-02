# T-junction MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a goal-conditioned T-junction MVP on the frozen V1 SRU + V62 chain.

**Architecture:** Add a symmetric fixed T geometry, an auditable teacher/data path, pure gate metrics, and a recurrent visual evaluator. Reuse the existing observation ABI, waypoint manager, real IsaacGym depth assertion, dataset schema, mixed BC trainer, and Frozen V62 controller.

**Tech Stack:** Python, NumPy, PyTorch, IsaacGym, unittest/pytest, existing legged_gym V1 navigation modules.

**Spec:** `docs/superpowers/specs/2026-09-02-t-junction-mvp-design.md`

## Global Constraints

- Real IsaacGym IMAGE_DEPTH only; assert `depth_backend_actual == "isaacgym"`.
- Keep T=16, 5 Hz high-level control, repeat=10, Frozen V62, current observation ABI, and done hidden resets.
- T_LEFT/T_RIGHT must share geometry, spawn, yaw, horizon, and seed; only goal differs.
- Do not add S/multi-junction/maze/planner/RL/new encoder/new SRU/new V62/ROS.
- Never stage unrelated existing `logs/` files.

---

### Task 1: Pure T geometry and gate metrics

**Files:**
- Create: `legged_gym/navigation/v1_t_junction.py`
- Create: `legged_gym/navigation/v1_t_junction_metrics.py`
- Test: `legged_gym/tests/test_t_junction_navigation.py`

**Interfaces:**
- `build_t_junction_geometry(branch: str, width_m=3.0, stem_length_m=2.5, branch_length_m=2.5, reach_radius_m=0.35) -> T junction geometry dataclass`.
- `classify_t_branch(...) -> "LEFT"|"RIGHT"|"UNDECIDED"` and `aggregate_t_gate(records, pairs, ablations) -> dict`.
- Metrics validate finite values and compute success, collision, timeout, wrong-turn, turn completion, exit reach, branch accuracy, and goal consistency.

- [ ] **Step 1: Write failing tests** for mirrored goals, identical wall segments, branch classification, paired consistency, gate thresholds, and finite/backend assertions.
- [ ] **Step 2: Run `pytest -q legged_gym/tests/test_t_junction_navigation.py`** and confirm failure because the T module is absent.
- [ ] **Step 3: Implement only the dataclasses, symmetric wall/waypoint construction, branch classifier, and pure metric aggregation required by the tests.
- [ ] **Step 4: Rerun the focused test and then the existing waypoint/dataset tests.
- [ ] **Step 5: Commit `feat: add auditable T-junction geometry and gates`.

### Task 2: T teacher and dataset collector

**Files:**
- Create: `legged_gym/scripts/collect_sru_visual_t_junction_teacher.py`
- Create: `legged_gym/scripts/audit_t_junction_teacher_dataset.py`
- Modify: `legged_gym/navigation/v1_teacher_dataset.py` only if a tested schema/audit helper is needed.
- Test: `legged_gym/tests/test_t_junction_navigation.py`

**Interfaces:**
- Teacher supports `T_LEFT` and `T_RIGHT`, uses `V1WaypointManager`, emits projected `(v_cmd,w_cmd)`, and records required episode fields and failure traces.
- Collector CLI accepts `--episodes`, `--seed`, `--output-dir`, and `--dataset-output`, configures real depth, and fails on any non-IsaacGym backend.
- Audit CLI writes `*_audit.json` with counts, T, finite ratio, ordering, backend, and command ranges.

- [ ] **Step 1: Add tests for teacher left/right sign, wrong-turn/turn/exit detection, dataset schema, finite checks, and backend rejection.
- [ ] **Step 2: Run focused tests and confirm the new collector/helper behaviors fail before implementation.
- [ ] **Step 3: Implement the minimal collector by adapting the existing L teacher path without changing the frozen control stack.
- [ ] **Step 4: Run focused tests and `py_compile` for the new scripts.
- [ ] **Step 5: Commit `feat: add real-depth T-junction teacher collection`.

### Task 3: Mixed Straight/L/T imitation and student evaluator

**Files:**
- Create: `legged_gym/scripts/train_sru_visual_t_junction_imitation.py`
- Create: `legged_gym/scripts/eval_sru_visual_t_junction.py`
- Create: `legged_gym/scripts/eval_t_junction_counterfactual.py` if kept separate for CLI clarity.
- Test: `legged_gym/tests/test_t_junction_navigation.py`

**Interfaces:**
- Training CLI accepts Straight/L/T datasets, applies integer-equivalent weights 1:3:5, preserves T=16, warm-starts the approved mixed checkpoint, and records train/validation/v/w MAE plus sampling distribution.
- Student evaluation writes T_LEFT/T_RIGHT episode CSV/trace, gate JSON, paired counterfactual results, and normal/zero/swapped goal ablation without retraining.
- All IsaacGym evaluation paths assert real depth and reset recurrent state per environment/episode.

- [ ] **Step 1: Add tests for sampler ratios, counterfactual pair construction, ablation goal modes, and student gate aggregation.
- [ ] **Step 2: Run focused tests and confirm the missing T trainer/evaluator APIs fail.
- [ ] **Step 3: Implement training/evaluation by reusing current imitation and L evaluator contracts; do not add policy inputs or network layers.
- [ ] **Step 4: Run focused tests, script compilation, and a CPU dataset-only audit.
- [ ] **Step 5: Commit `feat: add goal-conditioned T-junction student evaluation`.

### Task 4: Real IsaacGym gates and report

**Files:**
- Create: `logs/phase_c/t_junction_teacher_gate_20260902/t_junction_teacher_gate.json` (untracked artifact; never stage logs unless explicitly selected).
- Create: `logs/phase_c/t_junction_teacher_dataset_v1_20260902.pt` and its audit JSON.
- Create: `logs/phase_c/t_junction_student_20260902/t_junction_student_gate.json` and traces.
- Create: `logs/phase_c/T_JUNCTION_MVP_REPORT_20260902.md`.
- Modify: `docs/sru_visual_corridor_curriculum_report.md` with the verified ledger entry.

- [ ] **Step 1: Run 20+20 teacher gate and stop if teacher success, collision, or wrong-turn thresholds fail.
- [ ] **Step 2: Collect at least 20+20 finite ordered real-depth episodes and run dataset audit.
- [ ] **Step 3: Train the 1:3:5 mixed BC checkpoint with the approved parent and record all losses.
- [ ] **Step 4: Run 20+20 T student gate, paired counterfactuals, normal/zero/swapped ablation, and Straight/L regression in separate IsaacGym processes.
- [ ] **Step 5: Run full pytest excluding only the known ROS `test_nav.py` import failure, inspect all non-ROS failures, and write an exact PASS/FAIL report.
- [ ] **Step 6: Review `git diff`, stage only code/tests/docs, commit `feat: add goal-conditioned T-junction navigation MVP`, and try `git push origin codex/corridor-curriculum-navigation`.
