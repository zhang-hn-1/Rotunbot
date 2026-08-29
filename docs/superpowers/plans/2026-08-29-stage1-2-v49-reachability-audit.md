# Stage1.2 V49 Reachability Audit Implementation Plan

> Execute only after the approved design in
> `docs/superpowers/specs/2026-08-29-stage1-2-v49-reachability-audit-design.md`.

## Goal

Determine whether reset/state contamination explains the Stage1 A/B prefix
divergence and measure frozen V49's state-dependent response over a 0.2 s
command hold.

## Constraints

- Base `ac9e37a`, branch `codex/stage1-2-v49-reachability-audit`.
- No V49/control/reward/training/URDF/waypoint/damping changes.
- Runtime logs, plots, and checkpoints remain untracked.
- Use `/home/jason/legged_gym/.venv` and the existing frozen checkpoint.

## Tasks

### 1. Pure contracts (TDD)

- Add failing tests for snapshot comparison, first divergence, reset audit
  rows, 50 Hz alignment, ten-step holds, response metrics, sign correctness,
  low-speed bins, and CSV/JSON schema.
- Implement pure diagnostics and make all tests pass.

### 2. Part A state/reset audit

- Inventory environment reset buffers and locations without changing them.
- Add common-P1 A/B prefix capture and comparator through the first switch.
- Add isolated fresh-process cases A4/B5/B6/B7 and run-order sequences.
- Emit `A_B_prefix_equivalence.csv`, summary JSON, reset audit CSV/MD, and
  reproducibility CSV/JSON.
- Gate Part B on a clear equivalence/reproducibility result or explicit
  explanation of the remaining intrinsic forward-tracking failure.

### 3. Part B reachability sweep

- Drive direct projected commands using the existing task and policy.
- Establish initial states by V49 tracking, record stabilization status, then
  hold target commands for ten policy steps.
- Record raw 50 Hz traces and compute grid rows, response ratios, tracking and
  response reachable rates, sign failures, and low-speed/transition slices.
- Run the coarse sweep with five repeats per transition; do not expand to the
  full Cartesian product unless the runtime is demonstrably bounded.

### 4. Plots and report

- Generate the five required offline plots from the saved outputs.
- Write the executive conclusion, A/B/reset findings, dynamic envelope,
  revised H1-H8 ranking, and high-level-controller implications.
- Include explicit availability labels for every requested state buffer.

### 5. Verification and handoff

- Run V49 55-test regression, Stage1 9-test suite, Stage1.1 10-test suite,
  and new Stage1.2 tests.
- Run py_compile and output schema checks.
- Audit staged paths, commit the requested message, push the branch, and leave
  all logs/plots untracked.
