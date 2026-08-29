# Stage1.2 V49 State/Reset Validation and Dynamic Reachability Audit

## Objective

Validate whether the Stage1 failure traces are caused by episode-state/reset
contamination and characterize the frozen V49 controller's actual response over
one 0.2 s high-level hold. The work is diagnostic-only: it must not modify the
checkpoint, controller law, release limits, URDF, waypoint controller, or
training artifacts.

## Components

1. `v49_stage1_2_diagnostics.py` contains pure, CPU-testable comparators and
   reachability metrics. It compares nested runtime snapshots in a fixed data
   flow order and computes response ratios, sign correctness, and reachable
   rates without simulator access.
2. `audit_v49_state_reset.py` runs the common-P1 A/B prefix test, emits the
   reset-state inventory, and provides a fresh-process case runner. It records
   first divergence rather than declaring A/B equivalent from aggregate values.
3. `sweep_v49_dynamic_reachability.py` drives the existing task directly with
   commands, holds each command for exactly ten 50 Hz policy steps, and writes
   raw 50 Hz traces plus grid/summary outputs. Initial velocity states are
   established by V49 tracking; root velocity is never directly injected as a
   formal result.
4. `plot_stage1_2_reachability.py` reads the CSV/grid outputs and generates the
   five required diagnostic plots offline.

## Reset audit semantics

The audit reads existing buffers and reset code. It reports each requested
state as `available`, `not_available`, or `not_applicable`, with location,
observed before/after values, expected reset value, and status. A reset fix is
allowed only if a concrete omission is found and separately evidenced; no
control behavior is changed as part of instrumentation.

The A/B prefix comparator uses the same initial pose, seed, model, command,
and P1. It compares snapshots at every 50 Hz policy step until the first
waypoint switch, with absolute and relative differences and a configurable
floating-point tolerance. Fresh-process and run-order cases are recorded as
separate cases, not mixed into the prefix result.

## Reachability semantics

The coarse sweep uses stable initial states for `v0` in
`{0.06, 0.08, 0.10, 0.12}` and `w0` in `{0, -0.02, 0.02}`, and targets in
`v={0.06, 0.08, 0.10, 0.12}`, `w={0, -0.02, -0.01, 0.01, 0.02}`. Each
transition has five repeats unless an initial state cannot stabilize, in which
case it is explicitly recorded as `initial_state_not_stabilized`.

Every target is passed through the existing `project_velocity_commands()` and
both raw and projected targets are logged. Every 20 ms sample records command,
measured body/world velocity, yaw, joint state, existing action buffers, and
available contact/reset buffers. No missing runtime variable is fabricated.

`response_reachable` means the final response has target-direction sign
correctness and a nonzero target-directed response. `tracking_reachable` uses
the existing V49 tracking tolerances where available; any additional threshold
is labeled `analysis_threshold` in the output and report. The report also
shows continuous response ratios and 20/100/200 ms errors, avoiding a single
binary criterion.

## Outputs and verification

Runtime outputs stay under `logs/stage1_2_reachability/` and are not committed.
The committed report is `docs/stage1_2_v49_reachability_report.md`. Existing
V49, Stage1, and Stage1.1 tests must pass, along with new Stage1.2 unit/schema
tests. The final branch is committed with
`feat: add V49 state reset and dynamic reachability audit` and pushed without
merging.
