# Task 2 report: real-depth T-junction teacher collection

## Delivered

- Added `legged_gym/scripts/collect_sru_visual_t_junction_teacher.py`.
  - Supports only `T_LEFT` and `T_RIGHT`; both use the accepted
    `build_t_junction_geometry` geometry, identical seed/default horizon, and
    deterministic no-random-spawn configuration.
  - Uses `V1WaypointManager` to approach the junction, select the side's
    waypoint, then preserve the geometry's final global goal for termination.
  - Produces `teacher_velocity_diagnostics` commands, normalizes them to the
    V1 action range, and holds them through `env.step`.
  - Sets `corridor_explicit_wall_segments=geometry.wall_segments`, clears
    `corridor_wall_segments`, preserves `direct_obstacle_aabbs`, requests
    `camera.depth_backend="isaacgym"`, and rejects every non-IsaacGym actual
    backend both immediately after env creation and at rollout start.
  - Defaults to 20 episodes per side and applies the accepted strict teacher
    gate: per-side success >=95%, collision rate exactly zero, and wrong-turn
    rate exactly zero. Per-episode records include scenario, seed, goal,
    initial pose/yaw, horizon, primitive/macro steps, timeout, turn/exit
    outcomes, and detailed macro-step failure traces.
  - Optional `--dataset-output` writes the unchanged V1 schema through
    `TeacherSequenceWriter(sequence_length=16)`, with real-depth provenance,
    ordered per-episode scenario metadata, geometry, seed, and command ranges.

- Added `legged_gym/scripts/audit_t_junction_teacher_dataset.py`.
  - Its `--dataset`/`--output` CLI loads through `load_teacher_dataset`.
  - Audit rejects non-real-depth provenance, schema mismatches, non-finite
    values, non-terminal or early `done`, non-chronological step ids, missing
    T sides, and episode-to-scenario metadata mismatches.
  - JSON output reports backend provenance, finite/terminal/chronology checks,
    episode count, T_LEFT/T_RIGHT counts, macro-step count, and sequence
    length.

- Extended `legged_gym/tests/test_t_junction_navigation.py` with pure contract
  coverage for T yaw signs, wrong-turn/turn/exit classification, required
  evidence fields, strict backend rejection, and dataset schema/finiteness/
  terminal/chronology/side-count audit behavior.

## TDD evidence

1. Added the collector/audit contract tests before either CLI existed.
2. Ran an isolated pure-module unittest loader: RED was observed as one missing
   collector module assertion plus five missing-module contract errors; the 14
   pre-existing T tests passed.
3. Added the minimal collector and audit implementations.
4. Added an early-terminal-`done` audit regression test, observed its RED
   assertion, then made the audit reject any pre-terminal `done=True`.
5. Final isolated run: 20 tests passed.

## Verification

- Isolated focused unittest: `20` passed.
- `python3 -m py_compile legged_gym/scripts/collect_sru_visual_t_junction_teacher.py legged_gym/scripts/audit_t_junction_teacher_dataset.py`: passed.
- `git diff --check`: passed.
- Normal package unittest was attempted. It cannot begin in the current shell:
  `/usr/bin/python3` raises `ModuleNotFoundError: No module named 'isaacgym'`
  from `legged_gym/__init__.py`. This environment therefore did not reach the
  possible `gymtorch`/Ninja build stage and no IsaacGym rollout or formal gate
  was run here.

## Scope and preservation

- Did not modify V1 dataset schema, SRU, Depth Encoder, V62 controller, or
  any existing log directories.
- Existing untracked `logs/` content was preserved and excluded from the task
  commit.
