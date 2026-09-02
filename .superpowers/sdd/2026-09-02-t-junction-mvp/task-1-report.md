# Task 1 report

## Status

Implemented and committed as `feat: add auditable T-junction geometry and gates`.

## Changes

- Added `legged_gym/navigation/v1_t_junction.py` with finite-input validation,
  fixed symmetric 3.0 m T geometry, mirrored T_LEFT/T_RIGHT waypoints, shared
  wall segments/AABBs, branch classification, and reach radius metadata.
- Added `legged_gym/navigation/v1_t_junction_metrics.py` with episode rates,
  branch accuracy, paired goal consistency, backend assertion, finite check,
  ablation passthrough, and release-gate checks.
- Added `legged_gym/tests/test_t_junction_navigation.py` covering geometry
  mirroring, walls, classification, validation, metrics, thresholds, backend,
  and finite outputs.

## TDD evidence

RED focused attempt:

```text
$ python3 legged_gym/tests/test_t_junction_navigation.py
ModuleNotFoundError: No module named 'legged_gym.navigation.v1_t_junction'
```

The `pytest` executable and pytest module are unavailable in this environment;
the repository's unittest style was used instead. The normal package import
also requires Isaac Gym.

GREEN focused command (isolated package loading to avoid unrelated Isaac Gym
initialization):

```text
$ /home/jason/legged_gym/.venv/bin/python -c '...unittest...'
Ran 6 tests in 0.009s
OK
```

Additional syntax check:

```text
$ .../python -m py_compile <three Python files>
PASS
```

Relevant existing L test attempt:

```text
$ PYTHONPATH=. /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_l_turn_navigation
RuntimeError: Ninja is required to load C++ extensions
```

This is an environment dependency failure while importing the pre-existing
navigation package (`isaacgym.gymtorch`), not a T-junction test failure.

## Concerns

- Full standard unittest execution is blocked because the available Isaac Gym
  installation needs Ninja to build `gymtorch`; pytest is not installed.
- Existing untracked `logs/` directories were left untouched and were not
  included in the commit.

## Review remediation (2026-09-02)

### Fixed findings

- Rebuilt the T wall input as three **centreline** segments: the positive-x
  stem plus equal positive/negative-y branches.  This matches
  `RotunbotVelCorridor._create_envs`, which creates one actor on each side of
  every centreline.  For the default 3.0 m T, consumer-visible centres are
  `(1.25, +/-1.5)`, `(1.0, +/-1.25)`, and `(4.0, +/-1.25)`; there is no wall
  at the stem centreline.  The two branch-side actors at `x=4.0` form the
  forward junction wall while leaving both turns open.
- Replaced corner-pair obstacle output with the required
  `(center, half_extents)` format.  The six default fallback AABBs now exactly
  mirror the six wall actors and all half extents are positive.
- Added `wall_actor_centers`, a pure implementation of the production
  centreline-to-actor offset, so topology remains directly testable without
  importing Isaac Gym.
- Replaced the 80%/10% aggregate gate with per-`T_LEFT`/`T_RIGHT` checks:
  success >= 0.95, collision = 0, and wrong turn = 0 for both roles; student
  additionally requires timeout <= 0.05, turn completion >= 0.95, and paired
  goal consistency >= 0.95.
- `aggregate_t_gate` now rejects absent/non-`isaacgym`
  `depth_backend_actual`, recursively rejects non-finite Python/NumPy/
  tensor-like numeric payloads (including ablations), requires both sides and
  one policy role, and validates counterfactual pairs as distinct LEFT/RIGHT
  episodes with unique identities, side-matched expected branches, matching
  `seed`/`initial_pose`/`initial_yaw`/`horizon`, and opposite correct branch
  predictions.

### TDD and verification evidence

RED after adding the new consumer-visible contract test:

```text
ImportError: cannot import name 'wall_actor_centers' from
'legged_gym.navigation.v1_t_junction'
```

GREEN focused run, using an isolated package loader solely to bypass the
unrelated `legged_gym.navigation.__init__` Isaac Gym import chain:

```text
Ran 12 tests in 0.019s
OK
```

The focused suite covers the AABB centre/half-extent contract and actor-centre
topology, left/right mirroring, 19/20 pass and 18/20 fail per-side thresholds,
collision/wrong-turn zero tolerance, teacher zero-tolerance behavior,
backend/recursive-finite rejection, pair duplicate/side/identity/metadata
rejection, and true paired accuracy.

Additional checks:

```text
/home/jason/legged_gym/.venv/bin/python -m py_compile \
  legged_gym/navigation/v1_t_junction.py \
  legged_gym/navigation/v1_t_junction_metrics.py \
  legged_gym/tests/test_t_junction_navigation.py
exit 0

git diff --check
exit 0
```

The real requested command remains blocked before test collection:

```text
PYTHONPATH=. /home/jason/legged_gym/.venv/bin/python -m unittest -v \
  legged_gym.tests.test_t_junction_navigation
RuntimeError: Ninja is required to load C++ extensions
```

This is the pre-existing Isaac Gym `gymtorch` build dependency during package
initialization, not an isolated focused-test failure.  No `logs/`, SRU,
encoder, V62, or L files were changed.

## Fix round 2 (2026-09-02)

### Status and scope

Resolved the remaining Task 1 topology and paired-evidence findings.  The
only runtime consumer change is the backward-compatible explicit-wall option
in `legged_gym/navigation/v62_corridor_task.py`; legacy
`corridor_wall_segments` keeps its existing two-sided-offset behavior.
Untracked `logs/` directories and all SRU, encoder, V62-controller, and L
work were left untouched.

### TDD evidence

RED, after replacing the topology expectation and adding pair-reuse/coverage
regressions, ran through the isolated package loader:

```text
Ran 14 tests in 0.020s
FAILED (failures=3)
```

The three expected failures were the old double-offset actor layout, repeated
episode IDs accepted across pairs, and a student evidence set accepted with an
unpaired record.

GREEN, after the minimal implementation:

```text
Ran 14 tests in 0.020s
OK
```

### Changes

- Added optional `corridor_explicit_wall_segments` handling to
  `RotunbotVelCorridor._create_envs`.  Each non-empty `(start, end)` item now
  creates exactly one fixed actor at its supplied midpoint.  Legacy entries
  still create the previous two normal-offset actors.
- Rebuilt `TJunctionGeometry.wall_segments` as five direct physical walls:
  stem boundaries at `y=+/-1.5` from `x=0..2.5`; branch inner boundaries at
  `x=1.0` from `y=+/-1.5..+/-2.5`; and the continuous front boundary at
  `x=4.0` from `y=-2.5..+2.5`.  It emits one `(center, half_extent)` fallback
  AABB per explicit wall, using the fixed actor's 0.05 m thickness.
- Replaced the prior expected blocking layout with an actor/AABB occupancy
  regression: all sampled stem-centre points through the junction are open,
  while `(4.0, 0.0)` is occupied by the front wall.
- Hardened `_pair_consistency` so an episode ID cannot appear in different
  pairs; student pair IDs must exactly cover every release-gate record once.
  Existing same-side, duplicate-record, metadata, missing-field, and expected
  branch checks remain in force.

### Verification

```text
/home/jason/legged_gym/.venv/bin/python -m py_compile \
  legged_gym/navigation/v1_t_junction.py \
  legged_gym/navigation/v1_t_junction_metrics.py \
  legged_gym/navigation/v62_corridor_task.py \
  legged_gym/tests/test_t_junction_navigation.py
exit 0

git diff --check
exit 0
```

The normal focused unittest invocation remains blocked before collection by
the pre-existing Isaac Gym `gymtorch` extension build:

```text
RuntimeError: Ninja is required to load C++ extensions
```

This is an environment dependency blocker, not an isolated Task 1 test
failure.
