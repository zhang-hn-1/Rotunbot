# Task 7 Continuation Implementer Report

## Status

**BLOCKED.** The implementation and evaluator contract are complete and focused/full relevant CPU regressions pass, but the corrected bounded-reverse policy has not been trained or formally evaluated. The latest formal-format B3 diagnostic of the inherited checkpoint is 77/100, below the strict 90/100 B3 gate. Per the controller handoff instruction, no further long GPU job was started.

## Implementation / attempt

- Added deterministic fixed-seed formal evaluation for S1/S2/S2B with seeds `0,1,2`, exact S2B `70/20/10` mixture, checkpoint/parent SHA verification, wall-clock time, strict gate verdicts, episode CSV, aggregate and per-failure trajectories/plots, and explicit raw/requested/applied command instrumentation.
- Added independent counters for raw reverse requests, requested reverse commands, applied reverse commands, Transition Manager activation, reverse-transition activation, active transition steps, governor activation, projection activation, rate violations, feasible-domain violations, and hidden projection jumps.
- Corrected the minimum-radius direct-turn test to the hand-derived chord boundary `d < 2 R sin(|bearing|)`.
- Added a stateful bounded-reverse recovery phase with entry/exit hysteresis. Rewards consume the latched phase; reset clears it. Reverse remains a policy request and is applied only through the unchanged frozen V62 Transition Manager.
- Preserved the strict B1/B2/B3 chain: B3 `>=90%`, B2 regression `>=90%`, B1 regression `>=93%`, and all required V62 safety counters exactly zero. Missing safety fields fail closed.
- Preserved all untracked logs. No corridor stage was started.

## Physical diagnosis before reward/controller changes

The inherited B3 checkpoint was evaluated on the exact fixed-seed S2B mixture. It reached 77/100 with 23 timeouts and no collision or safety violation. All raw/requested/applied reverse counts were zero.

Inspection of the 23 complete failed trajectories found:

- all 23 began inside the corrected direct-turn circle; 21 were S2B samples and 2 were S2 samples;
- all 23 failures crossed into the rear half-plane, while none of the 77 successes did;
- after entering the rear half-plane, mean applied linear velocity remained `+0.1023 m/s` and mean actual velocity `+0.1017 m/s`;
- median remaining path after the rear crossing was `1.711 m`, while median goal-distance reduction was `-1.105 m` (distance worsened);
- 24 successful episodes also began inside the corrected boundary, motivating a latched phase with hysteresis rather than blanket reverse classification.

This supports bounded reverse recovery for the trapped phase, followed by forward motion once direct forward curvature is feasible. It does not establish gate passage without training.

## TDD RED and GREEN evidence

RED evidence captured before implementation:

- Existing direct-policy test modules initially failed import with `PyTorch was imported before isaacgym`; the test import order was then corrected.
- Geometry test failed because `inside_minimum_radius_turn_circle` did not exist.
- Evaluator tests failed because the deterministic goal builder, strict gate-chain evaluator, checkpoint identity loader, command diagnostics, failure artifact writer, and summary contract did not exist.
- Stateful recovery tests failed because `update_goal_recovery_phase` and the `recovery_active` reward inputs did not exist.
- Environment integration test showed an inactive fixture receiving approximately `0.999` recovery reward before the latched phase was wired into the environment.

GREEN evidence after implementation:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_direct_velocity_policy legged_gym.tests.test_direct_velocity_evaluation legged_gym.tests.test_corridor_artifacts
Ran 27 tests in 0.310s
OK
```

Earlier complete relevant regression run on the same implementation:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_direct_velocity_policy legged_gym.tests.test_direct_velocity_evaluation legged_gym.tests.test_corridor_artifacts legged_gym.tests.test_rotunbot_velocity_tracking legged_gym.tests.test_feasible_transition_manager legged_gym.tests.test_vel_sru50_structured_random
Ran 122 tests in 0.512s
OK
```

The main-thread handoff additionally reported `21/21` direct/V62 regression tests and `116/116` combined tests passing.

## Evaluation commands / results

Required runtime prefix was used: `PATH=/home/jason/legged_gym/.venv/bin:$PATH`, `/home/jason/legged_gym/.venv/bin/python`, GPU 0 only, serially.

Initial evaluator invocations established two environment issues before rollout: direct file-path invocation imported the installed package and hit legacy NumPy `np.float`; module invocation without `PYTHONPATH=.` hit unsupported NVRTC `sm_89`. Using the repository's tracked `sitecustomize.py` through `PYTHONPATH=.` resolved the RTX 4070 / PyTorch 1.10 compatibility path.

Successful diagnostic form:

```text
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m legged_gym.scripts.evaluate_sru_direct_velocity --checkpoint logs/rotunbot_sru_direct_velocity_s2b/Aug30_21-01-46_/model_3100.pt --stage S2B --episodes 100 --num_envs 16 --output_dir logs/phase_b/diagnostic_s2b_old_68b8379 --headless
```

Result: 100 episodes in `158.268 s`; 77 success, 23 timeout, 0 collision, 0 divergence, 0 rate violations, 0 feasible-domain violations, 0 hidden projection jumps. Raw/requested/applied reverse counts were all 0. The evaluator emitted 23 per-failure artifact directories plus aggregate CSV/plots and `summary.json`.

No corrected-policy training or post-change formal B3/B2/B1 evaluation was run before handoff.

## Checkpoint identity and gate rates

Diagnostic B3 checkpoint:

- path: `logs/rotunbot_sru_direct_velocity_s2b/Aug30_21-01-46_/model_3100.pt`
- SHA-256: `9031ed7f61d5e245d7eca1d856d9f8b42a8bce42278f928fe57867c0e5288b0c`
- parent: `logs/rotunbot_sru_direct_velocity_s2b/Aug30_20-53-11_/model_2800.pt`
- parent SHA-256: `7aac537a6523a36f78ca216489d939131e17ccd0552a5bdfc8a89dd1168767b7`

Gate evidence:

| Gate | Rate | Safety evidence | Verdict |
|---|---:|---|---|
| B3/S2B fresh diagnostic | 77/100 | collision/divergence/rate/domain/hidden-jump = 0; reverse = 0 | FAIL (`<90%`) |
| B2/S2 accepted prior evidence | 97/100 | accepted by addendum; not rerun with new evaluator | PASS evidence retained |
| B1/S1 regression accepted prior evidence | 100/100 | accepted by addendum; not rerun with new evaluator | PASS evidence retained |
| B1/S1 accepted stage evidence | 100/100 | accepted by addendum | PASS evidence retained |

Accepted prior checkpoint paths are `logs/sru_velocity/S2/S2_best.pt` and `logs/sru_velocity/S1/S1_best.pt`. They are not promoted as a completed post-change chain because B3 has not passed and the regressions were not rerun under the completed evaluator.

Frozen V62 checkpoint remains `/home/jason/Rotunbot_SRU50_V62_SafeYaw_Final_Verified_20260829/model/model_150.pt`, SHA-256 `d7173fbbb113ab790d25b0587e82a73abd7ffad9ab2ed148387ba04084944f1b`.

## Files changed

- `legged_gym/navigation/direct_velocity.py`
- `legged_gym/navigation/direct_velocity_evaluation.py`
- `legged_gym/envs/rotunbot/direct_velocity/rotunbot_direct_velocity.py`
- `legged_gym/envs/rotunbot/direct_velocity/rotunbot_direct_velocity_config.py`
- `legged_gym/scripts/evaluate_sru_direct_velocity.py`
- `legged_gym/tests/test_direct_velocity_policy.py`
- `legged_gym/tests/test_direct_velocity_evaluation.py`
- `.superpowers/sdd/2026-08-29-corridor-curriculum-navigation/task-7-implementer-report.md`

Implementation commit: `b33eca5` (`feat: add bounded B3 recovery evaluation`), parent `68b8379`.

## Self-review

- Scope: changes are limited to the direct-velocity controller reward/state, formal evaluator, and their tests/report.
- Architecture: recovery state is owned by the direct-velocity environment; the evaluator observes the full command path rather than replacing any V62 component.
- Gate compliance: thresholds are strict and missing safety fields fail. B3 failure prevents promotion and corridor curricula.
- Safety fields: raw/requested/applied reverse and transition/governor/projection activity are independently represented; rate/domain/hidden-jump violations remain zero in the diagnostic.
- Tests: hand-derived geometry boundaries, phase hysteresis, environment integration, exact mixtures, strict chain behavior, provenance, counters, and failure artifacts are covered.
- Frozen stack: no diff exists in the V62 Transition Manager or velocity-tracking controller. The external frozen checkpoint hash matches the required SHA.
- Artifacts: all pre-existing and newly generated logs remain untracked and preserved; no bulk log artifact was staged.
- Review limitation: no subagent/code-review dispatch was made because the task explicitly prohibited subagents.

## Issues / concerns

- The corrected recovery behavior is unit/integration tested but untrained. Its effect on B3 success and V62 transition activity is unknown until a single serial GPU training/evaluation chain is authorized and completed.
- The strict chain is still blocked at B3. Do not start B2/B1 regression promotion or corridor curricula until a corrected B3 checkpoint reaches at least 90/100 with all safety counters zero, followed by B2 >=90/100 and B1 >=93/100 under the completed evaluator.
- The inherited diagnostic checkpoint did not request reverse, so it cannot validate the newly instrumented reverse-transition path in physical rollout.

## Fix Round 1 — reviewer findings after `b33eca5`

### Status and scope

Fix Round 1 is implemented and CPU-verified. No S2B redesign, GPU training, checkpoint evaluation, or frozen V62 change was performed. Task 7 itself remains blocked at the previously reported B3 gate because this fix round was limited to evaluator correctness and tests.

Changed in this round:

- `CommandDiagnostics` now keeps separate prior-applied samples for rate checking and hidden projection-jump checking. A hidden jump compares the current projected-applied command with the independently retained prior applied command using explicit `(v, w)` thresholds `(0.02 m/s, 0.01 rad/s)`. Rate violations continue to use the configured acceleration limits times `policy_dt`.
- Direct-velocity environments now snapshot terminal applied command, measured velocity, position, command target, goal state, transition active/state, recovery phase, and terminal outcome before the inherited reset clears V62 runtime buffers.
- The evaluator no longer substitutes pre-step telemetry or forces `transition_active=False` on automatic termination. It selects the captured terminal post-step snapshot when available, otherwise the observed post-step data.
- A real integration-boundary test sends action `[-0.2, 1.0]` through `normalized_action_to_velocity_command` and the unchanged `project_velocity_commands`. The raw command is `[-0.05, 0.10]`; the requested/applied feasible command is approximately `[-0.05, 0.025]`. It is finite, bounded, reverse in all three command stages, and telemetry separately reports projection activity versus governor activity.

### Callback-order audit

`LeggedRobot.post_physics_step()` calls `_post_physics_step_callback()` before termination and reward. The direct-velocity callback first runs `RotunbotVel._post_physics_step_callback()`, which refreshes tracking motion and advances/copies the applied feasible command, then refreshes goal geometry and the recovery latch. Reward therefore sees the current recovery state, while command telemetry reflects the current V62 update. `reset_idx()` is called only after reward and now snapshots those values before delegating to the inherited reset. No callback reorder or extra ordering-only test was needed; the terminal-selector test covers the consumer boundary where stale data previously entered the artifact.

### TDD RED evidence

Independent jump behavior test before implementation:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_direct_velocity_evaluation.DirectVelocityEvaluationTests.test_projection_jump_and_rate_violation_can_fail_independently
TypeError: __init__() got an unexpected keyword argument 'projection_jump_threshold'
Ran 1 test in 0.001s
FAILED (errors=1)
```

Terminal post-step selector test before implementation:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_direct_velocity_evaluation.DirectVelocityEvaluationTests.test_terminal_step_selector_keeps_exposed_post_step_telemetry
AttributeError: module 'legged_gym.navigation.direct_velocity_evaluation' has no attribute 'select_step_telemetry'
Ran 1 test in 0.001s
FAILED (errors=1)
```

### GREEN and regression evidence

Independent jump/rate focused tests:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_direct_velocity_evaluation.DirectVelocityEvaluationTests.test_projection_jump_and_rate_violation_can_fail_independently legged_gym.tests.test_direct_velocity_evaluation.DirectVelocityEvaluationTests.test_reverse_requests_and_transition_application_are_measured_separately
Ran 2 tests in 0.001s
OK
```

Terminal selector focused test:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_direct_velocity_evaluation.DirectVelocityEvaluationTests.test_terminal_step_selector_keeps_exposed_post_step_telemetry
Ran 1 test in 0.001s
OK
```

Real negative-command projection boundary:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_direct_velocity_evaluation.DirectVelocityEvaluationTests.test_negative_action_crosses_real_projection_boundary_with_distinct_telemetry
Ran 1 test in 0.017s
OK
```

Focused evaluator module:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_direct_velocity_evaluation
Ran 9 tests in 0.202s
OK
```

Combined relevant direct-velocity/V62 suite:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_direct_velocity_policy legged_gym.tests.test_direct_velocity_evaluation legged_gym.tests.test_corridor_artifacts legged_gym.tests.test_rotunbot_velocity_tracking legged_gym.tests.test_feasible_transition_manager legged_gym.tests.test_vel_sru50_structured_random
Ran 125 tests in 0.525s
OK
```

Syntax verification:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m py_compile legged_gym/navigation/direct_velocity_evaluation.py legged_gym/scripts/evaluate_sru_direct_velocity.py legged_gym/envs/rotunbot/direct_velocity/rotunbot_direct_velocity.py legged_gym/tests/test_direct_velocity_evaluation.py
exit 0
```

`git diff --check` returned exit 0. `git diff --name-only b33eca5 -- legged_gym/envs/rotunbot/vel_tracking legged_gym/navigation/v62_corridor_controller.py` returned no paths, confirming that the frozen V62 implementation was not modified.

### Remaining evidence limitation

The old 77/100 diagnostic's `hidden_projection_jump_count=0` was produced by the pre-fix aliased definition. It has not been recomputed with the independent threshold-based metric because this round explicitly prohibited GPU evaluation/training. No replacement rollout metric is claimed.

## Fix Round 2 — real V62 transition-manager boundary

### Scope and result

Re-review finding 3 is covered by a new real integration-boundary test, `test_negative_action_crosses_real_v62_transition_manager_boundary`. Production code is unchanged. No GPU process, training, simulator rollout, checkpoint evaluation, or frozen V62 edit was performed.

The test uses the actual `FeasibleVelocityTransitionManager` with the frozen V62 transition parameters. It first establishes a feasible positive state `(v, w) = (0.14, 0.035)`, then sends normalized action `[-0.2, 1.0]` through `normalized_action_to_velocity_command`. This produces:

- raw command `[-0.05, 0.10]`;
- requested feasible command `[-0.05, 0.025]`;
- first manager-applied command still positive and smaller than `0.14 m/s`, while the manager is in `BRAKE_TO_ORIGIN` with `transition_active=True`;
- eventual finite, bounded negative applied command after the real brake, settle, and accelerate phases.

`CommandDiagnostics` observes the actual manager output. On transition activation it reports raw reverse `1`, requested reverse `1`, applied reverse `0`, projection active `1`, governor active `1`, transition active `1`, and transition activation event `1`. Once the manager begins applying reverse, applied reverse becomes `1`. The aggregate reports exactly one transition activation and one reverse-transition activation. The test would fail if the manager were bypassed, if the transition did not activate, if requested and applied commands were conflated, or if reverse never reached the applied command.

This characterization passed on its first run, demonstrating that the reviewer finding was a missing integration test rather than a missing production behavior:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_direct_velocity_evaluation.DirectVelocityEvaluationTests.test_negative_action_crosses_real_v62_transition_manager_boundary
Ran 1 test in 0.029s
OK
```

Focused evaluator suite:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_direct_velocity_evaluation
Ran 10 tests in 0.217s
OK
```

Combined relevant direct-velocity/V62 suite:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_direct_velocity_policy legged_gym.tests.test_direct_velocity_evaluation legged_gym.tests.test_corridor_artifacts legged_gym.tests.test_rotunbot_velocity_tracking legged_gym.tests.test_feasible_transition_manager legged_gym.tests.test_vel_sru50_structured_random
Ran 126 tests in 0.518s
OK
```

## Fix Round 3 — reverse-drive yaw alignment

### Root cause and bounded correction

Formal S2B evidence showed `model_900.pt` at 80/100 and `model_1100.pt`
at 65/100. Their traces contained raw and eventually applied reverse
commands, but failed trajectories drove away from the target. Inspection of
`goal_turn_alignment` led to an initial hypothesis that its
`sign(bearing) * w_cmd` term was wrong for reverse motion. The signed-unicycle
relation `beta_dot = -w - (v/r)sin(beta)` disproved that hypothesis: the yaw
term has the same convergence sign for either signed linear velocity.

The correction is confined to `goal_turn_alignment`: it keeps
`sign(bearing) * w_cmd` for forward, reverse, and near-zero commands. No
reward scale, recovery latch, action mapping, evaluator, Transition Manager,
governor, projector, frozen V62 implementation, or checkpoint was changed.

### Hand-derived TDD fixture

The regression uses positive and negative goal bearings with six literal
`(v_cmd, w_cmd)` fixtures: two forward commands steering toward the goal, two
meaningful reverse commands steering with the same goal-side yaw sign, and two
near-threshold reverse commands (`-0.01` and `-0.0101 m/s`). For
every fixture, `|w_cmd| = 0.05 rad/s`, so the independently derived expected
reward is the literal `tanh(0.05 / 0.10) = 0.462117`.

Before the production change, the focused test failed as intended because the
pre-fix reverse fixtures expected the incorrect sign convention:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_direct_velocity_policy.DirectVelocityPolicyTests.test_goal_turn_alignment_keeps_goal_side_yaw_sign_for_reverse_motion
Ran 1 test in 0.003s
FAILED (failures=1)
```

After the sign selection was removed, the same focused test passed:

```text
Ran 1 test in 0.022s
OK
```

The full focused policy module then passed:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_direct_velocity_policy
Ran 17 tests in 0.024s
OK
```

The established combined direct-velocity/V62 CPU regression passed:

```text
PATH=/home/jason/legged_gym/.venv/bin:$PATH /home/jason/legged_gym/.venv/bin/python -m unittest -v legged_gym.tests.test_direct_velocity_policy legged_gym.tests.test_direct_velocity_evaluation legged_gym.tests.test_corridor_artifacts legged_gym.tests.test_rotunbot_velocity_tracking legged_gym.tests.test_feasible_transition_manager legged_gym.tests.test_vel_sru50_structured_random
Ran 128 tests in 0.538s
OK
```

No GPU command, simulator rollout, training, or checkpoint evaluation was run.
Task 7 remains blocked on a future post-commit B3 training/evaluation gate; this
fix does not claim B3 passage.
