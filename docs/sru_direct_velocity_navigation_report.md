# SRU Direct Velocity Navigation Report

## Fixed interface

`Depth_t + goal_xy_robot_t + proprioception_t + previous (v,w)_t -> Depth Encoder -> single-step SRU block -> velocity head -> V62 transition manager/governor -> actuator`.

The current model is a stateless, length-one SRU-block baseline; it has no
persistent hidden state or temporal depth history. Training currently uses the
fallback backend of 32 horizontal rays replicated across 8 rows, not a
calibrated two-dimensional Isaac Gym depth image.

The SRU does not output actuator actions or local waypoints. The frozen V62 parent is `/home/jason/Rotunbot_SRU50_V62_SafeYaw_Final_Verified_20260829/model/model_150.pt`.

## Phase 0 / S0

The frozen V62 stack was spatially validated before navigation training:

- S0A straight: 20/20 success; collision, timeout, divergence, near-miss, stuck, oscillation, rate, domain, and hidden-projection counts all zero.
- S0B L corridor: 20/20 success; all safety/violation counts zero.
- S0C double turn: 30/30 success; all safety/violation counts zero.

Freeze Point 1 passed. Evidence is under `logs/formal_s0a`, `logs/formal_s0b_final`, and `logs/formal_s0c_final`.

## Stage S1

- Objective: open-space goal tracking, 0.5–1.0 m and ±10°.
- Training: 300 iterations, seed 4, from scratch.
- Formal evaluation: fixed 16-env protocol, 100 episodes.
- Result: SR 100%, collision 0%, timeout 0%, mean terminal distance 0.3481 m.
- Checkpoint: `logs/sru_velocity/S1/S1_best.pt`.
- SHA256: `9fc84118640671bdea0bc8ecde1d2bac1c2d79a01f10b0a09f0137ea345caaca`.
- Recorded decision under the evaluator/interface used at the time: PASS; allowed to enter S2. This stage must be revalidated under the forward-only and complete-safety protocol described below before it is treated as a current formal Gate result.

## Stage S2

- Objective: expand goal bearing to ±30°, distance 0.5–1.5 m.
- Training: loaded S1 and continued to 800 total iterations; 30% S1 replay.
- Current evaluation: 97/100 success, 3 timeout, 0 collision.
- S1 regression: 100/100 success, 0 timeout, 0 collision.
- Checkpoint: `logs/sru_velocity/S2/S2_best.pt`.
- SHA256: `38dc54b9cf55ea28203b73b670704db297fa527a0edc2e40e354bfbfa3ca407e`.
- Recorded decision under the evaluator/interface used at the time: PASS; allowed to enter S2B. This stage must also be revalidated under the corrected protocol before formal continuation.

The first S2 attempt reached only 82–86% because goal-side turn direction was not stable. A causal `goal_turn_alignment` shaping term was added and tested; the final S2 result passed without changing the V62 actuator path.

## Stage S2B status on 2026-08-31

- Objective: open-space direct-velocity goal tracking over 0.5–2.0 m and ±45°, with 70% S2B, 20% S2, and 10% S1 replay.
- Required Gate: S2B SR >= 90%, S2 regression >= 90%, S1 regression >= 93%, and all bottom-level violation counts zero.
- Latest full-distribution checkpoint: `logs/rotunbot_sru_direct_velocity_s2b/Aug30_21-01-46_/model_3100.pt`.
- Latest checkpoint SHA256: `9031ed7f61d5e245d7eca1d856d9f8b42a8bce42278f928fe57867c0e5288b0c`.
- Latest full evaluation: 79/100 success, 21 timeout, 0 collision; mean terminal distance 0.4260 m.
- Targeted 0.5–1.0 m, ±45° hard-set evaluation: 46/50 success, 4 timeout, 0 collision (92%). This is diagnostic only and is not the S2B Gate.
- Failure mode: large initial bearing produces a long-timeout orbit. Failed trajectories can finish with the goal 100–146° behind the robot even though the yaw-command sign remains goal-aligned.
- Decision: **FAIL**. No `S2B_best.pt` was created; the strict B3 checkpoint
  chain remains unavailable.

The best complete formal S2B artifact remains `logs/phase_b/S2B_clean_model900`:
80/100 success, 20 timeouts, 0 collisions, and zero rate, feasible-domain, and
hidden-projection-jump violations. The recovery-aware 66/100 and reverse-lateral
68/100 experiments also failed B3 and are not selected as a parent checkpoint.

The S2B experiment sequence was:

| Run | Main change | Full S2B SR | Collision | Timeout |
| --- | --- | ---: | ---: | ---: |
| `Aug30_19-26-19_` | first S2B continuation | 86% | 0% | 14% |
| `Aug30_19-38-52_` | stronger turn alignment | 82% | 0% | 18% |
| `Aug30_19-53-23_` | distance-dependent approach speed | 74% | 0% | 26% |
| `Aug30_20-06-24_` | complete V62 curvature projector at the SRU boundary | 74% | 0% | 26% |
| `Aug30_20-20-04_` | overshoot recovery shaping | 75% | 0% | 25% |
| `Aug30_20-28-54_` | kinematic-recovery shaping | 74% | 0% | 26% |
| `Aug30_20-44-46_` | intermediate-to-full curriculum | 77% | 0% | 23% |
| `Aug30_20-53-11_` | near/high-bearing hard-case curriculum | 77% | 0% | 23% |
| `Aug30_21-01-46_` | explicit hard-case recovery reward | 79% | 0% | 21% |

The first two rows predate the complete V62 curvature-projection alignment and are retained only as historical evidence. The retained best complete
full-distribution result is 80%, not the earlier 86% diagnostic result.

## Visual Entry Decision

The strict B3/S2B Gate remains independent and failed:

```text
S2B formal status: FAIL
best formal/diagnostic SR: 80/100
```

The new research-stage gate was evaluated from the current artifacts and the
2026-08-31 audit:

```text
VISUAL_ENTRY_GATE: PASS
```

Evidence:

- S0A straight, S0B L, and S0C double-turn are all PASS with zero collision,
  rate, feasible-domain, and hidden-projection-jump violations.
- S1 is 100/100 and S2 is 97/100 in the retained fixed evaluation artifacts;
  the complete S2B regression chain is PASS.
- Best S2B is 80%, above the visual-entry floor of 75%, while its formal 90%
  gate remains FAIL.
- The direct SRU → `(v_cmd,w_cmd)` → V62 command path, Depth observation
  contract, hidden-state reset, parallel-environment isolation, and corridor
  focused test coverage were verified.
- Current P0/P1 focused audit: 134/134 tests PASS across direct velocity,
  V62/corridor, depth, timing, curriculum, evaluation, and SRU-state checks.
  The repository-wide suite still contains unrelated legacy failures
  (`test_nav` requires ROS `rospy`; one V49 integration contract is missing a
  legacy method).

The V1 visual-corridor gate was then executed on a model-only warm-start
checkpoint using an external process for every evaluation. The 2.5m/3.0m
30+30 evaluation reached 18/30 and 14/30 successes respectively, with 1 and
5 collisions. The fixed 6m×100 formal evaluation used exact 6.0m initial
goals and reached 4/100 success, SPL 0.04, 37 collisions, and 59 timeouts.
These results are **V1 FAIL**; V2/L/double-turn/S curricula remain blocked.

The fallback depth normalization contract passes, but actual Isaac Gym
`IMAGE_DEPTH` currently returns `-Inf` for all 256 raw pixels under the V1
camera/scene, so the visual result is fallback-depth-only and real-depth
calibration is not passed. The SRU hidden-state audit confirms the current
policy is intentionally stateless (`is_recurrent=False`, fresh zero hidden on
each length-one call, no hidden state in rollout storage).

The former in-process V1 evaluator was removed after it caused a native Isaac
Gym segmentation fault when creating a second simulation before the training
simulation exited. The replacement is
`legged_gym/scripts/run_sru_visual_corridor_v1_curriculum.py`, which launches
training and evaluation as sequential isolated subprocesses and persists the
curriculum JSON between them.

This decision authorizes the V1 research probe and does not create an
`S2B_best.pt` or rewrite any historical B3 result.

## Obstacle avoidance and maze status

- S0 used corridor walls to validate scripted V62 spatial execution, but this is not learned obstacle avoidance.
- The reusable depth camera, fallback depth model, corridor geometry, and procedural maze infrastructure exist in the repository.
- The current direct-velocity task deliberately runs with `maze.enabled = False` and `scene_mode = "none"` during S1/S2/S2B.
- V1 learned depth-corridor training has started but its current formal Gate is
  FAIL; V2/L/double-turn/S are blocked. S2B remains a formal FAIL.
- M1 5×5, M2 9×9, and M3 15×15 direct-SRU maze training have not started. Existing maze/local-depth or actuator-action policies are historical infrastructure/baselines, not results for this new SRU→velocity→V62 route.

## Safety and evaluation protocol

All direct policies route desired `(v,w)` through the existing V62 command projection and transition manager. No direct SRU-to-actuator path was added. Evaluation disables observation/domain randomization and uses a fixed 16-environment protocol; single-env and hard-only results are not mixed with the full-distribution Gate. The evaluator now records independent rate, feasible-domain, and hidden-projection-jump counters plus transition/governor/projection activation and command-correction telemetry. Failed episodes retain replay artifacts.

## Current decision and next work

Current milestone: **S2B formal FAIL / VISUAL_ENTRY_GATE PASS / P0 timing correction PASS, 5 Hz-aligned S2 parent PASS / P1-A current-goal reward PASS / P1-B curriculum foundation PASS / V1 formal FAIL / Maze NOT STARTED**.

Next: improve V1 distance-transfer performance while preserving the independent
process evaluator, then rerun the exact 30+30 gate and fixed 6 m×100 formal
evaluation. The actual repository primitive clock is
`sim.dt=0.005 s`, decimation `4`,
`env.dt=0.02 s`; therefore the corrected 5 Hz PPO macro transition uses
`repeat=10`. The first fixed-6 m run was stopped for lack of learning evidence;
the training entry
now uses a bounded 2--6 m transfer curriculum while formal evaluation remains
fixed at 6 m. Oracle
waypoints remain diagnostic/reference only and are not actor inputs.
