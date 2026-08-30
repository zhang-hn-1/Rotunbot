# SRU Direct Velocity Navigation Report

## Fixed interface

`Depth + goal_xy_robot + proprioception + previous (v,w) -> Depth Encoder -> SRU -> velocity head -> V62 transition manager/governor -> actuator`.

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

## Stage S2B status on 2026-08-30

- Objective: open-space direct-velocity goal tracking over 0.5–2.0 m and ±45°, with 70% S2B, 20% S2, and 10% S1 replay.
- Required Gate: S2B SR >= 90%, S2 regression >= 90%, S1 regression >= 93%, and all bottom-level violation counts zero.
- Latest full-distribution checkpoint: `logs/rotunbot_sru_direct_velocity_s2b/Aug30_21-01-46_/model_3100.pt`.
- Latest checkpoint SHA256: `9031ed7f61d5e245d7eca1d856d9f8b42a8bce42278f928fe57867c0e5288b0c`.
- Latest full evaluation: 79/100 success, 21 timeout, 0 collision; mean terminal distance 0.4260 m.
- Targeted 0.5–1.0 m, ±45° hard-set evaluation: 46/50 success, 4 timeout, 0 collision (92%). This is diagnostic only and is not the S2B Gate.
- Failure mode: large initial bearing produces a long-timeout orbit. Failed trajectories can finish with the goal 100–146° behind the robot even though the yaw-command sign remains goal-aligned.
- Decision: **FAIL**. No `S2B_best.pt` was created and S3 is not authorized to start.

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

The first two rows predate the complete V62 curvature-projection alignment and are retained only as historical evidence. The corrected-interface best full-distribution result is 79%, not 86%.

## Important compliance note

The fixed specification requires `v_cmd >= 0` throughout S1–S8. The latest hard-case experiments added reverse-recovery shaping to diagnose the orbiting failure. Those changes are experimental and cannot become the accepted S2B solution. Before formal continuation, the direct action mapping must enforce the forward-only contract and S1/S2/S2B must be revalidated under that exact mapping.

The direct-policy evaluator currently reports success, collision, timeout, terminal distance, initial/final bearing, and command diagnostics. It does not yet emit the complete required `rate_violation`, `feasible_domain_violation`, `hidden_projection_jump`, manager activation, command correction, CSV, and trajectory artifact set. Therefore the direct-navigation Gate is not complete even if SR alone later crosses 90%.

## Obstacle avoidance and maze status

- S0 used corridor walls to validate scripted V62 spatial execution, but this is not learned obstacle avoidance.
- The reusable depth camera, fallback depth model, corridor geometry, and procedural maze infrastructure exist in the repository.
- The current direct-velocity task deliberately runs with `maze.enabled = False` and `scene_mode = "none"` during S1/S2/S2B.
- S3–S8 learned depth-corridor avoidance has not started because S2B is FAIL.
- M1 5×5, M2 9×9, and M3 15×15 direct-SRU maze training have not started. Existing maze/local-depth or actuator-action policies are historical infrastructure/baselines, not results for this new SRU→velocity→V62 route.

## Safety and evaluation protocol

All direct policies route desired `(v,w)` through the existing V62 command projection and transition manager. No direct SRU-to-actuator path was added. Evaluation disables observation/domain randomization and uses a fixed 16-environment protocol; single-env and hard-only results are not mixed with the full-distribution Gate. S0 supplied complete zero-violation evidence; the direct-policy evaluator still needs the missing safety fields described above.

## Current decision and next work

Current milestone: **S2B FAIL / S3 NOT STARTED / Maze NOT STARTED**.

The next accepted implementation should first restore the S1–S8 forward-only action contract, then complete the shared evaluator and rerun S1/S2 regression before attempting a new forward-only S2B curriculum. Only a full S2B PASS can unlock S3 depth-corridor training.
