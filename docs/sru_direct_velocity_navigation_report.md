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
- Decision: PASS; allowed to enter S2.

## Stage S2

- Objective: expand goal bearing to ±30°, distance 0.5–1.5 m.
- Training: loaded S1 and continued to 800 total iterations; 30% S1 replay.
- Current evaluation: 97/100 success, 3 timeout, 0 collision.
- S1 regression: 100/100 success, 0 timeout, 0 collision.
- Checkpoint: `logs/sru_velocity/S2/S2_best.pt`.
- SHA256: `38dc54b9cf55ea28203b73b670704db297fa527a0edc2e40e354bfbfa3ca407e`.
- Decision: PASS; allowed to enter S2B.

The first S2 attempt reached only 82–86% because goal-side turn direction was not stable. A causal `goal_turn_alignment` shaping term was added and tested; the final S2 result passed without changing the V62 actuator path.

## Safety and evaluation protocol

All direct policies route desired `(v,w)` through the existing V62 command projection and transition manager. No direct SRU-to-actuator path was added. Evaluation disables observation/domain randomization and uses a fixed 16-environment protocol; single-env results are not mixed with this Gate protocol.

## Next stage

S2B is next: 0.5–2.0 m, ±45°, with 70% S2B, 20% S2, and 10% S1 replay. It must load `S2_best.pt` and pass its current and regression gates before S3 corridor training.
