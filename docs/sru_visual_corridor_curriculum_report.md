# SRU Visual Corridor Curriculum Report

## Architecture contract

The visual curriculum keeps one learned control interface:

```text
Depth + goal_xy_robot + proprioception + previous (v,w) + approved state bit
    -> Depth Encoder -> single-step SRU block -> (v_cmd,w_cmd) -> frozen V62 -> actuator
```

The current temporal model is stateless: each policy call constructs a
length-one sequence and resets the SRU block's hidden tensor. It is not a
persistent recurrent SRU. The current training backend is fallback depth:
32 horizontal rays replicated across 8 rows, not a calibrated 2D depth image.

`CorridorWaypointOracle` is restricted to geometry diagnostics, upper-bound
reference, and future teacher-data generation. It is not an actor input and it
does not output actuator commands.

## Visual Entry Decision — 2026-08-31

```text
S2B formal status: FAIL
best formal/diagnostic SR: 80/100
VISUAL_ENTRY_GATE: PASS
```

The entry gate uses the retained S0A/S0B/S0C summaries, S1/S2 evaluation
artifacts, the complete S2B regression summary, and the current focused audit.
The gate passed because S2B is above the 75% research-entry floor and all
required safety/interface checks are present and zero where required. This
does not weaken the S2B formal threshold of 90%.

Audit evidence:

| Area | Result |
| --- | --- |
| V62 S0A/S0B/S0C | PASS; 20/20, 20/20, 30/30; safety counters zero |
| S1/S2 retained regression | 100/100 and 97/100 |
| S2B historical best | 80/100; formal FAIL at 90% |
| V1 warm-start probe A | historical 100-iteration probe; 18/20 short evaluation |
| V1 warm-start probe B | 100 iterations from historical S2B; 16/20 short evaluation |
| Direct/V62/corridor/Oracle tests | 83/83 PASS |
| Depth/visual-observation tests | 26/26 PASS |
| P0/P1 focused unit audit | 115/115 PASS |

## Stage ledger

| Stage | Parent checkpoint | Current Gate | Depth ablation | Memory ablation | Decision |
| --- | --- | --- | --- | --- | --- |
| V1 Depth Straight Corridor | 5 Hz-aligned S2 adaptation `model_100.pt` | S1/S2 regression passed; V1 formal evaluation pending | N/A | N/A | TRAINING IN PROGRESS |
| V2 Depth L Corridor | V1_best.pt | Blocked by V1 | N/A | N/A | NOT STARTED |
| V3 Depth Double-Turn | V2_best.pt | Blocked by V2 | N/A | N/A | NOT STARTED |
| V4 Depth S Corridor | V3_best.pt | Blocked by V3 | N/A | N/A | NOT STARTED |
| V5 Narrow Curriculum | V4_best.pt | Blocked by V4 | N/A | N/A | NOT STARTED |
| V6 Random Corridor | V5 composite | Blocked by V5 | N/A | N/A | NOT STARTED |

## Warm-start probe decision

Both candidates used the same 100-iteration S2B configuration, seed 4, 64
training environments, and model-only warm start:

| Candidate | Parent | Training safety | Fixed 20-episode probe |
| --- | --- | --- | --- |
| A | `logs/sru_velocity/S2/S2_best.pt` | collision/violation counts zero | 18/20; 2 timeout; no reverse command |
| B | `logs/rotunbot_sru_direct_velocity_s2b/Aug30_23-05-08_/model_900.pt` | collision/violation counts zero | 16/20; 4 timeout; 290 raw reverse commands |

The corrected 5 Hz-aligned S2 adaptation is now the V1 parent:
`logs/rotunbot_sru_direct_velocity_s2/Aug31_16-38-22_/model_100.pt`,
SHA256 `bddb58bc66e6f73371cca87ca941920844abce2c39b775edaa7e4c8887483e03`.
It was trained for 100 iterations from `logs/sru_velocity/S2/S2_best.pt`,
with model-only warm start and a fresh optimizer. Fixed 100-episode regression
is 100/100 on S1 and 99/100 on S2 (one timeout), with zero collision and zero
rate/domain/hidden-projection violations. These are regression results, not a
V1 result.

## V1 execution status

The V1 environment smoke completed with four parallel environments and a
finite 273-value observation through the direct `(v_cmd,w_cmd)` interface.
An initial 1500-iteration run was stopped at iteration 111: the inherited
policy produced near-zero forward motion and no success evidence, although
collision and safety counters remained zero. A targeted policy probe isolated
the issue to distance extrapolation: the selected S2 parent produced forward
commands around a 2 m goal but reverse commands at a 6 m goal. This is a
transfer-scale issue, not evidence that the V62 command chain is broken.

The V1 training entry point now enables the performance-gated curriculum:
70% uniform replay over 2.0--current-max m and 30% frontier sampling in the
last 0.25 m. The promotion levels are 2.5, 3, 4, 5, and 6 m; promotion
requires two consecutive 30-episode passes after at least 50 iterations,
with frontier >=26/30, replay >=27/30, collision <=1/30, and all safety
counters zero. Curriculum state is stored in the PPO environment checkpoint
and restored independently of model-only warm starts. Formal evaluation keeps
the curriculum disabled and the goal fixed at 6 m. The V1 Gate remains
`PENDING` until a converged checkpoint achieves the 100-episode fixed 6 m
evaluation and the same-checkpoint Depth ablation.

## P0 timing correction

The repository's actual V62-derived timing is `sim.dt=0.005 s`, control
decimation `4`, primitive env step `0.020 s` (50 Hz), and direct-velocity
command frequency `5 Hz`. The timing layer derives `repeat=10`, so PPO stores
one macro transition for ten held primitive steps. Reward aggregation uses
`sum(gamma_p**j * r_j)`, with `gamma_macro=gamma_p**10` and
`lambda_macro=lambda_p**10`. The deterministic timing artifact records 20
primitive rows for two policy samples, ten rows per sample; no action is
resampled inside a macro transition.

The original 6 m failure and the clipped-goal causal result remain diagnostic
evidence only; the corrected 5 Hz-aligned S2 parent has now passed S1/S2
regression, so V1 training may proceed from that parent.

## P0-A deterministic distance diagnostics

Using the selected S2 parent `model_100.pt`, fixed centered pose, zero
velocity, zero previous command, fixed fallback depth, and deterministic
`act_inference`, the 0.50--6.00 m scan (0.25 m increments) found the first
raw `a_v` and mapped `v_cmd` zero crossing at approximately `3.911 m`.
At 6.00 m the normal actor input produced `raw a_v=-0.305746` and
`mapped v_cmd=-0.076437 m/s`; at 2.00 m it produced `raw a_v=+0.305011` and
`mapped v_cmd=+0.076253 m/s`. The complete rows and plot are retained in
`logs/diagnostics/v1_distance_action_scan.csv` and
`logs/diagnostics/v1_distance_action_scan.png`.

In the observation-only causal control, the physical goal stayed at 6.00 m.
Normal visible 6.00 m gave `a_v=-0.305746`, `v_cmd=-0.076437 m/s`; temporary
visible 2.00 m clipping gave `a_v=+0.305011`, `v_cmd=+0.076253 m/s`. This
supports distance-observation OOD as a direct trigger of sign reversal, but
it is not a permanent clipping solution.
