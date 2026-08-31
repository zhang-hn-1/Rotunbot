# SRU Visual Corridor Curriculum Report

## Architecture contract

The visual curriculum keeps one learned control interface:

```text
Depth + goal_xy_robot + proprioception + previous (v,w) + approved state bit
    -> Depth Encoder -> SRU -> (v_cmd,w_cmd) -> frozen V62 -> actuator
```

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
| Direct/V62/corridor/Oracle tests | 83/83 PASS |
| Depth/visual-observation tests | 26/26 PASS |
| Total current focused audit | 109/109 PASS |

## Stage ledger

| Stage | Parent checkpoint | Current Gate | Depth ablation | Memory ablation | Decision |
| --- | --- | --- | --- | --- | --- |
| V1 Depth Straight Corridor | Pending warm-start probe | Not run | N/A | N/A | NOT STARTED |
| V2 Depth L Corridor | V1_best.pt | Blocked by V1 | N/A | N/A | NOT STARTED |
| V3 Depth Double-Turn | V2_best.pt | Blocked by V2 | N/A | N/A | NOT STARTED |
| V4 Depth S Corridor | V3_best.pt | Blocked by V3 | N/A | N/A | NOT STARTED |
| V5 Narrow Curriculum | V4_best.pt | Blocked by V4 | N/A | N/A | NOT STARTED |
| V6 Random Corridor | V5 composite | Blocked by V5 | N/A | N/A | NOT STARTED |

## Immediate next experiment

Run a 100–200 iteration probe with equal environment, seed, and configuration
for:

1. `logs/sru_velocity/S2/S2_best.pt`
2. `logs/rotunbot_sru_direct_velocity_s2b/Aug30_23-05-08_/model_900.pt`

The probe is only for parent selection. It must report reward/success trend,
collision, command stability, and V62 correction dependence. It is not a V1
formal result. The selected parent will then be used for V1 straight-corridor
training with randomized lateral offset up to ±0.30 m and yaw offset up to
±10°.
