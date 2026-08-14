# Rotunbot paper-level progress — 2026-08-14

## Accepted checkpoint

- Checkpoint: `logs/rotunbot_target_repro/Aug14_18-56-25_push_recovery_lownoise_from3806/model_3809.pt`
- Executor: `DIRECT_VP_TORQUE`
- Policy interface: unchanged 19-D observation frame stack and 2-D action output
- Training continuation: model 3800 -> latency model 3803 -> progress model 3806 -> low-noise push-recovery model 3809

## Nominal GPU evaluation (3 seeds x 40 episodes)

| Model | Success | SR | SPL | CLS (m) | Balance (%) |
|---|---:|---:|---:|---:|---:|
| model_3800 | 96/120 | 80.00% | 0.5687 | 0.2968 | 94.178 |
| model_3803 | 98/120 | 81.67% | 0.5642 | 0.2838 | 93.848 |
| model_3806 | 98/120 | 81.67% | 0.5750 | 0.2796 | 93.730 |
| model_3809 | 98/120 | 81.67% | 0.5853 | 0.2766 | 93.849 |
| Paper, simulation at 60 s | — | 88.87% | 0.6375 | 0.2092 | 75.52 ± 2.32 |

Relative to model_3800, model_3809 gains 1.67 percentage points SR,
0.0166 SPL, and reduces CLS by 0.0203 m. Relative to the paper's 60 s
simulation result, the remaining gaps are 7.20 percentage points SR, 0.0522
SPL, and 0.0674 m CLS. Balance values should not be treated as directly
comparable until the implementations are audited against the paper code.

Per-seed model_3809 nominal results:

| Seed | SR | SPL | CLS (m) | Notable failures |
|---:|---:|---:|---:|---|
| 3 | 87.5% | 0.6320 | 0.2208 | 4 slow timeout, 1 never reached |
| 7 | 82.5% | 0.5596 | 0.2427 | 5 slow timeout, 1 never reached, 1 instability |
| 11 | 75.0% | 0.5644 | 0.3662 | 5 slow timeout, 5 never reached |

### Current nominal-metric candidate (not yet evaluated)

The 120 nominal episodes above show that the remaining failures are not mainly
long-distance endurance failures.  Targets 4--7 m away are mostly successful,
while targets below 4 m and roughly 45--135 degrees to either side of the
initial heading are the weakest group.  Across all seeds, model 3809 has 14
slow timeouts, 7 never-reached failures, and 1 instability.

- Candidate: `logs/rotunbot_target_repro/Aug14_19-38-12_nominal_axis_alignment_1024_stage1_from3809/model_3813.pt`
- Source: accepted `model_3809.pt`
- Active pushes disabled; mild friction/noise and 0--2 step latency retained.
- Added a small 0.15 reward for aligning the target with either the forward or
  reverse rolling axis.  The absolute alignment preserves reverse approaches.
- Kept the 19-D observation and 2-D direct joint command interface unchanged.
- Reduced GPU parallel environments from 2048 to 1024.  At 2048, Isaac Gym
  warns that aggregate broadphase capacity is insufficient and interactions
  may be missed; the 1024-environment run has no such warning.

The three-update GPU run completed without NaN/CUDA errors.  Its internal
curriculum success-rate log remained about 72.4%.  No new evaluation was run,
as requested, so this checkpoint is a candidate and does not replace accepted
model 3809.

## Delay evaluation (seed 3, 40 episodes, random 0–0.1 s)

| Model | SR | SPL | CLS (m) | Instability |
|---|---:|---:|---:|---:|
| model_3800 | 77.5% | 0.4669 | 0.2173 | 0 |
| model_3803 | 80.0% | 0.5125 | 0.2077 | 0 |
| model_3806 | 80.0% | 0.5084 | 0.2837 | 0 |

Model 3806 remains better than the original checkpoint under delay, although
model 3803 is marginally better on delayed SPL/CLS and remains a useful
robustness reference.

## Flat combined stress evaluation (3 seeds x 40 episodes)

This profile combines friction/base-mass randomization, configured observation
noise, random 0–0.1 s observation/action delay, and 1.0 m/s pushes every 15 s.
It does not yet include rough terrain.

| Model | Success | SR | SPL | CLS (m) | Balance (%) |
|---|---:|---:|---:|---:|---:|
| model_3809 | 107/120 | 89.17% | 0.5028 | 0.1526 | 91.717 |
| Paper, simulation at 60 s | — | 88.87% | 0.6375 | 0.2092 | 75.52 ± 2.32 |

The combined-stress SR now reaches the paper's reported simulation SR, but SPL
remains 0.1347 lower. Pushes can help a stuck robot reach the target, so the
higher SR does not imply equal control quality; the path-efficiency gap remains
the main problem.

## Changes that survived evaluation

1. Explicit velocity/position target tracking through torque, matching the old
   R-controller dynamics while retaining the direct two-joint command interface.
2. Per-episode observation and action latency augmentation, initially staged at
   0–2 control steps (0–0.04 s), with evaluation support through 0–5 steps
   (0–0.1 s).
3. Progress-magnitude shaping: approach reward now scales with actual reduction
   in goal distance, rather than awarding the same value for every positive
   movement. This improved aggregate SPL without changing the network or action
   dimensions.
4. Conservative PPO fine-tuning: three iterations per stage, learning rate
   2e-4, checkpoint every iteration, with paired screening and stop criteria.
5. Low-noise push recovery: frequent 0.3 m/s training pushes improved combined
   robustness only after clamping resumed PPO exploration std from 1.5 to 0.5.

## Rejected or limited directions

- Identity action adapter: reduced strict success and remains disabled.
- Faster velocity slew rate: reduced success and caused instability; retained
  0.02 velocity and 0.04 position target increments.
- Blind longer continuation from model 3800: checkpoints 3805/3810 degraded.
- Latency-only model 3803 improved SR and delay robustness but reduced aggregate
  nominal SPL, so it was not the final nominal baseline.
- Push-recovery model 3809 trained with std=1.5 failed screening (75% SR,
  0.421 SPL on the paired 20-episode combined set) and was rejected. The
  otherwise identical std=0.5 run achieved 95% SR and 0.553 SPL on that set.

## Next highest-value work

The active direction is nominal SR/SPL/CLS improvement.  Priority order:

1. Continue only after the axis-alignment candidate can be compared on the
   fixed nominal manifest; do not stack further reward changes blindly.
2. If side-target failures persist, mix a bounded share of 1--4 m side targets
   into training while retaining the original full-map samples.
3. Tune terminal braking separately from heading/curvature so SR and SPL are
   not traded against one another.
4. Introduce the paper-style Transformer long-history encoder through staged
   teacher/student distillation from model 3806, retaining the current policy as
   a rollback baseline.
5. Run real-robot calibration/System-ID and 40-trial physical tests. Simulation
   metrics alone cannot establish paper-level real-world performance.
