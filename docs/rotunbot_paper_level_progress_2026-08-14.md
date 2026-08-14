# Rotunbot paper-level progress — 2026-08-14

## Accepted checkpoint

- Checkpoint: `logs/rotunbot_target_repro/Aug14_13-11-54_progress_stage1_from3803/model_3806.pt`
- Executor: `DIRECT_VP_TORQUE`
- Policy interface: unchanged 19-D observation frame stack and 2-D action output
- Training continuation: model 3800 -> latency stage model 3803 -> progress-reward model 3806

## Nominal GPU evaluation (3 seeds x 40 episodes)

| Model | Success | SR | SPL | CLS (m) | Balance (%) |
|---|---:|---:|---:|---:|---:|
| model_3800 | 96/120 | 80.00% | 0.5687 | 0.2968 | 94.178 |
| model_3803 | 98/120 | 81.67% | 0.5642 | 0.2838 | 93.848 |
| model_3806 | 98/120 | 81.67% | 0.5750 | 0.2796 | 93.730 |
| Paper, simulation at 60 s | — | 88.87% | 0.6375 | 0.2092 | 75.52 ± 2.32 |

Relative to model_3800, model_3806 gains 1.67 percentage points SR,
0.0063 SPL, and reduces CLS by 0.0173 m. Relative to the paper's 60 s
simulation result, the remaining gaps are 7.20 percentage points SR, 0.0625
SPL, and 0.0704 m CLS. Balance values should not be treated as directly
comparable until the implementations are audited against the paper code.

Per-seed model_3806 nominal results:

| Seed | SR | SPL | CLS (m) | Notable failures |
|---:|---:|---:|---:|---|
| 3 | 87.5% | 0.6117 | 0.2286 | 2 slow timeout, 3 never reached |
| 7 | 80.0% | 0.5308 | 0.2285 | 5 slow timeout, 2 out of bounds, 1 never reached |
| 11 | 77.5% | 0.5826 | 0.3816 | 3 slow timeout, 6 never reached |

## Delay evaluation (seed 3, 40 episodes, random 0–0.1 s)

| Model | SR | SPL | CLS (m) | Instability |
|---|---:|---:|---:|---:|
| model_3800 | 77.5% | 0.4669 | 0.2173 | 0 |
| model_3803 | 80.0% | 0.5125 | 0.2077 | 0 |
| model_3806 | 80.0% | 0.5084 | 0.2837 | 0 |

Model 3806 remains better than the original checkpoint under delay, although
model 3803 is marginally better on delayed SPL/CLS and remains a useful
robustness reference.

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

## Rejected or limited directions

- Identity action adapter: reduced strict success and remains disabled.
- Faster velocity slew rate: reduced success and caused instability; retained
  0.02 velocity and 0.04 position target increments.
- Blind longer continuation from model 3800: checkpoints 3805/3810 degraded.
- Latency-only model 3803 improved SR and delay robustness but reduced aggregate
  nominal SPL, so it was not the final nominal baseline.

## Next highest-value work

The next stage should target the remaining F1/F4 failures without extending PPO
training blindly. Priority order:

1. Add deterministic rough-terrain and combined paper-style validation with
   terrain-correct scenario manifests.
2. Separate heading/curvature efficiency from terminal braking in diagnostics;
   seed 11 F1 failures and seed 7 out-of-bounds failures should be treated as
   different problems.
3. Introduce the paper-style Transformer long-history encoder through staged
   teacher/student distillation from model 3806, retaining the current policy as
   a rollback baseline.
4. Run real-robot calibration/System-ID and 40-trial physical tests. Simulation
   metrics alone cannot establish paper-level real-world performance.
