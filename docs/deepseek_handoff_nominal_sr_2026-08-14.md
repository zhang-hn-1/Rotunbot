# DeepSeek handoff: nominal SR/SPL/CLS improvement

## Objective and hard gate

Improve the spherical robot on the fixed nominal point-to-point scenarios using
GPU training.  The acceptance set is seeds 3, 7, and 11, 40 episodes each.

- Required aggregate SR: at least 85% (`>=102/120`).
- Do not accept a candidate that materially reduces SPL or worsens CLS.
- Formal success: distance `<=0.20 m` and linear speed `<=0.10 m/s` within 60 s.
- Automatically reject weak candidates; do not ask the user about runs below
  the gate.

## Accepted rollback baseline

Checkpoint:

`logs/rotunbot_target_repro/Aug14_18-56-25_push_recovery_lownoise_from3806/model_3809.pt`

| Seed | Success | SR | SPL | CLS (m) |
|---:|---:|---:|---:|---:|
| 3 | 35/40 | 87.5% | 0.6320 | 0.2208 |
| 7 | 33/40 | 82.5% | 0.5596 | 0.2427 |
| 11 | 30/40 | 75.0% | 0.5644 | 0.3662 |
| Total | 98/120 | 81.67% | 0.5853 | 0.2766 |

Do not overwrite or delete this checkpoint.  It remains the accepted baseline.

## Architecture and controller

- Actor observation frame: 19 dimensions.
- History: 20 frames; actor recent stack is 5 frames; DWL-CNN long-history
  embedding is 16 dimensions; actor input is 111 dimensions.
- Action output remains 2-D: first-joint velocity command and second-joint
  position command.
- Executor: `DIRECT_VP_TORQUE`, with explicit velocity/position tracking torque.
- Control period: 0.02 s (50 Hz).
- Current training uses 1024 GPU environments.  At 2048 environments Isaac Gym
  reports insufficient aggregate broadphase capacity and possible missed
  interactions.

## Root cause found

The baseline checkpoint stores `training_success_distance=0.40`, while formal
evaluation requires 0.20 m.  Training success stayed near 72%, below the old
curriculum's 80% gate, so it never tightened from 0.40 to 0.20 m.  Current
experimental config therefore sets `target_curriculum=False`, which makes
training use the formal 0.20 m threshold.

Baseline failure distribution over 120 episodes:

- 14 slow timeouts (F4)
- 7 never reached (F1)
- 1 instability (F5)

The weakest seed-11 failures are concentrated around lateral bearings.  Six of
ten failures have initial relative bearings around +/-70 to +/-93 degrees.

## Training reliability fixes currently in the worktree

`legged_gym/dwl/on_policy_runner_dwl.py` supports runner option
`load_optimizer=False`.  This is important: loading a checkpoint previously
restored its old Adam state and learning rate, silently defeating low-learning-
rate experiments.

The conservative fine-tuning settings used in the latest branches are:

```text
learning_rate = 5e-5
schedule = fixed
entropy_coef = 0.002
action std range = 0.15..0.30
fresh optimizer when branching from model_3809
3 PPO updates, checkpoint every update
```

## Completed experiments

### Rejected axis-alignment reward

An absolute forward/reverse axis-alignment reward reduced seed-3 to 31/40,
SPL 0.530, with average path 8.08 m.  It induced unnecessary orientation
adjustments and has been removed from the code.

### Strict 0.20 m, old optimizer/high update branch

Run: `Aug14_19-54-12_nominal_strict020_stage1_from3809`

- model 3810, seed 3: 33/40, SPL 0.5641, CLS 0.2208
- model 3811, seed 3: 34/40, SPL 0.5898, CLS 0.2172
- model 3811, seed 7: 32/40, SPL 0.5155, CLS 0.2459
- model 3812/3813, seed 3: 34/40, SPL 0.5809, CLS 0.2034

Rejected for policy forgetting.

### Strict 0.20 m, fresh optimizer and low learning rate, train seed 3

Run: `Aug14_20-18-27_nominal_strict020_freshopt_lowlr_from3809`

- model 3810, seed 3: 34/40, SPL 0.5943, CLS 0.2202
- model 3811, seed 3: 35/40, SPL 0.6106, CLS 0.2040
- model 3811, seed 7: 32/40, SPL 0.5260, CLS 0.2453
- model 3812, seed 3: 34/40, SPL 0.5879, CLS 0.2261

The seed-3 3811 checkpoint is useful as a branch ingredient, but is not an
accepted multi-seed model.

### Same low-rate branch, train seed 7

Run: `Aug14_20-35-59_nominal_strict020_freshopt_lowlr_seed7_from3809`

- model 3811, eval seed 7: 34/40, SPL 0.5571, CLS 0.2200
- model 3811, eval seed 11: 31/40, SPL 0.5621, CLS 0.3902

Rejected as a standalone model.

### Same low-rate branch, train seed 11

Run: `Aug14_21-10-27_nominal_strict020_freshopt_lowlr_seed11_from3809`

- model 3811, eval seed 11: 30/40, SPL 0.5435, CLS 0.4033
- model 3812, eval seed 11: 31/40, SPL 0.5777, CLS 0.3818

Rejected as a standalone model.

### Three-seed delta average (model soup)

Checkpoint:
`logs/rotunbot_target_repro/multiseed_delta_soup_strict020/model_3811.pt`

Generated with `legged_gym/scripts/average_checkpoints.py` from the three
low-rate model-3811 branches relative to model 3809.

| Seed | Success | SR | SPL | CLS (m) |
|---:|---:|---:|---:|---:|
| 3 | 35/40 | 87.5% | 0.6178 | 0.2116 |
| 7 | 34/40 | 85.0% | 0.5649 | 0.2293 |
| 11 | 31/40 | 77.5% | 0.5677 | 0.3937 |
| Total | 100/120 | 83.33% | 0.5835 | 0.2782 |

This is the best new aggregate SR seen in this work, but it is below the 85%
gate and slightly worse than model 3809 in aggregate SPL/CLS.  Do not accept it.

### 35% hard lateral target sampling

The current environment code mixes 35% random hard lateral targets (distance
1--4 m, relative bearing +/-60 to +/-110 degrees) with 65% original uniform
full-map targets.  No heading reward is used.

Stage-1 run:
`Aug14_21-59-52_nominal_strict020_hardside35_seed11_from3809`

- model 3812, eval seed 11: 31/40, SPL 0.5850, CLS 0.4272

This improved successful-path SPL but did not improve SR enough.

Stage-2 continuation:
`Aug14_22-11-22_nominal_strict020_hardside35_seed11_stage2`

Models 3813--3816 exist.  Evaluation of model 3815 was explicitly interrupted
at the user's request before any complete episode set was written.  Its quality
is unknown.  If work resumes, evaluate model 3815 on seed 11 first; require at
least 33/40 before spending time on seeds 3 and 7.

### Rejected scripted stuck recovery

A deterministic saturated joint pulse was tested after detecting no progress.
It lengthened otherwise successful episodes even when delayed until 20 s.  The
evaluation was stopped early and all stuck-recovery code was removed.  Do not
reintroduce this exact approach.

## Current worktree state

The current config is an experimental continuation config, not the accepted
baseline configuration:

- training seed 11
- strict 0.20 m success (`target_curriculum=False`)
- 35% hard lateral target sampling enabled
- 1024 GPU environments
- push training disabled
- mild friction/noise and 0--2-step latency retained
- fixed learning rate 5e-5, action std max 0.30
- runner resumes stage-1 hard-side model 3812 with optimizer state enabled

There are intentional uncommitted source changes.  User-owned untracked paths
such as `.dsh/` and `deepseek-harness-master.zip` must not be modified.

## Fixed scenario manifests and evaluator

```text
/tmp/rotunbot_paper_gap/scenarios_seed_3.npz
/tmp/rotunbot_paper_gap/scenarios_seed_7.npz
/tmp/rotunbot_paper_gap/scenarios_seed_11.npz
```

Evaluator:
`legged_gym/scripts/evaluate_target_repro.py`

Example GPU worker:

```bash
/usr/bin/env \
  PATH=/home/jason/legged_gym/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  PYTHONPATH=/home/jason/SphericalRobot_LeggedGym-master-new-map:/home/jason/legged_gym/.venv/lib/python3.8/site-packages \
  /home/jason/legged_gym/.venv/bin/python \
  legged_gym/scripts/evaluate_target_repro.py \
  --mode worker \
  --run-dir <run-dir> \
  --output-dir <output-dir> \
  --seed 11 --checkpoint <checkpoint> --episodes 40 \
  --scenario-file /tmp/rotunbot_paper_gap/scenarios_seed_11.npz \
  --phase nominal_40 --perturbation nominal \
  --control-type DIRECT_VP_TORQUE --force
```

## Recommended next decisions

1. First finish only the interrupted stage-2 model-3815 seed-11 evaluation.
   Reject the branch immediately if it is below 33/40.
2. If hard-side continuation fails, stop extending PPO blindly.  The next
   architectural experiment should expose target-relative position directly
   to the actor.  Current observations force the network to subtract global
   target and global current position and combine that with quaternion.
3. Preserve checkpoint compatibility by either gradually blending the first
   target feature from absolute target toward `target-current`, or train a new
   observation adapter/student using model 3809 action distillation.  Do not
   abruptly replace all observation semantics and immediately trust the old
   weights.
4. A state-dependent teacher regularizer is preferable to unrestricted PPO:
   strongly preserve model-3809 actions beyond 1 m and permit adaptation only
   near the target or on lateral hard samples.
5. Keep the final acceptance gate at `>=102/120` and compare aggregate SPL/CLS
   against 0.5853/0.2766.  A single-seed 85% result is not completion.

## Process status at handoff

All training and evaluation processes were stopped.  No GPU worker is running.
The active model-3815 evaluation was interrupted and produced no valid summary.
