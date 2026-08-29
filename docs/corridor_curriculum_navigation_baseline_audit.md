# Corridor Curriculum Navigation Phase 0 Baseline Audit

日期：2026-08-29  
分支：`codex/corridor-curriculum-navigation`  
基线：`6aa9f531f991ae101053b1ebe3973cb340daa0d1`

## Identity

- 新 worktree：`/home/jason/.codex/worktrees/codex-corridor-curriculum-navigation`
- Git worktree 在创建时 clean；Phase 0 文档随后才产生未提交变更。
- 冻结 checkpoint 不在 Git 分支内，实际文件为：
  `/home/jason/Rotunbot_SRU50_V62_SafeYaw_Final_Verified_20260829/model/model_150.pt`
- checkpoint SHA256：`d7173fbbb113ab790d25b0587e82a73abd7ffad9ab2ed148387ba04084944f1b`

## Resolved V62 runtime

从 `RotunbotVelSRU50SafeYawResidualV62Cfg` 和 `RotunbotVelSRU50SafeYawResidualV62TransitionCfg` 实际解析得到：

| Parameter | Value |
|---|---:|
| physics dt | `0.005 s` |
| control decimation | `4` |
| low-level policy dt | `0.020 s` |
| low-level policy frequency | `50 Hz` |
| upper command frequency | `5 Hz` |
| low-level steps per upper command | `10` |
| maximum linear acceleration | `0.10 m/s²` |
| maximum yaw acceleration | `0.007 rad/s²` |
| max forward speed | `0.25 m/s` |
| max yaw rate | `0.10 rad/s` |
| minimum turn radius | `2.0 m` |
| feasible envelope fraction | `1.0` |
| manager in baseline task | `False` |
| manager in transition task | `True` |

The new corridor evaluator must use these values through the existing V62 config and `project_velocity_commands()`. It must not introduce a second command envelope.

## Existing regression

Command:

```bash
PATH=/home/jason/legged_gym/.venv/bin:$PATH \
/home/jason/legged_gym/.venv/bin/python -m unittest -v \
legged_gym.tests.test_rotunbot_velocity_tracking \
legged_gym.tests.test_feasible_transition_manager \
legged_gym.tests.test_vel_sru50_structured_random
```

Result:

```text
Ran 94 tests in 0.183s
OK
```

The first attempt using `pytest` was not a code failure: the requested venv has no pytest. The first `unittest` attempt failed before test collection because the venv `ninja` directory was not on `PATH`; adding `/home/jason/legged_gym/.venv/bin` allowed Isaac Gym's cached `gymtorch` extension to load and the full 94-test regression passed.

## Control path confirmed

The resolved transition task uses:

```text
upper request
  -> set_command_targets()
  -> project_velocity_commands()
  -> FeasibleVelocityTransitionManager.update_target()
  -> FeasibleVelocityTransitionManager.advance() at every 50 Hz step
  -> existing V62 governor/projector/controller
```

An upper 5 Hz request is held for exactly 10 low-level policy steps. Phase A scripted commands must enter through `set_command_targets()` and must not mutate `env.commands` as a bypass.

## Phase 0 decision

**PASS** — branch isolation, checkpoint identity, resolved runtime and existing 94-test regression are verified. The next permitted work is shared corridor infrastructure and Stage A0; no B1 training may start before A0/A1/A2 all pass.

