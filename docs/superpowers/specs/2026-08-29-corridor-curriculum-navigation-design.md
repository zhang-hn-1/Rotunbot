# Corridor Curriculum Navigation Design

来源：用户提供的完整串行规格 `/home/jason/.codex/attachments/d4f71254-2b34-4958-a9f9-c0f20812f163/pasted-text.txt`。

## Goal

在冻结的 V62 速度控制链之上，建立统一的 `local waypoint -> (v_cmd, w_cmd) -> V62 -> actuator` 接口，并按 Gate 串行完成空间底层验证、Velocity Local Goal、Oracle corridor、Single-Frame Depth、SRU Planner 和 Maze 接入。

## Immutable boundaries

- 基线 commit 为 `6aa9f531f991ae101053b1ebe3973cb340daa0d1`。
- 冻结 V62 checkpoint 为外置文件 `/home/jason/Rotunbot_SRU50_V62_SafeYaw_Final_Verified_20260829/model/model_150.pt`，SHA256 为 `d7173fbbb113ab790d25b0587e82a73abd7ffad9ab2ed148387ba04084944f1b`。
- V62 Transition Manager、Dynamic Governor、Feasibility Projector、V62 residual/inverse map 和冻结 checkpoint 在 Phase A 通过后不可由后续训练修改。
- 所有新策略只产生局部目标或 `(v_cmd, w_cmd)`；禁止直接产生 actuator action。
- 旧 actuator-action Local P2P 只保留为历史 baseline，不纳入新主线。
- 每个阶段必须完成代码、单元测试、smoke、正式训练/评估、固定评估集、旧任务回归、报告、checkpoint 元数据、commit 和 PASS/FAIL 决策。
- 前一阶段 Gate 未 PASS 时停止，不启动下一阶段训练。

## Runtime contract

V62 的真实 resolved runtime 为 physics `dt=0.005 s`、control decimation `4`、低层 policy `50 Hz`、上层 command `5 Hz`，即每个上层 request 保持 `10` 个低层 policy steps。所有命令通过 `set_command_targets()`，再经过 Transition Manager（启用任务）、Dynamic Governor、`project_velocity_commands()` 和现有 V62 controller。

新的外层 Local Goal 环境以高层 action `[a_v, a_w]` 产生 command target；每个外层 step 保持 target 一个 5 Hz tick，并在同一个物理环境内运行冻结 V62 actor 的 10 个低层 steps。冻结 V62 actor 只作为执行器，不参与外层 PPO 更新。

## Gate policy

每个评估器同时输出 Current Gate 和 Regression Gate。指标必须包括 success、collision、timeout、divergence、path length、goal distance、raw/applied command、actual velocity、rate violation、feasible-domain violation、hidden projection jump 和 Transition activation。任何安全指标失败都不能用平均 SR 掩盖。

## Stage order

1. Phase 0 baseline audit and shared infrastructure。
2. A0 straight corridor、A1 L corridor、A2 double-turn；全部 PASS 后 Freeze Point 1。
3. B1/B2/B3 Velocity Local Goal；B3 PASS 后形成 Local Goal Controller V1。
4. C1/C2/C3/C4 corridor curriculum、C5.1/C5.2/C5.3/C5.4 width boundary；PASS 后 Freeze Point 2。
5. D Oracle random corridor frozen-stack validation。
6. E1-E5 single-frame depth teacher-student、DAgger 和随机走廊评估。
7. F1-F3 SRU planner sequence imitation、DAgger 和公平对比。
8. G Maze 接入；只在所有前置 Gate PASS 后开始。

