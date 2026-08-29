# V62 Feasible Velocity Transition Manager 验证报告

日期：2026-08-29
分支：`codex/v62-feasible-transition-manager`

## 1. 范围与冻结项

本次只在冻结的 V62 Safe Yaw Residual 控制器和 `model/model_150.pt` 上增加 command-space Transition Manager。没有重新训练 PPO，没有建立 V63，没有修改 residual、inverse map、reward、机械参数或 V62 stable-curvature envelope。

由于 Verified 发布包不是 Git 仓库，工作分支以当前含 navigation/P2P 的 Git 仓库为基底，并导入 Verified V62 的 velocity-tracking 源文件；Verified 发布目录本身未被修改。

模型 SHA-256：`d7173fbbb113ab790d25b0587e82a73abd7ffad9ab2ed148387ba04084944f1b`。

## 2. 原 V62 问题复现

原路径是分别对 `v,w` 做 rate limit 后再做 feasible-domain projection。CPU 复现实验从 `(0.14, 0.035)` 直接切到 `(-0.14, 0.035)` 时，原路径出现 35 个 yaw rate-bound violation，并在第 69 个 step 才到 `v=0`；Transition Manager 经过原点，505 个 50-Hz steps（10.1 s）完成，硬 rate violation 为 0。

## 3. 实现

控制链路变为：

```text
raw request -> project_velocity_commands() -> projected target
             -> FeasibleVelocityTransitionManager
             -> applied_feasible_command -> existing V62 controller
```

Manager 为 GPU batch tensor 状态机，状态为 `TRACK`、`BRAKE_TO_ORIGIN`、`WAIT_SETTLED`、`ACCELERATE_FROM_ORIGIN`。同分支目标继续走 bounded feasible tracking；只有 `source_v * target_v < 0` 且达到已有 reversal threshold 才锁存跨分支制动。

BRAKE 使用 anchor 的径向缩放，WAIT 使用真实测量的 `|v|<=0.01 m/s`、`|w|<=0.005 rad/s` 连续 0.10 s，ACCELERATE 从原点沿最新 projected target 启动。新目标采用 latest-target-wins，不重置已锁存的制动进度。所有新增 runtime state 都在 `reset_idx()` 清零。

原 V62 任务的开关默认仍为 `feasible_transition_manager_enabled=False`；实验任务 `rotunbot_vel_sru50_v62_feasible_transition_manager` 继承 V62 并设为 `True`。

## 4. 单元测试与静态验证

- V62 velocity tracking regression：74 passed。
- Feasible Transition Manager：13 passed，覆盖同分支、固定 yaw 反向、原点路径、latest-target-wins、settle、reset、2048 batch 和无逐环境 Python loop。
- Structured-random evaluator tests：7 passed。
- 合并运行：94 passed。
- 修改源文件和脚本 `py_compile`：通过。
- Registry smoke：V62 baseline manager=False，实验 task manager=True，yaw acceleration 仍为 `0.007 rad/s²`。

## 5. 第一轮固定非零 yaw + v reversal

以下为相同 seed、相同 `model_150.pt`、16 个并行环境的 nominal/standard focused rollout。原 V62 的 16 个违规均来自该固定非零 yaw 反向族；manager 的硬验收计数为 0。

| noise | 版本 | v MAE (m/s) | w MAE (rad/s) | w P95 | hidden jump | rate violation | feasible violation | post-settle rebound |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| nominal | V62 | 0.012843 | 0.004325 | 0.010366 | 16 | 16 | 0 | 4034 |
| nominal | Transition | 0.000060 | 0.000003 | 0.000003 | 0 | 0 | 0 | 0 |
| standard | V62 | 0.012549 | 0.004243 | 0.010131 | 16 | 16 | 0 | 3959 |
| standard | Transition | 0.000644 | 0.000003 | 0.000005 | 0 | 0 | 0 | 0 |

Transition nominal 的全局 mean reversal completion time 为 3.187 s，P95 为 4.386 s；standard 分别为 3.175 s、4.380 s。固定 yaw 族在 8 s 观察窗内仍有部分环境处于完整加速阶段，因此该族自身的 completion count 不作为超时判定；CPU 长窗验证完成时间为 10.1 s。

## 6. 12 类 Structured Random

实验为每个 noise profile、每个 family 16 env、seed `20260829`，使用 50-Hz policy、200-Hz physics；为先完成 focused validation，本轮采用 1.0 s minimum / 2.0 s maximum precondition 和 8.0 s transition window。所有应用参考均通过 defensive projection invariance check，且没有 feasible-domain violation。

| family | nominal V62 v MAE | nominal Transition v MAE | nominal V62 w MAE | nominal Transition w MAE | Transition rate violations |
|---|---:|---:|---:|---:|---:|
| straight_v_reversal | 0.012818 | 0.012882 | 0.000000 | 0.000000 | 0 |
| fixed_w_v_reversal | 0.012843 | 0.000060 | 0.004325 | 0.000003 | 0 |
| constant_curvature_reversal | 0.011673 | 0.000061 | 0.004816 | 0.000003 | 0 |
| fixed_v_w_reversal | 0.000829 | 0.000087 | 0.005682 | 0.000003 | 0 |
| fixed_w_speed_change | 0.004913 | 0.000060 | 0.002544 | 0.000003 | 0 |
| fixed_v_yaw_magnitude_change | 0.002181 | 0.000057 | 0.002894 | 0.000003 | 0 |
| straight_stop_or_restart | 0.006743 | 0.006739 | 0.000004 | 0.000000 | 0 |
| turn_stop_or_restart | 0.006190 | 0.000029 | 0.002877 | 0.000002 | 0 |
| infeasible_low_speed_high_yaw | 0.006817 | 0.000271 | 0.002654 | 0.000002 | 0 |
| boundary_curvature_jump | 0.005906 | 0.000076 | 0.004923 | 0.000003 | 0 |
| all_quadrant_jump | 0.007916 | 0.000067 | 0.004713 | 0.000003 | 0 |
| independent_feasible_jump | 0.010104 | 0.000305 | 0.004193 | 0.000003 | 0 |

全局 nominal Transition：v MAE `0.001724 m/s`、w MAE `2.294e-6 rad/s`、hidden jump `0`、rate violation `0`、feasible-domain violation `0`、completion `15`、timeout `0`、mean completion `3.187 s`、P95 `4.386 s`。

全局 standard Transition：v MAE `0.002209 m/s`、w MAE `1.955e-5 rad/s`、hidden jump `0`、rate violation `0`、feasible-domain violation `0`、completion `15`、timeout `0`、mean completion `3.175 s`、P95 `4.380 s`。

## 7. Frozen Local P2P 闭环

当前项目中查到的、与 Local P2P task 匹配的冻结 checkpoint 为：

`/home/jason/SphericalRobot_LeggedGym-master-sru/SphericalRobot_LeggedGym-master-new-map/logs/rotunbot_local_p2p/Aug24_18-18-41_/model_3051.pt`

使用现有 `evaluate_gate0b_local_waypoint.py`，设置 `GATE0B_TASK=rotunbot_local_p2p`，40 个 episode；地图为 plane、无障碍、关闭摩擦/质量随机化，waypoint 距离 0.5--2.0 m、到达半径 0.35 m、episode 上限 6 s。该 Local P2P policy 直接输出 actuator action，未调用 `RotunbotVel.set_command_targets()`，因此本次 Transition Manager 不会激活；把它强行改成 velocity task 会同时改变 policy 输入/任务，不是隔离的 manager 对照。

实测 frozen P2P：success rate `30.0%`（12/40），timeout rate `70.0%`，divergence rate `40.0%`，near-miss rate `30.0%`，mean episode length `260.6 steps`。由于 manager activation count 为 0，该闭环的 Transition 对照为 N/A；该既有 P2P gate 仍未达到 95% 通过门槛，不能用它宣称 P2P PASS。

## 8. 验收判定

底层 command-transition 目标已满足：nominal 和 standard 下 hidden projection jump、rate-bound violation、feasible-domain violation 均为 0，固定非零 yaw + v reversal 的 yaw collapse/rebound 已消除，且没有看到新增的 transition timeout。

最终判定：**PARTIAL PASS**。

原因不是 Transition Manager 的底层安全指标失败，而是 Frozen Local P2P 使用另一条直接 actuator-action 控制链，无法在本次隔离改动下激活；该既有 P2P checkpoint 的 40-episode gate 也未通过。因此本报告不宣布 `BOTTOM-LEVEL CONTROL FINALIZED`，后续若要重新定义 P2P 对照，必须先明确 velocity-tracking 与 P2P policy 的接口边界，不能在本分支继续混合优化。

## 9. 结果与日志

代码报告：`docs/v62_feasible_transition_manager_report.md`。

主要结果：

- `logs/v62_baseline_nominal/structured_random_summary.json`
- `logs/v62_transition_nominal_final/structured_random_summary.json`
- `logs/v62_baseline_standard/structured_random_summary.json`
- `logs/v62_transition_standard/structured_random_summary.json`
- `logs/frozen_local_p2p_gate0b_correct_task.log`

运行 log：

- `logs/v62_baseline_nominal/run.log`
- `logs/v62_transition_nominal_final.log`
- `logs/v62_baseline_standard.log`
- `logs/v62_transition_standard.log`
- `logs/frozen_local_p2p_gate0b_correct_task.log`

上述 logs/、CSV、NPZ、TensorBoard/cache 均不纳入 Git commit。
