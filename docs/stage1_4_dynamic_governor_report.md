# Stage1.4 — V49 Dynamic Reachable Governor 报告

## 结论

Stage1.4 的 matched GPU A/B/C 实验已完成：Baseline（Governor OFF/raw request）、Static（现有 static projection）、Dynamic（Stage1.3 table + candidate search + final static projection）三种模式均使用同一 frozen checkpoint、同一 seed、同一初始状态协议和同一 10 场景 request sequence。

结果为 **PARTIAL PASS**：Dynamic 明显降低了“实际选中 command 相对测量速度”的误差和 yaw sign error，但它通过修改上层 command 实现这一点；相对原始导航 request 的 yaw error 几乎没有改善，forward P90 还变差。因此当前 governor 可以作为“可达 command 选择器”使用，不能宣称已经改善原始 request 的 tracking，也不应直接打开为默认行为。

## 实验协议

- task：`rotunbot_vel_sru50_v49_integration`
- checkpoint：`model_300.pt`
- physics/policy：约 200/50 Hz，`decimation=4`
- 每个上层 request 保持 10 个 policy steps（200 ms）
- 10 个固定场景，3 repeats，3 modes，共 90 个 mode-trials、3,060 个 policy samples
- 初始状态：按场景组使用 `.06 m/s`（low）、`.12 m/s`（high）或 `.08 m/s`（mixed），通过 V49 tracking stabilization 建立；没有 root velocity injection
- Dynamic candidate objective：
  `w_v e_v² + w_yaw e_w² + w_delta ||u-u_prev||²`
- Dynamic 保留方向约束、command-rate bounds，并在环境接口再次执行现有 geometric hard projection

## 总体结果

这里的 selected-command error 是 `selected command - measured velocity`；original-request error 是 `raw request - measured velocity`。

| mode | selected v MAE | selected w MAE | original v MAE | original w MAE | yaw sign errors | fallback |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.06678 | 0.05446 | 0.06678 | 0.05446 | 542 | 0 |
| Static | 0.06742 | 0.01529 | 0.06742 | 0.05489 | 316 | 0 |
| Dynamic | **0.02490** | **0.00584** | 0.06618 | 0.05468 | **155** | 0 |

Static projection发生在 930/1,020 个样本上；Dynamic 修改了 1,020/1,020 个 selected commands。没有 fallback、unstable 或 NaN。Dynamic 的 selected-command v MAE 相比 Static 下降约 63%，selected-command w MAE 下降约 62%；但这不是 free tracking gain，而是 command 被 governor 改成了模型预测更可达的值。

## 低速/高速分组

| group | mode | selected v MAE | selected w MAE | original v MAE | original w MAE | yaw sign errors |
|---|---|---:|---:|---:|---:|---:|
| low_speed | Baseline | .04407 | .05857 | .04407 | .05857 | 274 |
| low_speed | Static | .04572 | .00891 | .04572 | .05898 | 137 |
| low_speed | Dynamic | **.03312** | **.00485** | .04154 | .05852 | **96** |
| high_speed | Baseline | .10406 | .06037 | .10406 | .06037 | 127 |
| high_speed | Static | .10389 | .02992 | .10389 | .06064 | 123 |
| high_speed | Dynamic | **.01393** | **.00977** | .10888 | .06051 | **23** |

Dynamic 在 high-speed 组的 selected-v 误差改善最明显，但 original-v MAE 从 Static 的约 `.10389` 上升到约 `.10888`；这正是不能把“选中 command 更容易跟踪”误写成“原始 request 跟踪更好”的原因。

## 安全与行为检查

- 默认配置 `dynamic_governor_enabled=False`；原有 `set_command_targets()` 语义未改变。
- Dynamic 通过显式 `set_governed_command_targets()` 接入，之后仍经过 `project_velocity_commands()`。
- 候选命令不会反向切换当前运动方向；命令变化受显式 v/w step bound 限制。
- 三种 mode 的仿真均无 instability；Dynamic 表外查询会走 static fallback 并写出 coverage/fallback 标记，本次正式 10 场景没有触发表外 fallback。
- 按逐 policy sample 的 signed sequence 计数，selected-yaw oscillations 为 Baseline/Static/Dynamic = 28/20/4；该统计会受 hold sampling 影响。按每个 transition 的 sign-flip 标记，oscillation_count 为 21/15/0，正式结论采用后者。

## 判定与下一步

判定 **PARTIAL PASS**：

1. 作为 state-dependent reachable-command governor：通过。它改善 selected command 的可跟踪性、减少方向错误，并保持仿真稳定。
2. 作为原始导航 request 的 tracking improvement：未通过。original-request w MAE 与 Static 基本相同，original-request v P90 为 Dynamic `.20893`，高于 Static `.17162`。
3. 作为默认控制行为：不通过。保持 opt-in，直到补充更完整的 current-w 状态维度和 planner-level objective，并用 route-level success/overshoot 指标验证 command 修改不会损害导航。

## 产物

运行时产物位于 `logs/stage1_4_dynamic_governor/`：

- `stage1_4_trials.csv`
- `stage1_4_aggregate.json`
- `stage1_4_summary.json`
- `mode_yaw_error_comparison.png`
- `governor_counters.png`
- `low_speed_command_selection.png`
