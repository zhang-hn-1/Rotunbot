# Stage1.3 — V49 动态可达性测量报告

## 结论

Stage1.3 GPU 正式扫测完成，当前 frozen V49 integration task 在空地图上没有出现初始化失败或仿真不稳定。数据支持把动态响应建模为 `current_v + projected_v + projected_w` 的表格模型，并保留 projected forward command 作为输入；但本轮实验没有验证 `current_w` 的独立影响，因此模型暂不对 current yaw-rate 做插值。

本轮没有复现 Stage1.2 中“低速约 ±0.02 rad/s 后突然恢复”的清晰 cliff。原因是当前 integration task 的静态可行域先把最大 yaw command 限制到约 `±0.0351 rad/s`（在 `|v|=0.13 m/s` 时），而且本轮初始 yaw-rate 固定为 0。这个结果说明 Stage1.2 的现象不能直接当作当前 task 的完整动态 plant envelope；Stage1.4 必须把静态投影、动态 table 和 fallback 分开记录。

## 实验配置

- task：`rotunbot_vel_sru50_v49_integration`
- frozen policy：`model_300.pt`
- 初始 forward velocity request：`0, .02, .04, .06, .08, .10, .12, .14 m/s`
- 初始 yaw-rate request：`0 rad/s`
- forward command：9 个对称速度层，覆盖运行时 `±0.13 m/s`
- yaw command：`-0.10:0.02:0.10 rad/s`，随后经过现有 static projector
- 每个 raw request 至少 3 次重复；共 2,376 个 transition、23,760 个 50 Hz 样本
- physics：约 200 Hz，`dt=0.005 s`
- policy：约 50 Hz，`dt=0.02 s`，decimation=4
- 每个 transition：恰好 10 个 policy steps，即 200 ms

## 数据质量与覆盖

| 检查项 | 结果 |
|---|---:|
| transitions | 2,376 |
| completed transitions | 2,376 |
| initial-state-not-stabilized | 0 |
| simulation instability | 0 |
| compact projected-state cells | 200 |
| repeats per compact cell | 3、12 或 33（静态投影合并 raw alias 后仍不少于 3） |
| horizon fields | 50/100/150/200 ms |

`dynamic_reachability_table.csv` 的 current-v 轴为实验设计层 `0...0.14 m/s`，command 轴为经过 static projection 的 `(projected_v, projected_w)`。由于可行域是非矩形的，低速层只覆盖较小的 yaw 范围；模型在表外查询会标记 `out_of_coverage=True`，不会静默外推。

## 观测结果

运行时最大 yaw command 经静态 projection 后为约 `±0.0351 rad/s`。非零 projected yaw 在 200 ms 的绝对实际 yaw-rate 中位数随初始速度从约 `0.00016 rad/s`（`v0=0`）增加到约 `0.00119 rad/s`（`v0=.10`），在 `.12/.14 m/s` 分别约 `0.00096/0.00094 rad/s`。因此本轮测量显示的是平滑、很小的响应增强，而不是能够支撑一个固定阈值的二段式恢复。

正负 yaw response 已按 signed command 保留；当前数据存在方向相关的 asymmetry，不能把负 yaw 镜像成正 yaw。forward command 也必须保留：它改变 static yaw limit，并且在同一初始速度层内改变 measured 200 ms response。后续若要删除 projected-v 维度，必须先做固定 current-v、固定 projected-w 的 paired ablation，本轮没有这个证据。

## Stage1.4 使用约束

1. 先对候选 command 应用现有 `project_velocity_commands()`，再查询动态表。
2. 表外/稀疏区域必须走确定性的 static fallback，并写出 `coverage` 与 `fallback` 标记。
3. governor 只能修改上层 `[v,w]` command，不能修改 policy、action mapping、reward、gains 或物理参数。
4. 默认开关保持关闭；Baseline、Static、Dynamic 三种模式必须使用相同 seed、初始状态和 request sequence。
5. 本轮 table 不是 current-w 完整状态模型；Stage1.4 报告必须明确这一限制。

## 产物

运行时产物位于 `logs/stage1_3_dynamic_reachability/`：

- `dynamic_response_raw.csv`
- `dynamic_response_trials.csv`
- `dynamic_response_aggregated.csv`
- `dynamic_reachability_table.csv`
- `stage1_3_summary.json`
- `command_coverage_heatmap.png`
- `command_to_response_curves.png`
- `yaw_response_envelope.png`

Stage1.3 状态：**ACCEPTED WITH SCOPE LIMITATION**。表格模型和数据管线可供 Stage1.4 使用；current-w 独立状态效应和更高 yaw request 的动态 response 仍需后续实验验证。
