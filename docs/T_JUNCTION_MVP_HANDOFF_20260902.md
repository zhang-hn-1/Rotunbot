# Rotunbot V1 T-junction MVP 交接文档

更新时间：2026-09-02
仓库：`https://github.com/zhang-hn-1/Rotunbot`
工作分支：`codex/corridor-curriculum-navigation`
当前 HEAD：`e0e6699`

## 当前结论

本轮没有完成整个 T-junction MVP，不能写 `T_JUNCTION_MVP_PASS`。

已完成并通过代码审查的部分：

- T_LEFT/T_RIGHT 对称 T-junction 几何；
- 与旧 corridor consumer 隔离的 `corridor_explicit_wall_segments` 显式墙段模式；
- `(center, half_extent)` AABB 契约；
- 严格 95% / 零碰撞 / 零 wrong-turn 的 T gate 逻辑；
- paired counterfactual 的 side、metadata、唯一 ID、完整覆盖校验；
- T teacher collector、真实 IsaacGym backend fail-closed 断言；
- T=16 V1 teacher dataset audit：字段长度/shape、finite、done、`num_sequences`、递增 episode ID、episode provenance；
- active local waypoint 通过 `set_observation_goal_world` 进入 observation，global goal 仍只负责 terminal；
- wrong-branch 在每个 primitive step 检查，避免 macro interval blind spot。

尚未完成：

- 未运行真实 20+20 T Teacher Gate；
- 未生成正式 T dataset / audit JSON；
- 未训练 Straight:L:T = 1:3:5 mixed BC；
- 未实现/验证 T Student、paired 20+ counterfactual、normal/zero/swapped ablation；
- 未做 Straight/L regression；
- 未写最终 `T_JUNCTION_MVP_REPORT_20260902.md`；
- 未运行完整 pytest regression；
- 未 push 本轮新 commit。

## 已验证的代码提交

按时间顺序：

| Commit | 内容 | 状态 |
|---|---|---|
| `380ef1a` | T-junction 设计与实施计划 | 已提交 |
| `f944559` | 初版 T geometry / metrics / tests | 后续被 review 指出问题 |
| `36dfd6a` | AABB、严格 gate、pair/backend 初步加固 | 后续仍有拓扑/pair 问题 |
| `b8aae34` | 显式墙拓扑与向后兼容 consumer | Task 1 scoped review PASS |
| `3508c7e` | T teacher collector 与 dataset audit 初版 | 后续 audit/terminal 问题已修复 |
| `8105f60` | teacher terminal evidence、observation goal、provenance 初步加固 | 后续仍有 audit/primitive coverage 问题 |
| `e0e6699` | `num_sequences`、唯一 episode ID、primitive wrong-branch latch | Task 2 scoped review PASS |

Task 1/Task 2 的详细实施计划和审查记录位于：

- `docs/superpowers/specs/2026-09-02-t-junction-mvp-design.md`
- `docs/superpowers/plans/2026-09-02-t-junction-mvp.md`
- `.superpowers/sdd/2026-09-02-t-junction-mvp/progress.md`
- `.superpowers/sdd/2026-09-02-t-junction-mvp/task-1-round2-review-report.md`
- `.superpowers/sdd/2026-09-02-t-junction-mvp/task-2-round2-review-report.md`

## 关键代码入口

- `legged_gym/navigation/v1_t_junction.py`
  - `build_t_junction_geometry("T_LEFT"|"T_RIGHT")`
  - `wall_actor_centers(...)`
- `legged_gym/navigation/v1_t_junction_metrics.py`
  - `aggregate_t_gate(records, pairs, ablations)`
- `legged_gym/navigation/v62_corridor_task.py`
  - `corridor_explicit_wall_segments` 是新增可选模式；旧 `corridor_wall_segments` 行为未改变。
- `legged_gym/scripts/collect_sru_visual_t_junction_teacher.py`
  - collector CLI；默认 20 episodes/side，默认 2250 primitive steps。
- `legged_gym/scripts/audit_t_junction_teacher_dataset.py`
  - dataset audit CLI。
- `legged_gym/navigation/v1_teacher_dataset.py`
  - 现有 V1 schema / `TeacherSequenceWriter`，不要重设计 schema。
- `legged_gym/scripts/eval_sru_visual_l_turn.py`
  - 后续 T student evaluator 应复用的 recurrent rollout / Frozen V62 参考实现。
- `legged_gym/scripts/train_sru_visual_l_turn_imitation.py`
  - 后续 mixed BC trainer 应复用的 warm-start / imitation 参考实现。

## 下一 AI 必须做什么

停止范围外扩展，按以下顺序继续：

1. 读取本文件、T 设计/spec/plan 和 `progress.md`，确认当前 HEAD 为 `e0e6699`。
2. 先为 Task 3 写 failing tests：
   - Straight/L/T 采样比例 1:3:5；
   - T student episode record；
   - 20+ paired counterfactual 的唯一/完整覆盖；
   - normal/zero/swapped goal mode；
   - student gate aggregation。
3. 新增 `train_sru_visual_t_junction_imitation.py`：
   - 输入 Straight、L、T 三个真实 depth dataset；
   - `load_teacher_dataset` 后断言 metadata backend 为 `isaacgym` 且 T=16；
   - 用整数重采样实现 Straight:L:T = 1:3:5；
   - warm-start 当前已通过的 `logs/phase_c/v1_imitation_straight_l_balanced_20260901.pt`；
   - 保存 train/validation/masked-Huber、v MAE、w MAE、epoch/best epoch、sample distribution。
4. 新增 `eval_sru_visual_t_junction.py`：
   - 只用真实 IsaacGym depth，并严格断言 `env.depth_backend_actual == "isaacgym"`；
   - 设置 `corridor_explicit_wall_segments=geometry.wall_segments`，清空旧墙段；
   - 复用 275-wide recurrent ABI、T=16 temporal behavior、done hidden reset、Frozen V62；
   - 每个 macro decision 安装 active waypoint 并重新 `compute_observations()`；
   - terminal success 仍使用 global goal；
   - 记录 scenario/seed/goal/initial pose/yaw/horizon/success/collision/timeout/wrong_turn/turn completion/exit/failure trace；
   - 生成完整 pairs 和 `goal_consistency_rate`；
   - 同一 student 做 normal、zero、swapped goal ablation，不训练新模型。
5. 先运行 unit/CPU dataset audit，再按独立 IsaacGym 进程执行：
   - 20+20 Teacher Gate；
   - 正式 T dataset 与 audit；
   - mixed BC；
   - T Student 20+20；
   - paired counterfactual 至少 20 pairs；
   - Straight 20、L_LEFT 20、L_RIGHT 20 regression。
6. 只有所有非 ROS 测试和 gate 都有真实数值后，才写 `logs/phase_c/T_JUNCTION_MVP_REPORT_20260902.md`，最终 verdict 只能是 `T_JUNCTION_MVP_PASS` 或 `T_JUNCTION_MVP_FAIL`。

## 已知验证环境

必须使用：

```bash
PATH=/home/jason/legged_gym/.venv/bin:$PATH \
PYTHONPATH=/home/jason/.codex/worktrees/codex-corridor-curriculum-navigation \
python ...
```

此前子任务在错误的 system Python 下报告过缺少 `isaacgym`/`torch`；不能据此替代真实 gate。若 venv 仍失败，记录完整错误，不要 fake depth、ray fallback 或伪造 PASS。

已知 ROS `legged_gym/tests/test_nav.py` 因缺少 `rospy` 可以继续作为唯一环境 skip；不要为了它安装 ROS。其他任何 pytest failure 都阻止 PASS。

## Git 注意事项

- 当前分支比 origin 多本轮本地提交；本轮停止前没有执行 push。
- `logs/` 下已有大量用户实验目录，均为未跟踪内容；不要 `git add logs`，不要删除或 reset。
- 新增的交接文档本身需要提交；Task 3 不要覆盖本文件。
- 若最终仍失败，保留 checkpoint、trace、evaluation JSON，并明确失败归因，不要删除证据。
