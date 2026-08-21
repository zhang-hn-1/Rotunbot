# 变更报告：相对"一开始的文档"项目发生了多少变化

- 撰写时间：2026-08-16
- 报告基准：`HEAD = 0f6bde7`（16 个提交），工作区干净
- 三个对照基准点：
  1. 初始版本 `SphericalRobot_LeggedGym-master`（`CHANGELOG.md` 撰写于 2026-08-11）
  2. 会话开始时的文档状态（`docs/deepseek_handoff_nominal_sr_2026-08-14.md` +
     `docs/rotunbot_paper_level_progress_2026-08-14.md` 截至提交 `22d3415`，2026-08-15 18:51）
  3. 当前状态（2026-08-16 04:43 之后）

---

## 1. 总览：变化有多大

| 维度 | 会话开始时（22d3415） | 当前（0f6bde7） | 变化 |
|---|---|---|---|
| 提交数 | 9 | **16** | +7 个新提交 |
| 已接受模型 | model_3809 + 执行器增益 (100,600) | **uniform 4150**（新协议 SOTA） | 换了训练范式 |
| 目标采样 | 35% 硬侧向 + 65% 均匀 | **纯均匀**（论文协议） | 训练分布对齐评估分布 |
| 评估协议 | 固定清单（balance 用 y/z 轴） | **论文对齐**（均匀清单 + 随机朝向 + balance 改 x/y 轴） | 协议重新基线 |
| SOTA 指标 | SR 90.0%, SPL 0.5178, CLS 0.1909 | SR 88.33%, SPL 0.4984, CLS 0.1813, balance 81.8% | 新协议下重训提升 |
| 文档状态 | 2 份文档相对最新 | progress 文档补了 balance 勘误；**uniform 4150 阶段无文档**（本次报告补上） | 存在文档缺口 |

> 注意：新旧两组指标不在同一评估协议下，不能直接横向比较。旧数字在"固定清单 + 旧 balance 轴"下测得；
> 新数字在"论文对齐协议"下测得（见第 4 节）。

---

## 2. 相对初始版本 `SphericalRobot_LeggedGym-master` 的变化（CHANGELOG 已记录，未再变）

`CHANGELOG.md`（2026-08-11）记录的第一轮改造，之后没有再改动：

- 新增 2 个文件：`rotunbot_target_repro.py`（论文复现环境）、`rotunbot_target_repro_config.py`（配置）
- 修改 5 个文件：`actor_critic_dwl.py`（探索噪声钳制 0.2/1.5）、`on_policy_runner_dwl.py`（checkpoint 保存恢复）、
  `envs/__init__.py`（任务注册）、`rotunbot_target_lh.py`（速率限制逻辑修复）、`play.py`（自动评估）
- 移除：深度相机与视觉导航相关内容

这一层与"一开始的文档"一致，**没有变化**。

---

## 3. 相对会话开始时文档的变化：7 个新提交逐条

会话开始时 HEAD 停在 `22d3415`（"docs: final accepted baseline summary"）。之后仓库新增了 7 个提交，
全部发生在 2026-08-15 23:14 至 08-16 04:43 之间，构成一个新阶段：**"论文协议对齐 + uniform 重训链"**。

| 提交 | 时间 | 内容 | 结果 |
|---|---|---|---|
| `ad215b4` | 08-15 23:14 | 评估协议按论文对齐（均匀随机目标清单、随机初始朝向、balance 改为机体 roll/pitch 角速度 x/y 轴）；**修复 `direct_velocity_gain_randomize` 泄漏到评估**的问题（`_configure` 强制关闭）；新增 `balance_report.py`、`make_sim_effect.py`；3809 在 (100,600) 下做均匀分布重训 | uniform 3820：SR 106/120=88.33%、CLS 0.1843、balance 81.84%、SPL 0.4660 |
| `c2acd11` | 08-15 23:52 | 时间惩罚 `time: -0.5 → -1.0`，从 3820 继续训 | **ckpt 3830 = 新 SOTA**：SR 88.33%、SPL 0.4907（+0.025）、CLS 0.1820、balance 81.41% |
| `76cbce0` | 08-16 00:47 | 记录 uniform 基线上的否决：time-1.5 / detour / **brake05**（近距刹车距离 0.5） | 均被否决，SOTA 仍为 3830 |
| `56f078e` | 08-16 02:40 | 记录论文协议调参否决：**sig15**（tracking_sigma 1.5）、brake 恢复 0.20、从 3809 开新分支 | SOTA 仍为 uniform 3830 |
| `b737524` | 08-16 03:32 | **500 步长训**（uniform + time-1.0，`max_iterations=500`，从 3809 分支） | **ckpt 4150 = 新 SOTA** |
| `fd2e4f6` | 08-16 04:25 | 记录 rate-limit 改动否决；尝试 1000 步长训无提升 | SOTA = uniform 4150 确认 |
| `0f6bde7` | 08-16 04:43 | **body-frame 相对目标观测被否决**（`target_relative_blend=0.0`）；定稿 `ACCEPTED_uniform_4150` 配置 | 当前 HEAD |

### 3.1 新阶段"uniform 4150"是什么

- **训练分布 = 评估分布**：放弃 35% 硬侧向采样（`hard_side_target_probability: 0.35 → 0.0`），训练时目标
  就是论文的均匀全图随机分布——之前所有"21 个重训方向全崩"的结论，是在训练/评估分布不一致的协议下得出的；
  分布对齐后，重训第一次系统性地提升了 SPL。
- **时间惩罚链**：`time: -0.5 → -1.0`，直接给 SPL 训练信号，SPL 从 0.4660 → 0.4907 → 0.4984 阶梯上升。
- **长程续训**：从 3809 分支连续训 500 步（每 50 步存一次），4150 为峰值，继续到 1000 步无提升（确认 4150 为 SOTA）。
- **蒸馏锚定**：teacher 换成了 SOTA 3830（`distill_weight=0.5`、远距全锚定、近距 0.3 放松），防止微调漂移。
- **balance 奖励权重**：`0.1 → 0.4`（对齐论文 Table II）。

### 3.2 代码改动（相对 22d3415）

| 文件 | 改动 |
|---|---|
| `rotunbot_target_repro.py` | 目标观测支持 body-frame 相对目标（`target_body_frame`，最终 blend=0 停用）；balance 奖励与终端缓存改为机体 x/y 轴角速度 |
| `rotunbot_target_lh.py` | 增益随机化读取逻辑（配合 leak 修复） |
| `rotunbot_target_repro_config.py` | hardside 0.35→0.0、balance 0.1→0.4、time −0.5→−1.0、`target_relative_blend` 0.3→0.0、teacher→3830、`max_iterations` 30→1000、`save_interval` 5→50、`run_name=ACCEPTED_uniform_4150`、resume checkpoint 4150 |
| `evaluate_target_repro.py` | 论文协议清单（均匀 + 随机朝向）；balance 轴修正；`_configure` 强制关闭增益随机化；trace 额外保存机体角速度 |
| `play.py` | balance 指标轴修正 |
| **新增** `balance_report.py` | balance 指标对比报告（3809 基准 / g100 / 3816 等，含图表） |
| **新增** `make_sim_effect.py` | 把 3809@(100,600) 评估轨迹重放成图（40 条轨迹总览、基线对比、GIF/MP4 动画） |

---

## 4. 指标变化（新协议）

### 4.1 评估协议变更（ad215b4）

- 目标清单：改为论文式**均匀随机目标**（原先含训练混合分布痕迹）
- 初始朝向：随机 yaw
- **balance 指标勘误**：改为机体 roll/pitch（x/y 轴）角速度 `exp(-(ωx²+ωy²))`；此前所有 balance 数字
  （3809 名义 93.85、v100/p600 ~85.7）用的是 y/z 轴，与论文不可比（论文 75.52 ± 2.32）
- 修复：训练用的增益随机化曾泄漏进评估，`_configure` 现在强制关闭

### 4.2 同一模型在新协议下的重测

| 模型 | SR | SPL | CLS | Balance |
|---|---:|---:|---:|---:|
| 3809 @ (100,600)，旧协议（08-15 批准） | 90.0% (108/120) | 0.5178 | 0.1909 | ~85.7（旧轴，不可比） |
| 3809 @ (100,600)，新协议重测 | ≈83–84% | ≈0.50 | ≈0.25 | 新轴（见 4.3） |

### 4.3 uniform 重训链（全部为新协议）

| 模型 | SR | SPL | CLS (m) | Balance (%) |
|---|---:|---:|---:|---:|
| uniform 3820（ad215b4） | 88.33% (106/120) | 0.4660 | 0.1843 | 81.84 |
| uniform+time1 3830（c2acd11） | 88.33% (106/120) | 0.4907 | 0.1820 | 81.41 |
| **uniform+time1 long500 4150（接受）** | **88.33% (106/120)** | **0.4984** | **0.1813** | ≈81.8 |
| 论文（仿真 60 s） | 88.87% | 0.6375 | 0.2092 | 75.52 ± 2.32 |

uniform 4150 分种子：seed 3 = 92.5%（SPL 0.4985 / CLS 0.1844）、seed 7 = 80.0%（0.4603 / 0.2042）、
seed 11 = 92.5%（0.5364 / 0.1553）。

**结论**：SR 与论文基本持平（差 0.54pp）、CLS 与 balance 均超过论文，SPL 0.4984 仍是主要缺口（差 0.14）。

### 4.4 新阶段否决记录（uniform 基线上）

- time-1.5（时间惩罚过大）
- detour 惩罚
- brake05（近距刹车距离 0.50）
- sig15（tracking_sigma_main 1.5）
- rate-limit 改动
- body-frame 相对目标观测（`target_relative_blend` 路径，最终置 0）
- 1000 步长训（相对 500 步无提升）

---

## 5. 文档自身状态（重要）

| 文档 | 最后更新 | 状态 |
|---|---|---|
| `CHANGELOG.md` | 2026-08-11 | **过时**：只覆盖初始版本→08-11 的 7 文件改动 |
| `docs/deepseek_handoff_nominal_sr_2026-08-14.md` | 2026-08-14 | **过时**：描述的 worktree 状态、下一步（3815 评估、硬侧向续训）已被后续提交取代 |
| `docs/rotunbot_paper_level_progress_2026-08-14.md` | 2026-08-15（ad215b4 补 balance 勘误） | 部分过时：正文停留在 3809/3813 时代，补了 balance 勘误，**但没有 uniform 4150 阶段** |
| 本报告 `docs/change_report_vs_initial_2026-08-16.md` | 2026-08-16 | 补上 uniform 4150 阶段的文档缺口 |

---

## 6. 当前状态速览（HEAD = 0f6bde7）

- **接受配置**：`run_name = ACCEPTED_uniform_4150`，从 `Aug16_02-57-06_uniform_t1_long500_from3809/model_4150.pt` 续训
- **环境**：1024 环境、平面、19 维观测 × 20 帧、2 维动作、`DIRECT_VP_TORQUE` 执行器 (v100, p600)
- **训练**：均匀目标采样（hardside=0）、严格 0.20 m 成功半径、balance 权重 0.4、time −1.0、
  lr 5e-5 固定、探索 std 0.15–0.3、增益域随机化 [35,100]（仅训练）、蒸馏锚定 SOTA 3830
- **评估**：论文协议（均匀清单 + 随机朝向 + x/y 轴 balance），3 种子 × 40 episodes
- **后续方向**（延续 docs 建议）：SPL 是唯一未达论文的指标；需要新范式（如观测重构从零训练、Transformer
  长历史编码器蒸馏）；以及真机 System-ID 复测与 40 次实物测试
