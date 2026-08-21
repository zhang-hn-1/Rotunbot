#!/usr/bin/env bash
# ============================================================================
# SRU 方案 GPU 运行脚本（必须在有 GPU 的机器上运行：jason-Legion-Y9000X-IRX9）
#
# 运行方式（在你的终端）:
#   cd /home/jason/SphericalRobot_LeggedGym-master-new-map
#   bash run_sru_gpu.sh
#
# 流程:
#   0. 环境自检（Python/导入/SRU 网络快速前向）
#   1. 方案 B: SRU 调制（冻结 uniform 4150 base + SRU 残差调制）训练 50 迭代
#   2. 方案 B 评估（论文协议均匀清单, seeds 3/7/11 x 40）
#   3. 方案 A: SRU 直接控制训练 50 迭代
#   4. 方案 A 评估
#
# 说明: 本脚本在当前沙箱环境没有 GPU 设备权限，必须在你的真实终端执行。
# ============================================================================
set -u

REPO=/home/jason/SphericalRobot_LeggedGym-master-new-map
VENV=/home/jason/legged_gym/.venv/bin
export PATH="$VENV:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="$REPO:$VENV/../lib/python3.8/site-packages"
export LEGGED_GYM_ROOT_DIR="$REPO"
cd "$REPO" || exit 1

PY="$VENV/python"
EVAL="legged_gym/scripts/evaluate_target_repro.py"
SCEN_DIR=/tmp/rotunbot_paper_gap_sru
mkdir -p "$SCEN_DIR"

step() { echo; echo "========== $* =========="; }

# ---------------------------------------------------------------------------
step "0. 环境自检"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || { echo "!! 无 GPU，脚本中止"; exit 1; }
$PY - <<'EOF'
import sys, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault("LEGGED_GYM_ROOT_DIR", os.getcwd())
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())
assert torch.cuda.is_available(), "需要 CUDA GPU"
from legged_gym.dwl.actor_critic_sru_lh import ActorCriticSRULH, ActorCriticSRUModulate
m = ActorCriticSRUModulate(95, 19, 63, 2, in_channels=20,
    base_path="logs/rotunbot_target_repro/Aug16_02-57-06_uniform_t1_long500_from3809/model_4150.pt")
obs = torch.randn(2, 380)
with torch.no_grad():
    b = m.base.act_inference(obs); a = m.act_inference(obs)
assert float((a-b).abs().max()) < 1e-6, "SRU 调制初始必须为零"
print("SRU 网络自检 OK（初始 delta=0，行为==base 4150）")
EOF
[ $? -eq 0 ] || { echo "!! 自检失败"; exit 1; }

# ---------------------------------------------------------------------------
step "1. 方案 B: SRU 调制训练（base=uniform 4150 冻结, 50 迭代）"
$PY legged_gym/scripts/train.py --task rotunbot_target_sru_mod --headless

# 找到方案 B 最新训练目录
MOD_RUN=$(ls -dt logs/rotunbot_target_sru/*sru_modulate* 2>/dev/null | head -1)
echo "方案 B 训练目录: $MOD_RUN"
[ -n "$MOD_RUN" ] || { echo "!! 未找到方案 B 训练输出"; exit 1; }

# ---------------------------------------------------------------------------
step "2. 生成论文协议均匀清单（seeds 3/7/11）"
for S in 3 7 11; do
  $PY $EVAL --mode generate-scenarios --run-dir . --output-dir . \
      --seed $S --episodes 40 \
      --scenario-file "$SCEN_DIR/scenarios_seed_$S.npz" --uniform-targets
done

step "3. 方案 B 评估（nominal_40, 3 seeds）"
for S in 3 7 11; do
  $PY $EVAL --mode worker --run-dir "$MOD_RUN" --output-dir "logs/sru_eval_mod" \
      --seed $S --checkpoint -1 --episodes 40 \
      --scenario-file "$SCEN_DIR/scenarios_seed_$S.npz" \
      --phase nominal_40 --perturbation nominal \
      --control-type DIRECT_VP_TORQUE --task rotunbot_target_sru_mod --force
done

# ---------------------------------------------------------------------------
step "4. 方案 A: SRU 直接控制训练（50 迭代）"
$PY legged_gym/scripts/train.py --task rotunbot_target_sru --headless

DIR_RUN=$(ls -dt logs/rotunbot_target_sru/*sru_direct* 2>/dev/null | head -1)
echo "方案 A 训练目录: $DIR_RUN"
[ -n "$DIR_RUN" ] || { echo "!! 未找到方案 A 训练输出"; exit 1; }

step "5. 方案 A 评估（nominal_40, 3 seeds）"
for S in 3 7 11; do
  $PY $EVAL --mode worker --run-dir "$DIR_RUN" --output-dir "logs/sru_eval_direct" \
      --seed $S --checkpoint -1 --episodes 40 \
      --scenario-file "$SCEN_DIR/scenarios_seed_$S.npz" \
      --phase nominal_40 --perturbation nominal \
      --control-type DIRECT_VP_TORQUE --task rotunbot_target_sru --force
done

# ---------------------------------------------------------------------------
step "6. 迷宫 SRU 点到点训练（rotunbot_maze_sru, 200 迭代）"
echo "提示: 若方案 A/B 在平面上效果不佳，迷宫阶段可先跳过；"
echo "      若要继续，直接运行: python legged_gym/scripts/train.py --task rotunbot_maze_sru --headless"
$PY legged_gym/scripts/train.py --task rotunbot_maze_sru --headless

step "全部完成"
echo "方案 B (调制) 评估: logs/sru_eval_mod"
echo "方案 A (直接) 评估: logs/sru_eval_direct"
echo "迷宫 SRU 训练: logs/rotunbot_maze_sru/"
echo "训练输出: logs/rotunbot_target_sru/"
