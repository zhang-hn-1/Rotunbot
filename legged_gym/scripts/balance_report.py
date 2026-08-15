#!/usr/bin/env python3
"""Balance-metric evaluation report for model 3809 under executor (v100,p600).

Balance metric (as computed by evaluate_target_repro.py): per step
exp(-||roll_rate,pitch_rate||^2), averaged over the episode and scaled by 100
(100 = perfectly upright, no roll/pitch angular velocity).  Per-episode values
live in episode_details.csv (`balance_metric`, `mean_abs_roll`), and the trace
files additionally carry the raw roll signal.

Comparisons covered (all on the same fixed scenario manifests):
  * 3809 v100/p600  seeds 3/7/11      (_paired_eval_g100p600_other)
  * 3809 base       seed 7 only       (_paired_eval_base)
  * 3809 g100       seeds 3/7/11      (_paired_eval_g100_3809)
  * 3816 v100/p600  seeds 3/7/11      (_paired_eval_g100p600_other)
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Noto Sans CJK JP covers Latin + Greek + CJK in this matplotlib build, so it
# is used as the single family (per-glyph fallback lists do not work here).
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("logs/rotunbot_target_repro")
OUT = ROOT / "_paired_eval_g100p600_other/figures"
OUT.mkdir(parents=True, exist_ok=True)

EVALS = [
    # (dir, checkpoint, label)
    ("_paired_eval_g100p600_other", 3809, "3809 · v100/p600"),
    ("_paired_eval_base",           3809, "3809 · 基准增益"),
    ("_paired_eval_g100_3809",      3809, "3809 · v100"),
    ("_paired_eval_g100p600_other", 3816, "3816 · v100/p600"),
]


def load(evdir, ckpt, seed):
    root = ROOT / evdir / "raw/nominal_40" / f"seed_{seed}" / f"checkpoint_{ckpt:05d}"
    with open(root / "episode_details.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    with open(root / "summary.json") as fh:
        summ = json.load(fh)
    return rows, summ


def stats(rows):
    b = np.array([float(r["balance_metric"]) for r in rows])
    r = np.array([float(r["mean_abs_roll"]) for r in rows])
    ok = np.array([r_["success"] == "1" for r_ in rows])
    return {
        "n": len(b), "mean": b.mean(), "median": np.median(b), "std": b.std(),
        "min": b.min(), "mean_abs_roll": r.mean(),
        "bal_ok": b[ok].mean() if ok.any() else np.nan,
        "bal_fail": b[~ok].mean() if (~ok).any() else np.nan,
    }


def main():
    table = []
    for evdir, ckpt, label in EVALS:
        for seed in (3, 7, 11):
            try:
                rows, summ = load(evdir, ckpt, seed)
            except FileNotFoundError:
                continue
            st = stats(rows)
            table.append((label, seed, st, summ))
    table.sort(key=lambda t: (t[0], t[1]))

    # ---------------- markdown table + CSV --------------------------------
    print(f"{'配置':<16}{'seed':<6}{'SR':<8}{'平衡均值':<10}{'中位':<9}{'std':<8}"
          f"{'min':<8}{'成功时':<10}{'失败时':<10}{'平均|roll|(rad)':<16}")
    with open(OUT / "balance_report.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "seed", "success_rate", "balance_mean", "balance_median",
                    "balance_std", "balance_min", "balance_success", "balance_fail",
                    "mean_abs_roll_rad"])
        for label, seed, st, summ in table:
            print(f"{label:<16}{seed:<6}{summ['success_rate']:<8.1%}{st['mean']:<10.2f}"
                  f"{st['median']:<9.2f}{st['std']:<8.2f}{st['min']:<8.2f}"
                  f"{st['bal_ok']:<10.2f}{st['bal_fail']:<10.2f}{st['mean_abs_roll']:<16.4f}")
            w.writerow([label, seed, summ["success_rate"], round(st["mean"], 3),
                        round(st["median"], 3), round(st["std"], 3), round(st["min"], 3),
                        round(st["bal_ok"], 3) if np.isfinite(st["bal_ok"]) else "",
                        round(st["bal_fail"], 3) if np.isfinite(st["bal_fail"]) else "",
                        round(st["mean_abs_roll"], 4)])

    # ---------------- figures ---------------------------------------------
    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22)

    # A: 3809 v100/p600 per seed, success/failure split
    ax = fig.add_subplot(gs[0, 0])
    data_ok, data_fail, pos_ok, pos_fail, labels = [], [], [], [], []
    for i, (label, seed, st, summ) in enumerate(table):
        if label != "3809 · v100/p600":
            continue
        rows, _ = load("_paired_eval_g100p600_other", 3809, seed)
        b = np.array([float(r["balance_metric"]) for r in rows])
        ok = np.array([r["success"] == "1" for r in rows])
        data_ok.append(b[ok]); data_fail.append(b[~ok])
        pos_ok.append(i); pos_fail.append(i + 0.35)
        labels.append(f"seed {seed}")
    bp = ax.boxplot(data_ok, positions=pos_ok, widths=0.3, patch_artist=True,
                    medianprops=dict(color="k"))
    for p in bp["boxes"]:
        p.set_facecolor("#2e86c1")
    bp2 = ax.boxplot(data_fail, positions=pos_fail, widths=0.3, patch_artist=True,
                     medianprops=dict(color="k"))
    for p in bp2["boxes"]:
        p.set_facecolor("#e74c3c")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_ylabel("balance 指标")
    ax.set_title("(a) 3809 · v100/p600 各 seed 分布（蓝=成功 红=失败）", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # B: seed-7 paired comparison base / v100 / v100p600
    ax = fig.add_subplot(gs[0, 1])
    series, names = [], []
    for evdir, ckpt, label in EVALS:
        if "seed_7" not in str(ROOT / evdir / "raw/nominal_40"):
            pass
        try:
            rows, summ = load(evdir, ckpt, 7)
        except FileNotFoundError:
            continue
        series.append([float(r["balance_metric"]) for r in rows])
        names.append(label)
    bp = ax.boxplot(series, patch_artist=True, medianprops=dict(color="k"),
                    showfliers=False)
    cols = ["#95a5a6", "#f39c12", "#2e86c1"]
    for p, c in zip(bp["boxes"], cols):
        p.set_facecolor(c)
    for i, s in enumerate(series):
        x = np.random.default_rng(i).normal(i + 1, 0.04, len(s))
        ax.scatter(x, s, s=12, color="k", alpha=0.35, zorder=3)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=12, fontsize=9)
    ax.set_ylabel("balance 指标")
    ax.set_title("(b) seed 7 同场景 40 回合 · 仅换执行器增益", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # paired deltas for seed 7 (base vs v100/p600)
    base_rows, _ = load("_paired_eval_base", 3809, 7)
    high_rows, _ = load("_paired_eval_g100p600_other", 3809, 7)
    b_base = np.array([float(r["balance_metric"]) for r in base_rows])
    b_high = np.array([float(r["balance_metric"]) for r in high_rows])
    dlt = b_high - b_base
    print(f"\nseed7 paired delta (v100p600 - base): mean {dlt.mean():+.2f}, "
          f"median {np.median(dlt):+.2f}, worse-in {int((dlt < 0).sum())}/40 eps")

    # C: 3809 vs 3816 under v100/p600, all seeds
    ax = fig.add_subplot(gs[1, 0])
    pos, labs, datas = [], [], []
    for i, seed in enumerate((3, 7, 11)):
        r1, _ = load("_paired_eval_g100p600_other", 3809, seed)
        r2, _ = load("_paired_eval_g100p600_other", 3816, seed)
        datas.append([float(r["balance_metric"]) for r in r1])
        datas.append([float(r["balance_metric"]) for r in r2])
        pos += [2 * i + 1, 2 * i + 2]
        labs += [f"s{seed}\n3809", f"s{seed}\n3816"]
    bp = ax.boxplot(datas, positions=pos, widths=0.7, patch_artist=True,
                    medianprops=dict(color="k"), showfliers=False)
    for p, j in zip(bp["boxes"], range(len(pos))):
        p.set_facecolor("#2e86c1" if j % 2 == 0 else "#16a085")
    ax.set_xticks(pos)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("balance 指标")
    ax.set_title("(c) v100/p600 下 3809 vs 3816（同场景配对）", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # D: roll over time for 4 representative episodes (seed 11)
    ax = fig.add_subplot(gs[1, 1])
    traces = np.load(ROOT / "_paired_eval_g100p600_other/raw/nominal_40/seed_11/"
                     "checkpoint_03809/traces.npz", allow_pickle=True)["traces"]
    for ep_id, ls, lab in ((38, "-", "ep38 成功 SPL 0.969"), (6, "--", "ep6 过冲2次"),
                           (18, ":", "ep18 F4 超时"), (37, "-.", "ep37 F1 未达")):
        tr = traces[ep_id]
        t = np.arange(len(tr["roll"])) * 0.02
        ax.plot(t, tr["roll"], ls, lw=1.2, label=lab)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("roll [rad]")
    ax.set_title("(d) seed 11 代表性回合的 roll 时间序列", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("balance 指标测评 — 每步 exp(-||ω_roll/pitch||²) 的回合均值×100（100=完全直立稳定）",
                 fontsize=13)
    fig.savefig(OUT / "balance_report.png", dpi=130)
    plt.close(fig)
    print("\nbalance_report.png ->", OUT / "balance_report.png")
    print("balance_report.csv ->", OUT / "balance_report.csv")


if __name__ == "__main__":
    main()
