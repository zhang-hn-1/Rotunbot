#!/usr/bin/env python3
"""Replay model-3809 + executor (v100,p600) evaluation traces as visuals.

The paired evaluation (checkpoint 3809 under DIRECT_VP_TORQUE executor with
velocity gain 100 / position gain 600, 40 episodes x seeds 3/7/11) is already
stored under logs/rotunbot_target_repro/_paired_eval_g100p600_other.  This
script turns the saved 50 Hz traces into:

  figures/overview_seed11.png              all 40 trajectories of seed 11
  figures/overview_all_seeds.png           1x3 panel, seeds 3/7/11
  figures/comparison_seed7_base_vs_high.png base gains vs (v100,p600), seed 7
  figures/sim_effect_grid.gif              2x2 animated replay (seed 11)
  figures/sim_effect_grid.mp4              h264 version of the same frames

It is pure matplotlib (no GPU / no Isaac Gym required).
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

BALL_R = 0.4            # Rotunbot shell radius [m] (urdf sphere radius)
DT = 0.02               # control period [s] (50 Hz)
SUCCESS_R = 0.20        # formal success distance [m]
C_SUCCESS = "#1f77b4"
C_FAIL = "#d62728"
C_TRAIL = "#7fb3d5"
C_PATH = "#aeb6bf"

SRC = Path("logs/rotunbot_target_repro/_paired_eval_g100p600_other")
BASE = Path("logs/rotunbot_target_repro/_paired_eval_base")


def load_episodes(seed):
    """Return list of dicts with details + trace arrays for one seed."""
    root = SRC / "raw/nominal_40" / f"seed_{seed}" / "checkpoint_03809"
    traces = np.load(root / "traces.npz", allow_pickle=True)["traces"]
    with open(root / "episode_details.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    eps = []
    for r in rows:
        ep = dict(r)
        tr = traces[int(r["episode_id"])]
        ep["t"] = np.arange(len(tr["x"])) * DT
        ep["x"] = np.asarray(tr["x"]) + float(r["start_x"])
        ep["y"] = np.asarray(tr["y"]) + float(r["start_y"])
        ep["dist"] = np.asarray(tr["distance"])
        ep["speed"] = np.asarray(tr["speed"])
        ep["steps"] = int(r["steps"])
        eps.append(ep)
    return eps


def summary(seed):
    with open(SRC / "raw/nominal_40" / f"seed_{seed}" / "checkpoint_03809" / "summary.json") as fh:
        import json
        d = json.load(fh)
    return d


def plot_map(ax, eps, title, legend=True):
    """Overlay all episode trajectories on one equal-aspect map."""
    for ep in eps:
        ok = ep["success"] == "1"
        c = C_SUCCESS if ok else C_FAIL
        ax.plot(ep["x"], ep["y"], color=c, lw=0.9, alpha=0.85)
        ax.scatter(ep["x"][0], ep["y"][0], marker="s", s=18, color="k", zorder=5)
        ax.scatter(ep["target_x"], ep["target_y"], marker="*", s=90,
                   color="gold", edgecolor="k", zorder=6)
        circ = plt.Circle((float(ep["target_x"]), float(ep["target_y"])),
                          SUCCESS_R, fill=False, ls="--", lw=0.7,
                          color="#7f8c8d", zorder=4)
        ax.add_patch(circ)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if legend:
        ax.plot([], [], color=C_SUCCESS, lw=1.5, label="成功")
        ax.plot([], [], color=C_FAIL, lw=1.5, label="失败")
        ax.scatter([], [], marker="s", s=18, color="k", label="起点")
        ax.scatter([], [], marker="*", s=90, color="gold", edgecolor="k", label="目标 (0.20 m)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def render_overviews():
    out = SRC / "figures"
    out.mkdir(parents=True, exist_ok=True)

    # --- single best seed --------------------------------------------------
    eps11 = load_episodes(11)
    s = summary(11)
    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    plot_map(ax, eps11, f"model 3809 · 执行器 v100/p600 · seed 11 · 40 回合\n"
                        f"SR {s['success_rate']:.1%} · SPL {s['spl']:.3f} · CLS {s['cls_m']:.3f} m")
    fig.tight_layout()
    fig.savefig(out / "overview_seed11.png", dpi=140)
    plt.close(fig)

    # --- all seeds ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
    for ax, seed in zip(axes, (3, 7, 11)):
        eps = load_episodes(seed)
        s = summary(seed)
        plot_map(ax, eps, f"seed {seed} · SR {s['success_rate']:.1%} · "
                          f"SPL {s['spl']:.3f} · CLS {s['cls_m']:.3f} m",
                 legend=(seed == 11))
    fig.suptitle("model 3809 · 执行器增益 v100 / p600 · 固定清单 40 回合/seed", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "overview_all_seeds.png", dpi=130)
    plt.close(fig)

    # --- seed-7 comparison base vs high gain ------------------------------
    if (BASE / "raw/nominal_40/seed_7/checkpoint_03809/traces.npz").exists():
        def load_base_eps():
            root = BASE / "raw/nominal_40/seed_7/checkpoint_03809"
            traces = np.load(root / "traces.npz", allow_pickle=True)["traces"]
            with open(root / "episode_details.csv", newline="") as fh:
                rows = list(csv.DictReader(fh))
            eps = []
            for r in rows:
                ep = dict(r)
                tr = traces[int(r["episode_id"])]
                ep["x"] = np.asarray(tr["x"]) + float(r["start_x"])
                ep["y"] = np.asarray(tr["y"]) + float(r["start_y"])
                eps.append(ep)
            return eps

        base_eps = load_base_eps()
        with open(BASE / "raw/nominal_40/seed_7/checkpoint_03809/summary.json") as fh:
            import json
            sb = json.load(fh)
        sh = summary(7)
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6))
        plot_map(axes[0], base_eps, f"默认增益 (v35/p300) · SR {sb['success_rate']:.1%} · "
                                    f"SPL {sb['spl']:.3f} · CLS {sb['cls_m']:.3f} m")
        plot_map(axes[1], load_episodes(7), f"执行器 v100/p600 · SR {sh['success_rate']:.1%} · "
                                            f"SPL {sh['spl']:.3f} · CLS {sh['cls_m']:.3f} m")
        fig.suptitle("model 3809 · seed 7 · 同一固定清单 40 回合 · 仅更换执行器增益", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out / "comparison_seed7_base_vs_high.png", dpi=130)
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
def draw_episode_frame(ax_map, ax_dist, ax_spd, ep, idx, ball_r=BALL_R):
    x, y = ep["x"][: idx + 1], ep["y"][: idx + 1]
    tx, ty = float(ep["target_x"]), float(ep["target_y"])

    ax_map.cla()
    # whole path faint, traveled part coloured
    ax_map.plot(ep["x"], ep["y"], color=C_PATH, lw=1.0, zorder=1)
    ok = ep["success"] == "1"
    ax_map.plot(x, y, color=(C_SUCCESS if ok else C_FAIL), lw=1.6, zorder=2)
    # fading trail
    if len(x) > 30:
        ax_map.plot(x[-30:], y[-30:], color=C_TRAIL, lw=2.4, zorder=3)
    # target + success circle + start
    ax_map.scatter(tx, ty, marker="*", s=160, color="gold", edgecolor="k", zorder=6)
    ax_map.add_patch(plt.Circle((tx, ty), SUCCESS_R, fill=False, ls="--",
                                lw=1.0, color="#7f8c8d", zorder=4))
    ax_map.scatter(ep["x"][0], ep["y"][0], marker="s", s=40, color="k", zorder=5)
    # ball shell + heading arrow
    cx, cy = x[-1], y[-1]
    ax_map.add_patch(plt.Circle((cx, cy), ball_r, facecolor="#e74c3c" if not ok else "#2980b9",
                                edgecolor="k", lw=1.2, alpha=0.92, zorder=7))
    if idx >= 2:
        vx = x[-1] - x[-3]
        vy = y[-1] - y[-3]
        n = np.hypot(vx, vy)
        if n > 1e-6:
            vx, vy = vx / n, vy / n
            ax_map.annotate("", xy=(cx + vx * (ball_r + 0.35), cy + vy * (ball_r + 0.35)),
                            xytext=(cx + vx * ball_r, cy + vy * ball_r),
                            arrowprops=dict(arrowstyle="-|>", lw=1.6, color="k"))
    # window: fixed to start..target span
    lo_x = min(ep["x"][0], tx, x.min()) - 1.2
    hi_x = max(ep["x"][0], tx, x.max()) + 1.2
    lo_y = min(ep["y"][0], ty, y.min()) - 1.2
    hi_y = max(ep["y"][0], ty, y.max()) + 1.2
    ax_map.set_xlim(lo_x, hi_x)
    ax_map.set_ylim(lo_y, hi_y)
    ax_map.set_aspect("equal")
    t_now = idx * DT
    ax_map.set_title(f"ep {ep['episode_id']} · {ep['failure_mode'] or ('成功' if ok else '失败')} · "
                     f"t = {t_now:5.1f} s", fontsize=10)
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")

    ax_dist.cla()
    ax_dist.plot(ep["t"][: idx + 1], ep["dist"][: idx + 1], color="#8e44ad", lw=1.3)
    ax_dist.axhline(SUCCESS_R, ls="--", lw=0.8, color="#7f8c8d")
    ax_dist.axvline(t_now, ls=":", lw=0.8, color="k")
    ax_dist.set_title(f"到目标距离 {ep['dist'][idx]:.2f} m (阈值 0.20 m)", fontsize=9)
    ax_dist.set_xlim(0, max(60.0, ep["t"][-1]))
    ax_dist.set_ylim(0, max(2.0, ep["dist"][: idx + 1].max() * 1.1))
    ax_dist.set_xlabel("t [s]")
    ax_dist.set_ylabel("距离 [m]")
    ax_dist.tick_params(labelsize=8)

    ax_spd.cla()
    ax_spd.plot(ep["t"][: idx + 1], ep["speed"][: idx + 1], color="#16a085", lw=1.3)
    ax_spd.axvline(t_now, ls=":", lw=0.8, color="k")
    ax_spd.set_title(f"线速度 {ep['speed'][idx]:.2f} m/s", fontsize=9)
    ax_spd.set_xlim(0, max(60.0, ep["t"][-1]))
    ax_spd.set_ylim(0, max(1.0, ep["speed"][: idx + 1].max() * 1.15))
    ax_spd.set_xlabel("t [s]")
    ax_spd.set_ylabel("速度 [m/s]")
    ax_spd.tick_params(labelsize=8)


def render_animation(seed=11, episodes=(38, 6, 18, 37), fps=10, outdir=None):
    """2x2 animated replay grid.  ep ids: 38 clean success, 6 overshoot
    success, 18 F4 slow timeout, 37 F1 never reached (seed 11)."""
    outdir = outdir or (SRC / "figures")
    outdir.mkdir(parents=True, exist_ok=True)
    eps = {int(e["episode_id"]): e for e in load_episodes(seed)}
    picks = [eps[i] for i in episodes]
    max_idx = max(len(e["x"]) for e in picks)

    tmp = Path("/tmp/sim_frames_3809_v100p600")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15.5, 13.5))
    gs = GridSpec(8, 4, figure=fig, hspace=0.55, wspace=0.25)
    axes = []
    for i, e in enumerate(picks):
        r0 = 2 * i
        axes.append((fig.add_subplot(gs[r0:r0 + 2, 0:2]),
                     fig.add_subplot(gs[r0, 2:4]),
                     fig.add_subplot(gs[r0 + 1, 2:4])))
    fig.suptitle(f"model 3809 · 执行器增益 v100 / p600 · seed {seed} · 仿真回放 "
                 f"(50 Hz 轨迹, 球体半径 0.4 m, 成功圈 0.20 m)", fontsize=13)

    n_frames = max_idx // 5 + 1          # 50 Hz -> fps (every 5th step)
    for f in range(n_frames):
        idx = min(f * 5, max_idx - 1)
        for (ax_map, ax_dist, ax_spd), e in zip(axes, picks):
            draw_episode_frame(ax_map, ax_dist, ax_spd, e, min(idx, len(e["x"]) - 1))
        fig.savefig(tmp / f"frame_{f:04d}.png", dpi=95)
        if f % 60 == 0:
            print(f"  frame {f}/{n_frames - 1}")
    plt.close(fig)

    gif = outdir / "sim_effect_grid.gif"
    mp4 = outdir / "sim_effect_grid.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp / "frame_%04d.png"),
                    "-vf", "split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer",
                    str(gif)], check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp / "frame_%04d.png"),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-pix_fmt", "yuv420p", str(mp4)], check=True, capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return gif, mp4


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overview-only", action="store_true")
    args = ap.parse_args()
    out = render_overviews()
    print("figures ->", out)
    if not args.overview_only:
        gif, mp4 = render_animation()
        print("gif  ->", gif)
        print("mp4  ->", mp4)
