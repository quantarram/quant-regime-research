"""
Schematic figures + one new results plot for Paper 15, matching the
matplotlib-diagram style already established for this paper series
(schematic_fresh_vs_stale_design.png, Paper 13).

Produces:
  p15_schematic_five_levers.png     -- all 5 levers tested against the ceiling
                                        across Papers 14-15, and their verdicts
  p15_schematic_multiblock.png      -- tau*-scale block chaining design
  p15_schematic_three_fixes.png     -- the three calibration bugs found + fixed
  p15_crps_fix_progression.png      -- real numbers: JPM/AAPL CRPS at each fix stage
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

COL_WALL = "#4a4a4a"
COL_FAIL = "#8E8E8E"
COL_WIN = "#1B4F72"
COL_BOX_BG = "#EDEDED"


# ---------------------------------------------------------------------------
# Figure 1: five levers tested against the ceiling
# ---------------------------------------------------------------------------
def make_five_levers():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    wall_x = 8.6
    ax.plot([wall_x, wall_x], [0.6, 6.6], color=COL_WALL, lw=5, solid_capstyle="round", zorder=2)
    ax.text(wall_x, 6.85, "measured predictability limit  τ*", ha="center", fontsize=12.5,
            fontweight="bold", color=COL_WALL)

    levers = [
        ("Architecture",        "climatology / tree / RL / GAN / VAE",       "converges to climatology",        "Paper 14"),
        ("Depth",                "linear  →  1 hidden layer (tanh, real backprop)", "no change, still climatology", "Paper 14"),
        ("Training-window size", "0.5×τ*  →  8×τ*",                          "skill degrades past τ*, doesn't improve", "Paper 14"),
        ("Loss function",        "L2  →  quantile (pinball)  →  Lq, q=2..8", "median still = climatology; Lq destabilizes\nor does nothing, never helps", "Paper 15, §3"),
        ("Uncertainty quantification", "generative downscaler ensemble", "point forecast still = climatology exactly;\nCRPS beats all 5 architectures anyway", "Paper 15, §4"),
    ]
    y_positions = [6.3, 5.0, 3.7, 2.4, 1.1]
    ax.set_ylim(0.1, 7.2)
    for (name, tested, verdict, src), y in zip(levers, y_positions):
        ax.annotate("", xy=(wall_x - 0.15, y), xytext=(0.3, y),
                     arrowprops=dict(arrowstyle="-|>", color=COL_FAIL if "Paper 14" in src or "§3" in src else COL_WIN,
                                      lw=2.2, mutation_scale=18))
        name_fontsize = 11 if len(name) > 20 else 12
        ax.text(0.35, y + 0.34, name, fontsize=name_fontsize, fontweight="bold", color="#222222")
        ax.text(0.35, y - 0.02, tested, fontsize=9, color="#555555", style="italic")
        ax.text(wall_x + 0.3, y + 0.12, verdict, fontsize=9.3, color="#222222", va="center")
        ax.text(wall_x + 0.3, y - 0.32, src, fontsize=8, color="#8E8E8E", va="center")

    ax.text(6.5, 0.05, "Every lever stops at the same wall. The fifth wins a real, proper scoring rule (CRPS) --\n"
                        "but only by describing the wall honestly, never by getting past it (see §4.5).",
            ha="center", fontsize=9.5, color="#444444", style="italic")

    fig.suptitle("Five levers, one ceiling: architecture, depth, training-window size (Paper 14)\n"
                  "plus loss function and calibrated uncertainty (Paper 15)", fontsize=13.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, "p15_schematic_five_levers.png"), dpi=150)
    plt.close(fig)
    print("Saved p15_schematic_five_levers.png")


# ---------------------------------------------------------------------------
# Figure 2: multiblock tau*-scale chaining design
# ---------------------------------------------------------------------------
def make_multiblock():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.set_title("Building a full-horizon scenario from independent τ*-scale blocks", fontsize=13.5, pad=14)

    horizon_start, horizon_end = 0.8, 12.2
    y_top = 3.7
    ax.annotate("", xy=(horizon_end, y_top), xytext=(horizon_start, y_top),
                arrowprops=dict(arrowstyle="-", color="#333333", lw=1.2))
    ax.text((horizon_start + horizon_end) / 2, y_top + 0.35, "forecast horizon H  (e.g. 252 days)",
            ha="center", fontsize=10.5, color="#333333")

    n_blocks = 6
    block_w = (horizon_end - horizon_start) / n_blocks
    colors = ["#5B8DBE", "#2E8B57", "#C0392B", "#8E44AD", "#D68910", "#1B4F72"]
    for i in range(n_blocks):
        x0 = horizon_start + i * block_w
        rect = FancyBboxPatch((x0 + 0.05, y_top - 0.55), block_w - 0.1, 0.5,
                               boxstyle="round,pad=0.02,rounding_size=0.04",
                               linewidth=1.3, edgecolor=colors[i], facecolor=colors[i], alpha=0.18)
        ax.add_patch(rect)
        ax.text(x0 + block_w / 2, y_top - 0.3, f"block {i+1}\n(τ* days)", ha="center", va="center",
                fontsize=8.3, color=colors[i], fontweight="bold")

    y_mid = 2.1
    ax.text(horizon_start - 0.3, y_mid + 0.55, "each block: INDEPENDENT draw, own conditioning value",
            fontsize=9.5, color="#444444", style="italic")
    for i in range(n_blocks):
        x0 = horizon_start + i * block_w
        cx = x0 + block_w / 2
        ax.annotate("", xy=(cx, y_mid), xytext=(cx, y_top - 0.6),
                    arrowprops=dict(arrowstyle="-|>", color=colors[i], lw=1.4, mutation_scale=12))
        ax.text(cx, y_mid - 0.25, r"$c_i = r_{coarse}\!\cdot\!\frac{L_i}{H}$", ha="center", fontsize=7.6, color=colors[i])
        ax.text(cx, y_mid - 0.62, "shape-VAE\n(h = τ*)", ha="center", fontsize=7.6, color="#555555")

    y_bot = 0.55
    ax.annotate("", xy=(horizon_end, y_bot), xytext=(horizon_start, y_bot),
                arrowprops=dict(arrowstyle="-|>", color=COL_WIN, lw=2.2, mutation_scale=16))
    ax.text((horizon_start + horizon_end) / 2, y_bot - 0.35,
            "concatenated daily-return path  ->  one full-horizon scenario  (repeat K times for the ensemble)",
            ha="center", fontsize=9.5, color=COL_WIN)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "p15_schematic_multiblock.png"), dpi=150)
    plt.close(fig)
    print("Saved p15_schematic_multiblock.png")


# ---------------------------------------------------------------------------
# Figure 3: the three calibration fixes
# ---------------------------------------------------------------------------
def make_three_fixes():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.0))
    fig.suptitle("Three real calibration bugs found and fixed, in sequence", fontsize=14, y=0.99)

    # Panel 1: horizon/tau* mismatch
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Fix 1: model scale\nvs. τ*", fontsize=11.5, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.5, 6.3), 9, 1.6, boxstyle="round,pad=0.05", facecolor="#F5DADA", edgecolor="#C0392B"))
    ax.text(5, 7.1, "BEFORE: one linear VAE\nasked to model 252 days as ONE shape", ha="center", va="center", fontsize=9.3, color="#7B2C2C")
    ax.annotate("", xy=(5, 5.6), xytext=(5, 6.2), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2))
    ax.text(5, 5.3, "but dynamics decorrelate\npast τ* = 23d (~11 regimes, not 1)", ha="center", fontsize=8.8, color="#444444", style="italic")
    ax.annotate("", xy=(5, 4.2), xytext=(5, 4.9), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2))
    ax.add_patch(FancyBboxPatch((0.5, 1.9), 9, 1.6, boxstyle="round,pad=0.05", facecolor="#DAEAF5", edgecolor="#1B4F72"))
    ax.text(5, 2.7, "AFTER: shape-VAE trained at h=τ* only,\nfull horizon = chain of independent blocks", ha="center", va="center", fontsize=9.3, color="#1B4F72")
    ax.text(5, 0.9, "JPM CRPS: 46.0% -> 45.4%\n(necessary, not sufficient alone)", ha="center", fontsize=8.6, color="#666666")

    # Panel 2: decoder variance double-counting
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Fix 2: decoder variance\ndouble-counting", fontsize=11.5, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.5, 6.3), 9, 1.6, boxstyle="round,pad=0.05", facecolor="#F5DADA", edgecolor="#C0392B"))
    ax.text(5, 7.1, "BEFORE: Var(x) = Var$_z$(μ$_d$) + 1\n(full unit-variance noise ON TOP of z)", ha="center", va="center", fontsize=9.1, color="#7B2C2C")
    ax.annotate("", xy=(5, 5.6), xytext=(5, 6.2), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2))
    ax.text(5, 5.3, "law of total variance:\ntarget marginal variance = 1, not 1+extra", ha="center", fontsize=8.8, color="#444444", style="italic")
    ax.annotate("", xy=(5, 4.2), xytext=(5, 4.9), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2))
    ax.add_patch(FancyBboxPatch((0.5, 1.9), 9, 1.6, boxstyle="round,pad=0.05", facecolor="#DAEAF5", edgecolor="#1B4F72"))
    ax.text(5, 2.7, r"AFTER: $\sigma_d^2$ = max(1 $-$ Var$_z(\mu_d)$, ε)" + "\ndecoder noise shrinks to what's left over", ha="center", va="center", fontsize=9.1, color="#1B4F72")
    ax.text(5, 0.9, "JPM CRPS: 45.4% -> 42.0%\n(measured Var_z = 0.16, ~8%/day excess)", ha="center", fontsize=8.6, color="#666666")

    # Panel 3: recency-windowed calibration
    ax = axes[2]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Fix 3: sub-linear real\nvariance scaling", fontsize=11.5, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.5, 6.3), 9, 1.6, boxstyle="round,pad=0.05", facecolor="#F5DADA", edgecolor="#C0392B"))
    ax.text(5, 7.1, "BEFORE: independent-block chaining\n= linear variance scaling assumed", ha="center", va="center", fontsize=9.1, color="#7B2C2C")
    ax.annotate("", xy=(5, 5.6), xytext=(5, 6.2), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2))
    ax.text(5, 5.3, "real equities mean-revert beyond τ*\n(Fama-French / Poterba-Summers effect)", ha="center", fontsize=8.8, color="#444444", style="italic")
    ax.annotate("", xy=(5, 4.2), xytext=(5, 4.9), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2))
    ax.add_patch(FancyBboxPatch((0.5, 1.9), 9, 1.6, boxstyle="round,pad=0.05", facecolor="#DAEAF5", edgecolor="#1B4F72"))
    ax.text(5, 2.7, "AFTER: ensemble std rescaled to match\nreal RECENT-window horizon-scale std", ha="center", va="center", fontsize=9.1, color="#1B4F72")
    ax.text(5, 0.9, "JPM CRPS: 42.0% -> 25.6%\n(beats climatology's 27.7%)", ha="center", fontsize=8.6, color="#666666")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(OUT_DIR, "p15_schematic_three_fixes.png"), dpi=150)
    plt.close(fig)
    print("Saved p15_schematic_three_fixes.png")


# ---------------------------------------------------------------------------
# Figure 4: real CRPS numbers at each fix stage (JPM, AAPL) -- a plot, not a schematic
# ---------------------------------------------------------------------------
def make_crps_progression():
    stages = ["horizon-scale\nVAE\n(original)", "+ τ*-scale\nchaining\n(fix 1)",
              "+ decoder\nvariance fix\n(fix 2)", "+ recency-window\ncalibration\n(fix 3)"]
    jpm_downscaler = [46.039, 45.395, 41.963, 25.577]
    jpm_climatology = 27.738
    aapl_downscaler = [85.545, None, None, 38.478]  # fixes 1+2 tested together for AAPL in the session
    aapl_climatology = 39.104

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, downscaler_vals, clim, title, color in [
        (axes[0], jpm_downscaler, jpm_climatology, "JPM (τ*=23d, horizon=252d)", "#C0392B"),
        (axes[1], aapl_downscaler, aapl_climatology, "AAPL (τ*=22d, horizon=252d)", "#8E44AD"),
    ]:
        xs = list(range(len(stages)))
        ys = [v if v is not None else np.nan for v in downscaler_vals]
        ax.plot(xs, ys, marker="o", ms=9, lw=2.2, color=color, label="generative downscaler CRPS")
        ax.axhline(clim, color="#333333", lw=1.8, linestyle="--", label=f"climatology CRPS ({clim:.1f}%)")
        for x, y in zip(xs, ys):
            if not np.isnan(y):
                ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=9, color=color, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(stages, fontsize=8.6)
        ax.set_ylabel("CRPS (% of mean price)")
        ax.set_title(title, fontsize=11.5)
        ax.legend(fontsize=8.6, loc="upper right")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Each fix's real, measured effect on CRPS -- the two worst-hit (longest-horizon) instruments",
                 fontsize=13, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT_DIR, "p15_crps_fix_progression.png"), dpi=150)
    plt.close(fig)
    print("Saved p15_crps_fix_progression.png")


if __name__ == "__main__":
    make_five_levers()
    make_multiblock()
    make_three_fixes()
    make_crps_progression()
