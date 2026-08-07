"""
87_ppo_architecture_schematic.py
====================================
Schematic flowchart of the PPO actor-critic training loop used in
85_ppo_decision_layer.py -- state construction (all three predictive
signal types), the shared-trunk network with its policy and value
heads, the environment step, GAE advantage computation, and the
clipped-surrogate policy update. Purely illustrative of the pipeline
already described in prose in Section 3.1 -- no new data, no results.

Run: python 87_ppo_architecture_schematic.py
Output: 87_ppo_architecture_schematic.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BLUE = "#3B6FA0"
GREEN = "#2f8a4e"
GREY = "#6b7280"
INK = "#222222"
BOXFACE = "#EDF1F6"


def box(ax, xy, w, h, text, face=BOXFACE, edge=BLUE, fontsize=9.6, weight="normal"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                        linewidth=1.3, edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, weight=weight, zorder=3, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, p0, p1, color=GREY, style="-|>", lw=1.4, connectionstyle="arc3,rad=0.0", ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, color=color, lw=lw,
                         connectionstyle=connectionstyle, zorder=1, linestyle=ls,
                         mutation_scale=13)
    ax.add_patch(a)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(12, 7.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    ax.text(6, 8.3, "PPO actor-critic training loop (85_ppo_decision_layer.py)",
            ha="center", va="center", fontsize=12.5, weight="bold", color=INK)

    # --- state block (three signal types) ---
    s0 = box(ax, (0.3, 6.15), 3.0, 0.62, "predictor_v1 quantile\nforecast (median, dispersion)",
             face="#E7F0E9", edge=GREEN, fontsize=8.6)
    s1 = box(ax, (0.3, 5.35), 3.0, 0.62, "$\\tau^{*}$ / predictability-\npocket structure", face="#E7F0E9", edge=GREEN, fontsize=8.6)
    s2 = box(ax, (0.3, 4.55), 3.0, 0.62, "CPE conditional tail-\nexceedance probability", face="#E7F0E9", edge=GREEN, fontsize=8.6)
    s3 = box(ax, (0.3, 3.6), 3.0, 0.62, "current position,\nunrealized P&L (path-dependent)",
             face="#F7EFE3", edge="#B8863B", fontsize=8.6)

    state = box(ax, (3.9, 4.35), 2.0, 2.0, "state\n$s_t$", face="#DDE6F0", edge=BLUE, fontsize=10.5, weight="bold")
    for src in (s0, s1, s2, s3):
        x, y, w, h = src
        arrow(ax, (x + w, y + h / 2), (state[0], state[1] + state[3] * 0.5 + (0.4 if src in (s0, s1) else -0.4 if src in (s2, s3) else 0)))

    # --- network ---
    net = box(ax, (6.3, 4.85), 2.5, 1.5, "shared MLP trunk", face="#DDE6F0", edge=BLUE, fontsize=9.5)
    arrow(ax, (state[0] + state[2], state[1] + state[3] / 2), (net[0], net[1] + net[3] / 2))

    pol = box(ax, (9.2, 5.55), 2.3, 0.75, "policy head\n$\\tanh(\\mu_\\theta)$, learned $\\log\\sigma$", face="#E7F0E9", edge=GREEN, fontsize=8.3)
    val = box(ax, (9.2, 4.55), 2.3, 0.75, "value head\n$V_\\theta(s_t)$", face="#F0E7ED", edge="#8a3b6b", fontsize=8.3)
    arrow(ax, (net[0] + net[2], net[1] + net[3] * 0.75), (pol[0], pol[1] + pol[3] / 2))
    arrow(ax, (net[0] + net[2], net[1] + net[3] * 0.25), (val[0], val[1] + val[3] / 2))

    # --- action -> environment ---
    act = box(ax, (9.2, 3.25), 2.3, 0.66, "action $a_t$\n(position size)", face=BOXFACE, edge=BLUE, fontsize=8.8)
    arrow(ax, (pol[0] + pol[2] * 0.5, pol[1]), (act[0] + act[2] * 0.5, act[1] + act[3]))

    env = box(ax, (6.3, 2.15), 2.5, 0.75, "environment step\n$r_t = (a_t{-}1)R_t - c|a_t{-}a_{t-1}|$", face="#FDF3E3", edge="#B8863B", fontsize=8.3)
    arrow(ax, (act[0], act[1] + act[3] * 0.4), (env[0] + env[2], env[1] + env[3] * 0.6))

    roll = box(ax, (3.4, 2.15), 2.5, 0.75, "rollout buffer\n(one sequential episode / instrument)", face=BOXFACE, edge=GREY, fontsize=8.3)
    arrow(ax, (env[0], env[1] + env[3] * 0.5), (roll[0] + roll[2], roll[1] + roll[3] * 0.5))
    arrow(ax, (roll[0] + roll[2] * 0.5, roll[1] + roll[3]), (state[0] + state[2] * 0.5, 4.35),
          color=GREY, connectionstyle="arc3,rad=-0.35", ls="--")

    gae = box(ax, (0.3, 1.05), 3.0, 0.75, "GAE advantage\n$\\hat{A}_t = \\sum (\\gamma\\lambda)^l \\delta_{t+l}$", face="#DDE6F0", edge=BLUE, fontsize=8.6)
    arrow(ax, (roll[0], roll[1] + roll[3] * 0.5), (gae[0] + gae[2], gae[1] + gae[3] * 0.5))

    upd = box(ax, (0.3, 0.05), 3.0, 0.75, "clipped surrogate update\n$L^{\\mathrm{CLIP}}(\\theta)$ (3 epochs / rollout)", face="#DDE6F0", edge=BLUE, fontsize=8.6)
    arrow(ax, (gae[0] + gae[2] * 0.5, gae[1]), (upd[0] + upd[2] * 0.5, upd[1] + upd[3]))
    arrow(ax, (upd[0], upd[1] + upd[3] * 0.5), (net[0] + net[2] * 0.15, net[1]),
          color=BLUE, connectionstyle="arc3,rad=0.3", ls="--")

    ax.text(0.3, 0.9, "(3,000 epochs total)", fontsize=7.6, color=GREY, style="italic")
    ax.text(1.35, -0.15, "weight update, back to network", fontsize=7.4, color=BLUE, style="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "87_ppo_architecture_schematic.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: 87_ppo_architecture_schematic.png")
