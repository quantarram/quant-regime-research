"""
86_ppo_position_phaseplot.py
================================
A genuine 2D state-space view of the same real data already saved in
85_ppo_results.json (mean and std of position per instrument, OOS 2022+
holdout) -- std(position) on the x-axis (log scale, since several
instruments are ~1e-7), mean(position) on the y-axis. This is the
proper scientific-graphics way to show what Figure 3's bar chart shows
qualitatively: instruments cluster tightly around a single point
(near-constant ~1.52x, near-zero std) except for a small group that
shows genuine, substantial signal-conditioned variation. No new data --
same run, same JSON, a different and more informative projection of it.

Run: python 86_ppo_position_phaseplot.py
Output: 86_ppo_position_phaseplot.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
GREEN = "#2f8a4e"
RED = "#B0492F"

if __name__ == "__main__":
    r = json.load(open(os.path.join(OUT_DIR, "85_ppo_results.json")))
    items = r["per_instrument"]
    names = list(items.keys())
    means = np.array([items[n]["mean_position"] for n in names])
    stds = np.array([max(items[n]["std_position"], 1e-7) for n in names])
    collapsed = stds < 0.1

    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.axvspan(1e-8, 0.1, color=RED, alpha=0.06, zorder=0)
    ax.axvspan(0.1, 3, color=GREEN, alpha=0.06, zorder=0)
    ax.axvline(0.1, color="grey", lw=0.8, ls=":")

    ax.scatter(stds[collapsed], means[collapsed], s=95, color=RED, edgecolor="white",
               linewidth=0.8, zorder=3, label="std(position) < 0.1 -- beta capture")
    ax.scatter(stds[~collapsed], means[~collapsed], s=140, color=GREEN, edgecolor="white",
               linewidth=1.0, marker="D", zorder=3, label="std(position) $\\geq$ 0.1 -- real, signal-conditioned")
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="passive 1x long (reward baseline)")

    order = np.argsort(stds)
    for i, idx in enumerate(order):
        n, x, y = names[idx], stds[idx], means[idx]
        up = (i % 2 == 0)
        offset = (0, 10) if up else (0, -15)
        va = "bottom" if up else "top"
        ax.annotate(n, (x, y), textcoords="offset points", xytext=offset,
                    fontsize=8.4, ha="center", va=va)

    ax.set_xscale("log")
    ax.set_xlim(3e-8, 6)
    ax.set_ylim(0.55, 1.65)
    ax.set_xlabel("std(position), OOS 2022+ holdout (log scale)")
    ax.set_ylabel("mean(position), OOS 2022+ holdout")
    ax.set_title("PPO policy behavior in position state-space, all 12 instruments\n"
                  "Left cluster: a single near-constant ~1.5x position regardless of instrument or signal.\n"
                  "Right, separated by more than two orders of magnitude in std(position): real variation.")
    ax.legend(loc="lower left", fontsize=8.6, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "86_ppo_position_phaseplot.png"), dpi=150)
    plt.close(fig)
    print("Saved: 86_ppo_position_phaseplot.png")
