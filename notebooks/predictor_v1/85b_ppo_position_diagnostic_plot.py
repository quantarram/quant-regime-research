"""
85b_ppo_position_diagnostic_plot.py
======================================
The honest companion figure to 85_ppo_equity.png: that chart alone (12/12
instruments positive alpha) looks like a clean win, but it isn't one --
this plot shows WHY, using the mean/std position data already saved in
85_ppo_results.json. For 6 of 12 instruments, the trained policy's
std(position) over the entire 2022+ holdout is ~0 (its action never
meaningfully varies from a constant ~1.52x long, regardless of the day's
signals) -- real, positive alpha there is leveraged beta capture, not
signal-following. Only JPM, XOM, and EURUSD=X show real, state-conditioned
variation (std(position) > 0.5).

Run: python 85b_ppo_position_diagnostic_plot.py
Output: 85b_ppo_position_diagnostic.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    r = json.load(open(os.path.join(OUT_DIR, "85_ppo_results.json")))
    items = sorted(r["per_instrument"].items(), key=lambda kv: kv[1]["std_position"])
    names = [k for k, v in items]
    means = [v["mean_position"] for k, v in items]
    stds = [v["std_position"] for k, v in items]
    colors = ["#B0492F" if s < 0.1 else "#2f8a4e" for s in stds]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(range(len(names)), means, yerr=stds, color=colors, capsize=4)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="Passive 1x long (reward baseline)")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel("Position size, OOS 2022+ (mean ± std)")
    ax.set_title("What the PPO policy actually does, per instrument, 2022+ holdout\n"
                  "Red = std(position) < 0.1: a near-constant position regardless of the day's signals (beta capture, not skill).\n"
                  "Green = real, signal-conditioned variation. Same run as Figure 2's alpha chart.")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "85b_ppo_position_diagnostic.png"), dpi=140)
    plt.close(fig)
    print("Saved: 85b_ppo_position_diagnostic.png")
    for n, m, s in zip(names, means, stds):
        print(f"  {n:<10} mean_pos={m:+.3f}  std_pos={s:.3f}  {'<- collapsed' if s < 0.1 else '<- real variation'}")
