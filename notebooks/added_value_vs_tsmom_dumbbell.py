"""
added_value_vs_tsmom_dumbbell.py
====================================
Same real point estimates already computed and saved by
added_value_vs_tsmom.py (added_value_vs_tsmom_results.json) -- redrawn
as a paired dot ("dumbbell") plot instead of grouped bars, which is the
more standard scientific-graphics way to show a paired comparison (own-
signal method vs. standard TSMOM, same instrument, same holdout, same
benchmark): the connecting line makes the size of the gap the visual
subject, not just the two bar heights. No new data, no retraining --
reads the same JSON added_value_vs_tsmom.py already wrote.

Run: python added_value_vs_tsmom_dumbbell.py
Output: added_value_vs_tsmom_dumbbell.png
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEN = "#2f8a4e"
LGREEN = "#8FBF9F"
GREY = "#9AA1AD"
INK = "#222222"

if __name__ == "__main__":
    r = json.load(open("added_value_vs_tsmom_results.json"))
    instruments = sorted(r.keys(), key=lambda k: r[k]["tsmom_alpha_full_pct"])

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    y = np.arange(len(instruments))

    for i, tkr in enumerate(instruments):
        d = r[tkr]
        tsmom = d["tsmom_alpha_full_pct"]
        kelly = d["Kelly-sized"]
        master = d["Master model"]
        lo, hi = min(tsmom, kelly), max(tsmom, kelly)
        ax.plot([lo, hi], [i, i], color=GREY, lw=1.6, zorder=1)
        ax.scatter([tsmom], [i], color=GREY, s=90, zorder=3, edgecolor="white", linewidth=0.7,
                   label="Standard TSMOM (existing quant-shop tool)" if i == 0 else None)
        ax.scatter([kelly], [i], color=GREEN, s=110, marker="D", zorder=3, edgecolor="white", linewidth=0.7,
                   label="Kelly-sized (own method)" if i == 0 else None)
        ax.scatter([master], [i], color=LGREEN, s=70, marker="^", zorder=3, edgecolor="white", linewidth=0.7,
                   label="Master model (own method)" if i == 0 else None)

    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(instruments)
    ax.set_xlabel("Alpha vs. own buy-and-hold, 2022+ holdout (%/yr)")
    ax.set_title("Own-signal methods vs. standard TSMOM, same instrument, same holdout, same benchmark\n"
                  "connecting line spans the gap between standard TSMOM and this program's own Kelly-sized method")
    ax.legend(loc="lower right", fontsize=8.8, framealpha=0.95)
    ax.grid(axis="x", color="#e5e5e5", lw=0.7, zorder=0)
    fig.tight_layout()
    fig.savefig("added_value_vs_tsmom_dumbbell.png", dpi=150)
    plt.close(fig)
    print("Saved: added_value_vs_tsmom_dumbbell.png")
