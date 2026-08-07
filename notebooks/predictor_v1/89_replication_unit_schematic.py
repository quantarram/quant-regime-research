"""
89_replication_unit_schematic.py
====================================
Abstract, illustrative schematic of Section 4.2's general principle:
the replication unit used to judge a finding must match the phenomenon's
own claimed mechanism. This is a conceptual diagram, not a plot of real
experimental data -- the real data underlying this claim is in Figures
(TSMOM calendar-block vs. crisis-episode results, elsewhere in the
paper). Two synthetic illustrative timelines (arbitrary, round numbers,
clearly not tied to any real instrument) show how the same underlying
edge, concentrated in short triggered episodes, looks like noise when
chopped into equal calendar blocks but looks like a real, consistent
edge when chopped by the episodes that actually match its mechanism.

Run: python 89_replication_unit_schematic.py
Output: 89_replication_unit_schematic.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RED = "#B0492F"
GREEN = "#2f8a4e"
GREY = "#6b7280"
INK = "#222222"

rng = np.random.default_rng(3)

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)

    T = 300
    t = np.arange(T)
    # a synthetic edge that only exists (positive) inside three short
    # "episode" windows, and drags slightly negative the rest of the time
    # (illustrative of whipsaw/carry cost outside real trigger events) --
    # this is what makes equal calendar blocks land on different sides of
    # zero depending on how much of each block happens to overlap an episode.
    episodes = [(40, 65), (140, 175), (230, 250)]
    edge = np.full(T, -0.30)
    for a, b in episodes:
        edge[a:b] = 1.0
    noise = rng.normal(0, 0.45, T)
    cum = np.cumsum(edge + noise * 0.2)

    # --- Panel A: arbitrary equal calendar blocks ---
    ax = axes[0]
    ax.plot(t, cum, color=INK, lw=1.6)
    n_blocks = 5
    edges_cal = np.linspace(0, T, n_blocks + 1)
    for i in range(n_blocks):
        a, b = edges_cal[i], edges_cal[i + 1]
        seg = cum[int(a):int(b)]
        block_edge = seg[-1] - seg[0] if len(seg) > 1 else 0
        color = GREEN if block_edge > 0 else RED
        ax.axvspan(a, b, color=color, alpha=0.10)
        ax.text((a + b) / 2, ax.get_ylim()[1] if False else max(cum) * 1.12,
                ("+" if block_edge > 0 else "−"), ha="center", fontsize=13, color=color, weight="bold")
        if i > 0:
            ax.axvline(a, color=GREY, lw=0.7, ls=":")
    ax.set_title("Panel A -- wrong replication unit: arbitrary equal-length calendar blocks\n"
                  "same synthetic edge, chopped without regard to when it actually fires -- looks inconsistent, verdict flips block to block",
                  fontsize=9.6, loc="left")
    ax.set_ylabel("cumulative effect\n(illustrative units)")
    ax.set_ylim(top=max(cum) * 1.28)

    # --- Panel B: real triggered episodes ---
    ax2 = axes[1]
    ax2.plot(t, cum, color=INK, lw=1.6)
    for a, b in episodes:
        ax2.axvspan(a, b, color=GREEN, alpha=0.18)
        seg = cum[a:b]
        block_edge = seg[-1] - seg[0] if len(seg) > 1 else 0
        ax2.text((a + b) / 2, max(cum) * 1.12, "+", ha="center", fontsize=13, color=GREEN, weight="bold")
    ax2.set_title("Panel B -- matched replication unit: the episodes when the mechanism actually triggers\n"
                   "identical underlying series -- consistent, repeatable edge visible in every real episode",
                   fontsize=9.6, loc="left")
    ax2.set_ylabel("cumulative effect\n(illustrative units)")
    ax2.set_xlabel("time (illustrative, not a real instrument or date)")
    ax2.set_ylim(top=max(cum) * 1.28)

    fig.suptitle("Schematic: the same real, triggered edge looks absent or inconsistent under the wrong\n"
                  "replication unit, and consistent under the unit matching its own mechanism\n"
                  "(illustrative synthetic series -- the real TSMOM calendar-block vs. crisis-episode data behind this principle is elsewhere in this paper)",
                  fontsize=10.6, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "89_replication_unit_schematic.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: 89_replication_unit_schematic.png")
