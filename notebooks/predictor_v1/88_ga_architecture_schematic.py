"""
88_ga_architecture_schematic.py
===================================
Schematic flowchart of the memoryless genetic-algorithm population
search used in 82-84_ga_*.py -- population initialization, fully
vectorized fitness evaluation (einsum across the population dimension),
selection, crossover/mutation, and the generational loop. Purely
illustrative of the pipeline already described in prose in Section 3.1
-- no new data, no results.

Run: python 88_ga_architecture_schematic.py
Output: 88_ga_architecture_schematic.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BLUE = "#3B6FA0"
GREEN = "#2f8a4e"
GREY = "#6b7280"
INK = "#222222"
BOXFACE = "#EDF1F6"


def box(ax, xy, w, h, text, face=BOXFACE, edge=BLUE, fontsize=9.4, weight="normal"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                        linewidth=1.3, edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, weight=weight, zorder=3, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, p0, p1, color=GREY, lw=1.4, connectionstyle="arc3,rad=0.0", ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", color=color, lw=lw,
                         connectionstyle=connectionstyle, zorder=1, linestyle=ls,
                         mutation_scale=13)
    ax.add_patch(a)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.3, 6.6)
    ax.axis("off")

    ax.text(6, 6.35, "GA memoryless population search (82_ga_decision_layer_search.py)",
            ha="center", va="center", fontsize=12.5, weight="bold", color=INK)

    init = box(ax, (0.3, 4.35), 2.5, 1.0, "initialize population\n800 candidate policies\n(random linear weights)", fontsize=8.6)
    signals = box(ax, (0.3, 2.7), 2.5, 1.0, "same 3 signal types\n(same-day only --\nno path-dependent state)", face="#E7F0E9", edge=GREEN, fontsize=8.6)

    fit = box(ax, (3.6, 3.5), 3.0, 1.2, "vectorized fitness evaluation\neinsum across population $\\times$\n12 instruments $\\times$ time",
              face="#DDE6F0", edge=BLUE, fontsize=8.8, weight="bold")
    arrow(ax, (init[0] + init[2], init[1] + init[3] * 0.5), (fit[0], fit[1] + fit[3] * 0.65))
    arrow(ax, (signals[0] + signals[2], signals[1] + signals[3] * 0.5), (fit[0], fit[1] + fit[3] * 0.35))

    fiteq = box(ax, (3.6, 2.15), 3.0, 0.75, "fitness $= \\overline{\\mathrm{net\\_ret}}\\, /\\, \\overline{|a|}$\n(P&L per unit exposure)", face="#FDF3E3", edge="#B8863B", fontsize=8.6)
    arrow(ax, (fit[0] + fit[2] * 0.5, fit[1]), (fiteq[0] + fiteq[2] * 0.5, fiteq[1] + fiteq[3]))

    sel = box(ax, (7.4, 3.9), 2.0, 0.75, "selection\n(top performers)", fontsize=8.8)
    arrow(ax, (fit[0] + fit[2], fit[1] + fit[3] * 0.6), (sel[0], sel[1] + sel[3] * 0.5))

    xover = box(ax, (9.9, 3.9), 2.0, 0.75, "crossover +\nmutation", fontsize=8.8)
    arrow(ax, (sel[0] + sel[2], sel[1] + sel[3] * 0.5), (xover[0], xover[1] + xover[3] * 0.5))

    nextgen = box(ax, (7.4, 2.4), 4.5, 0.75, "next generation (800 policies)", face=BOXFACE, edge=GREY, fontsize=9.0)
    arrow(ax, (xover[0] + xover[2] * 0.5, xover[1]), (nextgen[0] + nextgen[2] * 0.85, nextgen[1] + nextgen[3]))
    arrow(ax, (nextgen[0], nextgen[1] + nextgen[3] * 0.5), (fit[0] + fit[2] * 0.5, fit[1]),
          color=BLUE, connectionstyle="arc3,rad=0.25", ls="--")
    ax.text(5.2, 1.55, "repeated 1,500 generations (1.2M policies evaluated total)", fontsize=8.0, color=BLUE, style="italic")

    best = box(ax, (7.4, 0.15), 4.5, 0.75, "single best-evolved individual\n$\\rightarrow$ evaluated once, real OOS 2022+ holdout", face="#F0E7ED", edge="#8a3b6b", fontsize=8.6)
    arrow(ax, (nextgen[0] + nextgen[2] * 0.5, nextgen[1]), (best[0] + best[2] * 0.5, best[1] + best[3]),
          color=GREY)

    ax.text(0.3, 1.9, "No day-by-day recurrence:\nposition is a pure function\nof same-day signals only\n(structurally memoryless,\nby construction, for\ntractability at this scale)",
            fontsize=8.0, color=GREY, style="italic", va="top")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "88_ga_architecture_schematic.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: 88_ga_architecture_schematic.png")
