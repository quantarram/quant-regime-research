"""
Two new, higher-impact figures for Paper 15, built after the paper's
content was already finalized and verified -- purely presentational,
no new claims, no new numbers beyond what Tables 1-2 already report.

  p15_graphical_abstract.png   -- unnumbered hero image, placed right after
                                   the title (standard graphical-abstract
                                   convention: not part of the numbered
                                   figure sequence).
  p15_victory_margins.png      -- Figure 9: a single ranked, diverging bar
                                   chart of Table 2's real result (climatology
                                   vs. the calibrated downscaler, own
                                   uncertainty), replacing "read 12 small
                                   subplots" with "read one chart."
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

INK = "#1A1A1A"
WALL = "#2C2C2C"
GRAY = "#8E8E8E"
BLUE = "#1B4F72"
AMBER = "#D68910"
BG = "#FCFBF9"


# ---------------------------------------------------------------------------
# Graphical abstract
# ---------------------------------------------------------------------------
def make_graphical_abstract():
    fig = plt.figure(figsize=(14, 8.4))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.4)
    ax.axis("off")
    ax.set_facecolor(BG)

    # Title block
    ax.text(7, 7.85, "THE CEILING HOLDS, AND SO DOES CLIMATOLOGY", ha="center", va="center",
            fontsize=21, fontweight="bold", color=INK, family="sans-serif")
    ax.text(7, 7.42, "Two papers, five levers, one measured predictability limit (τ*)",
            ha="center", va="center", fontsize=12.5, color="#555555", style="italic")

    # The wall -- levers now anchored at a left-aligned label column (x=3.3) with room
    # to spare, arrows running from there to the wall. Right-aligning long labels
    # against a small x anchor (the original design) pushed text off the left edge
    # of the canvas -- fixed by left-aligning against a safely-interior anchor instead.
    wall_x = 9.9
    label_x = 0.3
    arrow_start_x = 3.6
    ax.add_patch(Rectangle((wall_x, 2.6), 0.16, 4.2, facecolor=WALL, edgecolor="none", zorder=3))
    ax.text(wall_x + 0.08, 7.05, "τ*", ha="center", fontsize=15, fontweight="bold", color=WALL)
    ax.text(wall_x + 0.08, 6.75, "measured limit", ha="center", fontsize=8, color="#555555")

    levers = ["Architecture", "Depth", "Training window size", "Loss function"]
    ys = [6.35, 5.55, 4.75, 3.95]
    for name, y in zip(levers, ys):
        ax.annotate("", xy=(wall_x - 0.1, y), xytext=(arrow_start_x, y),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2.6, mutation_scale=16))
        ax.text(label_x, y, name, ha="left", va="center", fontsize=12, color="#333333")
    # 5th lever, distinct color, distinct verdict
    y5 = 3.15
    ax.annotate("", xy=(wall_x - 0.1, y5), xytext=(arrow_start_x, y5),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=3.0, mutation_scale=18))
    ax.text(label_x, y5, "Uncertainty quantification", ha="left", va="center", fontsize=12,
            color=BLUE, fontweight="bold")

    ax.text(label_x, 6.9, "TESTED AGAINST THE WALL", ha="left", va="center", fontsize=9.5,
            color="#999999", fontweight="bold")

    ax.text(wall_x + 0.5, 6.35, "no improvement", fontsize=9, color="#777777", va="center")
    ax.text(wall_x + 0.5, 5.55, "no improvement", fontsize=9, color="#777777", va="center")
    ax.text(wall_x + 0.5, 4.75, "degrades past it", fontsize=9, color="#777777", va="center")
    ax.text(wall_x + 0.5, 3.95, "no improvement", fontsize=9, color="#777777", va="center")
    ax.text(wall_x + 0.5, 3.15, "wins a real scoring rule*", fontsize=9.5, color=BLUE,
            fontweight="bold", va="center")

    # Twist callout box. zorder MUST be lower than the text drawn on top of it --
    # a first version set the box to zorder=4 with default-zorder text underneath,
    # which silently hid every word inside the box. Caught by looking at the actual
    # rendered image, not assumed from the code.
    box = FancyBboxPatch((0.6, 0.35), 12.8, 2.15, boxstyle="round,pad=0.08,rounding_size=0.12",
                          linewidth=1.6, edgecolor=AMBER, facecolor="#FFF8EC", zorder=2)
    ax.add_patch(box)
    ax.text(1.1, 2.05, "* THE TWIST", fontsize=12.5, fontweight="bold", color=AMBER, va="center", zorder=5)
    ax.text(1.1, 1.55,
            "Give every architecture its own honest uncertainty — no new calibration — and the win",
            fontsize=11, color="#333333", va="center", zorder=5)
    ax.text(1.1, 1.15,
            "holds against every sophisticated model, but climatology's own unmodeled real return",
            fontsize=11, color="#333333", va="center", zorder=5)
    ax.text(1.1, 0.75,
            "distribution beats even the calibrated downscaler outright, on the majority of instruments.",
            fontsize=11, color="#333333", va="center", zorder=5)

    ax.text(11.9, 1.35, "7", fontsize=42, fontweight="bold", color=GRAY, ha="center", va="center", zorder=5)
    ax.text(11.9, 0.72, "of 12 instruments:\nclimatology wins outright", fontsize=8.3, ha="center",
            va="center", color="#555555", zorder=5)

    fig.savefig(os.path.join(OUT_DIR, "p15_graphical_abstract.png"), dpi=170, facecolor=BG)
    plt.close(fig)
    print("Saved p15_graphical_abstract.png")


# ---------------------------------------------------------------------------
# Figure 9: victory margins, ranked
# ---------------------------------------------------------------------------
def make_victory_margins():
    # (instrument, climatology_own, downscaler_own) -- from Table 2, exact
    data = [
        ("GLD", 4.339, 4.613), ("JPM", 22.227, 25.749), ("AAPL", 31.118, 37.215),
        ("XLK", 13.254, 12.672), ("EUR/USD", 1.526, 1.679), ("IWM", 7.080, 6.439),
        ("MSFT", 35.754, 36.388), ("QQQ", 8.939, 8.421), ("SPY", 8.160, 7.615),
        ("XLE", 9.384, 9.590), ("XLF", 7.920, 8.023), ("XOM", 18.565, 16.270),
    ]
    rows = []
    for tkr, clim, dsc in data:
        winner_val, loser_val = (clim, dsc) if clim < dsc else (dsc, clim)
        margin = 100 * (loser_val - winner_val) / loser_val
        downscaler_wins = dsc < clim
        rows.append((tkr, margin, downscaler_wins))
    rows.sort(key=lambda r: (r[2], r[1]))  # climatology wins first (grouped), ascending margin within group

    fig, ax = plt.subplots(figsize=(11, 7.2))
    fig.patch.set_facecolor("white")
    ys = np.arange(len(rows))
    colors = [BLUE if dw else GRAY for _, _, dw in rows]
    signed = [m if dw else -m for _, m, dw in rows]
    bars = ax.barh(ys, signed, color=colors, height=0.62, zorder=3)
    for y, (tkr, m, dw) in zip(ys, rows):
        x = m if dw else -m
        label = f"{tkr}  (+{m:.0f}%)" if dw else f"{tkr}  (+{m:.0f}%)"
        ha = "left" if dw else "right"
        offset = 1.0 if dw else -1.0
        ax.text(x + offset, y, label, va="center", ha=ha, fontsize=10, color="#222222")

    ax.axvline(0, color="#333333", lw=1.2)
    ax.set_yticks([])
    ax.set_xlabel("margin of victory (%), winner over loser, own-uncertainty CRPS", fontsize=10.5)
    ax.set_xlim(-45, 45)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    n_clim = sum(1 for _, _, dw in rows if not dw)
    n_dsc = sum(1 for _, _, dw in rows if dw)
    ax.text(-22, len(rows) - 0.2, f"CLIMATOLOGY WINS ({n_clim})", fontsize=11.5, fontweight="bold",
            color=GRAY, ha="center")
    ax.text(22, len(rows) - 0.2, f"DOWNSCALER WINS ({n_dsc})", fontsize=11.5, fontweight="bold",
            color=BLUE, ha="center")

    ax.set_title("Climatology's own real return distribution vs. the calibrated downscaler:\n"
                 "who actually wins, instrument by instrument (Table 2)", fontsize=13, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "p15_victory_margins.png"), dpi=160)
    plt.close(fig)
    print("Saved p15_victory_margins.png")


if __name__ == "__main__":
    make_graphical_abstract()
    make_victory_margins()
