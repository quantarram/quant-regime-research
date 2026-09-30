"""
elnino_sugar_original_figures_updated.py
========================================
Regenerates the two "narrative" figures from the original working draft
(sugar price history with El Nino windows shaded; current signal-status
panel) using the corrected, current-vintage data: seven El Nino years
(2004 dropped -- see elnino_sugar_analysis.py), the full FRED+SB=F price
series through September 2026, and 2026's actual realized outcome in
place of the original forward-looking "odds of a rise" framing (the
event has now happened, so the panel reports what did happen, not what
might).

Run: python elnino_sugar_original_figures_updated.py
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from elnino_sugar_analysis import load_oni, load_sugar_price, forward_return, ONI_THRESHOLD

FLAGGED_YEARS = [1991, 1997, 2002, 2009, 2015, 2023, 2026]

if __name__ == "__main__":
    price, fred_cutoff, level_corr, ret_corr = load_sugar_price()
    d = json.load(open("elnino_sugar_results.json"))

    # ---------------------------------------------------------------
    # Figure A: full price history, corrected El Nino windows shaded
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6.5))
    monthly = price.resample("MS").mean() if not isinstance(price.index.freq, type(None)) else price
    ax.plot(price.index, price.values, color="#1B2A4A", lw=1.4)

    for y in FLAGGED_YEARS:
        start = pd.Timestamp(y, 6, 1)
        end = pd.Timestamp(y, 9, 30)
        ax.axvspan(start, end, color="#E8998D", alpha=0.45,
                   label="El Nino monsoon (JJAS), ONI >= 0.5" if y == FLAGGED_YEARS[0] else None)

    def annotate(year, month, text, xytext, color="#444"):
        x = pd.Timestamp(year, month, 1)
        y = price.reindex([x]).iloc[0] if x in price.index else price[price.index.get_indexer([x], method="nearest")[0]]
        ax.annotate(text, xy=(x, y), xytext=xytext, fontsize=9.5, ha="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.9),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

    annotate(1997, 8, "Super\nEl Nino", (pd.Timestamp(1997, 8, 1), 24))
    annotate(2009, 10, "India drought,\n30-yr price high", (pd.Timestamp(2012, 6, 1), 24))
    annotate(2023, 8, "Driest August\nin a century", (pd.Timestamp(2023, 8, 1), 32))
    annotate(2026, 9, "2026: +29% by Sept\n(June flag, 3mo\nforward, realized)", (pd.Timestamp(2024, 6, 1), 8), color="#2E7D4F")

    ax.set_ylabel("Cents / lb (IMF Sugar No. 11, FRED+SB=F)")
    ax.set_title("Global Sugar Price, 1992-2026 -- El Nino Monsoon Windows Shaded (corrected: 7 years, 2004 dropped)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 36)
    ax.set_xlim(pd.Timestamp(1991, 1, 1), pd.Timestamp(2028, 1, 1))
    fig.tight_layout()
    fig.savefig("elnino_sugar_price_history_plot.png", dpi=140)
    plt.close(fig)
    print("Saved: elnino_sugar_price_history_plot.png")

    # ---------------------------------------------------------------
    # Figure B: season outcome panel (realized, not forward-looking)
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.suptitle("2026 Season Outcome -- Late September 2026 (Resolved, Not a Live Signal)",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.barh(["2026\n(JJA final)"], [1.8], color="#B0492F", height=0.5)
    ax.axvline(ONI_THRESHOLD, color="#1B2A4A", ls="--", lw=1.5)
    ax.text(ONI_THRESHOLD / 2.2, 0.92, "NOAA El Nino\nthreshold (0.5C)", fontsize=8.5, color="#1B2A4A",
            ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.text(1.8 + 0.05, 0, "+1.8C", fontsize=11, fontweight="bold", va="center")
    ax.set_xlim(0, 2.2)
    ax.set_xlabel("JJA Nino 3.4 Index (C), current ONI v6 vintage")
    ax.set_title("El Nino Status\n(confirmed, not provisional)", fontsize=11, fontweight="bold")

    ax2 = axes[1]
    bars = ax2.bar(["Start\n(24 Jun)", "Season-end\n(~Sept)", "IMD full-season\nforecast"],
                    [-42, -13.5, -10], color=["#B0492F", "#c96a3a", "#2E7D4F"])
    for b, v in zip(bars, [-42, -13.5, -10]):
        ax2.text(b.get_x() + b.get_width() / 2, v - 2.5, f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("% departure from LPA")
    ax2.set_title("Monsoon Rainfall Deficit\n(realized: narrowed close to forecast)", fontsize=11, fontweight="bold")
    ax2.set_ylim(-48, 4)

    ax3 = axes[2]
    # each year's average 3-month-forward return across its flagged monsoon months --
    # same units throughout (% price return), not mixed with a hit-rate percentage
    per_year_avg = {1997: 2.2, 2002: 21.3, 2009: 20.6, 2015: 20.0, 2023: 3.9, 2026: 29.3}
    years = list(per_year_avg.keys())
    vals = list(per_year_avg.values())
    colors3 = ["#4a7a9d"] * (len(years) - 1) + ["#2E7D4F"]
    bars = ax3.bar([str(y) for y in years], vals, color=colors3)
    for b, v in zip(bars, vals):
        ax3.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:+.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax3.axhline(0, color="#444", lw=0.8)
    ax3.set_ylabel("Avg. 3-month-forward sugar price return (%)")
    ax3.set_title("Each Flagged Year's Real Move\n(2026, in green, is realized -- not a forecast)", fontsize=11, fontweight="bold")
    ax3.set_ylim(-5, 34)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("elnino_sugar_current_status_plot.png", dpi=140)
    plt.close(fig)
    print("Saved: elnino_sugar_current_status_plot.png")
