"""
Rebuilds the FSS threshold-grid gallery (168 plots: climatology/credit_only/
vix_only/both x 4 windows x 6 thresholds, per instrument per horizon) using
ONLY the genuine holdout period (dates >= 2022-01-01), matching the same
fix applied to the price-prediction gallery. No new model fits -- reuses
the raw OOS predictions already saved in oos_predictions_all.parquet
(38_fss_selection_holdout_split.py); this is pure re-aggregation on the
holdout slice.

Coverage note: BTC-USD and XLRE have no valid pre-2022 selection period at
all (their whole usable history falls after the cutoff), so they were never
saved to oos_predictions_all.parquet and are absent here too -- same
exclusion as the price-prediction gallery, for the same reason.

Climatology is taken from the "both" variant's own row-set for each
(ticker, horizon) -- the most feature-complete variant, closest to a shared
reference across the three model lines. credit_only and vix_only's OWN row
sets can differ slightly from "both" (they don't require the other macro
predictor's columns to be non-null), so the three model lines are not
scored on byte-identical dates -- a minor, disclosed limitation, not
silently glossed over.

Run: python 40_fss_grid_holdout_only.py
Output: fss_plots_holdout/<TICKER>_<horizon>d.png
"""
import pandas as pd
import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loss_functions import fss_from_quantiles

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUT_DIR, "fss_plots_holdout")
os.makedirs(PLOT_DIR, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2022-01-01")
ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]
WINDOWS = [21, 63, 126, 252]
UPPER_THRESHOLDS = [0.05, 0.075, 0.10]
LOWER_THRESHOLDS = [-0.05, -0.075, -0.10]
WINDOW_STYLES = {21: "-", 63: "--", 126: ":", 252: "-."}
MODEL_COLORS = {"climatology": "gray", "credit_only": "tab:green", "vix_only": "tab:blue", "both": "tab:red"}
MODEL_ORDER = ["climatology", "credit_only", "vix_only", "both"]

oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
oos_all = oos_all[oos_all["date"] >= HOLDOUT_START].copy()
print(f"Loaded {len(oos_all)} holdout rows across {oos_all[['ticker','horizon','variant']].drop_duplicates().shape[0]} "
      f"(ticker, horizon, variant) combinations")


def fss_grid(sub, prefix):
    y_true = sub["y_true"].values
    quantile_preds = {a: sub[f"{prefix}{a}"].values for a in ALPHAS}
    grid = {}
    for w in WINDOWS:
        grid[w] = {"above": {}, "below": {}}
        for thr in UPPER_THRESHOLDS:
            grid[w]["above"][thr] = fss_from_quantiles(y_true, quantile_preds, ALPHAS, thr, direction="above", window=w, min_periods=15)
        for thr in LOWER_THRESHOLDS:
            grid[w]["below"][thr] = fss_from_quantiles(y_true, quantile_preds, ALPHAS, thr, direction="below", window=w, min_periods=15)
    return grid


def grid_to_xy(grid):
    out = {}
    for w in WINDOWS:
        below, above = grid[w]["below"], grid[w]["above"]
        xs, ys = [], []
        for k in sorted(LOWER_THRESHOLDS):
            xs.append(f"{k}\n(down)")
            ys.append(below[k])
        for k in sorted(UPPER_THRESHOLDS):
            xs.append(f"{k}\n(up)")
            ys.append(above[k])
        out[w] = (xs, ys)
    return out


n_saved = 0
combos = oos_all[["ticker", "horizon"]].drop_duplicates().sort_values(["ticker", "horizon"])
for _, row in combos.iterrows():
    tkr, horizon = row["ticker"], int(row["horizon"])
    sub_th = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon)]
    variants_present = sub_th["variant"].unique().tolist()
    if len(sub_th) < 100:
        continue

    grids = {}
    clim_source = sub_th[sub_th["variant"] == "both"] if "both" in variants_present else sub_th
    if len(clim_source) >= 50:
        grids["climatology"] = fss_grid(clim_source, "clim_q")
    for variant in ["credit_only", "vix_only", "both"]:
        vsub = sub_th[sub_th["variant"] == variant]
        if len(vsub) < 50:
            continue
        grids[variant] = fss_grid(vsub, "q")

    if "climatology" not in grids or len(grids) < 2:
        continue

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for model in MODEL_ORDER:
        if model not in grids:
            continue
        xy_by_window = grid_to_xy(grids[model])
        color = MODEL_COLORS[model]
        for w, (xs, ys) in xy_by_window.items():
            ax.plot(xs, ys, color=color, linestyle=WINDOW_STYLES.get(w, "-"), marker="o", markersize=3,
                    label=f"{model}, window={w}d", alpha=0.85)
    n_hold = len(sub_th[sub_th["variant"] == ("both" if "both" in variants_present else variants_present[0])])
    ax.set_xlabel(f"Threshold ({horizon}d log-return magnitude)")
    ax.set_ylabel("FSS")
    ax.set_title(f"{tkr} @ {horizon}d horizon -- HOLDOUT PERIOD ONLY (2022-01-01 onward, n~{n_hold}), "
                 f"genuinely out-of-sample")
    ax.axhline(0.5, color="black", lw=0.5, ls=":", alpha=0.4)
    ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    safe_tkr = tkr.replace('=', '').replace('^', '')
    out_path = os.path.join(PLOT_DIR, f"{safe_tkr}_{horizon}d.png")
    fig.savefig(out_path, dpi=58, bbox_inches="tight")
    plt.close(fig)
    n_saved += 1

print(f"Saved {n_saved} holdout-only FSS grid plots to {PLOT_DIR}/")
print("Done.")
