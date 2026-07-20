"""
Final, honest price-prediction plots: replaces 37_price_prediction_model.py.
Uses best_config_selection_holdout.json (config chosen using ONLY the
selection period, dates < 2022-01-01) and the raw OOS predictions already
saved in oos_predictions_all.parquet (38_fss_selection_holdout_split.py) --
no new model fits, this just visualizes what's already been computed
honestly.

The selection period (used to pick the config) is shown muted/pale, since
it is NOT a fair test of that config -- it's the data the config was
chosen to look good on. The holdout period (2022-01-01 onward, never
touched during selection) is shown at full opacity -- that's the genuine,
one-time, honest result.

Run: python 39_price_prediction_honest.py
Output: price_predictions_honest/<TICKER>_price_prediction.png
"""
import pandas as pd
import numpy as np
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUT_DIR, "price_predictions_honest")
os.makedirs(PLOT_DIR, exist_ok=True)
HOLDOUT_START = pd.Timestamp("2022-01-01")
ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]

best_config = json.load(open(os.path.join(OUT_DIR, "best_config_selection_holdout.json")))
oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))
PROXY_TICKERS = ("IYR", "VOX")


def get_series(tkr):
    return (prices_proxy[tkr] if tkr in PROXY_TICKERS else prices[tkr]).dropna()


results_summary = {}
for tkr in sorted(best_config.keys(), key=lambda t: -best_config[t]["holdout_skill"]):
    cfg = best_config[tkr]
    horizon, variant = cfg["horizon"], cfg["variant"]
    sub = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon) & (oos_all["variant"] == variant)].copy()
    if sub.empty:
        print(f"  {tkr}: no matching rows in oos_predictions_all.parquet, skipping")
        continue
    sub = sub.sort_values("date").reset_index(drop=True)

    series = get_series(tkr)
    series_pos = {d: i for i, d in enumerate(series.index)}
    sub["price_now"] = series.reindex(sub["date"]).values
    target_idx = (sub["date"].map(series_pos) + horizon).astype(int)
    sub["target_date"] = series.index[target_idx.values]
    for a in ALPHAS:
        sub[f"pred_price_q{a}"] = sub["price_now"] * np.exp(sub[f"q{a}"])
        sub[f"clim_price_q{a}"] = sub["price_now"] * np.exp(sub[f"clim_q{a}"])

    is_selection = sub["date"] < HOLDOUT_START
    sel, hold = sub[is_selection], sub[~is_selection]
    is_real = cfg["holdout_skill"] > 0

    window_start = sub["target_date"].min() - pd.Timedelta(days=180)
    context_price = series[series.index >= window_start]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(context_price.index, context_price.values, color="black", lw=1.4, label="Actual price", zorder=4)

    for part, alpha_mult, tag in [(sel, 0.35, "selection period, used to pick config"),
                                    (hold, 1.0, "HOLDOUT period, honest result")]:
        if part.empty:
            continue
        ax.plot(part["target_date"], part["clim_price_q0.5"], color="tab:blue", lw=1.2, ls="--",
                 alpha=0.9 * alpha_mult, label=f"Climatology median ({tag})" if part is hold else None,
                 zorder=3)
        ax.plot(part["target_date"], part["pred_price_q0.5"], color="tab:red", lw=1.2,
                 alpha=1.0 * alpha_mult, label=f"Predicted median ({tag})", zorder=5)
        ax.fill_between(part["target_date"], part["pred_price_q0.1"], part["pred_price_q0.9"],
                         color="tab:red", alpha=0.15 * alpha_mult, zorder=2)

    ax.axvline(HOLDOUT_START, color="gray", lw=1.0, ls=":", zorder=1)
    ax.text(HOLDOUT_START, ax.get_ylim()[1], "  holdout starts", fontsize=8, color="gray", va="top")

    verdict = "REAL EDGE beyond climatology" if is_real else "NO real edge -- selection-period result did not hold up"
    ax.set_title(f"{tkr}: best config = {variant} @ {horizon}d  |  "
                 f"selection skill {cfg['selection_skill']:+.4f} -> HOLDOUT skill {cfg['holdout_skill']:+.4f}  |  {verdict}")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    safe_tkr = tkr.replace('=', '').replace('^', '')
    out_path = os.path.join(PLOT_DIR, f"{safe_tkr}_price_prediction.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    results_summary[tkr] = {**cfg, "real_edge": bool(is_real)}
    print(f"  {tkr}: {variant} @ {horizon}d, holdout_skill={cfg['holdout_skill']:+.4f} ({'REAL' if is_real else 'not real'})")

with open(os.path.join(OUT_DIR, "results_price_predictions_honest.json"), "w") as f:
    json.dump(results_summary, f, indent=2, default=float)
print(f"\nSaved {len(results_summary)} plots to {PLOT_DIR}/")
print("Done.")
