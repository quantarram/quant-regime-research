"""
Demonstrates why "the price prediction chart looks good" is not the same
claim as "there is tradeable alpha here" -- the same illusion this project
already caught once from the other direction (WEAK-flagged short-horizon
instruments look visually flawless because tomorrow's price is always
close to today's, a property of the task, not of the model). The mirror
case: at a long horizon, in a persistently trending market, almost ANY
reasonable forecast -- including the zero-information climatology
baseline -- will visually track the actual price reasonably well on a
price-LEVEL chart, because the chart's vertical scale is dominated by the
multi-year drift, not by the day-to-day precision that actually
determines whether trading the forecast beats simply holding the asset.

For each instrument, plots two panels on the same time axis:
  top:    actual price vs. informed-model predicted price vs. climatology
          predicted price (the "looks good" view)
  bottom: percentage forecast error (predicted/actual - 1) for both lines
          -- stripping out the shared drift reveals whether the informed
          model's errors are actually any tighter than climatology's, which
          the top panel alone cannot show.

Run: python 50_price_vs_error_demonstration.py
Output: pnl_plots/_ERRORVIEW_<TICKER>.png
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
PLOT_DIR = os.path.join(OUT_DIR, "pnl_plots")
HOLDOUT_START = pd.Timestamp("2022-01-01")

decisions = json.load(open(os.path.join(OUT_DIR, "master_model_final_decision.json")))
oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))
PROXY_TICKERS = ("IYR", "VOX")


def get_series(tkr):
    return (prices_proxy[tkr] if tkr in PROXY_TICKERS else prices[tkr]).dropna()


def build_view(tkr):
    dec = decisions[tkr]
    horizon, winner = dec["horizon"], dec["price_based_winner"]
    sub_th = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon)]
    variants_present = sub_th["variant"].unique().tolist()
    # climatology (clim_q*) columns are identical across variants for the same
    # ticker/horizon -- they don't depend on credit/vix features -- so any
    # available variant's rows carry the correct climatology reference. The
    # informed model's own q0.5, however, MUST come from the actual winning
    # variant, not an arbitrary one.
    variant_for_rows = winner if winner in variants_present else ("both" if "both" in variants_present else variants_present[0])
    sub = sub_th[sub_th["variant"] == variant_for_rows].sort_values("date").reset_index(drop=True)
    sub = sub[sub["date"] >= HOLDOUT_START]

    series = get_series(tkr)
    series_pos = {d: i for i, d in enumerate(series.index)}
    sub = sub[sub["date"].isin(series_pos)].copy()
    sub["price_now"] = series.reindex(sub["date"]).values
    target_idx = (sub["date"].map(series_pos) + horizon)
    valid = target_idx < len(series)
    sub = sub[valid.values]
    target_idx = target_idx[valid]
    sub["target_date"] = series.index[target_idx.values]
    sub["actual_target_price"] = series.reindex(sub["target_date"]).values

    sub["pred_price"] = sub["price_now"] * np.exp(sub["q0.5"])
    sub["clim_price"] = sub["price_now"] * np.exp(sub["clim_q0.5"])
    sub["pred_err_pct"] = (sub["pred_price"] / sub["actual_target_price"] - 1) * 100
    sub["clim_err_pct"] = (sub["clim_price"] / sub["actual_target_price"] - 1) * 100
    return sub, horizon, winner


for tkr in ["JPM", "XLE"]:
    sub, horizon, winner = build_view(tkr)
    sub = sub.dropna(subset=["actual_target_price", "pred_price", "clim_price"])

    mae_pred = sub["pred_err_pct"].abs().mean()
    mae_clim = sub["clim_err_pct"].abs().mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(sub["target_date"], sub["actual_target_price"], color="black", lw=1.4, label="Actual price")
    ax1.plot(sub["target_date"], sub["pred_price"], color="tab:red", lw=1.2, label=f"Informed model ({winner}@{horizon}d) predicted price")
    ax1.plot(sub["target_date"], sub["clim_price"], color="tab:blue", lw=1.2, ls="--", label="Climatology predicted price")
    ax1.set_title(f"{tkr}: does 'looks good' mean 'has an edge'? (holdout period)\n"
                   f"Top: price level -- both lines track actual price reasonably. MAE: informed={mae_pred:.2f}%, climatology={mae_clim:.2f}%")
    ax1.set_ylabel("Price")
    ax1.legend(fontsize=9, loc="upper left")

    ax2.plot(sub["target_date"], sub["pred_err_pct"], color="tab:red", lw=1.0, label="Informed model error (%)")
    ax2.plot(sub["target_date"], sub["clim_err_pct"], color="tab:blue", lw=1.0, ls="--", label="Climatology error (%)")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Forecast error (%)")
    ax2.set_title("Bottom: same two forecasts' error, with the shared drift removed -- this is what actually determines tradeable edge")
    ax2.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    safe_tkr = tkr.replace('=', '').replace('^', '')
    fig.savefig(os.path.join(PLOT_DIR, f"_ERRORVIEW_{safe_tkr}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"{tkr}: informed MAE={mae_pred:.2f}%, climatology MAE={mae_clim:.2f}%, "
          f"informed {'beats' if mae_pred < mae_clim else 'does not beat'} climatology on this metric")

print("\nDone.")
