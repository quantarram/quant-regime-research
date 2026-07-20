"""
Detail view of JPM's post-processing correction -- the one instrument
whose improvement survived proper method selection (chosen via an internal
calibration-period split, not by peeking at verification). Shows actual
price vs. raw-model predicted price vs. OLS-corrected predicted price on
the verification period (2024-01-01 onward, never touched while fitting
the correction), plus the error view with the shared drift removed.

Run: python 52_jpm_postproc_detail.py
Output: pnl_plots/_POSTPROC_JPM_detail.png
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
POSTPROC_SPLIT = pd.Timestamp("2024-01-01")
TKR = "JPM"

decisions = json.load(open(os.path.join(OUT_DIR, "master_model_final_decision.json")))
postproc = json.load(open(os.path.join(OUT_DIR, "post_processing_results.json")))
oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))

dec = decisions[TKR]
horizon, winner = dec["horizon"], dec["price_based_winner"]
a, b = postproc[TKR]["ols_params"]["a"], postproc[TKR]["ols_params"]["b"]

sub_th = oos_all[(oos_all["ticker"] == TKR) & (oos_all["horizon"] == horizon)]
sub = sub_th[sub_th["variant"] == winner].sort_values("date").reset_index(drop=True)
sub = sub[sub["date"] >= POSTPROC_SPLIT].copy()

series = prices[TKR].dropna()
series_pos = {d: i for i, d in enumerate(series.index)}
sub = sub[sub["date"].isin(series_pos)].copy()
sub["price_now"] = series.reindex(sub["date"]).values
target_idx = sub["date"].map(series_pos) + horizon
valid = target_idx < len(series)
sub = sub[valid.values]
target_idx = target_idx[valid]
sub["target_date"] = series.index[target_idx.values]
sub["actual_target_price"] = series.reindex(sub["target_date"]).values

sub["raw_pred_ret"] = sub["q0.5"]
sub["corrected_pred_ret"] = a + b * sub["q0.5"]
sub["raw_price"] = sub["price_now"] * np.exp(sub["raw_pred_ret"])
sub["corrected_price"] = sub["price_now"] * np.exp(sub["corrected_pred_ret"])
sub["raw_err_pct"] = (sub["raw_price"] / sub["actual_target_price"] - 1) * 100
sub["corrected_err_pct"] = (sub["corrected_price"] / sub["actual_target_price"] - 1) * 100

mae_raw = sub["raw_err_pct"].abs().mean()
mae_corr = sub["corrected_err_pct"].abs().mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
ax1.plot(sub["target_date"], sub["actual_target_price"], color="black", lw=1.4, label="Actual price")
ax1.plot(sub["target_date"], sub["raw_price"], color="tab:red", lw=1.1, ls="--", label=f"Raw model prediction ({winner}@{horizon}d)")
ax1.plot(sub["target_date"], sub["corrected_price"], color="tab:green", lw=1.3, label=f"OLS-corrected prediction (a={a:+.2f}, b={b:.2f})")
ax1.set_title(f"JPM post-processing: verification period only ({POSTPROC_SPLIT.date()} onward, correction never saw this data)\n"
              f"MAE: raw={mae_raw:.2f}%, corrected={mae_corr:.2f}% -- correction fit on 2022-2024, applied once here")
ax1.set_ylabel("Price")
ax1.legend(fontsize=9, loc="upper left")

ax2.plot(sub["target_date"], sub["raw_err_pct"], color="tab:red", lw=1.0, ls="--", label="Raw error (%)")
ax2.plot(sub["target_date"], sub["corrected_err_pct"], color="tab:green", lw=1.2, label="Corrected error (%)")
ax2.axhline(0, color="black", lw=0.8)
ax2.set_ylabel("Forecast error (%)")
ax2.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "_POSTPROC_JPM_detail.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"JPM verification-period MAE: raw={mae_raw:.2f}%, corrected={mae_corr:.2f}%")
print("Done.")
