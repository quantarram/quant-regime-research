"""
Post-processing / bias correction of the master model's raw point forecast
(q0.5, the winning variant per instrument -- clim_q0.5 for climatology
winners), using the (predicted, actual) pairs already available. Standard
model-output-statistics (MOS) techniques from numerical weather prediction,
ported to this forecast: the raw model can have a systematic, correctable
bias even when its instantaneous errors look like noise, and if so, a
calibration mapping fit on past (predicted, actual) pairs can remove it
without touching the underlying model at all.

CRITICAL: fitting a correction on the same data used to report the
correction's benefit would be exactly the selection-bias problem already
caught and fixed once this session (see feedback-selection-bias-caught).
So the existing HOLDOUT period (>=2022-01-01, already never touched during
model/config selection) is split AGAIN, chronologically:
  POSTPROC_CALIB : 2022-01-01 to 2024-01-01 -- fit each correction here
  FINAL_VERIFY   : 2024-01-01 onward         -- evaluate ONLY here, on
                   predictions the correction never saw

All three corrections operate on the LOG-RETURN scale (q0.5 vs y_true,
both already on that scale in oos_predictions_all.parquet) -- the natural,
roughly-stationary scale for this kind of calibration, not the raw
(non-stationary) price level.

Three methods:
  1. OLS (linear MOS)   : actual ~= a + b * raw_predicted, fit by ordinary
                          least squares on the calibration period.
  2. Moment/PDF matching: corrected = actual_mean_calib +
                          (raw - raw_mean_calib) * (actual_std_calib /
                          raw_std_calib). Matches the corrected series'
                          first two moments to the observed distribution
                          without regression-toward-the-mean attenuation
                          (unlike OLS, whose slope is damped by the
                          correlation) -- a distinct correction principle,
                          not a rephrasing of method 1.
  3. Quantile mapping   : empirical-CDF-to-empirical-CDF mapping between
     (with frequency       the calibration period's predicted and actual
      correction)          distributions (Piani et al. 2010-style), with
                          linear tail extrapolation beyond the calibration
                          range so verification-period predictions outside
                          the calibration data's observed range don't get
                          clipped or degenerate (the "frequency
                          correction").

Evaluated by MAE (log-return scale) and MAPE (price scale, converted back
via price(t)*exp(corrected_return)) on the verification period only,
against the RAW (uncorrected) forecast and climatology.

Run: python 51_post_processing.py
Output: post_processing_results.json, pnl_plots/_POSTPROC_SUMMARY.png,
        pnl_plots/_POSTPROC_<TICKER>.png (for the two/three most improved)
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
MIN_CALIB_ROWS = 60

decisions = json.load(open(os.path.join(OUT_DIR, "master_model_final_decision.json")))
oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))
PROXY_TICKERS = ("IYR", "VOX")


def get_series(tkr):
    return (prices_proxy[tkr] if tkr in PROXY_TICKERS else prices[tkr]).dropna()


def ols_correct(raw_calib, actual_calib, raw_verify):
    X = np.column_stack([np.ones_like(raw_calib), raw_calib])
    coef, *_ = np.linalg.lstsq(X, actual_calib, rcond=None)
    a, b = coef
    return a + b * raw_verify, (float(a), float(b))


def moment_match_correct(raw_calib, actual_calib, raw_verify):
    raw_mean, raw_std = raw_calib.mean(), raw_calib.std()
    act_mean, act_std = actual_calib.mean(), actual_calib.std()
    raw_std = max(raw_std, 1e-8)
    corrected = act_mean + (raw_verify - raw_mean) * (act_std / raw_std)
    return corrected, (float(raw_mean), float(raw_std), float(act_mean), float(act_std))


def quantile_map_correct(raw_calib, actual_calib, raw_verify, n_quantiles=50):
    qs = np.linspace(0, 1, n_quantiles)
    raw_q = np.quantile(raw_calib, qs)
    act_q = np.quantile(actual_calib, qs)
    raw_q_u, idx = np.unique(raw_q, return_index=True)
    act_q_u = act_q[idx]
    corrected = np.interp(raw_verify, raw_q_u, act_q_u)
    # linear tail extrapolation beyond the calibration range (the "frequency correction" --
    # otherwise np.interp would clip out-of-range verification predictions to the calibration
    # period's min/max, silently mishandling how often extreme predictions occur post-calibration)
    if len(raw_q_u) >= 2:
        lo_slope = (act_q_u[1] - act_q_u[0]) / max(raw_q_u[1] - raw_q_u[0], 1e-12)
        hi_slope = (act_q_u[-1] - act_q_u[-2]) / max(raw_q_u[-1] - raw_q_u[-2], 1e-12)
        below = raw_verify < raw_q_u[0]
        above = raw_verify > raw_q_u[-1]
        corrected = np.where(below, act_q_u[0] + (raw_verify - raw_q_u[0]) * lo_slope, corrected)
        corrected = np.where(above, act_q_u[-1] + (raw_verify - raw_q_u[-1]) * hi_slope, corrected)
    return corrected, None


results = {}
for tkr in sorted(decisions.keys()):
    dec = decisions[tkr]
    horizon, winner = dec["horizon"], dec["price_based_winner"]
    sub_th = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon)]
    variants_present = sub_th["variant"].unique().tolist()
    variant_for_rows = winner if winner in variants_present else ("both" if "both" in variants_present else variants_present[0])
    sub = sub_th[sub_th["variant"] == variant_for_rows].sort_values("date").reset_index(drop=True)
    pred_col = "clim_q0.5" if winner == "climatology" else "q0.5"
    sub = sub[sub["date"] >= HOLDOUT_START][["date", "y_true", pred_col, "clim_q0.5"]].dropna()
    sub.columns = ["date", "y_true", "raw_pred", "clim_pred"]

    calib = sub[sub["date"] < POSTPROC_SPLIT]
    verify = sub[sub["date"] >= POSTPROC_SPLIT]
    if len(calib) < MIN_CALIB_ROWS or len(verify) < MIN_CALIB_ROWS:
        continue

    # Select WHICH correction method to use via an internal split inside the
    # calibration period only -- never touching verification for method
    # selection. Fit each candidate on the first 70% of calib, score on the
    # last 30%, pick the winner, THEN refit that winner on the full calib
    # period before the one-time application to verification.
    split_i = int(len(calib) * 0.7)
    calib_train, calib_val = calib.iloc[:split_i], calib.iloc[split_i:]
    rt, at = calib_train["raw_pred"].values, calib_train["y_true"].values
    rv, av = calib_val["raw_pred"].values, calib_val["y_true"].values
    candidates_val = {
        "raw": float(np.mean(np.abs(rv - av))),
        "ols": float(np.mean(np.abs(ols_correct(rt, at, rv)[0] - av))),
        "moment_match": float(np.mean(np.abs(moment_match_correct(rt, at, rv)[0] - av))),
        "quantile_map": float(np.mean(np.abs(quantile_map_correct(rt, at, rv)[0] - av))),
    }
    selected_method = min(candidates_val, key=candidates_val.get)

    raw_calib, actual_calib = calib["raw_pred"].values, calib["y_true"].values
    raw_verify, actual_verify = verify["raw_pred"].values, verify["y_true"].values

    ols_pred, ols_params = ols_correct(raw_calib, actual_calib, raw_verify)
    mm_pred, mm_params = moment_match_correct(raw_calib, actual_calib, raw_verify)
    qm_pred, _ = quantile_map_correct(raw_calib, actual_calib, raw_verify)
    selected_pred = {"raw": raw_verify, "ols": ols_pred, "moment_match": mm_pred, "quantile_map": qm_pred}[selected_method]

    mae = lambda p: float(np.mean(np.abs(p - actual_verify)))
    series = get_series(tkr)
    series_pos = {d: i for i, d in enumerate(series.index)}
    verify_dates = verify["date"].values
    price_now = series.reindex(verify_dates).values
    target_idx = np.array([series_pos.get(pd.Timestamp(d), None) for d in verify_dates])
    valid_mask = np.array([t is not None and t + horizon < len(series) for t in target_idx])

    def mape_price(pred_ret):
        idxs = target_idx[valid_mask]
        tgt_price = series.values[np.array(idxs, dtype=int) + horizon]
        pred_price = price_now[valid_mask] * np.exp(pred_ret[valid_mask])
        return float(np.mean(np.abs(pred_price / tgt_price - 1)) * 100)

    entry = {
        "horizon": horizon, "winner": winner,
        "n_calib": int(len(calib)), "n_verify": int(len(verify)),
        "selected_method": selected_method,  # chosen via internal calib_train/calib_val split, NOT verification
        "internal_val_mae": candidates_val,
        "mae_return_raw": mae(raw_verify), "mae_return_climatology": mae(verify["clim_pred"].values),
        "mae_return_ols": mae(ols_pred), "mae_return_moment_match": mae(mm_pred), "mae_return_quantile_map": mae(qm_pred),
        "mape_price_raw": mape_price(raw_verify), "mape_price_climatology": mape_price(verify["clim_pred"].values),
        "mape_price_ols": mape_price(ols_pred), "mape_price_moment_match": mape_price(mm_pred), "mape_price_quantile_map": mape_price(qm_pred),
        "mape_price_selected": mape_price(selected_pred),
        "ols_params": {"a": ols_params[0], "b": ols_params[1]},
    }
    entry["best_method"] = entry["selected_method"]
    results[tkr] = entry

with open(os.path.join(OUT_DIR, "post_processing_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)

print(f"=== Post-processing, {len(results)} instruments, calib<{POSTPROC_SPLIT.date()}<=verify, method chosen by internal calib split, verify-only MAPE (price %) ===")
n_improved = 0
for tkr in sorted(results, key=lambda t: results[t]["mape_price_raw"]):
    r = results[tkr]
    improved = r["selected_method"] != "raw" and r["mape_price_selected"] < r["mape_price_raw"] - 0.01
    n_improved += improved
    print(f"  {tkr} ({r['winner']}@{r['horizon']}d, n_calib={r['n_calib']}, n_verify={r['n_verify']}): "
          f"raw={r['mape_price_raw']:.2f}% clim={r['mape_price_climatology']:.2f}% "
          f"[ols={r['mape_price_ols']:.2f}% moment={r['mape_price_moment_match']:.2f}% qmap={r['mape_price_quantile_map']:.2f}%] "
          f"-> selected={r['selected_method']} (chosen in-calib) giving {r['mape_price_selected']:.2f}%" + (" [IMPROVED]" if improved else ""))
print(f"\n{n_improved}/{len(results)} instruments improved on raw MAPE by >=0.01pp with the internally-selected post-processing method")

# summary plot: raw vs internally-selected post-processed MAPE, all instruments
tickers_sorted = sorted(results, key=lambda t: results[t]["mape_price_raw"] - results[t]["mape_price_selected"])
fig, ax = plt.subplots(figsize=(9, 8))
y_pos = np.arange(len(tickers_sorted))
raw_vals = [results[t]["mape_price_raw"] for t in tickers_sorted]
best_vals = [results[t]["mape_price_selected"] for t in tickers_sorted]
ax.barh(y_pos - 0.18, raw_vals, height=0.35, color="#9aa1ad", label="Raw (uncorrected)")
ax.barh(y_pos + 0.18, best_vals, height=0.35,
        color=["#2f8a4e" if b < r - 0.01 else "#b0492f" for r, b in zip(raw_vals, best_vals)],
        label="Selected post-processing (chosen in-calibration)")
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{t} ({results[t]['selected_method']})" for t in tickers_sorted], fontsize=9)
ax.set_xlabel("Verification-period MAPE (price, %) -- lower is better")
ax.set_title(f"Post-processing: raw vs. corrected MAPE, verification period ({POSTPROC_SPLIT.date()} onward)\n"
             f"Method chosen via internal calibration-period split (not verification) -- {n_improved}/{len(results)} improved")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "_POSTPROC_SUMMARY.png"), dpi=115, bbox_inches="tight")
plt.close(fig)
print("\nDone.")
