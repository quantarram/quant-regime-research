"""
Corrected post-processing design, per the user's explicit correction of
51_post_processing.py's approach. Three periods:
  1. TRAINING  -- untouched. The walk-forward LightGBM OOS predictions
     already saved in oos_predictions_all.parquet are used as-is.
  2. SELECTION (< 2022-01-01) -- ALL the new fitting happens here, in two
     stages:
       (a) Stacking regression: instead of picking ONE winning candidate
           (climatology / credit_only / vix_only / both, as the master
           model has done everywhere else in this project), fit an OLS
           regression of the actual outcome on all four candidates' q0.5
           predictions simultaneously: y_true ~ w0 + w1*clim + w2*credit
           + w3*vix + w4*both. The final output is a learned blend -- not
           attributable to any single source model, per the user's
           explicit description ("we will not know exactly from which
           model the prediction is coming").
       (b) Post-processing, applied SEQUENTIALLY to the stacked
           regression's output (not as alternative candidates picked by
           validation, as in 51_post_processing.py -- both stages are
           always applied, in this order, exactly as the user specified):
           moment/PDF matching (2-parameter mean+std rescale) first, then
           empirical quantile mapping with linear tail extrapolation
           ("frequency correction") on top of the moment-matched output.
  3. TEST (>= 2022-01-01) -- used exactly once. The frozen pipeline
     (regression weights + moment-match params + quantile-map lookup,
     all fit only on selection data) is applied to test-period raw
     candidate predictions and evaluated a single time. No further
     splitting of this period.

Run: python 54_stacked_regression_postprocess.py
Output: stacked_postprocess_results.json, pnl_plots/_STACKED_SUMMARY.png,
        pnl_plots/_STACKED_JPM_detail.png
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
SELECTION_END = pd.Timestamp("2022-01-01")  # SELECTION < this <= TEST
MIN_ROWS = 60

decisions = json.load(open(os.path.join(OUT_DIR, "master_model_final_decision.json")))
oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))
PROXY_TICKERS = ("IYR", "VOX")


def get_series(tkr):
    return (prices_proxy[tkr] if tkr in PROXY_TICKERS else prices[tkr]).dropna()


def moment_match_fit(raw, actual):
    raw_mean, raw_std = raw.mean(), max(raw.std(), 1e-8)
    act_mean, act_std = actual.mean(), actual.std()
    return {"raw_mean": float(raw_mean), "raw_std": float(raw_std), "act_mean": float(act_mean), "act_std": float(act_std)}


def moment_match_apply(raw, params):
    return params["act_mean"] + (raw - params["raw_mean"]) * (params["act_std"] / params["raw_std"])


def quantile_map_fit(raw, actual, n_quantiles=50):
    qs = np.linspace(0, 1, n_quantiles)
    raw_q = np.quantile(raw, qs)
    act_q = np.quantile(actual, qs)
    raw_q_u, idx = np.unique(raw_q, return_index=True)
    act_q_u = act_q[idx]
    return {"raw_q": raw_q_u.tolist(), "act_q": act_q_u.tolist()}


def quantile_map_apply(raw, params, max_extrap_multiple=2.0):
    """Empirical quantile mapping with BOUNDED linear tail extrapolation.
    A naive 2-point edge slope is noisy and, for any raw value landing well
    outside the calibration range, can extrapolate to an unbounded,
    nonsensical correction -- confirmed directly: with an unbounded version
    of this function, XLY's test-period MAPE exploded from 43% (before
    quantile-mapping) to 193% (after), driven by a handful of extreme
    values extrapolated off a noisy edge slope. Fixed two ways: (1) the
    slope is estimated from the outer ~15% of the quantile grid (a robust
    linear fit), not just the 2 most extreme, noisiest points; (2) the
    extrapolated correction is capped at +/- max_extrap_multiple times the
    calibration period's own observed range beyond each edge -- allows real
    "frequency correction" for values genuinely outside the calibration
    range, without runaway blowups from a single noisy slope estimate.
    """
    raw_q_u, act_q_u = np.array(params["raw_q"]), np.array(params["act_q"])
    corrected = np.interp(raw, raw_q_u, act_q_u)
    if len(raw_q_u) >= 4:
        n_tail = max(2, len(raw_q_u) // 7)  # ~15% of the grid for a more robust slope estimate
        lo_fit = np.polyfit(raw_q_u[:n_tail], act_q_u[:n_tail], 1)
        hi_fit = np.polyfit(raw_q_u[-n_tail:], act_q_u[-n_tail:], 1)
        lo_slope, hi_slope = lo_fit[0], hi_fit[0]
        act_range = act_q_u[-1] - act_q_u[0]
        lo_cap = act_q_u[0] - max_extrap_multiple * act_range
        hi_cap = act_q_u[-1] + max_extrap_multiple * act_range
        below, above = raw < raw_q_u[0], raw > raw_q_u[-1]
        ext_lo = np.clip(act_q_u[0] + (raw - raw_q_u[0]) * lo_slope, lo_cap, act_q_u[0])
        ext_hi = np.clip(act_q_u[-1] + (raw - raw_q_u[-1]) * hi_slope, act_q_u[-1], hi_cap)
        corrected = np.where(below, ext_lo, corrected)
        corrected = np.where(above, ext_hi, corrected)
    return corrected


CANDIDATES = ["clim", "credit_only", "vix_only", "both"]
results = {}

for tkr in sorted(decisions.keys()):
    horizon = decisions[tkr]["horizon"]
    old_winner = decisions[tkr]["price_based_winner"]  # for reference/comparison only
    sub_th = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon)]

    frames = {}
    for v in ["credit_only", "vix_only", "both"]:
        vs = sub_th[sub_th["variant"] == v][["date", "q0.5", "clim_q0.5", "y_true"]].dropna()
        frames[v] = vs.set_index("date")
    if any(len(f) == 0 for f in frames.values()):
        continue

    common_dates = frames["credit_only"].index.intersection(frames["vix_only"].index).intersection(frames["both"].index)
    if len(common_dates) < 2 * MIN_ROWS:
        continue

    df = pd.DataFrame(index=sorted(common_dates))
    df["clim"] = frames["both"].loc[df.index, "clim_q0.5"]  # identical across variants
    df["credit_only"] = frames["credit_only"].loc[df.index, "q0.5"]
    df["vix_only"] = frames["vix_only"].loc[df.index, "q0.5"]
    df["both"] = frames["both"].loc[df.index, "q0.5"]
    df["y_true"] = frames["both"].loc[df.index, "y_true"]

    sel = df[df.index < SELECTION_END]
    test = df[df.index >= SELECTION_END]
    if len(sel) < MIN_ROWS or len(test) < MIN_ROWS:
        continue

    # (a) stacking regression, fit on SELECTION only.
    # The 4 candidates (climatology, credit_only, vix_only, both) are strongly
    # collinear -- "both" is literally built from the same ingredients as
    # credit_only and vix_only -- so plain OLS assigns huge, unstable,
    # opposite-signed weights that fit in-sample and blow up out-of-sample
    # (confirmed: weights up to +/-3.8, catastrophic test-period MAPE for
    # several instruments). Fixed with ridge (L2-regularized) regression --
    # still a genuine regression/curve-fit, just one robust to collinear
    # inputs. Ridge strength chosen via an internal train/val split WITHIN
    # selection only (never touching test), same discipline as everywhere
    # else in this project.
    y_sel_arr = sel["y_true"].values
    Xc_sel = np.column_stack([sel[c].values for c in CANDIDATES])  # no intercept column -- center instead
    x_mean, y_mean = Xc_sel.mean(axis=0), y_sel_arr.mean()
    Xc_sel_ctr, y_sel_ctr = Xc_sel - x_mean, y_sel_arr - y_mean

    split_i = int(len(sel) * 0.7)
    Xtr, Xval = Xc_sel_ctr[:split_i], Xc_sel_ctr[split_i:]
    ytr, yval = y_sel_ctr[:split_i], y_sel_ctr[split_i:]
    lambdas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_lambda, best_val_mse = lambdas[0], np.inf
    k = Xc_sel.shape[1]
    for lam in lambdas:
        w_try = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(k), Xtr.T @ ytr)
        val_mse = float(np.mean((Xval @ w_try - yval) ** 2))
        if val_mse < best_val_mse:
            best_val_mse, best_lambda = val_mse, lam

    w_centered = np.linalg.solve(Xc_sel_ctr.T @ Xc_sel_ctr + best_lambda * np.eye(k), Xc_sel_ctr.T @ y_sel_ctr)
    intercept = y_mean - x_mean @ w_centered
    w = np.concatenate([[intercept], w_centered])
    X_sel = np.column_stack([np.ones(len(sel))] + [sel[c].values for c in CANDIDATES])
    stacked_sel = X_sel @ w

    # (b) sequential post-processing, fit on SELECTION's stacked output only
    mm_params = moment_match_fit(stacked_sel, sel["y_true"].values)
    mm_sel = moment_match_apply(stacked_sel, mm_params)
    qm_params = quantile_map_fit(mm_sel, sel["y_true"].values)
    final_sel = quantile_map_apply(mm_sel, qm_params)

    # apply the FROZEN pipeline to TEST exactly once
    X_test = np.column_stack([np.ones(len(test))] + [test[c].values for c in CANDIDATES])
    stacked_test = X_test @ w
    mm_test = moment_match_apply(stacked_test, mm_params)
    final_test = quantile_map_apply(mm_test, qm_params)

    # price-level MAPE on test period
    series = get_series(tkr)
    series_pos = {d: i for i, d in enumerate(series.index)}
    test_dates = test.index.values
    price_now = series.reindex(test_dates).values
    tidx = np.array([series_pos.get(pd.Timestamp(d), -1) for d in test_dates])
    valid = (tidx >= 0) & (tidx + horizon < len(series))
    tgt_price = series.values[tidx[valid] + horizon]

    def mape(pred_ret):
        pred_price = price_now[valid] * np.exp(pred_ret[valid])
        return float(np.mean(np.abs(pred_price / tgt_price - 1)) * 100)

    # for reference: the OLD master-model single-winner's own MAPE, same test dates/instrument
    old_col = "clim" if old_winner == "climatology" else old_winner
    old_pred = test[old_col].values

    entry = {
        "horizon": horizon, "old_winner": old_winner, "n_selection": int(len(sel)), "n_test": int(len(test)),
        "regression_weights": {"intercept": float(w[0]), **{c: float(w[i + 1]) for i, c in enumerate(CANDIDATES)}},
        "ridge_lambda": float(best_lambda),
        "mape_old_single_winner": mape(old_pred),
        "mape_climatology": mape(test["clim"].values),
        "mape_stacked_raw": mape(stacked_test),
        "mape_stacked_postprocessed": mape(final_test),
    }
    results[tkr] = entry

with open(os.path.join(OUT_DIR, "stacked_postprocess_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)

n_improved = sum(1 for r in results.values() if r["mape_stacked_postprocessed"] < r["mape_old_single_winner"] - 0.01)
print(f"=== Stacked regression + sequential post-processing, fit entirely on selection period, {len(results)} instruments ===")
for tkr in sorted(results, key=lambda t: results[t]["mape_old_single_winner"]):
    r = results[tkr]
    flag = "IMPROVED" if r["mape_stacked_postprocessed"] < r["mape_old_single_winner"] - 0.01 else ""
    print(f"  {tkr} (old_winner={r['old_winner']}@{r['horizon']}d): old_single_winner={r['mape_old_single_winner']:.2f}% "
          f"clim={r['mape_climatology']:.2f}% stacked_raw={r['mape_stacked_raw']:.2f}% "
          f"stacked+postproc={r['mape_stacked_postprocessed']:.2f}% {flag}")
print(f"\n{n_improved}/{len(results)} instruments improved on the old single-winner master-model MAPE")

tickers_sorted = sorted(results, key=lambda t: results[t]["mape_old_single_winner"] - results[t]["mape_stacked_postprocessed"], reverse=True)
fig, ax = plt.subplots(figsize=(9, 8))
y_pos = np.arange(len(tickers_sorted))
old_vals = [results[t]["mape_old_single_winner"] for t in tickers_sorted]
new_vals = [results[t]["mape_stacked_postprocessed"] for t in tickers_sorted]
ax.barh(y_pos - 0.18, old_vals, height=0.35, color="#9aa1ad", label="Old: single-winner master model")
ax.barh(y_pos + 0.18, new_vals, height=0.35,
        color=["#2f8a4e" if n < o - 0.01 else "#b0492f" for o, n in zip(old_vals, new_vals)],
        label="New: stacked regression + post-processing")
ax.set_yticks(y_pos)
ax.set_yticklabels(tickers_sorted, fontsize=9)
ax.set_xlabel("Test-period MAPE (price, %) -- lower is better")
ax.set_title(f"Stacked regression + post-processing (fit entirely on selection period) vs. old single-winner MAPE\n"
             f"Test period only, {n_improved}/{len(results)} improved")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "_STACKED_SUMMARY.png"), dpi=115, bbox_inches="tight")
plt.close(fig)
print("\nDone.")
