"""
Two things:

(1) Does JPM's post-processing correction (fit on 2022-2024, verified on
2024+ in 51_post_processing.py: MAPE 13.4% -> 7.7%) actually translate into
statistically significant alpha? Rebuilds the target-price buy-low/sell
-high strategy (45_target_price_trading_strategy.py's logic) using the
CORRECTED quantiles instead of raw, restricted to the verification period
only (2024-01-01 onward -- data the correction never saw). The single OLS
(a, b) fit on q0.5 is applied uniformly to q0.1..q0.9 (a standard, data-
efficient recalibration assumption: one affine correction applied to the
whole predictive distribution, not fit separately per quantile -- valid as
long as b>0, which it is, since that preserves quantile ordering). Compares
against the RAW forecast's strategy on the IDENTICAL verification-only
dates (not the fuller 2022-2026 window used in earlier backtests) so the
comparison isn't confounded by different sample periods.

(2) The user asked why not fit the post-processing correction on the
SELECTION period (pre-2022, already used to pick JPM's winning horizon/
variant) instead of carving the holdout into calibration/verification --
that would preserve the ENTIRE holdout for the final test, more
statistical power. Real tradeoff, checked empirically rather than argued
in the abstract: does the SAME correction (fit on 2022-2024) also reduce
error if applied backward to JPM's selection-period (pred, actual) pairs?
If yes, the bias is long-lived and the user's design would likely have
worked as well or better (more data, no loss of holdout power). If no,
the bias is regime-specific/recent, and fitting on old selection-period
data would not have captured it -- validating the recency-based design
instead. Answered with real data, not speculation.

Run: python 53_jpm_corrected_alpha_retest.py
Output: prints results; pnl_plots/_POSTPROC_JPM_alpha_retest.png
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
POSTPROC_SPLIT = pd.Timestamp("2024-01-01")
COST_BPS = 5
TKR = "JPM"

decisions = json.load(open(os.path.join(OUT_DIR, "master_model_final_decision.json")))
postproc = json.load(open(os.path.join(OUT_DIR, "post_processing_results.json")))
oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
series = prices[TKR].dropna()

dec = decisions[TKR]
horizon, winner = dec["horizon"], dec["price_based_winner"]
a, b = postproc[TKR]["ols_params"]["a"], postproc[TKR]["ols_params"]["b"]
print(f"JPM correction: a={a:+.4f}, b={b:.4f}\n")

sub_th = oos_all[(oos_all["ticker"] == TKR) & (oos_all["horizon"] == horizon)]
sub = sub_th[sub_th["variant"] == winner].sort_values("date").reset_index(drop=True)

QCOLS = ["q0.1", "q0.25", "q0.5", "q0.75", "q0.9"]
for c in QCOLS:
    sub[f"corr_{c}"] = a + b * sub[c]


def jensen_alpha(y_series, x_series):
    common = y_series.index.intersection(x_series.index)
    y = y_series.reindex(common).values
    x = x_series.reindex(common).values
    X = np.column_stack([np.ones_like(x), x])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_daily, beta = beta_hat
    resid = y - X @ beta_hat
    n, k = len(y), 2
    sigma2 = (resid @ resid) / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se_alpha = np.sqrt(max(sigma2 * XtX_inv[0, 0], 0))
    t_alpha = alpha_daily / se_alpha if se_alpha > 0 else np.nan
    return float(alpha_daily * 252 * 100), float(se_alpha * 252 * 100), float(beta), float(t_alpha)


IQR_TO_SIGMA = 1.349
HALF_KELLY = 0.5
KELLY_CAP = 2.0
SIGMA_FLOOR = 1e-4


def build_strategy(src, buy_col, sell_col, med_col, date_filter):
    """Same buy-low/sell-high entry/exit as 45_target_price_trading_strategy.py,
    but sized by the Kelly fraction (47_kelly_sized_strategy.py) instead of a flat
    100% -- required here because JPM's raw signal never changes sign or triggers
    a flat position in this window (confirmed: both raw and corrected are long
    100% of the time), so only a size-sensitive strategy can register a magnitude
    -only correction like this one."""
    s = src[date_filter(src["date"])].sort_values("date")
    idx = series.index[(series.index >= s["date"].min()) & (series.index <= s["date"].max())]
    price = series.reindex(idx)
    buy_ret_q = pd.Series(s[buy_col].values, index=s["date"].values).reindex(idx).ffill()
    sell_ret_q = pd.Series(s[sell_col].values, index=s["date"].values).reindex(idx).ffill()
    med_ret_q = pd.Series(s[med_col].values, index=s["date"].values).reindex(idx).ffill()
    buy_target = (price * np.exp(buy_ret_q)).shift(1)
    sell_target = (price * np.exp(sell_ret_q)).shift(1)

    sigma = ((sell_ret_q - buy_ret_q) / IQR_TO_SIGMA).clip(lower=SIGMA_FLOOR)
    kelly_frac = (HALF_KELLY * med_ret_q / (sigma ** 2)).clip(lower=0.0, upper=KELLY_CAP).shift(1)

    state, sizes = 0, []
    for i in range(len(idx)):
        px = price.iloc[i]
        if state == 0:
            bt = buy_target.iloc[i]
            if pd.notna(bt) and px <= bt:
                state = 1
        else:
            st = sell_target.iloc[i]
            if pd.notna(st) and px >= st:
                state = 0
        sizes.append(state)
    in_position = pd.Series(sizes, index=idx, dtype=float)
    position = (in_position * kelly_frac.reindex(idx)).fillna(0.0)
    daily_ret = price.pct_change().fillna(0.0)
    applied = position.shift(1).fillna(0.0)
    gross = applied * daily_ret
    turnover = applied.diff().abs().fillna(0.0)
    net = gross - turnover * (COST_BPS / 10000.0)
    return net, daily_ret, position.mean()


# (1) verification-only comparison, raw vs corrected, identical dates, Kelly-sized
verify_filter = lambda d: d >= POSTPROC_SPLIT
net_raw, own_ret, avg_size_raw = build_strategy(sub, "q0.25", "q0.75", "q0.5", verify_filter)
net_corr, _, avg_size_corr = build_strategy(sub, "corr_q0.25", "corr_q0.75", "corr_q0.5", verify_filter)
print(f"Average Kelly size: raw={avg_size_raw:.2f}x, corrected={avg_size_corr:.2f}x\n")

for label, net in [("RAW", net_raw), ("CORRECTED", net_corr)]:
    cum = (1 + net).cumprod()
    bh_cum = (1 + own_ret.reindex(net.index)).cumprod()
    a_own, se_own, b_own, t_own = jensen_alpha(net, own_ret)
    print(f"{label} (verification period {POSTPROC_SPLIT.date()} onward, n={len(net)} days):")
    print(f"  total return: {(cum.iloc[-1]-1)*100:+.1f}% vs buy&hold {(bh_cum.iloc[-1]-1)*100:+.1f}%")
    print(f"  alpha vs own buy&hold: {a_own:+.2f}%/yr, t={t_own:+.2f}, "
          f"{'SIGNIFICANT' if abs(t_own) >= 1.96 else 'not significant'}")

fig, ax = plt.subplots(figsize=(10, 5.5))
cum_raw = (1 + net_raw).cumprod()
cum_corr = (1 + net_corr).cumprod()
bh_cum = (1 + own_ret.reindex(cum_raw.index)).cumprod()
ax.plot(bh_cum.index, (bh_cum - 1) * 100, color="black", lw=1.3, label="JPM buy & hold")
ax.plot(cum_raw.index, (cum_raw - 1) * 100, color="tab:red", lw=1.2, ls="--", label="Raw forecast strategy, net")
ax.plot(cum_corr.index, (cum_corr - 1) * 100, color="tab:green", lw=1.3, label="Corrected forecast strategy, net")
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.set_title(f"JPM: does the post-processing correction produce real alpha?\nVerification period only ({POSTPROC_SPLIT.date()} onward), raw vs corrected on identical dates")
ax.set_ylabel("Cumulative return (%)")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "_POSTPROC_JPM_alpha_retest.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

# (2) does the correction generalize backward to the selection period?
print("\n--- Does the 2022-2024-fit correction generalize backward to the pre-2022 selection period? ---")
selection = sub[sub["date"] < pd.Timestamp("2022-01-01")].dropna(subset=["y_true", "q0.5"])
raw_err = (selection["q0.5"] - selection["y_true"]).abs().mean()
corr_err = (selection["corr_q0.5"] - selection["y_true"]).abs().mean()
print(f"Selection period (n={len(selection)} rows): raw MAE={raw_err:.4f}, corrected MAE={corr_err:.4f} "
      f"({'IMPROVES' if corr_err < raw_err else 'DOES NOT improve'} backward)")
print(f"(for reference, verification period improvement was raw MAE=0.313 -> corrected MAE=0.210 in return units, per the internal_val_mae check)")
