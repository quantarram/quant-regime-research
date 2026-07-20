"""
Cross-sectional relative-value long/short strategy -- a genuinely different
mechanism from the sign-rule / target-price / Kelly-sizing strategies
already tested, all of which were long-vs-cash bets on a single instrument
at a time and all converged to zero significant alpha. This strategy
instead asks a relative question: is the master model more bullish on
instrument A than instrument B *right now*, regardless of what the broad
market is doing? Going long the instruments it likes most and short the
ones it likes least, in equal size, cancels common market-wide moves and
isolates whatever idiosyncratic, cross-sectional information the model
actually has -- something a purely directional long/flat bet cannot
separate from beta.

Ranking score: instruments use different winning horizons H (1 to 252
days), so raw predicted returns aren't comparable across instruments --
a 252-day forecast's return figure is not on the same footing as a 1-day
one. Both are converted to an ANNUALIZED, risk-adjusted score (a "predicted
Sharpe ratio") before ranking:
  mu_H    = predicted median H-day log return (q0.5; clim_q0.5 for
            climatology winners)
  sigma_H = predicted H-day dispersion, (q0.75-q0.25)/1.349
  score   = (mu_H / sigma_H) * sqrt(252/H)
This is the correct annualization (mean scales with H, std with sqrt(H),
so their ratio needs the sqrt(252/H) correction to be comparable across
different H) -- without it, longer-horizon instruments would mechanically
dominate the ranking regardless of true conviction.

Construction: each day, rank all instruments with a valid score; go long
the top tercile, short the bottom tercile (equal weight within each leg,
dollar-neutral, ~200% gross exposure), flat the middle tercile. Same
no-look-ahead convention as every other backtest here (yesterday's score
decides today's weights, applied to today's return) and the same disclosed
5bps cost per unit of weight change. No borrow cost is modeled for the
short leg -- disclosed simplification.

Significance test: for a dollar-neutral book, the natural null is "mean
daily return is zero" (there's no single-asset buy-and-hold counterfactual
the way there was for the long-only strategies) -- tested with a plain
one-sample t-test, not a resampling test. A secondary regression against
SPY checks the book's realized beta is genuinely close to zero (validating
the market-neutral construction) and gives a cross-check alpha estimate.

Run: python 49_relative_value_long_short.py
Output: relative_value_results.json, pnl_plots/_RELVAL_long_short.png
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
COST_BPS = 5
IQR_TO_SIGMA = 1.349
SIGMA_FLOOR = 1e-4
MIN_AVAILABLE_FOR_TERCILES = 9  # need >=3 per leg to form a meaningful tercile split
REBAL_FREQ = 21  # trading days between full re-ranks (~monthly, matches this program's own horizon grid)

decisions = json.load(open(os.path.join(OUT_DIR, "master_model_final_decision.json")))
oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))
PROXY_TICKERS = ("IYR", "VOX")
spy_ret = prices["SPY"].dropna().pct_change().dropna()

TICKERS = sorted(decisions.keys())


def get_series(tkr):
    return (prices_proxy[tkr] if tkr in PROXY_TICKERS else prices[tkr]).dropna()


def max_drawdown(cum):
    running_max = cum.cummax()
    return float((cum / running_max - 1).min())


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


# --- build each instrument's daily annualized score series ---
score_by_ticker = {}
for tkr in TICKERS:
    dec = decisions[tkr]
    horizon, winner = dec["horizon"], dec["price_based_winner"]
    sub = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon)]
    variants_present = sub["variant"].unique().tolist()
    if winner == "climatology":
        src = sub[sub["variant"] == "both"] if "both" in variants_present else sub[sub["variant"] == variants_present[0]]
        med_col, lo_col, hi_col = "clim_q0.5", "clim_q0.25", "clim_q0.75"
    else:
        src = sub[sub["variant"] == winner]
        med_col, lo_col, hi_col = "q0.5", "q0.25", "q0.75"
    src = src[src["date"] >= HOLDOUT_START].sort_values("date")
    if len(src) < 100:
        continue
    med = pd.Series(src[med_col].values, index=src["date"].values)
    sigma = pd.Series(((src[hi_col] - src[lo_col]) / IQR_TO_SIGMA).values, index=src["date"].values).clip(lower=SIGMA_FLOOR)
    score_by_ticker[tkr] = (med / sigma) * np.sqrt(252.0 / horizon)

score_df = pd.DataFrame(score_by_ticker)
all_dates = score_df.index.sort_values()
score_df = score_df.reindex(all_dates).ffill(limit=5)  # tolerate brief gaps only, no long forward-fill

ret_df = pd.DataFrame({tkr: get_series(tkr).pct_change() for tkr in score_by_ticker}).reindex(all_dates)

weights = pd.DataFrame(0.0, index=all_dates, columns=score_df.columns)
n_long_used, n_short_used = [], []
for i, dt in enumerate(all_dates):
    # Rebalance only every REBAL_FREQ trading days -- most of these forecasts are
    # multi-week to multi-month horizons, so re-ranking and fully reshuffling the
    # book from scratch every day (as an earlier version of this script did) is an
    # unrealistic execution assumption that let transaction costs alone swamp the
    # result (see git history / memory for the diagnosis). Hold positions between
    # rebalances instead.
    if i % REBAL_FREQ != 0:
        weights.iloc[i] = weights.iloc[i - 1]
        n_long_used.append(n_long_used[-1])
        n_short_used.append(n_short_used[-1])
        continue
    row = score_df.loc[dt].dropna()
    n_avail = len(row)
    if n_avail < MIN_AVAILABLE_FOR_TERCILES:
        weights.iloc[i] = weights.iloc[i - 1] if i > 0 else 0.0
        n_long_used.append(0)
        n_short_used.append(0)
        continue
    n_leg = max(1, n_avail // 3)
    ranked = row.sort_values(ascending=False)
    long_names = ranked.index[:n_leg]
    short_names = ranked.index[-n_leg:]
    weights.loc[dt, long_names] = 1.0 / n_leg
    weights.loc[dt, short_names] = -1.0 / n_leg
    n_long_used.append(n_leg)
    n_short_used.append(n_leg)

applied_weights = weights.shift(1).fillna(0.0)  # yesterday's score decides today's weights, no look-ahead
gross_ret = (applied_weights * ret_df).sum(axis=1)
turnover = applied_weights.diff().abs().sum(axis=1).fillna(0.0)
cost = turnover * (COST_BPS / 10000.0)
net_ret = gross_ret - cost

cum = (1 + net_ret).cumprod()

n_days = len(net_ret)
mean_daily = net_ret.mean()
se_daily = net_ret.std() / np.sqrt(n_days)
t_stat_zero = mean_daily / se_daily if se_daily > 0 else np.nan
ann_mean_pct = mean_daily * 252 * 100
ann_se_pct = se_daily * 252 * 100

a_spy, se_spy, b_spy, t_spy = jensen_alpha(net_ret, spy_ret)

gross_cum = (1 + gross_ret).cumprod()
gross_mean_daily = gross_ret.mean()
gross_se_daily = gross_ret.std() / np.sqrt(len(gross_ret))
gross_t_stat = gross_mean_daily / gross_se_daily if gross_se_daily > 0 else np.nan

stats = {
    "n_instruments_universe": len(score_by_ticker),
    "n_days": int(n_days),
    "avg_n_long": float(np.mean([n for n in n_long_used if n > 0])) if any(n_long_used) else 0.0,
    "avg_n_short": float(np.mean([n for n in n_short_used if n > 0])) if any(n_short_used) else 0.0,
    "total_return_pct": float((cum.iloc[-1] - 1) * 100),
    "gross_total_return_pct": float((gross_cum.iloc[-1] - 1) * 100),
    "gross_annualized_mean_return_pct": float(gross_mean_daily * 252 * 100),
    "gross_t_stat_vs_zero": float(gross_t_stat),
    "gross_significant_vs_zero_95": bool(abs(gross_t_stat) >= 1.96),
    "total_cost_drag_pct_of_year": float((mean_daily - gross_mean_daily) * 252 * 100),
    "annualized_mean_return_pct": float(ann_mean_pct),
    "annualized_return_ci_lo_pct": float(ann_mean_pct - 1.96 * ann_se_pct),
    "annualized_return_ci_hi_pct": float(ann_mean_pct + 1.96 * ann_se_pct),
    "t_stat_vs_zero": float(t_stat_zero),
    "significant_vs_zero_95": bool(abs(t_stat_zero) >= 1.96),
    "sharpe": float(net_ret.mean() / net_ret.std() * np.sqrt(252)) if net_ret.std() > 0 else float("nan"),
    "max_dd_pct": max_drawdown(cum) * 100,
    "avg_daily_turnover": float(turnover.mean()),
    "beta_vs_spy": float(b_spy),
    "alpha_vs_spy_pct": float(a_spy),
    "t_vs_spy": float(t_spy),
    "significant_vs_spy_95": bool(abs(t_spy) >= 1.96),
}
with open(os.path.join(OUT_DIR, "relative_value_results.json"), "w") as f:
    json.dump(stats, f, indent=2, default=float)

print("=== Cross-sectional relative-value long/short (tercile, dollar-neutral) ===")
print(f"Universe: {stats['n_instruments_universe']} instruments, avg {stats['avg_n_long']:.1f} long / {stats['avg_n_short']:.1f} short per day")
print(f"Total return: {stats['total_return_pct']:+.1f}% NET vs {stats['gross_total_return_pct']:+.1f}% GROSS (before costs) over {n_days} days")
print(f"Annualized mean return (net): {ann_mean_pct:+.2f}%/yr (95% CI {stats['annualized_return_ci_lo_pct']:+.1f}% to {stats['annualized_return_ci_hi_pct']:+.1f}%), t={t_stat_zero:+.2f} vs zero -- "
      f"{'SIGNIFICANT' if stats['significant_vs_zero_95'] else 'not significant'}")
print(f"Annualized mean return (gross, before costs): {stats['gross_annualized_mean_return_pct']:+.2f}%/yr, t={gross_t_stat:+.2f} -- "
      f"{'SIGNIFICANT' if stats['gross_significant_vs_zero_95'] else 'not significant'}")
print(f"Cost drag: {stats['total_cost_drag_pct_of_year']:+.2f}%/yr from turnover")
print(f"Sharpe: {stats['sharpe']:+.2f}, max DD: {stats['max_dd_pct']:.1f}%, avg daily turnover: {stats['avg_daily_turnover']:.2f}")
print(f"Beta vs SPY: {b_spy:+.3f} (should be near zero if genuinely market-neutral)")
print(f"Diagnostic alpha vs SPY: {a_spy:+.2f}%/yr, t={t_spy:+.2f} -- {'SIGNIFICANT' if stats['significant_vs_spy_95'] else 'not significant'}")

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(cum.index, (cum - 1) * 100, color="teal", lw=1.4, label="Relative-value long/short, net of costs")
spy_cum_aligned = (1 + spy_ret.reindex(cum.index).fillna(0.0)).cumprod()
ax.plot(spy_cum_aligned.index, (spy_cum_aligned - 1) * 100, color="tab:gray", lw=1.0, ls="--", label="SPY buy&hold (context, not the benchmark)")
ax.axhline(0, color="black", lw=0.8)
sig = "significant" if stats["significant_vs_zero_95"] else "not significant"
ax.set_title(f"Cross-sectional relative-value long/short, {stats['n_instruments_universe']}-instrument universe\n"
             f"Annualized return {ann_mean_pct:+.2f}%/yr, t={t_stat_zero:+.2f} vs zero ({sig}) -- beta to SPY {b_spy:+.2f}")
ax.set_ylabel("Cumulative return (%)")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "_RELVAL_long_short.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print("\nDone.")
