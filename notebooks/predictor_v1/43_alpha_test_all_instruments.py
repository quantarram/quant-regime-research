"""
Does ANY instrument's master-model prediction (including climatology winners,
not just the ones that already beat buy-and-hold) generate real alpha --
i.e. risk-adjusted excess return left over after removing market-beta
exposure -- rather than just beating its own buy-and-hold in raw return?

"Beats buy-and-hold" (42_pnl_backtest.py) and "has alpha" are different
questions: a strategy can beat its own instrument's buy-and-hold purely by
being exposed to a name that outperformed the market anyway. The correct
test is a market-model (Jensen's alpha) regression: strategy net daily
return regressed on SPY's daily return (the market proxy), for the same
long/flat, shift(1), 5bps-cost rule used throughout. The intercept is
alpha; a plain analytic OLS standard error and t-stat, NOT a
resampling/bootstrap test, decide significance -- consistent with this
project's standing rejection of randomization testing.

Run: python 43_alpha_test_all_instruments.py
Output: alpha_test_results.json, pnl_plots/_ALPHA_test_all.png
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

decisions = json.load(open(os.path.join(OUT_DIR, "master_model_final_decision.json")))
oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))
PROXY_TICKERS = ("IYR", "VOX")

spy_ret = prices["SPY"].dropna().pct_change().dropna()


def get_series(tkr):
    return (prices_proxy[tkr] if tkr in PROXY_TICKERS else prices[tkr]).dropna()


def strategy_daily_ret(tkr):
    dec = decisions[tkr]
    horizon, winner = dec["horizon"], dec["price_based_winner"]
    sub = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon)]
    variants = sub["variant"].unique().tolist()
    if winner == "climatology":
        src = sub[sub["variant"] == "both"] if "both" in variants else sub[sub["variant"] == variants[0]]
        pred_col = "clim_q0.5"
    else:
        src = sub[sub["variant"] == winner]
        pred_col = "q0.5"
    src = src[src["date"] >= HOLDOUT_START].sort_values("date")

    series = get_series(tkr)
    daily_ret = series.pct_change().dropna()
    position = pd.Series((src[pred_col].values > 0).astype(float), index=src["date"].values)
    position = position[~position.index.duplicated(keep="last")]
    idx = daily_ret.index[(daily_ret.index >= position.index.min()) & (daily_ret.index <= position.index.max())]
    pos_daily = position.reindex(idx).ffill().fillna(0.0)
    ret = daily_ret.reindex(idx)
    applied = pos_daily.shift(1).fillna(0.0)
    gross = applied * ret
    turnover = applied.diff().abs().fillna(0.0)
    net = gross - turnover * (COST_BPS / 10000.0)
    return net, winner, horizon, int(turnover.sum())


results = {}
for tkr in sorted(decisions.keys()):
    strat_ret, winner, horizon, n_changes = strategy_daily_ret(tkr)
    common = strat_ret.index.intersection(spy_ret.index)
    y = strat_ret.reindex(common).values
    x = spy_ret.reindex(common).values
    X = np.column_stack([np.ones_like(x), x])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_daily, beta = beta_hat
    resid = y - X @ beta_hat
    n, k = len(y), 2
    sigma2 = (resid @ resid) / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se_alpha = np.sqrt(sigma2 * XtX_inv[0, 0])
    t_alpha = alpha_daily / se_alpha if se_alpha > 0 else np.nan
    alpha_ann = alpha_daily * 252 * 100
    se_ann = se_alpha * 252 * 100
    results[tkr] = {
        "winner": winner, "horizon": horizon, "n_position_changes": n_changes, "n_days": int(n),
        "alpha_annualized_pct": float(alpha_ann), "alpha_se_annualized_pct": float(se_ann),
        "alpha_ci_lo_pct": float(alpha_ann - 1.96 * se_ann), "alpha_ci_hi_pct": float(alpha_ann + 1.96 * se_ann),
        "beta_vs_spy": float(beta), "t_stat": float(t_alpha),
        "significant_95": bool(abs(t_alpha) >= 1.96),
    }

with open(os.path.join(OUT_DIR, "alpha_test_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)

n_sig_pos = sum(1 for r in results.values() if r["significant_95"] and r["alpha_annualized_pct"] > 0)
n_sig_neg = sum(1 for r in results.values() if r["significant_95"] and r["alpha_annualized_pct"] < 0)
print(f"=== Alpha test (Jensen's alpha vs SPY, net of costs), all {len(results)} instruments ===")
for tkr in sorted(results, key=lambda t: -results[t]["alpha_annualized_pct"]):
    r = results[tkr]
    flag = "SIGNIFICANT+" if (r["significant_95"] and r["alpha_annualized_pct"] > 0) else \
        ("SIGNIFICANT-" if r["significant_95"] else "not significant")
    print(f"  {tkr} ({r['winner']}@{r['horizon']}d): alpha={r['alpha_annualized_pct']:+.2f}%/yr "
          f"t={r['t_stat']:+.2f} beta={r['beta_vs_spy']:.2f} n_trades={r['n_position_changes']} [{flag}]")
print(f"\n{n_sig_pos}/{len(results)} significant POSITIVE alpha, {n_sig_neg}/{len(results)} significant NEGATIVE alpha "
      f"at 95% (|t|>=1.96), {len(results) - n_sig_pos - n_sig_neg} indistinguishable from zero")

tickers_sorted = sorted(results, key=lambda t: results[t]["alpha_annualized_pct"])
alphas = [results[t]["alpha_annualized_pct"] for t in tickers_sorted]
ci_lo = [results[t]["alpha_ci_lo_pct"] for t in tickers_sorted]
ci_hi = [results[t]["alpha_ci_hi_pct"] for t in tickers_sorted]
tstats = [results[t]["t_stat"] for t in tickers_sorted]
labels = [f"{t} ({results[t]['winner']}@{results[t]['horizon']}d)" for t in tickers_sorted]

fig, ax = plt.subplots(figsize=(10, 9))
y_pos = np.arange(len(tickers_sorted))


def color_for(t, a):
    if t >= 1.96 and a > 0:
        return "#2f8a4e"
    if t <= -1.96 and a < 0:
        return "#b0492f"
    return "#9aa1ad"


colors = [color_for(t, a) for t, a in zip(tstats, alphas)]
ax.barh(y_pos, alphas, color=colors, height=0.6, zorder=3)
for i, (lo, hi) in enumerate(zip(ci_lo, ci_hi)):
    ax.plot([lo, hi], [i, i], color="black", lw=1.0, zorder=4)
    ax.plot([lo, lo], [i - 0.1, i + 0.1], color="black", lw=1.0, zorder=4)
    ax.plot([hi, hi], [i - 0.1, i + 0.1], color="black", lw=1.0, zorder=4)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Annualized Jensen alpha vs SPY, net of costs (%/yr) -- 95% CI shown")
ax.set_title(f"Does our master model generate real alpha? All {len(results)} instruments, holdout period\n"
             f"(green = significant positive, red = significant negative, gray = not distinguishable from zero)")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "_ALPHA_test_all.png"), dpi=115, bbox_inches="tight")
plt.close(fig)
print("\nDone.")
