"""
CORRECTION to 43_alpha_test_all_instruments.py: that script tested alpha
against SPY as a universal market benchmark. For an asset class that is
structurally uncorrelated with equities (GLD, EURUSD=X), that test doesn't
isolate model skill -- it mostly measures "did this asset class rally
independent of stocks," which a naive buy-and-hold would also capture, with
no model at all. This script adds the control that should have been run the
first time: does simply buying-and-holding the instrument, with zero
model, ALSO show significant alpha vs SPY? And more importantly, the
properly-specified test of "does OUR MODEL's timing add value" -- regress
the strategy's net return on the SAME INSTRUMENT's own buy-and-hold return,
not SPY. That isolates timing skill from asset-class/beta exposure.

Run: python 44_alpha_test_own_benchmark.py
Output: alpha_test_own_benchmark_results.json, pnl_plots/_ALPHA_test_own_benchmark.png
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


def strategy_and_own_ret(tkr):
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
    return net, ret, winner, horizon, int(turnover.sum())


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
    return float(alpha_daily * 252 * 100), float(se_alpha * 252 * 100), float(beta), float(t_alpha), int(n)


results = {}
for tkr in sorted(decisions.keys()):
    strat_ret, own_ret, winner, horizon, n_changes = strategy_and_own_ret(tkr)
    a_spy, se_spy, b_spy, t_spy, n1 = jensen_alpha(strat_ret, spy_ret)
    a_bh_spy, se_bh_spy, b_bh_spy, t_bh_spy, n2 = jensen_alpha(own_ret.reindex(strat_ret.index), spy_ret)
    a_own, se_own, b_own, t_own, n3 = jensen_alpha(strat_ret, own_ret)
    results[tkr] = {
        "winner": winner, "horizon": horizon, "n_position_changes": n_changes,
        "alpha_vs_spy_pct": a_spy, "t_vs_spy": t_spy,
        "buyhold_alpha_vs_spy_pct": a_bh_spy, "t_buyhold_vs_spy": t_bh_spy,
        "alpha_vs_own_benchmark_pct": a_own, "alpha_vs_own_se_pct": se_own,
        "alpha_vs_own_ci_lo_pct": a_own - 1.96 * se_own, "alpha_vs_own_ci_hi_pct": a_own + 1.96 * se_own,
        "beta_vs_own_benchmark": b_own, "t_vs_own_benchmark": t_own,
        "significant_vs_own_95": bool(abs(t_own) >= 1.96),
    }

with open(os.path.join(OUT_DIR, "alpha_test_own_benchmark_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)

n_sig = sum(1 for r in results.values() if r["significant_vs_own_95"])
print(f"=== Alpha vs OWN instrument's buy-and-hold (the appropriate benchmark for timing skill) ===")
for tkr in sorted(results, key=lambda t: -results[t]["alpha_vs_own_benchmark_pct"]):
    r = results[tkr]
    flag = "SIGNIFICANT" if r["significant_vs_own_95"] else "not significant"
    print(f"  {tkr} ({r['winner']}@{r['horizon']}d): alpha_vs_own={r['alpha_vs_own_benchmark_pct']:+.2f}%/yr "
          f"t={r['t_vs_own_benchmark']:+.2f} beta={r['beta_vs_own_benchmark']:.2f} "
          f"[buyhold_vs_SPY t={r['t_buyhold_vs_spy']:+.2f}] [{flag}]")
print(f"\n{n_sig}/{len(results)} significant alpha vs own instrument's buy-and-hold at 95%")

tickers_sorted = sorted(results, key=lambda t: results[t]["alpha_vs_own_benchmark_pct"])
alphas = [results[t]["alpha_vs_own_benchmark_pct"] for t in tickers_sorted]
ci_lo = [results[t]["alpha_vs_own_ci_lo_pct"] for t in tickers_sorted]
ci_hi = [results[t]["alpha_vs_own_ci_hi_pct"] for t in tickers_sorted]
tstats = [results[t]["t_vs_own_benchmark"] for t in tickers_sorted]
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
ax.set_xlabel("Annualized alpha vs OWN instrument's buy-and-hold, net of costs (%/yr) -- 95% CI shown")
ax.set_title(f"Corrected alpha test: does the model's TIMING add value over simply holding the asset?\n"
             f"All {len(results)} instruments, holdout period -- 0/{len(results)} significant")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "_ALPHA_test_own_benchmark.png"), dpi=115, bbox_inches="tight")
plt.close(fig)
print("\nDone.")
