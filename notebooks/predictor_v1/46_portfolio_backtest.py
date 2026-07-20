"""
Multi-instrument portfolio backtest, per the user's request: combine the
individual target-price strategies (45_target_price_trading_strategy.py)
across ALL 22 evaluable instruments, each contributing its own master-model
-selected winner (climatology included -- NOT filtered to "informed"
credit/vix winners only).

CORRECTED from the first version of this script, which restricted the
universe to non-climatology winners on the theory that climatology-winning
instruments had "no real skill." That contradicts this project's own
master-model framing: climatology winning for an instrument means seasonal
patterns are genuinely the best available predictor there, not that there's
no skill -- it's still real, tradeable information from the master model,
on equal footing with the credit/vix-informed variants. Excluding it was an
arbitrary, inconsistent filter invented for this one script. User caught
this directly.

Equal-weight, daily-rebalanced combination of each instrument's individual
target-price strategy net return. Benchmarked against (a) an equal-weight
buy-and-hold of the SAME basket (the correct, asset-matched benchmark per
the prior alpha-test correction) and (b) SPY buy-and-hold, for context.

Run: python 46_portfolio_backtest.py
Output: portfolio_backtest_results.json, pnl_plots/_PORTFOLIO_backtest.png
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

decisions = json.load(open(os.path.join(OUT_DIR, "master_model_final_decision.json")))
strat_results = json.load(open(os.path.join(OUT_DIR, "target_price_strategy_results.json")))
strat_returns = pd.read_parquet(os.path.join(OUT_DIR, "target_price_strategy_returns.parquet"))
prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))
PROXY_TICKERS = ("IYR", "VOX")
spy_ret = prices["SPY"].dropna().pct_change().dropna()

PORTFOLIO_TICKERS = sorted(decisions.keys())
print(f"All {len(PORTFOLIO_TICKERS)} evaluable instruments used for portfolio (each using its own master-model winner, climatology included): {PORTFOLIO_TICKERS}")


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


strat_panel = strat_returns[PORTFOLIO_TICKERS]
portfolio_net = strat_panel.mean(axis=1, skipna=True)  # equal-weight across whichever instruments have data that day

bh_panel = pd.DataFrame({tkr: get_series(tkr).pct_change() for tkr in PORTFOLIO_TICKERS})
bh_panel = bh_panel.reindex(portfolio_net.index)
portfolio_bh = bh_panel.mean(axis=1, skipna=True)

portfolio_cum = (1 + portfolio_net).cumprod()
bh_cum = (1 + portfolio_bh).cumprod()
spy_cum_aligned = (1 + spy_ret.reindex(portfolio_net.index).fillna(0.0)).cumprod()

n_days = len(portfolio_net)
a_bh, se_bh, b_bh, t_bh = jensen_alpha(portfolio_net, portfolio_bh)
a_spy, se_spy, b_spy, t_spy = jensen_alpha(portfolio_net, spy_ret)

stats = {
    "instruments": PORTFOLIO_TICKERS, "n_instruments": len(PORTFOLIO_TICKERS), "n_days": int(n_days),
    "portfolio_net_total_return_pct": float((portfolio_cum.iloc[-1] - 1) * 100),
    "basket_buy_hold_total_return_pct": float((bh_cum.iloc[-1] - 1) * 100),
    "spy_buy_hold_total_return_pct": float((spy_cum_aligned.iloc[-1] - 1) * 100),
    "portfolio_sharpe": float(portfolio_net.mean() / portfolio_net.std() * np.sqrt(252)),
    "basket_buy_hold_sharpe": float(portfolio_bh.mean() / portfolio_bh.std() * np.sqrt(252)),
    "portfolio_max_dd_pct": max_drawdown(portfolio_cum) * 100,
    "basket_buy_hold_max_dd_pct": max_drawdown(bh_cum) * 100,
    "beats_basket_buy_hold": bool(portfolio_cum.iloc[-1] > bh_cum.iloc[-1]),
    "alpha_vs_basket_pct": a_bh, "alpha_vs_basket_ci_lo_pct": a_bh - 1.96 * se_bh,
    "alpha_vs_basket_ci_hi_pct": a_bh + 1.96 * se_bh, "beta_vs_basket": b_bh, "t_vs_basket": t_bh,
    "significant_vs_basket_95": bool(abs(t_bh) >= 1.96),
    "alpha_vs_spy_pct": a_spy, "beta_vs_spy": b_spy, "t_vs_spy": t_spy,
    "significant_vs_spy_95": bool(abs(t_spy) >= 1.96),
}
with open(os.path.join(OUT_DIR, "portfolio_backtest_results.json"), "w") as f:
    json.dump(stats, f, indent=2, default=float)

print(f"\n=== {len(PORTFOLIO_TICKERS)}-instrument equal-weight portfolio, target-price strategies combined ===")
print(f"Portfolio net total return: {stats['portfolio_net_total_return_pct']:+.1f}% "
      f"vs basket buy&hold: {stats['basket_buy_hold_total_return_pct']:+.1f}% "
      f"vs SPY buy&hold: {stats['spy_buy_hold_total_return_pct']:+.1f}%")
print(f"Sharpe: portfolio={stats['portfolio_sharpe']:+.2f} basket_buy_hold={stats['basket_buy_hold_sharpe']:+.2f}")
print(f"Max DD: portfolio={stats['portfolio_max_dd_pct']:.1f}% basket_buy_hold={stats['basket_buy_hold_max_dd_pct']:.1f}%")
print(f"Beats basket buy&hold: {stats['beats_basket_buy_hold']}")
print(f"Alpha vs basket buy&hold (correct benchmark): {a_bh:+.2f}%/yr, t={t_bh:+.2f}, "
      f"{'SIGNIFICANT' if stats['significant_vs_basket_95'] else 'not significant'}")
print(f"Alpha vs SPY (context only): {a_spy:+.2f}%/yr, t={t_spy:+.2f}, "
      f"{'SIGNIFICANT' if stats['significant_vs_spy_95'] else 'not significant'}")

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(bh_cum.index, (bh_cum - 1) * 100, color="black", lw=1.4, label=f"Equal-weight buy&hold, same {len(PORTFOLIO_TICKERS)} instruments")
ax.plot(portfolio_cum.index, (portfolio_cum - 1) * 100, color="tab:purple", lw=1.5, label="Combined target-price strategy portfolio, net")
ax.plot(spy_cum_aligned.index, (spy_cum_aligned - 1) * 100, color="tab:gray", lw=1.0, ls="--", label="SPY buy&hold (context)")
ax.axhline(0, color="gray", lw=0.5, ls=":")
beats = "beats" if stats["beats_basket_buy_hold"] else "does NOT beat"
ax.set_title(f"{len(PORTFOLIO_TICKERS)}-instrument portfolio (all evaluable instruments, own master-model winner each)\n"
             f"Combined strategy {beats} its own basket buy&hold -- alpha vs basket={a_bh:+.1f}%/yr (t={t_bh:+.2f}, "
             f"{'significant' if stats['significant_vs_basket_95'] else 'not significant'})")
ax.set_ylabel("Cumulative return (%)")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "_PORTFOLIO_backtest.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print("\nDone.")
