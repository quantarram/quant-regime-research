"""
CORRECTION to 42_pnl_backtest.py: that script used the crudest possible
rule -- go long/flat based only on the SIGN of the predicted median return.
It never used the fact that the master model outputs a full predicted
price distribution (q0.1..q0.9), and it never modeled the "buy low, sell
high" behavior an actual trader would use: prepare to buy when the model
signals a low is coming, actually buy once price genuinely reaches that
low; prepare to sell once holding, actually sell once price reaches the
model's predicted high.

Strategy (a rolling limit-order rule driven by the model's own predicted
bands, not a fixed external indicator):
  - Each day t, the winning (horizon H, variant) config's q0.25/q0.75
    predicted H-day-ahead log-return quantiles are converted to price
    targets: buy_target(t) = price(t) * exp(q0.25), sell_target(t) =
    price(t) * exp(q0.75). (climatology winners use clim_q0.25/clim_q0.75.)
  - State machine, FLAT or LONG, starting FLAT:
      FLAT: if tomorrow's close <= today's buy_target -> BUY at that close.
      LONG: if tomorrow's close >= today's sell_target -> SELL at that close.
  - Targets refresh daily with the freshest forecast until filled (a
    resting order that's repriced every day, not frozen at entry) --
    this is what "be prepared to buy/sell, then act when the actual price
    gets there" means mechanically.
  - No look-ahead: day t's forecast (and the price target built from it)
    is only used to trigger a trade on day t+1's close.
  - Same disclosed 5bps cost per position change as the baseline test.
q0.25/q0.75 (not more extreme quantiles) is a disclosed, deliberately
moderate choice -- narrow enough to reflect a real, not extreme, forecast
belief; wide enough to trigger sometimes rather than never.

Also runs the CORRECT alpha test from the prior correction (regress
strategy net return on the SAME INSTRUMENT's own buy-and-hold, not a
generic market index) on this new strategy.

Run: python 45_target_price_trading_strategy.py
Output: target_price_strategy_results.json, pnl_plots/_TARGET_<TICKER>.png,
        pnl_plots/_TARGET_SUMMARY_all_instruments.png
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
os.makedirs(PLOT_DIR, exist_ok=True)
HOLDOUT_START = pd.Timestamp("2022-01-01")
COST_BPS = 5
BUY_Q, SELL_Q = "0.25", "0.75"

decisions = json.load(open(os.path.join(OUT_DIR, "master_model_final_decision.json")))
oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))
PROXY_TICKERS = ("IYR", "VOX")


def get_series(tkr):
    return (prices_proxy[tkr] if tkr in PROXY_TICKERS else prices[tkr]).dropna()


def max_drawdown(cum):
    running_max = cum.cummax()
    return float((cum / running_max - 1).min())


def build_strategy(tkr):
    dec = decisions[tkr]
    horizon, winner = dec["horizon"], dec["price_based_winner"]
    sub = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon)]
    variants_present = sub["variant"].unique().tolist()
    if winner == "climatology":
        src = sub[sub["variant"] == "both"] if "both" in variants_present else sub[sub["variant"] == variants_present[0]]
        buy_col, sell_col = f"clim_q{BUY_Q}", f"clim_q{SELL_Q}"
    else:
        src = sub[sub["variant"] == winner]
        buy_col, sell_col = f"q{BUY_Q}", f"q{SELL_Q}"
    src = src[src["date"] >= HOLDOUT_START].sort_values("date")
    if len(src) < 100:
        return None

    series = get_series(tkr)
    idx = series.index[(series.index >= src["date"].min()) & (series.index <= src["date"].max())]
    price = series.reindex(idx)

    buy_ret_q = pd.Series(src[buy_col].values, index=src["date"].values).reindex(idx).ffill()
    sell_ret_q = pd.Series(src[sell_col].values, index=src["date"].values).reindex(idx).ffill()
    buy_target = (price * np.exp(buy_ret_q)).shift(1)   # yesterday's forecast, no look-ahead
    sell_target = (price * np.exp(sell_ret_q)).shift(1)

    state = 0  # 0=flat, 1=long
    positions = []
    n_buys, n_sells = 0, 0
    for i in range(len(idx)):
        px = price.iloc[i]
        if state == 0:
            bt = buy_target.iloc[i]
            if pd.notna(bt) and px <= bt:
                state = 1
                n_buys += 1
        else:
            st = sell_target.iloc[i]
            if pd.notna(st) and px >= st:
                state = 0
                n_sells += 1
        positions.append(state)
    position = pd.Series(positions, index=idx, dtype=float)

    daily_ret = price.pct_change().fillna(0.0)
    applied = position.shift(1).fillna(0.0)  # position decided using day t info, applied to day t+1 return
    gross = applied * daily_ret
    turnover = applied.diff().abs().fillna(0.0)
    net = gross - turnover * (COST_BPS / 10000.0)

    buyhold_cum = (1 + daily_ret).cumprod()
    strat_net_cum = (1 + net).cumprod()
    n_days = len(daily_ret)
    stats = {
        "horizon": horizon, "winner": winner, "n_days": int(n_days),
        "n_buys": n_buys, "n_sells": n_sells, "n_trades": n_buys + n_sells,
        "pct_days_long": float(applied.mean() * 100),
        "buy_hold_total_return_pct": float((buyhold_cum.iloc[-1] - 1) * 100),
        "strategy_net_total_return_pct": float((strat_net_cum.iloc[-1] - 1) * 100),
        "strategy_net_sharpe": float(net.mean() / net.std() * np.sqrt(252)) if net.std() > 0 else float("nan"),
        "buy_hold_sharpe": float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else float("nan"),
        "strategy_net_max_dd_pct": max_drawdown(strat_net_cum) * 100,
        "buy_hold_max_dd_pct": max_drawdown(buyhold_cum) * 100,
        "beats_buy_hold_net": bool(strat_net_cum.iloc[-1] > buyhold_cum.iloc[-1]),
    }
    return net, daily_ret, stats


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


results = {}
net_returns_by_ticker = {}
for tkr in sorted(decisions.keys()):
    out = build_strategy(tkr)
    if out is None:
        continue
    net, own_ret, stats = out
    a_own, se_own, b_own, t_own = jensen_alpha(net, own_ret)
    stats.update({
        "alpha_vs_own_pct": a_own, "alpha_vs_own_ci_lo_pct": a_own - 1.96 * se_own,
        "alpha_vs_own_ci_hi_pct": a_own + 1.96 * se_own, "beta_vs_own": b_own,
        "t_vs_own": t_own, "significant_vs_own_95": bool(abs(t_own) >= 1.96),
    })
    results[tkr] = stats
    net_returns_by_ticker[tkr] = net

    price = get_series(tkr).reindex(net.index)
    buyhold_cum = (1 + own_ret).cumprod()
    strat_cum = (1 + net).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(buyhold_cum.index, (buyhold_cum - 1) * 100, color="black", lw=1.3, label="Buy & hold")
    ax.plot(strat_cum.index, (strat_cum - 1) * 100, color="tab:purple", lw=1.3, label=f"Target-price strategy ({stats['winner']}), net")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    beats = "beats buy & hold" if stats["beats_buy_hold_net"] else "does NOT beat buy & hold"
    ax.set_title(f"{tkr} @ {stats['horizon']}d ({stats['winner']}): buy-low/sell-high target strategy -- {beats}\n"
                 f"{stats['n_trades']} trades, alpha vs own buy&hold={a_own:+.1f}%/yr (t={t_own:+.2f})")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    safe_tkr = tkr.replace('=', '').replace('^', '')
    fig.savefig(os.path.join(PLOT_DIR, f"_TARGET_{safe_tkr}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

with open(os.path.join(OUT_DIR, "target_price_strategy_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)

n_beat = sum(1 for r in results.values() if r["beats_buy_hold_net"])
n_alpha = sum(1 for r in results.values() if r["significant_vs_own_95"])
print(f"=== Target-price (buy-low/sell-high) strategy, {len(results)} instruments ===")
for tkr in sorted(results, key=lambda t: -results[t]["strategy_net_total_return_pct"]):
    r = results[tkr]
    flag = "BEATS" if r["beats_buy_hold_net"] else "does not beat"
    sig = "SIGNIFICANT" if r["significant_vs_own_95"] else "not sig"
    print(f"  {tkr} ({r['winner']}@{r['horizon']}d): net={r['strategy_net_total_return_pct']:+.1f}% "
          f"vs buyhold={r['buy_hold_total_return_pct']:+.1f}%, {r['n_trades']} trades, {flag} buy&hold, "
          f"alpha={r['alpha_vs_own_pct']:+.1f}%/yr t={r['t_vs_own']:+.2f} [{sig}]")
print(f"\n{n_beat}/{len(results)} beat buy-and-hold net of costs; {n_alpha}/{len(results)} significant alpha vs own buy&hold")

# Save the per-ticker net return series for the portfolio backtest
pd.DataFrame(net_returns_by_ticker).to_parquet(os.path.join(OUT_DIR, "target_price_strategy_returns.parquet"))

tickers_sorted = sorted(results, key=lambda t: -results[t]["strategy_net_total_return_pct"])
fig, ax = plt.subplots(figsize=(9, 8))
y_pos = np.arange(len(tickers_sorted))
colors = ["#2f8a4e" if results[t]["beats_buy_hold_net"] else "#b0492f" for t in tickers_sorted]
ax.barh(y_pos, [results[t]["strategy_net_total_return_pct"] for t in tickers_sorted], color=colors, height=0.6, zorder=3)
for i, t in enumerate(tickers_sorted):
    ax.plot(results[t]["buy_hold_total_return_pct"], i, marker="|", color="black", markersize=14, mew=2, zorder=4)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{t} ({results[t]['winner']}@{results[t]['horizon']}d)" for t in tickers_sorted], fontsize=9)
ax.axvline(0, color="gray", lw=0.6)
ax.set_xlabel("Target-price strategy net total return (%) -- black tick = buy & hold")
ax.set_title(f"Buy-low/sell-high target-price strategy vs buy & hold, all {len(results)} instruments\n"
             f"green = beats buy&hold net of costs, red = does not")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "_TARGET_SUMMARY_all_instruments.png"), dpi=115, bbox_inches="tight")
plt.close(fig)
print("\nDone.")
