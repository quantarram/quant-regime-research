"""
Kelly-style position sizing on top of the target-price strategy
(45_target_price_trading_strategy.py). Instead of a fixed 100% position
whenever the buy-low/sell-high state machine says "in", size the position
by conviction: f* = mu / sigma^2, the growth-optimal (Kelly) fraction,
using the model's own forecast each day as mu and sigma.

mu(t) = predicted median H-day-ahead log return (q0.5).
sigma(t) = predicted H-day-ahead dispersion, from the interquartile range
  (q0.75 - q0.25) / 1.349 (the normal-distribution IQR-to-sigma conversion).
Because both mu and sigma^2 scale linearly with the horizon H under a
random-walk/i.i.d.-increments assumption, their ratio mu_H/sigma_H^2 is
time-scale invariant -- it equals the daily-rebalanced Kelly fraction
directly, with no separate per-horizon rescaling needed. (Standard result
behind the continuous-time/Merton form of the Kelly criterion.)

Two disclosed, standard practitioner modifications to raw/full Kelly (full
Kelly is well known to be too aggressive under real-world parameter
uncertainty): HALF-Kelly (a conventional 0.5x multiplier) and a hard cap on
leverage (KELLY_CAP). Long-only, floored at 0 (matches the existing
strategy's long/flat design -- no shorting).

The Kelly fraction only scales the SIZE of exposure while the existing
buy-low/sell-high state machine says "in a position" -- it does not create
a new entry/exit signal. This only reallocates risk given the same
(already statistically insignificant, per 44/45/46) underlying signal; it
cannot manufacture edge that isn't there. Report honestly either way.

Run: python 47_kelly_sized_strategy.py
Output: kelly_strategy_results.json, kelly_strategy_returns.parquet,
        pnl_plots/_KELLY_<TICKER>.png, pnl_plots/_KELLY_SUMMARY_all_instruments.png
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
BUY_Q, SELL_Q = "0.25", "0.75"
IQR_TO_SIGMA = 1.349
HALF_KELLY = 0.5
KELLY_CAP = 2.0
SIGMA_FLOOR = 1e-4

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


def build_strategy(tkr):
    dec = decisions[tkr]
    horizon, winner = dec["horizon"], dec["price_based_winner"]
    sub = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon)]
    variants_present = sub["variant"].unique().tolist()
    if winner == "climatology":
        src = sub[sub["variant"] == "both"] if "both" in variants_present else sub[sub["variant"] == variants_present[0]]
        buy_col, sell_col, med_col = f"clim_q{BUY_Q}", f"clim_q{SELL_Q}", "clim_q0.5"
    else:
        src = sub[sub["variant"] == winner]
        buy_col, sell_col, med_col = f"q{BUY_Q}", f"q{SELL_Q}", "q0.5"
    src = src[src["date"] >= HOLDOUT_START].sort_values("date")
    if len(src) < 100:
        return None

    series = get_series(tkr)
    idx = series.index[(series.index >= src["date"].min()) & (series.index <= src["date"].max())]
    price = series.reindex(idx)

    buy_ret_q = pd.Series(src[buy_col].values, index=src["date"].values).reindex(idx).ffill()
    sell_ret_q = pd.Series(src[sell_col].values, index=src["date"].values).reindex(idx).ffill()
    med_ret_q = pd.Series(src[med_col].values, index=src["date"].values).reindex(idx).ffill()
    buy_target = (price * np.exp(buy_ret_q)).shift(1)
    sell_target = (price * np.exp(sell_ret_q)).shift(1)

    sigma = ((sell_ret_q - buy_ret_q) / IQR_TO_SIGMA).clip(lower=SIGMA_FLOOR)
    kelly_raw = med_ret_q / (sigma ** 2)
    kelly_frac = (HALF_KELLY * kelly_raw).clip(lower=0.0, upper=KELLY_CAP).shift(1)  # no look-ahead

    state = 0
    sizes = []
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
        sizes.append(state)
    in_position = pd.Series(sizes, index=idx, dtype=float)
    position_size = (in_position * kelly_frac.reindex(idx)).fillna(0.0)

    daily_ret = price.pct_change().fillna(0.0)
    applied = position_size.shift(1).fillna(0.0)
    gross = applied * daily_ret
    turnover = applied.diff().abs().fillna(0.0)
    net = gross - turnover * (COST_BPS / 10000.0)

    buyhold_cum = (1 + daily_ret).cumprod()
    strat_net_cum = (1 + net).cumprod()
    stats = {
        "horizon": horizon, "winner": winner, "n_days": int(len(daily_ret)),
        "n_buys": n_buys, "n_sells": n_sells,
        "avg_kelly_fraction_when_in": float(kelly_frac.reindex(idx)[in_position > 0].mean()) if (in_position > 0).any() else 0.0,
        "max_kelly_fraction": float(kelly_frac.reindex(idx).max()),
        "buy_hold_total_return_pct": float((buyhold_cum.iloc[-1] - 1) * 100),
        "strategy_net_total_return_pct": float((strat_net_cum.iloc[-1] - 1) * 100),
        "strategy_net_sharpe": float(net.mean() / net.std() * np.sqrt(252)) if net.std() > 0 else float("nan"),
        "buy_hold_sharpe": float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else float("nan"),
        "strategy_net_max_dd_pct": max_drawdown(strat_net_cum) * 100,
        "buy_hold_max_dd_pct": max_drawdown(buyhold_cum) * 100,
        "beats_buy_hold_net": bool(strat_net_cum.iloc[-1] > buyhold_cum.iloc[-1]),
    }
    return net, daily_ret, stats


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

    buyhold_cum = (1 + own_ret).cumprod()
    strat_cum = (1 + net).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(buyhold_cum.index, (buyhold_cum - 1) * 100, color="black", lw=1.3, label="Buy & hold")
    ax.plot(strat_cum.index, (strat_cum - 1) * 100, color="tab:orange", lw=1.3, label=f"Kelly-sized target-price strategy ({stats['winner']}), net")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    beats = "beats buy & hold" if stats["beats_buy_hold_net"] else "does NOT beat buy & hold"
    ax.set_title(f"{tkr} @ {stats['horizon']}d ({stats['winner']}): Kelly-sized strategy -- {beats}\n"
                 f"avg size when in={stats['avg_kelly_fraction_when_in']:.2f}x, alpha vs own buy&hold={a_own:+.1f}%/yr (t={t_own:+.2f})")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    safe_tkr = tkr.replace('=', '').replace('^', '')
    fig.savefig(os.path.join(PLOT_DIR, f"_KELLY_{safe_tkr}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

with open(os.path.join(OUT_DIR, "kelly_strategy_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
pd.DataFrame(net_returns_by_ticker).to_parquet(os.path.join(OUT_DIR, "kelly_strategy_returns.parquet"))

n_beat = sum(1 for r in results.values() if r["beats_buy_hold_net"])
n_alpha = sum(1 for r in results.values() if r["significant_vs_own_95"])
print(f"=== Kelly-sized (half-Kelly, capped at {KELLY_CAP}x) target-price strategy, {len(results)} instruments ===")
for tkr in sorted(results, key=lambda t: -results[t]["strategy_net_total_return_pct"]):
    r = results[tkr]
    flag = "BEATS" if r["beats_buy_hold_net"] else "does not beat"
    sig = "SIGNIFICANT" if r["significant_vs_own_95"] else "not sig"
    print(f"  {tkr} ({r['winner']}@{r['horizon']}d): net={r['strategy_net_total_return_pct']:+.1f}% "
          f"vs buyhold={r['buy_hold_total_return_pct']:+.1f}%, avg_size={r['avg_kelly_fraction_when_in']:.2f}x, "
          f"{flag} buy&hold, alpha={r['alpha_vs_own_pct']:+.1f}%/yr t={r['t_vs_own']:+.2f} [{sig}]")
print(f"\n{n_beat}/{len(results)} beat buy-and-hold net of costs; {n_alpha}/{len(results)} significant alpha vs own buy&hold")

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
ax.set_xlabel("Kelly-sized strategy net total return (%) -- black tick = buy & hold")
ax.set_title(f"Half-Kelly-sized buy-low/sell-high strategy vs buy & hold, all {len(results)} instruments\n"
             f"green = beats buy&hold net of costs, red = does not")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "_KELLY_SUMMARY_all_instruments.png"), dpi=115, bbox_inches="tight")
plt.close(fig)
print("\nDone.")
