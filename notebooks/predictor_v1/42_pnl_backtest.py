"""
Economic backtest: for each instrument, use the master model's actual
holdout-period winner (climatology or one of credit_only/vix_only/both,
from 41_final_decision_master_model.py) to drive a simple long/flat
trading rule, and see whether it would have actually made money on the
genuinely held-out period -- not just scored well on FSS or MAPE.

Rule (simple, standard, not over-engineered): each day, take a position
based on the SIGN of that day's predicted median H-day-forward return from
the winning model (long if positive, flat if <=0). Positions rebalance
daily using the freshest available forecast, held one day at a time --
this is the standard way to convert a continuously-updating multi-day-
ahead forecast into a live daily P&L series without double-counting
overlapping windows. Position decided using info as of day t is applied to
day t+1's realized return (no look-ahead).

Reports gross (no costs) and net-of-costs (5bps per position change, a
standard round-number assumption, disclosed not hidden) total return,
annualized Sharpe (a plain point-estimate descriptive statistic -- not a
resampling-based significance test, consistent with this project's
standing rejection of randomization testing), max drawdown, and turnover,
each vs. buy-and-hold over the identical holdout window. This is exactly
the check that should kill any "prediction" that is really just
yesterday's price restated: a signal with real turnover but no genuine
edge will underperform buy-and-hold once costs are included, regardless of
how good its FSS or MAPE numbers looked.

Run: python 42_pnl_backtest.py
Output: pnl_backtest_results.json, pnl_plots/<TICKER>.png
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
COST_BPS = 5  # per position change, one-way, a standard disclosed assumption

decisions = json.load(open(os.path.join(OUT_DIR, "master_model_final_decision.json")))
oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))
PROXY_TICKERS = ("IYR", "VOX")


def get_series(tkr):
    return (prices_proxy[tkr] if tkr in PROXY_TICKERS else prices[tkr]).dropna()


def max_drawdown(cum):
    running_max = cum.cummax()
    dd = cum / running_max - 1
    return float(dd.min())


results = {}
for tkr in sorted(decisions.keys()):
    dec = decisions[tkr]
    horizon, winner = dec["horizon"], dec["price_based_winner"]

    sub_th = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon)]
    variants_present = sub_th["variant"].unique().tolist()
    if winner == "climatology":
        src = sub_th[sub_th["variant"] == "both"] if "both" in variants_present else \
            sub_th[sub_th["variant"] == variants_present[0]]
        pred_col = "clim_q0.5"
    else:
        src = sub_th[sub_th["variant"] == winner]
        pred_col = "q0.5"
    src = src[src["date"] >= HOLDOUT_START].sort_values("date")
    if len(src) < 100:
        continue

    series = get_series(tkr)
    daily_ret = series.pct_change().dropna()

    position = pd.Series((src[pred_col].values > 0).astype(float), index=src["date"].values)
    position = position[~position.index.duplicated(keep="last")]

    common_idx = daily_ret.index[(daily_ret.index >= position.index.min()) & (daily_ret.index <= position.index.max())]
    position_daily = position.reindex(common_idx).ffill().fillna(0.0)
    realized_ret = daily_ret.reindex(common_idx)

    # position decided at close of day t (using that day's forecast) applied to day t+1's return
    applied_position = position_daily.shift(1).fillna(0.0)
    strategy_ret_gross = applied_position * realized_ret

    turnover = applied_position.diff().abs().fillna(0.0)
    cost = turnover * (COST_BPS / 10000.0)
    strategy_ret_net = strategy_ret_gross - cost

    buyhold_cum = (1 + realized_ret).cumprod()
    strat_gross_cum = (1 + strategy_ret_gross).cumprod()
    strat_net_cum = (1 + strategy_ret_net).cumprod()

    n_days = len(realized_ret)
    ann_factor = 252 / n_days
    stats = {
        "n_days": int(n_days),
        "buy_hold_total_return_pct": float((buyhold_cum.iloc[-1] - 1) * 100),
        "strategy_gross_total_return_pct": float((strat_gross_cum.iloc[-1] - 1) * 100),
        "strategy_net_total_return_pct": float((strat_net_cum.iloc[-1] - 1) * 100),
        "strategy_gross_sharpe": float(strategy_ret_gross.mean() / strategy_ret_gross.std() * np.sqrt(252)) if strategy_ret_gross.std() > 0 else float("nan"),
        "strategy_net_sharpe": float(strategy_ret_net.mean() / strategy_ret_net.std() * np.sqrt(252)) if strategy_ret_net.std() > 0 else float("nan"),
        "buy_hold_sharpe": float(realized_ret.mean() / realized_ret.std() * np.sqrt(252)) if realized_ret.std() > 0 else float("nan"),
        "strategy_gross_max_dd_pct": max_drawdown(strat_gross_cum) * 100,
        "strategy_net_max_dd_pct": max_drawdown(strat_net_cum) * 100,
        "buy_hold_max_dd_pct": max_drawdown(buyhold_cum) * 100,
        "pct_days_long": float(applied_position.mean() * 100),
        "n_position_changes": int(turnover.sum()),
        "beats_buy_hold_net": bool(strat_net_cum.iloc[-1] > buyhold_cum.iloc[-1]),
    }
    results[tkr] = {"horizon": horizon, "winner": winner, **stats}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(buyhold_cum.index, (buyhold_cum - 1) * 100, color="black", lw=1.3, label="Buy & hold")
    ax.plot(strat_gross_cum.index, (strat_gross_cum - 1) * 100, color="tab:blue", lw=1.2, ls="--",
             label=f"Strategy ({winner}), gross")
    ax.plot(strat_net_cum.index, (strat_net_cum - 1) * 100, color="tab:red", lw=1.3,
             label=f"Strategy ({winner}), net of {COST_BPS}bps costs")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    beats = "beats buy & hold" if stats["beats_buy_hold_net"] else "does NOT beat buy & hold"
    ax.set_title(f"{tkr} @ {horizon}d ({winner}): cumulative return, holdout period -- {beats} net of costs")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    safe_tkr = tkr.replace('=', '').replace('^', '')
    fig.savefig(os.path.join(PLOT_DIR, f"{safe_tkr}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

with open(os.path.join(OUT_DIR, "pnl_backtest_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)

print("=== P&L backtest, holdout period, net of 5bps costs per position change ===")
n_beat = sum(1 for r in results.values() if r["beats_buy_hold_net"])
for tkr in sorted(results, key=lambda t: -results[t]["strategy_net_total_return_pct"]):
    r = results[tkr]
    flag = "BEATS buy&hold" if r["beats_buy_hold_net"] else "does not beat buy&hold"
    print(f"  {tkr} ({r['winner']}@{r['horizon']}d): strategy_net={r['strategy_net_total_return_pct']:+.1f}%, "
          f"buy_hold={r['buy_hold_total_return_pct']:+.1f}%, net_sharpe={r['strategy_net_sharpe']:+.2f}, "
          f"n_position_changes={r['n_position_changes']}, {flag}")
print(f"\n{n_beat}/{len(results)} instruments' strategy beats buy-and-hold net of costs over the holdout period")
print("\nDone.")
