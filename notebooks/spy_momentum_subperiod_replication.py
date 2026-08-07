"""
spy_momentum_subperiod_replication.py
========================================
Re-does the pocket-momentum group comparison's SPY result without a
Welch's t-test or any confidence interval on a group mean. Instead: does
SPY's own momentum result -- the standout of the 12-instrument panel --
show up repeatedly across genuinely separate, non-overlapping real
historical periods of SPY's own history, or was it earned in one lucky
stretch? Real out-of-sample replication, not a classical significance
test on a single pooled estimate.

Method: split SPY's full available momentum-evaluation history into
five non-overlapping ~6-7-year real calendar blocks (a mechanical,
roughly-equal-length split, not chosen to flatter any particular block),
and report each block's real point-estimate momentum return and Sharpe
directly, side by side -- the same standard TSMOM signal (252d lookback,
63d vol window) used throughout tonight, unchanged.

No significance/randomisation-test games, no p-values, no t-statistics --
real point estimates, block by block, reported directly.

Run: python spy_momentum_subperiod_replication.py
Output: spy_momentum_subperiod_results.json, spy_momentum_subperiod_plot.png
"""
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOOKBACK, VOL_WINDOW, TARGET_VOL, MAX_LEV, COST_BPS = 252, 63, 0.10, 2.0, 5
N_BLOCKS = 5


def month_end_dates(index):
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).last().values)


def sharpe(ret):
    r = ret.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")


if __name__ == "__main__":
    prices = pd.read_parquet("multiasset_prices.parquet")
    spy = prices["SPY"].dropna()
    daily_ret = np.log(spy).diff()

    start = spy.index.min() + pd.Timedelta(days=int(1.5 * (LOOKBACK + VOL_WINDOW)))
    rebal = month_end_dates(spy.index[spy.index >= start])
    trading_index = spy.index[spy.index >= rebal[0]]

    rows = {}
    for d in rebal:
        s = spy.loc[:d]
        if len(s) < LOOKBACK + VOL_WINDOW + 5:
            rows[d] = np.nan
            continue
        trail = np.log(s.iloc[-1] / s.iloc[-LOOKBACK])
        r = daily_ret.loc[:d].tail(VOL_WINDOW)
        vol = r.std() * np.sqrt(252)
        rows[d] = float(np.clip(np.sign(trail) * (TARGET_VOL / vol), -MAX_LEV, MAX_LEV)) if vol > 1e-6 else np.nan
    w = pd.Series(rows).reindex(trading_index, method="ffill").shift(1)
    strat_ret = w * daily_ret.reindex(trading_index)
    turnover = w.diff().abs().fillna(0.0)
    strat_ret = (strat_ret - turnover * (COST_BPS / 10000.0)).dropna()
    bh_ret = daily_ret.reindex(strat_ret.index)

    print(f"SPY momentum, full sample {strat_ret.index.min().date()} to {strat_ret.index.max().date()}: "
          f"Sharpe={sharpe(strat_ret):.3f}, ann_ret={strat_ret.mean()*252*100:+.2f}%")
    print(f"SPY buy-and-hold, same period: Sharpe={sharpe(bh_ret):.3f}, ann_ret={bh_ret.mean()*252*100:+.2f}%")

    block_edges = pd.date_range(strat_ret.index.min(), strat_ret.index.max(), periods=N_BLOCKS + 1)
    results = []
    print(f"\n{N_BLOCKS} non-overlapping real historical blocks (mechanical equal split, not curated):")
    for i in range(N_BLOCKS):
        b_start, b_end = block_edges[i], block_edges[i + 1]
        mask = (strat_ret.index >= b_start) & (strat_ret.index < b_end if i < N_BLOCKS - 1 else strat_ret.index <= b_end)
        s_block, bh_block = strat_ret[mask], bh_ret[mask]
        excess_ann = (s_block.mean() - bh_block.mean()) * 252 * 100
        print(f"  {b_start.date()} to {b_end.date()} (n={mask.sum()}d): "
              f"momentum Sharpe={sharpe(s_block):+.2f} ann_ret={s_block.mean()*252*100:+6.2f}%  |  "
              f"buy-hold Sharpe={sharpe(bh_block):+.2f} ann_ret={bh_block.mean()*252*100:+6.2f}%  |  "
              f"excess over buy-hold={excess_ann:+6.2f}%/yr")
        results.append({
            "block_start": str(b_start.date()), "block_end": str(b_end.date()), "n_days": int(mask.sum()),
            "momentum_sharpe": sharpe(s_block), "momentum_ann_ret_pct": float(s_block.mean() * 252 * 100),
            "buyhold_sharpe": sharpe(bh_block), "buyhold_ann_ret_pct": float(bh_block.mean() * 252 * 100),
            "excess_over_buyhold_pct": float(excess_ann),
        })

    n_blocks_positive_excess = sum(1 for r in results if r["excess_over_buyhold_pct"] > 0)
    print(f"\nMomentum beat SPY's own buy-and-hold in {n_blocks_positive_excess} of {N_BLOCKS} real, non-overlapping historical blocks")

    with open("spy_momentum_subperiod_results.json", "w") as f:
        json.dump({"full_sample": {"momentum_sharpe": sharpe(strat_ret), "buyhold_sharpe": sharpe(bh_ret)},
                    "blocks": results, "n_blocks_positive_excess": n_blocks_positive_excess}, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [f"{r['block_start'][:4]}-{r['block_end'][:4]}" for r in results]
    x = np.arange(N_BLOCKS)
    ax.bar(x - 0.2, [r["momentum_ann_ret_pct"] for r in results], width=0.4, label="Momentum", color="#2E6DA4")
    ax.bar(x + 0.2, [r["buyhold_ann_ret_pct"] for r in results], width=0.4, label="Buy-and-hold", color="#9AA1AD")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Annualized return (%)")
    ax.set_title(f"SPY momentum vs. buy-and-hold, {N_BLOCKS} real non-overlapping historical blocks\n"
                  f"Momentum beat buy-and-hold in {n_blocks_positive_excess}/{N_BLOCKS} blocks -- no significance test, direct replication")
    ax.legend()
    fig.tight_layout()
    fig.savefig("spy_momentum_subperiod_plot.png", dpi=140)
    plt.close(fig)
    print("\nSaved: spy_momentum_subperiod_results.json, spy_momentum_subperiod_plot.png")
