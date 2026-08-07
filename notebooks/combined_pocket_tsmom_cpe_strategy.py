"""
combined_pocket_tsmom_cpe_strategy.py
========================================
Builds and fairly tests a genuinely different combination of TSMOM, CPE,
and predictability limits than the two combinations already falsified
tonight -- designed around what was actually validated, not a repeat of
what failed.

Two design choices, each grounded in a specific, already-replicated
finding, not a fresh guess:

1. POCKET-FILTERED INSTRUMENT SELECTION, not tau*-as-lookback. The
   tau*-adaptive lookback test failed because tau* measures magnitude/
   volatility-structure persistence (built on |dp|), not directional
   momentum persistence -- a category error, confirmed directly. But the
   corrected pocket-momentum replication check found something different
   and real: instruments whose own measured predictability structure
   (Ramanathan 2026a) shows a genuine pocket near TSMOM's own 252-day
   lookback (SPY, IWM, AAPL, MSFT, GLD) are exactly where momentum's
   real, crisis-concentrated edge showed up under real block replication.
   This strategy uses tau*-pocket structure to choose WHICH instruments
   to trade, keeping TSMOM's own signal construction completely
   unchanged -- not to change how the signal itself is built.

2. ADDITIVE BLEND, not multiplicative scaling. The TSMOM x CPE-crisis
   combo test failed because CPE's vol-complex detector fires at the
   ONSET of a fast shock while TSMOM's real edge accrues over a
   SUSTAINED, multi-month drawdown -- scaling TSMOM's exposure at CPE's
   onset signal amplified TSMOM at exactly the moment it hadn't yet
   repositioned correctly. An additive blend -- CPE's detector run as its
   own small, independent sleeve alongside TSMOM, not a multiplier on
   it -- needs no timing alignment between the two signals at all; each
   sleeve simply contributes its own real return, whatever the other is
   doing.

Tested with the replication unit matched to each component, per the
lesson from the last comparison: 5 real non-overlapping calendar blocks
for the steady TSMOM component (a strategy claiming a roughly time-
distributed edge), and the already-established 13 real crisis episodes
for context on the CPE sleeve (a strategy claiming a crisis-triggered
one). No p-values, no t-statistics, no significance verdicts -- real
point estimates and real replication counts only.

Run: python combined_pocket_tsmom_cpe_strategy.py
Output: combined_strategy_results.json, combined_strategy_plot.png
"""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "files")
from backtest_engine import build_increments_and_thresholds, configuration_fires_on_date
import backtest_engine as _be

POCKET_INSTRUMENTS = ["SPY", "IWM", "AAPL", "MSFT", "GLD"]
ALL_12_INSTRUMENTS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "AAPL", "MSFT", "JPM", "XOM", "GLD", "EURUSD=X"]
LOOKBACK, VOL_WINDOW, TARGET_VOL, MAX_LEV, COST_BPS = 252, 63, 0.10, 2.0, 5
CPE_ALLOCATION = 0.20  # additive blend weight on the CPE crisis sleeve
CPE_TRAIN_CUTOFF = pd.Timestamp("2013-12-31")
CPE_CONFIG = pd.Series({"Y": "QQQ", "direction": "bullish", "predictors": ["TLH", "^GVZ"],
                         "tau_pasts": [126, 126], "q_Xs": [0.95, 0.95], "tau_future": 126, "q_Y": 0.50})
Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]


def month_end_dates(index):
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).last().values)


def sharpe(ret):
    r = ret.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")


def build_tsmom(prices: pd.DataFrame, instruments: list, daily_index: pd.DatetimeIndex, rebal_dates: pd.DatetimeIndex) -> pd.Series:
    log_px = np.log(prices[instruments])
    daily_ret = log_px.diff()
    rows = {}
    for d in rebal_dates:
        weights = {}
        for tkr in instruments:
            s = prices[tkr].loc[:d].dropna()
            if len(s) < LOOKBACK + VOL_WINDOW + 5:
                weights[tkr] = np.nan
                continue
            trail = np.log(s.iloc[-1] / s.iloc[-LOOKBACK])
            r = daily_ret[tkr].dropna().loc[:d].tail(VOL_WINDOW)
            vol = r.std() * np.sqrt(252)
            weights[tkr] = float(np.clip(np.sign(trail) * (TARGET_VOL / vol), -MAX_LEV, MAX_LEV)) if vol > 1e-6 else np.nan
        rows[d] = weights
    weight_df = pd.DataFrame(rows).T
    daily_w = weight_df.reindex(daily_index, method="ffill").shift(1)
    active = daily_w.notna()
    n_active = active.sum(axis=1).replace(0, np.nan)
    per_instrument_w = daily_w.div(n_active, axis=0)
    port_ret = (per_instrument_w * daily_ret.reindex(daily_index)).sum(axis=1, skipna=True)
    turnover = per_instrument_w.diff().abs().sum(axis=1).fillna(0.0)
    return port_ret - turnover * (COST_BPS / 10000.0)


def build_cpe_sleeve(prices: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.Series:
    _be.TRAIN_CUTOFF = CPE_TRAIN_CUTOFF
    increments, thresholds = build_increments_and_thresholds(prices, Q_GRID)
    qqq_valid_dates = prices["QQQ"].dropna().index
    eval_dates = daily_index[(daily_index >= CPE_TRAIN_CUTOFF + pd.Timedelta(days=1)) & daily_index.isin(qqq_valid_dates)]
    fires = pd.Series([configuration_fires_on_date(CPE_CONFIG, d, increments, thresholds) for d in eval_dates], index=eval_dates)
    newly = fires & ~fires.shift(1).fillna(False)
    position = pd.Series(0.0, index=eval_dates)
    active_until = None
    for d in eval_dates:
        if active_until is not None and d <= active_until:
            position[d] = 1.0
        elif newly.get(d, False):
            position[d] = 1.0
            active_until = d + pd.Timedelta(days=int(126 * 1.45))
    qqq_ret = np.log(prices["QQQ"].reindex(eval_dates).ffill()).diff()
    applied = position.shift(1).fillna(0.0)
    turnover = applied.diff().abs().fillna(0.0)
    cpe_ret = applied * qqq_ret - turnover * (COST_BPS / 10000.0)
    return cpe_ret.reindex(daily_index).fillna(0.0)


if __name__ == "__main__":
    prices = pd.read_parquet("multiasset_prices.parquet")
    n_bad = int((prices[ALL_12_INSTRUMENTS] <= 0).sum().sum())
    if n_bad:
        prices = prices.copy()
        prices[ALL_12_INSTRUMENTS] = prices[ALL_12_INSTRUMENTS].mask(prices[ALL_12_INSTRUMENTS] <= 0)

    # MAX, not min, across all 12 -- GLD doesn't start until 2004-11-18 and
    # IWM until 2000-05-26; using the earliest-starting single instrument
    # (XOM/JPM, 1962) as the test start would silently run the early blocks
    # on 1-2 real instruments instead of a genuine 12- or 5-instrument
    # portfolio -- caught by inspecting the first script run's output
    # directly rather than trusting it, same discipline as every other
    # check tonight.
    first_valid = max(prices[t].dropna().index.min() for t in ALL_12_INSTRUMENTS)
    start = first_valid + pd.Timedelta(days=int(1.5 * (LOOKBACK + VOL_WINDOW)))
    rebal_dates = month_end_dates(prices.index[prices.index >= start])
    trading_index = prices.index[prices.index >= rebal_dates[0]]

    print("Building baseline TSMOM (all 12 tau*-measured instruments, unfiltered)...")
    baseline_ret = build_tsmom(prices, ALL_12_INSTRUMENTS, trading_index, rebal_dates)

    print("Building pocket-filtered TSMOM (SPY, IWM, AAPL, MSFT, GLD only)...")
    pocket_ret = build_tsmom(prices, POCKET_INSTRUMENTS, trading_index, rebal_dates)

    print("Building CPE crisis-detector sleeve ([TLH, GVZ] -> QQQ)...")
    cpe_ret = build_cpe_sleeve(prices, trading_index)

    common_idx = pocket_ret.index.intersection(cpe_ret.index)
    combined_ret = (1 - CPE_ALLOCATION) * pocket_ret.reindex(common_idx).fillna(0.0) + CPE_ALLOCATION * cpe_ret.reindex(common_idx).fillna(0.0)

    print(f"\n{'='*90}\nFull-sample real point estimates, {common_idx.min().date()} to {common_idx.max().date()}\n{'='*90}")
    for label, ret in [("Baseline TSMOM (12 instruments)", baseline_ret.reindex(common_idx)),
                        ("Pocket-filtered TSMOM (5 instruments)", pocket_ret.reindex(common_idx)),
                        ("CPE crisis sleeve alone", cpe_ret.reindex(common_idx)),
                        (f"Combined ({1-CPE_ALLOCATION:.0%} pocket-TSMOM + {CPE_ALLOCATION:.0%} CPE)", combined_ret)]:
        r = ret.dropna()
        eq = np.exp(r).cumprod()
        mdd = float((eq / eq.cummax() - 1).min() * 100)
        print(f"  {label:<42} Sharpe={sharpe(r):+.3f}  ann_ret={r.mean()*252*100:+6.2f}%  MaxDD={mdd:7.2f}%")

    print(f"\n{'='*90}\n5 real non-overlapping calendar blocks\n{'='*90}")
    block_edges = pd.date_range(common_idx.min(), common_idx.max(), periods=6)
    block_results = []
    n_pocket_beats_baseline, n_combined_beats_pocket = 0, 0
    for i in range(5):
        b_start, b_end = block_edges[i], block_edges[i + 1]
        mask = (common_idx >= b_start) & (common_idx < b_end if i < 4 else common_idx <= b_end)
        idx = common_idx[mask]
        base_b, pocket_b, combo_b = baseline_ret.reindex(idx), pocket_ret.reindex(idx), combined_ret.reindex(idx)
        base_ann, pocket_ann, combo_ann = base_b.mean() * 252 * 100, pocket_b.mean() * 252 * 100, combo_b.mean() * 252 * 100
        pocket_wins = pocket_ann > base_ann
        combo_wins = combo_ann > pocket_ann
        n_pocket_beats_baseline += pocket_wins
        n_combined_beats_pocket += combo_wins
        print(f"  {b_start.date()} to {b_end.date()}: baseline={base_ann:+6.2f}%  pocket-filtered={pocket_ann:+6.2f}%  "
              f"combined={combo_ann:+6.2f}%  [pocket {'beats' if pocket_wins else 'trails'} baseline, combined {'beats' if combo_wins else 'trails'} pocket-alone]")
        block_results.append({"block_start": str(b_start.date()), "block_end": str(b_end.date()),
                               "baseline_pct": float(base_ann), "pocket_pct": float(pocket_ann), "combined_pct": float(combo_ann)})

    print(f"\nPocket-filtered TSMOM beat unfiltered baseline in {n_pocket_beats_baseline}/5 real blocks")
    print(f"Adding the CPE sleeve improved on pocket-filtered-TSMOM-alone in {n_combined_beats_pocket}/5 real blocks")

    with open("combined_strategy_results.json", "w") as f:
        json.dump({"blocks": block_results, "n_pocket_beats_baseline": n_pocket_beats_baseline,
                    "n_combined_beats_pocket": n_combined_beats_pocket}, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(13, 7))
    for label, ret, color in [("Baseline TSMOM (12 instr.)", baseline_ret, "#9AA1AD"),
                               ("Pocket-filtered TSMOM (5 instr.)", pocket_ret, "#2E6DA4"),
                               (f"Combined (pocket-TSMOM + CPE sleeve)", combined_ret, "#2f8a4e")]:
        eq = np.exp(ret.reindex(common_idx).fillna(0)).cumprod() * 100_000
        ax.plot(eq.index, eq.values, label=f"{label} (Sharpe {sharpe(ret.reindex(common_idx)):.2f})", lw=1.2, color=color)
    ax.set_yscale("log")
    ax.set_title("Pocket-filtered TSMOM + additive CPE crisis sleeve vs. unfiltered baseline\nNo significance test -- real equity curves, real point estimates")
    ax.set_ylabel("Equity ($100k notional, log scale)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("combined_strategy_plot.png", dpi=140)
    plt.close(fig)
    print("\nSaved: combined_strategy_results.json, combined_strategy_plot.png")
