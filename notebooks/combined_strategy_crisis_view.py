"""
combined_strategy_crisis_view.py
==================================
A Figure-1-style crisis view (tsmom_crisis_alpha_check.py's two-panel
equity-curve + shaded-drawdown convention) for this program's own
flagship decision-layer construction: pocket-filtered TSMOM with an
additive CPE crisis sleeve (combined_pocket_tsmom_cpe_strategy.py),
shown against the unfiltered TSMOM baseline and a passive equal-weight
benchmark of the identical 12-instrument universe.

Reuses combined_pocket_tsmom_cpe_strategy.py's own return-series
functions directly (build_tsmom, build_cpe_sleeve) rather than
re-deriving them, so the equity curves here are exactly the same
series already reported in Section 5.1/5.2 of the paper -- no new
numbers, just a different, more diagnostic view of the same result.

Crisis episodes are defined the same way as Figure 1: passive
equal-weight drawdown <= -10% from its own running peak, on THIS
strategy's own 12-instrument universe (not the 29-instrument universe
used for the original TSMOM benchmark) -- the internally consistent
choice, since the combined strategy and its baseline both trade this
12-instrument universe specifically.

Run: python combined_strategy_crisis_view.py
Output: combined_strategy_crisis_view.png
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from combined_pocket_tsmom_cpe_strategy import (
    ALL_12_INSTRUMENTS, POCKET_INSTRUMENTS, CPE_ALLOCATION,
    build_tsmom, build_cpe_sleeve, month_end_dates, sharpe,
)

CRISIS_THRESHOLD = -0.10
MIN_EPISODE_DAYS = 15


def cluster_episodes(mask: pd.Series, min_days: int) -> list:
    dates = mask.index[mask.values]
    if len(dates) == 0:
        return []
    episodes, current = [], [dates[0]]
    for d in dates[1:]:
        if (d - current[-1]).days > 10:
            if len(current) >= min_days:
                episodes.append((current[0], current[-1]))
            current = [d]
        else:
            current.append(d)
    if len(current) >= min_days:
        episodes.append((current[0], current[-1]))
    return episodes


if __name__ == "__main__":
    prices = pd.read_parquet("multiasset_prices.parquet")
    n_bad = int((prices[ALL_12_INSTRUMENTS] <= 0).sum().sum())
    if n_bad:
        prices = prices.copy()
        prices[ALL_12_INSTRUMENTS] = prices[ALL_12_INSTRUMENTS].mask(prices[ALL_12_INSTRUMENTS] <= 0)

    first_valid = max(prices[t].dropna().index.min() for t in ALL_12_INSTRUMENTS)
    start = first_valid + pd.Timedelta(days=int(1.5 * (252 + 63)))
    rebal_dates = month_end_dates(prices.index[prices.index >= start])
    trading_index = prices.index[prices.index >= rebal_dates[0]]

    print("Rebuilding baseline TSMOM, pocket-filtered TSMOM, CPE sleeve, and combined strategy...")
    baseline_ret = build_tsmom(prices, ALL_12_INSTRUMENTS, trading_index, rebal_dates)
    pocket_ret = build_tsmom(prices, POCKET_INSTRUMENTS, trading_index, rebal_dates)
    cpe_ret = build_cpe_sleeve(prices, trading_index)

    common_idx = pocket_ret.index.intersection(cpe_ret.index)
    combined_ret = (1 - CPE_ALLOCATION) * pocket_ret.reindex(common_idx).fillna(0.0) + CPE_ALLOCATION * cpe_ret.reindex(common_idx).fillna(0.0)

    print("Building passive equal-weight benchmark (same 12-instrument universe)...")
    log_px = np.log(prices[ALL_12_INSTRUMENTS])
    daily_ret_all = log_px.diff().reindex(common_idx)
    active = daily_ret_all.notna()
    n_active = active.sum(axis=1).replace(0, np.nan)
    passive_ret = (daily_ret_all.where(active, 0.0).sum(axis=1)) / n_active

    passive_equity = np.exp(passive_ret.fillna(0)).cumprod()
    drawdown = passive_equity / passive_equity.cummax() - 1.0
    crisis_mask = drawdown <= CRISIS_THRESHOLD
    episodes = cluster_episodes(crisis_mask, MIN_EPISODE_DAYS)
    print(f"\n{len(episodes)} real crisis episodes identified (passive 12-instrument-universe drawdown <= {CRISIS_THRESHOLD:.0%})")

    n_combined_won, n_baseline_won = 0, 0
    for start_d, end_d in episodes:
        window = (common_idx >= start_d) & (common_idx <= end_d)
        combo_cum = float((np.exp(combined_ret[window].fillna(0)).prod() - 1) * 100)
        base_cum = float((np.exp(baseline_ret.reindex(common_idx)[window].fillna(0)).prod() - 1) * 100)
        pass_cum = float((np.exp(passive_ret[window].fillna(0)).prod() - 1) * 100)
        combo_won = combo_cum > base_cum
        n_combined_won += combo_won
        n_baseline_won += (not combo_won)
        print(f"  {start_d.date()} to {end_d.date()} ({window.sum():>4}d): passive={pass_cum:+6.1f}%  "
              f"baseline TSMOM={base_cum:+6.1f}%  combined={combo_cum:+6.1f}%  "
              f"{'<- combined ahead' if combo_won else '<- baseline ahead'}")

    print(f"\nCombined (pocket-filtered TSMOM + CPE sleeve) ahead of unfiltered baseline in "
          f"{n_combined_won}/{len(episodes)} identified crisis episodes")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[2, 1])
    series = [
        ("Passive equal-weight (12 instr.)", passive_ret, "#9AA1AD"),
        ("Baseline TSMOM (12 instr., unfiltered)", baseline_ret.reindex(common_idx), "#B0492F"),
        ("Pocket-filtered TSMOM (5 instr.)", pocket_ret.reindex(common_idx), "#2E6DA4"),
        ("Combined (pocket-filtered + CPE sleeve)", combined_ret, "#2f8a4e"),
    ]
    for label, ret, color in series:
        eq = np.exp(ret.fillna(0)).cumprod() * 100_000
        ax1.plot(eq.index, eq.values, label=f"{label} (Sharpe {sharpe(ret):.2f})", lw=1.3, color=color)
    for start_d, end_d in episodes:
        ax1.axvspan(start_d, end_d, color="#B0492F", alpha=0.10)
    ax1.set_yscale("log")
    ax1.set_ylabel("Equity ($100k notional, log)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title(f"Does the combined pocket-filtered TSMOM + CPE sleeve actually help where it should?\n"
                  f"{len(episodes)} real crisis episodes shaded (passive 12-instrument drawdown <= {CRISIS_THRESHOLD:.0%}) -- "
                  f"combined ahead of unfiltered baseline in {n_combined_won}/{len(episodes)} of them")

    ax2.fill_between(drawdown.index, drawdown.values * 100, 0, color="#B0492F", alpha=0.3)
    ax2.set_ylabel("Passive drawdown (%)")
    ax2.axhline(CRISIS_THRESHOLD * 100, color="black", lw=0.7, linestyle="--")
    fig.tight_layout()
    fig.savefig("combined_strategy_crisis_view.png", dpi=140)
    plt.close(fig)
    print("\nSaved: combined_strategy_crisis_view.png")
