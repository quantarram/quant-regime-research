"""
tsmom_block_count_sensitivity.py
========================================
Robustness check on Paper 16's central result: pocket-filtered TSMOM (SPY,
IWM, AAPL, MSFT, GLD) vs. unfiltered TSMOM (12 instruments), replicated
across several different mechanical calendar-block counts, not just the
N_BLOCKS=5 used in the paper. Same construction, same data, same rule for
building block edges (pd.date_range(start, end, periods=N+1), a mechanical
equal-length split, not curated) -- only N_BLOCKS varies.

No p-values, no significance test -- real point estimates and real win
counts at each block count, reported directly.

Run: python tsmom_block_count_sensitivity.py
Output: tsmom_block_count_sensitivity_results.json, printed table
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from combined_pocket_tsmom_cpe_strategy import (
    ALL_12_INSTRUMENTS, POCKET_INSTRUMENTS, LOOKBACK, VOL_WINDOW,
    build_tsmom, month_end_dates,
)

BLOCK_COUNTS = [3, 4, 5, 6, 7, 8, 10, 13, 16, 20]

if __name__ == "__main__":
    prices = pd.read_parquet("multiasset_prices.parquet")
    n_bad = int((prices[ALL_12_INSTRUMENTS] <= 0).sum().sum())
    if n_bad:
        prices = prices.copy()
        prices[ALL_12_INSTRUMENTS] = prices[ALL_12_INSTRUMENTS].mask(prices[ALL_12_INSTRUMENTS] <= 0)

    first_valid = max(prices[t].dropna().index.min() for t in ALL_12_INSTRUMENTS)
    start = first_valid + pd.Timedelta(days=int(1.5 * (LOOKBACK + VOL_WINDOW)))
    rebal_dates = month_end_dates(prices.index[prices.index >= start])
    trading_index = prices.index[prices.index >= rebal_dates[0]]

    print("Building baseline TSMOM (12 instruments, unfiltered)...")
    baseline_ret = build_tsmom(prices, ALL_12_INSTRUMENTS, trading_index, rebal_dates)
    print("Building pocket-filtered TSMOM (SPY, IWM, AAPL, MSFT, GLD)...")
    pocket_ret = build_tsmom(prices, POCKET_INSTRUMENTS, trading_index, rebal_dates)

    common_idx = baseline_ret.index.intersection(pocket_ret.index)
    baseline_ret, pocket_ret = baseline_ret.reindex(common_idx), pocket_ret.reindex(common_idx)

    print(f"\n{'='*100}\nBlock-count sensitivity, {common_idx.min().date()} to {common_idx.max().date()}\n{'='*100}")
    all_results = {}
    for n_blocks in BLOCK_COUNTS:
        block_edges = pd.date_range(common_idx.min(), common_idx.max(), periods=n_blocks + 1)
        n_pocket_wins = 0
        block_detail = []
        for i in range(n_blocks):
            b_start, b_end = block_edges[i], block_edges[i + 1]
            mask = (common_idx >= b_start) & (common_idx < b_end if i < n_blocks - 1 else common_idx <= b_end)
            idx = common_idx[mask]
            base_ann = baseline_ret.reindex(idx).mean() * 252 * 100
            pocket_ann = pocket_ret.reindex(idx).mean() * 252 * 100
            wins = bool(pocket_ann > base_ann)
            n_pocket_wins += wins
            block_detail.append({"start": str(b_start.date()), "end": str(b_end.date()),
                                  "baseline_pct": float(base_ann), "pocket_pct": float(pocket_ann), "pocket_wins": wins})
        frac = n_pocket_wins / n_blocks
        print(f"  N_BLOCKS={n_blocks:>3}  (~{(common_idx.max()-common_idx.min()).days/365.25/n_blocks:4.1f}yr/block)  "
              f"pocket-filtered wins {n_pocket_wins:>2}/{n_blocks:<2}  ({frac:5.1%})")
        all_results[str(n_blocks)] = {"n_blocks": n_blocks, "n_pocket_wins": n_pocket_wins,
                                        "win_fraction": frac, "blocks": block_detail}

    with open("tsmom_block_count_sensitivity_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print("\nSaved: tsmom_block_count_sensitivity_results.json")
