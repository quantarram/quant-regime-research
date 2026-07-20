"""
Wave-2 predictor panel: systemic coupling / realized-correlation coherence.

Rolling 60d average pairwise realized correlation across a core cross-asset
subset (SPY, QQQ, IWM, XLK, XLF, XLE, GLD, TLT, BTC-USD -- chosen for deep
history + spanning multiple asset classes) -- the well-documented "everything
correlates in a crisis" regularity (dynamic conditional correlation, Engle
2002), structurally analogous to large-scale atmospheric organization
preceding a storm. Market-wide (one series).

Run: python 08d_systemic_correlation_features.py
Output: features_systemic_correlation.parquet
"""
import pandas as pd
import numpy as np
import itertools
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

CORE = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "GLD", "TLT", "BTC-USD"]
WINDOW = 60

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
# Restrict to weekday rows (business-day calendar) before computing returns -- once crypto (24/7)
# joined the shared index (~2014-09-17) it started including weekend rows where the other 8 core
# tickers are NaN, which silently breaks positional .shift(1)/.rolling() on the raw columns from
# that date forward if not handled first (same issue found and fixed in 08a_vrp_features.py).
prices_wd = prices[prices.index.dayofweek < 5][CORE].ffill()
rets = pd.DataFrame({t: np.log(prices_wd[t] / prices_wd[t].shift(1)) for t in CORE})

pair_corrs = []
for a, b in itertools.combinations(CORE, 2):
    pair_corrs.append(rets[a].rolling(WINDOW, min_periods=30).corr(rets[b]))
avg_corr = pd.concat(pair_corrs, axis=1).mean(axis=1).rename("systemic_avg_corr_60d")
avg_corr_velocity = avg_corr.diff(21).rename("systemic_avg_corr_velocity_21d")

feat = pd.concat([avg_corr, avg_corr_velocity], axis=1).reset_index().rename(columns={"index": "date"})
out_path = os.path.join(OUT_DIR, "features_systemic_correlation.parquet")
feat.to_parquet(out_path)
print(f"Saved {len(feat)} rows x {len(feat.columns)} cols to {out_path}")
print(feat.dropna().tail(3))
print(f"\nRange check: min={feat['systemic_avg_corr_60d'].min():.3f}, max={feat['systemic_avg_corr_60d'].max():.3f} (should be within [-1,1])")
