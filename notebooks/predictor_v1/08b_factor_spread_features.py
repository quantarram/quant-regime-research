"""
Wave-2 predictor panel: priced-risk-factor spreads (Fama-French/Carhart style).

value_growth_spread_mom21 = 21d change in log(VTV/VUG)
momentum_factor_rel21     = MTUM's 21d log return minus SPY's
quality_factor_rel21      = QUAL's 21d log return minus SPY's
size_factor_rel21         = SIZE's 21d log return minus SPY's
Market-wide (one series each), no per-instrument variant.

Run: python 08b_factor_spread_features.py
Output: features_factor_spreads.parquet
"""
import pandas as pd
import numpy as np
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
# Restrict to weekday rows before diff/shift -- see 08a_vrp_features.py's comment: the shared
# calendar started including weekend rows (NaN for these equity/ETF tickers) once crypto joined
# it (~2014-09-17), which silently breaks positional .diff()/.shift() on the raw columns from
# that date forward if not handled first.
FACTOR_COLS = ["VTV", "VUG", "SPY", "MTUM", "QUAL", "SIZE"]
prices = prices[prices.index.dayofweek < 5][FACTOR_COLS].ffill()

vg_ratio = np.log(prices["VTV"] / prices["VUG"])
value_growth_spread_mom21 = vg_ratio.diff(21)

spy_ret21 = np.log(prices["SPY"] / prices["SPY"].shift(21))
momentum_factor_rel21 = np.log(prices["MTUM"] / prices["MTUM"].shift(21)) - spy_ret21
quality_factor_rel21 = np.log(prices["QUAL"] / prices["QUAL"].shift(21)) - spy_ret21
size_factor_rel21 = np.log(prices["SIZE"] / prices["SIZE"].shift(21)) - spy_ret21

feat = pd.DataFrame({
    "value_growth_spread_mom21": value_growth_spread_mom21,
    "momentum_factor_rel21": momentum_factor_rel21,
    "quality_factor_rel21": quality_factor_rel21,
    "size_factor_rel21": size_factor_rel21,
}).reset_index().rename(columns={"index": "date"})

out_path = os.path.join(OUT_DIR, "features_factor_spreads.parquet")
feat.to_parquet(out_path)
print(f"Saved {len(feat)} rows x {len(feat.columns)} cols to {out_path}")
print(feat.dropna().tail(3))
