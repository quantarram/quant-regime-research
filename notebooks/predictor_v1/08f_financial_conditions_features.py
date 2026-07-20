"""
Wave-2 predictor panel: composite financial-conditions index.

A combined credit-spread + vol + dollar + curve-slope index, in the spirit
of the Chicago Fed NFCI / St. Louis Fed STLFSI (Brave & Butters 2011) -- a
synthesized "large-scale steering state" built from several coupled
variables at once, unlike items 1-4's single-relationship features.
Higher = tighter/more stressed financial conditions (each component
oriented and z-scored the same direction before averaging).

Components (each z-scored vs its own 200d trailing mean/std):
  credit_component = -(HYG/LQD ratio)   [ratio UP = credit conditions BETTER,
                                          so negate for "tighter=positive" polarity]
  vol_component     = ^VIX level
  dollar_component  = UUP level
  curve_component   = -(10y-3m slope)   [inverted/flat curve = tighter conditions]

Run: python 08f_financial_conditions_features.py
Output: features_financial_conditions.parquet
"""
import pandas as pd
import numpy as np
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def zscore(s, window=200, min_periods=100):
    return (s - s.rolling(window, min_periods=min_periods).mean()) / s.rolling(window, min_periods=min_periods).std()


prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))

credit_ratio = (prices["HYG"] / prices["LQD"]).ffill()
credit_component = -zscore(credit_ratio)
vol_component = zscore(prices["^VIX"].ffill())
dollar_component = zscore(prices["UUP"].ffill())
curve_slope = (prices["^TNX"] - prices["^IRX"]).ffill()
curve_component = -zscore(curve_slope)

fci = pd.concat([credit_component, vol_component, dollar_component, curve_component], axis=1)
fci.columns = ["credit_component", "vol_component", "dollar_component", "curve_component"]
fci["financial_conditions_index"] = fci.mean(axis=1)
fci["fci_velocity_21d"] = fci["financial_conditions_index"].diff(21)

feat = fci[["financial_conditions_index", "fci_velocity_21d"]].reset_index().rename(columns={"index": "date"})
out_path = os.path.join(OUT_DIR, "features_financial_conditions.parquet")
feat.to_parquet(out_path)
print(f"Saved {len(feat)} rows x {len(feat.columns)} cols to {out_path}")
print(feat.dropna().tail(5))
