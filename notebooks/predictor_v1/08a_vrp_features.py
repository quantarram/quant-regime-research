"""
Wave-2 predictor panel: volatility risk premium (VRP) as the "fuel" variable.

VRP = implied vol (^VIX) - realized vol (from SPY's own trailing return
series, annualized to the same percentage-point scale as VIX) -- the
liquidity/risk-appetite proxy identified in the plan's data audit, since this
dataset has no true volume/positioning data. Market-wide (one series), plus
an interaction with each instrument's own extreme-dynamics feature
(gap_tau21_q4_z), matching the same "trigger x fuel" interaction pattern as
04c_macro_interaction_features.py.

Run: python 08a_vrp_features.py
Output: features_vrp.parquet
"""
import pandas as pd
import numpy as np
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
baseline = pd.read_parquet(os.path.join(OUT_DIR, "features_daily_panel.parquet"))
baseline["date"] = pd.to_datetime(baseline["date"])

spy = prices["SPY"].dropna()  # dropna BEFORE shift/rolling -- once crypto (24/7) joined the
spy_ret = np.log(spy / spy.shift(1))  # shared calendar (~2014-09-17), it started including
realized_vol_21d = spy_ret.rolling(21).std() * np.sqrt(252) * 100  # weekend rows where SPY is
vrp = (prices["^VIX"].reindex(spy.index) - realized_vol_21d).rename("vrp_market")  # NaN, which
# silently breaks positional .shift(1)/.rolling(21) on the raw column from that date forward if
# not dropna'd first (same convention 01_rolling_features.py already uses for this reason).

macro = vrp.reset_index().rename(columns={"index": "date"})
panel = baseline[["ticker", "date", "gap_tau21_q4_z"]].merge(macro, on="date", how="left")
panel["interact_gap21q4_vrp"] = panel["gap_tau21_q4_z"] * panel["vrp_market"]

out = panel[["ticker", "date", "vrp_market", "interact_gap21q4_vrp"]]
out_path = os.path.join(OUT_DIR, "features_vrp.parquet")
out.to_parquet(out_path)
print(f"Saved {len(out)} rows x {len(out.columns)} cols to {out_path}")
print(out.dropna().tail(3))
