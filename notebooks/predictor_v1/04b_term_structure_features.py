"""
Wave-1 predictor panel: term-structure curvature as literal "vertical levels."

Different bond tenors (or vol-surface instruments across asset classes) are
genuinely stacked "levels" the way atmospheric pressure levels are -- this
computes curvature/convergence-divergence directly on those levels, not a
metaphorical analog. Unlike 01_rolling_features.py and 04_cross_field_features.py,
these are direct algebraic transforms of already-available level/yield data
(no rolling multifractal fit needed), so no lookahead risk beyond a plain
backward .diff().

Rate curve "levels" (13wk / 5y / 10y / 30y):
  curve_slope_10y3m       = TNX - IRX          (classic 2s10s-style slope)
  curve_slope_30y10y      = TYX - TNX
  curve_curvature_5ybelly = 2*FVX - IRX - TNX   (5y belly butterfly)
  curve_curvature_10ybelly= 2*TNX - FVX - TYX   (10y belly butterfly)
  *_velocity_21d          = 21-trading-day change in the above (roll-down velocity)

Cross-asset vol "levels" (equity vol vs Nasdaq/gold/oil vol):
  vix_vxn_spread, vix_gvz_spread, vix_ovx_spread, and their 21d velocities

Run: python 04b_term_structure_features.py
Requires: ../multiasset_prices.parquet
Output: features_term_structure.parquet
"""
import pandas as pd
import numpy as np
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("  WAVE-1 PREDICTOR PANEL: TERM-STRUCTURE CURVATURE")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))

RATE_COLS = ["^IRX", "^FVX", "^TNX", "^TYX"]
VOL_COLS = ["^VIX", "^VXN", "^GVZ", "^OVX"]
rates = prices[RATE_COLS].ffill()
vols = prices[VOL_COLS].ffill()

feat = pd.DataFrame(index=prices.index)
feat["curve_slope_10y3m"] = rates["^TNX"] - rates["^IRX"]
feat["curve_slope_30y10y"] = rates["^TYX"] - rates["^TNX"]
feat["curve_curvature_5ybelly"] = 2 * rates["^FVX"] - rates["^IRX"] - rates["^TNX"]
feat["curve_curvature_10ybelly"] = 2 * rates["^TNX"] - rates["^FVX"] - rates["^TYX"]
for c in ["curve_slope_10y3m", "curve_slope_30y10y", "curve_curvature_5ybelly", "curve_curvature_10ybelly"]:
    feat[f"{c}_velocity_21d"] = feat[c].diff(21)

feat["vix_vxn_spread"] = vols["^VIX"] - vols["^VXN"]
feat["vix_gvz_spread"] = vols["^VIX"] - vols["^GVZ"]
feat["vix_ovx_spread"] = vols["^VIX"] - vols["^OVX"]
for c in ["vix_vxn_spread", "vix_gvz_spread", "vix_ovx_spread"]:
    feat[f"{c}_velocity_21d"] = feat[c].diff(21)

feat = feat.reset_index().rename(columns={"index": "date"})
out_path = os.path.join(OUT_DIR, "features_term_structure.parquet")
feat.to_parquet(out_path)
print(f"Saved {len(feat)} rows x {len(feat.columns)} cols to {out_path}")
print(feat.tail(3).T)
