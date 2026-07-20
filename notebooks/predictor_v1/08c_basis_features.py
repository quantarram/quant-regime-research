"""
Wave-2 predictor panel: futures-spot basis/carry, a physically-constrained
convergence process by construction (storage theory / normal backwardation).

oil_basis_z  = z-scored deviation of log(CL=F/USO) from its own 200d mean
gold_basis_z = z-scored deviation of log(GC=F/GLD) from its own 200d mean
plus 21d velocity of each. Market-wide.

Run: python 08c_basis_features.py
Output: features_basis.parquet
"""
import pandas as pd
import numpy as np
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))


def basis_z(fut_col, etf_col):
    # CL=F/USO and GC=F/GLD don't share an identical trading calendar (scattered
    # gap days), which starves a strict rolling(200) window of ever finding 200
    # consecutive non-null values -- ffill both legs first, same convention
    # 04b_term_structure_features.py uses for its rate/vol series.
    fut = prices[fut_col].ffill()
    etf = prices[etf_col].ffill()
    ratio = np.log(fut / etf)
    return (ratio - ratio.rolling(200, min_periods=100).mean()) / ratio.rolling(200, min_periods=100).std()


oil_basis_z = basis_z("CL=F", "USO")
gold_basis_z = basis_z("GC=F", "GLD")

feat = pd.DataFrame({
    "oil_basis_z": oil_basis_z,
    "gold_basis_z": gold_basis_z,
})
feat["oil_basis_z_velocity_21d"] = feat["oil_basis_z"].diff(21)
feat["gold_basis_z_velocity_21d"] = feat["gold_basis_z"].diff(21)
feat = feat.reset_index().rename(columns={"index": "date"})

out_path = os.path.join(OUT_DIR, "features_basis.parquet")
feat.to_parquet(out_path)
print(f"Saved {len(feat)} rows x {len(feat.columns)} cols to {out_path}")
print(feat.dropna().tail(3))
