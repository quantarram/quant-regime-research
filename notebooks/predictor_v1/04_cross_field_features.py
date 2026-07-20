"""
Wave-1 predictor panel: cross-field multifractal coupling.

The direct translation of atmospheric convergence/divergence into this
program's own multifractal-cascade language: instead of a structure function
of one field with itself (what 01_rolling_features.py already computes --
self-referential, "one variable measured on itself"), a genuine bivariate
cross-moment between two economically coupled fields, in the spirit of
Multifractal Cross-Correlation Analysis (MF-DCCA: Zhou 2008; Podobnik &
Stanley 2008). Simplified from full MF-DCCA (which fits a multiscale
detrended-fluctuation exponent) to a single-scale rolling normalized cross
q-th moment, for the same reason the multifractal predictability paper itself
abandoned the parametric route for a direct empirical construction
(cpe_paper_multifractal_predictability_draft.md, Section 4) -- tractable and
still a genuine two-field coupling term, not a per-instrument statistic.

Pairs chosen for real economic coupling (not arbitrary):
  SPY <-> TLT   risk-on/risk-off flow
  GLD <-> TIP   opportunity-cost / real-yield coupling
  CL=F <-> TIP  inflation-expectations coupling
  UUP <-> GLD   dollar as a "steering flow" for gold
  HYG <-> SPY   credit-equity risk-appetite coupling

For each pair (X,Y), lag tau, and moment order q, computes the normalized
cross moment of absolute price increments:
  M_xy(tau,q) = mean[ |Δx(t)|^(q/2) * |Δy(t+tau)|^(q/2) ]
                / sqrt( mean[|Δx(t)|^q] * mean[|Δy(t+tau)|^q] )
a genuine X-leads-Y coupling coefficient (tau=0 is contemporaneous), computed
on a trailing window only -- point-in-time, no lookahead, matching
01_rolling_features.py's convention exactly (same LOOKBACK, same master
date grid, same STRIDE=1) so this panel merges cleanly onto the baseline.

Run: python 04_cross_field_features.py
Requires: ../multiasset_prices.parquet
Output: features_cross_field.parquet
"""
import pandas as pd
import numpy as np
import os
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PAIRS = [("SPY", "TLT"), ("GLD", "TIP"), ("CL=F", "TIP"), ("UUP", "GLD"), ("HYG", "SPY")]
LAGS = [0, 5, 21]
QS = [2.0, 4.0]
LOOKBACK = 512
STRIDE = 1
MIN_WINDOW = 60  # minimum overlapping obs required inside a window to trust the moment

print("=" * 60)
print("  WAVE-1 PREDICTOR PANEL: CROSS-FIELD MULTIFRACTAL COUPLING")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
master_dates = prices.index[::STRIDE]


def cross_moment(fx_win, fy_win, tau, q):
    n = min(len(fx_win), len(fy_win))
    if tau == 0:
        a, b = fx_win[:n], fy_win[:n]
    else:
        if n <= tau:
            return np.nan
        a, b = fx_win[:n - tau], fy_win[tau:n]
    if len(a) < MIN_WINDOW:
        return np.nan
    a_h, b_h = a ** (q / 2), b ** (q / 2)
    cross = np.mean(a_h * b_h)
    norm = np.sqrt(np.mean(a ** q) * np.mean(b ** q))
    if norm <= 0 or not np.isfinite(norm):
        return np.nan
    return float(cross / norm)


def compute_pair_features(x_ticker, y_ticker):
    sx = prices[x_ticker].dropna()
    sy = prices[y_ticker].dropna()
    common_idx = sx.index.intersection(sy.index)
    sx, sy = sx.reindex(common_idx), sy.reindex(common_idx)
    fx = np.concatenate([[np.nan], np.abs(np.diff(sx.values))])
    fy = np.concatenate([[np.nan], np.abs(np.diff(sy.values))])
    idx = sx.index
    n = len(idx)
    if n < LOOKBACK + max(LAGS) + MIN_WINDOW:
        print(f"  {x_ticker}-{y_ticker}: insufficient overlapping history ({n} obs), skipping")
        return pd.DataFrame()

    pair_dates = [d for d in master_dates if d in set(idx)]
    idx_pos = idx.searchsorted(pair_dates, side="right") - 1
    rows = []
    for d, pos in zip(pair_dates, idx_pos):
        if pos < LOOKBACK - 1 or pos >= n:
            continue
        fx_win = fx[max(0, pos - LOOKBACK + 1): pos + 1]
        fy_win = fy[max(0, pos - LOOKBACK + 1): pos + 1]
        fx_win = fx_win[~np.isnan(fx_win)]
        fy_win = fy_win[~np.isnan(fy_win)]
        feat = {"date": d}
        for tau in LAGS:
            for q in QS:
                feat[f"xmom_{x_ticker}_{y_ticker}_tau{tau}_q{int(q)}"] = cross_moment(fx_win, fy_win, tau, q)
        rows.append(feat)
    return pd.DataFrame(rows)


t0 = time.time()
frames = []
for x, y in PAIRS:
    print(f"Computing cross-moment features for {x} <-> {y}...")
    df = compute_pair_features(x, y)
    if not df.empty:
        frames.append(df)
    print(f"  {len(df)} rows, {time.time()-t0:.1f}s elapsed")

panel = frames[0]
for df in frames[1:]:
    panel = panel.merge(df, on="date", how="outer")
panel = panel.sort_values("date").reset_index(drop=True)

out_path = os.path.join(OUT_DIR, "features_cross_field.parquet")
panel.to_parquet(out_path)
print(f"\nSaved {len(panel)} rows x {len(panel.columns)} cols to {out_path}")
print(f"Total time: {time.time()-t0:.1f}s")
