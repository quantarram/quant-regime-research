"""
Paper 7 — Permutation Test for Multiple Testing Correction
===========================================================
Tests whether the 172 surviving signals could arise by chance
under random shuffling of temperature exceedance flags.

Method: Circular block permutation
  - Shuffle the temperature exceedance series by random circular shift
  - This preserves autocorrelation structure and seasonal patterns
  - Re-run the full CPE analysis on each shuffled dataset
  - Count how many signals survive at the same quality filters
  - Compare empirical signal count (172) to permutation distribution

If median permutation signals << 172 → results are genuine
If median permutation signals ≈ 172 → multiple testing problem

N_PERMS = 1000 recommended (takes ~30-60 min on local machine)
N_PERMS = 100  for quick check (~3-6 min)

Run on your LOCAL machine.
Required:
  - multiasset_returns.parquet
  - data/paper7_agri_exceedances_aligned.parquet
"""

import pandas as pd
import numpy as np
import warnings
import os
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# ── CONFIGURATION ──────────────────────────────────────────────────────
TRAIN_END  = '2024-12-31'
EVAL_START = '2025-01-01'
MIN_N_COND = 30
MIN_CPE    = 0.58
MIN_LIFT   = 1.15
QUANTILES  = [0.50, 0.55, 0.60, 0.65, 0.70]
HORIZONS   = [21, 42, 63, 126]
N_PERMS    = 1000  # set to 100 for quick test

# Same channel pairs as paper7_cpe_analysis.py
CHANNEL_PAIRS = [
    ('EU_Wheat_Belt',          'WEAT',  HORIZONS),
    ('EU_Wheat_Belt',          'ZW=F',  HORIZONS),
    ('EU_Wheat_Belt',          'GC=F',  HORIZONS),
    ('EU_Wheat_Belt',          'NG=F',  HORIZONS),
    ('Ukraine_Russia_Wheat',   'WEAT',  HORIZONS),
    ('Ukraine_Russia_Wheat',   'ZW=F',  HORIZONS),
    ('Ukraine_Russia_Wheat',   'GC=F',  HORIZONS),
    ('US_Great_Plains_Wheat',  'WEAT',  HORIZONS),
    ('US_Great_Plains_Wheat',  'ZW=F',  HORIZONS),
    ('US_Great_Plains_Wheat',  'CORN',  HORIZONS),
    ('US_Great_Plains_Wheat',  'ZC=F',  HORIZONS),
    ('US_Corn_Belt',           'CORN',  HORIZONS),
    ('US_Corn_Belt',           'ZC=F',  HORIZONS),
    ('US_Corn_Belt',           'SOYB',  HORIZONS),
    ('US_Corn_Belt',           'ZS=F',  HORIZONS),
    ('US_Corn_Belt',           'WEAT',  HORIZONS),
    ('US_Corn_Belt',           'GC=F',  HORIZONS),
    ('Brazil_Corn',            'CORN',  HORIZONS),
    ('Brazil_Corn',            'ZC=F',  HORIZONS),
    ('Brazil_Corn',            'SOYB',  HORIZONS),
    ('Thailand_Sugar',         'CANE',  HORIZONS),
    ('Thailand_Sugar',         'GC=F',  HORIZONS),
    ('Thailand_Sugar',         'DBB',   HORIZONS),
    ('India_Sugar',            'CANE',  HORIZONS),
    ('India_Sugar',            'GC=F',  HORIZONS),
    ('EU_Urban_Energy',        'NG=F',  HORIZONS),
    ('EU_Urban_Energy',        'UNG',   HORIZONS),
    ('EU_Urban_Energy',        'XLU',   HORIZONS),
    ('EU_Urban_Energy',        'XLE',   HORIZONS),
    ('EU_Urban_Energy',        'ICLN',  HORIZONS),
    ('EU_Urban_Energy',        'DBB',   HORIZONS),
    ('EU_Urban_Energy',        'GC=F',  HORIZONS),
    ('US_Urban_Energy',        'NG=F',  HORIZONS),
    ('US_Urban_Energy',        'UNG',   HORIZONS),
    ('US_Urban_Energy',        'XLU',   HORIZONS),
    ('US_Urban_Energy',        'CORN',  HORIZONS),
    ('US_Urban_Energy',        'GC=F',  HORIZONS),
]

PREDICTOR_TYPES = [
    'tmax_q80','tmax_q90','tmax_q95',
    'seasonal_q80','seasonal_q90',
    'heatstress_30C','heatstress_32C','heatstress_35C','heatstress_38C',
    'GDD30_q80','GDD30_q90',
]

# ── LOAD DATA ─────────────────────────────────────────────────────────
def load_data():
    print("Loading data...")
    returns = pd.read_parquet('multiasset_returns.parquet')
    temp    = pd.read_parquet('data/paper7_agri_exceedances_aligned.parquet')
    print(f"  Returns: {returns.shape}")
    print(f"  Temp:    {temp.shape}")

    all_h = sorted(set(h for _,_,hs in CHANNEL_PAIRS for h in hs))
    print(f"  Pre-computing forward returns for horizons {all_h}...")
    cum_fwd = {}
    tickers = list(set([p[1] for p in CHANNEL_PAIRS]))
    for ticker in tickers:
        if ticker not in returns.columns:
            continue
        daily_clean = returns[ticker].dropna()
        cum_fwd[ticker] = {}
        for h in all_h:
            td   = max(1, int(round(h * 252 / 365)))
            roll = daily_clean.rolling(window=td, min_periods=td).sum().shift(-td)
            cum_fwd[ticker][h] = roll.reindex(returns[ticker].index)

    return returns, temp, cum_fwd

# ── SINGLE CPE CHECK ──────────────────────────────────────────────────
def cpe_passes(temp_series, fwd_series, q_target):
    """Return True if this configuration passes quality filters."""
    both  = pd.DataFrame({'temp': temp_series, 'fwd': fwd_series}).dropna()
    train = both[both.index <= TRAIN_END]
    if len(train) < MIN_N_COND:
        return False
    thr  = train['fwd'].quantile(q_target)
    up   = (train['fwd'] > thr).mean()
    if up <= 0:
        return False
    cond = train[train['temp'] == 1]
    if len(cond) < MIN_N_COND:
        return False
    cpe  = (cond['fwd'] > thr).mean()
    lift = cpe / up
    return cpe >= MIN_CPE and lift >= MIN_LIFT

# ── COUNT SIGNALS for one temp dataset ────────────────────────────────
def count_signals(temp_df, cum_fwd):
    """Count surviving signals across all channel pairs and predictor types."""
    n = 0
    for zone_prefix, ticker, horizons in CHANNEL_PAIRS:
        if ticker not in cum_fwd:
            continue
        zone_cols = [c for c in temp_df.columns
                     if c.startswith(zone_prefix + '_')
                     and c[len(zone_prefix)+1:] in PREDICTOR_TYPES]
        for pred_col in zone_cols:
            for h in horizons:
                if h not in cum_fwd[ticker]:
                    continue
                for q in QUANTILES:
                    if cpe_passes(temp_df[pred_col], cum_fwd[ticker][h], q):
                        n += 1
    return n

# ── CIRCULAR BLOCK SHUFFLE ────────────────────────────────────────────
def circular_shift_temp(temp_df, rng):
    """
    Circular block permutation: randomly shift the entire temperature
    dataset by a random number of trading days.

    This preserves:
    - Autocorrelation structure within the series
    - Seasonal patterns (approximately — shift is random)
    - Cross-predictor correlations (all columns shift together)

    This destroys:
    - The alignment between temperature events and forward financial returns
    """
    n     = len(temp_df)
    shift = rng.integers(low=int(n * 0.1), high=int(n * 0.9))
    # Circular shift: move rows by `shift`, wrapping around
    shifted_values = np.roll(temp_df.values, shift, axis=0)
    return pd.DataFrame(shifted_values, index=temp_df.index,
                        columns=temp_df.columns)

# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("PAPER 7 — PERMUTATION TEST FOR MULTIPLE TESTING CORRECTION")
    print("=" * 65)
    print(f"N permutations : {N_PERMS}")
    print(f"Min CPE        : {MIN_CPE}")
    print(f"Min lift       : {MIN_LIFT}×")
    print(f"Min n_cond     : {MIN_N_COND}")
    print()

    returns, temp, cum_fwd = load_data()

    # Step 1: Count empirical signals (observed result)
    print("\nStep 1: Counting empirical signals (observed data)...")
    t0 = time.time()
    n_empirical = count_signals(temp, cum_fwd)
    print(f"  Empirical signal count: {n_empirical}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Step 2: Permutation distribution
    print(f"\nStep 2: Running {N_PERMS} permutations...")
    print("  (Circular shift permutation — preserves autocorrelation)")
    rng = np.random.default_rng(seed=42)
    perm_counts = []

    for i in range(N_PERMS):
        temp_shuffled = circular_shift_temp(temp, rng)
        n_perm        = count_signals(temp_shuffled, cum_fwd)
        perm_counts.append(n_perm)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i+1) * (N_PERMS - i - 1)
            print(f"  [{i+1:4d}/{N_PERMS}] median={np.median(perm_counts):.0f} "
                  f"max={max(perm_counts)} "
                  f"ETA={eta/60:.1f}min")

    perm_counts = np.array(perm_counts)

    # Step 3: Results
    p_value   = np.mean(perm_counts >= n_empirical)
    median_p  = np.median(perm_counts)
    p95_perm  = np.percentile(perm_counts, 95)
    p99_perm  = np.percentile(perm_counts, 99)
    ratio     = n_empirical / median_p if median_p > 0 else float('inf')

    print("\n" + "=" * 65)
    print("PERMUTATION TEST RESULTS")
    print("=" * 65)
    print(f"Empirical signal count : {n_empirical}")
    print(f"Permutation median     : {median_p:.1f}")
    print(f"Permutation 95th pct   : {p95_perm:.1f}")
    print(f"Permutation 99th pct   : {p99_perm:.1f}")
    print(f"Empirical / median     : {ratio:.2f}×")
    print(f"p-value                : {p_value:.4f}  "
          f"(fraction of perms ≥ empirical)")
    print()

    if p_value < 0.001:
        print("✓ HIGHLY SIGNIFICANT (p<0.001)")
        print("  The empirical signal count far exceeds the permutation")
        print("  distribution. Results are not a multiple testing artefact.")
    elif p_value < 0.01:
        print("✓ SIGNIFICANT (p<0.01)")
        print("  Results survive multiple testing correction.")
    elif p_value < 0.05:
        print("~ MARGINAL (p<0.05)")
        print("  Results survive at conventional threshold but borderline.")
    else:
        print("✗ NOT SIGNIFICANT (p>{:.3f})".format(p_value))
        print("  Results do not survive permutation test.")
        print("  Multiple testing is a concern.")

    # Save results
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame({
        'permutation': range(N_PERMS),
        'n_signals':   perm_counts,
    })
    results_df.to_csv('results/paper7_permutation_test.csv', index=False)

    summary = {
        'n_empirical':   n_empirical,
        'n_perms':       N_PERMS,
        'perm_median':   float(median_p),
        'perm_p95':      float(p95_perm),
        'perm_p99':      float(p99_perm),
        'ratio':         float(ratio),
        'p_value':       float(p_value),
        'timestamp':     datetime.now().isoformat(),
    }
    pd.Series(summary).to_csv('results/paper7_permutation_summary.csv')

    print(f"\nSaved permutation distribution → results/paper7_permutation_test.csv")
    print(f"Saved summary → results/paper7_permutation_summary.csv")
    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} minutes")

    # Print distribution summary
    print("\nPermutation distribution:")
    for pct in [5,10,25,50,75,90,95,99]:
        print(f"  {pct:3d}th percentile: {np.percentile(perm_counts,pct):.0f} signals")
