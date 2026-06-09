"""
Joint CPE Engine — Greedy Predictor Selection (Clean v2)
=========================================================
Fixes from v1:
  - Deduplicate: same ticker cannot appear twice in predictor set
  - Leveraged/inverse ETFs excluded from X as well as Y
  - Greedy stops at MAX_PREDICTORS (default 6)

For each (Y, tau_future, q_Y, direction):
  1. Load pairwise signals that passed filters for this Y
  2. Greedy: start with best single predictor (highest pairwise CPE)
  3. At each step, try adding each remaining predictor (unique tickers only):
     - Compute joint conditioning event (intersection of all predictor conditions)
     - Keep if n_joint >= MIN_N and joint CPE >= CPE_THRESH and lift >= MIN_LIFT
     - Select the addition that maximises joint CPE
  4. Stop when no predictor can be added OR n_predictors == MAX_PREDICTORS
  5. Save all intermediate joint sets (size 2, 3, ..., MAX_PREDICTORS) that pass filters

Output: joint_cpe_results.parquet
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings, os, time
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MIN_N          = 100
CPE_THRESH     = 0.80
MIN_LIFT       = 1.5
MAX_PREDICTORS = 10

RATE_INDEX_TICKERS = {
    "^VIX","^VXN","^OVX","^GVZ","^EVZ","^VVIX","^SKEW",
    "^TNX","^TYX","^FVX","^IRX"
}

# Excluded from both Y and X
EXCLUDE_TICKERS = {
    "SSO","SDS","TQQQ","TMF","TBT","TBF",
    "UVXY","SVXY","VIXY","VIXM","VXX",
    "THBUSD=X","CNYUSD=X","KRWUSD=X",
}

print(f"\n{'='*65}")
print(f"  JOINT CPE ENGINE (GREEDY v2)  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  MAX_PREDICTORS={MAX_PREDICTORS}  MIN_N={MIN_N}  CPE_THRESH={CPE_THRESH}  MIN_LIFT={MIN_LIFT}")
print(f"{'='*65}")

# ── LOAD PRICES ───────────────────────────────────────────────────────────────
print("\n  Loading price data...")
prices = pd.read_parquet("multiasset_prices.parquet")
all_tickers   = list(prices.columns)
price_tickers = [t for t in all_tickers if t not in RATE_INDEX_TICKERS]
rate_tickers  = [t for t in all_tickers if t in RATE_INDEX_TICKERS]

# ── PRE-COMPUTE INCREMENTS ────────────────────────────────────────────────────
TAU_LIST = [1, 5, 10, 21, 63, 126, 252, 300]
Q_GRID   = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]

print("  Pre-computing increments...")
increments = {}
for tau in TAU_LIST:
    inc = pd.DataFrame(index=prices.index)
    for t in price_tickers:
        s = prices[t]
        inc[t] = np.log(s / s.shift(tau))
    for t in rate_tickers:
        s = prices[t]
        inc[t] = s - s.shift(tau)
    increments[tau] = inc

print("  Pre-computing forward increments...")
predicted_tickers = [t for t in price_tickers if t not in EXCLUDE_TICKERS]
future_inc = {}
for tau_f in TAU_LIST:
    future_inc[tau_f] = increments[tau_f][predicted_tickers].shift(-tau_f)

print("  Pre-computing quantile thresholds...")
full_q_grid = sorted(set(Q_GRID + [round(1 - q, 10) for q in Q_GRID]))
thresholds = {}
for tau in TAU_LIST:
    for q in full_q_grid:
        thresholds[(tau, q)] = increments[tau].quantile(q, numeric_only=True)

# ── LOAD PAIRWISE CPE RESULTS ─────────────────────────────────────────────────
print("\n  Loading pairwise CPE results...")
pairwise = pd.read_parquet("cpe_results.parquet")

# Remove excluded tickers from predictor side
pairwise = pairwise[~pairwise["X"].isin(EXCLUDE_TICKERS)].copy()
print(f"  Pairwise rows after X exclusion: {len(pairwise):,}")
print(f"  Directions: {pairwise['direction'].value_counts().to_dict()}")

# ── HELPER: GET CONDITION MASK ────────────────────────────────────────────────
def get_condition_mask(x, tau_p, q_x, direction, common_idx):
    if x not in increments[tau_p].columns:
        return None
    px = increments[tau_p][x].loc[common_idx].values
    valid = ~np.isnan(px)
    thresh_up = thresholds[(tau_p, q_x)].get(x, np.nan)
    thresh_dn = thresholds[(tau_p, round(1 - q_x, 10))].get(x, np.nan)
    if direction == "bullish":
        if np.isnan(thresh_up): return None
        return valid & (px > thresh_up)
    else:
        if np.isnan(thresh_dn): return None
        return valid & (px < thresh_dn)

# ── GREEDY JOINT CPE ──────────────────────────────────────────────────────────
results = []
t0 = time.time()

groups = pairwise.groupby(["Y", "tau_future", "q_Y", "direction"])
print(f"\n  Total groups: {len(groups)}")
print(f"\n  Running greedy (max {MAX_PREDICTORS} predictors per set)...\n")

n_groups = 0
n_joint  = 0

for (y, tau_f, q_y, direction), group in groups:
    n_groups += 1

    if n_groups % 200 == 0:
        elapsed = time.time() - t0
        print(f"  [{n_groups:>5}/{len(groups)}]  "
              f"joint results: {n_joint:>5,}  "
              f"elapsed: {elapsed:.0f}s", end="\r")

    # Sort by pairwise CPE descending
    candidates = (group
                  .sort_values("CPE", ascending=False)
                  .reset_index(drop=True))
    if len(candidates) < 2:
        continue

    # Common date index
    fy_series  = future_inc[tau_f][y].dropna()
    common_idx = fy_series.index
    for tau_p in TAU_LIST:
        common_idx = common_idx.intersection(
            increments[tau_p].dropna(how="all").index)
    if len(common_idx) < MIN_N:
        continue

    uncond_prob = 1.0 - q_y

    # Greedy selection
    # selected: list of dicts with keys X, tau_past, q_X, CPE
    # selected_tickers: set of unique tickers already selected
    first_row   = candidates.iloc[0]
    selected    = [first_row.to_dict()]
    selected_tickers = {first_row["X"]}

    # Pre-compute joint mask for selected set
    joint_mask = get_condition_mask(
        first_row["X"], int(first_row["tau_past"]),
        first_row["q_X"], direction, common_idx)
    if joint_mask is None:
        continue

    while len(selected) < MAX_PREDICTORS:
        best_cpe  = -1
        best_row  = None
        best_mask = None
        best_n    = 0

        for _, cand in candidates.iterrows():
            # Skip if ticker already in selected set
            if cand["X"] in selected_tickers:
                continue

            cand_mask = get_condition_mask(
                cand["X"], int(cand["tau_past"]),
                cand["q_X"], direction, common_idx)
            if cand_mask is None:
                continue

            trial_mask = joint_mask & cand_mask
            n_trial    = trial_mask.sum()
            if n_trial < MIN_N:
                continue

            # Compute joint CPE
            fy_vals = future_inc[tau_f][y].loc[common_idx].values
            thresh_y_up = thresholds[(tau_f, q_y)].get(y, np.nan)
            thresh_y_dn = thresholds[(tau_f, round(1 - q_y, 10))].get(y, np.nan)
            if direction == "bullish":
                if np.isnan(thresh_y_up): continue
                event = fy_vals > thresh_y_up
            else:
                if np.isnan(thresh_y_dn): continue
                event = fy_vals < thresh_y_dn

            cpe  = float(np.nanmean(event[trial_mask]))
            lift = cpe / uncond_prob if uncond_prob > 0 else np.nan

            if cpe >= CPE_THRESH and lift >= MIN_LIFT and cpe > best_cpe:
                best_cpe  = cpe
                best_row  = cand
                best_mask = trial_mask
                best_n    = int(n_trial)

        if best_row is None:
            break  # no valid addition found

        # Accept best addition
        selected.append(best_row.to_dict())
        selected_tickers.add(best_row["X"])
        joint_mask = best_mask

        # Save this joint set (size >= 2)
        if len(selected) >= 2:
            lift = best_cpe / uncond_prob
            results.append({
                "Y":             y,
                "direction":     direction,
                "tau_future":    int(tau_f),
                "q_Y":           q_y,
                "n_predictors":  len(selected),
                "predictors":    [r["X"]         for r in selected],
                "tau_pasts":     [int(r["tau_past"]) for r in selected],
                "q_Xs":          [r["q_X"]        for r in selected],
                "pairwise_CPEs": [round(float(r["CPE"]), 4) for r in selected],
                "joint_CPE":     round(best_cpe, 4),
                "uncond_prob":   round(uncond_prob, 4),
                "lift":          round(lift, 4),
                "n_joint":       best_n,
                "n_total":       len(common_idx),
            })
            n_joint += 1

elapsed = time.time() - t0
print(f"\n\n  Done. Elapsed: {elapsed:.0f}s  ({elapsed/60:.1f} min)")
print(f"  Groups processed : {n_groups:,}")
print(f"  Joint results    : {n_joint:,}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
if results:
    df = pd.DataFrame(results)
    df = df.sort_values(
        ["direction","n_predictors","joint_CPE","n_joint"],
        ascending=[True,True,False,False]
    ).reset_index(drop=True)

    out = "joint_cpe_results.parquet"
    df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)

    print(f"\n{'='*65}")
    print(f"  COMPLETE")
    print(f"  Saved {len(df):,} rows → {out}  ({os.path.getsize(out)/1e6:.2f} MB)")

    print(f"\n  Direction × n_predictors breakdown:")
    print(df.groupby(["direction","n_predictors"]).size()
            .unstack(fill_value=0).to_string())

    print(f"\n  Mean joint CPE by n_predictors:")
    print(df.groupby(["direction","n_predictors"])["joint_CPE"]
            .mean().round(4).unstack().to_string())

    for direction in ["bullish","bearish"]:
        print(f"\n  Top 10 {direction} (size=2, ranked by joint_CPE then n_joint):")
        top = (df[(df["direction"]==direction) & (df["n_predictors"]==2)]
               .sort_values(["joint_CPE","n_joint"], ascending=[False,False])
               .head(10))
        for _, r in top.iterrows():
            print(f"    Y={r['Y']:<14} τf={r['tau_future']:>3}  qY={r['q_Y']}"
                  f"  CPE={r['joint_CPE']:.4f}  n={r['n_joint']}"
                  f"  predictors={list(zip(r['predictors'],r['tau_pasts'],r['q_Xs']))}")

    print(f"{'='*65}\n")
else:
    print("\n  No joint results passed the filters.")

if __name__ == "__main__":
    pass
