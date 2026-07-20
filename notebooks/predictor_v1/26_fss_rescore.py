"""
Re-score the SPY conditional-distribution model (19_spy_distribution_model_refined.py)
using Fractional Skill Score instead of R2/RMSE-family metrics, per explicit
user request: R2 rewards nailing the exact-day value of an overlapping,
noisy 21d-forward return -- FSS instead asks whether the model gets the RATE
of extreme moves right over a trailing window, which is a much closer match
to what these predictors were validated for (SPY future realized
volatility/regime, not exact return magnitude -- see causal_validation_results.json).

Identical walk-forward/purge machinery to 19_*.py (same three variants:
vix_only, credit_only, both), so results are directly comparable to that
script's R2/twCRPS numbers. Two thresholds per variant: upper-tail (predict
large positive 21d moves) and lower-tail (predict large negative 21d moves),
each at the 90th/10th percentile of the full OOS y_true distribution.

Run: python 26_fss_rescore.py
Output: results_fss_rescore.json
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from loss_functions import fss_from_quantiles

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
HORIZON = 21
INITIAL_TRAIN_YEARS = 6
STEP_YEARS = 1
ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]

print("=" * 60)
print("  FSS RE-SCORE: SPY distribution model, vix_only vs credit_only vs both")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
spy = prices["SPY"].dropna()
fwd_ret = np.log(spy.shift(-HORIZON) / spy).rename("fwd_ret")
date_pos = {d: i for i, d in enumerate(prices.index)}

macro = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro["date"] = pd.to_datetime(macro["date"])
credit_spread_regime = macro[["date", "credit_spread_regime"]].drop_duplicates("date").set_index("date")["credit_spread_regime"]

vixy = prices["VIXY"].dropna()
vixm = prices["VIXM"].dropna()
common = vixy.index.intersection(vixm.index)
raw_ratio = np.log(vixm.reindex(common) / vixy.reindex(common))
vix_term_slope = ((raw_ratio - raw_ratio.rolling(200, min_periods=100).mean())
                   / raw_ratio.rolling(200, min_periods=100).std()).rename("vix_term_slope")

d_full = pd.concat([fwd_ret, credit_spread_regime, vix_term_slope], axis=1).dropna()
d_full["date"] = d_full.index
d_full["pos"] = d_full["date"].map(date_pos)
d_full = d_full.reset_index(drop=True)
print(f"Dataset: {len(d_full)} rows, {d_full['date'].min().date()} .. {d_full['date'].max().date()}")

min_year, max_year = d_full["date"].dt.year.min(), d_full["date"].dt.year.max()
first_test_year = min_year + INITIAL_TRAIN_YEARS

VARIANTS = {
    "climatology": [],  # no-skill baseline: unconditional training-quantiles, no vix/credit info
    "vix_only": ["vix_term_slope"],
    "credit_only": ["credit_spread_regime"],
    "both": ["credit_spread_regime", "vix_term_slope"],
}

# True climatology reference, day-of-year granularity (the user's explicit
# choice -- "monthly" in the original explanation was just one illustration
# of the general principle, not the spec): each calendar (month, day) gets
# its own value, averaged across the reference years only, applied to the
# matching calendar day in the OOS period. With ~6 reference years, most
# buckets have only ~5-6 raw observations -- computed as-is, no small-n
# fallback or correction, a sample is a sample. The only fallback is for
# calendar days with ZERO observations across all reference years (e.g.
# market holidays like Dec 25/Jan 1, which never appear in a trading price
# series at all) -- that's a data-availability necessity, not a distrust of
# small samples, so it falls back to the pooled reference distribution only
# in that literal zero-observation case.
first_test_start = pd.Timestamp(f"{first_test_year}-01-01")
first_test_start_pos = date_pos.get(first_test_start, None)
if first_test_start_pos is None:
    candidates = [p for dt, p in date_pos.items() if dt >= first_test_start]
    first_test_start_pos = min(candidates)
initial_train_mask = (d_full["date"] < first_test_start) & (d_full["pos"] + HORIZON < first_test_start_pos)
initial_train = d_full.loc[initial_train_mask, ["date", "fwd_ret"]].copy()
initial_train["mmdd"] = list(zip(initial_train["date"].dt.month, initial_train["date"].dt.day))
pooled_vals = initial_train["fwd_ret"].values
climatology_quantiles_by_day = {}
n_empty = 0
for m in range(1, 13):
    days_in_month = 29 if m == 2 else (30 if m in (4, 6, 9, 11) else 31)
    for dd in range(1, days_in_month + 1):
        vals = initial_train.loc[initial_train["mmdd"] == (m, dd), "fwd_ret"].values
        if len(vals) == 0:
            vals = pooled_vals
            n_empty += 1
        climatology_quantiles_by_day[(m, dd)] = {a: float(np.quantile(vals, a)) for a in ALPHAS}
print(f"\nFixed climatology reference period: {initial_train['date'].min().date()} "
      f".. {initial_train['date'].max().date()} (n={len(initial_train)}), day-of-year climatology "
      f"({len(climatology_quantiles_by_day)} calendar-day buckets, {n_empty} empty -> pooled fallback), "
      f"held constant for the entire OOS period thereafter.")
sample_days = [(1, 15), (7, 15), (12, 25)]
for m, dd in sample_days:
    n_d = (initial_train["mmdd"] == (m, dd)).sum()
    print(f"  e.g. {m:02d}-{dd:02d} (n={n_d}): " +
          ", ".join(f"q{a}={climatology_quantiles_by_day[(m,dd)][a]:.4f}" for a in ALPHAS))

all_results = {}
oos_by_variant = {}

for variant_name, feature_cols in VARIANTS.items():
    print(f"\n=== Variant: {variant_name} ({feature_cols}) ===")
    oos_rows = []
    test_year = first_test_year
    while test_year <= max_year:
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year + STEP_YEARS}-01-01")
        test_mask = (d_full["date"] >= test_start) & (d_full["date"] < test_end)
        if test_mask.sum() < 20:
            test_year += STEP_YEARS
            continue
        test_start_pos = date_pos.get(test_start, None)
        if test_start_pos is None:
            candidates = [p for dt, p in date_pos.items() if dt >= test_start]
            test_start_pos = min(candidates) if candidates else None
        train_mask = (d_full["date"] < test_start) & (d_full["pos"] + HORIZON < test_start_pos)
        if train_mask.sum() < 200:
            test_year += STEP_YEARS
            continue

        ytr = d_full.loc[train_mask, "fwd_ret"]
        yte = d_full.loc[test_mask, "fwd_ret"]
        row = d_full.loc[test_mask, ["date"]].copy()
        row["y_true"] = yte.values

        if variant_name == "climatology":
            # Each row gets ITS OWN calendar day's fixed reference quantiles
            # (computed once above, never refit) -- day-of-year granularity,
            # not a flat monthly or yearly number.
            row_mmdd = list(zip(row["date"].dt.month, row["date"].dt.day))
            for a in ALPHAS:
                row[f"q{a}"] = [climatology_quantiles_by_day[k][a] for k in row_mmdd]
        else:
            Xtr, Xte = d_full.loc[train_mask, feature_cols], d_full.loc[test_mask, feature_cols]
            Xtr_sm = sm.add_constant(Xtr)
            Xte_sm = sm.add_constant(Xte, has_constant="add")
            for a in ALPHAS:
                qr_model = QuantReg(ytr, Xtr_sm).fit(q=a)
                row[f"q{a}"] = qr_model.predict(Xte_sm).values
        oos_rows.append(row)
        test_year += STEP_YEARS

    oos = pd.concat(oos_rows, ignore_index=True).sort_values("date").reset_index(drop=True)
    oos_by_variant[variant_name] = oos
    print(f"  n_oos={len(oos)}")

# Multi-scale, multi-threshold FSS grid, per user direction: don't collapse
# to one window/one threshold, sweep both and average -- and prefer
# thresholds/windows with actual financial meaning over arbitrary percentiles.
#   Windows: 21/63/126/252d ~ 1mo/1q/6mo/1yr, the horizons that actually matter
#   for position-sizing/risk decisions.
#   Thresholds: fixed 21d log-return magnitudes (not percentiles, so they mean
#   the same thing in every fold) at moderate/large/severe move categories.
#   -0.10 on the downside is not arbitrary -- it's the exact cutoff
#   17_stress_episode_composite.py used to define a real SPY drawdown episode.
WINDOWS = [21, 63, 126, 252]
UPPER_THRESHOLDS = [0.05, 0.075, 0.10]
LOWER_THRESHOLDS = [-0.05, -0.075, -0.10]

print(f"\nGrid: windows={WINDOWS}, upper thresholds={UPPER_THRESHOLDS}, lower thresholds={LOWER_THRESHOLDS}")

grid = {}  # grid[variant][window][direction][threshold] = fss
for variant_name, oos in oos_by_variant.items():
    y_true = oos["y_true"].values
    quantile_preds = {a: oos[f"q{a}"].values for a in ALPHAS}
    grid[variant_name] = {}
    for w in WINDOWS:
        grid[variant_name][w] = {"above": {}, "below": {}}
        for thr in UPPER_THRESHOLDS:
            grid[variant_name][w]["above"][thr] = fss_from_quantiles(
                y_true, quantile_preds, ALPHAS, thr, direction="above", window=w)
        for thr in LOWER_THRESHOLDS:
            grid[variant_name][w]["below"][thr] = fss_from_quantiles(
                y_true, quantile_preds, ALPHAS, thr, direction="below", window=w)

print("\n=== FSS grid (rows=window, cols=threshold), skill above climatology in parentheses ===")
for variant_name in oos_by_variant:
    if variant_name == "climatology":
        continue
    print(f"\n--- {variant_name} ---")
    header = "  window | " + " | ".join(f"up{t:+.3f}" for t in UPPER_THRESHOLDS) + " | " + \
             " | ".join(f"dn{t:+.3f}" for t in LOWER_THRESHOLDS)
    print(header)
    for w in WINDOWS:
        cells = []
        for thr in UPPER_THRESHOLDS:
            v, base = grid[variant_name][w]["above"][thr], grid["climatology"][w]["above"][thr]
            cells.append(f"{v:.3f}({v - base:+.3f})")
        for thr in LOWER_THRESHOLDS:
            v, base = grid[variant_name][w]["below"][thr], grid["climatology"][w]["below"][thr]
            cells.append(f"{v:.3f}({v - base:+.3f})")
        print(f"  {w:6d} | " + " | ".join(cells))

print("\n=== Aggregated across the full grid (24 window x threshold x direction cells) ===")
for variant_name in oos_by_variant:
    all_vals, upper_vals, lower_vals, skill_above_base = [], [], [], []
    by_window = {w: [] for w in WINDOWS}
    for w in WINDOWS:
        for thr in UPPER_THRESHOLDS:
            v = grid[variant_name][w]["above"][thr]
            base = grid["climatology"][w]["above"][thr]
            all_vals.append(v); upper_vals.append(v); by_window[w].append(v)
            skill_above_base.append(v - base)
        for thr in LOWER_THRESHOLDS:
            v = grid[variant_name][w]["below"][thr]
            base = grid["climatology"][w]["below"][thr]
            all_vals.append(v); lower_vals.append(v); by_window[w].append(v)
            skill_above_base.append(v - base)

    grid_avg = float(np.mean(all_vals))
    upper_avg = float(np.mean(upper_vals))
    lower_avg = float(np.mean(lower_vals))
    skill_avg = float(np.mean(skill_above_base))
    window_curve = {w: float(np.mean(by_window[w])) for w in WINDOWS}
    med_r2 = 1 - np.sum((oos_by_variant[variant_name]["y_true"].values - oos_by_variant[variant_name]["q0.5"].values) ** 2) / \
        np.sum((oos_by_variant[variant_name]["y_true"].values - oos_by_variant[variant_name]["y_true"].values.mean()) ** 2)

    print(f"  {variant_name}: grid_avg_FSS={grid_avg:.4f}, avg_skill_above_climatology={skill_avg:+.4f}, "
          f"upper_avg={upper_avg:.4f}, lower_avg={lower_avg:.4f}  (median R2={med_r2:.4f})")
    print(f"    FSS-vs-window curve: " + ", ".join(f"{w}d={window_curve[w]:.4f}" for w in WINDOWS))

    all_results[variant_name] = {
        "n_oos": int(len(oos_by_variant[variant_name])), "grid_avg_fss": grid_avg,
        "avg_skill_above_climatology": skill_avg, "upper_avg": upper_avg, "lower_avg": lower_avg,
        "fss_vs_window": window_curve, "median_r2": float(med_r2),
        "grid": grid[variant_name],  # full per-cell (window x direction x threshold) grid, for plotting
    }

print("\n=== Ranking by avg_skill_above_climatology vs. by median R2 ===")
by_skill = sorted([v for v in all_results if v != "climatology"],
                   key=lambda v: all_results[v]["avg_skill_above_climatology"], reverse=True)
by_r2 = sorted([v for v in all_results if v != "climatology"],
               key=lambda v: all_results[v]["median_r2"], reverse=True)
print(f"  By avg_skill_above_climatology: {by_skill}")
print(f"  By R2:                          {by_r2}")

out_path = os.path.join(OUT_DIR, "results_fss_rescore.json")
with open(out_path, "w") as f:
    json.dump({"windows": WINDOWS, "upper_thresholds": UPPER_THRESHOLDS,
                "lower_thresholds": LOWER_THRESHOLDS, "results": all_results}, f, indent=2, default=float)
print(f"\nSaved results to {out_path}")
print("\nDone.")
