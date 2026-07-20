"""
Stepping back to finalize the model on the aggregate statistical
relationship, per the user's explicit direction after the event-level
mechanism search (17/18_*.py) came up empty: credit_spread_regime showed no
consistent individual-episode signature and its trend "build-up" pattern
didn't survive scrutiny (Mann-Whitney p=0.135, poor hit/false-alarm
tradeoffs), while vix_term_slope's upper-tail-suppression asymmetry
robustly reproduced out-of-sample in 15_spy_distribution_model.py. That's a
real distinction, but it doesn't automatically mean credit_spread_regime
should be dropped -- the event-study tested a different question (does it
have a crisis-precursor MECHANISM) than the Granger/VAR tests did (does it
have real DAY-TO-DAY predictive content, which it does, independently of
vix_term_slope, per 12_vix_term_structure_causal.py's multivariate test).

So: let the walk-forward evidence decide, the same way every other
predictor-inclusion question in this program has been settled. Three
variants, identical methodology (QuantReg, the model family that already
beat LightGBM in 15_spy_distribution_model.py for this small, clean feature
set) and identical walk-forward/purge machinery:
  vix_only     -- vix_term_slope alone
  credit_only  -- credit_spread_regime alone
  both         -- both together (15_spy_distribution_model.py's original spec)

Run: python 19_spy_distribution_model_refined.py
Output: results_spy_distribution_refined.json, spy_distribution_refined_plot.png
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from loss_functions import pinball_loss, crps_from_quantiles_weighted, calibration_check

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
HORIZON = 21
INITIAL_TRAIN_YEARS = 6
STEP_YEARS = 1
ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]

print("=" * 60)
print("  SPY DISTRIBUTION MODEL, REFINED: vix_only vs credit_only vs both")
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
    "vix_only": ["vix_term_slope"],
    "credit_only": ["credit_spread_regime"],
    "both": ["credit_spread_regime", "vix_term_slope"],
}

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

        Xtr, ytr = d_full.loc[train_mask, feature_cols], d_full.loc[train_mask, "fwd_ret"]
        Xte, yte = d_full.loc[test_mask, feature_cols], d_full.loc[test_mask, "fwd_ret"]
        Xtr_sm = sm.add_constant(Xtr)
        Xte_sm = sm.add_constant(Xte, has_constant="add")

        row = d_full.loc[test_mask, ["date"]].copy()
        row["y_true"] = yte.values
        for a in ALPHAS:
            qr_model = QuantReg(ytr, Xtr_sm).fit(q=a)
            row[f"q{a}"] = qr_model.predict(Xte_sm).values
        oos_rows.append(row)
        test_year += STEP_YEARS

    oos = pd.concat(oos_rows, ignore_index=True)
    oos_by_variant[variant_name] = oos
    y_true = oos["y_true"].values
    quantile_preds = {a: oos[f"q{a}"].values for a in ALPHAS}

    cal = calibration_check(y_true, quantile_preds, ALPHAS)
    pinballs = {a: pinball_loss(y_true, quantile_preds[a], a) for a in ALPHAS}
    twcrps = crps_from_quantiles_weighted(y_true, quantile_preds, ALPHAS)
    med_r2 = 1 - np.sum((y_true - quantile_preds[0.5]) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

    print(f"  n_oos={len(oos)}")
    print("  Calibration (empirical vs nominal):")
    for a in ALPHAS:
        print(f"    q={a}: nominal={a}, empirical={cal[a]:.3f}, pinball={pinballs[a]:.5f}")
    print(f"  twCRPS={twcrps:.5f}, median R2={med_r2:.4f}")

    all_results[variant_name] = {"n_oos": int(len(oos)), "calibration": cal, "pinball": pinballs,
                                  "twcrps": float(twcrps), "median_r2": float(med_r2)}

print("\n=== Summary: which variant wins on each metric ===")
for metric in ["twcrps", "median_r2"]:
    best = min(all_results, key=lambda v: all_results[v][metric]) if metric == "twcrps" else \
        max(all_results, key=lambda v: all_results[v][metric])
    print(f"  Best by {metric}: {best} ({all_results[best][metric]:.5f})")
    for v in VARIANTS:
        print(f"    {v}: {all_results[v][metric]:.5f}")

out_path = os.path.join(OUT_DIR, "results_spy_distribution_refined.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, default=float)
print(f"\nSaved results to {out_path}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for ax, variant_name in zip(axes, VARIANTS):
        oos = oos_by_variant[variant_name].sort_values("date")
        ax.plot(oos["date"], oos["y_true"], color="black", lw=0.8, label="Actual fwd 21d return")
        ax.plot(oos["date"], oos["q0.5"], color="tab:blue", lw=1, label="Predicted median")
        ax.fill_between(oos["date"], oos["q0.1"], oos["q0.9"], alpha=0.2, color="tab:blue", label="10-90% band")
        ax.set_title(f"{variant_name}: OOS SPY fwd 21d return, predicted vs actual "
                     f"(twCRPS={all_results[variant_name]['twcrps']:.5f}, R2={all_results[variant_name]['median_r2']:.4f})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spy_distribution_refined_plot.png"), dpi=120)
    print(f"Saved plot to {os.path.join(OUT_DIR, 'spy_distribution_refined_plot.png')}")
except Exception as e:
    print(f"Plot failed (non-fatal): {e}")

print("\nDone.")
