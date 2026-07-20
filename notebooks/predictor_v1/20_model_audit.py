"""
Self-audit of the final model (19_spy_distribution_model_refined.py), in
response to being asked directly whether this was actually done properly.
Four checks I had NOT done before declaring vix_only the winner:

1. Quantile crossing: each alpha was fit as a SEPARATE QuantReg with no
   joint monotonicity constraint. Does q0.1 <= q0.25 <= q0.5 <= q0.75 <= q0.9
   actually hold row-by-row, or does "a coherent predicted distribution"
   silently not hold in some fraction of rows?
2. Paired significance: I declared vix_only's twCRPS edge over "both" as
   the tiebreaker, but never tested whether that gap is distinguishable
   from noise on n_oos=2345 -- a block bootstrap (blocked by year, since
   the 21-day-forward-return target is heavily overlapping and observations
   within a year are not independent -- NOT an n_condition/effective-N
   correction, just correct block resampling for a paired comparison).
3. Leakage re-check on THIS exact script: shift the label by one extra
   period and confirm OOS R2 collapses toward the shifted-baseline level,
   the same style of check used earlier in this program, run fresh here
   rather than assumed inherited from an earlier script.
4. Economic sanity check: does a trivial long/flat strategy driven by the
   model's predicted median actually beat buy-and-hold OOS, transaction
   costs aside -- the question that matters in the end, which I never
   circled back to for this specific final model.

Run: python 20_model_audit.py
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

REPO_DIR = "/Users/arrams/Documents/personal_work/quant/quant-regime-research"
OUT_DIR = "/Users/arrams/Documents/personal_work/quant/quant-regime-research/notebooks/predictor_v1"
HORIZON = 21
INITIAL_TRAIN_YEARS = 6
STEP_YEARS = 1
ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]

print("=" * 60)
print("  MODEL AUDIT")
print("=" * 60)

prices = pd.read_parquet(f"{REPO_DIR}/notebooks/multiasset_prices.parquet")
spy = prices["SPY"].dropna()
date_pos = {d: i for i, d in enumerate(prices.index)}

vixy = prices["VIXY"].dropna()
vixm = prices["VIXM"].dropna()
common = vixy.index.intersection(vixm.index)
raw_ratio = np.log(vixm.reindex(common) / vixy.reindex(common))
vix_term_slope = ((raw_ratio - raw_ratio.rolling(200, min_periods=100).mean())
                   / raw_ratio.rolling(200, min_periods=100).std()).rename("vix_term_slope")

macro = pd.read_parquet(f"{OUT_DIR}/features_macro_interaction.parquet")
macro["date"] = pd.to_datetime(macro["date"])
credit_spread_regime = macro[["date", "credit_spread_regime"]].drop_duplicates("date").set_index("date")["credit_spread_regime"]


def run_walkforward(feature_cols, label_shift=0):
    fwd_ret = np.log(spy.shift(-HORIZON - label_shift) / spy).rename("fwd_ret")
    d_full = pd.concat([fwd_ret, credit_spread_regime, vix_term_slope], axis=1).dropna()
    d_full["date"] = d_full.index
    d_full["pos"] = d_full["date"].map(date_pos)
    d_full = d_full.reset_index(drop=True)
    min_year, max_year = d_full["date"].dt.year.min(), d_full["date"].dt.year.max()
    first_test_year = min_year + INITIAL_TRAIN_YEARS

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
        train_mask = (d_full["date"] < test_start) & (d_full["pos"] + HORIZON + label_shift < test_start_pos)
        if train_mask.sum() < 200:
            test_year += STEP_YEARS
            continue

        Xtr, ytr = d_full.loc[train_mask, feature_cols], d_full.loc[train_mask, "fwd_ret"]
        Xte, yte = d_full.loc[test_mask, feature_cols], d_full.loc[test_mask, "fwd_ret"]
        Xtr_sm = sm.add_constant(Xtr)
        Xte_sm = sm.add_constant(Xte, has_constant="add")

        row = d_full.loc[test_mask, ["date"]].copy()
        row["y_true"] = yte.values
        row["year"] = test_year
        for a in ALPHAS:
            qr_model = QuantReg(ytr, Xtr_sm).fit(q=a)
            row[f"q{a}"] = qr_model.predict(Xte_sm).values
        oos_rows.append(row)
        test_year += STEP_YEARS

    return pd.concat(oos_rows, ignore_index=True)


print("\nRefitting vix_only and both (needed: I didn't save raw OOS predictions the first time)...")
oos_vix = run_walkforward(["vix_term_slope"])
oos_both = run_walkforward(["credit_spread_regime", "vix_term_slope"])

# ── Check 1: quantile crossing ───────────────────────────────────────────
print("\n--- Check 1: quantile monotonicity ---")
for name, oos in [("vix_only", oos_vix), ("both", oos_both)]:
    q_cols = [f"q{a}" for a in ALPHAS]
    vals = oos[q_cols].values
    violations = (np.diff(vals, axis=1) < 0).any(axis=1)
    print(f"  {name}: {violations.sum()} / {len(oos)} rows ({100*violations.mean():.2f}%) have a quantile-crossing violation")

# ── Check 2: paired significance, block bootstrap by year ───────────────
print("\n--- Check 2: is vix_only's twCRPS edge over 'both' real, or noise? ---")


def pinball(y, pred, a):
    r = y - pred
    return np.mean(np.maximum(a * r, (a - 1) * r))


def twcrps(oos):
    losses = []
    weights = []
    for a in ALPHAS:
        w = 3.0 if a in (0.1, 0.9) else 1.0
        losses.append(pinball(oos["y_true"].values, oos[f"q{a}"].values, a) * w)
        weights.append(w)
    return sum(losses) / sum(weights)


merged = oos_vix[["date", "year", "y_true"] + [f"q{a}" for a in ALPHAS]].merge(
    oos_both[["date"] + [f"q{a}" for a in ALPHAS]], on="date", suffixes=("_vix", "_both"))
years = sorted(merged["year"].unique())
rng = np.random.default_rng(0)
diffs = []
for _ in range(2000):
    sampled_years = rng.choice(years, size=len(years), replace=True)
    boot = pd.concat([merged[merged["year"] == y] for y in sampled_years], ignore_index=True)
    vix_boot = boot.rename(columns={f"q{a}_vix": f"q{a}" for a in ALPHAS})
    both_boot = boot.rename(columns={f"q{a}_both": f"q{a}" for a in ALPHAS})
    diffs.append(twcrps(both_boot) - twcrps(vix_boot))  # positive = vix_only better (lower twCRPS)
diffs = np.array(diffs)
point_estimate = twcrps(oos_both.rename(columns={f"q{a}": f"q{a}" for a in ALPHAS})) - twcrps(oos_vix)
print(f"  Point estimate (both_twcrps - vix_only_twcrps): {point_estimate:.6f} (positive = vix_only better)")
print(f"  Block-bootstrap (by year, n=2000): 95% CI = [{np.percentile(diffs, 2.5):.6f}, {np.percentile(diffs, 97.5):.6f}]")
print(f"  Fraction of bootstrap draws where vix_only is better: {(diffs > 0).mean():.1%}")
if np.percentile(diffs, 2.5) < 0 < np.percentile(diffs, 97.5):
    print("  -> CI spans zero: the 'vix_only wins' claim is NOT statistically distinguishable from noise at this sample size.")
else:
    print("  -> CI excludes zero: vix_only's edge is real at conventional confidence.")

# ── Check 3: leakage re-check on this exact script ───────────────────────
print("\n--- Check 3: leakage re-check (shifted label, vix_only) ---")
oos_shifted = run_walkforward(["vix_term_slope"], label_shift=1)
r2_real = 1 - np.sum((oos_vix["y_true"] - oos_vix["q0.5"]) ** 2) / np.sum((oos_vix["y_true"] - oos_vix["y_true"].mean()) ** 2)
r2_shifted = 1 - np.sum((oos_shifted["y_true"] - oos_shifted["q0.5"]) ** 2) / np.sum((oos_shifted["y_true"] - oos_shifted["y_true"].mean()) ** 2)
print(f"  Real (unshifted) median R2: {r2_real:.4f}")
print(f"  Shifted-label median R2:    {r2_shifted:.4f}")
print(f"  {'OK -- shifted result is not better than real, no evidence of leakage' if r2_shifted <= r2_real + 0.01 else 'WARNING -- shifted result is suspiciously close to or better than real, investigate'}")

# ── Check 4: economic sanity check ────────────────────────────────────────
print("\n--- Check 4: does this beat buy-and-hold OOS? ---")
oos_vix_sorted = oos_vix.sort_values("date").drop_duplicates("date")
strat_ret = np.where(oos_vix_sorted["q0.5"] > 0, oos_vix_sorted["y_true"], 0.0)
bh_ret = oos_vix_sorted["y_true"].values


def sharpe_from_overlapping(returns, horizon=21, periods_per_year=252):
    non_overlap = returns[::horizon]
    if non_overlap.std() == 0:
        return np.nan
    return float(np.mean(non_overlap) / non_overlap.std() * np.sqrt(periods_per_year / horizon))


print(f"  Strategy (long when predicted median > 0, else flat), non-overlapping subsample Sharpe: "
      f"{sharpe_from_overlapping(strat_ret):.3f}")
print(f"  Buy & hold, same non-overlapping subsample Sharpe: {sharpe_from_overlapping(bh_ret):.3f}")
print(f"  Strategy total return (sum of log returns, non-overlapping): {strat_ret[::21].sum():.3f}")
print(f"  Buy & hold total return (sum of log returns, non-overlapping): {bh_ret[::21].sum():.3f}")
print(f"  Fraction of time strategy is long: {(oos_vix_sorted['q0.5'] > 0).mean():.1%}")

print("\nDone.")
