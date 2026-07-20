"""
The first predictive model in this program built on causally-validated
inputs, targeting the right thing for what was actually found.

Inputs: exactly the two predictors that survived rigorous causal validation
(11_causal_validation.py, 12_vix_term_structure_causal.py) -- nothing else.
No engineered combinations, no extra columns "just in case" -- that pattern
was tested repeatedly earlier in this program (Wave 2, Phase C, the
nonlinear-combination test) and never outperformed a small, validated
feature set.
  credit_spread_regime  -- HYG/LQD nowcast, Granger-leads SPY realized vol
  vix_term_slope        -- VIXM/VIXY term structure, genuinely forecast-like,
                            Granger-leads SPY realized vol
  (confirmed independent of each other via a multivariate VAR test)

Target: SPY's forward 21-trading-day return DISTRIBUTION, not a point
estimate -- because that's what the causal evidence actually supports.
14_leverage_effect_test.py found neither predictor cleanly signals
direction, but both signal an asymmetric shape: elevated stress suppresses
the upper tail of forward returns without materially worsening the lower
tail. A single point forecast would throw that away; quantile regression at
{0.1, 0.25, 0.5, 0.75, 0.9} keeps it.

Scope: SPY only. Causality was only established for SPY -- extending this
to other instruments would require repeating the causal validation for each
one first, not just assuming a market-wide feature transfers.

Two model families compared at each quantile: classical linear QuantReg
(the same tool used in the causal-validation stage, giving continuity) and
LightGBM quantile regression (the nonlinear ML approach already established
as well-calibrated and best-behaved in this program's own Phase A ablation).
Same walk-forward-with-purge machinery already leakage-audited throughout
this program.

Run: python 15_spy_distribution_model.py
Output: results_spy_distribution_model.json, spy_distribution_oos_plot.png
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
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
print("  SPY CONDITIONAL RETURN-DISTRIBUTION MODEL")
print("  (credit_spread_regime + vix_term_slope -> SPY fwd 21d return)")
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

FEATURE_COLS = ["credit_spread_regime", "vix_term_slope"]
d = pd.concat([fwd_ret, credit_spread_regime, vix_term_slope], axis=1).dropna()
d["date"] = d.index
d["pos"] = d["date"].map(date_pos)
d = d.reset_index(drop=True)
print(f"Dataset: {len(d)} rows, {d['date'].min().date()} .. {d['date'].max().date()}")

min_year, max_year = d["date"].dt.year.min(), d["date"].dt.year.max()
first_test_year = min_year + INITIAL_TRAIN_YEARS
print(f"First OOS test year: {first_test_year}, last: {max_year}")

LGB_BASE = dict(n_estimators=150, max_depth=3, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=1.0, random_state=0, verbosity=-1)

oos_rows = []
test_year = first_test_year
while test_year <= max_year:
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year + STEP_YEARS}-01-01")
    test_mask = (d["date"] >= test_start) & (d["date"] < test_end)
    if test_mask.sum() < 20:
        test_year += STEP_YEARS
        continue
    test_start_pos = date_pos.get(test_start, None)
    if test_start_pos is None:
        candidates = [p for dt, p in date_pos.items() if dt >= test_start]
        test_start_pos = min(candidates) if candidates else None
    train_mask = (d["date"] < test_start) & (d["pos"] + HORIZON < test_start_pos)
    if train_mask.sum() < 200:
        test_year += STEP_YEARS
        continue

    Xtr, ytr = d.loc[train_mask, FEATURE_COLS], d.loc[train_mask, "fwd_ret"]
    Xte, yte = d.loc[test_mask, FEATURE_COLS], d.loc[test_mask, "fwd_ret"]
    Xtr_sm = sm.add_constant(Xtr)
    Xte_sm = sm.add_constant(Xte, has_constant="add")

    row = d.loc[test_mask, ["date"]].copy()
    row["y_true"] = yte.values
    for a in ALPHAS:
        qr_model = QuantReg(ytr, Xtr_sm).fit(q=a)
        row[f"qr_q{a}"] = qr_model.predict(Xte_sm).values

        lgb_model = lgb.LGBMRegressor(**LGB_BASE, objective="quantile", alpha=a)
        lgb_model.fit(Xtr, ytr)
        row[f"lgb_q{a}"] = lgb_model.predict(Xte)

    oos_rows.append(row)
    print(f"  {test_year}: n={int(test_mask.sum())} trained (train n={int(train_mask.sum())})")
    test_year += STEP_YEARS

oos = pd.concat(oos_rows, ignore_index=True)
print(f"\nTotal OOS rows: {len(oos)}")

results = {"n_oos": int(len(oos)), "models": {}}
for model_name in ["qr", "lgb"]:
    quantile_preds = {a: oos[f"{model_name}_q{a}"].values for a in ALPHAS}
    y_true = oos["y_true"].values

    cal = calibration_check(y_true, quantile_preds, ALPHAS)
    pinballs = {a: pinball_loss(y_true, quantile_preds[a], a) for a in ALPHAS}
    twcrps = crps_from_quantiles_weighted(y_true, quantile_preds, ALPHAS)

    print(f"\n--- {model_name} ---")
    print("  Calibration (empirical coverage, should match nominal alpha):")
    for a in ALPHAS:
        print(f"    q={a}: nominal={a}, empirical={cal[a]:.3f}, pinball={pinballs[a]:.5f}")
    print(f"  twCRPS = {twcrps:.5f}")

    med_r2 = 1 - np.sum((y_true - quantile_preds[0.5]) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    print(f"  median (q0.5) R2 = {med_r2:.4f}")

    spread = quantile_preds[0.9] - quantile_preds[0.1]
    results["models"][model_name] = {
        "calibration": cal, "pinball": pinballs, "twcrps": float(twcrps), "median_r2": float(med_r2),
        "mean_predicted_spread_q90_q10": float(spread.mean()),
    }

# ── Does the OOS model reproduce the upper-tail-suppression asymmetry found in 14_leverage_effect_test.py? ──
print("\n--- Asymmetry check: do OOS predicted quantiles react asymmetrically to the predictors? ---")
merged = oos.merge(d[["date"] + FEATURE_COLS], on="date", how="left")
for model_name in ["qr", "lgb"]:
    for feat in FEATURE_COLS:
        corr_q10 = np.corrcoef(merged[feat], merged[f"{model_name}_q0.1"])[0, 1]
        corr_q90 = np.corrcoef(merged[feat], merged[f"{model_name}_q0.9"])[0, 1]
        print(f"  {model_name}: corr({feat}, predicted_q0.1)={corr_q10:.3f}, "
              f"corr({feat}, predicted_q0.9)={corr_q90:.3f} "
              f"{'-- upside reacts more (matches leverage-effect test)' if abs(corr_q90) > abs(corr_q10) else '-- downside reacts more (does NOT match)'}")
        results["models"][model_name].setdefault("asymmetry_check", {})[feat] = {
            "corr_with_q0.1": float(corr_q10), "corr_with_q0.9": float(corr_q90)}

out_path = os.path.join(OUT_DIR, "results_spy_distribution_model.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved results to {out_path}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    oos_sorted = oos.sort_values("date")
    price_now = spy.reindex(oos_sorted["date"]).values
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, model_name, title in [(axes[0], "qr", "Linear QuantReg"), (axes[1], "lgb", "LightGBM quantile")]:
        ax.plot(oos_sorted["date"], oos_sorted["y_true"], color="black", lw=0.8, label="Actual fwd 21d return")
        ax.plot(oos_sorted["date"], oos_sorted[f"{model_name}_q0.5"], color="tab:blue", lw=1, label="Predicted median")
        ax.fill_between(oos_sorted["date"], oos_sorted[f"{model_name}_q0.1"], oos_sorted[f"{model_name}_q0.9"],
                         alpha=0.2, color="tab:blue", label="Predicted 10-90% band")
        ax.set_title(f"{title}: OOS SPY fwd 21d return, predicted vs actual")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spy_distribution_oos_plot.png"), dpi=120)
    print(f"Saved plot to {os.path.join(OUT_DIR, 'spy_distribution_oos_plot.png')}")
except Exception as e:
    print(f"Plot failed (non-fatal): {e}")

print("\nDone.")
