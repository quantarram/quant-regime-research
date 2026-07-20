"""
Diagnose WHY macro_interaction + L2 was the standout Wave-1 cell at the
21-day horizon, before deciding on Wave 2.

Four things checked, all against the same walk-forward/purge machinery used
throughout this program:

1. Is it robust, or a few folds/instruments carrying the whole result?
   Per-fold and per-instrument breakdown of extreme_hit_rate (pooled Wave-1
   number was 0.204 vs ~0.10 chance).

2. IMPORTANT GAP CLOSED: features_macro_interaction.parquet actually contains
   BOTH the raw regime columns (uup_trend_regime, credit_spread_regime --
   literally parallel columns, the exact naive pattern this whole redesign
   was reacting against) AND the multiplicative interaction terms
   (interact_gap_tau21_q4_*, interact_xi_q4_*). Wave 1 tested them bundled
   together, so it can't distinguish "the interaction structure matters" from
   "any macro feature at all would have helped." This re-runs three variants
   at 21d/L2/macro config: raw-only, interaction-only, both (=Wave 1's number).

3. Feature importance from the actual fitted models -- which specific
   feature(s) the trees are actually splitting on.

4. Direct economic interpretation: conditional on credit_spread_regime /
   uup_trend_regime being in a stressed state, is the realized rate of
   extreme downside 21-day moves actually elevated? (a plain crosstab, no
   model involved, the most basic possible check that this is econometrically
   sensible and not just an ML artifact)

Run: python 07_diagnose_macro_interaction.py
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.metrics import r2_score

from loss_functions import extreme_hit_rate, tail_pinball_loss

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
H = 21
INITIAL_TRAIN_YEARS = 6
STEP_YEARS = 1
LGB_BASE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)

print("=" * 60)
print("  DIAGNOSING macro_interaction + L2 @ 21d")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
baseline = pd.read_parquet(os.path.join(OUT_DIR, "features_daily_panel.parquet"))
baseline["date"] = pd.to_datetime(baseline["date"])
TICKERS = sorted(baseline["ticker"].unique())
date_pos = {d: i for i, d in enumerate(prices.index)}

s_map = {t: prices[t] for t in TICKERS}
ret_frames = []
for t in TICKERS:
    s = s_map[t]
    df = pd.DataFrame({"date": s.index, "ticker": t, "price_now": s.values})
    fwd_price = s.shift(-H)
    df[f"fwd_{H}"] = np.log(fwd_price.values / s.values)
    ret_frames.append(df)
ret_df = pd.concat(ret_frames, ignore_index=True)
baseline = baseline.merge(ret_df, on=["ticker", "date"], how="left")

z_cols = [c for c in baseline.columns if c.endswith("_z")]
ctx_cols = [c for c in baseline.columns if c.startswith("ctx_")]
for c in ctx_cols:
    baseline[c] = np.sign(baseline[c]) * np.log1p(np.abs(baseline[c]))
BASELINE_FEATURE_COLS = z_cols + ctx_cols + ["self_ref_score"]

macro_interaction = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro_interaction["date"] = pd.to_datetime(macro_interaction["date"])

RAW_COLS = ["uup_trend_regime", "credit_spread_regime"]
INTERACT_COLS = ["interact_gap_tau21_q4_uup", "interact_gap_tau21_q4_credit",
                  "interact_xi_q4_uup", "interact_xi_q4_credit"]

d_full = baseline.merge(macro_interaction, on=["ticker", "date"], how="left")

# same fair-comparison floor as the Wave-1 grid
floor = macro_interaction.dropna(subset=RAW_COLS + INTERACT_COLS, how="all")["date"].min()
d_full = d_full[d_full["date"] >= floor].copy()
label_col = f"fwd_{H}"
print(f"Date floor: {floor.date()}")


def walkforward(feature_cols, collect_importance=False):
    dd = d_full.dropna(subset=feature_cols + [label_col]).copy()
    dd["pos"] = dd["date"].map(date_pos)
    min_year, max_year = dd["date"].dt.year.min(), dd["date"].dt.year.max()
    first_test_year = min_year + INITIAL_TRAIN_YEARS

    oos_rows = []
    importances = []
    test_year = first_test_year
    while test_year <= max_year:
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year + STEP_YEARS}-01-01")
        test_mask = (dd["date"] >= test_start) & (dd["date"] < test_end)
        if test_mask.sum() < 30:
            test_year += STEP_YEARS
            continue
        test_start_pos = date_pos.get(test_start, None)
        if test_start_pos is None:
            candidates = [p for dt, p in date_pos.items() if dt >= test_start]
            test_start_pos = min(candidates) if candidates else None
        train_mask = (dd["date"] < test_start) & (dd["pos"] + H < test_start_pos)
        if train_mask.sum() < 200:
            test_year += STEP_YEARS
            continue

        Xtr, ytr = dd.loc[train_mask, feature_cols], dd.loc[train_mask, label_col]
        Xte, yte = dd.loc[test_mask, feature_cols], dd.loc[test_mask, label_col]
        m = lgb.LGBMRegressor(**LGB_BASE, objective="regression")
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)

        rec = dd.loc[test_mask, ["ticker", "date"]].copy()
        rec["y_true"] = yte.values
        rec["y_pred"] = pred
        rec["test_year"] = test_year
        oos_rows.append(rec)
        if collect_importance:
            importances.append(pd.Series(m.feature_importances_, index=feature_cols, name=test_year))
        test_year += STEP_YEARS

    pooled = pd.concat(oos_rows, ignore_index=True)
    imp_df = pd.concat(importances, axis=1) if importances else None
    return pooled, imp_df


# ── 1&2. Ablation: raw-only vs interaction-only vs both (Wave-1's actual config) ──
print("\n--- Ablation: raw macro columns vs interaction terms vs both ---")
variants = {
    "raw_only": BASELINE_FEATURE_COLS + RAW_COLS,
    "interaction_only": BASELINE_FEATURE_COLS + INTERACT_COLS,
    "both (Wave-1)": BASELINE_FEATURE_COLS + RAW_COLS + INTERACT_COLS,
}
pooled_by_variant = {}
for name, cols in variants.items():
    pooled, imp = walkforward(cols, collect_importance=(name == "both (Wave-1)"))
    pooled_by_variant[name] = pooled
    hr = extreme_hit_rate(pooled["y_true"].values, pooled["y_pred"].values)
    tp10 = tail_pinball_loss(pooled["y_true"].values, pooled["y_pred"].values, 0.1)
    r2 = r2_score(pooled["y_true"], pooled["y_pred"])
    print(f"  {name}: n={len(pooled)}, extreme_hit_rate={hr:.4f}, tail_pinball_0.1={tp10:.5f}, r2={r2:.4f}")
    if imp is not None:
        importance_summary = imp.mean(axis=1).sort_values(ascending=False)

print("\n--- Feature importance (mean gain-based split count across folds, 'both' model) ---")
print(importance_summary.head(15))
print("\n  ...macro-related features specifically:")
macro_related = [c for c in importance_summary.index if any(k in c for k in ["uup", "credit", "interact"])]
print(importance_summary.loc[macro_related])

# ── 3. Per-fold and per-instrument breakdown of the "both" (Wave-1) result ──
pooled_both = pooled_by_variant["both (Wave-1)"]
print("\n--- Per-fold extreme_hit_rate (both/Wave-1 config) ---")
for yr, g in pooled_both.groupby("test_year"):
    hr = extreme_hit_rate(g["y_true"].values, g["y_pred"].values)
    print(f"  {yr}: n={len(g)}, extreme_hit_rate={hr:.4f}")

print("\n--- Per-instrument extreme_hit_rate (both/Wave-1 config, pooled across folds) ---")
for tkr, g in pooled_both.groupby("ticker"):
    if len(g) < 50:
        continue
    hr = extreme_hit_rate(g["y_true"].values, g["y_pred"].values)
    print(f"  {tkr}: n={len(g)}, extreme_hit_rate={hr:.4f}")

# ── 4. Direct economic check: does realized extreme-downside rate rise when ──
# ── credit_spread_regime / uup_trend_regime are themselves in a stressed state? ──
print("\n--- Direct economic check: extreme-downside rate conditional on macro regime state ---")
econ = d_full.dropna(subset=[label_col] + RAW_COLS).copy()
lo_thresh = econ[label_col].quantile(0.1)
econ["is_extreme_down"] = econ[label_col] <= lo_thresh
base_rate = econ["is_extreme_down"].mean()
print(f"  Unconditional P(extreme downside 21d move) = {base_rate:.4f} (should be ~0.10 by construction)")
for col in RAW_COLS:
    stressed = econ[col] > 1.0  # more than 1 std above its own 200d mean
    cond_rate = econ.loc[stressed, "is_extreme_down"].mean()
    calm_rate = econ.loc[~stressed, "is_extreme_down"].mean()
    print(f"  {col} > +1 std: P(extreme downside) = {cond_rate:.4f} (n={stressed.sum()}) "
          f"vs <= +1 std: {calm_rate:.4f} (n={(~stressed).sum()})")

print("\nDone.")
