"""
Direct test: explicit nonlinear cross-family products (10_cross_family_nonlinear_features.py)
vs. Phase C's simple concatenation (macro_term_regime, from 06_phase_c_combinations.py) vs.
the single-family bests -- does genuine multiplicative combination beat just dumping all
three families' columns into one tree model?

Two predictor configs tested here, same 6 loss options and walk-forward machinery as
06_phase_c_combinations.py:
  nonlinear_only     = baseline + the 6 explicit cross-family product terms only
  nonlinear_plus_all = baseline + macro_interaction + term_structure + regime_switching
                        + the 6 explicit product terms (concatenation AND explicit products together)

Run: python 06b_nonlinear_combination_grid.py
Output: results_nonlinear_combination.json
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error

from loss_functions import make_composite_objective, extreme_hit_rate, tail_pinball_loss

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

HORIZONS = [1, 5, 21]
INITIAL_TRAIN_YEARS = 6
STEP_YEARS = 1
LGB_BASE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)

print("=" * 60)
print("  EXPLICIT NONLINEAR CROSS-FAMILY COMBINATION vs. CONCATENATION")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
baseline = pd.read_parquet(os.path.join(OUT_DIR, "features_daily_panel.parquet"))
baseline["date"] = pd.to_datetime(baseline["date"])
TICKERS = sorted(baseline["ticker"].unique())
date_pos = {d: i for i, d in enumerate(prices.index)}

ret_frames = []
for t in TICKERS:
    s = prices[t]
    df = pd.DataFrame({"date": s.index, "ticker": t, "price_now": s.values})
    for h in HORIZONS:
        fwd_price = s.shift(-h)
        df[f"fwd_{h}"] = np.log(fwd_price.values / s.values)
    ret_frames.append(df)
ret_df = pd.concat(ret_frames, ignore_index=True)
baseline = baseline.merge(ret_df, on=["ticker", "date"], how="left")

z_cols = [c for c in baseline.columns if c.endswith("_z")]
ctx_cols = [c for c in baseline.columns if c.startswith("ctx_")]
for c in ctx_cols:
    baseline[c] = np.sign(baseline[c]) * np.log1p(np.abs(baseline[c]))
BASELINE_FEATURE_COLS = z_cols + ctx_cols + ["self_ref_score"]

nonlinear = pd.read_parquet(os.path.join(OUT_DIR, "features_cross_family_nonlinear.parquet"))
nonlinear["date"] = pd.to_datetime(nonlinear["date"])
nonlinear_cols = [c for c in nonlinear.columns if c != "date"]

macro = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro["date"] = pd.to_datetime(macro["date"])
macro_cols = [c for c in macro.columns if c not in ("ticker", "date")]
term = pd.read_parquet(os.path.join(OUT_DIR, "features_term_structure.parquet"))
term["date"] = pd.to_datetime(term["date"])
term_cols = [c for c in term.columns if c != "date"]
regime = pd.read_parquet(os.path.join(OUT_DIR, "features_regime_switching.parquet"))
regime["date"] = pd.to_datetime(regime["date"])
regime_cols = [c for c in regime.columns if c != "date"]

GLOBAL_START = nonlinear.dropna(subset=nonlinear_cols, how="all")["date"].min()
print(f"Fair-comparison date floor: {GLOBAL_START.date()}")

CONFIGS = {
    "nonlinear_only": lambda: (baseline.merge(nonlinear, on="date", how="left"),
                                list(BASELINE_FEATURE_COLS) + nonlinear_cols),
    "nonlinear_plus_all": lambda: (
        baseline.merge(macro, on=["ticker", "date"], how="left")
                .merge(term, on="date", how="left")
                .merge(regime, on="date", how="left")
                .merge(nonlinear, on="date", how="left"),
        list(BASELINE_FEATURE_COLS) + macro_cols + term_cols + regime_cols + nonlinear_cols),
}

LOSS_OPTIONS = {
    "l2": lambda: dict(objective="regression"),
    "combo_l2_lq4": lambda: dict(objective=make_composite_objective(
        [{"type": "l2", "weight": 0.5}, {"type": "lq", "q": 4, "weight": 0.5}])),
    "combo_l2_tail": lambda: dict(objective=make_composite_objective(
        [{"type": "l2", "weight": 0.5}, {"type": "tail_l2", "weight": 0.5}])),
    "combo_l2_pinball90": lambda: dict(objective=make_composite_objective(
        [{"type": "l2", "weight": 0.5}, {"type": "pinball", "alpha": 0.9, "weight": 0.5}])),
    "combo_lq4_pinball90": lambda: dict(objective=make_composite_objective(
        [{"type": "lq", "q": 4, "weight": 0.5}, {"type": "pinball", "alpha": 0.9, "weight": 0.5}])),
    "combo_all4": lambda: dict(objective=make_composite_objective(
        [{"type": "l2", "weight": 0.25}, {"type": "lq", "q": 4, "weight": 0.25},
         {"type": "pinball", "alpha": 0.1, "weight": 0.25}, {"type": "pinball", "alpha": 0.9, "weight": 0.25}])),
}


def run_walkforward(d, feature_cols, label_col, horizon, loss_builder):
    dd = d[d["date"] >= GLOBAL_START].dropna(subset=feature_cols + [label_col]).copy()
    dd["pos"] = dd["date"].map(date_pos)
    if len(dd) == 0:
        return None
    min_year, max_year = dd["date"].dt.year.min(), dd["date"].dt.year.max()
    first_test_year = min_year + INITIAL_TRAIN_YEARS

    oos_rows = []
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
        train_mask = (dd["date"] < test_start) & (dd["pos"] + horizon < test_start_pos)
        if train_mask.sum() < 200:
            test_year += STEP_YEARS
            continue

        Xtr, ytr = dd.loc[train_mask, feature_cols], dd.loc[train_mask, label_col]
        Xte, yte = dd.loc[test_mask, feature_cols], dd.loc[test_mask, label_col]

        m = lgb.LGBMRegressor(**LGB_BASE, **loss_builder())
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        oos_rows.append(pd.DataFrame({"y_true": yte.values, "y_pred": pred}))
        test_year += STEP_YEARS

    if not oos_rows:
        return None
    return pd.concat(oos_rows, ignore_index=True)


def score(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "dir_acc": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
        "extreme_hit_rate": float(extreme_hit_rate(y_true, y_pred)),
        "tail_pinball_0.1": float(tail_pinball_loss(y_true, y_pred, 0.1)),
        "tail_pinball_0.9": float(tail_pinball_loss(y_true, y_pred, 0.9)),
        "n": int(len(y_true)),
    }


results = {"global_start": str(GLOBAL_START.date()), "horizons": {}}

for h in HORIZONS:
    print(f"\n=== Horizon {h}d ===")
    label_col = f"fwd_{h}"
    results["horizons"][str(h)] = {}
    for cfg_name, cfg_builder in CONFIGS.items():
        d, feature_cols = cfg_builder()
        results["horizons"][str(h)][cfg_name] = {}
        for loss_name, loss_builder in LOSS_OPTIONS.items():
            pooled = run_walkforward(d, feature_cols, label_col, h, loss_builder)
            if pooled is None:
                print(f"  {cfg_name}/{loss_name}: no valid folds, skipping.")
                continue
            s = score(pooled["y_true"].values, pooled["y_pred"].values)
            results["horizons"][str(h)][cfg_name][loss_name] = s
            print(f"  {cfg_name}/{loss_name}: n={s['n']} hit={s['extreme_hit_rate']:.3f} r2={s['r2']:.3f}")

out_path = os.path.join(OUT_DIR, "results_nonlinear_combination.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved results to {out_path}")
print("\nDone.")
