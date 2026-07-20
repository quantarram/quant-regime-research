"""
Wave-1 grid: every predictor configuration x every loss function, evaluated
primarily on tail-aware metrics (not pooled R2, per the user's correction
that R2/dir_acc quietly reward missing the extremes).

Predictor configs: baseline (existing features_daily_panel.parquet),
  + cross-field multifractal coupling, + term-structure curvature,
  + macro regime multiplicative interaction (04_*.py panels built this round).
Loss variants: l2, lq_q4, tail_weighted (1 model/fold each) and a shared
  5-quantile grid (alphas 0.1/0.25/0.5/0.75/0.9, 1 fit-set/fold) that derives
  quantile_0.1, quantile_0.9, and distributional as three named readings of
  the SAME fit rather than three redundant re-fits.

All predictor configs are restricted to the SAME date range (the latest
common start date across the Wave-1 panels, ~2008, driven by ^GVZ) so the
grid is a genuinely fair apples-to-apples comparison -- this means baseline
numbers here will differ from v1/Phase A's full-history numbers, and that's
disclosed rather than papered over.

Run: python 05_grid_runner.py
Requires: features_daily_panel.parquet, features_cross_field.parquet,
          features_term_structure.parquet, features_macro_interaction.parquet,
          ../multiasset_prices.parquet
Output: results_grid_wave1.json
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error

from loss_functions import (make_lq_objective, tail_weights, pinball_loss,
                             crps_from_quantiles_weighted, calibration_check,
                             extreme_hit_rate, tail_pinball_loss)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

HORIZONS = [1, 5, 21]
INITIAL_TRAIN_YEARS = 6   # shorter than v1's 10y since the fair-comparison date floor (~2008) leaves less history
STEP_YEARS = 1
DIST_ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]
LGB_BASE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)

print("=" * 60)
print("  WAVE-1 GRID: 4 PREDICTOR CONFIGS x 6 LOSS VARIANTS")
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

cross_field = pd.read_parquet(os.path.join(OUT_DIR, "features_cross_field.parquet"))
cross_field["date"] = pd.to_datetime(cross_field["date"])
term_structure = pd.read_parquet(os.path.join(OUT_DIR, "features_term_structure.parquet"))
term_structure["date"] = pd.to_datetime(term_structure["date"])
macro_interaction = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro_interaction["date"] = pd.to_datetime(macro_interaction["date"])

cf_cols = [c for c in cross_field.columns if c != "date"]
ts_cols = [c for c in term_structure.columns if c != "date"]
mi_cols = [c for c in macro_interaction.columns if c not in ("ticker", "date")]

# Fair-comparison date floor: latest date at which EVERY Wave-1 panel has data.
floors = [
    cross_field.dropna(subset=cf_cols, how="all")["date"].min(),
    term_structure.dropna(subset=ts_cols, how="all")["date"].min(),
    macro_interaction.dropna(subset=mi_cols, how="all")["date"].min(),
]
GLOBAL_START = max(floors)
print(f"Fair-comparison date floor (latest common start across Wave-1 panels): {GLOBAL_START.date()}")


def build_config(name):
    if name == "baseline":
        d = baseline.copy()
        feature_cols = list(BASELINE_FEATURE_COLS)
    elif name == "cross_field":
        d = baseline.merge(cross_field, on="date", how="left")
        feature_cols = list(BASELINE_FEATURE_COLS) + cf_cols
    elif name == "term_structure":
        d = baseline.merge(term_structure, on="date", how="left")
        feature_cols = list(BASELINE_FEATURE_COLS) + ts_cols
    elif name == "macro_interaction":
        d = baseline.merge(macro_interaction, on=["ticker", "date"], how="left")
        feature_cols = list(BASELINE_FEATURE_COLS) + mi_cols
    else:
        raise ValueError(name)
    d = d[d["date"] >= GLOBAL_START].copy()
    return d, feature_cols


def run_walkforward(d, feature_cols, label_col, horizon):
    """Walk-forward with purge, same logic as 02_train_predict_daily.py /
    03_ablation_loss.py. Returns pooled OOS predictions across all folds."""
    dd = d.dropna(subset=feature_cols + [label_col]).copy()
    dd["pos"] = dd["date"].map(date_pos)
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

        preds = {"y_true": yte.values}

        for name, params in [("l2", dict(objective="regression")),
                              ("lq_q4", dict(objective=make_lq_objective(4))),
                              ("tail_weighted", dict(objective="regression"))]:
            m = lgb.LGBMRegressor(**LGB_BASE, **params)
            if name == "tail_weighted":
                m.fit(Xtr, ytr, sample_weight=tail_weights(ytr.values))
            else:
                m.fit(Xtr, ytr)
            preds[name] = m.predict(Xte)

        for a in DIST_ALPHAS:
            m = lgb.LGBMRegressor(**LGB_BASE, objective="quantile", alpha=a)
            m.fit(Xtr, ytr)
            preds[f"q{a}"] = m.predict(Xte)

        oos_rows.append(pd.DataFrame(preds))
        test_year += STEP_YEARS

    if not oos_rows:
        return None
    return pd.concat(oos_rows, ignore_index=True)


def score_variant(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "dir_acc": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
        "extreme_hit_rate": float(extreme_hit_rate(y_true, y_pred)),
        "tail_pinball_0.1": float(tail_pinball_loss(y_true, y_pred, 0.1)),
        "tail_pinball_0.9": float(tail_pinball_loss(y_true, y_pred, 0.9)),
        "n": int(len(y_true)),
    }


CONFIGS = ["baseline", "cross_field", "term_structure", "macro_interaction"]
results = {"global_start": str(GLOBAL_START.date()), "horizons": {}}

for h in HORIZONS:
    print(f"\n=== Horizon {h}d ===")
    label_col = f"fwd_{h}"
    results["horizons"][str(h)] = {}
    for cfg_name in CONFIGS:
        print(f"  Config: {cfg_name}")
        d, feature_cols = build_config(cfg_name)
        pooled = run_walkforward(d, feature_cols, label_col, h)
        if pooled is None:
            print("    No valid folds, skipping.")
            continue

        cell = {}
        for name in ["l2", "lq_q4", "tail_weighted"]:
            cell[name] = score_variant(pooled["y_true"].values, pooled[name].values)
        cell["quantile_0.1"] = score_variant(pooled["y_true"].values, pooled["q0.1"].values)
        cell["quantile_0.1"]["coverage"] = float(np.mean(pooled["y_true"].values <= pooled["q0.1"].values))
        cell["quantile_0.9"] = score_variant(pooled["y_true"].values, pooled["q0.9"].values)
        cell["quantile_0.9"]["coverage"] = float(np.mean(pooled["y_true"].values <= pooled["q0.9"].values))

        quantile_preds = {a: pooled[f"q{a}"].values for a in DIST_ALPHAS}
        dist_cell = score_variant(pooled["y_true"].values, pooled["q0.5"].values)
        dist_cell["twcrps"] = float(crps_from_quantiles_weighted(pooled["y_true"].values, quantile_preds, DIST_ALPHAS))
        cell["distributional"] = dist_cell

        results["horizons"][str(h)][cfg_name] = cell
        print(f"    n={cell['l2']['n']} | l2 r2={cell['l2']['r2']:.3f} hit={cell['l2']['extreme_hit_rate']:.3f} | "
              f"lq_q4 r2={cell['lq_q4']['r2']:.3f} hit={cell['lq_q4']['extreme_hit_rate']:.3f} | "
              f"distributional r2={dist_cell['r2']:.3f} hit={dist_cell['extreme_hit_rate']:.3f} twcrps={dist_cell['twcrps']:.5f}")

out_path = os.path.join(OUT_DIR, "results_grid_wave1.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved results to {out_path}")
print("\nDone.")
