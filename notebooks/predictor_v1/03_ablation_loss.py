"""
Phase A: loss-function ablation. Features and model family held FIXED (the
existing, already-validated daily 15-instrument panel; LightGBM, since it
natively supports custom/quantile objectives) -- only the training objective
changes between runs, so any difference in outcome is attributable to the
loss function alone.

Variants tested, all against the same walk-forward/purge folds already
leakage-audited in 02_train_predict_daily.py:
  l2                    -- control (reproduces a v1 number)
  lq_q3 / q4 / q5 / q6   -- custom L_q loss (loss_functions.make_lq_objective)
  tail_weighted          -- L2 with sample weights proportional to |actual return|
  quantile_0.5/0.75/0.9  -- native LightGBM pinball loss at those quantiles
  distributional         -- 5-quantile grid (0.1/0.25/0.5/0.75/0.9), evaluated
                             via CRPS (avg pinball across the grid) + calibration

Run: python 03_ablation_loss.py
Requires: features_daily_panel.parquet, ../multiasset_prices.parquet
Output: results_ablation_loss.json
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error

from loss_functions import make_lq_objective, tail_weights, pinball_loss, crps_from_quantiles, calibration_check

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

HORIZONS = [1, 5, 21]
INITIAL_TRAIN_YEARS = 10
STEP_YEARS = 1
DIST_ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]
LGB_BASE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)

print("=" * 60)
print("  PHASE A: LOSS FUNCTION ABLATION (features fixed)")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
panel = pd.read_parquet(os.path.join(OUT_DIR, "features_daily_panel.parquet"))
panel["date"] = pd.to_datetime(panel["date"])
TICKERS = sorted(panel["ticker"].unique())
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
panel = panel.merge(ret_df, on=["ticker", "date"], how="left")

z_cols = [c for c in panel.columns if c.endswith("_z")]
ctx_cols = [c for c in panel.columns if c.startswith("ctx_")]
for c in ctx_cols:
    panel[c] = np.sign(panel[c]) * np.log1p(np.abs(panel[c]))
FEATURE_COLS = z_cols + ctx_cols + ["self_ref_score"]
print(f"Feature columns ({len(FEATURE_COLS)}), unchanged from v1 control")

panel["year"] = panel["date"].dt.year
min_year, max_year = panel["year"].min(), panel["year"].max()
first_test_year = min_year + INITIAL_TRAIN_YEARS
print(f"Panel spans {min_year}-{max_year}; first OOS test year: {first_test_year}")

results = {"horizons": {}}

for h in HORIZONS:
    print(f"\n--- Horizon {h}d ---")
    label_col = f"fwd_{h}"
    d = panel.dropna(subset=FEATURE_COLS + [label_col]).copy()
    d["pos"] = d["date"].map(date_pos)

    fold_records = []
    test_year = first_test_year
    while test_year <= max_year:
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year + STEP_YEARS}-01-01")
        test_mask = (d["date"] >= test_start) & (d["date"] < test_end)
        if test_mask.sum() < 30:
            test_year += STEP_YEARS
            continue
        test_start_pos = date_pos.get(test_start, None)
        if test_start_pos is None:
            candidates = [p for dt, p in date_pos.items() if dt >= test_start]
            test_start_pos = min(candidates) if candidates else None
        train_mask = (d["date"] < test_start) & (d["pos"] + h < test_start_pos)
        if train_mask.sum() < 200:
            test_year += STEP_YEARS
            continue

        Xtr, ytr = d.loc[train_mask, FEATURE_COLS], d.loc[train_mask, label_col]
        Xte, yte = d.loc[test_mask, FEATURE_COLS], d.loc[test_mask, label_col]

        point_variants = {
            "l2": dict(objective="regression"),
            "lq_q3": dict(objective=make_lq_objective(3)),
            "lq_q4": dict(objective=make_lq_objective(4)),
            "lq_q5": dict(objective=make_lq_objective(5)),
            "lq_q6": dict(objective=make_lq_objective(6)),
        }
        for name, params in point_variants.items():
            m = lgb.LGBMRegressor(**LGB_BASE, **params)
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            fold_records.append(dict(variant=name, test_year=test_year, n=len(yte),
                                      r2=r2_score(yte, pred), rmse=mean_squared_error(yte, pred) ** 0.5,
                                      dir_acc=float(np.mean(np.sign(pred) == np.sign(yte.values)))))

        w = tail_weights(ytr.values)
        m = lgb.LGBMRegressor(**LGB_BASE, objective="regression")
        m.fit(Xtr, ytr, sample_weight=w)
        pred = m.predict(Xte)
        fold_records.append(dict(variant="tail_weighted", test_year=test_year, n=len(yte),
                                  r2=r2_score(yte, pred), rmse=mean_squared_error(yte, pred) ** 0.5,
                                  dir_acc=float(np.mean(np.sign(pred) == np.sign(yte.values)))))

        quantile_preds = {}
        for a in DIST_ALPHAS:
            m = lgb.LGBMRegressor(**LGB_BASE, objective="quantile", alpha=a)
            m.fit(Xtr, ytr)
            quantile_preds[a] = m.predict(Xte)

        for a in [0.5, 0.75, 0.9]:
            pred = quantile_preds[a]
            rec = dict(variant=f"quantile_{a}", test_year=test_year, n=len(yte),
                       pinball=pinball_loss(yte.values, pred, a),
                       coverage=float(np.mean(yte.values <= pred)))
            if a == 0.5:
                rec.update(r2=r2_score(yte, pred), rmse=mean_squared_error(yte, pred) ** 0.5,
                           dir_acc=float(np.mean(np.sign(pred) == np.sign(yte.values))))
            fold_records.append(rec)

        crps = crps_from_quantiles(yte.values, quantile_preds, DIST_ALPHAS)
        cal = calibration_check(yte.values, quantile_preds, DIST_ALPHAS)
        fold_records.append(dict(variant="distributional", test_year=test_year, n=len(yte),
                                  crps=crps, calibration=cal,
                                  r2=r2_score(yte, quantile_preds[0.5]),
                                  rmse=mean_squared_error(yte, quantile_preds[0.5]) ** 0.5,
                                  dir_acc=float(np.mean(np.sign(quantile_preds[0.5]) == np.sign(yte.values)))))

        print(f"  {test_year}: n={len(yte)} -- l2 r2={fold_records[-10]['r2']:.3f}, "
              f"lq_q4 r2={fold_records[-8]['r2']:.3f}, "
              f"quantile_0.5 r2={fold_records[-4]['r2']:.3f}, "
              f"distributional crps={crps:.5f}")
        test_year += STEP_YEARS

    fm = pd.DataFrame(fold_records)
    summary = {}
    for v in fm["variant"].unique():
        sub = fm[fm["variant"] == v]
        n_total = sub["n"].sum()
        s = {"n_folds": int(len(sub)), "n_obs": int(n_total)}
        if "r2" in sub.columns and sub["r2"].notna().any():
            s["pooled_r2"] = float((sub["r2"] * sub["n"]).sum() / n_total)
            s["pooled_rmse"] = float((sub["rmse"] * sub["n"]).sum() / n_total)
            s["pooled_dir_acc"] = float((sub["dir_acc"] * sub["n"]).sum() / n_total)
        if "pinball" in sub.columns and sub["pinball"].notna().any():
            s["pooled_pinball"] = float((sub["pinball"] * sub["n"]).sum() / n_total)
            s["pooled_coverage"] = float((sub["coverage"] * sub["n"]).sum() / n_total)
        if "crps" in sub.columns and sub["crps"].notna().any():
            s["pooled_crps"] = float((sub["crps"] * sub["n"]).sum() / n_total)
        summary[v] = s
    results["horizons"][str(h)] = summary
    print(f"\n--- Horizon {h}d pooled summary ---")
    for v, s in summary.items():
        print(f"  {v}: {s}")

results["feature_cols"] = FEATURE_COLS
out_path = os.path.join(OUT_DIR, "results_ablation_loss.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved results to {out_path}")
print("\nDone.")
