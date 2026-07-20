"""
Pooled, walk-forward REGRESSION model on top of the rolling multifractal
feature panel (01_rolling_features.py). One model per horizon in {1,5,21}
trading days, trained jointly across all 15 instruments in the panel
(cross-sectional pooling).

Predicts actual forward log-return (continuous), converted back to an actual
predicted price (current_price * exp(pred_return)) -- not just a direction
signal -- per the user's explicit requirement that this model output real
price forecasts, the same way their AI/NWP-based rainfall models output an
actual rainfall amount rather than a rain/no-rain category.

Model progression matches the user's own established rainfall-prediction
workflow: OLS multiple regression first (on the same physically-meaningful
feature set -- structural/dynamical multifractal features, not raw price/
technical-indicator dumps), then gradient boosting (XGBoost + LightGBM), then
an MLP neural net, to see whether the same "simple regression on the right
features already works, boosting/NN add a bit more" pattern holds here.

Run: python 02_train_predict_daily.py
Requires: features_daily_panel.parquet (01_rolling_features.py), ../multiasset_prices.parquet
Output: results_daily.json, oos_price_daily_<TICKER>.png
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

HORIZONS = [1, 5, 21]
INITIAL_TRAIN_YEARS = 10
STEP_YEARS = 1
LABEL_LEAK_SHIFT = int(os.environ.get("LABEL_LEAK_SHIFT", 0))   # set to 1 by the synthetic leakage test
PLOT_TICKERS = ["SPY", "GLD", "BTC-USD"]

print("=" * 60)
print("  POOLED DAILY WALK-FORWARD REGRESSION MODEL")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
panel = pd.read_parquet(os.path.join(OUT_DIR, "features_daily_panel.parquet"))
panel["date"] = pd.to_datetime(panel["date"])

TICKERS = sorted(panel["ticker"].unique())
date_pos = {d: i for i, d in enumerate(prices.index)}

# ── Forward-return labels (regression target) + current/future price for ──
# ── converting back to an actual predicted price ────────────────────────
ret_frames = []
for t in TICKERS:
    s = prices[t]
    df = pd.DataFrame({"date": s.index, "ticker": t, "price_now": s.values})
    for h in HORIZONS:
        fwd_price = s.shift(-h - LABEL_LEAK_SHIFT)
        df[f"fwd_{h}"] = np.log(fwd_price.values / s.values)
        df[f"price_fut_{h}"] = fwd_price.values
    ret_frames.append(df)
ret_df = pd.concat(ret_frames, ignore_index=True)
panel = panel.merge(ret_df, on=["ticker", "date"], how="left")

z_cols = [c for c in panel.columns if c.endswith("_z")]
ctx_cols = [c for c in panel.columns if c.startswith("ctx_")]
for c in ctx_cols:
    panel[c] = np.sign(panel[c]) * np.log1p(np.abs(panel[c]))
FEATURE_COLS = z_cols + ctx_cols + ["self_ref_score"]
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

panel["year"] = panel["date"].dt.year
min_year, max_year = panel["year"].min(), panel["year"].max()
first_test_year = min_year + INITIAL_TRAIN_YEARS
print(f"Panel spans {min_year}-{max_year}; first OOS test year: {first_test_year}")

MODEL_BUILDERS = {
    "ols": lambda: LinearRegression(),
    "xgboost": lambda: xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8,
                                         random_state=0, n_jobs=-1, verbosity=0),
    "lightgbm": lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8,
                                           random_state=0, verbosity=-1),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, max_depth=6,
                                                     min_samples_leaf=20, max_features="sqrt",
                                                     random_state=0, n_jobs=-1),
    "mlp": lambda: MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-3,
                                 max_iter=500, early_stopping=True, random_state=0),
}
NEEDS_SCALING = {"ols", "mlp"}

results = {"horizons": {}}
oos_records = []

for h in HORIZONS:
    print(f"\n--- Horizon {h}d ---")
    label_col = f"fwd_{h}"
    d = panel.dropna(subset=FEATURE_COLS + [label_col, "price_now", f"price_fut_{h}"]).copy()
    d["pos"] = d["date"].map(date_pos)

    fold_metrics = []
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

        # purge: drop training rows whose forward-return window (pos+h) crosses
        # into the test period -- leakage prevention at the split boundary only,
        # NOT an effective-sample-size / autocorrelation correction.
        train_mask = (d["date"] < test_start) & (d["pos"] + h < test_start_pos)
        if train_mask.sum() < 200:
            test_year += STEP_YEARS
            continue

        Xtr, ytr = d.loc[train_mask, FEATURE_COLS], d.loc[train_mask, label_col]
        Xte, yte = d.loc[test_mask, FEATURE_COLS], d.loc[test_mask, label_col]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        fold_preds = {}
        for name, builder in MODEL_BUILDERS.items():
            model = builder()
            if name in NEEDS_SCALING:
                model.fit(Xtr_s, ytr)
                pred = model.predict(Xte_s)
            else:
                model.fit(Xtr, ytr)
                pred = model.predict(Xte)
            fold_preds[name] = pred

            r2 = r2_score(yte, pred)
            rmse = mean_squared_error(yte, pred) ** 0.5
            dir_acc = float(np.mean(np.sign(pred) == np.sign(yte.values)))
            fold_metrics.append({"model": name, "test_year": test_year, "n": int(test_mask.sum()),
                                  "r2": r2, "rmse": rmse, "dir_acc": dir_acc})

        rec = d.loc[test_mask, ["ticker", "date", label_col, "price_now", f"price_fut_{h}"]].copy()
        rec = rec.rename(columns={label_col: "actual_return", f"price_fut_{h}": "actual_price"})
        rec["horizon"] = h
        for name, pred in fold_preds.items():
            rec[f"pred_return_{name}"] = pred
            rec[f"pred_price_{name}"] = rec["price_now"].values * np.exp(pred)
        oos_records.append(rec)

        line = f"  {test_year}: n={int(test_mask.sum())} | " + " | ".join(
            f"{m}: r2={r2_score(yte, fold_preds[m]):.3f} rmse={mean_squared_error(yte, fold_preds[m])**0.5:.4f} "
            f"dir={np.mean(np.sign(fold_preds[m])==np.sign(yte.values)):.3f}"
            for m in MODEL_BUILDERS
        )
        print(line)
        test_year += STEP_YEARS

    fm = pd.DataFrame(fold_metrics)
    summary = {}
    for name in MODEL_BUILDERS:
        sub = fm[fm.model == name]
        n_total = sub["n"].sum()
        summary[name] = {
            "pooled_r2": float((sub["r2"] * sub["n"]).sum() / n_total) if n_total else np.nan,
            "pooled_rmse": float((sub["rmse"] * sub["n"]).sum() / n_total) if n_total else np.nan,
            "pooled_dir_acc": float((sub["dir_acc"] * sub["n"]).sum() / n_total) if n_total else np.nan,
            "n_folds": int(len(sub)),
            "n_obs": int(n_total),
        }
    results["horizons"][str(h)] = summary
    print(f"  Pooled ({h}d): " + " | ".join(
        f"{m}: r2={summary[m]['pooled_r2']:.4f} rmse={summary[m]['pooled_rmse']:.4f} "
        f"dir_acc={summary[m]['pooled_dir_acc']:.4f}" for m in MODEL_BUILDERS))

oos_df = pd.concat(oos_records, ignore_index=True) if oos_records else pd.DataFrame()

# ── Price-level accuracy (MAPE) per horizon/model, computed from the same OOS predictions ──
price_level_summary = {}
if not oos_df.empty:
    for h in HORIZONS:
        sub = oos_df[oos_df.horizon == h]
        if sub.empty:
            continue
        price_level_summary[str(h)] = {}
        for name in MODEL_BUILDERS:
            mape = float(np.mean(np.abs((sub[f"pred_price_{name}"] - sub["actual_price"]) / sub["actual_price"])) * 100)
            price_level_summary[str(h)][name] = {"mape_pct": mape}
results["price_level_mape"] = price_level_summary

results["feature_cols"] = FEATURE_COLS
results["label_leak_shift"] = LABEL_LEAK_SHIFT

out_path = os.path.join(OUT_DIR, "results_daily.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved results to {out_path}")

# ── Predicted-vs-actual price plots for representative instruments, best model ──
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not oos_df.empty:
        best_h = 21 if 21 in HORIZONS else HORIZONS[-1]
        sub_h = oos_df[oos_df.horizon == best_h]
        for t in PLOT_TICKERS:
            sub = sub_h[sub_h.ticker == t].sort_values("date")
            if len(sub) < 30:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(sub["date"], sub["actual_price"], label="Actual price", linewidth=1.5)
            for name in ["xgboost", "lightgbm", "ols"]:
                ax.plot(sub["date"], sub[f"pred_price_{name}"], label=f"Predicted ({name})", alpha=0.7, linewidth=1)
            ax.set_title(f"{t}: actual vs. predicted price, {best_h}-day-ahead OOS walk-forward")
            ax.legend()
            fig.tight_layout()
            fname = os.path.join(OUT_DIR, f"oos_price_daily_{t.replace('=','').replace('^','')}.png")
            fig.savefig(fname, dpi=120)
            plt.close(fig)
            print(f"Saved {fname}")
except Exception as e:
    print(f"Plot failed (non-fatal): {e}")

print("\nDone.")
