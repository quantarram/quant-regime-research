"""
Pooled, walk-forward REGRESSION model on top of the rolling intraday
multifractal feature panel (01b_rolling_features_intraday.py). One model per
horizon in {5,15,60} minutes, trained jointly across BTC/ETH/SOL/BNB.

Same regression-to-actual-price pivot and OLS -> XGBoost/LightGBM -> MLP
model progression as 02_train_predict_daily.py. This is the genuinely new
test of intraday predictability referenced in 01b's docstring -- higher-order
multifractal/structure-function features instead of the naive quantile-
exceedance approach that previously found nothing at these horizons
(files/cpe_engine_intraday_btc.py). No result is assumed going in.

Run: python 02b_train_predict_intraday.py
Requires: features_intraday_panel.parquet, ../data/intraday_*_1m.parquet
Output: results_intraday.json, oos_price_intraday_<SYMBOL>.png
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
DATA_DIR = os.path.join(REPO_DIR, "data")

SYMBOLS = {"BTC": "intraday_btc_1m.parquet", "ETH": "intraday_eth_1m.parquet",
           "SOL": "intraday_sol_1m.parquet", "BNB": "intraday_bnb_1m.parquet"}
HORIZONS = [5, 15, 60]
INITIAL_TRAIN_FRAC = 0.5
N_TEST_FOLDS = 6
LABEL_LEAK_SHIFT = 0

print("=" * 60)
print("  POOLED INTRADAY WALK-FORWARD REGRESSION MODEL")
print("=" * 60)

panel = pd.read_parquet(os.path.join(OUT_DIR, "features_intraday_panel.parquet"))
panel["ts"] = pd.to_datetime(panel["ts"])

price_series = {}
for label, fname in SYMBOLS.items():
    raw = pd.read_parquet(os.path.join(DATA_DIR, fname))
    raw["ts"] = pd.to_datetime(raw["open_time"], unit="ms")
    raw = raw.sort_values("ts").drop_duplicates("ts").set_index("ts")
    price_series[label] = raw["close"].astype(float)

ret_frames = []
for label, s in price_series.items():
    df = pd.DataFrame({"ts": s.index, "symbol": label, "price_now": s.values})
    for h in HORIZONS:
        fwd_price = s.shift(-h - LABEL_LEAK_SHIFT)
        df[f"fwd_{h}"] = np.log(fwd_price.values / s.values)
        df[f"price_fut_{h}"] = fwd_price.values
    ret_frames.append(df)
ret_df = pd.concat(ret_frames, ignore_index=True)
panel = panel.merge(ret_df, on=["symbol", "ts"], how="left")

FEATURE_COLS = ["alpha", "C1", "H", "xi_q2", "xi_q4",
                 "gap_tau5_q2", "gap_tau5_q4", "gap_tau15_q2", "gap_tau15_q4",
                 "gap_tau60_q2", "gap_tau60_q4"]
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

ts_min, ts_max = panel["ts"].min(), panel["ts"].max()
total_span = ts_max - ts_min
train_end = ts_min + total_span * INITIAL_TRAIN_FRAC
test_span = (ts_max - train_end) / N_TEST_FOLDS
print(f"Panel spans {ts_min} .. {ts_max}; initial train end: {train_end}")

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
    print(f"\n--- Horizon {h}min ---")
    label_col = f"fwd_{h}"
    d = panel.dropna(subset=FEATURE_COLS + [label_col, "price_now", f"price_fut_{h}"]).copy()

    fold_metrics = []
    fold_start = train_end
    for fold in range(N_TEST_FOLDS):
        test_start = fold_start
        test_end = fold_start + test_span
        test_mask = (d["ts"] >= test_start) & (d["ts"] < test_end)
        if test_mask.sum() < 30:
            fold_start = test_end
            continue

        purge_boundary = test_start - pd.Timedelta(minutes=h)
        train_mask = d["ts"] < purge_boundary
        if train_mask.sum() < 200:
            fold_start = test_end
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
            fold_metrics.append({"model": name, "fold": fold, "n": int(test_mask.sum()),
                                  "r2": r2, "rmse": rmse, "dir_acc": dir_acc})

        rec = d.loc[test_mask, ["symbol", "ts", label_col, "price_now", f"price_fut_{h}"]].copy()
        rec = rec.rename(columns={label_col: "actual_return", f"price_fut_{h}": "actual_price"})
        rec["horizon"] = h
        for name, pred in fold_preds.items():
            rec[f"pred_return_{name}"] = pred
            rec[f"pred_price_{name}"] = rec["price_now"].values * np.exp(pred)
        oos_records.append(rec)

        line = f"  fold {fold} ({test_start.date()}..{test_end.date()}): n={int(test_mask.sum())} | " + " | ".join(
            f"{m}: r2={r2_score(yte, fold_preds[m]):.3f} rmse={mean_squared_error(yte, fold_preds[m])**0.5:.5f} "
            f"dir={np.mean(np.sign(fold_preds[m])==np.sign(yte.values)):.3f}"
            for m in MODEL_BUILDERS
        )
        print(line)
        fold_start = test_end

    fm = pd.DataFrame(fold_metrics)
    summary = {}
    if len(fm):
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
        print(f"  Pooled ({h}min): " + " | ".join(
            f"{m}: r2={summary[m]['pooled_r2']:.4f} rmse={summary[m]['pooled_rmse']:.5f} "
            f"dir_acc={summary[m]['pooled_dir_acc']:.4f}" for m in MODEL_BUILDERS))
    else:
        print("  No valid folds.")
    results["horizons"][str(h)] = summary

oos_df = pd.concat(oos_records, ignore_index=True) if oos_records else pd.DataFrame()

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

out_path = os.path.join(OUT_DIR, "results_intraday.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved results to {out_path}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not oos_df.empty:
        best_h = HORIZONS[0]
        sub_h = oos_df[oos_df.horizon == best_h]
        for sym in SYMBOLS:
            sub = sub_h[sub_h.symbol == sym].sort_values("ts")
            if len(sub) < 30:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(sub["ts"], sub["actual_price"], label="Actual price", linewidth=1.5)
            for name in ["xgboost", "lightgbm", "ols"]:
                ax.plot(sub["ts"], sub[f"pred_price_{name}"], label=f"Predicted ({name})", alpha=0.7, linewidth=1)
            ax.set_title(f"{sym}: actual vs. predicted price, {best_h}-min-ahead OOS walk-forward")
            ax.legend()
            fig.tight_layout()
            fname = os.path.join(OUT_DIR, f"oos_price_intraday_{sym}.png")
            fig.savefig(fname, dpi=120)
            plt.close(fig)
            print(f"Saved {fname}")
except Exception as e:
    print(f"Plot failed (non-fatal): {e}")

print("\nDone.")
