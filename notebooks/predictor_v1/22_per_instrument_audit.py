"""
Per-instrument bootstrap audit of the one finding that survived
21_grid_claims_audit.py: macro_interaction @ 21d, pooled hit_rate edge
[+0.013, +0.110] over baseline (CI excludes zero). Earlier diagnostic work
(mid-session) eyeballed per-instrument point estimates and suggested the
effect was concentrated in XLF/XLE and near-chance for BTC/EUR/GLD -- never
actually bootstrap-tested. Same audit discipline as everything else in this
self-check pass: don't trust a per-instrument point estimate without a CI.

Run: python 22_per_instrument_audit.py
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb

from loss_functions import extreme_hit_rate

REPO_DIR = "/Users/arrams/Documents/personal_work/quant/quant-regime-research"
OUT_DIR = "/Users/arrams/Documents/personal_work/quant/quant-regime-research/notebooks/predictor_v1"
HORIZON = 21
INITIAL_TRAIN_YEARS = 6
STEP_YEARS = 1
LGB_BASE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
GLOBAL_START = pd.Timestamp("2007-12-12")

print("=" * 60)
print("  PER-INSTRUMENT AUDIT: macro_interaction @ 21d")
print("=" * 60)

prices = pd.read_parquet(f"{REPO_DIR}/notebooks/multiasset_prices.parquet")
baseline = pd.read_parquet(f"{OUT_DIR}/features_daily_panel.parquet")
baseline["date"] = pd.to_datetime(baseline["date"])
TICKERS = sorted(baseline["ticker"].unique())
date_pos = {d: i for i, d in enumerate(prices.index)}

z_cols = [c for c in baseline.columns if c.endswith("_z")]
ctx_cols = [c for c in baseline.columns if c.startswith("ctx_")]
for c in ctx_cols:
    baseline[c] = np.sign(baseline[c]) * np.log1p(np.abs(baseline[c]))
BASELINE_FEATURE_COLS = z_cols + ctx_cols + ["self_ref_score"]

macro = pd.read_parquet(f"{OUT_DIR}/features_macro_interaction.parquet")
macro["date"] = pd.to_datetime(macro["date"])
macro_cols = [c for c in macro.columns if c not in ("ticker", "date")]

s_map = {t: prices[t] for t in TICKERS}
ret_frames = []
for t in TICKERS:
    s = s_map[t]
    df = pd.DataFrame({"date": s.index, "ticker": t})
    df["fwd_ret"] = np.log(s.shift(-HORIZON).values / s.values)
    ret_frames.append(df)
ret_df = pd.concat(ret_frames, ignore_index=True)
d = baseline.merge(ret_df, on=["ticker", "date"], how="left").merge(macro, on=["ticker", "date"], how="left")
feature_cols = BASELINE_FEATURE_COLS + macro_cols
d = d[d["date"] >= GLOBAL_START].copy()

dd = d.dropna(subset=feature_cols + ["fwd_ret"]).copy()
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
    train_mask = (dd["date"] < test_start) & (dd["pos"] + HORIZON < test_start_pos)
    if train_mask.sum() < 200:
        test_year += STEP_YEARS
        continue

    Xtr, ytr = dd.loc[train_mask, feature_cols], dd.loc[train_mask, "fwd_ret"]
    Xte, yte = dd.loc[test_mask, feature_cols], dd.loc[test_mask, "fwd_ret"]
    m = lgb.LGBMRegressor(**LGB_BASE, objective="regression")
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)

    rec = dd.loc[test_mask, ["ticker", "date"]].copy()
    rec["y_true"] = yte.values
    rec["y_pred"] = pred
    rec["year"] = test_year
    oos_rows.append(rec)
    test_year += STEP_YEARS

oos = pd.concat(oos_rows, ignore_index=True)
print(f"Total OOS rows: {len(oos)}, tickers: {sorted(oos['ticker'].unique())}")

print("\n--- Per-instrument hit_rate, block-bootstrap (by year) 95% CI ---")
rng_global = np.random.default_rng(0)
results = []
for tkr in sorted(oos["ticker"].unique()):
    sub = oos[oos["ticker"] == tkr]
    if len(sub) < 100:
        print(f"  {tkr}: n={len(sub)} too small, skipping")
        continue
    point = extreme_hit_rate(sub["y_true"].values, sub["y_pred"].values)
    years = sorted(sub["year"].unique())
    rng = np.random.default_rng(hash(tkr) % (2**32))
    boot = []
    for _ in range(1000):
        sampled = rng.choice(years, size=len(years), replace=True)
        b = pd.concat([sub[sub["year"] == y] for y in sampled], ignore_index=True)
        boot.append(extreme_hit_rate(b["y_true"].values, b["y_pred"].values))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    above_chance = lo > 0.10
    results.append((tkr, len(sub), point, lo, hi, above_chance))
    print(f"  {tkr}: n={len(sub)}, hit_rate={point:.3f}, 95% CI=[{lo:.3f}, {hi:.3f}] "
          f"{'-- excludes 0.10 chance floor' if above_chance else '-- CI includes chance floor, not distinguishable'}")

n_above = sum(1 for r in results if r[5])
print(f"\n{n_above} / {len(results)} instruments have a 95% CI that excludes the ~0.10 chance floor.")
