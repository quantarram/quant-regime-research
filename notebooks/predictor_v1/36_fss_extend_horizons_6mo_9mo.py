"""
Add 6-month and 9-month horizons (126 and 189 trading days, 21 trading
days/month, same convention as every other horizon in this pipeline) to
results_fss_true_unified.json. Identical methodology to 32_fss_true_unified.py
and 35_fss_extend_horizon_252.py -- merges into the existing results file
rather than recomputing 1/5/21/63/252d again.

Run: python 36_fss_extend_horizons_6mo_9mo.py
Output: results_fss_true_unified.json (merged in place)
"""
import pandas as pd
import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb

from loss_functions import fss_from_quantiles

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_HORIZONS = [126, 189]
INITIAL_TRAIN_YEARS = 6
STEP_YEARS = 1
ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]
LGB_BASE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
WINDOWS = [21, 63, 126, 252]
UPPER_THRESHOLDS = [0.05, 0.075, 0.10]
LOWER_THRESHOLDS = [-0.05, -0.075, -0.10]
MIN_OOS_ROWS = 300

t0 = time.time()
print("=" * 60)
print(f"  FSS EXTEND: adding {NEW_HORIZONS} horizons to existing results")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))
date_pos_main = {d: i for i, d in enumerate(prices.index)}
date_pos_proxy = {d: i for i, d in enumerate(prices_proxy.index)}

macro_existing = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro_existing["date"] = pd.to_datetime(macro_existing["date"])
credit_spread_regime = macro_existing[["date", "credit_spread_regime"]].drop_duplicates("date").set_index("date")["credit_spread_regime"]

vixy = prices["VIXY"].dropna()
vixm = prices["VIXM"].dropna()
common = vixy.index.intersection(vixm.index)
raw_ratio = np.log(vixm.reindex(common) / vixy.reindex(common))
vix_term_slope = ((raw_ratio - raw_ratio.rolling(200, min_periods=100).mean())
                   / raw_ratio.rolling(200, min_periods=100).std()).rename("vix_term_slope")
regimes = pd.concat([credit_spread_regime, vix_term_slope], axis=1).reset_index().rename(columns={"index": "date"})

orig_baseline = pd.read_parquet(os.path.join(OUT_DIR, "features_daily_panel.parquet"))
orig_baseline["date"] = pd.to_datetime(orig_baseline["date"])
ORIG_TICKERS = sorted(orig_baseline["ticker"].unique())

new_baseline_raw = pd.read_parquet(os.path.join(OUT_DIR, "features_new_tickers_baseline_cache.parquet"))
new_baseline_raw["date"] = pd.to_datetime(new_baseline_raw["date"])
NEW_TICKERS = sorted(new_baseline_raw["ticker"].unique())
zscore_cols = ["alpha", "C1", "H", "xi_q2", "xi_q4", "gap_tau1_q2", "gap_tau1_q4",
               "gap_tau5_q2", "gap_tau5_q4", "gap_tau21_q2", "gap_tau21_q4"]


def zscore_group(g):
    for c in zscore_cols:
        mu, sd = g[c].mean(), g[c].std()
        g[c + "_z"] = (g[c] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
    return g


new_baseline = new_baseline_raw.groupby("date", group_keys=False).apply(zscore_group)
ALL_TICKERS = ORIG_TICKERS + NEW_TICKERS
print(f"All {len(ALL_TICKERS)} tickers: {ALL_TICKERS}")

ticker_registry = {}
for tkr in ORIG_TICKERS:
    df = orig_baseline[orig_baseline["ticker"] == tkr].merge(regimes, on="date", how="left")
    baseline_cols = [c for c in orig_baseline.columns if c.endswith("_z")] + \
        [c for c in orig_baseline.columns if c.startswith("ctx_")] + ["self_ref_score"]
    ticker_registry[tkr] = {"df": df, "baseline_cols": baseline_cols, "series": prices[tkr].dropna(),
                             "date_pos": date_pos_main}
for tkr in NEW_TICKERS:
    df = new_baseline[new_baseline["ticker"] == tkr].merge(regimes, on="date", how="left")
    baseline_cols = [c for c in new_baseline.columns if c.endswith("_z")]
    is_proxy = tkr in ("IYR", "VOX")
    ticker_registry[tkr] = {
        "df": df, "baseline_cols": baseline_cols,
        "series": (prices_proxy[tkr] if is_proxy else prices[tkr]).dropna(),
        "date_pos": date_pos_proxy if is_proxy else date_pos_main,
    }

for tkr, reg in ticker_registry.items():
    df = reg["df"]
    df["interact_gap21q4_credit"] = df["gap_tau21_q4_z"] * df["credit_spread_regime"]
    df["interact_xiq4_credit"] = df["xi_q4_z"] * df["credit_spread_regime"]
    df["interact_gap21q4_vix"] = df["gap_tau21_q4_z"] * df["vix_term_slope"]
    df["interact_xiq4_vix"] = df["xi_q4_z"] * df["vix_term_slope"]

CREDIT_TERMS = ["interact_gap21q4_credit", "interact_xiq4_credit", "credit_spread_regime"]
VIX_TERMS = ["interact_gap21q4_vix", "interact_xiq4_vix", "vix_term_slope"]
VARIANT_EXTRA_COLS = {"climatology": [], "credit_only": CREDIT_TERMS, "vix_only": VIX_TERMS,
                       "both": CREDIT_TERMS + VIX_TERMS}


def run_instrument(tkr, horizon):
    reg = ticker_registry[tkr]
    series, date_pos = reg["series"], reg["date_pos"]
    fwd_ret = np.log(series.shift(-horizon) / series).rename("fwd_ret")
    d_full = reg["df"].merge(fwd_ret, left_on="date", right_index=True, how="left")
    d_full["pos"] = d_full["date"].map(date_pos)
    d_full = d_full.dropna(subset=["pos"]).copy()
    d_full["pos"] = d_full["pos"].astype(int)

    all_cols = reg["baseline_cols"] + CREDIT_TERMS + VIX_TERMS
    dd = d_full.dropna(subset=all_cols + ["fwd_ret"]).reset_index(drop=True)
    if len(dd) < 500:
        return None

    min_year, max_year = dd["date"].dt.year.min(), dd["date"].dt.year.max()
    first_test_year = min_year + INITIAL_TRAIN_YEARS
    first_test_start = pd.Timestamp(f"{first_test_year}-01-01")
    cands0 = [p for dt, p in date_pos.items() if dt >= first_test_start]
    if not cands0:
        return None
    first_test_start_pos = min(cands0)

    initial_train_mask = (dd["date"] < first_test_start) & (dd["pos"] + horizon < first_test_start_pos)
    if initial_train_mask.sum() < 150:
        return None

    oos_by_variant = {}
    for variant_name, extra_cols in VARIANT_EXTRA_COLS.items():
        feature_cols = reg["baseline_cols"] + extra_cols
        if variant_name == "climatology":
            initial_train = dd.loc[initial_train_mask, ["date", "fwd_ret"]].copy()
            initial_train["mmdd"] = list(zip(initial_train["date"].dt.month, initial_train["date"].dt.day))
            pooled_vals = initial_train["fwd_ret"].values
            climatology_quantiles_by_day = {}
            for m in range(1, 13):
                days_in_month = 29 if m == 2 else (30 if m in (4, 6, 9, 11) else 31)
                for da in range(1, days_in_month + 1):
                    vals = initial_train.loc[initial_train["mmdd"] == (m, da), "fwd_ret"].values
                    if len(vals) == 0:
                        vals = pooled_vals
                    climatology_quantiles_by_day[(m, da)] = {a: float(np.quantile(vals, a)) for a in ALPHAS}

        oos_rows = []
        test_year = first_test_year
        while test_year <= max_year:
            test_start = pd.Timestamp(f"{test_year}-01-01")
            test_end = pd.Timestamp(f"{test_year + STEP_YEARS}-01-01")
            test_mask = (dd["date"] >= test_start) & (dd["date"] < test_end)
            if test_mask.sum() < 20:
                test_year += STEP_YEARS
                continue
            cands = [p for dt, p in date_pos.items() if dt >= test_start]
            if not cands:
                test_year += STEP_YEARS
                continue
            test_start_pos = min(cands)
            train_mask = (dd["date"] < test_start) & (dd["pos"] + horizon < test_start_pos)
            if train_mask.sum() < 200:
                test_year += STEP_YEARS
                continue

            ytr = dd.loc[train_mask, "fwd_ret"]
            yte = dd.loc[test_mask, "fwd_ret"]
            row = dd.loc[test_mask, ["date"]].copy()
            row["y_true"] = yte.values

            if variant_name == "climatology":
                row_mmdd = list(zip(row["date"].dt.month, row["date"].dt.day))
                for a in ALPHAS:
                    row[f"q{a}"] = [climatology_quantiles_by_day[k][a] for k in row_mmdd]
            else:
                Xtr, Xte = dd.loc[train_mask, feature_cols], dd.loc[test_mask, feature_cols]
                for a in ALPHAS:
                    m = lgb.LGBMRegressor(**LGB_BASE, objective="quantile", alpha=a)
                    m.fit(Xtr, ytr)
                    row[f"q{a}"] = m.predict(Xte)
            oos_rows.append(row)
            test_year += STEP_YEARS

        if not oos_rows:
            return None
        oos_by_variant[variant_name] = pd.concat(oos_rows, ignore_index=True).sort_values("date").reset_index(drop=True)

    if any(len(oos_by_variant[v]) < MIN_OOS_ROWS for v in VARIANT_EXTRA_COLS):
        return None

    grid = {}
    for variant_name, oos in oos_by_variant.items():
        y_true = oos["y_true"].values
        quantile_preds = {a: oos[f"q{a}"].values for a in ALPHAS}
        grid[variant_name] = {}
        for w in WINDOWS:
            grid[variant_name][w] = {"above": {}, "below": {}}
            for thr in UPPER_THRESHOLDS:
                grid[variant_name][w]["above"][thr] = fss_from_quantiles(
                    y_true, quantile_preds, ALPHAS, thr, direction="above", window=w)
            for thr in LOWER_THRESHOLDS:
                grid[variant_name][w]["below"][thr] = fss_from_quantiles(
                    y_true, quantile_preds, ALPHAS, thr, direction="below", window=w)

    n_oos = {v: int(len(oos_by_variant[v])) for v in VARIANT_EXTRA_COLS}
    return {"grid": grid, "n_oos": n_oos}


def grid_avg(res, variant):
    vals = [v for w in WINDOWS for d in ("above", "below") for v in res["grid"][variant][w][d].values()
            if np.isfinite(v)]
    return float(np.mean(vals))


results_path = os.path.join(OUT_DIR, "results_fss_true_unified.json")
existing = json.load(open(results_path))
existing_results = existing["results"]
for nh in NEW_HORIZONS:
    if nh not in existing["horizons"]:
        existing["horizons"].append(nh)
existing["horizons"].sort()

n_added = 0
for horizon in NEW_HORIZONS:
    for tkr in ALL_TICKERS:
        res = run_instrument(tkr, horizon)
        if res is None:
            print(f"  {tkr} @ {horizon}d: skipped (insufficient data)  [{time.time()-t0:.0f}s]")
            continue
        existing_results.setdefault(tkr, {})
        existing_results[tkr][str(horizon)] = res
        n_added += 1
        avgs = {v: grid_avg(res, v) for v in VARIANT_EXTRA_COLS}
        print(f"  {tkr} @ {horizon}d: n={res['n_oos']['both']}, clim={avgs['climatology']:.4f}, "
              f"credit_only={avgs['credit_only']:.4f} ({avgs['credit_only']-avgs['climatology']:+.4f}), "
              f"vix_only={avgs['vix_only']:.4f} ({avgs['vix_only']-avgs['climatology']:+.4f}), "
              f"both={avgs['both']:.4f} ({avgs['both']-avgs['climatology']:+.4f})  [{time.time()-t0:.0f}s]")

existing["results"] = existing_results
with open(results_path, "w") as f:
    json.dump(existing, f, indent=2, default=float)
print(f"\nAdded {NEW_HORIZONS} horizons, {n_added} (ticker,horizon) combinations total, merged into {results_path}")
print(f"Total time: {time.time()-t0:.1f}s")
print("\nDone.")
