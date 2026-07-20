"""
Fixes a real selection-bias problem: "best config per instrument" was
chosen by scanning the ENTIRE walk-forward OOS period across 1000+
(horizon x variant x window x threshold x direction) combinations, then
that SAME period's performance was reported as if it were a fair test.
That's data snooping -- picking the best of many candidates on a period and
then reporting that period's result for the winner guarantees an inflated
number, independent of whether real skill exists.

Fix: a genuine chronological three-way split, real data only, no
randomization (per standing methodology in this project):
  - training: unchanged, walk-forward already trains strictly before each
    test fold
  - SELECTION period: OOS dates < HOLDOUT_START -- used ONLY to pick, per
    instrument, which (horizon, variant) has the highest skill above
    climatology
  - HOLDOUT period: OOS dates >= HOLDOUT_START -- NEVER touched during
    selection, used exactly once to report the selected configuration's
    real, honest performance

HOLDOUT_START is a single fixed calendar date applied identically to every
instrument (not tuned per instrument, which would just relocate the same
bias).

Refits every (ticker, horizon, variant) combination from scratch (24 x 7 x 3
= 504) because only the aggregated FSS grid was saved before, not the raw
per-row OOS predictions needed to score the two periods separately.

Run: python 38_fss_selection_holdout_split.py
Output: oos_predictions_all.parquet, results_selection_holdout.json
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
HORIZONS = [1, 5, 21, 63, 126, 189, 252]
HOLDOUT_START = pd.Timestamp("2022-01-01")
INITIAL_TRAIN_YEARS = 6
STEP_YEARS = 1
ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]
LGB_BASE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
WINDOWS = [21, 63, 126, 252]
UPPER_THRESHOLDS = [0.05, 0.075, 0.10]
LOWER_THRESHOLDS = [-0.05, -0.075, -0.10]
MIN_OOS_ROWS_TOTAL = 300
MIN_OOS_ROWS_PER_PERIOD = 100  # each side of the split must itself be usable

t0 = time.time()
print("=" * 60)
print("  FSS SELECTION/HOLDOUT SPLIT: fixing data-snooping in best-config choice")
print(f"  Selection period: OOS dates < {HOLDOUT_START.date()}")
print(f"  Holdout period:   OOS dates >= {HOLDOUT_START.date()} (never used in selection)")
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
VARIANT_EXTRA_COLS = {"credit_only": CREDIT_TERMS, "vix_only": VIX_TERMS, "both": CREDIT_TERMS + VIX_TERMS}
ALL_TICKERS = ORIG_TICKERS + NEW_TICKERS


def run_variant(tkr, horizon, variant):
    """Walk-forward LightGBM quantile fit for one (ticker, horizon, variant),
    PLUS climatology's day-of-year quantiles computed on the identical rows
    (fixed from the initial pre-sample reference period, as established
    throughout this project). Returns the raw per-row OOS dataframe."""
    reg = ticker_registry[tkr]
    series, date_pos = reg["series"], reg["date_pos"]
    fwd_ret = np.log(series.shift(-horizon) / series).rename("fwd_ret")
    d_full = reg["df"].merge(fwd_ret, left_on="date", right_index=True, how="left")
    d_full["pos"] = d_full["date"].map(date_pos)
    d_full = d_full.dropna(subset=["pos"]).copy()
    d_full["pos"] = d_full["pos"].astype(int)

    feature_cols = reg["baseline_cols"] + VARIANT_EXTRA_COLS[variant]
    dd = d_full.dropna(subset=feature_cols + ["fwd_ret"]).reset_index(drop=True)
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

        Xtr, ytr = dd.loc[train_mask, feature_cols], dd.loc[train_mask, "fwd_ret"]
        Xte, yte = dd.loc[test_mask, feature_cols], dd.loc[test_mask, "fwd_ret"]
        row = dd.loc[test_mask, ["date"]].copy()
        row["y_true"] = yte.values
        for a in ALPHAS:
            m = lgb.LGBMRegressor(**LGB_BASE, objective="quantile", alpha=a)
            m.fit(Xtr, ytr)
            row[f"q{a}"] = m.predict(Xte)
        row_mmdd = list(zip(row["date"].dt.month, row["date"].dt.day))
        for a in ALPHAS:
            row[f"clim_q{a}"] = [climatology_quantiles_by_day[k][a] for k in row_mmdd]
        oos_rows.append(row)
        test_year += STEP_YEARS

    if not oos_rows:
        return None
    oos = pd.concat(oos_rows, ignore_index=True).sort_values("date").reset_index(drop=True)
    oos["ticker"] = tkr
    oos["horizon"] = horizon
    oos["variant"] = variant
    return oos


def fss_grid_for_slice(sub, quantile_cols_prefix):
    """FSS grid (window x direction x threshold) for one dataframe slice."""
    y_true = sub["y_true"].values
    quantile_preds = {a: sub[f"{quantile_cols_prefix}{a}"].values for a in ALPHAS}
    grid = {}
    for w in WINDOWS:
        grid[w] = {"above": {}, "below": {}}
        for thr in UPPER_THRESHOLDS:
            grid[w]["above"][thr] = fss_from_quantiles(y_true, quantile_preds, ALPHAS, thr, direction="above", window=w)
        for thr in LOWER_THRESHOLDS:
            grid[w]["below"][thr] = fss_from_quantiles(y_true, quantile_preds, ALPHAS, thr, direction="below", window=w)
    return grid


def grid_avg(grid):
    vals = [v for w in WINDOWS for d in ("above", "below") for v in grid[w][d].values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


all_oos_frames = []
skill_records = []  # one row per (ticker, horizon, variant): selection_skill, holdout_skill, n_selection, n_holdout

for tkr in ALL_TICKERS:
    for horizon in HORIZONS:
        for variant in VARIANT_EXTRA_COLS:
            oos = run_variant(tkr, horizon, variant)
            if oos is None or len(oos) < MIN_OOS_ROWS_TOTAL:
                print(f"  {tkr} {variant} @ {horizon}d: skipped (insufficient data)  [{time.time()-t0:.0f}s]")
                continue

            sel = oos[oos["date"] < HOLDOUT_START]
            hold = oos[oos["date"] >= HOLDOUT_START]
            if len(sel) < MIN_OOS_ROWS_PER_PERIOD or len(hold) < MIN_OOS_ROWS_PER_PERIOD:
                print(f"  {tkr} {variant} @ {horizon}d: skipped (n_sel={len(sel)}, n_hold={len(hold)}, "
                      f"one side too thin)  [{time.time()-t0:.0f}s]")
                continue

            sel_grid_model = fss_grid_for_slice(sel, "q")
            sel_grid_clim = fss_grid_for_slice(sel, "clim_q")
            hold_grid_model = fss_grid_for_slice(hold, "q")
            hold_grid_clim = fss_grid_for_slice(hold, "clim_q")

            sel_skill = grid_avg(sel_grid_model) - grid_avg(sel_grid_clim)
            hold_skill = grid_avg(hold_grid_model) - grid_avg(hold_grid_clim)

            skill_records.append({
                "ticker": tkr, "horizon": horizon, "variant": variant,
                "n_selection": int(len(sel)), "n_holdout": int(len(hold)),
                "selection_skill": sel_skill, "holdout_skill": hold_skill,
            })
            all_oos_frames.append(oos)
            print(f"  {tkr} {variant} @ {horizon}d: n_sel={len(sel)}, n_hold={len(hold)}, "
                  f"selection_skill={sel_skill:+.4f}, holdout_skill={hold_skill:+.4f}  [{time.time()-t0:.0f}s]")

print(f"\nSaving raw OOS predictions ({sum(len(f) for f in all_oos_frames)} rows total)...")
all_oos = pd.concat(all_oos_frames, ignore_index=True)
all_oos.to_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))

skill_df = pd.DataFrame(skill_records)
skill_df.to_csv(os.path.join(OUT_DIR, "selection_holdout_skill.csv"), index=False)

# Best config per instrument, chosen using SELECTION-period skill ONLY --
# then report that SAME config's HOLDOUT-period skill as the honest result.
best_config = {}
for tkr in sorted(skill_df["ticker"].unique()):
    sub = skill_df[skill_df["ticker"] == tkr]
    winner = sub.loc[sub["selection_skill"].idxmax()]
    best_config[tkr] = {
        "horizon": int(winner["horizon"]), "variant": winner["variant"],
        "selection_skill": float(winner["selection_skill"]),
        "holdout_skill": float(winner["holdout_skill"]),
        "n_selection": int(winner["n_selection"]), "n_holdout": int(winner["n_holdout"]),
    }

with open(os.path.join(OUT_DIR, "best_config_selection_holdout.json"), "w") as f:
    json.dump(best_config, f, indent=2, default=float)

print("\n=== Best config per instrument (chosen on SELECTION period, reported on HOLDOUT period) ===")
for tkr in sorted(best_config, key=lambda t: -best_config[t]["holdout_skill"]):
    c = best_config[tkr]
    gap = c["holdout_skill"] - c["selection_skill"]
    print(f"  {tkr}: {c['variant']} @ {c['horizon']}d -- selection_skill={c['selection_skill']:+.4f}, "
          f"HOLDOUT_skill={c['holdout_skill']:+.4f} (gap={gap:+.4f})")

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("\nDone.")
