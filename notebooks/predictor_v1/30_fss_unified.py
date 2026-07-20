"""
ONE consistent FSS methodology applied identically to every instrument,
replacing the two divergent pipelines used earlier (26_fss_rescore.py for
SPY only: QuantReg on credit_spread_regime/vix_term_slope; 27_*.py for the
other 24: LightGBM on each instrument's own multifractal features plus a
DIFFERENT macro pair, with an unclearly-named "baseline" tier). User
correctly flagged this as inconsistent -- different color schemes, different
model sets, no single shared methodology, four-panel-per-instrument plots
that didn't match SPY's single-panel one.

This script is exactly 26_fss_rescore.py's method (QuantReg walk-forward,
day-of-year climatology held fixed, same FSS window/threshold grid),
generalized to loop over all 25 instruments. Same four variants everywhere,
same colors, same everything:
  climatology  -- no-skill, day-of-year reference, no macro info
  credit_only  -- credit_spread_regime alone
  vix_only     -- vix_term_slope alone
  both         -- both combined
credit_spread_regime and vix_term_slope are used for every instrument
because they are the two CAUSALLY VALIDATED predictors from this program's
earlier Granger/CCF work (see causal_validation_results.json) -- not
uup_trend_regime, which was an earlier, never-causally-tested ad hoc feature
from before that validation phase existed. Retiring it here removes the
second source of inconsistency, not just the color scheme.

No instrument-specific multifractal features are used -- this deliberately
tests whether the market-wide credit/vix regime alone predicts EACH
instrument's own future return distribution, a clean, uniform, well-posed
question, applied the same way to all 25.

Run: python 30_fss_unified.py
Output: results_fss_unified.json
"""
import pandas as pd
import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from loss_functions import fss_from_quantiles

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
HORIZON = 21
INITIAL_TRAIN_YEARS = 6
STEP_YEARS = 1
ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]
WINDOWS = [21, 63, 126, 252]
UPPER_THRESHOLDS = [0.05, 0.075, 0.10]
LOWER_THRESHOLDS = [-0.05, -0.075, -0.10]
VARIANTS = {"climatology": [], "credit_only": ["credit_spread_regime"],
            "vix_only": ["vix_term_slope"], "both": ["credit_spread_regime", "vix_term_slope"]}
MIN_OOS_ROWS = 300

t0 = time.time()
print("=" * 60)
print("  FSS UNIFIED: credit_only / vix_only / both / climatology, ALL instruments")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))
date_pos_main = {d: i for i, d in enumerate(prices.index)}
date_pos_proxy = {d: i for i, d in enumerate(prices_proxy.index)}

# ── market-wide macro predictors, identical construction to 26_fss_rescore.py ──
macro = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro["date"] = pd.to_datetime(macro["date"])
credit_spread_regime = macro[["date", "credit_spread_regime"]].drop_duplicates("date").set_index("date")["credit_spread_regime"]

vixy = prices["VIXY"].dropna()
vixm = prices["VIXM"].dropna()
common = vixy.index.intersection(vixm.index)
raw_ratio = np.log(vixm.reindex(common) / vixy.reindex(common))
vix_term_slope = ((raw_ratio - raw_ratio.rolling(200, min_periods=100).mean())
                   / raw_ratio.rolling(200, min_periods=100).std()).rename("vix_term_slope")

MAIN_TICKERS = ["AAPL", "BTC-USD", "EURUSD=X", "GLD", "IWM", "JPM", "MSFT", "QQQ", "SPY", "TLT",
                 "XLE", "XLF", "XLK", "XOM", "^VIX", "XLI", "XLB", "XLY", "XLP", "XLU", "XLV",
                 "XLC", "XLRE", "DIA", "VTI"]
PROXY_TICKERS = ["IYR", "VOX"]
ALL_TICKERS = MAIN_TICKERS + PROXY_TICKERS


def run_instrument(ticker):
    series = (prices_proxy[ticker] if ticker in PROXY_TICKERS else prices[ticker]).dropna()
    date_pos = date_pos_proxy if ticker in PROXY_TICKERS else date_pos_main
    fwd_ret = np.log(series.shift(-HORIZON) / series).rename("fwd_ret")

    d_full = pd.concat([fwd_ret, credit_spread_regime, vix_term_slope], axis=1).dropna()
    d_full["date"] = d_full.index
    d_full["pos"] = d_full["date"].map(date_pos)
    d_full = d_full.dropna(subset=["pos"]).reset_index(drop=True)
    d_full["pos"] = d_full["pos"].astype(int)
    if len(d_full) < 500:
        return None

    min_year, max_year = d_full["date"].dt.year.min(), d_full["date"].dt.year.max()
    first_test_year = min_year + INITIAL_TRAIN_YEARS
    first_test_start = pd.Timestamp(f"{first_test_year}-01-01")
    cands0 = [p for dt, p in date_pos.items() if dt >= first_test_start]
    if not cands0:
        return None
    first_test_start_pos = min(cands0)
    initial_train_mask = (d_full["date"] < first_test_start) & (d_full["pos"] + HORIZON < first_test_start_pos)
    if initial_train_mask.sum() < 150:
        return None

    initial_train = d_full.loc[initial_train_mask, ["date", "fwd_ret"]].copy()
    initial_train["mmdd"] = list(zip(initial_train["date"].dt.month, initial_train["date"].dt.day))
    pooled_vals = initial_train["fwd_ret"].values
    climatology_quantiles_by_day = {}
    for m in range(1, 13):
        days_in_month = 29 if m == 2 else (30 if m in (4, 6, 9, 11) else 31)
        for dd in range(1, days_in_month + 1):
            vals = initial_train.loc[initial_train["mmdd"] == (m, dd), "fwd_ret"].values
            if len(vals) == 0:
                vals = pooled_vals
            climatology_quantiles_by_day[(m, dd)] = {a: float(np.quantile(vals, a)) for a in ALPHAS}

    max_year_local = d_full["date"].dt.year.max()
    oos_by_variant = {}
    for variant_name, feature_cols in VARIANTS.items():
        oos_rows = []
        test_year = first_test_year
        while test_year <= max_year_local:
            test_start = pd.Timestamp(f"{test_year}-01-01")
            test_end = pd.Timestamp(f"{test_year + STEP_YEARS}-01-01")
            test_mask = (d_full["date"] >= test_start) & (d_full["date"] < test_end)
            if test_mask.sum() < 20:
                test_year += STEP_YEARS
                continue
            cands = [p for dt, p in date_pos.items() if dt >= test_start]
            if not cands:
                test_year += STEP_YEARS
                continue
            test_start_pos = min(cands)
            train_mask = (d_full["date"] < test_start) & (d_full["pos"] + HORIZON < test_start_pos)
            if train_mask.sum() < 200:
                test_year += STEP_YEARS
                continue

            ytr = d_full.loc[train_mask, "fwd_ret"]
            yte = d_full.loc[test_mask, "fwd_ret"]
            row = d_full.loc[test_mask, ["date"]].copy()
            row["y_true"] = yte.values

            if variant_name == "climatology":
                row_mmdd = list(zip(row["date"].dt.month, row["date"].dt.day))
                for a in ALPHAS:
                    row[f"q{a}"] = [climatology_quantiles_by_day[k][a] for k in row_mmdd]
            else:
                Xtr, Xte = d_full.loc[train_mask, feature_cols], d_full.loc[test_mask, feature_cols]
                Xtr_sm = sm.add_constant(Xtr)
                Xte_sm = sm.add_constant(Xte, has_constant="add")
                for a in ALPHAS:
                    qr_model = QuantReg(ytr, Xtr_sm).fit(q=a)
                    row[f"q{a}"] = qr_model.predict(Xte_sm).values
            oos_rows.append(row)
            test_year += STEP_YEARS

        if not oos_rows:
            return None
        oos_by_variant[variant_name] = pd.concat(oos_rows, ignore_index=True).sort_values("date").reset_index(drop=True)

    if any(len(oos_by_variant[v]) < MIN_OOS_ROWS for v in VARIANTS):
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

    n_oos = {v: int(len(oos_by_variant[v])) for v in VARIANTS}
    return {"grid": grid, "n_oos": n_oos}


all_results = {}
for tkr in ALL_TICKERS:
    res = run_instrument(tkr)
    if res is None:
        print(f"  {tkr}: skipped (insufficient data)")
        continue
    all_results[tkr] = res

    def grid_avg(variant):
        vals = [v for w in WINDOWS for d in ("above", "below") for v in res["grid"][variant][w][d].values()
                if np.isfinite(v)]
        return float(np.mean(vals))

    avgs = {v: grid_avg(v) for v in VARIANTS}
    print(f"  {tkr}: n={res['n_oos']['both']}, clim={avgs['climatology']:.4f}, "
          f"credit_only={avgs['credit_only']:.4f} ({avgs['credit_only']-avgs['climatology']:+.4f}), "
          f"vix_only={avgs['vix_only']:.4f} ({avgs['vix_only']-avgs['climatology']:+.4f}), "
          f"both={avgs['both']:.4f} ({avgs['both']-avgs['climatology']:+.4f})  [{time.time()-t0:.0f}s]")

out_path = os.path.join(OUT_DIR, "results_fss_unified.json")
with open(out_path, "w") as f:
    json.dump({"windows": WINDOWS, "upper_thresholds": UPPER_THRESHOLDS,
               "lower_thresholds": LOWER_THRESHOLDS, "results": all_results}, f, indent=2, default=float)
print(f"\nSaved results to {out_path}")
print(f"Total time: {time.time()-t0:.1f}s")
print("\nDone.")
