"""
cpe_hth_multi_year_test.py
============================
Does CPE hold-to-horizon's HAC-robust alpha (2025: +23.87%/yr, t=+3.06 to
+3.57 across lags) hold up in a DIFFERENT year, or was it specific to
2025's particular gold bull market? Reruns the exact same strategy
(original and tau*-aware) with TRAIN_CUTOFF/eval window shifted back one
year at a time -- train through Dec 31 of year Y-1, evaluate all of year
Y, strictly out-of-sample, same discipline as the original 2025 test, just
repeated at three additional non-overlapping years: 2022, 2023, 2024.

One disclosed methodological limitation, unchanged from every other test
in this program: the underlying joint CPE screen (joint_cpe_results.parquet
-- which predictor-target configurations exist at all) was built using
full-sample quantile thresholds, not year-by-year train-only ones (a
known, pre-existing property of that artifact, confirmed in
apply_episode_conviction_to_joint.py). Testing additional years with the
same screen doesn't introduce a NEW leak relative to the already-accepted
2025 test -- it's the identical methodology, just re-run on more windows
-- but it does mean the SET of configurations being tested was chosen with
some knowledge of their full-history behavior. The FIRING check and
episode-conviction weighting themselves ARE properly train-cutoff-frozen
for whichever year is under test (backtest_engine.py's own discipline,
unchanged here).

TSMOM and XSMOM are computed for the exact same calendar years directly
from their own already-validated return series, for a genuine apples-to-
apples read on whether ANY of these strategies -- ours or the published
controls -- show a consistent, not-just-2025 edge.

No significance/randomisation-test games -- real train/test splits, one
per year, reported honestly whichever way each one comes out.

Run (from notebooks/files/):
    python cpe_hth_multi_year_test.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

import backtest_engine as _be
from backtest_engine import build_increments_and_thresholds, SLEEVES
from run_backtest import load_and_filter_joint, run_hold_to_horizon, Q_GRID

sys.path.insert(0, ".")
from run_backtest_tau_aware_hth import run_hold_to_horizon_tau_aware

sys.path.insert(0, "..")
import tsmom_crisis_alpha_check as _tmc
import xsmom_benchmark as _xm

YEARS = [2022, 2023, 2024, 2025]


def get_eval_dates_for_year(prices: pd.DataFrame, year: int) -> pd.DatetimeIndex:
    start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    mask = (prices.index >= start) & (prices.index <= end)
    window = prices.index[mask]
    spy_valid = prices["SPY"].notna()
    return window[window.isin(prices.index[spy_valid])]


def hac_alpha(strat_ret: pd.Series, bench_ret: pd.Series, maxlags: int) -> dict:
    common = strat_ret.index.intersection(bench_ret.index)
    y = strat_ret.reindex(common).fillna(0.0).values
    x = bench_ret.reindex(common).fillna(0.0).values
    if len(y) < 20 or np.std(y) == 0:
        return {"alpha_annualized_pct": float("nan"), "t_hac": float("nan"), "significant_95": False, "n": len(y)}
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    alpha_daily = model.params[0]
    t_hac = model.tvalues[0]
    return {"alpha_annualized_pct": float(alpha_daily * 252 * 100), "t_hac": float(t_hac),
            "significant_95": bool(abs(t_hac) >= 1.96), "n": len(y)}


if __name__ == "__main__":
    prices = pd.read_parquet("../multiasset_prices.parquet")
    joint = load_and_filter_joint("../joint_cpe_results.parquet")
    print(f"Joint configs loaded (n_predictors<=6): {len(joint)}")

    # tsmom_crisis_alpha_check.build_series() and xsmom_benchmark's helpers
    # read "multiasset_prices.parquet" relative to notebooks/ (their own
    # normal working directory) -- hop over there just for these two calls,
    # then back to notebooks/files/ for the rest of this script.
    _here = os.getcwd()
    os.chdir("..")
    tsmom_ret, tsmom_passive = _tmc.build_series()
    os.chdir(_here)

    xs_first_valid = min(prices[t].dropna().index.min() for t in _xm.UNIVERSE)
    xs_start = xs_first_valid + pd.Timedelta(days=int(1.5 * _xm.MIN_HISTORY_DAYS))
    from tsmom_benchmark import month_end_dates
    xs_rebal_dates = month_end_dates(prices.index[prices.index >= xs_start])
    xs_weight_df = _xm.build_monthly_positions(prices, xs_rebal_dates)
    xs_trading_index = prices.index[prices.index >= xs_rebal_dates[0]]
    xsmom_ret = _xm.simulate_xsmom(xs_weight_df, prices, xs_trading_index)
    xsmom_passive = _xm.simulate_passive(prices, xs_trading_index, xs_weight_df)

    all_rows = []
    for year in YEARS:
        print(f"\n{'='*90}\n  YEAR {year}  (train through {year-1}-12-31, evaluate all of {year})\n{'='*90}")
        _be.TRAIN_CUTOFF = pd.Timestamp(f"{year-1}-12-31")
        eval_dates = get_eval_dates_for_year(prices, year)
        print(f"  Eval dates: {eval_dates.min().date()} to {eval_dates.max().date()} ({len(eval_dates)} days)")

        increments, thresholds = build_increments_and_thresholds(prices, Q_GRID)

        orig = run_hold_to_horizon(joint, prices, increments, thresholds, eval_dates)
        tau_aware = run_hold_to_horizon_tau_aware(joint, prices, increments, thresholds, eval_dates)

        from run_backtest import run_no_tilt_benchmark
        no_tilt = run_no_tilt_benchmark(prices, eval_dates)
        bench_ret = np.log(no_tilt["equity_curve"]["equity"]).diff()

        for label, res in [("CPE HTH original", orig), ("CPE HTH tau*-aware", tau_aware)]:
            strat_ret = np.log(res["equity_curve"]["equity"]).diff()
            hac63 = hac_alpha(strat_ret, bench_ret, maxlags=63)
            print(f"  {label:<22} alpha={hac63['alpha_annualized_pct']:+7.2f}%/yr  "
                  f"t_HAC63={hac63['t_hac']:+.2f}  [{'SIGNIFICANT' if hac63['significant_95'] else 'not significant'}]  "
                  f"Sharpe={res['stats']['sharpe']}")
            all_rows.append((year, label, hac63["alpha_annualized_pct"], hac63["t_hac"], hac63["significant_95"]))

        year_mask_t = (tsmom_ret.index >= pd.Timestamp(f"{year}-01-01")) & (tsmom_ret.index <= pd.Timestamp(f"{year}-12-31"))
        year_mask_x = (xsmom_ret.index >= pd.Timestamp(f"{year}-01-01")) & (xsmom_ret.index <= pd.Timestamp(f"{year}-12-31"))
        t_hac = hac_alpha(tsmom_ret[year_mask_t], tsmom_passive[year_mask_t], maxlags=21)
        x_hac = hac_alpha(xsmom_ret[year_mask_x], xsmom_passive[year_mask_x], maxlags=21)
        print(f"  {'TSMOM':<22} alpha={t_hac['alpha_annualized_pct']:+7.2f}%/yr  t_HAC21={t_hac['t_hac']:+.2f}  "
              f"[{'SIGNIFICANT' if t_hac['significant_95'] else 'not significant'}]")
        print(f"  {'XSMOM':<22} alpha={x_hac['alpha_annualized_pct']:+7.2f}%/yr  t_HAC21={x_hac['t_hac']:+.2f}  "
              f"[{'SIGNIFICANT' if x_hac['significant_95'] else 'not significant'}]")
        all_rows.append((year, "TSMOM", t_hac["alpha_annualized_pct"], t_hac["t_hac"], t_hac["significant_95"]))
        all_rows.append((year, "XSMOM", x_hac["alpha_annualized_pct"], x_hac["t_hac"], x_hac["significant_95"]))

    print(f"\n{'='*90}\n  SUMMARY ACROSS ALL {len(YEARS)} YEARS\n{'='*90}")
    df = pd.DataFrame(all_rows, columns=["year", "strategy", "alpha_pct", "t_hac", "significant"])
    pivot = df.pivot(index="strategy", columns="year", values="alpha_pct")
    print(pivot.round(2).to_string())
    print("\nSignificant (HAC-corrected, 95%) by year:")
    sig_pivot = df.pivot(index="strategy", columns="year", values="significant")
    print(sig_pivot.to_string())
