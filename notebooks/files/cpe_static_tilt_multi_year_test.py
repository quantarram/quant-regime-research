"""
cpe_static_tilt_multi_year_test.py
=====================================
Completes the gauntlet for the ORIGINAL portfolio-tilt work (Papers 3-4)
that cpe_hth_multi_year_test.py already ran for hold-to-horizon (Paper
3's headline mechanism): this script does the same for STATIC TILT
(Paper 4's mechanism: "+30.6% vs +17.4% neutral equal-weight... Sharpe
1.43 vs 1.03"), across the same four non-overlapping years -- 2022, 2023,
2024, 2025 -- each trained honestly through the prior year-end.

Confirmed directly from source (joint_cpe_engine.py line 298,
`prior_mask = pairwise.apply(lambda r: is_admissible(r["X"], r["Y"], ...))`)
before running this: joint_cpe_results.parquet, the screen used
throughout this session, ALREADY has Papers 3-4's economic-prior gate
baked in -- it is not the raw, unrestricted 169k-signal screen. So this
and the earlier hold-to-horizon test both are honest re-tests of the
actual Paper 3/4 methodology, using the best currently available
reimplementation (backtest_engine.py, since the original scoring code no
longer exists per that module's own docstring), not a weaker substitute.

TSMOM and XSMOM are included for the same years, for the same reason as
before: so "did this survive" is judged on the same scale as the
published controls, not in isolation.

No significance/randomisation-test games -- Newey-West HAC correction on
every alpha reported, same discipline as every other test this session --
real train/test splits, reported honestly whichever way each year comes
out.

Run (from notebooks/files/):
    python cpe_static_tilt_multi_year_test.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

import backtest_engine as _be
from backtest_engine import build_increments_and_thresholds
from run_backtest import load_and_filter_joint, run_static_tilt, run_no_tilt_benchmark, Q_GRID

sys.path.insert(0, "..")
import tsmom_crisis_alpha_check as _tmc
import xsmom_benchmark as _xm
from tsmom_benchmark import month_end_dates

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
    return {"alpha_annualized_pct": float(model.params[0] * 252 * 100), "t_hac": float(model.tvalues[0]),
            "significant_95": bool(abs(model.tvalues[0]) >= 1.96), "n": len(y)}


if __name__ == "__main__":
    prices = pd.read_parquet("../multiasset_prices.parquet")
    joint = load_and_filter_joint("../joint_cpe_results.parquet")
    print(f"Joint configs loaded (n_predictors<=6, economic-prior-gated): {len(joint)}")

    _here = os.getcwd()
    os.chdir("..")
    tsmom_ret, tsmom_passive = _tmc.build_series()
    os.chdir(_here)

    xs_first_valid = min(prices[t].dropna().index.min() for t in _xm.UNIVERSE)
    xs_start = xs_first_valid + pd.Timedelta(days=int(1.5 * _xm.MIN_HISTORY_DAYS))
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
        static = run_static_tilt(joint, prices, increments, thresholds, eval_dates)
        no_tilt = run_no_tilt_benchmark(prices, eval_dates)
        bench_ret = np.log(no_tilt["equity_curve"]["equity"]).diff()
        strat_ret = np.log(static["equity_curve"]["equity"]).diff()

        hac63 = hac_alpha(strat_ret, bench_ret, maxlags=63)
        print(f"  CPE static tilt        alpha={hac63['alpha_annualized_pct']:+7.2f}%/yr  "
              f"t_HAC63={hac63['t_hac']:+.2f}  [{'SIGNIFICANT' if hac63['significant_95'] else 'not significant'}]  "
              f"Sharpe={static['stats']['sharpe']}  non-neutral days={static['n_nonneutral_days']}")
        all_rows.append((year, "CPE static tilt", hac63["alpha_annualized_pct"], hac63["t_hac"], hac63["significant_95"]))

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
    print(df.pivot(index="strategy", columns="year", values="alpha_pct").round(2).to_string())
    print("\nSignificant (HAC-corrected, 95%) by year:")
    print(df.pivot(index="strategy", columns="year", values="significant").to_string())
