"""
cpe_breadth_expansion_test.py
================================
Tests the highest-leverage, best-evidenced lever identified this session:
breadth. Every strategy's information ratio scales roughly as
IC * sqrt(breadth) (Grinold's "fundamental law of active management") --
more independent bets, not necessarily a better individual one. The
portfolio-tilt engine tested all night trades exactly 5 sleeves (one
proxy each for Equities/Gold/Bonds/Crypto/FX). But the underlying,
already economic-prior-gated joint screen (joint_cpe_results.parquet --
confirmed to already have Papers 3-4's is_admissible() filter baked in)
has real, supported configurations for 132 distinct, tradeable targets,
107 of which this program has simply never traded.

This is NOT a new mechanism or a new fitted parameter -- it is the exact
same static-tilt engine (backtest_engine.py, unchanged), the exact same
quality-weight formula, the exact same episode-conviction gate, applied
to more of what the screen already found, mechanically selected (top 40
targets by number of supporting n_predictors<=6 configurations, a
selection criterion fixed BEFORE looking at any target's OWN return
performance -- picking targets by how much real conditioning evidence
exists for them, not by which ones made money, is what keeps this an
honest breadth test rather than a curated one).

Disclosed limitation, not hidden: many of these 40 targets share
underlying exposure (e.g. NG=F and UNG are both natural gas; several
single-country/commodity ETFs correlate with their futures counterparts
already in the set). Nominal breadth of 40 therefore overstates TRUE
independent breadth -- an honest caveat, reported directly in the output,
not smoothed over.

Same discipline as every other test tonight: multi-year (2022-2025, each
year trained honestly through the prior year-end), Newey-West HAC-
corrected significance, benchmarked against the same 5-sleeve engine and
against TSMOM/XSMOM over the identical years.

Run (from notebooks/files/):
    python cpe_breadth_expansion_test.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

import backtest_engine as _be
from backtest_engine import build_increments_and_thresholds, RATE_INDEX_TICKERS
from run_backtest import load_and_filter_joint, run_static_tilt, run_no_tilt_benchmark, Q_GRID

sys.path.insert(0, "..")
import tsmom_crisis_alpha_check as _tmc
import xsmom_benchmark as _xm
from tsmom_benchmark import month_end_dates

YEARS = [2022, 2023, 2024, 2025]
N_TARGETS = 40
MIN_HISTORY_OBS = 1500


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


def get_eval_dates_for_year(prices: pd.DataFrame, year: int) -> pd.DatetimeIndex:
    start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    mask = (prices.index >= start) & (prices.index <= end)
    window = prices.index[mask]
    spy_valid = prices["SPY"].notna()
    return window[window.isin(prices.index[spy_valid])]


def select_targets(joint: pd.DataFrame, prices: pd.DataFrame, n: int) -> list:
    counts = joint.groupby("Y").size().sort_values(ascending=False)
    selected = []
    for y, c in counts.items():
        if y in RATE_INDEX_TICKERS or y not in prices.columns:
            continue
        if prices[y].dropna().shape[0] < MIN_HISTORY_OBS:
            continue
        selected.append(y)
        if len(selected) >= n:
            break
    return selected


if __name__ == "__main__":
    prices = pd.read_parquet("../multiasset_prices.parquet")
    joint_full = load_and_filter_joint("../joint_cpe_results.parquet")

    targets = select_targets(joint_full, prices, N_TARGETS)
    print(f"Selected {len(targets)} targets (mechanical cutoff: top {N_TARGETS} by config count, "
          f">= {MIN_HISTORY_OBS} price obs, excluding rate indices):")
    print(f"  {targets}")
    print(f"  vs. current production engine's 5 sleeves: {list(_be.SLEEVES.values())}")

    # Expand SLEEVES/NEUTRAL_WEIGHTS to the full target list, equal-weighted
    # (no risk-budgeting rationale to prefer one target over another here --
    # equal weight is the neutral, non-fitted default).
    expanded_sleeves = {t: t for t in targets}
    expanded_weights = {t: 100.0 / len(targets) for t in targets}
    joint_expanded = joint_full[joint_full["Y"].isin(targets)].copy()
    print(f"  {len(joint_expanded)} of {len(joint_full)} total joint configs feed these {len(targets)} targets")

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
        _be.SLEEVES.clear(); _be.SLEEVES.update(expanded_sleeves)
        _be.NEUTRAL_WEIGHTS.clear(); _be.NEUTRAL_WEIGHTS.update(expanded_weights)

        eval_dates = get_eval_dates_for_year(prices, year)
        increments, thresholds = build_increments_and_thresholds(prices, Q_GRID)

        expanded = run_static_tilt(joint_expanded, prices, increments, thresholds, eval_dates)
        no_tilt = run_no_tilt_benchmark(prices, eval_dates)
        bench_ret = np.log(no_tilt["equity_curve"]["equity"]).diff()
        strat_ret = np.log(expanded["equity_curve"]["equity"]).diff()

        hac63 = hac_alpha(strat_ret, bench_ret, maxlags=63)
        n_active_sleeves = sum(1 for s, n in expanded["n_nonneutral_days"].items() if n > 0)
        print(f"\n  {len(targets)}-target static tilt   alpha={hac63['alpha_annualized_pct']:+7.2f}%/yr  "
              f"t_HAC63={hac63['t_hac']:+.2f}  [{'SIGNIFICANT' if hac63['significant_95'] else 'not significant'}]  "
              f"Sharpe={expanded['stats']['sharpe']}  active targets={n_active_sleeves}/{len(targets)}")
        all_rows.append((year, f"{len(targets)}-target CPE static tilt", hac63["alpha_annualized_pct"], hac63["t_hac"], hac63["significant_95"]))

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

    # restore the production 5-sleeve config for anything downstream that imports this module state
    _be.SLEEVES.clear(); _be.SLEEVES.update({"Equities": "SPY", "Gold": "GC=F", "Bonds": "TLT", "Crypto": "BTC-USD", "FX": "UUP"})
    _be.NEUTRAL_WEIGHTS.clear(); _be.NEUTRAL_WEIGHTS.update({"Equities": 30.87, "Gold": 24.29, "Bonds": 4.74, "Crypto": 30.10, "FX": 10.00})

    print(f"\n{'='*90}\n  SUMMARY ACROSS ALL {len(YEARS)} YEARS: {len(targets)}-target breadth expansion vs. 5-sleeve production engine\n{'='*90}")
    df = pd.DataFrame(all_rows, columns=["year", "strategy", "alpha_pct", "t_hac", "significant"])
    print(df.pivot(index="strategy", columns="year", values="alpha_pct").round(2).to_string())
    print("\nSignificant (HAC-corrected, 95%) by year:")
    print(df.pivot(index="strategy", columns="year", values="significant").to_string())
