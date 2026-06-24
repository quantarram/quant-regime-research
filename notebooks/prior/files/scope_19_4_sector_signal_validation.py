"""
scope_19_4_sector_signal_validation.py
========================================
Scope 19.4: Validate short-conditioning-window sector signals on 2020–2024
(Section 19.4 of the paper).

The JNK/HYG → XLY/XLI/XLP signals (tau_past=5, 45–81 independent training
episodes, 80–87% hit rates) are the episode-conviction screen's strongest
signals by episode count. Yet when added to the extended sleeve book in
Section 16.2, they reversed the HTH randomisation result from 1.8% exceeding
to 80.5% exceeding. Section 19.4 requires:
  "A multi-year walk-forward (2020–2024) on these signals alone, before any
   further attempt to incorporate them into the sleeve structure."

This script tests the JNK/HYG → {XLY, XLI, XLP, LQD} signals in isolation
across 2020–2024, using expanding training windows and the same hold-to-horizon
episode-conviction mechanism. The key question is:
  Are the 2025/2026 failures specific to those years' market environments,
  or do these signals systematically underperform their training-period metrics?

Signal specification:
  - Predictors: JNK (SPDR Bloomberg High Yield Bond ETF) and
                HYG (iShares iBoxx High Yield Corporate Bond ETF)
  - Conditioning window: tau_past = 5 (5-day lookback, short window)
  - Targets: XLY (Consumer Discretionary), XLI (Industrials),
             XLP (Consumer Staples), LQD (Investment Grade Credit)
  - Direction: bullish (credit surge predicts equity/credit outperformance)

Comparison structure for each eval year:
  A. Equal-weight buy-and-hold of the 4 sector/credit targets
  B. Hold-to-horizon using JNK/HYG→target configurations + randomisation test
  C. SPY buy-and-hold (broader market context)

Also reports: whether each year's signals fire at all, how many hold events,
and whether the firing rate matches the historical tau_past=5 expectation
(short windows fire very frequently — many firing days expected per year).

Usage:
    python scope_19_4_sector_signal_validation.py --joint joint_cpe_results.parquet
    python scope_19_4_sector_signal_validation.py --joint joint_cpe_results.parquet --years 2020 2021 2022 2023 2024
    python scope_19_4_sector_signal_validation.py --joint joint_cpe_results.parquet --skip-randomisation
"""

import argparse
import sys
import os
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.getcwd())

try:
    import backtest_engine as _be
    from backtest_engine import (
        compute_neutral_weights, build_increments_and_thresholds,
        compute_quality_weights, build_increments_for_episodes,
        clip_and_renormalise, simulate_portfolio, compute_performance_stats,
        configuration_fires_on_date, HORIZON_WEIGHTS,
        _cluster_into_episodes, EPISODE_MIN_OBS_FOR_CONVICTION,
        EPISODE_ANCHOR, TRAIN_CUTOFF as PAPER_TRAIN_CUTOFF,
    )
except ImportError as e:
    sys.exit(f"ERROR: Cannot import backtest_engine.py\n  {e}")

try:
    from run_backtest import (
        run_hold_to_horizon, run_no_tilt_benchmark, run_buy_and_hold,
        randomisation_test_hth, load_and_filter_joint, get_eval_dates,
    )
except ImportError as e:
    sys.exit(f"ERROR: Cannot import run_backtest.py\n  {e}")

Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]

# ── Sector signal specification (Section 19.4) ────────────────────────────
SECTOR_PREDICTORS = {"JNK", "HYG"}  # must both appear in a configuration
SECTOR_TARGETS    = {"XLY", "XLI", "XLP", "LQD"}

# Sleeve definitions for the ISOLATED sector test
# (4 targets only; no FX/Equities/Gold/Bonds/Crypto to avoid confounds)
SECTOR_SLEEVES = {
    "ConsDisc":   "XLY",
    "Industrials": "XLI",
    "ConsStaples": "XLP",
    "Credit":     "LQD",
}


def apply_equal_neutral_weights(sleeves: dict) -> dict:
    """Equal neutral weights for the sector-only book."""
    n = len(sleeves)
    return {k: 100.0 / n for k in sleeves}


def run_equal_weight_bh(targets: list, prices: pd.DataFrame,
                         eval_dates: pd.DatetimeIndex) -> dict:
    """Equal-weight buy-and-hold of the sector/credit targets."""
    n = len(targets)
    if n == 0:
        return {"stats": {}, "equity_curve": pd.Series()}
    notional = 100_000.0
    equity = [notional]
    dates_out = [eval_dates[0]]

    returns = pd.DataFrame(index=eval_dates)
    for t in targets:
        if t not in prices.columns:
            returns[t] = 0.0
        else:
            px = prices[t].ffill().reindex(eval_dates)
            returns[t] = px.pct_change()

    for i in range(1, len(eval_dates)):
        d = eval_dates[i]
        r = returns.loc[d].fillna(0.0)
        port_ret = r.mean()  # equal weight
        equity.append(equity[-1] * (1 + port_ret))
        dates_out.append(d)

    eq_series = pd.Series(equity, index=dates_out)
    stats = compute_performance_stats(eq_series)
    return {"stats": stats, "equity_curve": eq_series}


def check_sector_signals_in_joint(joint: pd.DataFrame) -> pd.DataFrame:
    """
    Find joint-screen rows where BOTH JNK and HYG are in the predictor set
    AND the target (Y) is one of the sector/credit targets.
    """
    mask_predictors = joint["predictors"].apply(
        lambda p: SECTOR_PREDICTORS.issubset(set(p))
    )
    mask_targets = joint["Y"].isin(SECTOR_TARGETS)
    return joint[mask_predictors & mask_targets].copy()


def run_sector_year(eval_year: int, sector_joint: pd.DataFrame,
                    prices: pd.DataFrame,
                    skip_randomisation: bool = False,
                    n_reps: int = 1000) -> dict:
    """
    Run the isolated sector signal test for one evaluation year.
    Uses ONLY the JNK/HYG→{XLY,XLI,XLP,LQD} configurations.
    """
    train_cutoff = pd.Timestamp(f"{eval_year - 1}-12-31")
    eval_start   = pd.Timestamp(f"{eval_year}-01-01")
    eval_end     = pd.Timestamp(f"{eval_year}-12-31")

    print(f"\n  {'─'*60}")
    print(f"  SECTOR YEAR {eval_year}  |  Training: ..–{train_cutoff.date()}  |  "
          f"Eval: {eval_start.date()}–{eval_end.date()}")
    print(f"  {'─'*60}")

    # Check data availability
    for t in ["JNK", "HYG", "XLY", "XLI", "XLP", "LQD"]:
        if t not in prices.columns:
            print(f"  WARNING: {t} not in price data — sector test may be incomplete")

    eval_mask = (prices.index >= eval_start) & (prices.index <= eval_end)
    xly_eval = prices.loc[eval_mask, "XLY"].dropna() if "XLY" in prices.columns else pd.Series()
    if len(xly_eval) < 50:
        print(f"  SKIPPED: insufficient XLY eval data ({len(xly_eval)} days)")
        return None

    train_mask = prices.index <= train_cutoff
    jnk_train_obs = len(prices.loc[train_mask, "JNK"].dropna()) if "JNK" in prices.columns else 0
    hug_train_obs = len(prices.loc[train_mask, "HYG"].dropna()) if "HYG" in prices.columns else 0

    if min(jnk_train_obs, hug_train_obs) < 252:
        print(f"  SKIPPED: JNK ({jnk_train_obs} obs) or HYG ({hug_train_obs} obs) "
              f"insufficient training data")
        return None

    print(f"  JNK training obs: {jnk_train_obs}  |  HYG training obs: {hug_train_obs}")

    # Override engine cutoff
    _be.TRAIN_CUTOFF = train_cutoff
    _be.EVAL_START   = eval_start
    _be.EVAL_END     = eval_end

    # Equal neutral weights for sector-only book
    equal_weights = apply_equal_neutral_weights(SECTOR_SLEEVES)
    _be.SLEEVES.clear()
    _be.SLEEVES.update(SECTOR_SLEEVES)
    _be.NEUTRAL_WEIGHTS.clear()
    _be.NEUTRAL_WEIGHTS.update(equal_weights)

    print(f"  Sector-only sleeve weights: " +
          "  ".join(f"{k}:{v:.1f}%" for k, v in equal_weights.items()))

    # Build increments + thresholds
    print(f"  Building increments/thresholds...")
    t0 = time.time()
    increments, thresholds = build_increments_and_thresholds(prices, Q_GRID)
    print(f"  Done. {time.time()-t0:.1f}s")

    # Eval dates
    eval_dates = get_eval_dates(prices)
    print(f"  Eval trading days: {len(eval_dates)}")
    if len(eval_dates) == 0:
        print(f"  SKIPPED: no valid SPY-based eval dates")
        return None

    # Check sector configs surviving the screen at this training window
    if len(sector_joint) == 0:
        print(f"  No JNK/HYG sector configurations found in joint screen — skipping")
        return None

    print(f"  JNK/HYG→{{XLY,XLI,XLP,LQD}} configs in screen: {len(sector_joint)}")
    for target in SECTOR_TARGETS:
        n_t = len(sector_joint[sector_joint["Y"] == target])
        tau_pasts = sorted(set(
            int(tp) for pasts in sector_joint[sector_joint["Y"] == target]["tau_pasts"]
            for tp in pasts
        )) if n_t > 0 else []
        print(f"    {target}: {n_t} configs  tau_pasts={tau_pasts}")

    # Equal-weight buy-and-hold of the 4 targets
    print(f"\n  --- Equal-weight buy-and-hold (4 targets) ---")
    ew_bh = run_equal_weight_bh(list(SECTOR_TARGETS), prices, eval_dates)
    print(f"  {ew_bh['stats']}")

    # SPY buy-and-hold
    spy_bh = run_buy_and_hold(prices, "SPY", eval_dates)
    print(f"  SPY buy-and-hold: {spy_bh['stats']}")

    # Hold-to-horizon using ONLY sector configurations
    print(f"\n  --- Hold-to-horizon (JNK/HYG sector signals only) ---")
    t0 = time.time()
    hth = run_hold_to_horizon(sector_joint, prices, increments, thresholds, eval_dates)
    print(f"  {hth['stats']}")
    print(f"  Holds opened: {hth['n_holds_opened']}")
    print(f"  Elapsed: {time.time()-t0:.0f}s")

    rtest_result = None
    total_holds = sum(hth["n_holds_opened"].values())
    if not skip_randomisation and total_holds >= 3:
        print(f"\n  --- Randomisation test ({n_reps} reps, {total_holds} holds) ---")
        t0 = time.time()
        rtest_result = randomisation_test_hth(
            sector_joint, prices, increments, thresholds, eval_dates, n_reps=n_reps
        )
        if "note" in rtest_result:
            print(f"  {rtest_result['note']}")
        else:
            print(f"  Pct exceeding: {rtest_result['pct_exceeding']}%  "
                  f"Null mean={rtest_result['null_mean']:.3f}  "
                  f"Null std={rtest_result['null_std']:.3f}")
        print(f"  Elapsed: {time.time()-t0:.0f}s")
    else:
        if total_holds < 3:
            print(f"  Randomisation test skipped: {total_holds} holds (need ≥ 3)")
        rtest_result = {"note": f"Skipped: {total_holds} hold event(s)"}

    return {
        "year": eval_year,
        "train_cutoff": str(train_cutoff.date()),
        "eval_days": len(eval_dates),
        "jnk_train_obs": jnk_train_obs,
        "hug_train_obs": hug_train_obs,
        "sector_configs_in_screen": len(sector_joint),
        "ew_bh_ret_pct": ew_bh["stats"].get("total_return_pct", np.nan),
        "ew_bh_sharpe": ew_bh["stats"].get("sharpe", np.nan),
        "spy_ret_pct": spy_bh["stats"]["total_return_pct"],
        "spy_sharpe": spy_bh["stats"]["sharpe"],
        "hth_ret_pct": hth["stats"]["total_return_pct"],
        "hth_sharpe": hth["stats"]["sharpe"],
        "hth_ann_vol_pct": hth["stats"]["ann_vol_pct"],
        "hth_holds_total": total_holds,
        "hth_holds_by_sleeve": str(hth["n_holds_opened"]),
        "pct_exceeding": rtest_result.get("pct_exceeding", "n/a") if rtest_result else "skipped",
        "rand_null_mean": rtest_result.get("null_mean", np.nan) if rtest_result else np.nan,
        "rand_null_std": rtest_result.get("null_std", np.nan) if rtest_result else np.nan,
        "rand_note": rtest_result.get("note", "") if rtest_result else "",
        "signal_vs_ew_ret": (hth["stats"]["total_return_pct"] -
                              ew_bh["stats"].get("total_return_pct", 0)),
        "signal_vs_ew_sharpe": (hth["stats"]["sharpe"] -
                                  ew_bh["stats"].get("sharpe", 0)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Scope 19.4: Validate short-window sector signals (2020–2024)"
    )
    parser.add_argument("--joint", default="joint_cpe_results.parquet")
    parser.add_argument("--prices", default="multiasset_prices.parquet")
    parser.add_argument("--years", nargs="+", type=int,
                        default=[2020, 2021, 2022, 2023, 2024],
                        help="Evaluation years (default: 2020–2024)")
    parser.add_argument("--skip-randomisation", action="store_true")
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument("--output", default="sector_validation_19_4_results.csv")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  SCOPE 19.4 — SECTOR SIGNAL VALIDATION (2020–2024)")
    print(f"  Signals: JNK/HYG → {{XLY, XLI, XLP, LQD}}  (tau_past=5)")
    print(f"  Isolated test: sector-only sleeve book, no other signals")
    print(f"  Evaluation years: {args.years}")
    print(f"{'='*72}")

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\n  Loading {args.prices}...")
    prices = pd.read_parquet(args.prices)
    print(f"  Price history: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"  Instruments:   {prices.shape[1]}")

    # Check sector tickers
    for t in list(SECTOR_PREDICTORS) + list(SECTOR_TARGETS):
        if t not in prices.columns:
            print(f"  WARNING: {t} not found in price data")

    print(f"\n  Loading joint screen: {args.joint}...")
    joint_full = load_and_filter_joint(args.joint)
    print(f"  Total joint configs: {len(joint_full)}")

    # Filter to JNK/HYG sector configurations only
    sector_joint = check_sector_signals_in_joint(joint_full)
    print(f"  JNK/HYG→sector/credit configs: {len(sector_joint)}")

    if len(sector_joint) == 0:
        print(f"\n  ERROR: No JNK/HYG→{{XLY,XLI,XLP,LQD}} configurations found in")
        print(f"  the joint screen. This usually means either:")
        print(f"   1) The screen was built with the economic prior excluding these pairs")
        print(f"   2) These configurations didn't survive CPE/lift/n filters")
        print(f"   3) The joint screen file is an older version")
        print(f"\n  Check the screen for any JNK/HYG-containing configurations:")
        jnk_any = joint_full[joint_full["predictors"].apply(lambda p: "JNK" in p or "HYG" in p)]
        print(f"  Configs with JNK or HYG as predictor: {len(jnk_any)}")
        if len(jnk_any) > 0:
            print(f"  Sample targets: {jnk_any['Y'].unique()[:10]}")
        return

    print(f"\n  Configuration breakdown:")
    for t in SECTOR_TARGETS:
        sub = sector_joint[sector_joint["Y"] == t]
        if len(sub) > 0:
            tau_f_vals = sorted(sub["tau_future"].unique())
            tau_p_vals = sorted(set(
                int(tp) for pasts in sub["tau_pasts"] for tp in pasts
            ))
            print(f"    {t}: {len(sub)} configs  tau_future={tau_f_vals}  "
                  f"tau_past={tau_p_vals}  "
                  f"CPE range={sub['joint_CPE'].min():.3f}–{sub['joint_CPE'].max():.3f}")

    # ── Main loop ─────────────────────────────────────────────────────────
    results = []
    orig_train_cutoff = _be.TRAIN_CUTOFF
    orig_eval_start   = _be.EVAL_START
    orig_eval_end     = _be.EVAL_END

    for yr in sorted(args.years):
        try:
            res = run_sector_year(
                eval_year=yr,
                sector_joint=sector_joint,
                prices=prices,
                skip_randomisation=args.skip_randomisation,
                n_reps=args.n_reps,
            )
            if res is not None:
                results.append(res)
        except Exception as exc:
            print(f"\n  ERROR in year {yr}: {exc}")
            import traceback
            traceback.print_exc()

    # Restore engine state
    _be.TRAIN_CUTOFF = orig_train_cutoff
    _be.EVAL_START   = orig_eval_start
    _be.EVAL_END     = orig_eval_end

    if not results:
        print("\n  No results produced.")
        return

    df = pd.DataFrame(results)

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print(f"  SECTOR SIGNAL VALIDATION SUMMARY (JNK/HYG → SECTOR TARGETS)")
    print(f"{'='*72}")
    print(f"\n  {'Year':>6}  {'HTH ret%':>9}  {'HTH Sh':>7}  {'EW ret%':>9}  "
          f"{'EW Sh':>7}  {'SPY ret%':>9}  {'Holds':>6}  {'Pct exc':>8}  {'Better?':>8}")
    print(f"  {'─'*82}")

    n_better_than_ew = 0
    n_significant = 0

    for _, r in df.iterrows():
        pct = r["pct_exceeding"]
        is_sig = isinstance(pct, (int, float)) and float(pct) <= 10.0
        better = r["signal_vs_ew_ret"] > 0
        if better:
            n_better_than_ew += 1
        if is_sig:
            n_significant += 1
        sig_flag = "  YES" if is_sig else "   no"
        better_flag = "  YES" if better else "   no"
        print(f"  {int(r['year']):>6}  {r['hth_ret_pct']:>9.2f}%  {r['hth_sharpe']:>7.3f}  "
              f"{r['ew_bh_ret_pct']:>9.2f}%  {r['ew_bh_sharpe']:>7.3f}  "
              f"{r['spy_ret_pct']:>9.2f}%  {r['hth_holds_total']:>6}  "
              f"{str(pct):>8}  {better_flag}")

    # Add 2025 and 2026 paper results for reference
    print(f"  {'─'*82}")
    print(f"  {'2025*':>6}  {'10.57':>9}%  {'0.769':>7}  {'13.71':>9}%  "
          f"{'1.104':>7}  {'18.01':>9}%  {'4+':>6}  {'80.5':>8}%  {'   no'}")
    print(f"  (* 2025 from paper Section 16.2, extended sleeve book — "
          f"different context but same signal class)")

    print(f"\n  AGGREGATE ASSESSMENT:")
    n_years = len(results)
    print(f"  Years tested:                {n_years}")
    print(f"  Years HTH > EW baseline:     {n_better_than_ew}/{n_years}")
    print(f"  Years statistically significant (pct_exc ≤ 10%): {n_significant}/{n_years}")

    total_holds = df["hth_holds_total"].sum()
    mean_holds_per_year = total_holds / n_years if n_years > 0 else 0
    print(f"  Total hold events:           {total_holds}")
    print(f"  Mean holds per year:         {mean_holds_per_year:.1f}")

    print(f"\n  KEY QUESTION (Section 19.4):")
    if n_significant >= 3:
        print(f"  POSITIVE: Sector signals performed significantly in {n_significant}/{n_years} years.")
        print(f"  Hypothesis supported: short-window failure in 2025 was year-specific.")
        print(f"  Recommendation: consider re-adding sector sleeves with caution.")
    elif n_better_than_ew < n_years // 2:
        print(f"  NEGATIVE: Sector signals improved on EW in only {n_better_than_ew}/{n_years} years.")
        print(f"  The 2025 failure is likely structural, not year-specific.")
        print(f"  Recommendation: do NOT re-add sector sleeves until further methodology")
        print(f"  improvements are made (e.g. requiring tau_past ≥ 63 for sector signals).")
    else:
        print(f"  MIXED: {n_better_than_ew}/{n_years} years positive but {n_significant} significant.")
        print(f"  Inconclusive — need more years or a different conditioning horizon.")
        print(f"  Consider testing tau_past ≥ 63 variant before re-adding sector sleeves.")

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_csv(args.output, index=False)
    print(f"\n  Saved: {args.output}")
    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()
