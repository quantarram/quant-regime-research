"""
run_improvements.py
===================
Runs the three Phase-1/2/3 improvements against the existing pipeline and
produces a comparison table. Requires the standard data files in the
working directory:
  - multiasset_prices.parquet
  - joint_cpe_results.parquet  (rebuilt with episode_utils if needed)

USAGE
-----
    # Full run (all three improvements, 1000 randomisation reps):
    python run_improvements.py

    # Skip randomisation tests (faster, for development):
    python run_improvements.py --skip-rand

    # Single improvement only:
    python run_improvements.py --phases 1        # episode consistency check only
    python run_improvements.py --phases 1,2      # episode + detrending diagnostic
    python run_improvements.py --phases 1,2,3    # all three

WHAT EACH PHASE DOES
--------------------
Phase 1 — Episode counting consistency audit
    Loads joint_cpe_results.parquet and re-computes episode conviction for
    every row using the canonical episode_utils.py implementation. Reports:
      - How many rows have stored n_episodes that differ from the re-computed
        value (the discrepancy flagged in paper Section 20.6).
      - For the VIXM+VIXY->SPY tau_f=63 configurations specifically, what
        the stored vs. re-computed episode count is.
    Does NOT modify any files; this is a read-only audit.

Phase 2 — Detrending diagnostic
    Calls detrend_utils.decay_detrend_report() on the training window and
    prints the raw vs. detrended 95th-percentile thresholds for each decay
    instrument at tau=252.
    Also runs build_increments_and_thresholds() with detrending applied
    and counts how many 2025 days each decay instrument clears its threshold
    under raw vs. detrended increments.
    Does NOT re-run the full CPE screen (that requires joint_cpe_engine.py;
    see the section below on how to do a full re-run).

Phase 3 — Vol-targeted neutral weights backtest
    Runs the hold-to-horizon backtest using vol-targeted neutral weights
    (from vol_targeting.py) instead of the static Sharpe-derived weights.
    Compares against:
      (a) Static Sharpe-derived neutral weights (paper Section 16.1 result)
      (b) EWMA inverse-variance weights (paper Section 6 result, reproduced)
      (c) 63-day vol-targeted weights (this fix)
    Runs the randomisation test for each.

HOW TO DO A FULL RE-RUN WITH ALL FIXES
---------------------------------------
To regenerate joint_cpe_results.parquet with:
  - Consistent episode counting (episode_utils.py shared by both engines)
  - Detrended increments for VIX-complex ETPs

1. Copy episode_utils.py and detrend_utils.py to the directory containing
   cpe_engine_parallel.py and joint_cpe_engine.py.

2. In cpe_engine_parallel.py, add to the increment-building section:
       from detrend_utils import integrate_detrending, DECAY_INSTRUMENTS
       ...
       # After building inc_df for each tau:
       for ticker in DECAY_INSTRUMENTS:
           if ticker in inc_df.columns:
               inc_df[ticker] = get_detrended_increment(prices[ticker], tau, ticker)

3. In joint_cpe_engine.py, replace the local cluster_into_episodes and
   compute_episode_stats functions with imports from episode_utils:
       from episode_utils import (
           cluster_into_episodes, compute_episode_conviction,
           episode_conviction_for_row
       )

4. In backtest_engine.py, replace _cluster_into_episodes and
   _episode_conviction_for_row with imports from episode_utils:
       from episode_utils import (
           cluster_into_episodes as _cluster_into_episodes,
           episode_conviction_for_row as _episode_conviction_for_row,
       )

5. Re-run: python cpe_engine_parallel.py && python joint_cpe_engine.py

The resulting joint_cpe_results.parquet will have stored n_episodes
values that are consistent with what backtest_engine.py computes at
runtime, eliminating the Section 20.6 discrepancy.
"""

import argparse
import sys
import os
import time
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Optional, Set
warnings.filterwarnings("ignore")

# ── BOOTSTRAP: add the cpe_improvements directory to the path ────────────────
# This allows importing from the improvement modules regardless of where
# the script is run from, provided the improvements directory is accessible.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from episode_utils import (
    cluster_into_episodes,
    compute_episode_conviction,
    episode_conviction_for_row,
    EPISODE_MIN_CONVICTION,
    EPISODE_GAP_MULTIPLIER,
)
from detrend_utils import (
    decay_detrend_report,
    integrate_detrending,
    DECAY_INSTRUMENTS,
    get_detrended_increment,
)
from vol_targeting import (
    build_vol_targeted_neutral_weights_v2,
    build_vol_targeted_weights_v3,
    build_vol_targeted_neutral_weights,
    compare_sizing_schemes,
    PORTFOLIO_VOL_TARGET,
    VOL_LOOKBACK,
    FLOOR_WEIGHT,
    CEILING_WEIGHT,
)

# ── PARSE ARGS ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Run CPE improvement phases 1-3")
parser.add_argument("--phases",   default="1,2,3",
                    help="Comma-separated phases to run (default: 1,2,3)")
parser.add_argument("--joint",    default="joint_cpe_results.parquet")
parser.add_argument("--prices",   default="multiasset_prices.parquet")
parser.add_argument("--skip-rand", action="store_true",
                    help="Skip randomisation tests (faster)")
parser.add_argument("--n-reps",   type=int, default=1000)
args = parser.parse_args()

PHASES = set(int(p.strip()) for p in args.phases.split(","))

print(f"\n{'='*70}")
print(f"  CPE IMPROVEMENT PHASES: {sorted(PHASES)}")
print(f"{'='*70}")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print(f"\n  Loading {args.prices}...")
try:
    prices = pd.read_parquet(args.prices)
    print(f"  Price history: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"  Instruments: {prices.shape[1]}")
except FileNotFoundError:
    print(f"  ERROR: {args.prices} not found in {os.getcwd()}")
    sys.exit(1)

joint = None
if 1 in PHASES or 3 in PHASES:
    print(f"\n  Loading {args.joint}...")
    try:
        joint = pd.read_parquet(args.joint)
        print(f"  Joint configs loaded: {len(joint):,}")
    except FileNotFoundError:
        print(f"  WARNING: {args.joint} not found — phases requiring it will be skipped.")

# ── PHASE 1: EPISODE COUNTING CONSISTENCY AUDIT ──────────────────────────────

if 1 in PHASES:
    print(f"\n{'='*70}")
    print(f"  PHASE 1: EPISODE COUNTING CONSISTENCY AUDIT")
    print(f"{'='*70}")

    if joint is None:
        print("  Skipped (no joint parquet).")
    else:
        from backtest_engine import (
            TRAIN_CUTOFF, build_increments_for_episodes,
        )

        print(f"\n  Re-computing episode conviction for all {len(joint):,} joint configs")
        print(f"  using episode_utils.py (canonical implementation)...")

        # Build increments once (expensive but needed for per-row episode recomputation)
        needed_taus = sorted(set(
            int(t) for taus in joint["tau_pasts"] for t in taus
        ) | set(int(t) for t in joint["tau_future"]))
        t0 = time.time()
        increments_ep = build_increments_for_episodes(prices, needed_taus)
        print(f"  Increments built in {time.time()-t0:.1f}s")

        t0 = time.time()
        recomputed = joint.apply(
            lambda row: episode_conviction_for_row(row, increments_ep, TRAIN_CUTOFF),
            axis=1
        )
        print(f"  Re-computation done in {time.time()-t0:.1f}s")

        if "episode_conviction" in joint.columns:
            stored = joint["episode_conviction"].fillna(0.0)
            delta  = (recomputed - stored).abs()
            n_discrepant = (delta > 1e-4).sum()
            print(f"\n  Stored vs re-computed episode_conviction discrepancies:")
            print(f"    Total rows:            {len(joint):,}")
            print(f"    Rows with delta > 1e-4: {n_discrepant:,} "
                  f"({100*n_discrepant/len(joint):.1f}%)")
            print(f"    Max delta:             {delta.max():.6f}")
            if n_discrepant > 0:
                print(f"\n  *** DISCREPANCY FOUND (confirms Section 20.6 finding) ***")
                print(f"  The stored n_episodes column in joint_cpe_results.parquet")
                print(f"  was computed by a different implementation than backtest_engine.py.")
                print(f"  Re-run joint_cpe_engine.py with episode_utils.py to fix.")
        else:
            print(f"  (no stored 'episode_conviction' column — fresh screen, "
                  f"or older format)")

        # Specific check: VIXM+VIXY->SPY tau_f=63
        vv_spy = joint[
            (joint["Y"] == "SPY") &
            (joint["tau_future"] == 63) &
            (joint["direction"] == "bullish") &
            (joint["predictors"].apply(lambda p: "VIXM" in p and "VIXY" in p))
        ]
        print(f"\n  VIXM+VIXY->SPY, tau_f=63, bullish configs: {len(vv_spy)}")
        if len(vv_spy) > 0:
            for _, r in vv_spy.iterrows():
                conv_stored  = r.get("episode_conviction", "n/a")
                conv_recomp  = episode_conviction_for_row(r, increments_ep, TRAIN_CUTOFF)
                n_ep_stored  = r.get("n_episodes", "n/a")
                print(f"    predictors={list(r['predictors'])} "
                      f"n_episodes_stored={n_ep_stored} "
                      f"conv_stored={conv_stored} "
                      f"conv_recomputed={conv_recomp:.6f}")
                if isinstance(conv_stored, float):
                    print(f"    delta={abs(conv_recomp - conv_stored):.2e}")

        # Run the self-test cross-check
        print(f"\n  Running episode_utils cross-check (Section 14.3 verification)...")
        try:
            from episode_utils import _run_self_test as ep_selftest
            ep_selftest()
        except AssertionError as e:
            print(f"  CROSS-CHECK FAILED: {e}")

# ── PHASE 2: DETRENDING DIAGNOSTIC ────────────────────────────────────────────

if 2 in PHASES:
    print(f"\n{'='*70}")
    print(f"  PHASE 2: DETRENDING DIAGNOSTIC FOR VIX-COMPLEX ETPs")
    print(f"{'='*70}")

    from backtest_engine import TRAIN_CUTOFF

    print(f"\n  Detrend report — tau=252, q=0.95 (the Section 3.3 failure case):")
    report = decay_detrend_report(prices, TRAIN_CUTOFF, tau=252, q=0.95)
    print(report.to_string(index=False))

    print(f"\n  Detrend report — tau=126, q=0.95 (the validated signal's tau_past):")
    report_126 = decay_detrend_report(prices, TRAIN_CUTOFF, tau=126, q=0.95)
    print(report_126.to_string(index=False))

    # Run detrend self-test
    print(f"\n  Running detrend_utils self-test...")
    try:
        from detrend_utils import _run_self_test as det_selftest
        det_selftest()
    except AssertionError as e:
        print(f"  SELF-TEST FAILED: {e}")

    # Count firing days in 2025 with raw vs detrended thresholds
    from backtest_engine import build_increments_and_thresholds, Q_GRID_FOR_THRESHOLDS

    print(f"\n  Building increments (raw)...")
    increments_raw, thresholds_raw = build_increments_and_thresholds(
        prices, Q_GRID_FOR_THRESHOLDS
    )

    print(f"  Building increments (detrended decay instruments)...")
    import copy
    increments_det = {tau: df.copy() for tau, df in increments_raw.items()}
    from backtest_engine import RATE_INDEX_TICKERS as RATE_TICKERS
    increments_det = integrate_detrending(
        increments_det, prices, list(increments_det.keys()),
        rate_index_tickers=RATE_TICKERS,
    )

    # Rebuild thresholds from detrended increments
    q_grid_full = sorted(set(Q_GRID_FOR_THRESHOLDS + [round(1-q, 10) for q in Q_GRID_FOR_THRESHOLDS]))
    train_mask = prices.index <= TRAIN_CUTOFF
    eval_mask_2025 = (prices.index >= pd.Timestamp("2025-01-01")) & (prices.index <= pd.Timestamp("2025-12-31"))

    thresholds_det = {}
    for tau in increments_det:
        train_inc = increments_det[tau].loc[train_mask]
        for q in q_grid_full:
            thresholds_det[(tau, q)] = train_inc.quantile(q, numeric_only=True).to_dict()

    print(f"\n  Firing-day comparison — 2025 evaluation window:")
    print(f"  {'Ticker':<8} {'Tau':>5} {'Q':>5}  {'Raw thresh':>12}  {'Det thresh':>12}  "
          f"{'Raw fires':>10}  {'Det fires':>10}")
    print(f"  {'-'*80}")

    for ticker in sorted(DECAY_INSTRUMENTS & set(prices.columns)):
        for tau in [126, 252]:
            for q in [0.90, 0.95]:
                raw_t = thresholds_raw.get((tau, q), {}).get(ticker, np.nan)
                det_t = thresholds_det.get((tau, q), {}).get(ticker, np.nan)
                if np.isnan(raw_t) or tau not in increments_raw:
                    continue
                if ticker not in increments_raw[tau].columns:
                    continue

                eval_raw = increments_raw[tau][ticker][eval_mask_2025].dropna()
                eval_det = increments_det[tau][ticker][eval_mask_2025].dropna()

                raw_fires = int((eval_raw > raw_t).sum())
                det_fires = int((eval_det > det_t).sum()) if not np.isnan(det_t) else -1

                print(f"  {ticker:<8} {tau:>5} {q:>5.2f}  {raw_t:>12.4f}  "
                      f"{det_t:>12.4f}  {raw_fires:>10}  {det_fires:>10}")

# ── PHASE 3: VOL-TARGETED BACKTEST ────────────────────────────────────────────

if 3 in PHASES:
    print(f"\n{'='*70}")
    print(f"  PHASE 3: VOL-TARGETED NEUTRAL WEIGHTS BACKTEST")
    print(f"{'='*70}")

    if joint is None:
        print("  Skipped (no joint parquet).")
    else:
        import backtest_engine as _be
        from backtest_engine import (
            TRAIN_CUTOFF, EVAL_START, EVAL_END,
            BASE_SLEEVES, compute_neutral_weights,
            build_increments_and_thresholds,
            Q_GRID_FOR_THRESHOLDS,
        )
        from run_backtest import (
            run_hold_to_horizon, run_no_tilt_benchmark, run_buy_and_hold,
            randomisation_test_hth, get_eval_dates, load_and_filter_joint,
        )

        _be.SLEEVES.clear()
        _be.SLEEVES.update(BASE_SLEEVES)

        eval_dates = get_eval_dates(prices)
        joint_f = load_and_filter_joint(args.joint)

        print(f"\n  Building increments and thresholds...")
        t0 = time.time()
        increments, thresholds = build_increments_and_thresholds(prices, Q_GRID_FOR_THRESHOLDS)
        print(f"  Done ({time.time()-t0:.1f}s)")

        # ── (A) Static Sharpe-derived weights (paper Section 16.1) ──────────
        print(f"\n  (A) Static Sharpe-derived neutral weights [paper Section 16.1]")
        static_w = compute_neutral_weights(BASE_SLEEVES, prices)
        _be.NEUTRAL_WEIGHTS.clear()
        _be.NEUTRAL_WEIGHTS.update(static_w)

        hth_a = run_hold_to_horizon(joint_f, prices, increments, thresholds, eval_dates)
        bench_a = run_no_tilt_benchmark(prices, eval_dates)
        spy = run_buy_and_hold(prices, "SPY", eval_dates)
        print(f"  HTH:       {hth_a['stats']}")
        print(f"  No-tilt:   {bench_a['stats']}")
        print(f"  SPY:       {spy['stats']}")

        if not args.skip_rand:
            print(f"  Running randomisation test (HTH, static weights)...")
            t0 = time.time()
            rand_a = randomisation_test_hth(
                joint_f, prices, increments, thresholds, eval_dates, n_reps=args.n_reps
            )
            print(f"  pct_exceeding={rand_a.get('pct_exceeding')}%  "
                  f"n_holds={rand_a.get('n_holds')}  ({time.time()-t0:.0f}s)")

        # ── (B) EWMA inverse-variance (reproducing Section 6 failure) ───────
        print(f"\n  (B) EWMA inverse-variance weights (λ=0.94) [paper Section 6 comparison]")
        lam = 0.94
        ewma_weights_by_day: Dict = {}
        for sleeve, ticker in BASE_SLEEVES.items():
            if ticker not in prices.columns:
                continue
            px = prices[ticker].ffill()
            rets = np.log(px / px.shift(1))
            ewma_weights_by_day[sleeve] = rets.ewm(alpha=1 - lam, adjust=False).var() * 252

        # Build daily weight DF for EWMA
        ewma_daily_df_rows = []
        for d in eval_dates:
            raw = {}
            for sleeve in BASE_SLEEVES:
                v_s = ewma_weights_by_day.get(sleeve)
                if v_s is not None and d in v_s.index:
                    var_val = float(v_s[d])
                    raw[sleeve] = 1.0 / var_val if var_val > 0 else 0.0
                else:
                    raw[sleeve] = 0.0
            total = sum(raw.values())
            ewma_daily_df_rows.append(
                {k: v / total * 100.0 if total > 0 else 100.0 / len(raw)
                 for k, v in raw.items()}
            )
        ewma_weights_df = pd.DataFrame(ewma_daily_df_rows, index=eval_dates)
        print(f"  Mean weights (EWMA inv-var):")
        for sleeve in BASE_SLEEVES:
            if sleeve in ewma_weights_df.columns:
                print(f"    {sleeve}: {ewma_weights_df[sleeve].mean():.1f}%  "
                      f"(range {ewma_weights_df[sleeve].min():.1f}–{ewma_weights_df[sleeve].max():.1f}%)")

        # ── (C) Vol-targeted weights (the fix) ───────────────────────────────
        print(f"\n  (C) 63-day vol-targeted weights [this improvement]")
        print(f"  Portfolio vol target: {PORTFOLIO_VOL_TARGET}%  |  "
              f"Lookback: {VOL_LOOKBACK}d  |  "
              f"Bounds: [{FLOOR_WEIGHT}%, {CEILING_WEIGHT}%]")

        vol_target_df = build_vol_targeted_weights_v3(
            prices, eval_dates, BASE_SLEEVES,
            portfolio_vol_target=PORTFOLIO_VOL_TARGET,
            vol_lookback=VOL_LOOKBACK,
            floor_weight=FLOOR_WEIGHT,
            ceiling_weight=CEILING_WEIGHT,
        )

        print(f"  Mean weights (vol-targeted):")
        for sleeve in BASE_SLEEVES:
            if sleeve in vol_target_df.columns:
                print(f"    {sleeve}: {vol_target_df[sleeve].mean():.1f}%  "
                      f"(range {vol_target_df[sleeve].min():.1f}–{vol_target_df[sleeve].max():.1f}%)")

        # Run backtest with dynamic vol-targeted weights
        # We override NEUTRAL_WEIGHTS in the engine with the day-specific values
        # by patching simulate_portfolio's weight lookup.
        # Strategy: run hold_to_horizon to get hold events, then replay
        # with vol-targeted base weights.
        from backtest_engine import (
            compute_quality_weights, build_increments_for_episodes,
            configuration_fires_on_date, clip_and_renormalise,
            simulate_portfolio, compute_performance_stats, HORIZON_WEIGHTS,
        )

        # Set static fallback for quality weight computation
        _be.NEUTRAL_WEIGHTS.clear()
        _be.NEUTRAL_WEIGHTS.update(static_w)

        # Get quality weights and hold events (same as run_hold_to_horizon)
        needed_taus = sorted(set(
            int(t) for taus in joint_f["tau_pasts"] for t in taus
        ) | set(int(t) for t in joint_f["tau_future"]))
        inc_ep = build_increments_for_episodes(prices, needed_taus)
        quality_weights = compute_quality_weights(joint_f, prices, inc_ep)

        # Find all signal fires during eval window
        open_holds = {}     # sleeve -> (expiry_date, tilt_pp)
        daily_weights = pd.DataFrame(
            index=eval_dates, columns=list(BASE_SLEEVES.keys()), dtype=float
        )

        TILT_95th = float(quality_weights.quantile(0.95)) if len(quality_weights) > 0 else 1.0

        for i, d in enumerate(eval_dates):
            # Use vol-targeted weights as the base for this day (lagged)
            if i == 0:
                base_w = {s: 100.0/len(BASE_SLEEVES) for s in BASE_SLEEVES}
            else:
                prev_d = eval_dates[i - 1]
                base_w = {s: float(vol_target_df.at[prev_d, s])
                          for s in BASE_SLEEVES if s in vol_target_df.columns}

            # Check for new fires
            for _, row in joint_f.iterrows():
                y = row["Y"]
                tau_f = int(row["tau_future"])
                direction = row["direction"]

                # Map target ticker to sleeve
                sleeve = None
                for sl, tk in BASE_SLEEVES.items():
                    if tk == y:
                        sleeve = sl
                        break
                if sleeve is None:
                    continue

                qw = quality_weights.get(row.name, 0.0)
                if qw <= 0:
                    continue

                if configuration_fires_on_date(row, d, increments, thresholds):
                    conv_fraction = min(qw / TILT_95th, 1.0)
                    tilt_pp = conv_fraction * 15.0
                    if direction == "bearish":
                        tilt_pp = -tilt_pp

                    expiry_td = pd.bdate_range(d, periods=tau_f + 1)[-1]

                    if sleeve not in open_holds or expiry_td > open_holds[sleeve][0]:
                        open_holds[sleeve] = (expiry_td, tilt_pp)

            # Remove expired holds
            open_holds = {s: (exp, tp) for s, (exp, tp) in open_holds.items() if exp >= d}

            # Build weight: base + tilt
            raw_w = dict(base_w)
            for sleeve, (_, tilt_pp) in open_holds.items():
                raw_w[sleeve] = raw_w.get(sleeve, 0.0) + tilt_pp

            final_w = clip_and_renormalise(raw_w)
            for sleeve in BASE_SLEEVES:
                daily_weights.at[d, sleeve] = final_w.get(sleeve, 0.0)

        eq_c = simulate_portfolio(daily_weights, prices, eval_dates)
        stats_c = compute_performance_stats(eq_c["equity"])
        print(f"  HTH (vol-targeted base): {stats_c}")

        # No-tilt benchmark with vol-targeted weights
        bench_weights = pd.DataFrame(
            {s: vol_target_df[s].shift(1).fillna(100.0/len(BASE_SLEEVES))
             for s in BASE_SLEEVES if s in vol_target_df.columns},
            index=eval_dates,
        )
        bench_vt = simulate_portfolio(bench_weights, prices, eval_dates)
        bench_stats_c = compute_performance_stats(bench_vt["equity"])
        print(f"  No-tilt (vol-targeted base): {bench_stats_c}")

        # Vol-targeted self-test
        print(f"\n  Running vol_targeting self-test...")
        try:
            from vol_targeting import _run_self_test as vt_selftest
            vt_selftest()
        except AssertionError as e:
            print(f"  SELF-TEST FAILED: {e}")

        # ── SUMMARY TABLE ─────────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"  PHASE 3 SUMMARY — 2025 HOLD-TO-HORIZON PERFORMANCE")
        print(f"{'='*70}")
        rows_summary = [
            {"Specification": "Paper Sec 16.1 (static Sharpe weights)",
             **hth_a["stats"]},
            {"Specification": "Paper Sec 16.1 no-tilt benchmark",
             **bench_a["stats"]},
            {"Specification": "Vol-targeted HTH (this improvement, C)",
             **stats_c},
            {"Specification": "Vol-targeted no-tilt benchmark",
             **bench_stats_c},
            {"Specification": "SPY buy-and-hold",
             **spy["stats"]},
        ]
        print(pd.DataFrame(rows_summary).to_string(index=False))

        print(f"\n  Weight comparison:")
        sizing_cmp = compare_sizing_schemes(prices, eval_dates, BASE_SLEEVES, TRAIN_CUTOFF)
        print(sizing_cmp[["sleeve", "ticker", "static_weight_pct",
                            "ewma_invvar_mean_pct", "vol_target_mean_pct",
                            "vol_target_min_pct", "vol_target_max_pct"]].to_string(index=False))

print(f"\n{'='*70}")
print(f"  DONE")
print(f"{'='*70}\n")
