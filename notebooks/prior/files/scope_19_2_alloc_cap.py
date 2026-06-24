"""
scope_19_2_alloc_cap.py
========================
Scope 19.2: Base allocation cap test (Section 19.2 of the paper).

The paper's 5-sleeve corrected-weight book carries ~28% Crypto (BTC-USD)
and ~25% Gold. Both dragged performance in 2026 H1 when equities ran hard.
Section 19.2 specifies a single, pre-specified structural change:
  - Cap Crypto weight at 15%
  - Cap Gold weight at 20%
  - Reallocate the surplus proportionally to the remaining sleeves

This script:
  1. Loads multiasset_prices.parquet and checks data history depth
  2. Computes corrected neutral weights (same as backtest_engine.py)
  3. Applies the cap, reallocates surplus proportionally
  4. Runs the full 2025 walk-forward under the capped weights, comparing:
       (a) No-tilt capped-weight baseline
       (b) Static tilt + capped weights
       (c) Hold-to-horizon + capped weights
  5. Reports the same performance stats + randomisation tests as the paper

Usage:
    python scope_19_2_alloc_cap.py --joint joint_cpe_results.parquet
    python scope_19_2_alloc_cap.py --joint joint_cpe_results.parquet --skip-randomisation
"""

import argparse
import sys
import os
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Allow running from any directory by looking for the data in CWD ───────
sys.path.insert(0, os.getcwd())

# Import from the uploaded engines (copied to CWD)
try:
    import backtest_engine as _be
    from backtest_engine import (
        TRAIN_CUTOFF, EVAL_START, EVAL_END,
        BASE_SLEEVES, compute_neutral_weights,
        build_increments_and_thresholds, compute_quality_weights,
        build_increments_for_episodes,
        compute_daily_class_scores, _static_tilt_delta, clip_and_renormalise,
        simulate_portfolio, compute_performance_stats,
        configuration_fires_on_date, Q_GRID_FOR_THRESHOLDS,
        HORIZON_WEIGHTS,
    )
except ImportError as e:
    sys.exit(f"ERROR: Could not import backtest_engine.py — make sure it is in the "
             f"current working directory.\n  {e}")

try:
    from run_backtest import (
        run_static_tilt, run_hold_to_horizon,
        run_no_tilt_benchmark, run_buy_and_hold,
        randomisation_test, randomisation_test_hth,
        load_and_filter_joint, get_eval_dates,
    )
except ImportError as e:
    sys.exit(f"ERROR: Could not import run_backtest.py — make sure it is in the "
             f"current working directory.\n  {e}")

Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]

# ── CAP SPECIFICATION (Section 19.2, pre-specified) ───────────────────────
CRYPTO_CAP_PCT = 15.0
GOLD_CAP_PCT   = 20.0

# Map from sleeve name → cap (None = uncapped)
WEIGHT_CAPS = {
    "Equities": None,
    "Gold":     GOLD_CAP_PCT,
    "Bonds":    None,
    "Crypto":   CRYPTO_CAP_PCT,
    "FX":       None,
}


def apply_weight_cap(weights: dict, caps: dict) -> dict:
    """
    Apply per-sleeve weight caps. Surplus from capped sleeves is
    reallocated proportionally to the UNCAPPED sleeves (those whose
    raw Sharpe-derived weight is below their cap and is not at the cap).

    This mirrors how risk-limit-based weight adjustments are handled in
    portfolio construction practice: do not spread surplus to ALL other
    sleeves uniformly, but only to those not at their own ceiling.
    """
    capped = {}
    surplus = 0.0

    # First pass: apply caps, accumulate surplus
    for sleeve, w in weights.items():
        cap = caps.get(sleeve)
        if cap is not None and w > cap:
            surplus += w - cap
            capped[sleeve] = cap
        else:
            capped[sleeve] = w

    if surplus <= 0:
        return capped

    # Second pass: distribute surplus proportionally to uncapped sleeves
    uncapped_sleeves = [s for s, w in weights.items()
                        if caps.get(s) is None or w < (caps.get(s) or float("inf"))]
    uncapped_total = sum(capped[s] for s in uncapped_sleeves)

    if uncapped_total <= 0:
        # Edge case: all sleeves at cap — distribute uniformly
        per_sleeve = surplus / len(capped)
        return {s: v + per_sleeve for s, v in capped.items()}

    for sleeve in uncapped_sleeves:
        share = capped[sleeve] / uncapped_total
        capped[sleeve] += surplus * share

    return capped


def main():
    parser = argparse.ArgumentParser(
        description="Scope 19.2: Base allocation cap test (Crypto ≤15%, Gold ≤20%)"
    )
    parser.add_argument("--joint", default="joint_cpe_results.parquet",
                        help="Joint CPE screen parquet file")
    parser.add_argument("--prices", default="multiasset_prices.parquet")
    parser.add_argument("--skip-randomisation", action="store_true")
    parser.add_argument("--n-reps", type=int, default=1000)
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  SCOPE 19.2 — BASE ALLOCATION CAP TEST")
    print(f"  Crypto ≤ {CRYPTO_CAP_PCT}%, Gold ≤ {GOLD_CAP_PCT}%")
    print(f"  Surplus reallocated proportionally to uncapped sleeves")
    print(f"{'='*72}")

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\n  Loading prices from {args.prices}...")
    prices = pd.read_parquet(args.prices)
    print(f"  Price history: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"  Instruments:   {prices.shape[1]}")

    joint = load_and_filter_joint(args.joint)
    print(f"  Joint configs (n_pred ≤ 6): {len(joint)}")

    # ── Wire up sleeves in the engine ────────────────────────────────────
    _be.SLEEVES.clear()
    _be.SLEEVES.update(BASE_SLEEVES)

    # ── Compute UNCAPPED corrected neutral weights (paper Section 16.1) ───
    uncapped_weights = compute_neutral_weights(BASE_SLEEVES, prices)
    print(f"\n  Uncapped Sharpe-derived weights (full training history):")
    for k, v in uncapped_weights.items():
        cap = WEIGHT_CAPS.get(k)
        flag = f"  *** ABOVE {cap}% cap" if cap and v > cap else ""
        print(f"    {k:<15}: {v:.2f}%{flag}")

    # ── Apply caps ────────────────────────────────────────────────────────
    capped_weights = apply_weight_cap(uncapped_weights, WEIGHT_CAPS)
    print(f"\n  Capped weights (Crypto ≤ {CRYPTO_CAP_PCT}%, Gold ≤ {GOLD_CAP_PCT}%):")
    for k, v in capped_weights.items():
        delta = v - uncapped_weights[k]
        arrow = f"  ({delta:+.2f}pp)" if abs(delta) > 0.01 else ""
        print(f"    {k:<15}: {v:.2f}%{arrow}")

    total = sum(capped_weights.values())
    print(f"    Total:           {total:.2f}%  (should be 100.00%)")

    # ── Wire capped weights into engine ───────────────────────────────────
    _be.NEUTRAL_WEIGHTS.clear()
    _be.NEUTRAL_WEIGHTS.update(capped_weights)

    eval_dates = get_eval_dates(prices)
    print(f"\n  Evaluation window: {eval_dates.min().date()} to {eval_dates.max().date()}")
    print(f"  Trading days: {len(eval_dates)}")

    # ── Build increments + thresholds ─────────────────────────────────────
    print(f"\n  Building increments and training-frozen thresholds...")
    t0 = time.time()
    increments, thresholds = build_increments_and_thresholds(prices, Q_GRID)
    print(f"  Done. {time.time()-t0:.1f}s")

    # ── No-tilt capped benchmark ──────────────────────────────────────────
    print(f"\n  --- No-tilt capped-weight benchmark ---")
    bench_capped = run_no_tilt_benchmark(prices, eval_dates)
    print(f"  {bench_capped['stats']}")

    # ── Uncapped baseline for comparison ─────────────────────────────────
    print(f"\n  --- No-tilt UNCAPPED (corrected) weights benchmark ---")
    _be.NEUTRAL_WEIGHTS.clear()
    _be.NEUTRAL_WEIGHTS.update(uncapped_weights)
    bench_uncapped = run_no_tilt_benchmark(prices, eval_dates)
    print(f"  {bench_uncapped['stats']}")

    # Restore capped weights for signal runs
    _be.NEUTRAL_WEIGHTS.clear()
    _be.NEUTRAL_WEIGHTS.update(capped_weights)

    # ── SPY benchmark ─────────────────────────────────────────────────────
    print(f"\n  --- SPY buy-and-hold ---")
    spy = run_buy_and_hold(prices, "SPY", eval_dates)
    print(f"  {spy['stats']}")

    # ── Static tilt + capped weights ─────────────────────────────────────
    print(f"\n  --- Static tilt (capped weights) ---")
    t0 = time.time()
    static = run_static_tilt(joint, prices, increments, thresholds, eval_dates)
    print(f"  {static['stats']}")
    print(f"  Non-neutral days: {static['n_nonneutral_days']}")
    print(f"  Elapsed: {time.time()-t0:.0f}s")

    if not args.skip_randomisation:
        print(f"\n  --- Randomisation test (static tilt, capped weights) ---")
        t0 = time.time()
        rtest = randomisation_test(static["tilt_df"], prices, eval_dates, n_reps=args.n_reps)
        print(f"  Actual={rtest['actual_sharpe']}  Null mean={rtest['null_mean']:.3f}  "
              f"Null std={rtest['null_std']:.3f}  Pct exceeding={rtest['pct_exceeding']}%  "
              f"({rtest['n_reps']} reps)")
        print(f"  Elapsed: {time.time()-t0:.0f}s")

    # ── Hold-to-horizon + capped weights ─────────────────────────────────
    print(f"\n  --- Hold-to-horizon (capped weights) ---")
    t0 = time.time()
    hth = run_hold_to_horizon(joint, prices, increments, thresholds, eval_dates)
    print(f"  {hth['stats']}")
    print(f"  Holds opened: {hth['n_holds_opened']}")
    print(f"  Elapsed: {time.time()-t0:.0f}s")

    if not args.skip_randomisation:
        print(f"\n  --- Randomisation test (hold-to-horizon, capped weights) ---")
        t0 = time.time()
        rtest_hth = randomisation_test_hth(joint, prices, increments, thresholds, eval_dates, n_reps=args.n_reps)
        if "note" in rtest_hth:
            print(f"  {rtest_hth['note']}")
        else:
            print(f"  Actual={rtest_hth['actual_sharpe']}  Null mean={rtest_hth['null_mean']:.3f}  "
                  f"Null std={rtest_hth['null_std']:.3f}  Pct exceeding={rtest_hth['pct_exceeding']}%  "
                  f"({rtest_hth['n_reps']} reps, {rtest_hth['n_holds']} holds shuffled)")
        print(f"  Elapsed: {time.time()-t0:.0f}s")

    # ── Summary comparison ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  SUMMARY: CAPPED vs UNCAPPED WEIGHTS — 2025 PERFORMANCE")
    print(f"{'='*72}")

    rows = {
        "No-tilt UNCAPPED (paper Sec 16.1)": {**bench_uncapped["stats"],
                                                "total_return_pct": bench_uncapped["stats"]["total_return_pct"]},
        "No-tilt CAPPED (Crypto≤15%, Gold≤20%)": bench_capped["stats"],
        "SPY buy-and-hold": spy["stats"],
        "Static tilt CAPPED": static["stats"],
        "Hold-to-horizon CAPPED": hth["stats"],
    }
    # Add paper's uncapped HTH result for reference
    rows["Hold-to-horizon UNCAPPED (paper Sec 16.1)"] = {
        "total_return_pct": 18.77,
        "ann_vol_pct": 15.15,
        "sharpe": 1.224,
    }

    summary = pd.DataFrame(rows).T
    print(summary.to_string())

    print(f"\n  KEY INTERPRETATION:")
    print(f"  - Weight cap isolates base-allocation effect from signal effect.")
    print(f"  - Comparing 'No-tilt CAPPED' vs 'No-tilt UNCAPPED' shows")
    print(f"    the pure base-allocation value of the cap (no signal involved).")
    print(f"  - Comparing 'HTH CAPPED' vs 'HTH UNCAPPED' shows whether the cap")
    print(f"    helps the signal strategy or merely reshuffles beta exposure.")
    print(f"  - If 'HTH CAPPED' HTH randomisation pct_exceeding ≤ 5%, the signal")
    print(f"    remains significant even after restricting the base allocation.")
    print(f"    If it rises above 10%, the signal's significance was partly")
    print(f"    driven by elevated Crypto/Gold acting as market-direction noise.")

    # ── Save ──────────────────────────────────────────────────────────────
    out = pd.DataFrame({
        "no_tilt_capped":   bench_capped["equity_curve"]["equity"],
        "no_tilt_uncapped": bench_uncapped["equity_curve"]["equity"],
        "spy_bh":           spy["equity_curve"],
        "static_capped":    static["equity_curve"]["equity"],
        "hth_capped":       hth["equity_curve"]["equity"],
    })
    fname = "backtest_scope_19_2_alloc_cap.csv"
    out.to_csv(fname)
    print(f"\n  Saved equity curves → {fname}")
    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()
