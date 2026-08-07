"""
run_backtest_tau_aware.py
==========================
Paper 16 candidate: apply what Ramanathan (2026a, 2026c) measured about
predictability limits to the ONE part of this whole research program with
a real, demonstrated Sharpe -- the CPE portfolio-tilt engine (Papers 3-4)
-- rather than to price forecasting, which has found no tradeable alpha in
five independent tests since Paper 12.

The static-tilt mechanism (backtest_engine.py, spec A.5/B.2) combines each
sleeve's daily class score across four horizons (21/63/126/252d) using
ONE FIXED, universal weighting -- HORIZON_WEIGHTS = {21:0.20, 63:0.30,
126:0.30, 252:0.20} -- applied identically to every sleeve regardless of
that sleeve's own measured predictability limit tau* (Equities/SPY=22d,
Gold/GLD-proxy=22d, Bonds/TLT=28d, Crypto/BTC-USD=35d, FX/EURUSD-proxy=
43d). A configuration targeting a sleeve tau_future days ahead is
conditioning on real, causally-supported co-movement out to that horizon
-- but this whole research program's own finding (2026a) is that price
dynamics decorrelate past an instrument's own tau*, so a configuration
whose tau_future sits far beyond that sleeve's tau* is asking the same
"is this still connected to what's happening now" question the
prediction-layer papers already answered "less and less so" for pure
point forecasting. This script tests whether the SAME logic, applied here
to which HORIZON of an already-real signal to trust most, does anything
for the one part of this program that has actual demonstrated alpha.

Design, kept deliberately simple and non-tunable to avoid the selection-
bias trap this program has been caught by before: each sleeve's horizon
weight is a single, parameter-free function of that sleeve's own tau*,

    w(h) ~ 1 / (1 + |log(h) - log(tau*)|),  normalized to sum to 1
    over h in {21, 63, 126, 252}

-- concentrating weight on whichever available horizon is closest (in
log-distance) to the sleeve's own measured decorrelation scale, with nc
free parameters fit to the evaluation window. Everything else -- the
joint screen, episode-conviction weights w(Pi), train/test cutoff
(2024-12-31 / 2025 eval), tilt tiers, position lag -- is IDENTICAL to
backtest_engine.py's validated original, so any difference in Sharpe is
attributable to this one change alone. Only the STATIC TILT mechanism
uses HORIZON_WEIGHTS at all; hold-to-horizon holds each firing
configuration to its own tau_future directly and is untouched by this
lever (reported here unchanged, for reference).

No significance/randomisation-test games -- one real comparison, same
train/test split as the original, reported honestly either way.

Usage:
    python run_backtest_tau_aware.py --joint ../joint_cpe_results.parquet
"""
import argparse
import time

import numpy as np
import pandas as pd

import backtest_engine as _be
from backtest_engine import (
    SLEEVES, NEUTRAL_WEIGHTS, TRAIN_CUTOFF, EVAL_START, EVAL_END,
    build_increments_and_thresholds, compute_quality_weights,
    compute_daily_class_scores, _static_tilt_delta, clip_and_renormalise,
    simulate_portfolio, compute_performance_stats,
)
from run_backtest import load_and_filter_joint, get_eval_dates, run_no_tilt_benchmark, run_buy_and_hold, Q_GRID

# Measured predictability limits (tau*, trading days), Ramanathan (2026a),
# via the same sleeve-proxy tickers backtest_engine.py already trades:
# Equities=SPY, Gold-proxy=GLD (sleeve trades GC=F), Bonds=TLT,
# Crypto=BTC-USD, FX-proxy=EURUSD=X (sleeve trades UUP). Gold futures and
# the dollar index aren't in the 2026a panel directly; GLD and EURUSD=X
# are the same proxy substitutions this codebase already uses elsewhere
# (e.g. predictor_v1's PROXY_TICKERS convention) for instruments outside
# that panel.
SLEEVE_TAU_STAR = {
    "Equities": 22,   # SPY
    "Gold": 22,       # GLD proxy for GC=F
    "Bonds": 28,      # TLT
    "Crypto": 35,     # BTC-USD
    "FX": 43,         # EURUSD=X proxy for UUP
}

AVAILABLE_HORIZONS = [21, 63, 126, 252]


def tau_aware_horizon_weights(tau_star: int) -> dict:
    """w(h) ~ 1 / (1 + |log(h) - log(tau*)|), normalized over the four
    horizons this codebase's joint screen actually uses (300d excluded,
    same as the original HORIZON_WEIGHTS, per spec A.5)."""
    log_tau = np.log(tau_star)
    raw = {h: 1.0 / (1.0 + abs(np.log(h) - log_tau)) for h in AVAILABLE_HORIZONS}
    total = sum(raw.values())
    return {h: w / total for h, w in raw.items()}


SLEEVE_HORIZON_WEIGHTS = {s: tau_aware_horizon_weights(t) for s, t in SLEEVE_TAU_STAR.items()}


def run_static_tilt_tau_aware(joint, prices, increments, thresholds, eval_dates) -> dict:
    print("  Computing quality weights w(Pi)  [episode-conviction mode, unchanged]...")
    weights = compute_quality_weights(joint, prices, precomputed_increments=increments)

    sleeve_scores = {}
    for sleeve, proxy in SLEEVES.items():
        sub = joint[joint["Y"] == proxy]
        hw = SLEEVE_HORIZON_WEIGHTS[sleeve]
        print(f"  [{sleeve:<10}] {len(sub)} joint configs for proxy {proxy} -- "
              f"tau*={SLEEVE_TAU_STAR[sleeve]}d, horizon weights={{{', '.join(f'{h}d:{w:.2f}' for h, w in hw.items())}}}")
        # Temporarily override the module-global HORIZON_WEIGHTS that
        # compute_daily_class_scores reads internally, restoring it right
        # after -- this reuses the validated per-horizon scoring logic
        # unchanged, only varying which horizons matter most per sleeve.
        original_hw = dict(_be.HORIZON_WEIGHTS)
        _be.HORIZON_WEIGHTS.clear()
        _be.HORIZON_WEIGHTS.update(hw)
        try:
            scores = compute_daily_class_scores(proxy, sub, weights, eval_dates, increments, thresholds)
        finally:
            _be.HORIZON_WEIGHTS.clear()
            _be.HORIZON_WEIGHTS.update(original_hw)
        sleeve_scores[sleeve] = scores
        n_nonzero = (scores != 0).sum()
        print(f"    -> {n_nonzero} non-neutral-score days")

    score_df = pd.DataFrame(sleeve_scores)
    tilt_df = score_df.map(_static_tilt_delta)

    raw_weight_df = pd.DataFrame(index=eval_dates, columns=SLEEVES.keys(), dtype=float)
    for sleeve in SLEEVES:
        raw_weight_df[sleeve] = NEUTRAL_WEIGHTS[sleeve] + tilt_df[sleeve]

    clipped_rows = [clip_and_renormalise(raw_weight_df.loc[d].to_dict()) for d in eval_dates]
    final_weight_df = pd.DataFrame(clipped_rows, index=eval_dates)

    lagged_weight_df = final_weight_df.shift(1)
    lagged_weight_df.iloc[0] = pd.Series(NEUTRAL_WEIGHTS)

    equity = simulate_portfolio(lagged_weight_df, prices, eval_dates)
    stats = compute_performance_stats(equity["equity"])

    return {
        "equity_curve": equity, "stats": stats, "tilt_df": tilt_df, "score_df": score_df,
        "n_nonneutral_days": {s: int((tilt_df[s] != 0).sum()) for s in SLEEVES},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", required=True)
    args = parser.parse_args()

    print(f"\n{'='*72}\n  tau*-aware horizon weighting vs. original static-tilt strategy\n{'='*72}")
    print(f"  Per-sleeve tau* and derived horizon weights:")
    for s in SLEEVES:
        hw = SLEEVE_HORIZON_WEIGHTS[s]
        print(f"    {s:<10} tau*={SLEEVE_TAU_STAR[s]:>3}d  " +
              "  ".join(f"{h}d={w:.3f}" for h, w in hw.items()))
    print(f"    {'(original)':<10}{'':>7}  " +
          "  ".join(f"{h}d={w:.3f}" for h, w in _be.HORIZON_WEIGHTS.items()))

    prices = pd.read_parquet("../multiasset_prices.parquet")
    eval_dates = get_eval_dates(prices)
    print(f"\n  Evaluation window: {eval_dates.min().date()} to {eval_dates.max().date()} ({len(eval_dates)} trading days)")

    joint = load_and_filter_joint(args.joint)
    print(f"  Joint configs loaded (n_predictors<=6): {len(joint)}")

    t0 = time.time()
    print("\n  Building increments and training-frozen thresholds...")
    increments, thresholds = build_increments_and_thresholds(prices, Q_GRID)
    print(f"  Done. {time.time()-t0:.0f}s")

    print("\n  --- No-tilt benchmark ---")
    bench = run_no_tilt_benchmark(prices, eval_dates)
    print(f"  {bench['stats']}")

    print("\n  --- Original static tilt (fixed, universal horizon weights) ---")
    original_static = _be
    from run_backtest import run_static_tilt as _run_static_orig
    orig = _run_static_orig(joint, prices, increments, thresholds, eval_dates)
    print(f"  {orig['stats']}")
    print(f"  Non-neutral days by sleeve: {orig['n_nonneutral_days']}")

    print("\n  --- tau*-aware static tilt (per-sleeve horizon weights) ---")
    t0 = time.time()
    tau_aware = run_static_tilt_tau_aware(joint, prices, increments, thresholds, eval_dates)
    print(f"  {tau_aware['stats']}")
    print(f"  Non-neutral days by sleeve: {tau_aware['n_nonneutral_days']}")
    print(f"  Elapsed: {time.time()-t0:.0f}s")

    print(f"\n{'='*72}\n  SUMMARY\n{'='*72}")
    summary = pd.DataFrame({
        "No-tilt benchmark": bench["stats"],
        "Original static tilt (fixed horizon weights)": orig["stats"],
        "tau*-aware static tilt (per-sleeve horizon weights)": tau_aware["stats"],
    }).T
    print(summary.to_string())

    out = pd.DataFrame({
        "no_tilt": bench["equity_curve"]["equity"],
        "static_tilt_original": orig["equity_curve"]["equity"],
        "static_tilt_tau_aware": tau_aware["equity_curve"]["equity"],
    })
    out.to_csv("backtest_result_tau_aware_horizon_weights.csv")
    print("\n  Saved equity curves -> backtest_result_tau_aware_horizon_weights.csv")


if __name__ == "__main__":
    main()
