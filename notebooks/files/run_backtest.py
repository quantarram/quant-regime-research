"""
run_backtest.py
================
Driver script. Loads a joint CPE screen (either the original unrestricted
one or the prior-gated one), runs the static-tilt strategy AND the
hold-to-horizon strategy against it for the full 2025 evaluation window,
plus the no-tilt benchmark, and reports results in the same format as
the paper's own tables for direct comparison.

Usage:
    python run_backtest.py --joint joint_cpe_results_ORIGINAL.parquet --label "Run 1: Original unrestricted screen"
    python run_backtest.py --joint joint_cpe_results_PRIOR_GATED.parquet --label "Run 2: Prior-gated screen"
"""

import argparse
import pandas as pd
import numpy as np
import time
import backtest_engine as _be
from backtest_engine import (
    SLEEVES, NEUTRAL_WEIGHTS, HORIZON_WEIGHTS, TRAIN_CUTOFF, EVAL_START, EVAL_END,
    build_increments_and_thresholds, compute_quality_weights,
    compute_daily_class_scores, _static_tilt_delta, clip_and_renormalise,
    simulate_portfolio, compute_performance_stats, configuration_fires_on_date,
)

Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]


def load_and_filter_joint(joint_path: str) -> pd.DataFrame:
    """Load the joint screen and restrict to size <= 6 predictors, the
    same MAX_PREDICTORS cap used by cpe_signal_score.py (spec consistency
    with the rest of the pipeline)."""
    joint = pd.read_parquet(joint_path)
    joint = joint[joint["n_predictors"] <= 6].copy()
    return joint


def get_eval_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Evaluation calendar = SPY's own trading days within the window. This
    matters because multiasset_prices.parquet's row index spans every
    calendar day (crypto trades 7 days/week), so naively using the full
    panel index includes weekends/holidays where SPY/GC=F/TLT/UUP have
    no print -- forward-filling those onto a window that starts on a
    non-trading day (e.g. 2025-01-01, a holiday) leaves the very first
    evaluation date NaN with nothing prior to fill from, which silently
    corrupts the simulated return series. Anchoring eval_dates to SPY's
    own non-null trading days avoids this for the four business-day
    sleeves; BTC-USD (which trades every day) is reindexed onto this
    calendar instead, which is the correct direction for a book whose
    other four sleeves only trade on business days anyway.
    """
    mask = (prices.index >= EVAL_START) & (prices.index <= EVAL_END)
    window = prices.index[mask]
    spy_valid = prices["SPY"].notna()
    return window[window.isin(prices.index[spy_valid])]


# ── STATIC TILT STRATEGY (paper Sections 2-9) ────────────────────────────

def run_static_tilt(joint: pd.DataFrame, prices: pd.DataFrame,
                     increments, thresholds, eval_dates) -> dict:
    print("  Computing quality weights w(Pi)  [episode-conviction mode]...")
    weights = compute_quality_weights(joint, prices, precomputed_increments=increments)

    sleeve_scores = {}
    for sleeve, proxy in SLEEVES.items():
        sub = joint[joint["Y"] == proxy]
        print(f"  [{sleeve:<10}] {len(sub)} joint configs for proxy {proxy} -- scoring {len(eval_dates)} days...")
        t0 = time.time()
        scores = compute_daily_class_scores(proxy, sub, weights, eval_dates, increments, thresholds)
        sleeve_scores[sleeve] = scores
        n_nonzero = (scores != 0).sum()
        print(f"    -> {n_nonzero} non-neutral-score days, elapsed {time.time()-t0:.0f}s")

    score_df = pd.DataFrame(sleeve_scores)
    tilt_df = score_df.map(_static_tilt_delta)

    raw_weight_df = pd.DataFrame(index=eval_dates, columns=SLEEVES.keys(), dtype=float)
    for sleeve in SLEEVES:
        raw_weight_df[sleeve] = NEUTRAL_WEIGHTS[sleeve] + tilt_df[sleeve]

    clipped_rows = []
    for d in eval_dates:
        clipped_rows.append(clip_and_renormalise(raw_weight_df.loc[d].to_dict()))
    final_weight_df = pd.DataFrame(clipped_rows, index=eval_dates)

    # Spec A.7: lag weights by one trading day before applying to returns
    lagged_weight_df = final_weight_df.shift(1)
    lagged_weight_df.iloc[0] = pd.Series(NEUTRAL_WEIGHTS)

    equity = simulate_portfolio(lagged_weight_df, prices, eval_dates)
    stats = compute_performance_stats(equity["equity"])

    return {
        "equity_curve": equity,
        "stats": stats,
        "tilt_df": tilt_df,
        "score_df": score_df,
        "n_nonneutral_days": {s: int((tilt_df[s] != 0).sum()) for s in SLEEVES},
    }


# ── HOLD-TO-HORIZON STRATEGY (paper Section 10.3) ────────────────────────

def run_hold_to_horizon(joint: pd.DataFrame, prices: pd.DataFrame,
                         increments, thresholds, eval_dates) -> dict:
    weights = compute_quality_weights(joint, prices, precomputed_increments=increments)

    # 95th percentile benchmark computed ONLY over tradeable-sleeve-target
    # configurations, per spec A.10
    sleeve_proxies = set(SLEEVES.values())
    tradeable_mask = joint["Y"].isin(sleeve_proxies)
    tradeable_weights = weights[tradeable_mask]
    if len(tradeable_weights) == 0:
        w95 = 1.0
    else:
        w95 = np.percentile(tradeable_weights, 95)
        if w95 <= 0:
            w95 = tradeable_weights.max() if tradeable_weights.max() > 0 else 1.0

    # Active holds per sleeve: list of dicts {expiry_date, tilt_value}
    active_holds = {s: [] for s in SLEEVES}
    raw_weight_df = pd.DataFrame(index=eval_dates, columns=SLEEVES.keys(), dtype=float)

    # Pre-check firing on every date for every tradeable-target config,
    # so we can detect "newly fires" (today fires, yesterday didn't)
    fire_cache = {}  # (config_idx) -> {date: bool}
    for sleeve, proxy in SLEEVES.items():
        sub = joint[(joint["Y"] == proxy) & tradeable_mask.reindex(joint.index, fill_value=False)]
        print(f"  [{sleeve:<10}] hold-to-horizon: checking {len(sub)} configs across {len(eval_dates)} days...")
        for idx, row in sub.iterrows():
            fires_series = {}
            prev_fired = False
            for d in eval_dates:
                fired_today = configuration_fires_on_date(row, d, increments, thresholds)
                fires_series[d] = (fired_today, fired_today and not prev_fired)  # (fires, newly_fires)
                prev_fired = fired_today
            fire_cache[idx] = (row, fires_series)

    for sleeve, proxy in SLEEVES.items():
        relevant_idx = [idx for idx, (row, _) in fire_cache.items() if row["Y"] == proxy]
        for d in eval_dates:
            # Expire holds past their horizon
            active_holds[sleeve] = [h for h in active_holds[sleeve] if h["expiry"] > d]

            # Check for new fires today among this sleeve's configs
            for idx in relevant_idx:
                row, fires_series = fire_cache[idx]
                fired_today, newly_fires = fires_series[d]
                if newly_fires:
                    tau_f = int(row["tau_future"])
                    expiry = d + pd.Timedelta(days=int(tau_f * 1.45))  # approx trading->calendar days
                    direction_sign = 1.0 if row["direction"] == "bullish" else -1.0
                    conviction = min(weights.loc[idx] / w95, 1.0) if w95 > 0 else 0.0
                    tilt_value = direction_sign * conviction * 100.0  # scaled into [-100,100] pp space, clipped later
                    active_holds[sleeve].append({"expiry": expiry, "tilt": tilt_value})

            # Tilt for today = largest-MAGNITUDE active hold (spec A.9), or 0 if none
            if active_holds[sleeve]:
                tilt_today = max(active_holds[sleeve], key=lambda h: abs(h["tilt"]))["tilt"]
            else:
                tilt_today = 0.0
            raw_weight_df.at[d, sleeve] = NEUTRAL_WEIGHTS[sleeve] + tilt_today

    clipped_rows = []
    for d in eval_dates:
        clipped_rows.append(clip_and_renormalise(raw_weight_df.loc[d].to_dict()))
    final_weight_df = pd.DataFrame(clipped_rows, index=eval_dates)

    lagged_weight_df = final_weight_df.shift(1)
    lagged_weight_df.iloc[0] = pd.Series(NEUTRAL_WEIGHTS)

    equity = simulate_portfolio(lagged_weight_df, prices, eval_dates)
    stats = compute_performance_stats(equity["equity"])

    n_holds_opened = {s: sum(1 for idx in fire_cache if fire_cache[idx][0]["Y"] == SLEEVES[s]
                              and any(v[1] for v in [fire_cache[idx][1][d] for d in eval_dates]))
                       for s in SLEEVES}

    return {"equity_curve": equity, "stats": stats, "n_holds_opened": n_holds_opened}


# ── NO-TILT BENCHMARK ─────────────────────────────────────────────────────

def run_no_tilt_benchmark(prices: pd.DataFrame, eval_dates: pd.DatetimeIndex) -> dict:
    weight_df = pd.DataFrame([NEUTRAL_WEIGHTS] * len(eval_dates), index=eval_dates)
    equity = simulate_portfolio(weight_df, prices, eval_dates)
    stats = compute_performance_stats(equity["equity"])
    return {"equity_curve": equity, "stats": stats}


# ── SPY / GOLD BUY-AND-HOLD COMPARATORS ──────────────────────────────────

def run_buy_and_hold(prices: pd.DataFrame, ticker: str, eval_dates: pd.DatetimeIndex) -> dict:
    px = prices[ticker].ffill().reindex(eval_dates)
    rets = px.pct_change().fillna(0)
    equity = 100_000 * (1 + rets).cumprod()
    stats = compute_performance_stats(equity)
    return {"equity_curve": equity, "stats": stats}


# ── RANDOMISATION TEST (spec A.11) ───────────────────────────────────────

def randomisation_test(tilt_df: pd.DataFrame, prices: pd.DataFrame,
                        eval_dates: pd.DatetimeIndex, n_reps: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    actual_weight_rows = []
    for d in eval_dates:
        raw = {s: NEUTRAL_WEIGHTS[s] + tilt_df.loc[d, s] for s in SLEEVES}
        actual_weight_rows.append(clip_and_renormalise(raw))
    actual_weights = pd.DataFrame(actual_weight_rows, index=eval_dates).shift(1)
    actual_weights.iloc[0] = pd.Series(NEUTRAL_WEIGHTS)
    actual_equity = simulate_portfolio(actual_weights, prices, eval_dates)
    actual_sharpe = compute_performance_stats(actual_equity["equity"])["sharpe"]

    n_days = len(eval_dates)
    null_sharpes = []
    for rep in range(n_reps):
        perm = rng.permutation(n_days)
        permuted_tilt = tilt_df.iloc[perm].reset_index(drop=True)
        permuted_tilt.index = eval_dates
        rows = []
        for d in eval_dates:
            raw = {s: NEUTRAL_WEIGHTS[s] + permuted_tilt.loc[d, s] for s in SLEEVES}
            rows.append(clip_and_renormalise(raw))
        w = pd.DataFrame(rows, index=eval_dates).shift(1)
        w.iloc[0] = pd.Series(NEUTRAL_WEIGHTS)
        eq = simulate_portfolio(w, prices, eval_dates)
        s = compute_performance_stats(eq["equity"])["sharpe"]
        if not np.isnan(s):
            null_sharpes.append(s)

    null_sharpes = np.array(null_sharpes)
    pct_exceeding = (null_sharpes >= actual_sharpe).mean() * 100

    return {
        "actual_sharpe": actual_sharpe,
        "null_mean": float(null_sharpes.mean()),
        "null_std": float(null_sharpes.std()),
        "pct_exceeding": round(pct_exceeding, 1),
        "n_reps": len(null_sharpes),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--skip-randomisation", action="store_true")
    parser.add_argument("--exclude-sleeve", action="append", default=[],
                         help="Sleeve name to drop entirely (e.g. Crypto). "
                              "Repeatable: --exclude-sleeve Crypto --exclude-sleeve FX. "
                              "Remaining sleeves' neutral weights are renormalised to "
                              "sum to 100%% so the comparison isn't biased by leftover cash.")
    args = parser.parse_args()

    if args.exclude_sleeve:
        for s in args.exclude_sleeve:
            if s not in _be.SLEEVES:
                raise SystemExit(f"--exclude-sleeve '{s}' is not a known sleeve. "
                                  f"Valid options: {list(_be.SLEEVES.keys())}")
        kept = {k: v for k, v in _be.SLEEVES.items() if k not in args.exclude_sleeve}
        kept_weights_raw = {k: v for k, v in _be.NEUTRAL_WEIGHTS.items() if k in kept}
        total = sum(kept_weights_raw.values())
        kept_weights = {k: v * 100.0 / total for k, v in kept_weights_raw.items()}

        print(f"\n  *** --exclude-sleeve {args.exclude_sleeve}: dropping from the book ***")
        print(f"  Remaining sleeves, renormalised neutral weights: {kept_weights}")

        # Mutate the SHARED module objects in place (not reassign the
        # name) so every function in backtest_engine.py that reads
        # SLEEVES / NEUTRAL_WEIGHTS as a bare global sees the same
        # updated dict, since they imported the dict object itself.
        for s in args.exclude_sleeve:
            _be.SLEEVES.pop(s, None)
            _be.NEUTRAL_WEIGHTS.pop(s, None)
        for k, v in kept_weights.items():
            _be.NEUTRAL_WEIGHTS[k] = v

    print(f"\n{'='*70}")
    print(f"  {args.label or args.joint}")
    print(f"{'='*70}")

    prices = pd.read_parquet("multiasset_prices.parquet")
    eval_dates = get_eval_dates(prices)
    print(f"  Evaluation window: {eval_dates.min().date()} to {eval_dates.max().date()} ({len(eval_dates)} trading days)")
    print(f"  Active sleeves: {list(_be.SLEEVES.keys())}")

    joint = load_and_filter_joint(args.joint)
    print(f"  Joint configs loaded (n_predictors<=6): {len(joint)}")

    t0 = time.time()
    print("\n  Building increments and training-frozen thresholds...")
    increments, thresholds = build_increments_and_thresholds(prices, Q_GRID)
    print(f"  Done. {time.time()-t0:.0f}s")

    print("\n  --- No-tilt benchmark ---")
    bench = run_no_tilt_benchmark(prices, eval_dates)
    print(f"  {bench['stats']}")

    print("\n  --- SPY buy-and-hold ---")
    spy_bh = run_buy_and_hold(prices, "SPY", eval_dates)
    print(f"  {spy_bh['stats']}")

    print("\n  --- Static tilt strategy ---")
    t0 = time.time()
    static = run_static_tilt(joint, prices, increments, thresholds, eval_dates)
    print(f"  {static['stats']}")
    print(f"  Non-neutral days by sleeve: {static['n_nonneutral_days']}")
    print(f"  Elapsed: {time.time()-t0:.0f}s")

    if not args.skip_randomisation:
        print("\n  --- Randomisation test (static tilt) ---")
        t0 = time.time()
        rtest = randomisation_test(static["tilt_df"], prices, eval_dates, n_reps=1000)
        print(f"  Actual Sharpe={rtest['actual_sharpe']}  Null mean={rtest['null_mean']:.3f}  "
              f"Null std={rtest['null_std']:.3f}  Pct exceeding={rtest['pct_exceeding']}%  "
              f"({rtest['n_reps']} reps)")
        print(f"  Elapsed: {time.time()-t0:.0f}s")

    print("\n  --- Hold-to-horizon strategy ---")
    t0 = time.time()
    hth = run_hold_to_horizon(joint, prices, increments, thresholds, eval_dates)
    print(f"  {hth['stats']}")
    print(f"  Holds opened by sleeve: {hth['n_holds_opened']}")
    print(f"  Elapsed: {time.time()-t0:.0f}s")

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    summary = pd.DataFrame({
        "No-tilt benchmark": bench["stats"],
        "SPY buy-and-hold": spy_bh["stats"],
        "Static tilt": static["stats"],
        "Hold-to-horizon": hth["stats"],
    }).T
    print(summary.to_string())

    # Save equity curves for inspection
    out = pd.DataFrame({
        "no_tilt": bench["equity_curve"]["equity"],
        "spy_bh": spy_bh["equity_curve"],
        "static_tilt": static["equity_curve"]["equity"],
        "hold_to_horizon": hth["equity_curve"]["equity"],
    })
    fname = f"backtest_result_{args.joint.replace('.parquet','')}.csv"
    out.to_csv(fname)
    print(f"\n  Saved equity curves -> {fname}")


if __name__ == "__main__":
    main()
