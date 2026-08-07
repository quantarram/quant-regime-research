"""
run_backtest_tau_aware_hth.py
==============================
Continues run_backtest_tau_aware.py's static-tilt test onto the mechanism
that's actually more likely to feel it. Static tilt's per-sleeve horizon
reweighting was real (daily scores measurably shifted) but had zero effect
on the final Sharpe, because static tilt only has two coarse tiers
(0.05->8pp, 0.30->15pp) and every affected day already sat inside the same
tier before and after. Hold-to-horizon (spec A.9, Section 10.3) doesn't
use HORIZON_WEIGHTS or discrete tiers at all -- its conviction is
CONTINUOUS: min(w(Pi)/w95, 1.0), and on any day with multiple
simultaneously active holds, the single LARGEST-magnitude one wins
outright. That's exactly the kind of mechanism a continuous tau*-based
adjustment can actually move: it can shift a hold's magnitude enough to
change which hold wins the "largest active" comparison on a given day,
not just nudge a number that gets swallowed by a threshold.

The lever: a hold-to-horizon position is opened when a joint configuration
fires, targeting that configuration's own tau_future days ahead. Nothing
in the current mechanism asks whether tau_future is within the sleeve's
own measured predictability limit tau* (Ramanathan 2026a) or far beyond
it -- a config targeting 252 days ahead for a sleeve whose own dynamics
decorrelate at 22 days gets the exact same conviction treatment as one
targeting 21 days ahead. This script discounts a hold's conviction by how
far tau_future sits beyond that sleeve's own tau*, with the same
parameter-free logic as the static-tilt test:

    freshness(tau_future, tau*) = 1 / (1 + max(0, tau_future/tau* - 1))

-- 1.0 (full trust) for any tau_future within tau*, decaying smoothly as
tau_future grows past it (2x tau* -> 0.5, 4x tau* -> 0.25). Applied as a
pure multiplicative discount on top of the existing min(w/w95, 1.0)
conviction, AFTER w95 is computed the original (undiscounted) way, so the
normalization scale itself is untouched -- isolating the test to whether
tau*-discounting far-horizon holds changes which hold wins each day and,
through that, the realized Sharpe. Everything else -- the joint screen,
episode-conviction weights, train/test cutoff, expiry logic, position lag
-- is identical to backtest_engine.py's validated original.

No significance/randomisation-test games -- one real comparison, same
train/test split as the original, reported honestly either way.

Usage:
    python run_backtest_tau_aware_hth.py --joint ../joint_cpe_results.parquet
"""
import argparse
import time

import numpy as np
import pandas as pd

from backtest_engine import (
    SLEEVES, NEUTRAL_WEIGHTS, build_increments_and_thresholds,
    compute_quality_weights, configuration_fires_on_date,
    clip_and_renormalise, simulate_portfolio, compute_performance_stats,
)
from run_backtest import (
    load_and_filter_joint, get_eval_dates, run_no_tilt_benchmark,
    run_buy_and_hold, run_hold_to_horizon, Q_GRID,
)

# Same measured predictability limits (tau*, trading days) and sleeve-proxy
# mapping as run_backtest_tau_aware.py.
SLEEVE_TAU_STAR = {"Equities": 22, "Gold": 22, "Bonds": 28, "Crypto": 35, "FX": 43}


def freshness_discount(tau_future: int, tau_star: int) -> float:
    return 1.0 / (1.0 + max(0.0, tau_future / tau_star - 1.0))


def run_hold_to_horizon_tau_aware(joint: pd.DataFrame, prices: pd.DataFrame,
                                   increments, thresholds, eval_dates) -> dict:
    weights = compute_quality_weights(joint, prices, precomputed_increments=increments)

    sleeve_proxies = set(SLEEVES.values())
    tradeable_mask = joint["Y"].isin(sleeve_proxies)
    tradeable_weights = weights[tradeable_mask]
    if len(tradeable_weights) == 0:
        w95 = 1.0
    else:
        w95 = np.percentile(tradeable_weights, 95)
        if w95 <= 0:
            w95 = tradeable_weights.max() if tradeable_weights.max() > 0 else 1.0

    active_holds = {s: [] for s in SLEEVES}
    raw_weight_df = pd.DataFrame(index=eval_dates, columns=SLEEVES.keys(), dtype=float)

    fire_cache = {}
    for sleeve, proxy in SLEEVES.items():
        sub = joint[(joint["Y"] == proxy) & tradeable_mask.reindex(joint.index, fill_value=False)]
        print(f"  [{sleeve:<10}] hold-to-horizon (tau*-aware): checking {len(sub)} configs "
              f"across {len(eval_dates)} days, sleeve tau*={SLEEVE_TAU_STAR[sleeve]}d...")
        for idx, row in sub.iterrows():
            fires_series = {}
            prev_fired = False
            for d in eval_dates:
                fired_today = configuration_fires_on_date(row, d, increments, thresholds)
                fires_series[d] = (fired_today, fired_today and not prev_fired)
                prev_fired = fired_today
            fire_cache[idx] = (row, fires_series)

    discount_log = []
    for sleeve, proxy in SLEEVES.items():
        tau_star = SLEEVE_TAU_STAR[sleeve]
        relevant_idx = [idx for idx, (row, _) in fire_cache.items() if row["Y"] == proxy]
        for d in eval_dates:
            active_holds[sleeve] = [h for h in active_holds[sleeve] if h["expiry"] > d]

            for idx in relevant_idx:
                row, fires_series = fire_cache[idx]
                fired_today, newly_fires = fires_series[d]
                if newly_fires:
                    tau_f = int(row["tau_future"])
                    expiry = d + pd.Timedelta(days=int(tau_f * 1.45))
                    direction_sign = 1.0 if row["direction"] == "bullish" else -1.0
                    base_conviction = min(weights.loc[idx] / w95, 1.0) if w95 > 0 else 0.0
                    disc = freshness_discount(tau_f, tau_star)
                    conviction = base_conviction * disc
                    tilt_value = direction_sign * conviction * 100.0
                    active_holds[sleeve].append({"expiry": expiry, "tilt": tilt_value})
                    discount_log.append({"sleeve": sleeve, "tau_future": tau_f, "tau_star": tau_star,
                                          "discount": disc, "base_conviction": base_conviction})

            if active_holds[sleeve]:
                tilt_today = max(active_holds[sleeve], key=lambda h: abs(h["tilt"]))["tilt"]
            else:
                tilt_today = 0.0
            raw_weight_df.at[d, sleeve] = NEUTRAL_WEIGHTS[sleeve] + tilt_today

    clipped_rows = [clip_and_renormalise(raw_weight_df.loc[d].to_dict()) for d in eval_dates]
    final_weight_df = pd.DataFrame(clipped_rows, index=eval_dates)

    lagged_weight_df = final_weight_df.shift(1)
    lagged_weight_df.iloc[0] = pd.Series(NEUTRAL_WEIGHTS)

    equity = simulate_portfolio(lagged_weight_df, prices, eval_dates)
    stats = compute_performance_stats(equity["equity"])

    n_holds_opened = {s: sum(1 for idx in fire_cache if fire_cache[idx][0]["Y"] == SLEEVES[s]
                              and any(v[1] for v in [fire_cache[idx][1][d] for d in eval_dates]))
                       for s in SLEEVES}

    disc_df = pd.DataFrame(discount_log)
    return {"equity_curve": equity, "stats": stats, "n_holds_opened": n_holds_opened, "discount_log": disc_df}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", required=True)
    args = parser.parse_args()

    print(f"\n{'='*72}\n  tau*-aware conviction discounting vs. original hold-to-horizon\n{'='*72}")
    print(f"  Sleeve tau* values: {SLEEVE_TAU_STAR}")
    print(f"  freshness(tau_future) at each sleeve's own tau*, 2x, 4x, 8x:")
    for s, ts in SLEEVE_TAU_STAR.items():
        vals = [freshness_discount(m * ts, ts) for m in (1, 2, 4, 8)]
        print(f"    {s:<10} tau*={ts:>3}d  " + "  ".join(f"{m}x={v:.2f}" for m, v in zip((1, 2, 4, 8), vals)))

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

    print("\n  --- Original hold-to-horizon (no tau* discount) ---")
    t0 = time.time()
    orig = run_hold_to_horizon(joint, prices, increments, thresholds, eval_dates)
    print(f"  {orig['stats']}")
    print(f"  Holds opened by sleeve: {orig['n_holds_opened']}")
    print(f"  Elapsed: {time.time()-t0:.0f}s")

    print("\n  --- tau*-aware hold-to-horizon (far-horizon holds discounted) ---")
    t0 = time.time()
    tau_aware = run_hold_to_horizon_tau_aware(joint, prices, increments, thresholds, eval_dates)
    print(f"  {tau_aware['stats']}")
    print(f"  Holds opened by sleeve: {tau_aware['n_holds_opened']}")
    dlog = tau_aware["discount_log"]
    if len(dlog):
        print(f"\n  Discount summary across {len(dlog)} holds opened:")
        print(dlog.groupby("sleeve")[["discount", "base_conviction"]].mean().to_string())
    print(f"  Elapsed: {time.time()-t0:.0f}s")

    print(f"\n{'='*72}\n  SUMMARY\n{'='*72}")
    summary = pd.DataFrame({
        "No-tilt benchmark": bench["stats"],
        "Original hold-to-horizon": orig["stats"],
        "tau*-aware hold-to-horizon": tau_aware["stats"],
    }).T
    print(summary.to_string())

    out = pd.DataFrame({
        "no_tilt": bench["equity_curve"]["equity"],
        "hth_original": orig["equity_curve"]["equity"],
        "hth_tau_aware": tau_aware["equity_curve"]["equity"],
    })
    out.to_csv("backtest_result_tau_aware_hth.csv")
    print("\n  Saved equity curves -> backtest_result_tau_aware_hth.csv")


if __name__ == "__main__":
    main()
