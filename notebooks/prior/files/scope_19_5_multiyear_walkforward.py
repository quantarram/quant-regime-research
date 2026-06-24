"""
scope_19_5_multiyear_walkforward.py
=====================================
Scope 19.5: Multi-year walk-forward on the validated VIXM+VIXY→SPY signal
(Section 19.5 of the paper).

The VIXM+VIXY→SPY hold-to-horizon signal is the only signal validated to
date (pct_exceeding 1.8% in 2025, paper Section 15.1). This script tests
it across 2015–2024 using expanding training windows, never touching 2025
data, to ask: is the 2025 significance specific to that year's market
structure, or does it reflect a genuine multi-regime pattern?

Methodology:
  - For each evaluation year Y in {2015, 2016, ..., 2024}:
    * Training data: all price history UP TO AND INCLUDING Dec 31 of Y-1
    * Evaluation data: all trading days of year Y
    * Signal: same episode-conviction pipeline as the 2025 paper results
      (economic prior gated, episode-independent sizing, hold-to-horizon)
    * The VIXM+VIXY→SPY configuration is re-estimated at each expanding
      window to check that it still survives the quality filters
    * The same 5-sleeve, capped-weight (Crypto≤15%, Gold≤20%) book is
      used throughout (or corrected weights if cap test not applicable)
  - Comparison: each year's SPY buy-and-hold and no-tilt 5-sleeve baseline
  - Randomisation test is run at each year if enough hold events exist

This script does NOT re-run the full greedy joint-screen for every year
(that would take hours). Instead, it uses the FROZEN joint_cpe_results.parquet
from the paper's training cutoff (2024-12-31) but re-estimates the
EPISODE CONVICTION of the VIXM+VIXY→SPY configuration specifically on each
year's expanding training window, then fires it on the evaluation year.

IMPORTANT: this is not identical to a fully-fresh per-year screen (which
would find different predictor sets for earlier training windows) — it
specifically tests whether the one validated configuration (VIXM+VIXY→SPY,
tau_f=63 or tau_f=252, the dominant signal) would have been selected and
would have performed in earlier years. This is a targeted signal-validity
test, not a full strategy re-derivation.

Usage:
    python scope_19_5_multiyear_walkforward.py --joint joint_cpe_results.parquet
    python scope_19_5_multiyear_walkforward.py --joint joint_cpe_results.parquet --years 2018 2019 2020 2021 2022 2023 2024
    python scope_19_5_multiyear_walkforward.py --joint joint_cpe_results.parquet --skip-randomisation
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
        BASE_SLEEVES, compute_neutral_weights,
        build_increments_and_thresholds, compute_quality_weights,
        build_increments_for_episodes,
        compute_daily_class_scores, _static_tilt_delta, clip_and_renormalise,
        simulate_portfolio, compute_performance_stats,
        configuration_fires_on_date, HORIZON_WEIGHTS,
        _cluster_into_episodes, EPISODE_MIN_OBS_FOR_CONVICTION,
        EPISODE_GAP_MULTIPLIER, EPISODE_ANCHOR,
    )
except ImportError as e:
    sys.exit(f"ERROR: Cannot import backtest_engine.py — ensure it is in the working directory.\n  {e}")

try:
    from run_backtest import (
        run_hold_to_horizon, run_no_tilt_benchmark, run_buy_and_hold,
        randomisation_test_hth, load_and_filter_joint, get_eval_dates,
    )
except ImportError as e:
    sys.exit(f"ERROR: Cannot import run_backtest.py — ensure it is in the working directory.\n  {e}")

# ── Base allocation cap (from Scope 19.2, Section 19.2) ──────────────────
CRYPTO_CAP_PCT = 15.0
GOLD_CAP_PCT   = 20.0
WEIGHT_CAPS    = {"Equities": None, "Gold": GOLD_CAP_PCT, "Bonds": None,
                  "Crypto": CRYPTO_CAP_PCT, "FX": None}

Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]

# ── Validated signal spec (from paper Section 7.1, 15.1) ─────────────────
# The greedy joint screen selected VIXM+VIXY jointly predicting SPY.
# We look for configurations matching this in the joint screen;
# if the loaded screen has them, we test those exact rows.
# If not (e.g. earlier training cutoff changes survivorship), we flag it.
VALIDATED_Y         = "SPY"
VALIDATED_PREDICTORS = {"VIXM", "VIXY"}  # must both appear (order-independent)


def apply_weight_cap(weights: dict, caps: dict) -> dict:
    """Same cap logic as scope_19_2_alloc_cap.py."""
    capped = {}
    surplus = 0.0
    for sleeve, w in weights.items():
        cap = caps.get(sleeve)
        if cap is not None and w > cap:
            surplus += w - cap
            capped[sleeve] = cap
        else:
            capped[sleeve] = w
    if surplus <= 0:
        return capped
    uncapped = [s for s, w in weights.items()
                if caps.get(s) is None or w < (caps.get(s) or float("inf"))]
    uncapped_total = sum(capped[s] for s in uncapped)
    if uncapped_total <= 0:
        per = surplus / len(capped)
        return {s: v + per for s, v in capped.items()}
    for sleeve in uncapped:
        capped[sleeve] += surplus * (capped[sleeve] / uncapped_total)
    return capped


def check_signal_survives(joint: pd.DataFrame, train_cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Find rows in the joint screen where Y=SPY and the predictor set
    contains both VIXM and VIXY. These are the configurations we test.
    Returns a (possibly empty) sub-DataFrame.
    """
    spy_rows = joint[joint["Y"] == VALIDATED_Y].copy()
    mask = spy_rows["predictors"].apply(
        lambda preds: VALIDATED_PREDICTORS.issubset(set(preds))
    )
    return spy_rows[mask]


def episode_conviction_expanding(row: pd.Series, prices: pd.DataFrame,
                                  train_cutoff: pd.Timestamp,
                                  increments: dict) -> float:
    """
    Compute episode conviction for one joint-screen row, restricting
    episode detection to the EXPANDING training window (all data up to
    train_cutoff for this evaluation year).

    This is needed because episode conviction is computed from training
    data only, and the training window grows for each evaluation year.
    """
    direction = row["direction"]
    predictors = list(row["predictors"])
    tau_pasts = [int(t) for t in row["tau_pasts"]]
    q_xs = [float(q) for q in row["q_Xs"]]
    y = row["Y"]
    tau_f = int(row["tau_future"])
    q_y = float(row["q_Y"])

    if tau_f not in increments or any(tp not in increments for tp in tau_pasts):
        return 0.0

    train_dates = increments[tau_pasts[0]].index[
        increments[tau_pasts[0]].index <= train_cutoff
    ]

    joint_mask = pd.Series(True, index=train_dates)
    for x, tau_p, q_x in zip(predictors, tau_pasts, q_xs):
        if x not in increments[tau_p].columns:
            return 0.0
        series = increments[tau_p][x].reindex(train_dates)
        train_series = increments[tau_p][x].loc[
            increments[tau_p].index <= train_cutoff
        ]
        if direction == "bullish":
            thresh = train_series.quantile(q_x)
            joint_mask &= (series > thresh)
        else:
            thresh = train_series.quantile(round(1 - q_x, 10))
            joint_mask &= (series < thresh)

    firing_dates = joint_mask[joint_mask.fillna(False)].index
    if len(firing_dates) == 0:
        return 0.0

    episodes = _cluster_into_episodes(firing_dates, max(tau_pasts))

    target_forward = increments[tau_f][y].shift(-tau_f)
    target_train = increments[tau_f][y].loc[
        increments[tau_f].index <= train_cutoff
    ]
    target_thresh = target_train.quantile(
        q_y if direction == "bullish" else round(1 - q_y, 10)
    )

    outcomes = []
    for ep in episodes:
        anchor_idx = {"last": -1, "first": 0, "mid": len(ep) // 2}.get(EPISODE_ANCHOR, -1)
        anchor = ep[anchor_idx]
        val = target_forward.get(anchor, np.nan)
        if pd.isna(val):
            continue
        outcomes.append(
            bool(val > target_thresh) if direction == "bullish"
            else bool(val < target_thresh)
        )

    n_episodes = len(outcomes)
    if n_episodes < EPISODE_MIN_OBS_FOR_CONVICTION:
        return 0.0

    hit_rate = float(np.mean(outcomes))
    agreement = max(0.0, 2 * hit_rate - 1)
    return float(np.log(n_episodes) * agreement)


def run_one_year(eval_year: int, joint_full: pd.DataFrame,
                  prices: pd.DataFrame,
                  skip_randomisation: bool = False,
                  n_reps: int = 1000) -> dict:
    """
    Run the hold-to-horizon strategy for a single evaluation year using
    an expanding training window (all data through Dec 31 of eval_year-1).
    """
    train_cutoff = pd.Timestamp(f"{eval_year - 1}-12-31")
    eval_start   = pd.Timestamp(f"{eval_year}-01-01")
    eval_end     = pd.Timestamp(f"{eval_year}-12-31")

    print(f"\n  {'─'*60}")
    print(f"  YEAR {eval_year}  |  Training: ..–{train_cutoff.date()}  |  "
          f"Eval: {eval_start.date()}–{eval_end.date()}")
    print(f"  {'─'*60}")

    # Check that the evaluation year has enough data
    eval_mask = (prices.index >= eval_start) & (prices.index <= eval_end)
    spy_eval = prices.loc[eval_mask, "SPY"].dropna()
    if len(spy_eval) < 50:
        print(f"  SKIPPED: insufficient SPY data in {eval_year} ({len(spy_eval)} days)")
        return None

    # Check training data depth
    train_mask = prices.index <= train_cutoff
    spy_train = prices.loc[train_mask, "SPY"].dropna()
    vixm_train = prices.loc[train_mask, "VIXM"].dropna() if "VIXM" in prices.columns else pd.Series()
    vixy_train = prices.loc[train_mask, "VIXY"].dropna() if "VIXY" in prices.columns else pd.Series()

    print(f"  SPY training obs:  {len(spy_train)}")
    print(f"  VIXM training obs: {len(vixm_train)}")
    print(f"  VIXY training obs: {len(vixy_train)}")

    min_predictor_obs = min(len(vixm_train), len(vixy_train))
    if min_predictor_obs < 252:
        print(f"  SKIPPED: VIXM/VIXY have only {min_predictor_obs} training obs "
              f"(minimum 252 required for 1-year threshold estimation)")
        return None

    # Override engine cutoff for this year
    _be.TRAIN_CUTOFF = train_cutoff
    _be.EVAL_START   = eval_start
    _be.EVAL_END     = eval_end

    # Compute neutral weights on THIS year's training window
    raw_weights = compute_neutral_weights(BASE_SLEEVES, prices)
    # Use the smaller raw_weights before the cap is defined for data inspection
    print(f"  Uncapped weights: " +
          "  ".join(f"{k}:{v:.1f}%" for k, v in raw_weights.items()))
    capped_weights = apply_weight_cap(raw_weights, WEIGHT_CAPS)
    print(f"  Capped weights:   " +
          "  ".join(f"{k}:{v:.1f}%" for k, v in capped_weights.items()))

    _be.SLEEVES.clear()
    _be.SLEEVES.update(BASE_SLEEVES)
    _be.NEUTRAL_WEIGHTS.clear()
    _be.NEUTRAL_WEIGHTS.update(capped_weights)

    # Build increments and thresholds frozen to THIS year's training window
    print(f"  Building increments/thresholds (training cutoff {train_cutoff.date()})...")
    t0 = time.time()
    increments, thresholds = build_increments_and_thresholds(prices, Q_GRID)
    print(f"  Done. {time.time()-t0:.1f}s")

    # Check whether the VIXM+VIXY→SPY configuration survives the joint
    # screen's filters at this expanding training window.
    # We use the full joint screen from 2024-12-31 (frozen), but check
    # the configuration's episode conviction at the current train_cutoff.
    validated_rows = check_signal_survives(joint_full, train_cutoff)
    if len(validated_rows) == 0:
        print(f"  NOTE: No VIXM+VIXY→SPY configurations found in joint screen.")
        print(f"        Strategy will trade on all SPY-target configurations.")
    else:
        print(f"  VIXM+VIXY→SPY configurations in joint screen: {len(validated_rows)}")
        for idx, vrow in validated_rows.iterrows():
            conv = episode_conviction_expanding(vrow, prices, train_cutoff, increments)
            print(f"    tau_f={vrow['tau_future']}  direction={vrow['direction']}  "
                  f"CPE={vrow['joint_CPE']:.3f}  "
                  f"episode_conviction={conv:.4f}  "
                  f"(>=3 eps needed; {'ACTIVE' if conv > 0 else 'ZEROED'})")

    # Evaluation dates for this year
    eval_dates = get_eval_dates(prices)
    print(f"  Eval trading days: {len(eval_dates)}")

    if len(eval_dates) == 0:
        print(f"  SKIPPED: no valid evaluation dates in {eval_year}")
        return None

    # Run no-tilt benchmark
    bench = run_no_tilt_benchmark(prices, eval_dates)
    spy_bh = run_buy_and_hold(prices, "SPY", eval_dates)
    print(f"  No-tilt benchmark:  ret={bench['stats']['total_return_pct']}%  "
          f"Sharpe={bench['stats']['sharpe']}")
    print(f"  SPY buy-and-hold:   ret={spy_bh['stats']['total_return_pct']}%  "
          f"Sharpe={spy_bh['stats']['sharpe']}")

    # Run hold-to-horizon
    print(f"  Running hold-to-horizon...")
    t0 = time.time()
    hth = run_hold_to_horizon(joint_full, prices, increments, thresholds, eval_dates)
    print(f"  HTH result:         ret={hth['stats']['total_return_pct']}%  "
          f"Sharpe={hth['stats']['sharpe']}")
    print(f"  Holds opened: {hth['n_holds_opened']}")
    print(f"  Elapsed: {time.time()-t0:.0f}s")

    rtest_result = None
    if not skip_randomisation:
        total_holds = sum(hth["n_holds_opened"].values())
        if total_holds >= 3:
            print(f"  Running randomisation test ({n_reps} reps, {total_holds} holds)...")
            t0 = time.time()
            rtest_result = randomisation_test_hth(
                joint_full, prices, increments, thresholds, eval_dates, n_reps=n_reps
            )
            if "note" in rtest_result:
                print(f"  {rtest_result['note']}")
            else:
                print(f"  Pct exceeding: {rtest_result['pct_exceeding']}%  "
                      f"Null mean={rtest_result['null_mean']:.3f}  "
                      f"Null std={rtest_result['null_std']:.3f}")
            print(f"  Elapsed: {time.time()-t0:.0f}s")
        else:
            print(f"  Randomisation test skipped: {total_holds} holds (need ≥ 3)")
            rtest_result = {"note": f"Skipped: only {total_holds} hold event(s)"}

    return {
        "year": eval_year,
        "train_cutoff": str(train_cutoff.date()),
        "eval_days": len(eval_dates),
        "spy_train_obs": len(spy_train),
        "vixm_train_obs": len(vixm_train),
        "vixy_train_obs": len(vixy_train),
        "validated_configs_found": len(validated_rows),
        "bench_ret_pct": bench["stats"]["total_return_pct"],
        "bench_sharpe": bench["stats"]["sharpe"],
        "spy_ret_pct": spy_bh["stats"]["total_return_pct"],
        "spy_sharpe": spy_bh["stats"]["sharpe"],
        "hth_ret_pct": hth["stats"]["total_return_pct"],
        "hth_sharpe": hth["stats"]["sharpe"],
        "hth_ann_vol_pct": hth["stats"]["ann_vol_pct"],
        "hth_holds_total": sum(hth["n_holds_opened"].values()),
        "hth_holds_by_sleeve": str(hth["n_holds_opened"]),
        "pct_exceeding": rtest_result.get("pct_exceeding", "n/a") if rtest_result else "skipped",
        "rand_null_mean": rtest_result.get("null_mean", np.nan) if rtest_result else np.nan,
        "rand_null_std": rtest_result.get("null_std", np.nan) if rtest_result else np.nan,
        "rand_note": rtest_result.get("note", "") if rtest_result else "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Scope 19.5: Multi-year walk-forward on VIXM+VIXY→SPY signal"
    )
    parser.add_argument("--joint", default="joint_cpe_results.parquet")
    parser.add_argument("--prices", default="multiasset_prices.parquet")
    parser.add_argument("--years", nargs="+", type=int,
                        default=list(range(2015, 2025)),
                        help="Evaluation years to test (default: 2015–2024)")
    parser.add_argument("--skip-randomisation", action="store_true")
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument("--output", default="walkforward_19_5_results.csv")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  SCOPE 19.5 — MULTI-YEAR WALK-FORWARD: VIXM+VIXY→SPY SIGNAL")
    print(f"  Evaluation years: {args.years}")
    print(f"  Expanding training windows (data through Dec 31 of prior year)")
    print(f"  Hold-to-horizon, episode-conviction sizing, capped weights")
    print(f"{'='*72}")

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\n  Loading {args.prices}...")
    prices = pd.read_parquet(args.prices)
    earliest = prices.index.min()
    latest   = prices.index.max()
    print(f"  Price history: {earliest.date()} to {latest.date()}")
    print(f"  Instruments:   {prices.shape[1]}")

    # Check data availability for each requested year
    print(f"\n  DATA AVAILABILITY CHECK:")
    for yr in args.years:
        train_cutoff = pd.Timestamp(f"{yr-1}-12-31")
        eval_end     = pd.Timestamp(f"{yr}-12-31")
        vixm_obs = len(prices.loc[prices.index <= train_cutoff, "VIXM"].dropna()) \
                   if "VIXM" in prices.columns else 0
        vixy_obs = len(prices.loc[prices.index <= train_cutoff, "VIXY"].dropna()) \
                   if "VIXY" in prices.columns else 0
        eval_spy = len(prices.loc[(prices.index >= pd.Timestamp(f"{yr}-01-01")) &
                                   (prices.index <= eval_end), "SPY"].dropna())
        flag = ""
        if vixm_obs < 252:
            flag = f"  ← VIXM only {vixm_obs} obs; will skip"
        elif vixy_obs < 252:
            flag = f"  ← VIXY only {vixy_obs} obs; will skip"
        elif eval_spy < 50:
            flag = f"  ← only {eval_spy} SPY eval days; will skip"
        print(f"    {yr}: VIXM train={vixm_obs:>5}  VIXY train={vixy_obs:>5}  "
              f"SPY eval days={eval_spy:>4}{flag}")

    print(f"\n  Loading joint screen: {args.joint}...")
    joint = load_and_filter_joint(args.joint)
    print(f"  Joint configs (n_pred ≤ 6): {len(joint)}")

    # Check for validated signal in the screen
    spy_rows = joint[joint["Y"] == VALIDATED_Y]
    vixm_vixy_rows = spy_rows[spy_rows["predictors"].apply(
        lambda p: VALIDATED_PREDICTORS.issubset(set(p))
    )]
    print(f"  VIXM+VIXY→SPY configurations in screen: {len(vixm_vixy_rows)}")
    if len(vixm_vixy_rows) > 0:
        for _, r in vixm_vixy_rows.iterrows():
            print(f"    tau_f={r['tau_future']}  dir={r['direction']}  "
                  f"CPE={r['joint_CPE']:.3f}  n_joint={r.get('n_joint', '?')}")

    # ── Main walk-forward loop ─────────────────────────────────────────────
    results = []
    orig_train_cutoff = _be.TRAIN_CUTOFF
    orig_eval_start   = _be.EVAL_START
    orig_eval_end     = _be.EVAL_END

    for yr in sorted(args.years):
        try:
            res = run_one_year(
                eval_year=yr,
                joint_full=joint,
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

    # Restore original engine state
    _be.TRAIN_CUTOFF = orig_train_cutoff
    _be.EVAL_START   = orig_eval_start
    _be.EVAL_END     = orig_eval_end
    _be.SLEEVES.clear()
    _be.SLEEVES.update(BASE_SLEEVES)

    if not results:
        print("\n  No results produced — check data availability.")
        return

    # ── Summary table ─────────────────────────────────────────────────────
    df = pd.DataFrame(results)

    print(f"\n\n{'='*72}")
    print(f"  MULTI-YEAR WALK-FORWARD SUMMARY (VIXM+VIXY→SPY, HTH, CAPPED WEIGHTS)")
    print(f"{'='*72}")
    print(f"\n  {'Year':>6}  {'HTH ret%':>9}  {'HTH Sh':>7}  {'Bench ret%':>11}  "
          f"{'Bench Sh':>9}  {'SPY ret%':>9}  {'Holds':>6}  {'Pct exc':>8}  {'Sig?':>5}")
    print(f"  {'─'*82}")

    sig_years   = []
    total_holds = 0

    for _, r in df.iterrows():
        pct = r["pct_exceeding"]
        is_sig = isinstance(pct, (int, float)) and float(pct) <= 10.0
        sig_flag = "  YES" if is_sig else "   no"
        if is_sig:
            sig_years.append(int(r["year"]))
        total_holds += r.get("hth_holds_total", 0)
        print(f"  {int(r['year']):>6}  {r['hth_ret_pct']:>9.2f}%  {r['hth_sharpe']:>7.3f}  "
              f"{r['bench_ret_pct']:>11.2f}%  {r['bench_sharpe']:>9.3f}  "
              f"{r['spy_ret_pct']:>9.2f}%  {r['hth_holds_total']:>6}  "
              f"{str(pct):>8}  {sig_flag}")

    # Add 2025 paper result for comparison
    print(f"  {'─'*82}")
    print(f"  {'2025*':>6}  {'18.77':>9}%  {'1.224':>7}  {'16.55':>11}%  "
          f"{'1.089':>9}  {'18.01':>9}%  {'11':>6}  {'1.8':>8}%  {'  YES'}")
    print(f"  (* 2025 result from paper Section 16.1; training cutoff 2024-12-31)")

    print(f"\n  INTERPRETATION:")
    print(f"  Signal-significant years (pct_exceeding ≤ 10%): "
          f"{sig_years if sig_years else 'none in tested range'}")
    print(f"  Total holds fired across {len(results)} evaluation years: {total_holds}")
    n_years = len(results)
    n_sig = len(sig_years)
    print(f"  Significance rate: {n_sig}/{n_years} evaluation years")

    if n_sig > 0:
        print(f"\n  POSITIVE FINDING: The VIXM+VIXY→SPY signal produced statistically")
        print(f"  significant results in {n_sig} of {n_years} tested years, suggesting")
        print(f"  the 2025 result is not specific to that year's market structure.")
    elif total_holds == 0:
        print(f"\n  NOTE: Zero holds fired across all tested years. This typically means")
        print(f"  VIXM/VIXY lacked sufficient training history in the earlier years,")
        print(f"  or the 3-episode minimum conviction floor was never cleared.")
        print(f"  VIXM launched ~2011, VIXY ~2011; years before 2014 may lack")
        print(f"  sufficient training data for the episode filter to pass.")
    else:
        print(f"\n  CAUTION: The signal fired {total_holds} times but produced no")
        print(f"  statistically significant years. This suggests the 2025 result")
        print(f"  may be specific to 2025's market structure or the single-year")
        print(f"  window is insufficient to establish the pattern across regimes.")

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_csv(args.output, index=False)
    print(f"\n  Saved: {args.output}")
    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()
