"""
scope_episode_floor.py
=======================
Section 20.6: Effect of raising the minimum episode floor.

Tests three combinations:
  A. 2024-trained screen (joint_cpe_results.parquet), floor=5
  B. 2024-trained screen (joint_cpe_results.parquet), floor=6
  C. 2014-trained screen (joint_cpe_results_train2014_final.parquet), floor=5

For each:
  - Filter joint screen to configs with n_episodes >= floor
  - Report surviving configs per sleeve
  - Run hold-to-horizon backtest over the appropriate eval window
  - Run randomisation test
  - Report year-by-year breakdown

Eval windows:
  A & B: 2025-01-01 to 2025-12-31  (paper's primary eval window)
  C:     2015-01-01 to 2025-12-31  (single-screen continuous eval)

Key question for each:
  A/B: Does the paper's 2025 result (pct_exc 1.8%) survive a stricter floor?
       VIXM+VIXY→SPY has 6 episodes at 2024 cutoff → survives floor=5 and 6
  C:   Does floor=5 reduce the 785-hold saturation in the 2014→2025 test?
       VIXM+VIXY→SPY has 3 episodes at 2014 cutoff → zeroed out at floor=5

Usage:
    python scope_episode_floor.py
    python scope_episode_floor.py --skip-randomisation
    python scope_episode_floor.py --n-reps 1000
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
        clip_and_renormalise, simulate_portfolio, compute_performance_stats,
        configuration_fires_on_date,
    )
except ImportError as e:
    sys.exit(f"ERROR: Cannot import backtest_engine.py\n  {e}")

try:
    from run_backtest import (
        run_no_tilt_benchmark, run_buy_and_hold,
        load_and_filter_joint, get_eval_dates,
    )
except ImportError as e:
    sys.exit(f"ERROR: Cannot import run_backtest.py\n  {e}")

Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]

# Base allocation caps from Scope 19.2
CRYPTO_CAP_PCT = 15.0
GOLD_CAP_PCT   = 20.0
WEIGHT_CAPS    = {"Equities": None, "Gold": GOLD_CAP_PCT,
                  "Bonds": None, "Crypto": CRYPTO_CAP_PCT, "FX": None}

# The three test configurations
TESTS = [
    {
        "label":       "A",
        "description": "2024-trained screen, episode floor = 5",
        "joint_file":  "joint_cpe_results.parquet",
        "train_cutoff": pd.Timestamp("2024-12-31"),
        "eval_start":  pd.Timestamp("2025-01-01"),
        "eval_end":    pd.Timestamp("2025-12-31"),
        "floor":       5,
        "note":        "Paper's primary eval window. VIXM+VIXY has 6 episodes → survives.",
    },
    {
        "label":       "B",
        "description": "2024-trained screen, episode floor = 6",
        "joint_file":  "joint_cpe_results.parquet",
        "train_cutoff": pd.Timestamp("2024-12-31"),
        "eval_start":  pd.Timestamp("2025-01-01"),
        "eval_end":    pd.Timestamp("2025-12-31"),
        "floor":       6,
        "note":        "Strictest floor. VIXM+VIXY has exactly 6 episodes → borderline.",
    },
    {
        "label":       "C",
        "description": "2014-trained screen, episode floor = 5",
        "joint_file":  "joint_cpe_results_train2014_final.parquet",
        "train_cutoff": pd.Timestamp("2014-12-31"),
        "eval_start":  pd.Timestamp("2015-01-01"),
        "eval_end":    pd.Timestamp("2025-12-31"),
        "floor":       5,
        "note":        "Tests whether floor=5 fixes 785-hold saturation. "
                       "VIXM+VIXY has 3 episodes at 2014 cutoff → zeroed out.",
    },
]


def apply_weight_cap(weights, caps):
    capped, surplus = {}, 0.0
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
    for s in uncapped:
        capped[s] += surplus * (capped[s] / uncapped_total)
    return capped


def apply_episode_floor(joint: pd.DataFrame, floor: int) -> pd.DataFrame:
    """
    Filter joint screen to configs with n_episodes >= floor.
    Also zeros out episode_conviction for configs below floor
    (defensive: should already be 0 from engine, but be explicit).
    """
    if "n_episodes" not in joint.columns:
        print(f"  WARNING: n_episodes column not found — cannot apply floor={floor}")
        print(f"  Available columns: {list(joint.columns)}")
        return joint

    before = len(joint)
    filtered = joint[joint["n_episodes"] >= floor].copy()
    after = len(filtered)
    print(f"  Episode floor={floor}: {before} → {after} configs "
          f"({before-after} removed, {after/before*100:.1f}% retained)")
    return filtered


def screen_diagnostics(joint: pd.DataFrame, floor: int,
                        train_cutoff: pd.Timestamp) -> None:
    """Print diagnostic breakdown of surviving configs."""
    sleeve_proxies = set(BASE_SLEEVES.values())
    print(f"\n  Screen diagnostics (floor={floor}, "
          f"train cutoff {train_cutoff.date()}):")

    for proxy in sorted(sleeve_proxies):
        sub = joint[joint["Y"] == proxy]
        if len(sub) == 0:
            print(f"    {proxy:<10}: 0 configs")
            continue
        active = sub[sub.get("episode_conviction", pd.Series(
            dtype=float)).reindex(sub.index, fill_value=0) > 0] \
            if "episode_conviction" in sub.columns else sub
        vv = sub[sub["predictors"].apply(
            lambda p: {"VIXM", "VIXY"}.issubset(set(p))
        )]
        ep_range = f"{sub['n_episodes'].min()}–{sub['n_episodes'].max()}" \
                   if "n_episodes" in sub.columns else "?"
        print(f"    {proxy:<10}: {len(sub):>3} configs  "
              f"ep_range={ep_range}  "
              f"VV+VIXY configs: {len(vv)}")
        if len(vv) > 0 and "episode_conviction" in vv.columns:
            for _, r in vv.iterrows():
                print(f"      VV: tau_f={r['tau_future']}  "
                      f"CPE={r['joint_CPE']:.3f}  "
                      f"n_ep={r['n_episodes']}  "
                      f"conv={r['episode_conviction']:.4f}  "
                      f"{'ACTIVE' if r['episode_conviction'] > 0 else 'ZEROED'}")


def run_hth(joint: pd.DataFrame, prices: pd.DataFrame,
            eval_dates: pd.DatetimeIndex,
            neutral_weights: dict) -> dict:
    """Hold-to-horizon backtest. Reuses run_backtest's logic via engine."""
    sleeve_proxies = set(BASE_SLEEVES.values())
    tradeable = joint[joint["Y"].isin(sleeve_proxies)].copy()

    increments, thresholds = build_increments_and_thresholds(prices, Q_GRID)
    weights = compute_quality_weights(joint, prices,
                                       precomputed_increments=increments)

    # Normalise conviction weights against tradeable universe
    trad_w = weights[weights.index.isin(tradeable.index)]
    w95 = float(np.percentile(trad_w, 95)) if len(trad_w) > 0 and trad_w.max() > 0 else 1.0
    if w95 <= 0:
        w95 = trad_w.max() if len(trad_w) > 0 else 1.0

    raw = pd.DataFrame({s: neutral_weights[s] for s in BASE_SLEEVES},
                       index=eval_dates, dtype=float)
    active_holds = {s: [] for s in BASE_SLEEVES}
    hold_events  = []
    fire_cache   = {}

    # Pre-compute firing for all tradeable configs
    for sleeve, proxy in BASE_SLEEVES.items():
        sub = tradeable[tradeable["Y"] == proxy]
        print(f"  [{sleeve:<10}] {len(sub)} configs × "
              f"{len(eval_dates)} days...", end=" ", flush=True)
        t0 = time.time()
        for idx, row in sub.iterrows():
            fires, prev = {}, False
            for d in eval_dates:
                today = configuration_fires_on_date(row, d, increments, thresholds)
                fires[d] = (today, today and not prev)
                prev = today
            fire_cache[idx] = (row, fires)
        print(f"{time.time()-t0:.0f}s")

    # Build weight path
    for sleeve, proxy in BASE_SLEEVES.items():
        rel = [idx for idx, (row, _) in fire_cache.items()
               if row["Y"] == proxy]
        for d in eval_dates:
            active_holds[sleeve] = [
                h for h in active_holds[sleeve] if h["expiry"] > d
            ]
            for idx in rel:
                row, fires = fire_cache[idx]
                fired_today, newly = fires[d]
                if newly:
                    tau_f   = int(row["tau_future"])
                    expiry  = d + pd.Timedelta(days=int(tau_f * 1.45))
                    sign    = 1.0 if row["direction"] == "bullish" else -1.0
                    w_idx   = weights.loc[idx] if idx in weights.index else 0.0
                    conv    = min(w_idx / w95, 1.0) if w95 > 0 else 0.0
                    tilt    = sign * conv * 100.0
                    active_holds[sleeve].append({"expiry": expiry, "tilt": tilt})
                    hold_events.append({
                        "sleeve":       sleeve,
                        "entry_date":   d,
                        "duration_days": int(tau_f * 1.45),
                        "tilt":         tilt,
                        "tau_future":   tau_f,
                        "Y":            proxy,
                        "predictors":   list(row["predictors"]),
                        "n_episodes":   int(row.get("n_episodes", 0)),
                        "episode_conviction": float(
                            row.get("episode_conviction", np.nan)),
                        "joint_CPE":    float(row["joint_CPE"]),
                    })
            if active_holds[sleeve]:
                tilt_today = max(active_holds[sleeve],
                                 key=lambda h: abs(h["tilt"]))["tilt"]
            else:
                tilt_today = 0.0
            raw.at[d, sleeve] = neutral_weights[sleeve] + tilt_today

    clipped = pd.DataFrame(
        [clip_and_renormalise(raw.loc[d].to_dict()) for d in eval_dates],
        index=eval_dates)
    lagged = clipped.shift(1)
    lagged.iloc[0] = pd.Series(neutral_weights)

    equity = simulate_portfolio(lagged, prices, eval_dates)
    stats  = compute_performance_stats(equity["equity"])
    return {"equity": equity, "stats": stats,
            "hold_events": hold_events,
            "increments": increments, "thresholds": thresholds}


def randomisation_test(hold_events: list, prices: pd.DataFrame,
                        eval_dates: pd.DatetimeIndex,
                        neutral_weights: dict,
                        actual_sharpe: float,
                        n_reps: int = 1000) -> dict:
    """Shuffle hold entry dates across eval window, preserving duration/magnitude."""
    if not hold_events:
        return {"note": "No hold events"}

    n_eval    = len(eval_dates)
    eval_list = list(eval_dates)
    rng       = np.random.default_rng(42)
    null_sharpes = []
    t0 = time.time()

    print(f"  Randomisation test: {len(hold_events)} holds, "
          f"{n_eval} days, {n_reps} reps...")

    for rep in range(n_reps):
        raw = pd.DataFrame({s: neutral_weights[s] for s in BASE_SLEEVES},
                           index=eval_dates, dtype=float)
        active = {s: [] for s in BASE_SLEEVES}
        shuffled = [{**ev, "entry_date": eval_list[int(rng.integers(0, n_eval))]}
                    for ev in hold_events]

        for d in eval_dates:
            for s in BASE_SLEEVES:
                active[s] = [h for h in active[s] if h["expiry"] > d]
            for ev in shuffled:
                if ev["entry_date"] == d:
                    s      = ev["sleeve"]
                    expiry = d + pd.Timedelta(days=ev["duration_days"])
                    active[s].append({"expiry": expiry, "tilt": ev["tilt"]})
            for s in BASE_SLEEVES:
                tilt = max(active[s], key=lambda h: abs(h["tilt"]))["tilt"] \
                       if active[s] else 0.0
                raw.at[d, s] = neutral_weights[s] + tilt

        clipped = pd.DataFrame(
            [clip_and_renormalise(raw.loc[d].to_dict()) for d in eval_dates],
            index=eval_dates)
        lagged = clipped.shift(1)
        lagged.iloc[0] = pd.Series(neutral_weights)
        eq = simulate_portfolio(lagged, prices, eval_dates)
        sh = compute_performance_stats(eq["equity"])["sharpe"]
        if not np.isnan(sh):
            null_sharpes.append(sh)

        if (rep + 1) % 200 == 0:
            print(f"    {rep+1}/{n_reps}  elapsed {time.time()-t0:.0f}s",
                  end="\r")

    print()
    null = np.array(null_sharpes)
    pct  = float((null >= actual_sharpe).mean() * 100)
    return {
        "actual_sharpe": actual_sharpe,
        "null_mean":     float(null.mean()),
        "null_std":      float(null.std()),
        "pct_exceeding": round(pct, 1),
        "n_reps":        len(null),
        "n_holds":       len(hold_events),
    }


def yearly_breakdown(equity_series: pd.Series, eval_start, eval_end,
                     hold_events: list) -> pd.DataFrame:
    """Return year-by-year return and hold count."""
    rows = []
    for yr in range(eval_start.year, eval_end.year + 1):
        ys = pd.Timestamp(f"{yr}-01-01")
        ye = pd.Timestamp(f"{yr}-12-31")
        sub = equity_series.loc[(equity_series.index >= ys) &
                                 (equity_series.index <= ye)].dropna()
        if len(sub) < 2:
            continue
        ret = round((sub.iloc[-1] / sub.iloc[0] - 1) * 100, 2)
        r   = sub.pct_change().dropna()
        sh  = round((r.mean() / r.std()) * np.sqrt(252), 3) \
              if r.std() > 0 else np.nan
        n_h = sum(1 for ev in hold_events if ev["entry_date"].year == yr)
        rows.append({"year": yr, "ret_pct": ret, "sharpe": sh, "holds": n_h})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Episode floor sensitivity test for Section 20.6"
    )
    parser.add_argument("--prices",             default="multiasset_prices.parquet")
    parser.add_argument("--skip-randomisation", action="store_true")
    parser.add_argument("--n-reps",   type=int, default=1000)
    parser.add_argument("--output",             default="episode_floor_results.csv")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  EPISODE FLOOR SENSITIVITY TEST — Section 20.6")
    print(f"  Three combinations: A (2024/floor=5), B (2024/floor=6), "
          f"C (2014/floor=5)")
    print(f"{'='*72}")

    prices = pd.read_parquet(args.prices)
    print(f"\n  Price history: {prices.index.min().date()} to "
          f"{prices.index.max().date()}")

    all_results = []
    orig_cutoff = _be.TRAIN_CUTOFF
    orig_start  = _be.EVAL_START
    orig_end    = _be.EVAL_END

    for test in TESTS:
        print(f"\n{'='*72}")
        print(f"  TEST {test['label']}: {test['description']}")
        print(f"  {test['note']}")
        print(f"{'='*72}")

        # Load and check joint screen
        if not os.path.exists(test["joint_file"]):
            print(f"  ERROR: {test['joint_file']} not found — skipping")
            continue

        joint_raw = pd.read_parquet(test["joint_file"])
        joint_raw = joint_raw[joint_raw["n_predictors"] <= 6].copy()
        print(f"\n  Joint screen loaded: {len(joint_raw)} configs")

        # Check n_episodes availability
        if "n_episodes" not in joint_raw.columns:
            print(f"  WARNING: n_episodes not in screen — was this built with "
                  f"the episode-conviction pipeline?")
            print(f"  Columns: {list(joint_raw.columns)}")
            print(f"  Skipping this test.")
            continue

        ep_dist = joint_raw["n_episodes"].value_counts().sort_index()
        print(f"\n  Episode distribution (top 10 values):")
        for ep, cnt in ep_dist.head(10).items():
            print(f"    n_episodes={ep}: {cnt} configs")

        # Apply floor
        joint = apply_episode_floor(joint_raw, test["floor"])

        if len(joint) == 0:
            print(f"  No configs survive floor={test['floor']} — skipping")
            continue

        # Screen diagnostics
        screen_diagnostics(joint, test["floor"], test["train_cutoff"])

        # Set engine state
        _be.TRAIN_CUTOFF = test["train_cutoff"]
        _be.EVAL_START   = test["eval_start"]
        _be.EVAL_END     = test["eval_end"]
        _be.SLEEVES.clear()
        _be.SLEEVES.update(BASE_SLEEVES)

        raw_weights    = compute_neutral_weights(BASE_SLEEVES, prices)
        capped_weights = apply_weight_cap(raw_weights, WEIGHT_CAPS)
        _be.NEUTRAL_WEIGHTS.clear()
        _be.NEUTRAL_WEIGHTS.update(capped_weights)

        print(f"\n  Neutral weights (capped):")
        for k, v in capped_weights.items():
            print(f"    {k:<12}: {v:.1f}%")

        mask       = ((prices.index >= test["eval_start"]) &
                      (prices.index <= test["eval_end"]))
        eval_dates = prices.index[mask & prices["SPY"].notna()]
        print(f"\n  Eval dates: {eval_dates[0].date()} to "
              f"{eval_dates[-1].date()} ({len(eval_dates)} days)")

        # Benchmarks
        bench  = run_no_tilt_benchmark(prices, eval_dates)
        spy_bh = run_buy_and_hold(prices, "SPY", eval_dates)
        print(f"\n  Benchmark (no-tilt): "
              f"ret={bench['stats']['total_return_pct']}%  "
              f"Sharpe={bench['stats']['sharpe']}")
        print(f"  SPY buy-and-hold:    "
              f"ret={spy_bh['stats']['total_return_pct']}%  "
              f"Sharpe={spy_bh['stats']['sharpe']}")

        # HTH backtest
        print(f"\n  Hold-to-horizon backtest...")
        t0  = time.time()
        hth = run_hth(joint, prices, eval_dates, capped_weights)
        print(f"  Done in {time.time()-t0:.0f}s")
        print(f"  HTH result: ret={hth['stats']['total_return_pct']}%  "
              f"Sharpe={hth['stats']['sharpe']}  "
              f"vol={hth['stats']['ann_vol_pct']}%")
        print(f"  Total hold events: {len(hth['hold_events'])}")

        if hth["hold_events"]:
            by_sleeve = {}
            for ev in hth["hold_events"]:
                by_sleeve[ev["sleeve"]] = by_sleeve.get(ev["sleeve"], 0) + 1
            print(f"  Holds by sleeve: " +
                  "  ".join(f"{k}:{v}" for k, v in sorted(by_sleeve.items())))

            by_year = {}
            for ev in hth["hold_events"]:
                yr = ev["entry_date"].year
                by_year[yr] = by_year.get(yr, 0) + 1
            print(f"  Holds by year: " +
                  "  ".join(f"{yr}:{n}" for yr, n in sorted(by_year.items())))

        # Randomisation test
        rtest = None
        n_holds = len(hth["hold_events"])
        if not args.skip_randomisation and n_holds >= 3:
            print(f"\n  Running randomisation test ({args.n_reps} reps)...")
            t0    = time.time()
            rtest = randomisation_test(
                hth["hold_events"], prices, eval_dates, capped_weights,
                actual_sharpe=hth["stats"]["sharpe"],
                n_reps=args.n_reps,
            )
            print(f"  Done in {(time.time()-t0)/60:.1f} min")
            print(f"\n  Actual Sharpe:   {rtest['actual_sharpe']}")
            print(f"  Null mean:       {rtest['null_mean']:.3f}")
            print(f"  Null std:        {rtest['null_std']:.3f}")
            print(f"  Pct exceeding:   {rtest['pct_exceeding']}%")
            print(f"  N holds pooled:  {rtest['n_holds']}")
            if rtest["pct_exceeding"] <= 5.0:
                print(f"  *** SIGNIFICANT at 5% ***")
            elif rtest["pct_exceeding"] <= 10.0:
                print(f"  *** SIGNIFICANT at 10% ***")
            else:
                print(f"  Not significant at 10%")
        elif n_holds < 3:
            print(f"\n  Randomisation test skipped: {n_holds} hold events (need ≥ 3)")

        # Year-by-year
        if len(hth["hold_events"]) > 0:
            yearly = yearly_breakdown(
                hth["equity"]["equity"], test["eval_start"],
                test["eval_end"], hth["hold_events"]
            )
            print(f"\n  Year-by-year breakdown:")
            print(f"  {'Year':>6}  {'HTH ret%':>9}  {'HTH Sh':>7}  {'Holds':>6}")
            print(f"  {'─'*35}")
            for _, r in yearly.iterrows():
                print(f"  {int(r['year']):>6}  {r['ret_pct']:>9.2f}%  "
                      f"{r['sharpe']:>7.3f}  {int(r['holds']):>6}")

        # Collect results
        row = {
            "test":          test["label"],
            "description":   test["description"],
            "floor":         test["floor"],
            "joint_file":    test["joint_file"],
            "train_cutoff":  str(test["train_cutoff"].date()),
            "eval_start":    str(test["eval_start"].date()),
            "eval_end":      str(test["eval_end"].date()),
            "configs_before_floor": len(joint_raw),
            "configs_after_floor":  len(joint),
            "total_holds":   n_holds,
            "hth_ret_pct":   hth["stats"]["total_return_pct"],
            "hth_sharpe":    hth["stats"]["sharpe"],
            "hth_vol_pct":   hth["stats"]["ann_vol_pct"],
            "bench_ret_pct": bench["stats"]["total_return_pct"],
            "bench_sharpe":  bench["stats"]["sharpe"],
            "spy_ret_pct":   spy_bh["stats"]["total_return_pct"],
            "spy_sharpe":    spy_bh["stats"]["sharpe"],
            "pct_exceeding": rtest["pct_exceeding"] if rtest else
                             (f"<3 holds ({n_holds})" if n_holds < 3
                              else "skipped"),
            "rand_null_mean": rtest["null_mean"] if rtest else np.nan,
            "rand_null_std":  rtest["null_std"]  if rtest else np.nan,
            "rand_n_holds":   rtest["n_holds"]   if rtest else 0,
        }
        all_results.append(row)

    # Restore engine state
    _be.TRAIN_CUTOFF = orig_cutoff
    _be.EVAL_START   = orig_start
    _be.EVAL_END     = orig_end

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print(f"  SUMMARY: EPISODE FLOOR SENSITIVITY")
    print(f"{'='*72}")

    # Reference rows from paper
    print(f"\n  Reference results (floor=3, existing paper/Section 20):")
    print(f"  {'Spec':<45}  {'Holds':>6}  {'HTH Sh':>7}  {'Bench Sh':>8}  "
          f"{'Pct exc':>8}")
    print(f"  {'─'*80}")
    refs = [
        ("2024-trained, floor=3 (paper Sec 16.1, 2025)", 11, 1.224, 1.089, "1.8%"),
        ("2014-trained, floor=3 (Sec 20.4, 2015-2025)", 785, 0.904, 1.095, "72.7%"),
    ]
    for label, holds, hsh, bsh, pct in refs:
        print(f"  {label:<45}  {holds:>6}  {hsh:>7.3f}  {bsh:>8.3f}  {pct:>8}")

    print(f"\n  New results (this section):")
    print(f"  {'Spec':<45}  {'Holds':>6}  {'HTH Sh':>7}  {'Bench Sh':>8}  "
          f"{'Pct exc':>8}")
    print(f"  {'─'*80}")
    for r in all_results:
        label = f"{r['description']} ({r['eval_start'][:4]}–{r['eval_end'][:4]})"
        print(f"  {label:<45}  {r['total_holds']:>6}  "
              f"{r['hth_sharpe']:>7.3f}  {r['bench_sharpe']:>8.3f}  "
              f"{str(r['pct_exceeding']):>8}")

    print(f"\n  INTERPRETATION:")
    for r in all_results:
        pct = r["pct_exceeding"]
        try:
            pct_f = float(pct)
            sig = pct_f <= 5.0
            border = 5.0 < pct_f <= 10.0
        except (TypeError, ValueError):
            sig = border = False

        print(f"\n  Test {r['test']} ({r['description']}):")
        print(f"    Configs: {r['configs_before_floor']} → "
              f"{r['configs_after_floor']} (floor={r['floor']})")
        print(f"    Holds:   {r['total_holds']}")
        print(f"    Sharpe:  {r['hth_sharpe']} vs benchmark {r['bench_sharpe']}")
        print(f"    Pct exc: {pct}")

        if r["test"] in ("A", "B"):
            if sig:
                print(f"    VERDICT: 2025 result SURVIVES floor={r['floor']}. "
                      f"The validated signal passes the stricter criterion.")
            elif border:
                print(f"    VERDICT: 2025 result borderline at floor={r['floor']}. "
                      f"Signal weakened but not eliminated.")
            else:
                print(f"    VERDICT: 2025 result does NOT survive floor={r['floor']}. "
                      f"The VIXM+VIXY config may have been zeroed out or "
                      f"insufficient configs remained.")
        else:  # Test C
            if r["total_holds"] < 50:
                print(f"    VERDICT: Floor=5 dramatically reduced saturation "
                      f"({785} → {r['total_holds']} holds). "
                      f"The recommendation is confirmed: "
                      f"a higher floor is needed for the 2014-trained screen.")
            else:
                print(f"    VERDICT: Floor=5 reduced but did not eliminate "
                      f"saturation ({785} → {r['total_holds']} holds).")
            if sig:
                print(f"    BONUS: With reduced saturation, the 2014→2025 "
                      f"test is now SIGNIFICANT (pct_exc={pct}%). "
                      f"This is a strong additional finding.")

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\n  Saved: {args.output}")
    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()
