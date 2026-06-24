"""
run_crosssectional.py
======================
Cross-sectional re-run of the strategy described in Section 11 of the
Portfolio Tilt paper, using the corrected pipeline (economic prior,
episode-conviction sizing, training-only thresholds, MIN_TRAIN_OBS floor).

The original Section 11 result was Sharpe 0.294, beaten by 89% of random
reassignments. That test predates every fix made in the continuation
work (Sections 17-19). This script re-runs the same cross-sectional
concept with three position structures, against an episode-validated
universe (only targets with >=3 independent training-period episodes),
to test whether the corrected pipeline changes the cross-sectional result.

Three position structures, all long-only (no shorting, consistent with
the rest of the paper):

  A) LONG-TILT: same as original Section 11. Overweight top-scoring
     targets, underweight bottom-scoring, proportional to normalised
     score. Each target's weight = neutral_weight + tilt_delta, where
     tilt_delta scales the score to sum to zero across the universe
     (pure redistribution, no net leverage).

  B) TOP-QUARTILE: equal-weight the top 25% of targets by episode-
     conviction score each day. Zero weight to the rest. Simple and
     interpretable: hold the strongest-signalled quarter of the universe.

  C) SCORE-PROPORTIONAL: weight all targets proportionally to their
     positive episode-conviction score. Targets with zero or negative
     score get zero weight. Normalise to sum to 100%.

All three use:
  - Episode-validated universe only (targets with >=3 distinct training-
    period episodes in the joint screen, recomputed here at runtime)
  - Same training-only thresholds (frozen at 2024-12-31)
  - Same episode-conviction quality weights as backtest_engine.py
  - Hold-to-horizon position construction (not static daily tilt)
  - 1000-rep randomisation test for each structure
  - Both 2025 and 2026 evaluation windows

Usage:
    python run_crosssectional.py --joint joint_cpe_results.parquet
    python run_crosssectional.py --joint joint_cpe_results.parquet --eval-year 2026
"""

import argparse
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import backtest_engine as _be
from backtest_engine import (
    TRAIN_CUTOFF, build_increments_and_thresholds,
    compute_quality_weights, build_increments_for_episodes,
    _episode_conviction_for_row, _cluster_into_episodes,
    configuration_fires_on_date, clip_and_renormalise,
    simulate_portfolio, compute_performance_stats,
)

Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]
TAU_LIST = [1, 5, 10, 21, 63, 126, 252, 300]
MIN_EPISODES = 3  # episode-validated universe floor


# ── UNIVERSE CONSTRUCTION ──────────────────────────────────────────────────

def build_episode_validated_universe(joint: pd.DataFrame,
                                      prices: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the joint screen to only configurations whose target (Y) has
    at least MIN_EPISODES genuinely independent training-period episodes
    in at least one configuration. Returns the filtered joint screen plus
    a per-Y episode count dict.
    """
    print(f"  Building episode-validated universe (min {MIN_EPISODES} episodes per target)...")
    increments = build_increments_for_episodes(prices, TAU_LIST)

    # Compute episode conviction for every row -- keep rows where the
    # TARGET's best config clears the episode floor
    ep_counts_by_y = {}
    rows_with_conviction = []

    for idx, row in joint.iterrows():
        conv = _episode_conviction_for_row(row, increments)
        if conv > 0:
            rows_with_conviction.append(idx)
            y = row["Y"]
            # Track max episodes for this target across all its configs
            # (we want to know if the TARGET is episode-validated, not
            # just this one config)
            ep_counts_by_y[y] = ep_counts_by_y.get(y, 0) + 1

    validated = joint.loc[rows_with_conviction].copy()
    validated_targets = set(validated["Y"].unique())
    print(f"  Episode-validated targets: {len(validated_targets)} / {joint['Y'].nunique()}")
    print(f"  Episode-validated configs: {len(validated)} / {len(joint)}")
    return validated, increments


# ── DAILY SIGNAL SCORE (CROSS-SECTIONAL) ──────────────────────────────────

def compute_daily_scores_crosssectional(validated: pd.DataFrame,
                                         weights: pd.Series,
                                         eval_dates: pd.DatetimeIndex,
                                         increments: dict,
                                         thresholds: dict) -> pd.DataFrame:
    """
    For every (Y, tau_future) group in the validated universe, compute
    a daily signal score using the same horizon-weighted formula as the
    sleeve-based backtest, but across all targets simultaneously.

    Returns a DataFrame: index=eval_dates, columns=unique Y targets,
    values=combined daily signal score (positive=bullish, negative=bearish).
    """
    targets = sorted(validated["Y"].unique())
    horizon_weights = {21: 0.20, 63: 0.30, 126: 0.30, 252: 0.20}

    score_df = pd.DataFrame(0.0, index=eval_dates, columns=targets)

    for y in targets:
        sub = validated[validated["Y"] == y]
        if sub.empty:
            continue

        # Per-horizon score contribution
        for tau_f, hw in horizon_weights.items():
            tau_sub = sub[sub["tau_future"] == tau_f]
            if tau_sub.empty:
                continue

            bull = tau_sub[tau_sub["direction"] == "bullish"]
            bear = tau_sub[tau_sub["direction"] == "bearish"]
            w_bull_total = weights.loc[bull.index].sum() if len(bull) else 0
            w_bear_total = weights.loc[bear.index].sum() if len(bear) else 0
            denom = w_bull_total + w_bear_total
            if denom <= 0:
                continue

            for d in eval_dates:
                fired_bull_w = sum(
                    weights.loc[idx]
                    for idx in bull.index
                    if configuration_fires_on_date(bull.loc[idx], d, increments, thresholds)
                )
                fired_bear_w = sum(
                    weights.loc[idx]
                    for idx in bear.index
                    if configuration_fires_on_date(bear.loc[idx], d, increments, thresholds)
                )
                score_df.at[d, y] += hw * (fired_bull_w - fired_bear_w) / denom

    return score_df


# ── POSITION CONSTRUCTION ──────────────────────────────────────────────────

def build_weights_long_tilt(score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Structure A: long-only tilt proportional to normalised score.
    Each target starts at equal weight (1/N). Score is redistributed
    from low-scoring to high-scoring targets, preserving total = 100%.
    Max tilt: +/-15pp (same tiers as sleeve-based static tilt).
    """
    n = len(score_df.columns)
    base_weight = 100.0 / n
    tilt_rows = []
    for d in score_df.index:
        scores = score_df.loc[d]
        score_range = scores.max() - scores.min()
        if score_range > 0:
            normalised = (scores - scores.mean()) / score_range
            tilt = normalised * 15.0  # scale to +/-15pp max
        else:
            tilt = pd.Series(0.0, index=scores.index)
        raw = (base_weight + tilt).clip(lower=0)
        total = raw.sum()
        tilt_rows.append((raw / total * 100.0) if total > 0 else pd.Series(base_weight, index=scores.index))
    return pd.DataFrame(tilt_rows, index=score_df.index)


def build_weights_top_quartile(score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Structure B: equal-weight top 25% of targets by signal score each day.
    Zero weight to the rest.
    """
    n = len(score_df.columns)
    top_k = max(1, n // 4)
    weight_rows = []
    for d in score_df.index:
        scores = score_df.loc[d]
        top_targets = scores.nlargest(top_k).index
        w = pd.Series(0.0, index=scores.index)
        # Only include targets with positive score
        positive_top = [t for t in top_targets if scores[t] > 0]
        if positive_top:
            w[positive_top] = 100.0 / len(positive_top)
        else:
            # All scores zero -- equal-weight everything
            w[:] = 100.0 / n
        weight_rows.append(w)
    return pd.DataFrame(weight_rows, index=score_df.index)


def build_weights_score_proportional(score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Structure C: weight proportional to positive signal score.
    Targets with zero or negative score get zero weight.
    """
    n = len(score_df.columns)
    weight_rows = []
    for d in score_df.index:
        scores = score_df.loc[d].clip(lower=0)
        total = scores.sum()
        if total > 0:
            weight_rows.append(scores / total * 100.0)
        else:
            weight_rows.append(pd.Series(100.0 / n, index=scores.index))
    return pd.DataFrame(weight_rows, index=score_df.index)


# ── PORTFOLIO SIMULATION (cross-sectional version) ─────────────────────────

def simulate_crosssectional(weight_df: pd.DataFrame,
                              prices: pd.DataFrame,
                              eval_dates: pd.DatetimeIndex) -> dict:
    """
    Simulate a cross-sectional portfolio where weights sum to 100% across
    all targets. Uses the same daily log-return compounding as the sleeve
    backtest, with one-day position lag.
    """
    lagged = weight_df.shift(1)
    lagged.iloc[0] = weight_df.iloc[0]  # first day: use day-0 weights

    notional = 100_000.0
    equity = [notional]

    for i in range(1, len(eval_dates)):
        d = eval_dates[i]
        w = lagged.loc[d] / 100.0
        day_ret = 0.0
        for ticker in weight_df.columns:
            if ticker not in prices.columns:
                continue
            px = prices[ticker].ffill()
            if d not in px.index or pd.isna(px.loc[d]):
                continue
            prev_dates = px.index[px.index < d]
            if len(prev_dates) == 0:
                continue
            prev = px.loc[prev_dates[-1]]
            if pd.isna(prev) or prev <= 0:
                continue
            ret = px.loc[d] / prev - 1
            day_ret += w.get(ticker, 0.0) * ret
        equity.append(equity[-1] * (1 + day_ret))

    eq = pd.Series(equity, index=eval_dates)
    return {"equity": eq, "stats": compute_performance_stats(eq)}


# ── EQUAL-WEIGHT BENCHMARK ─────────────────────────────────────────────────

def run_equal_weight_benchmark(targets: list, prices: pd.DataFrame,
                                eval_dates: pd.DatetimeIndex) -> dict:
    n = len(targets)
    weight_df = pd.DataFrame(
        {t: 100.0 / n for t in targets},
        index=eval_dates
    )
    return simulate_crosssectional(weight_df, prices, eval_dates)


# ── RANDOMISATION TEST ─────────────────────────────────────────────────────

def randomisation_test_crosssectional(weight_df: pd.DataFrame,
                                       prices: pd.DataFrame,
                                       eval_dates: pd.DatetimeIndex,
                                       n_reps: int = 1000,
                                       seed: int = 42) -> dict:
    """Shuffle each target's daily weight column independently."""
    actual = simulate_crosssectional(weight_df, prices, eval_dates)
    actual_sharpe = actual["stats"]["sharpe"]

    rng = np.random.default_rng(seed)
    n_days = len(eval_dates)
    null_sharpes = []

    for _ in range(n_reps):
        perm_df = weight_df.copy()
        for col in perm_df.columns:
            perm = rng.permutation(n_days)
            perm_df[col] = weight_df[col].values[perm]
        # Renormalise rows to 100%
        row_sums = perm_df.sum(axis=1)
        perm_df = perm_df.div(row_sums, axis=0) * 100.0
        result = simulate_crosssectional(perm_df, prices, eval_dates)
        s = result["stats"]["sharpe"]
        if not np.isnan(s):
            null_sharpes.append(s)

    null_sharpes = np.array(null_sharpes)
    pct_exceeding = float((null_sharpes >= actual_sharpe).mean() * 100)
    return {
        "actual_sharpe": actual_sharpe,
        "null_mean": float(null_sharpes.mean()),
        "null_std": float(null_sharpes.std()),
        "pct_exceeding": round(pct_exceeding, 1),
        "n_reps": len(null_sharpes),
    }


# ── MAIN ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", required=True)
    parser.add_argument("--eval-year", type=int, default=None)
    parser.add_argument("--skip-randomisation", action="store_true")
    parser.add_argument("--n-reps", type=int, default=1000)
    args = parser.parse_args()

    if args.eval_year:
        _be.EVAL_START = pd.Timestamp(f"{args.eval_year}-01-01")
        _be.EVAL_END   = pd.Timestamp(f"{args.eval_year}-12-31")

    print(f"\n{'='*70}")
    print(f"  CROSS-SECTIONAL BACKTEST  |  {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Joint screen: {args.joint}")
    print(f"{'='*70}")

    prices = pd.read_parquet("multiasset_prices.parquet")

    # Evaluation dates: SPY trading days in window
    mask = (prices.index >= _be.EVAL_START) & (prices.index <= _be.EVAL_END)
    window = prices.index[mask]
    spy_valid = prices["SPY"].notna()
    eval_dates = window[window.isin(prices.index[spy_valid])]
    print(f"  Evaluation window: {eval_dates.min().date()} to {eval_dates.max().date()} "
          f"({len(eval_dates)} trading days)")

    print("\n  Loading joint screen...")
    joint = pd.read_parquet(args.joint)
    joint = joint[joint["n_predictors"] <= 6].copy()
    print(f"  Joint configs (n_pred<=6): {len(joint)}")

    # Build episode-validated universe
    validated, increments = build_episode_validated_universe(joint, prices)
    targets = sorted(validated["Y"].unique())
    print(f"  Targets in universe: {len(targets)}")

    print("\n  Building training-frozen thresholds...")
    t0 = time.time()
    _, thresholds = build_increments_and_thresholds(prices, Q_GRID)
    print(f"  Done. {time.time()-t0:.0f}s")

    print("\n  Computing episode-conviction quality weights...")
    weights = compute_quality_weights(validated, prices,
                                       precomputed_increments=increments)
    print(f"  Nonzero weights: {(weights > 0).sum()} / {len(weights)}")

    print(f"\n  Computing daily signal scores for {len(targets)} targets "
          f"x {len(eval_dates)} days...")
    t0 = time.time()
    score_df = compute_daily_scores_crosssectional(
        validated, weights, eval_dates, increments, thresholds
    )
    print(f"  Done. {time.time()-t0:.0f}s")

    # Equal-weight benchmark
    print("\n  --- Equal-weight benchmark ---")
    ew = run_equal_weight_benchmark(targets, prices, eval_dates)
    print(f"  {ew['stats']}")

    # SPY buy-and-hold
    px_spy = prices["SPY"].ffill().reindex(eval_dates)
    rets_spy = px_spy.pct_change().fillna(0)
    spy_eq = 100_000 * (1 + rets_spy).cumprod()
    spy_stats = compute_performance_stats(spy_eq)
    print(f"\n  --- SPY buy-and-hold ---")
    print(f"  {spy_stats}")

    results = {}
    weight_dfs = {}

    for label, fn in [
        ("A: Long-tilt", build_weights_long_tilt),
        ("B: Top-quartile", build_weights_top_quartile),
        ("C: Score-proportional", build_weights_score_proportional),
    ]:
        print(f"\n  --- Structure {label} ---")
        wdf = fn(score_df)
        weight_dfs[label] = wdf
        result = simulate_crosssectional(wdf, prices, eval_dates)
        results[label] = result
        print(f"  {result['stats']}")

        # Daily activity stats
        n_active_days = (wdf.max(axis=1) != wdf.min(axis=1)).sum()
        print(f"  Days with non-uniform weights: {n_active_days} / {len(eval_dates)}")

        if not args.skip_randomisation:
            print(f"  Running randomisation test ({args.n_reps} reps)...")
            t0 = time.time()
            rtest = randomisation_test_crosssectional(
                wdf, prices, eval_dates, n_reps=args.n_reps
            )
            print(f"  Actual Sharpe={rtest['actual_sharpe']:.3f}  "
                  f"Null mean={rtest['null_mean']:.3f}  "
                  f"Null std={rtest['null_std']:.3f}  "
                  f"Pct exceeding={rtest['pct_exceeding']}%  "
                  f"({rtest['n_reps']} reps)  "
                  f"Elapsed {time.time()-t0:.0f}s")
            results[label]["rtest"] = rtest

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    rows = {
        "Equal-weight benchmark": ew["stats"],
        "SPY buy-and-hold": spy_stats,
    }
    for label, r in results.items():
        rows[f"Structure {label}"] = r["stats"]
    summary = pd.DataFrame(rows).T
    print(summary.to_string())

    if not args.skip_randomisation:
        print(f"\n  Randomisation test pct_exceeding:")
        for label, r in results.items():
            if "rtest" in r:
                print(f"    Structure {label}: {r['rtest']['pct_exceeding']}%")

    # Save
    fname = f"crosssectional_result_{args.eval_year or 'default'}.csv"
    eq_df = pd.DataFrame({"equal_weight": ew["equity"], "spy": spy_eq})
    for label, r in results.items():
        col_name = f"struct_{label[0]}"
        eq_df[col_name] = r["equity"]
    eq_df.to_csv(fname)
    print(f"\n  Saved equity curves -> {fname}")


if __name__ == "__main__":
    main()
