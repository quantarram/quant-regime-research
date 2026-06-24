"""
run_benchmarks.py
==================
Implements four standard quantitative strategies against the same
2025 evaluation window and 61-target episode-validated universe used
by the CPE cross-sectional strategy, for direct comparison.

All strategies:
  - Long-only (no shorting, consistent with the CPE paper)
  - Zero transaction costs (consistent with the CPE paper)
  - One-day position lag (signal computed at close of day t, applied
    to return of day t+1)
  - Evaluated on 2025-01-02 to 2025-12-31 (250 trading days)
  - Parameters estimated from training data only (<=2024-12-31)
  - Annualised Sharpe, zero risk-free rate (CPE paper convention)

Strategies implemented:
  1. Time-series momentum (TSMOM): long if trailing 12-month return
     positive, flat otherwise. Equal-weight among long positions.
     (Moskowitz, Ooi & Pedersen 2012)

  2. Cross-sectional momentum (XSMOM): long top quartile by trailing
     12-month return. Equal-weight among top-quartile positions.
     (Jegadeesh & Titman 1993, extended to multi-asset)

  3. Trend-following (EWMA crossover): long when 63-day EWMA > 252-day
     EWMA, flat otherwise. Equal-weight among trending positions.
     Standard CTA / managed futures implementation.

  4. Risk parity (equal-vol): weights inversely proportional to trailing
     63-day realised volatility, rebalanced daily. Normalised to sum
     to 100% (no leverage).

  5. Equal-weight (naive diversification): equal weight across all 61
     targets, no signal. Baseline / benchmark for all strategies.

Usage:
    python run_benchmarks.py --universe universe_tickers.txt
    python run_benchmarks.py  # auto-builds episode-validated universe
"""

import argparse
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import backtest_engine as _be
from backtest_engine import TRAIN_CUTOFF, EVAL_START, EVAL_END, compute_performance_stats


# ── UNIVERSE ───────────────────────────────────────────────────────────────

def load_episode_validated_universe(joint_path: str, prices: pd.DataFrame) -> list:
    """Load the joint screen and return tickers of episode-validated targets."""
    from backtest_engine import (build_increments_for_episodes,
                                  _episode_conviction_for_row, TAU_LIST)
    joint = pd.read_parquet(joint_path)
    joint = joint[joint["n_predictors"] <= 6].copy()

    print(f"  Loading episode-validated universe from {joint_path}...")
    increments = build_increments_for_episodes(prices, TAU_LIST)

    validated_targets = set()
    for idx, row in joint.iterrows():
        conv = _episode_conviction_for_row(row, increments)
        if conv > 0:
            validated_targets.add(row["Y"])

    # Keep only tickers actually present in prices with sufficient history
    targets = sorted(
        t for t in validated_targets
        if t in prices.columns and
        prices[t].loc[prices.index <= TRAIN_CUTOFF].dropna().__len__() >= 252
    )
    print(f"  Episode-validated targets with >= 252 training days: {len(targets)}")
    return targets


# ── COMMON HELPERS ──────────────────────────────────────────────────────────

def get_eval_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    mask = (prices.index >= EVAL_START) & (prices.index <= EVAL_END)
    window = prices.index[mask]
    spy_valid = prices["SPY"].notna()
    return window[window.isin(prices.index[spy_valid])]


def simulate_portfolio(weight_df: pd.DataFrame, prices: pd.DataFrame,
                        eval_dates: pd.DatetimeIndex) -> pd.Series:
    """Simulate $100k notional, one-day lag, no transaction costs."""
    lagged = weight_df.shift(1)
    lagged.iloc[0] = weight_df.iloc[0]

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
            if d not in px.index:
                continue
            prev_idx = px.index[px.index < d]
            if len(prev_idx) == 0:
                continue
            prev = px.loc[prev_idx[-1]]
            if pd.isna(prev) or prev <= 0:
                continue
            curr = px.loc[d]
            if pd.isna(curr):
                continue
            day_ret += w.get(ticker, 0.0) * (curr / prev - 1)
        equity.append(equity[-1] * (1 + day_ret))

    return pd.Series(equity, index=eval_dates)


def randomisation_test(weight_df: pd.DataFrame, prices: pd.DataFrame,
                        eval_dates: pd.DatetimeIndex,
                        actual_sharpe: float,
                        n_reps: int = 1000, seed: int = 42) -> dict:
    """Shuffle each column's weight series independently, preserving marginal."""
    rng = np.random.default_rng(seed)
    n_days = len(eval_dates)
    null_sharpes = []

    for _ in range(n_reps):
        perm_df = weight_df.copy()
        for col in perm_df.columns:
            perm_df[col] = weight_df[col].values[rng.permutation(n_days)]
        row_sums = perm_df.sum(axis=1)
        perm_df = perm_df.div(row_sums.replace(0, np.nan), axis=0).fillna(0) * 100.0
        eq = simulate_portfolio(perm_df, prices, eval_dates)
        s = compute_performance_stats(eq)["sharpe"]
        if not np.isnan(s):
            null_sharpes.append(s)

    null_sharpes = np.array(null_sharpes)
    pct_exceeding = float((null_sharpes >= actual_sharpe).mean() * 100)
    return {
        "null_mean": round(float(null_sharpes.mean()), 3),
        "null_std": round(float(null_sharpes.std()), 3),
        "pct_exceeding": round(pct_exceeding, 1),
        "n_reps": len(null_sharpes),
    }


# ── STRATEGY 1: TIME-SERIES MOMENTUM ───────────────────────────────────────

def run_tsmom(targets: list, prices: pd.DataFrame,
               eval_dates: pd.DatetimeIndex, lookback: int = 252) -> dict:
    """
    Long if trailing lookback-day return is positive, flat otherwise.
    Equal-weight among long positions. Parameters computed from full
    price history up to each day (rolling, not fixed training-period).
    No look-ahead: signal on day t uses return ending at close of t.
    """
    # Compute rolling lookback-day returns for all targets
    returns = pd.DataFrame(index=prices.index)
    for t in targets:
        if t in prices.columns:
            px = prices[t].ffill()
            returns[t] = px / px.shift(lookback) - 1

    weight_rows = []
    for d in eval_dates:
        if d not in returns.index:
            weight_rows.append(pd.Series(100.0 / len(targets), index=targets))
            continue
        long_pos = [t for t in targets if returns.loc[d, t] > 0
                     and not pd.isna(returns.loc[d, t])]
        w = pd.Series(0.0, index=targets)
        if long_pos:
            w[long_pos] = 100.0 / len(long_pos)
        else:
            w[:] = 100.0 / len(targets)  # no signal: equal-weight
        weight_rows.append(w)

    wdf = pd.DataFrame(weight_rows, index=eval_dates)
    eq = simulate_portfolio(wdf, prices, eval_dates)
    stats = compute_performance_stats(eq)

    n_active = (wdf.max(axis=1) != wdf.min(axis=1)).sum()
    return {"equity": eq, "stats": stats, "weights": wdf,
            "n_active_days": int(n_active)}


# ── STRATEGY 2: CROSS-SECTIONAL MOMENTUM ────────────────────────────────────

def run_xsmom(targets: list, prices: pd.DataFrame,
               eval_dates: pd.DatetimeIndex, lookback: int = 252,
               top_frac: float = 0.25) -> dict:
    """
    Long top top_frac of assets by trailing lookback-day return.
    Equal-weight among top positions.
    """
    returns = pd.DataFrame(index=prices.index)
    for t in targets:
        if t in prices.columns:
            px = prices[t].ffill()
            returns[t] = px / px.shift(lookback) - 1

    top_k = max(1, int(len(targets) * top_frac))
    weight_rows = []
    for d in eval_dates:
        if d not in returns.index:
            weight_rows.append(pd.Series(100.0 / len(targets), index=targets))
            continue
        day_rets = returns.loc[d].dropna()
        top = day_rets.nlargest(top_k).index.tolist()
        w = pd.Series(0.0, index=targets)
        if top:
            w[top] = 100.0 / len(top)
        else:
            w[:] = 100.0 / len(targets)
        weight_rows.append(w)

    wdf = pd.DataFrame(weight_rows, index=eval_dates)
    eq = simulate_portfolio(wdf, prices, eval_dates)
    stats = compute_performance_stats(eq)

    n_active = (wdf.max(axis=1) != wdf.min(axis=1)).sum()
    return {"equity": eq, "stats": stats, "weights": wdf,
            "n_active_days": int(n_active)}


# ── STRATEGY 3: TREND-FOLLOWING (EWMA CROSSOVER) ────────────────────────────

def run_trend(targets: list, prices: pd.DataFrame,
               eval_dates: pd.DatetimeIndex,
               fast: int = 63, slow: int = 252) -> dict:
    """
    Long when fast EWMA > slow EWMA, flat otherwise.
    Equal-weight among trending positions.
    """
    fast_ewma = pd.DataFrame(index=prices.index)
    slow_ewma = pd.DataFrame(index=prices.index)
    for t in targets:
        if t in prices.columns:
            px = prices[t].ffill()
            fast_ewma[t] = px.ewm(span=fast, adjust=False).mean()
            slow_ewma[t] = px.ewm(span=slow, adjust=False).mean()

    weight_rows = []
    for d in eval_dates:
        if d not in fast_ewma.index:
            weight_rows.append(pd.Series(100.0 / len(targets), index=targets))
            continue
        trending = [t for t in targets
                    if not pd.isna(fast_ewma.loc[d, t])
                    and not pd.isna(slow_ewma.loc[d, t])
                    and fast_ewma.loc[d, t] > slow_ewma.loc[d, t]]
        w = pd.Series(0.0, index=targets)
        if trending:
            w[trending] = 100.0 / len(trending)
        else:
            w[:] = 100.0 / len(targets)
        weight_rows.append(w)

    wdf = pd.DataFrame(weight_rows, index=eval_dates)
    eq = simulate_portfolio(wdf, prices, eval_dates)
    stats = compute_performance_stats(eq)

    n_active = (wdf.max(axis=1) != wdf.min(axis=1)).sum()
    return {"equity": eq, "stats": stats, "weights": wdf,
            "n_active_days": int(n_active)}


# ── STRATEGY 4: RISK PARITY (EQUAL-VOL) ─────────────────────────────────────

def run_risk_parity(targets: list, prices: pd.DataFrame,
                     eval_dates: pd.DatetimeIndex, vol_lookback: int = 63) -> dict:
    """
    Weights inversely proportional to trailing vol_lookback-day realised
    volatility. Rebalanced daily. Normalised to sum to 100% (no leverage).
    """
    log_rets = pd.DataFrame(index=prices.index)
    for t in targets:
        if t in prices.columns:
            px = prices[t].ffill()
            log_rets[t] = np.log(px / px.shift(1))

    rolling_vol = log_rets.rolling(vol_lookback).std() * np.sqrt(252)

    weight_rows = []
    for d in eval_dates:
        if d not in rolling_vol.index:
            weight_rows.append(pd.Series(100.0 / len(targets), index=targets))
            continue
        vols = rolling_vol.loc[d].dropna()
        vols = vols[vols > 0]
        if vols.empty:
            weight_rows.append(pd.Series(100.0 / len(targets), index=targets))
            continue
        inv_vol = 1.0 / vols
        normed = inv_vol / inv_vol.sum() * 100.0
        w = pd.Series(0.0, index=targets)
        w[normed.index] = normed.values
        weight_rows.append(w)

    wdf = pd.DataFrame(weight_rows, index=eval_dates)
    eq = simulate_portfolio(wdf, prices, eval_dates)
    stats = compute_performance_stats(eq)
    return {"equity": eq, "stats": stats, "weights": wdf}


# ── MAIN ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", default="joint_cpe_results.parquet",
                         help="Joint CPE screen to derive episode-validated universe")
    parser.add_argument("--skip-randomisation", action="store_true")
    parser.add_argument("--n-reps", type=int, default=1000)
    args = parser.parse_args()

    prices = pd.read_parquet("multiasset_prices.parquet")
    eval_dates = get_eval_dates(prices)
    print(f"Evaluation window: {eval_dates.min().date()} to {eval_dates.max().date()} "
          f"({len(eval_dates)} trading days)")

    # Build episode-validated universe
    targets = load_episode_validated_universe(args.joint, prices)

    # CPE reference numbers (from Section 20 of the paper, 2025)
    cpe_reference = {
        "CPE: 5-sleeve HTH (corrected weights)":
            {"total_return_pct": 18.77, "ann_vol_pct": 15.15, "sharpe": 1.224,
             "note": "pct_exceeding=1.8%"},
        "CPE: Cross-sectional top-quartile (Structure B)":
            {"total_return_pct": 24.58, "ann_vol_pct": 14.42, "sharpe": 1.613,
             "note": "pct_exceeding=2.2%"},
        "CPE: Equal-weight episode-validated universe":
            {"total_return_pct": 14.45, "ann_vol_pct": 10.81, "sharpe": 1.317,
             "note": "no signal applied"},
    }

    print(f"\n{'='*70}")
    print("  BENCHMARK STRATEGY COMPARISON  |  2025")
    print(f"{'='*70}")

    # SPY buy-and-hold
    px_spy = prices["SPY"].ffill().reindex(eval_dates)
    spy_eq = 100_000 * (1 + px_spy.pct_change().fillna(0)).cumprod()
    spy_stats = compute_performance_stats(spy_eq)
    print(f"\n  SPY buy-and-hold: {spy_stats}")

    results = {}

    # Equal-weight baseline
    print(f"\n  --- Equal-weight ({len(targets)} targets) ---")
    ew_wdf = pd.DataFrame(100.0 / len(targets), index=eval_dates, columns=targets)
    ew_eq = simulate_portfolio(ew_wdf, prices, eval_dates)
    ew_stats = compute_performance_stats(ew_eq)
    results["Equal-weight (61 targets)"] = {"stats": ew_stats}
    print(f"  {ew_stats}")

    strategies = [
        ("TSMOM (12-month lookback)", run_tsmom, {"lookback": 252}),
        ("XSMOM top-quartile (12-month lookback)", run_xsmom,
         {"lookback": 252, "top_frac": 0.25}),
        ("Trend-following EWMA (63d/252d)", run_trend,
         {"fast": 63, "slow": 252}),
        ("Risk parity (equal-vol, 63d)", run_risk_parity, {"vol_lookback": 63}),
    ]

    for name, fn, kwargs in strategies:
        print(f"\n  --- {name} ---")
        t0 = time.time()
        result = fn(targets, prices, eval_dates, **kwargs)
        results[name] = result
        print(f"  {result['stats']}")
        if "n_active_days" in result:
            print(f"  Active days (non-uniform weights): {result['n_active_days']} / {len(eval_dates)}")

        if not args.skip_randomisation:
            rtest = randomisation_test(
                result["weights"], prices, eval_dates,
                result["stats"]["sharpe"], n_reps=args.n_reps
            )
            results[name]["rtest"] = rtest
            print(f"  Randomisation: null_mean={rtest['null_mean']}  "
                  f"null_std={rtest['null_std']}  "
                  f"pct_exceeding={rtest['pct_exceeding']}%  "
                  f"({rtest['n_reps']} reps)  {time.time()-t0:.0f}s")

    # Summary table
    print(f"\n{'='*70}")
    print("  SUMMARY: CPE vs Standard Quant Strategies (2025)")
    print(f"{'='*70}")
    print(f"\n  {'Strategy':<45} {'Return':>8} {'Vol':>7} {'Sharpe':>7} {'pct_exc':>9}")
    print(f"  {'-'*45} {'-'*8} {'-'*7} {'-'*7} {'-'*9}")

    # SPY
    print(f"  {'SPY buy-and-hold':<45} "
          f"{spy_stats['total_return_pct']:>7.2f}% "
          f"{spy_stats['ann_vol_pct']:>6.2f}% "
          f"{spy_stats['sharpe']:>7.3f}  {'--':>9}")

    # Equal-weight
    print(f"  {'Equal-weight (61-target EV universe)':<45} "
          f"{ew_stats['total_return_pct']:>7.2f}% "
          f"{ew_stats['ann_vol_pct']:>6.2f}% "
          f"{ew_stats['sharpe']:>7.3f}  {'--':>9}")

    # Benchmarks
    for name, r in results.items():
        if name == "Equal-weight (61 targets)":
            continue
        s = r["stats"]
        exc = f"{r['rtest']['pct_exceeding']}%" if "rtest" in r else "--"
        print(f"  {name:<45} "
              f"{s['total_return_pct']:>7.2f}% "
              f"{s['ann_vol_pct']:>6.2f}% "
              f"{s['sharpe']:>7.3f}  {exc:>9}")

    # CPE reference
    print(f"\n  {'--- CPE strategy results (from paper) ---':<45}")
    for name, s in cpe_reference.items():
        note = s.get("note", "")
        print(f"  {name:<45} "
              f"{s['total_return_pct']:>7.2f}% "
              f"{s['ann_vol_pct']:>6.2f}% "
              f"{s['sharpe']:>7.3f}  {note:>9}")

    # Save equity curves
    equity_df = pd.DataFrame({"SPY": spy_eq, "Equal_weight": ew_eq})
    for name, r in results.items():
        if name != "Equal-weight (61 targets)" and "equity" in r:
            equity_df[name.replace(" ", "_")[:30]] = r["equity"]
    equity_df.to_csv("benchmark_equity_curves_2025.csv")
    print(f"\n  Saved equity curves -> benchmark_equity_curves_2025.csv")


if __name__ == "__main__":
    main()
