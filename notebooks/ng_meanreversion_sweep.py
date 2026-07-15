"""
ng_meanreversion_sweep.py
===========================
Robustness check on ng_meanreversion_backtest.py: was Sharpe=0.487 at
(roll_window=252, z_entry=1.5) a lucky pick, or does the edge hold up
across nearby, equally-reasonable parameter choices? Sweeps a grid of
(roll_window, z_entry) combinations, runs the same walk-forward
backtest + circular-shift permutation test at each point (reduced to
300 reps/cell to keep the full 5x5 grid tractable -- the single-config
deep dive already used 1000 reps), and reports Sharpe plus permutation
p-value for every cell.

A real edge should look like a smooth, broadly-positive region across
this grid. A result that's only good in one narrow corner and falls
apart everywhere else is the classic signature of a parameter that got
fit to this data, not a genuine effect.

Usage:
    ../.venv/bin/python ng_meanreversion_sweep.py
"""

import numpy as np
import pandas as pd

from ng_meanreversion_backtest import (
    load_prices, build_signal, run_strategy, perf_stats, TX_COST_BPS,
)

ROLL_WINDOWS = [126, 189, 252, 378, 504]     # ~0.5, 0.75, 1, 1.5, 2 years
Z_ENTRIES = [1.0, 1.25, 1.5, 1.75, 2.0]
N_PERMUTATIONS_SWEEP = 300
SEED = 42


def circular_shift_pvalue(logp, position, actual_sharpe, n_reps=N_PERMUTATIONS_SWEEP, seed=SEED):
    ret = logp.diff().fillna(0).values
    pos = position.values
    n = len(pos)
    rng = np.random.default_rng(seed)
    null_sharpes = []
    for _ in range(n_reps):
        shift = rng.integers(1, n - 1)
        shifted_pos = np.roll(pos, shift)
        strat_ret = shifted_pos * ret
        trade = np.abs(np.diff(shifted_pos, prepend=shifted_pos[0]))
        cost = trade * (TX_COST_BPS / 10_000.0)
        strat_ret = strat_ret - cost
        eq = 100_000 * np.exp(np.cumsum(strat_ret))
        r = pd.Series(eq).pct_change().dropna()
        s = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan
        if not np.isnan(s):
            null_sharpes.append(s)
    null_sharpes = np.array(null_sharpes)
    if len(null_sharpes) == 0:
        return np.nan
    return float((null_sharpes >= actual_sharpe).mean())


def main():
    price = load_prices()
    logp = np.log(price)

    print(f"\nSweeping {len(ROLL_WINDOWS)}x{len(Z_ENTRIES)} = "
          f"{len(ROLL_WINDOWS)*len(Z_ENTRIES)} (roll_window, z_entry) combinations...")
    print(f"({N_PERMUTATIONS_SWEEP} permutation reps per cell)\n")

    rows = []
    for w in ROLL_WINDOWS:
        for z in Z_ENTRIES:
            sig = build_signal(logp, roll_window=w, z_entry=z)
            position = sig["position"]
            n_trades = int((position.diff().fillna(position.iloc[0]) != 0).sum())
            time_in_market = float((position != 0).mean() * 100)

            if n_trades < 5:
                rows.append(dict(roll_window=w, z_entry=z, n_trades=n_trades,
                                  time_in_market_pct=round(time_in_market, 1),
                                  CAGR_pct=np.nan, Sharpe=np.nan, max_dd_pct=np.nan, p_value=np.nan))
                continue

            eq = run_strategy(logp, position, roll_drag_annual=0.0)
            stats = perf_stats(eq)
            pval = circular_shift_pvalue(logp, position, stats["Sharpe"])

            rows.append(dict(
                roll_window=w, z_entry=z, n_trades=n_trades,
                time_in_market_pct=round(time_in_market, 1),
                CAGR_pct=stats["CAGR_%"], Sharpe=stats["Sharpe"],
                max_dd_pct=stats["max_drawdown_%"], p_value=round(pval, 3),
            ))
            print(f"  window={w:>4}d  z={z:.2f}  ->  Sharpe={stats['Sharpe']:.3f}  "
                  f"CAGR={stats['CAGR_%']:.1f}%  maxDD={stats['max_drawdown_%']:.1f}%  "
                  f"p={pval:.3f}  n_trades={n_trades}")

    df = pd.DataFrame(rows)
    df.to_csv("ng_meanreversion_sweep_results.csv", index=False)

    print(f"\n{'='*78}")
    print("  SHARPE, BY (roll_window x z_entry)")
    print(f"{'='*78}")
    pivot_sharpe = df.pivot(index="roll_window", columns="z_entry", values="Sharpe")
    print(pivot_sharpe.to_string())

    print(f"\n{'='*78}")
    print("  PERMUTATION P-VALUE, BY (roll_window x z_entry)")
    print(f"{'='*78}")
    pivot_p = df.pivot(index="roll_window", columns="z_entry", values="p_value")
    print(pivot_p.to_string())

    print(f"\n{'='*78}")
    print("  SUMMARY")
    print(f"{'='*78}")
    valid = df.dropna(subset=["Sharpe"])
    n_positive_sharpe = (valid["Sharpe"] > 0).sum()
    n_sig_at_05 = (valid["p_value"] < 0.05).sum()
    n_sig_at_10 = (valid["p_value"] < 0.10).sum()
    print(f"  Cells with positive Sharpe: {n_positive_sharpe}/{len(valid)}")
    print(f"  Cells significant at p<0.05: {n_sig_at_05}/{len(valid)}")
    print(f"  Cells significant at p<0.10: {n_sig_at_10}/{len(valid)}")
    print(f"  Sharpe range: [{valid['Sharpe'].min():.3f}, {valid['Sharpe'].max():.3f}], "
          f"median={valid['Sharpe'].median():.3f}")

    print(f"\nSaved full grid -> ng_meanreversion_sweep_results.csv")


if __name__ == "__main__":
    main()
