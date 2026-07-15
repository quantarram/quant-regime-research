"""
ng_meanreversion_backtest.py
=============================
Tests the concrete question from the natural-gas discussion: if you buy
NG=F when it's statistically "low" and sell/short when it's statistically
"high," repeated over time, is that a profitable strategy net of realistic
costs -- or does the answer only look good until costs are included?

This is a STANDALONE script, not built on backtest_engine.py -- that
engine implements a specific multi-sleeve (Equities/Gold/Bonds/Crypto/FX)
joint-CPE portfolio-tilt strategy with its own fixed 2025 eval window and
episode-conviction machinery. Natural gas isn't one of its sleeves and the
question here (single-instrument threshold-crossing with futures roll
cost) doesn't fit that architecture. This reuses the same conventions
(multiasset_prices.parquet, cost-in-bps labeling per ibkr_paper_ledger.py)
but is its own, simpler, purpose-built backtest.

Methodology notes (read before trusting the numbers):
  - Signal is a WALK-FORWARD rolling z-score: at each date t, "high"/"low"
    is judged only against the trailing ROLL_WINDOW days of history up to
    t-1 -- never the full-sample statistics the earlier screener used
    (that was fine for a descriptive diagnostic, not for a trading rule,
    since a real trader at date t doesn't know the future or the
    full-sample trend).
  - Position for day t is decided from z computed through t-1, then
    applied to day t's return -- one-day lag, same discipline as
    backtest_engine.py's spec A.7.
  - Three-state band strategy: long below -Z_ENTRY std, short above
    +Z_ENTRY std, flat in between. This is one clean, textbook parameter
    choice (Z_ENTRY=1.5, ROLL_WINDOW=252d), NOT fit/optimized on this
    data -- no parameter search was run to cherry-pick a flattering
    result.
  - Costs: TX_COST_BPS per unit of position change (round-trip realistic
    estimate for a liquid futures contract, consistent with the 2-8bps
    range already used per-asset-class in ibkr_paper_ledger.py). Futures
    roll/contango drag is a SEPARATE, deliberately uncertain add-on: NG=F's
    own history shows repeated >10% single-day jumps clustered near
    month-end (contract-expiry) dates, suggesting Yahoo's continuous
    series is NOT smoothly back-adjusted and likely already embeds some
    real historical roll jumps -- but this wasn't independently confirmed
    against actual futures-curve data. Rather than assert a precise
    number, this script reports the strategy at three assumed additional
    annualized roll-drag levels (0%, -10%, -20%/yr on long exposure,
    symmetric credit on short exposure) so the reader can see how much of
    the theoretical edge, if any, survives realistic cost uncertainty.

Usage:
    ../.venv/bin/python ng_meanreversion_backtest.py
"""

import numpy as np
import pandas as pd

ROLL_WINDOW = 252       # trading days, rolling lookback for the z-score
Z_ENTRY = 1.5            # std devs from rolling mean to trigger long/short
TX_COST_BPS = 5.0        # round-trip cost estimate per position change, in bps
ROLL_DRAG_SCENARIOS = [0.0, -0.10, -0.20]   # extra assumed annualized drag on long exposure
N_PERMUTATIONS = 1000
TICKER = "NG=F"


def load_prices() -> pd.Series:
    prices = pd.read_parquet("multiasset_prices.parquet")
    s = prices[TICKER].dropna()
    print(f"Loaded {TICKER}: {len(s)} obs, {s.index.min().date()} to {s.index.max().date()}")
    return s


def build_signal(logp: pd.Series, roll_window: int = ROLL_WINDOW,
                  z_entry: float = Z_ENTRY) -> pd.DataFrame:
    """Walk-forward rolling z-score and resulting position, using only
    information available strictly before each date (shift(1) on the
    rolling stats themselves, since pandas .rolling() at row t includes
    row t by default).

    Explicit sequential state machine rather than a vectorised ffill trick
    -- three states (long/flat/short), entered at |z|>=z_entry, exited
    back to flat when z crosses 0. ~6,500 rows, trivially fast as a loop,
    and far less error-prone than encoding hysteresis via replace/ffill
    chains (an earlier vectorised version of this had a bug where the
    flat-exit state got silently re-filled over by the next ffill pass)."""
    roll_mean = logp.rolling(roll_window).mean().shift(1)
    roll_std = logp.rolling(roll_window).std().shift(1)
    z = (logp.shift(1) - roll_mean) / roll_std   # z as of t-1, applied to trade at t

    position = np.zeros(len(z), dtype=int)
    cur = 0
    for i, zi in enumerate(z.values):
        if np.isnan(zi):
            cur = 0
        elif cur == 0:
            if zi <= -z_entry:
                cur = 1
            elif zi >= z_entry:
                cur = -1
        elif cur == 1 and zi >= 0:
            cur = 0
        elif cur == -1 and zi <= 0:
            cur = 0
        position[i] = cur

    return pd.DataFrame({"z": z, "position": position}, index=logp.index)


def run_strategy(logp: pd.Series, position: pd.Series, roll_drag_annual: float,
                  tx_cost_bps: float = TX_COST_BPS) -> pd.Series:
    ret = logp.diff().fillna(0.0)  # daily log return; first row has no prior price, fill 0
    strat_ret = position * ret  # position already lagged into "at t" in build_signal
    # extra assumed roll drag: hurts long exposure, credits short exposure
    daily_drag = roll_drag_annual / 252.0
    strat_ret = strat_ret + position * daily_drag

    trade = position.diff().fillna(position.iloc[0]).abs()  # units of position change
    cost = trade * (tx_cost_bps / 10_000.0)
    strat_ret = strat_ret - cost

    equity = 100_000 * np.exp(strat_ret.cumsum())
    return equity


def perf_stats(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else np.nan
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    return {
        "CAGR_%": round(cagr * 100, 2),
        "ann_vol_%": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "max_drawdown_%": round(max_dd * 100, 2),
        "final_equity": round(equity.iloc[-1], 0),
    }


def trade_stats(position: pd.Series, logp: pd.Series) -> dict:
    pos_diff = position.diff().fillna(0)
    entries = pos_diff[pos_diff != 0].index
    n_trades = len(entries)
    time_in_market_pct = (position != 0).mean() * 100

    # per-trade P&L for a rough win-rate (ignoring costs, just direction)
    segments = []
    cur_pos = position.iloc[0]
    seg_start = position.index[0]
    for d in position.index[1:]:
        if position.loc[d] != cur_pos:
            if cur_pos != 0:
                seg_ret = cur_pos * (logp.loc[d] - logp.loc[seg_start])
                segments.append(seg_ret)
            cur_pos = position.loc[d]
            seg_start = d
    win_rate = (np.array(segments) > 0).mean() * 100 if segments else np.nan
    avg_holding_days = None
    if segments:
        avg_holding_days = len(position) / max(len(segments), 1)  # rough

    return {
        "n_position_changes": int(n_trades),
        "n_completed_segments": len(segments),
        "win_rate_%": round(win_rate, 1) if segments else np.nan,
        "time_in_market_%": round(time_in_market_pct, 1),
    }


def circular_shift_permutation_test(logp: pd.Series, position: pd.Series,
                                     n_reps: int = N_PERMUTATIONS, seed: int = 42) -> dict:
    """
    Null: keep the exact position path (same trades, same holding
    durations, same total time in market) but shift WHEN in history it
    occurs, via a random circular shift. This tests whether the actual
    entry/exit TIMING carries real information, versus just having been
    long/short/flat with this trade cadence at random points in the
    series (which would already capture NG's average drift/vol without
    any genuine mean-reversion timing skill).
    """
    ret = logp.diff().fillna(0).values
    pos = position.values
    n = len(pos)
    actual_equity = run_strategy(logp, position, roll_drag_annual=0.0)
    actual_sharpe = perf_stats(actual_equity)["Sharpe"]

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
    pct_exceeding = (null_sharpes >= actual_sharpe).mean() * 100
    return {
        "actual_sharpe": actual_sharpe,
        "null_mean": float(null_sharpes.mean()),
        "null_std": float(null_sharpes.std()),
        "pct_exceeding": round(pct_exceeding, 1),
        "n_reps": len(null_sharpes),
    }


def main():
    price = load_prices()
    logp = np.log(price)

    sig = build_signal(logp)
    position = sig["position"]

    print(f"\nSignal: rolling {ROLL_WINDOW}d z-score, enter at |z|>={Z_ENTRY}, exit at z crossing 0")
    print(f"First valid signal date: {sig['z'].dropna().index.min().date()}")

    tstats = trade_stats(position, logp)
    print(f"\nTrade stats: {tstats}")

    print(f"\n{'='*78}")
    print("  RESULTS")
    print(f"{'='*78}")

    rows = {}
    for drag in ROLL_DRAG_SCENARIOS:
        eq = run_strategy(logp, position, roll_drag_annual=drag)
        label = f"Strategy (extra roll drag {drag*100:.0f}%/yr)"
        rows[label] = perf_stats(eq)

    bh_ret = logp.diff().fillna(0)
    bh_equity = 100_000 * np.exp(bh_ret.cumsum())
    bh_equity.iloc[0] *= (1 - TX_COST_BPS / 10_000.0)  # one-time entry cost
    rows["Buy & hold NG=F"] = perf_stats(bh_equity)

    summary = pd.DataFrame(rows).T
    print(summary.to_string())

    print(f"\n{'='*78}")
    print("  PERMUTATION TEST (base case, 0% extra drag, position-timing null)")
    print(f"{'='*78}")
    ptest = circular_shift_permutation_test(logp, position)
    print(f"  Actual Sharpe = {ptest['actual_sharpe']}")
    print(f"  Null Sharpe: mean={ptest['null_mean']:.3f}, std={ptest['null_std']:.3f} "
          f"({ptest['n_reps']} circular-shift reps)")
    print(f"  Pct of null reps with Sharpe >= actual: {ptest['pct_exceeding']}%")
    print(f"  (i.e. actual result is at roughly the {100-ptest['pct_exceeding']:.0f}th percentile "
          f"of a same-trade-cadence random-timing null)")

    out = pd.DataFrame({
        "price": price,
        "z": sig["z"],
        "position": position,
    })
    out.to_csv("ng_meanreversion_backtest_detail.csv")
    print(f"\nSaved daily detail -> ng_meanreversion_backtest_detail.csv")


if __name__ == "__main__":
    main()
