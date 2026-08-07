"""
xsmom_benchmark.py
===================
Second control strategy alongside tsmom_benchmark.py: Cross-Sectional
Momentum (Jegadeesh & Titman, 1993, "Returns to Buying Winners and Selling
Losers"), the other pillar of published, decades-validated, industry-scale
momentum investing (distinct from TSMOM -- this ranks assets AGAINST EACH
OTHER, not against their own history, and is dollar-neutral by
construction rather than directionally exposed).

Standard "12-1" specification, unchanged from the published literature:
  - formation return = trailing 12-month return, SKIPPING the most recent
    1 month (the paper's own refinement -- the last month alone shows
    short-term reversal, contaminating a pure momentum signal if included)
  - each month, rank the universe by formation return
  - go long the top tercile, short the bottom tercile, equal-weighted
    within each leg, dollar-neutral (long leg sums to +100%, short leg to
    -100%)
  - rebalanced monthly

Universe: restricted to the 16 equity/sector/international-equity
instruments in this program's own panel (SPY, QQQ, IWM, DIA, 9 sector
ETFs, EWJ/EWY/EWZ) -- NOT the full 29-instrument multi-asset set used for
TSMOM. This is a disclosed, deliberate choice: cross-sectional momentum's
published validation ranks assets WITHIN a single, comparable asset class
(equities), where a "top tercile vs bottom tercile" comparison is
economically meaningful. Ranking a 29-instrument set that mixes BTC-USD's
80%+ annualized vol against TLT's ~8% would produce a long-short book
dominated by which asset CLASS happened to be in the top/bottom tercile,
not genuine cross-sectional selection within a comparable universe -- not
a fair test of the strategy as actually practiced or published.

No significance/randomisation-test games -- Jensen's alpha (OLS intercept
vs. the same universe's passive equal-weight benchmark, for consistency
with tsmom_benchmark.py, even though a dollar-neutral book's primary
judgment is its own raw Sharpe, not benchmark-relative alpha), one real
comparison, reported honestly whichever way it comes out.

Run: python xsmom_benchmark.py
Output: xsmom_benchmark_results.json, xsmom_benchmark_equity.png
"""
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tsmom_benchmark as _tm  # reuse month_end_dates, mean_diff_alpha, sharpe, COST_BPS

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FORMATION_DAYS = 252   # 12 months
SKIP_DAYS = 21         # skip most recent 1 month (standard 12-1 spec)
MIN_HISTORY_DAYS = FORMATION_DAYS + SKIP_DAYS + 21
TERCILE_FRAC = 1.0 / 3.0

UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU",
    "EWJ", "EWY", "EWZ",
]


def build_monthly_positions(prices: pd.DataFrame, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = {}
    for d in rebal_dates:
        formation_scores = {}
        for tkr in UNIVERSE:
            s = prices[tkr].dropna().loc[:d]
            if len(s) < MIN_HISTORY_DAYS:
                continue
            # 12-month return ending SKIP_DAYS before d (skip the most recent month)
            p_end = s.iloc[-1 - SKIP_DAYS] if SKIP_DAYS > 0 else s.iloc[-1]
            p_start = s.iloc[-1 - SKIP_DAYS - FORMATION_DAYS]
            formation_scores[tkr] = float(np.log(p_end / p_start))
        if len(formation_scores) < 6:  # need enough names for a meaningful tercile split
            rows[d] = {t: np.nan for t in UNIVERSE}
            continue
        ranked = sorted(formation_scores.items(), key=lambda kv: kv[1])
        n = len(ranked)
        n_leg = max(1, int(round(n * TERCILE_FRAC)))
        losers = [t for t, _ in ranked[:n_leg]]
        winners = [t for t, _ in ranked[-n_leg:]]
        weights = {t: np.nan for t in UNIVERSE}
        for t in UNIVERSE:
            if t in winners:
                weights[t] = 1.0 / len(winners)
            elif t in losers:
                weights[t] = -1.0 / len(losers)
            elif t in formation_scores:
                weights[t] = 0.0
        rows[d] = weights
    return pd.DataFrame(rows).T


def simulate_xsmom(weight_df: pd.DataFrame, prices: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.Series:
    daily_w = weight_df.reindex(daily_index, method="ffill").shift(1)
    log_px = np.log(prices[UNIVERSE])
    daily_ret = log_px.diff().reindex(daily_index)
    port_ret = (daily_w * daily_ret).sum(axis=1, skipna=True)
    turnover = daily_w.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (_tm.COST_BPS / 10000.0)
    return port_ret - cost


def simulate_passive(prices: pd.DataFrame, daily_index: pd.DatetimeIndex, weight_df: pd.DataFrame) -> pd.Series:
    active = weight_df.reindex(daily_index, method="ffill").notna().shift(1).fillna(False)
    n_active = active.sum(axis=1).replace(0, np.nan)
    w = active.div(n_active, axis=0)
    log_px = np.log(prices[UNIVERSE])
    daily_ret = log_px.diff().reindex(daily_index)
    return (w * daily_ret).sum(axis=1, skipna=True)


def run_window(strat_ret, passive_ret, label):
    s_sharpe, p_sharpe = _tm.sharpe(strat_ret), _tm.sharpe(passive_ret)
    alpha = _tm.mean_diff_alpha(strat_ret, passive_ret)
    strat_cum = float((np.exp(strat_ret.fillna(0)).cumprod().iloc[-1] - 1) * 100) if len(strat_ret) else float("nan")
    passive_cum = float((np.exp(passive_ret.fillna(0)).cumprod().iloc[-1] - 1) * 100) if len(passive_ret) else float("nan")
    print(f"\n  [{label}] n_days={alpha['n_days']}")
    print(f"    XSMOM (long-short, dollar-neutral): cum_ret={strat_cum:+.1f}%  Sharpe={s_sharpe:.3f}")
    print(f"    Passive (same universe, long-only): cum_ret={passive_cum:+.1f}%  Sharpe={p_sharpe:.3f}")
    print(f"    Alpha vs passive (mean-return difference, annualized): {alpha['alpha_annualized_pct']:+.2f}%/yr")
    return {"xsmom_cum_ret_pct": strat_cum, "passive_cum_ret_pct": passive_cum,
            "xsmom_sharpe": s_sharpe, "passive_sharpe": p_sharpe, **alpha}


if __name__ == "__main__":
    prices = pd.read_parquet("multiasset_prices.parquet")
    daily_index = prices.index

    first_valid = min(prices[t].dropna().index.min() for t in UNIVERSE)
    start = first_valid + pd.Timedelta(days=int(1.5 * MIN_HISTORY_DAYS))
    rebal_dates = _tm.month_end_dates(daily_index[daily_index >= start])
    print(f"Universe: {len(UNIVERSE)} equity/sector instruments. First rebalance: {rebal_dates[0].date()}, "
          f"last: {rebal_dates[-1].date()} ({len(rebal_dates)} months)")

    weight_df = build_monthly_positions(prices, rebal_dates)
    trading_index = daily_index[daily_index >= rebal_dates[0]]

    xsmom_ret = simulate_xsmom(weight_df, prices, trading_index)
    passive_ret = simulate_passive(prices, trading_index, weight_df)

    results = {}
    results["full_sample"] = run_window(xsmom_ret, passive_ret, f"Full sample, {rebal_dates[0].date()}-{rebal_dates[-1].date()}")
    for label, start_d, end_d in [("Since 2010", "2010-01-01", None), ("Since 2020", "2020-01-01", None),
                                   ("Since 2022", "2022-01-01", None), ("2025 only", "2025-01-01", "2025-12-31")]:
        mask = trading_index >= pd.Timestamp(start_d)
        if end_d:
            mask &= trading_index <= pd.Timestamp(end_d)
        sub_idx = trading_index[mask]
        if len(sub_idx) < 60:
            continue
        results[label.lower().replace(" ", "_")] = run_window(xsmom_ret.reindex(sub_idx), passive_ret.reindex(sub_idx), label)

    with open(os.path.join(OUT_DIR, "xsmom_benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(12, 6))
    xsmom_eq = np.exp(xsmom_ret.fillna(0)).cumprod() * 100_000
    passive_eq = np.exp(passive_ret.fillna(0)).cumprod() * 100_000
    ax.plot(passive_eq.index, passive_eq.values, color="#9AA1AD", lw=1.1, label=f"Passive equal-weight (Sharpe {results['full_sample']['passive_sharpe']:.2f})")
    ax.plot(xsmom_eq.index, xsmom_eq.values, color="#2E6DA4", lw=1.1, label=f"XSMOM long-short (Sharpe {results['full_sample']['xsmom_sharpe']:.2f})")
    ax.set_yscale("log")
    ax.set_title(f"Cross-Sectional Momentum (Jegadeesh-Titman 1993 '12-1' spec, no fitted parameters)\n"
                 f"{len(UNIVERSE)}-instrument equity/sector universe, {rebal_dates[0].date()} to {rebal_dates[-1].date()}, net of {_tm.COST_BPS}bps\n"
                 f"Full-sample alpha vs passive (mean-return difference): {results['full_sample']['alpha_annualized_pct']:+.2f}%/yr")
    ax.set_ylabel("Equity ($100k notional, log scale)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "xsmom_benchmark_equity.png"), dpi=140)
    plt.close(fig)
    print("\nSaved: xsmom_benchmark_results.json, xsmom_benchmark_equity.png")
