"""
tsmom_benchmark.py
===================
Control experiment: before concluding this program's own signals can't
generate alpha, replicate a genuinely well-established, published,
out-of-sample-validated quant strategy on the SAME instrument universe and
data source (multiasset_prices.parquet), with NO fitted parameters, and
see whether it produces real, honest alpha here. If it does, that tells us
something concrete about what's different in our own approach (breadth,
construction, universe). If it doesn't either, that's real evidence this
isn't a methodology problem specific to CPE or predictor_v1 -- it's these
specific instruments, this specific period, or genuine market efficiency.

Strategy: Time-Series Momentum (TSMOM), Moskowitz, Ooi & Pedersen (2012,
"Time Series Momentum", Journal of Financial Economics) -- the actual
textbook specification behind a large share of the managed-futures/CTA
industry's real, multi-decade track record (AQR, Man AHL, Winton, etc.),
not a strategy invented for this program:

  - signal(i, t)   = sign(12-month trailing return of instrument i)
  - vol(i, t)      = 3-month trailing realized daily-return volatility,
                     annualized
  - weight(i, t)   = signal(i, t) * TARGET_VOL / vol(i, t), capped at
                     MAX_LEVERAGE per instrument (a standard, disclosed
                     practitioner modification -- uncapped inverse-vol
                     sizing blows up during low-vol regimes)
  - portfolio(t)   = equal-weighted average of all active instruments'
                     vol-scaled positions
  - rebalanced monthly (the paper's own cadence), positions formed from
    information available at the END of the prior month, applied with a
    1-trading-day lag (same discipline as backtest_engine.py)

Every parameter above (12m lookback, 3m vol window, monthly rebalance) is
the paper's own published specification, not fit to this data -- the
whole point of this test is that nothing here is tuned.

Universe: 29 liquid, multi-asset-class instruments already in this
program's own price panel (equity indices, sector ETFs, international
equities, commodities, a bond ETF, crypto, FX) -- real breadth, spanning
asset classes, the structural ingredient this program's own signals have
mostly lacked (at most a handful of sleeves/instruments at a time).

Benchmark (the correct "own benchmark" for a multi-asset timing book,
analogous to Jensen's alpha vs. an instrument's own buy-and-hold used
elsewhere in this program): a static, equal-weighted, monthly-rebalanced
LONG-ONLY portfolio of the exact same universe. The question is whether
TIMING (trend direction) adds value over simply holding the same
instruments passively -- not whether the asset class itself went up.

No significance/randomisation-test games -- Jensen's alpha (OLS intercept
of strategy returns on the passive benchmark's returns), one real
comparison, reported honestly whichever way it comes out.

Run: python tsmom_benchmark.py
Output: tsmom_benchmark_results.json, tsmom_benchmark_equity.png
"""
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
COST_BPS = 5
LOOKBACK_DAYS = 252
VOL_WINDOW_DAYS = 63
TARGET_VOL = 0.10       # 10% annualized per-instrument vol target, standard practitioner simplification
MAX_LEVERAGE = 2.0       # cap on |weight| per instrument, disclosed modification (uncapped inverse-vol sizing is unstable)
MIN_HISTORY_DAYS = LOOKBACK_DAYS + VOL_WINDOW_DAYS + 21  # must have enough history to form a signal at all

UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU",
    "EWJ", "EWY", "EWZ",
    "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZW=F", "ZS=F",
    "TLT", "BTC-USD", "EURUSD=X", "JPYUSD=X", "UUP",
]


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    df = pd.Series(index, index=index)
    return pd.DatetimeIndex(df.groupby([index.year, index.month]).last().values)


def build_monthly_signals(prices: pd.DataFrame, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """weight(i, rebal_date), formed from info available strictly at/before
    that rebal date (month-end trailing windows), one row per rebalance
    date. NaN where an instrument has insufficient history yet."""
    log_px = np.log(prices[UNIVERSE])
    daily_ret = log_px.diff()

    rows = {}
    for d in rebal_dates:
        px_hist = prices[UNIVERSE].loc[:d]
        weights = {}
        for tkr in UNIVERSE:
            s = px_hist[tkr].dropna()
            if len(s) < MIN_HISTORY_DAYS:
                weights[tkr] = np.nan
                continue
            trail_ret = np.log(s.iloc[-1] / s.iloc[-LOOKBACK_DAYS])
            r = daily_ret[tkr].dropna().loc[:d].tail(VOL_WINDOW_DAYS)
            vol = r.std() * np.sqrt(252)
            if not np.isfinite(vol) or vol <= 1e-6:
                weights[tkr] = np.nan
                continue
            sig = np.sign(trail_ret)
            raw_w = sig * (TARGET_VOL / vol)
            weights[tkr] = float(np.clip(raw_w, -MAX_LEVERAGE, MAX_LEVERAGE))
        rows[d] = weights
    return pd.DataFrame(rows).T


def simulate(weight_df: pd.DataFrame, prices: pd.DataFrame, daily_index: pd.DatetimeIndex,
             equal_weight_passive: bool = False) -> pd.Series:
    """weight_df: rows=rebalance dates, cols=instruments, vol-scaled target
    weights. Forward-filled to daily, applied with a 1-day lag, equal-
    weighted average across active (non-NaN) instruments each day, net of
    turnover cost on rebalance days."""
    daily_w = weight_df.reindex(daily_index, method="ffill")
    daily_w = daily_w.shift(1)  # 1-day lag: today's position was DECIDED using info through yesterday's close

    log_px = np.log(prices[UNIVERSE])
    daily_ret = log_px.diff().reindex(daily_index)

    active = daily_w.notna()
    n_active = active.sum(axis=1).replace(0, np.nan)
    if equal_weight_passive:
        per_instrument_w = active.div(n_active, axis=0)  # simple equal weight, sign always +1, no vol scaling
    else:
        # Equal-weighted AVERAGE of each instrument's vol-scaled position,
        # as documented -- dividing by n_active is what keeps a 29-name
        # book from silently running at up to MAX_LEVERAGE * 29 gross
        # exposure; without it this is not the equal-weighted combination
        # the docstring (and the published TSMOM spec) describes.
        per_instrument_w = daily_w.div(n_active, axis=0)

    port_ret = (per_instrument_w * daily_ret).sum(axis=1, skipna=True)

    turnover = per_instrument_w.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (COST_BPS / 10000.0)
    net_ret = port_ret - cost
    return net_ret


def mean_diff_alpha(strat_ret: pd.Series, bench_ret: pd.Series) -> dict:
    """Real point-estimate alpha: the raw annualized difference in mean
    daily return, no fitted coefficient, no standard error, no
    significance verdict. Deliberately NOT an OLS/Jensen's-alpha
    regression -- that family of statistic assumes a single constant,
    stationary beta over the whole window and is estimated by minimizing
    squared residuals, both of which are exactly the mainstream Gaussian-
    adjacent machinery this project's methodology (Section 2 of the
    Paper 16 draft) rejects, regardless of sample size. See
    tsmom_stat_choice_check.py for a direct, disclosed demonstration of
    why: on the smaller per-instrument samples used elsewhere in that
    paper, Jensen's alpha flips sign relative to this statistic on more
    than half of a 22-instrument panel."""
    common = strat_ret.index.intersection(bench_ret.index)
    y = strat_ret.reindex(common).fillna(0.0)
    x = bench_ret.reindex(common).fillna(0.0)
    return {
        "alpha_annualized_pct": float((y.mean() - x.mean()) * 252 * 100),
        "n_days": int(len(common)),
    }


def sharpe(ret: pd.Series) -> float:
    r = ret.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")


def run_window(strat_ret, passive_ret, label):
    s_sharpe, p_sharpe = sharpe(strat_ret), sharpe(passive_ret)
    alpha = mean_diff_alpha(strat_ret, passive_ret)
    strat_cum = float((np.exp(strat_ret.fillna(0)).cumprod().iloc[-1] - 1) * 100) if len(strat_ret) else float("nan")
    passive_cum = float((np.exp(passive_ret.fillna(0)).cumprod().iloc[-1] - 1) * 100) if len(passive_ret) else float("nan")
    print(f"\n  [{label}] n_days={alpha['n_days']}")
    print(f"    TSMOM:   cum_ret={strat_cum:+.1f}%  Sharpe={s_sharpe:.3f}")
    print(f"    Passive: cum_ret={passive_cum:+.1f}%  Sharpe={p_sharpe:.3f}")
    print(f"    Alpha vs passive (mean-return difference, annualized): {alpha['alpha_annualized_pct']:+.2f}%/yr")
    return {
        "tsmom_cum_ret_pct": strat_cum, "passive_cum_ret_pct": passive_cum,
        "tsmom_sharpe": s_sharpe, "passive_sharpe": p_sharpe, **alpha,
    }


if __name__ == "__main__":
    prices = pd.read_parquet("multiasset_prices.parquet")
    # WTI crude futures (CL=F) printed a NEGATIVE settlement price on
    # 2020-04-20 (the widely-documented April 2020 negative-oil-price
    # event, a real contract-mechanics artifact, not a data error to
    # silently trust). log() of a non-positive price is undefined and
    # corrupts every downstream return/vol/leverage calculation that
    # touches it. Treated as missing, same as this program's convention
    # elsewhere for genuinely invalid observations -- not smoothed,
    # interpolated, or otherwise disguised.
    n_bad = int((prices <= 0).sum().sum())
    if n_bad:
        print(f"Masking {n_bad} non-positive price observation(s) as missing (e.g. CL=F's 2020-04-20 negative print)...")
        prices = prices.mask(prices <= 0)
    daily_index = prices.index

    first_valid = min(prices[t].dropna().index.min() for t in UNIVERSE)
    start = first_valid + pd.Timedelta(days=int(1.5 * (LOOKBACK_DAYS + VOL_WINDOW_DAYS)))
    rebal_dates = month_end_dates(daily_index[(daily_index >= start)])
    print(f"Universe: {len(UNIVERSE)} instruments. First rebalance: {rebal_dates[0].date()}, "
          f"last: {rebal_dates[-1].date()} ({len(rebal_dates)} months)")

    weight_df = build_monthly_signals(prices, rebal_dates)
    trading_index = daily_index[daily_index >= rebal_dates[0]]

    tsmom_ret = simulate(weight_df, prices, trading_index, equal_weight_passive=False)
    passive_ret = simulate(weight_df, prices, trading_index, equal_weight_passive=True)

    results = {}
    results["full_sample"] = run_window(tsmom_ret, passive_ret, f"Full sample, {rebal_dates[0].date()}-{rebal_dates[-1].date()}")

    for label, start_d in [("Since 2010", "2010-01-01"), ("Since 2020", "2020-01-01"), ("2025 only", "2025-01-01")]:
        mask = trading_index >= pd.Timestamp(start_d)
        end_mask = trading_index <= pd.Timestamp("2025-12-31") if label == "2025 only" else pd.Series(True, index=trading_index).values
        sub_idx = trading_index[mask & (end_mask if isinstance(end_mask, np.ndarray) else np.array(end_mask))]
        if len(sub_idx) < 60:
            continue
        results[label.lower().replace(" ", "_")] = run_window(
            tsmom_ret.reindex(sub_idx), passive_ret.reindex(sub_idx), label)

    with open(os.path.join(OUT_DIR, "tsmom_benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(12, 6))
    tsmom_eq = np.exp(tsmom_ret.fillna(0)).cumprod() * 100_000
    passive_eq = np.exp(passive_ret.fillna(0)).cumprod() * 100_000
    ax.plot(tsmom_eq.index, tsmom_eq.values, label=f"TSMOM (Sharpe {results['full_sample']['tsmom_sharpe']:.2f})", color="#2E6DA4", lw=1.2)
    ax.plot(passive_eq.index, passive_eq.values, label=f"Passive equal-weight (Sharpe {results['full_sample']['passive_sharpe']:.2f})", color="#9AA1AD", lw=1.2)
    ax.set_yscale("log")
    ax.set_title(f"Time-Series Momentum (Moskowitz-Ooi-Pedersen 2012 spec, no fitted parameters)\n"
                 f"{len(UNIVERSE)}-instrument multi-asset universe, {rebal_dates[0].date()} to {rebal_dates[-1].date()}, "
                 f"net of {COST_BPS}bps costs\n"
                 f"Full-sample alpha vs passive (mean-return difference): {results['full_sample']['alpha_annualized_pct']:+.2f}%/yr")
    ax.set_ylabel("Equity ($100k notional, log scale)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tsmom_benchmark_equity.png"), dpi=140)
    plt.close(fig)
    print("\nSaved: tsmom_benchmark_results.json, tsmom_benchmark_equity.png")
