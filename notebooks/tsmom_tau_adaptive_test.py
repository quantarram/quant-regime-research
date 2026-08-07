"""
tsmom_tau_adaptive_test.py
=============================
Builds on tonight's TSMOM control strategy with the one thing this whole
research program has that the published literature doesn't: independently
measured, per-instrument predictability limits (tau*, Ramanathan 2026a).

Standard TSMOM uses ONE uniform lookback (12 months) and vol window (3
months) for every instrument -- an industry-standard choice, not tailored
to any instrument's own dynamics. This tests a direct, principled
alternative: use each instrument's own tau* as its trend-signal lookback,
and its own half_window = floor(tau*/2) as its vol-estimation window --
the exact "honest, non-stale" convention Papers 13-15 already validated
for prediction, applied here to trend-signal construction instead. A
trend measured over a window LONGER than tau* is partly built on data the
instrument's own measured dynamics say has already decorrelated.

Nothing else changes: same monthly rebalance calendar, same vol-target
sizing formula, same leverage cap, same cost model, same benchmark
(passive equal-weight of the identical universe) as tsmom_benchmark.py.
Restricted to the 12 instruments with an existing, already-published tau*
(Ramanathan 2026a/c) -- not re-estimated or fit here, just reused --
rather than a new estimation exercise on the full 29-instrument universe.

No significance/randomisation-test games -- Newey-West HAC-corrected
comparison, same multi-year discipline as every other test tonight,
reported honestly whichever way it comes out.

Run: python tsmom_tau_adaptive_test.py
Output: tsmom_tau_adaptive_results.json, tsmom_tau_adaptive_equity.png
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COST_BPS = 5
TARGET_VOL = 0.10
MAX_LEVERAGE = 2.0
BASE_LOOKBACK, BASE_VOL_WINDOW = 252, 63

INSTRUMENTS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "AAPL", "MSFT", "JPM", "XOM", "GLD", "EURUSD=X"]


def load_tau_star():
    d = json.load(open("predictability_paper/results_correlated_decorrelated.json"))
    return {t: d[t]["2"]["top5_tradeable"][0][0] for t in INSTRUMENTS}


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).last().values)


def build_monthly_signals(prices: pd.DataFrame, rebal_dates: pd.DatetimeIndex, lookbacks: dict, vol_windows: dict) -> pd.DataFrame:
    log_px = np.log(prices[INSTRUMENTS])
    daily_ret = log_px.diff()
    rows = {}
    for d in rebal_dates:
        weights = {}
        for tkr in INSTRUMENTS:
            lb, vw = lookbacks[tkr], vol_windows[tkr]
            s = prices[tkr].loc[:d].dropna()
            if len(s) < lb + vw + 5:
                weights[tkr] = np.nan
                continue
            trail_ret = np.log(s.iloc[-1] / s.iloc[-lb])
            r = daily_ret[tkr].dropna().loc[:d].tail(vw)
            vol = r.std() * np.sqrt(252)
            if not np.isfinite(vol) or vol <= 1e-6:
                weights[tkr] = np.nan
                continue
            raw_w = np.sign(trail_ret) * (TARGET_VOL / vol)
            weights[tkr] = float(np.clip(raw_w, -MAX_LEVERAGE, MAX_LEVERAGE))
        rows[d] = weights
    return pd.DataFrame(rows).T


def simulate(weight_df: pd.DataFrame, prices: pd.DataFrame, daily_index: pd.DatetimeIndex, passive: bool = False) -> pd.Series:
    daily_w = weight_df.reindex(daily_index, method="ffill").shift(1)
    log_px = np.log(prices[INSTRUMENTS])
    daily_ret = log_px.diff().reindex(daily_index)
    active = daily_w.notna()
    n_active = active.sum(axis=1).replace(0, np.nan)
    per_instrument_w = active.div(n_active, axis=0) if passive else daily_w.div(n_active, axis=0)
    port_ret = (per_instrument_w * daily_ret).sum(axis=1, skipna=True)
    turnover = per_instrument_w.diff().abs().sum(axis=1).fillna(0.0)
    return port_ret - turnover * (COST_BPS / 10000.0)


def hac_alpha(strat_ret: pd.Series, bench_ret: pd.Series, maxlags: int) -> dict:
    common = strat_ret.index.intersection(bench_ret.index)
    y = strat_ret.reindex(common).fillna(0.0).values
    x = bench_ret.reindex(common).fillna(0.0).values
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {"alpha_annualized_pct": float(model.params[0] * 252 * 100), "t_hac": float(model.tvalues[0]),
            "significant_95": bool(abs(model.tvalues[0]) >= 1.96), "n": len(y)}


def sharpe(ret: pd.Series) -> float:
    r = ret.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")


if __name__ == "__main__":
    prices = pd.read_parquet("multiasset_prices.parquet")
    n_bad = int((prices[INSTRUMENTS] <= 0).sum().sum())
    if n_bad:
        prices = prices.copy()
        prices[INSTRUMENTS] = prices[INSTRUMENTS].mask(prices[INSTRUMENTS] <= 0)

    tau_star = load_tau_star()
    print("Per-instrument tau* (trend lookback) and half_window (vol window) vs. baseline TSMOM's uniform 252d/63d:")
    for t in INSTRUMENTS:
        print(f"  {t:<10} tau*={tau_star[t]:>3}d  half_window={max(tau_star[t]//2,5):>3}d")

    baseline_lb = {t: BASE_LOOKBACK for t in INSTRUMENTS}
    baseline_vw = {t: BASE_VOL_WINDOW for t in INSTRUMENTS}
    tau_lb = {t: tau_star[t] for t in INSTRUMENTS}
    tau_vw = {t: max(tau_star[t] // 2, 5) for t in INSTRUMENTS}

    daily_index = prices.index
    first_valid = min(prices[t].dropna().index.min() for t in INSTRUMENTS)
    start = first_valid + pd.Timedelta(days=int(1.5 * (BASE_LOOKBACK + BASE_VOL_WINDOW)))
    rebal_dates = month_end_dates(daily_index[daily_index >= start])
    trading_index = daily_index[daily_index >= rebal_dates[0]]
    print(f"\nEvaluation: {rebal_dates[0].date()} to {rebal_dates[-1].date()} ({len(rebal_dates)} months), {len(INSTRUMENTS)} instruments")

    baseline_weights = build_monthly_signals(prices, rebal_dates, baseline_lb, baseline_vw)
    tau_weights = build_monthly_signals(prices, rebal_dates, tau_lb, tau_vw)

    baseline_ret = simulate(baseline_weights, prices, trading_index, passive=False)
    tau_ret = simulate(tau_weights, prices, trading_index, passive=False)
    passive_ret = simulate(baseline_weights, prices, trading_index, passive=True)

    results = {}
    windows = [("full_sample", rebal_dates[0], rebal_dates[-1]), ("since_2022", pd.Timestamp("2022-01-01"), rebal_dates[-1]),
               ("2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")), ("2023", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
               ("2024", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")), ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))]

    print(f"\n{'window':<14}{'baseline TSMOM (252d/63d)':<38}{'tau*-adaptive TSMOM':<38}")
    for label, start_d, end_d in windows:
        mask = (trading_index >= start_d) & (trading_index <= end_d)
        idx = trading_index[mask]
        if len(idx) < 40:
            continue
        b_hac = hac_alpha(baseline_ret.reindex(idx), passive_ret.reindex(idx), maxlags=21)
        t_hac = hac_alpha(tau_ret.reindex(idx), passive_ret.reindex(idx), maxlags=21)
        b_str = f"Sharpe={sharpe(baseline_ret.reindex(idx)):.2f} alpha={b_hac['alpha_annualized_pct']:+.2f}%/yr t={b_hac['t_hac']:+.2f}{'*' if b_hac['significant_95'] else ''}"
        t_str = f"Sharpe={sharpe(tau_ret.reindex(idx)):.2f} alpha={t_hac['alpha_annualized_pct']:+.2f}%/yr t={t_hac['t_hac']:+.2f}{'*' if t_hac['significant_95'] else ''}"
        print(f"{label:<14}{b_str:<38}{t_str:<38}")
        results[label] = {"baseline": b_hac, "tau_adaptive": t_hac,
                           "baseline_sharpe": sharpe(baseline_ret.reindex(idx)), "tau_adaptive_sharpe": sharpe(tau_ret.reindex(idx))}

    head_to_head = hac_alpha(tau_ret, baseline_ret, maxlags=21)
    print(f"\nDirect head-to-head, full sample (tau*-adaptive vs. baseline TSMOM as the benchmark): "
          f"alpha={head_to_head['alpha_annualized_pct']:+.2f}%/yr  t={head_to_head['t_hac']:+.2f}  "
          f"[{'SIGNIFICANT' if head_to_head['significant_95'] else 'not significant'}]")
    results["head_to_head_vs_baseline"] = head_to_head

    with open("tsmom_tau_adaptive_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(12, 6))
    for ret, label, color in [(passive_ret, f"Passive equal-weight (Sharpe {sharpe(passive_ret):.2f})", "#9AA1AD"),
                               (baseline_ret, f"TSMOM baseline, 252d/63d (Sharpe {sharpe(baseline_ret):.2f})", "#2E6DA4"),
                               (tau_ret, f"TSMOM tau*-adaptive (Sharpe {sharpe(tau_ret):.2f})", "#2f8a4e")]:
        eq = np.exp(ret.fillna(0)).cumprod() * 100_000
        ax.plot(eq.index, eq.values, label=label, lw=1.2, color=color)
    ax.set_yscale("log")
    ax.set_title(f"TSMOM: uniform 252d/63d lookback vs. each instrument's own tau*/half_window\n"
                 f"{len(INSTRUMENTS)} instruments with published predictability limits, net of {COST_BPS}bps")
    ax.set_ylabel("Equity ($100k notional, log scale)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("tsmom_tau_adaptive_equity.png", dpi=140)
    plt.close(fig)
    print("\nSaved: tsmom_tau_adaptive_results.json, tsmom_tau_adaptive_equity.png")
