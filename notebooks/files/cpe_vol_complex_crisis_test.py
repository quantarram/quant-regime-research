"""
cpe_vol_complex_crisis_test.py
================================
Tests CPE's own validated "vol-complex -> equity" channel (README: one of
the two channels that survived the economic-prior gate, alongside
crypto->gold) the same rigorous way TSMOM's crisis-alpha was just tested:
does it actually concentrate in crisis/stress episodes, and does a real,
cost-adjusted, HAC-corrected backtest of it hold up out-of-sample and
across multiple years -- or is it another single-window artifact?

The channel, as it actually exists in joint_cpe_results.parquet (not as a
hand-simplified proxy): predictors [TLH, ^GVZ] (a long-duration Treasury
ETF and gold's own option-implied volatility index) both in their own
95th-percentile upper tail over a trailing 126-day window, jointly predict
QQQ's own forward 126-day return with a BULLISH bias (joint_CPE=1.00,
lift=2.0, 105 supporting historical observations in the original,
full-sample-thresholded screen). Economically: elevated gold-vol AND
elevated long-bond prices together (both markets pricing in stress/flight-
to-safety) has historically been followed by an ABOVE-median subsequent
move in QQQ -- a contrarian, buy-the-stress signal, not a simple "vol
spike -> get defensive" one.

Method:
  1. Re-derive REAL firing dates for this exact configuration using
     backtest_engine.py's own threshold/firing machinery, with thresholds
     frozen at a single training cutoff (2013-12-31, giving ^GVZ/TLH's
     post-2008 history room to establish stable quantiles) and firing
     checked strictly out-of-sample from 2014 onward -- not the full-
     sample thresholds the original joint screen itself was built with.
  2. Check whether firing dates are enriched in QQQ's own crisis/drawdown
     periods (QQQ drawdown <= -10% from its running peak, same definition
     used for the TSMOM crisis-alpha check) relative to their base rate
     across all days.
  3. Build the actual tradeable strategy this signal implies -- long QQQ
     for 126 trading days from each new firing, flat otherwise -- and
     backtest it net of costs, both as one continuous OOS stretch and
     split year by year, with Newey-West HAC-corrected significance (the
     same correction applied to CPE hold-to-horizon), against QQQ's own
     buy-and-hold.

No significance/randomisation-test games -- real thresholds, real
firing dates, real P&L, reported honestly whichever way it comes out.

Run (from notebooks/files/):
    python cpe_vol_complex_crisis_test.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest_engine import build_increments_and_thresholds, configuration_fires_on_date

TRAIN_CUTOFF = pd.Timestamp("2013-12-31")
COST_BPS = 5
CRISIS_THRESHOLD = -0.10

# The exact config row from joint_cpe_results.parquet (index 2059):
CONFIG = pd.Series({
    "Y": "QQQ", "direction": "bullish",
    "predictors": ["TLH", "^GVZ"], "tau_pasts": [126, 126], "q_Xs": [0.95, 0.95],
    "tau_future": 126, "q_Y": 0.50,
})

Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]


def hac_alpha(strat_ret: pd.Series, bench_ret: pd.Series, maxlags: int) -> dict:
    common = strat_ret.index.intersection(bench_ret.index)
    y = strat_ret.reindex(common).fillna(0.0).values
    x = bench_ret.reindex(common).fillna(0.0).values
    if len(y) < 20 or np.std(y) == 0:
        return {"alpha_annualized_pct": float("nan"), "t_hac": float("nan"), "significant_95": False, "n": len(y)}
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {"alpha_annualized_pct": float(model.params[0] * 252 * 100), "t_hac": float(model.tvalues[0]),
            "significant_95": bool(abs(model.tvalues[0]) >= 1.96), "n": len(y)}


def sharpe(ret: pd.Series) -> float:
    r = ret.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")


if __name__ == "__main__":
    prices = pd.read_parquet("../multiasset_prices.parquet")

    import backtest_engine as _be
    _be.TRAIN_CUTOFF = TRAIN_CUTOFF
    increments, thresholds = build_increments_and_thresholds(prices, Q_GRID)

    eval_start = TRAIN_CUTOFF + pd.Timedelta(days=1)
    eval_dates = prices.index[(prices.index >= eval_start) & prices["QQQ"].notna()]
    print(f"Evaluating [TLH, ^GVZ] (95th pctile, 126d) -> QQQ (bullish, 126d fwd) "
          f"OOS from {eval_dates.min().date()} to {eval_dates.max().date()} ({len(eval_dates)} days)")

    fires = pd.Series(
        [configuration_fires_on_date(CONFIG, d, increments, thresholds) for d in eval_dates],
        index=eval_dates,
    )
    n_fire_days = int(fires.sum())
    print(f"Fires on {n_fire_days}/{len(eval_dates)} days ({n_fire_days/len(eval_dates)*100:.1f}%)")

    # -- crisis enrichment check --
    qqq = prices["QQQ"].reindex(eval_dates).ffill()
    qqq_ret = np.log(qqq).diff()
    running_peak = qqq.cummax()
    drawdown = qqq / running_peak - 1.0
    crisis_mask = drawdown <= CRISIS_THRESHOLD
    base_rate = crisis_mask.mean()
    fire_crisis_rate = fires[fires].index.isin(crisis_mask[crisis_mask].index)
    fire_in_crisis_rate = crisis_mask.reindex(fires[fires].index).mean() if n_fire_days else float("nan")
    print(f"\nBase rate: {base_rate*100:.1f}% of all days are 'crisis' (QQQ drawdown <= {CRISIS_THRESHOLD:.0%})")
    print(f"Of days this channel FIRES: {fire_in_crisis_rate*100:.1f}% are crisis days "
          f"({'ENRICHED' if fire_in_crisis_rate > base_rate * 1.3 else ('DEPLETED' if fire_in_crisis_rate < base_rate * 0.7 else 'roughly proportional')} "
          f"vs base rate, {fire_in_crisis_rate/base_rate:.2f}x)")

    # -- build the tradeable strategy: long QQQ for 126d from each NEW firing --
    newly_fires = fires & ~fires.shift(1).fillna(False)
    position = pd.Series(0.0, index=eval_dates)
    active_until = None
    for d in eval_dates:
        if active_until is not None and d <= active_until:
            position[d] = 1.0
        elif newly_fires.get(d, False):
            position[d] = 1.0
            active_until = d + pd.Timedelta(days=int(126 * 1.45))
        else:
            position[d] = 0.0
    n_episodes_opened = int(newly_fires.sum())
    print(f"\n{n_episodes_opened} new firing episodes opened, {int((position > 0).sum())} days with an active long position")

    applied = position.shift(1).fillna(0.0)
    turnover = applied.diff().abs().fillna(0.0)
    strat_ret = applied * qqq_ret - turnover * (COST_BPS / 10000.0)

    print(f"\n--- Full OOS period ({eval_dates.min().date()} to {eval_dates.max().date()}) ---")
    full_hac = hac_alpha(strat_ret, qqq_ret, maxlags=63)
    print(f"  Strategy Sharpe={sharpe(strat_ret):.3f}  QQQ buy-hold Sharpe={sharpe(qqq_ret):.3f}")
    print(f"  Alpha vs QQQ buy-hold: {full_hac['alpha_annualized_pct']:+.2f}%/yr  t_HAC63={full_hac['t_hac']:+.2f}  "
          f"[{'SIGNIFICANT' if full_hac['significant_95'] else 'not significant'}]")

    print(f"\n--- Crisis days vs calm days (this strategy's own returns) ---")
    for label, mask in [("crisis", crisis_mask), ("calm", ~crisis_mask)]:
        sub_s, sub_q = strat_ret[mask], qqq_ret[mask]
        print(f"  [{label:>6}] n={mask.sum():>5}  strategy ann_ret={sub_s.mean()*252*100:+7.2f}%  "
              f"Sharpe={sharpe(sub_s):+.2f}  |  QQQ ann_ret={sub_q.mean()*252*100:+7.2f}%  Sharpe={sharpe(sub_q):+.2f}")

    print(f"\n--- Year by year ---")
    years = sorted(set(eval_dates.year))
    for y in years:
        m = eval_dates.year == y
        if m.sum() < 60:
            continue
        yr_hac = hac_alpha(strat_ret[m], qqq_ret[m], maxlags=21)
        n_fires_yr = int(newly_fires[m].sum())
        print(f"  {y}: alpha={yr_hac['alpha_annualized_pct']:+7.2f}%/yr  t_HAC21={yr_hac['t_hac']:+.2f}  "
              f"[{'SIGNIFICANT' if yr_hac['significant_95'] else 'not significant'}]  "
              f"new episodes opened={n_fires_yr}  crisis days in year={int(crisis_mask[m].sum())}")
