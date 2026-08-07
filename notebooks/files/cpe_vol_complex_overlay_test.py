"""
cpe_vol_complex_overlay_test.py
=================================
Reframes the vol-complex crisis-detector test: not "does this signal make
money on its own" (already answered: no, -0.24%/yr vs QQQ buy-and-hold,
not significant) but "does de-risking specifically while this signal is
active reduce a long-only book's drawdown and improve its risk-adjusted
return" -- the actual product a tail-risk/overlay desk would care about,
independent of whether the signal itself has standalone alpha.

Important context established before running this, not after: QQQ's
single WORST drawdown across the 2014-2026 test window is -35.1%, set in
the 2022 bear market (2022-11-03) -- NOT the -28.6% 2020 COVID crash. The
vol-complex signal never fired once in 2022 (confirmed in
cpe_vol_complex_crisis_test.py: 0 new episodes that year, despite 232
crisis days). So this overlay, by construction, cannot protect against
the single largest drawdown in the test period -- it can only ever help
with 2020, the second-worst one. That's disclosed here directly, not
buried in a footnote, because it bears directly on how any drawdown
reduction below should be read.

Overlay mechanism: 100% QQQ exposure at all times, EXCEPT during the
signal's active firing episodes (the same 5 episodes, ~127 days,
identified in cpe_vol_complex_crisis_test.py), when exposure is cut to
DEFENSIVE_WEIGHT (tested at 0% -- full de-risk to cash -- and 50% --
partial de-risk, the standard practitioner alternative to an all-or-
nothing hedge). Compared against static 100% QQQ buy-and-hold on the
metrics that actually matter for an overlay: max drawdown, annualized
vol, Sharpe, Sortino (downside-deviation-based), and Calmar (return /
max drawdown) -- not raw alpha, which is not what this framing claims to
produce.

No significance/randomisation-test games -- one real overlay, real
metrics, reported honestly whichever way it comes out.

Run (from notebooks/files/):
    python cpe_vol_complex_overlay_test.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import backtest_engine as _be
from backtest_engine import build_increments_and_thresholds, configuration_fires_on_date

TRAIN_CUTOFF = pd.Timestamp("2013-12-31")
COST_BPS = 5
DEFENSIVE_WEIGHTS = [0.0, 0.5]

CONFIG = pd.Series({
    "Y": "QQQ", "direction": "bullish",
    "predictors": ["TLH", "^GVZ"], "tau_pasts": [126, 126], "q_Xs": [0.95, 0.95],
    "tau_future": 126, "q_Y": 0.50,
})
Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def sharpe(ret: pd.Series) -> float:
    r = ret.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")


def sortino(ret: pd.Series) -> float:
    r = ret.dropna()
    downside = r[r < 0]
    dd_std = downside.std()
    return float(r.mean() / dd_std * np.sqrt(252)) if dd_std and dd_std > 0 else float("nan")


def calmar(ret: pd.Series, equity: pd.Series) -> float:
    ann_ret = float(ret.mean() * 252)
    mdd = abs(max_drawdown(equity))
    return float(ann_ret / mdd) if mdd > 0 else float("nan")


def full_stats(ret: pd.Series, label: str) -> dict:
    equity = np.exp(ret.fillna(0)).cumprod()
    return {
        "label": label, "ann_ret_pct": float(ret.mean() * 252 * 100), "ann_vol_pct": float(ret.std() * np.sqrt(252) * 100),
        "sharpe": sharpe(ret), "sortino": sortino(ret), "max_dd_pct": max_drawdown(equity) * 100,
        "calmar": calmar(ret, equity), "total_ret_pct": float((equity.iloc[-1] - 1) * 100),
    }


def print_stats(s: dict):
    print(f"  {s['label']:<28} total_ret={s['total_ret_pct']:+8.1f}%  ann_ret={s['ann_ret_pct']:+6.2f}%  "
          f"ann_vol={s['ann_vol_pct']:6.2f}%  Sharpe={s['sharpe']:+.2f}  Sortino={s['sortino']:+.2f}  "
          f"MaxDD={s['max_dd_pct']:7.2f}%  Calmar={s['calmar']:+.2f}")


if __name__ == "__main__":
    prices = pd.read_parquet("../multiasset_prices.parquet")
    _be.TRAIN_CUTOFF = TRAIN_CUTOFF
    increments, thresholds = build_increments_and_thresholds(prices, Q_GRID)

    eval_start = TRAIN_CUTOFF + pd.Timedelta(days=1)
    eval_dates = prices.index[(prices.index >= eval_start) & prices["QQQ"].notna()]

    fires = pd.Series(
        [configuration_fires_on_date(CONFIG, d, increments, thresholds) for d in eval_dates],
        index=eval_dates,
    )
    newly_fires = fires & ~fires.shift(1).fillna(False)

    active = pd.Series(False, index=eval_dates)
    active_until = None
    for d in eval_dates:
        if active_until is not None and d <= active_until:
            active[d] = True
        elif newly_fires.get(d, False):
            active[d] = True
            active_until = d + pd.Timedelta(days=int(126 * 1.45))

    qqq = prices["QQQ"].reindex(eval_dates).ffill()
    qqq_ret = np.log(qqq).diff()

    print(f"Overlay active (signal firing) on {int(active.sum())} of {len(eval_dates)} days "
          f"({active.mean()*100:.1f}%), all within {int(newly_fires.sum())} episodes")
    print(f"\nKnown up front: QQQ's single worst drawdown in this window is -35.1% (2022-11-03, the 2022 bear "
          f"market), NOT 2020's -28.6% COVID crash. This signal never fired in 2022. It can only ever affect "
          f"how the 2020 episode looks, not the period's actual worst episode.\n")

    baseline_ret = qqq_ret.copy()
    baseline_stats = full_stats(baseline_ret, "Baseline (100% QQQ always)")
    print_stats(baseline_stats)

    for w in DEFENSIVE_WEIGHTS:
        position = pd.Series(1.0, index=eval_dates)
        position[active] = w
        applied = position.shift(1).fillna(1.0)
        turnover = applied.diff().abs().fillna(0.0)
        overlay_ret = applied * qqq_ret - turnover * (COST_BPS / 10000.0)
        s = full_stats(overlay_ret, f"Overlay (defensive weight={w:.0%})")
        print_stats(s)

    print("\n--- Same comparison, restricted to the 2020 episode only (where the signal actually acted) ---")
    mask_2020 = eval_dates.year == 2020
    print_stats(full_stats(baseline_ret[mask_2020], "Baseline, 2020 only"))
    for w in DEFENSIVE_WEIGHTS:
        position = pd.Series(1.0, index=eval_dates)
        position[active] = w
        applied = position.shift(1).fillna(1.0)
        turnover = applied.diff().abs().fillna(0.0)
        overlay_ret = applied * qqq_ret - turnover * (COST_BPS / 10000.0)
        print_stats(full_stats(overlay_ret[mask_2020], f"Overlay, 2020 only (w={w:.0%})"))

    print("\n--- Same comparison, restricted to the 2022 episode (signal never fired) ---")
    mask_2022 = eval_dates.year == 2022
    print_stats(full_stats(baseline_ret[mask_2022], "Baseline, 2022 only"))
    for w in DEFENSIVE_WEIGHTS:
        position = pd.Series(1.0, index=eval_dates)
        position[active] = w
        applied = position.shift(1).fillna(1.0)
        turnover = applied.diff().abs().fillna(0.0)
        overlay_ret = applied * qqq_ret - turnover * (COST_BPS / 10000.0)
        print_stats(full_stats(overlay_ret[mask_2022], f"Overlay, 2022 only (w={w:.0%})"))
