"""
tsmom_cpe_crisis_combo_test.py
================================
Lever 2 from the "how do we push this further" discussion: TSMOM has
real, HAC-significant crisis alpha (+2.57%/yr full-sample, t=+3.22) but
pays a persistent drag during calm markets (+4.33%/yr vs. passive's
+10.94%/yr on non-crisis days, confirmed in tsmom_crisis_alpha_check.py).
CPE's own validated vol-complex channel ([TLH, ^GVZ] -> QQQ) is a real,
4.1x crisis-enriched detector, but fires far too rarely (5 episodes in 12
years) to be a strategy or a useful overlay on its own
(cpe_vol_complex_overlay_test.py). Neither works alone. This tests
whether combining them does: use CPE's crisis signal to SCALE UP TSMOM's
exposure specifically when it fires, rather than running TSMOM at a
constant weight all the time.

Design, kept deliberately simple and disclosed rather than tuned: TSMOM
position sizing is UNCHANGED (same signal, same vol-target sizing) except
that overall book leverage is multiplied by a SCALE_UP factor (tested at
1.5x and 2.0x, two round, non-optimized choices, not searched over a
grid) specifically on days CPE's channel is actively firing (the exact
same firing-episode logic as cpe_vol_complex_overlay_test.py). This tests
one specific, economically motivated hypothesis -- crisis-alpha
strategies should be sized UP exactly when a real crisis is starting, not
run at constant exposure -- not a parameter search for the best multiplier.

CPE's channel targets QQQ specifically and only fired in 2020 within the
12-year OOS window already established. TSMOM is evaluated on its own
29-instrument multi-asset universe (tsmom_benchmark.py), unchanged. The
overlay is applied to TSMOM's AGGREGATE exposure (all legs scaled
together), not just its QQQ-related position, since CPE's channel is
read here as a signal about "is a real market-wide crisis underway", not
a QQQ-specific timing call.

No significance/randomisation-test games -- Newey-West HAC-corrected
comparison, same years as every other test tonight, reported honestly
whichever way it comes out.

Run (from notebooks/files/):
    python tsmom_cpe_crisis_combo_test.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest_engine import build_increments_and_thresholds, configuration_fires_on_date
import backtest_engine as _be

sys.path.insert(0, "..")
import tsmom_crisis_alpha_check as _tmc

TRAIN_CUTOFF = pd.Timestamp("2013-12-31")
SCALE_FACTORS = [1.5, 2.0]
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


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min())


if __name__ == "__main__":
    prices = pd.read_parquet("../multiasset_prices.parquet")

    print("Rebuilding TSMOM and its passive benchmark (unchanged spec)...")
    _here = os.getcwd()
    os.chdir("..")
    tsmom_ret, tsmom_passive = _tmc.build_series()
    os.chdir(_here)

    print("Rebuilding CPE crisis-channel firing dates ([TLH, ^GVZ] -> QQQ, 2014-2026 OOS)...")
    _be.TRAIN_CUTOFF = TRAIN_CUTOFF
    increments, thresholds = build_increments_and_thresholds(prices, Q_GRID)
    eval_start = TRAIN_CUTOFF + pd.Timedelta(days=1)
    signal_dates = prices.index[(prices.index >= eval_start) & prices["QQQ"].notna()]
    fires = pd.Series(
        [configuration_fires_on_date(CONFIG, d, increments, thresholds) for d in signal_dates],
        index=signal_dates,
    )
    newly_fires = fires & ~fires.shift(1).fillna(False)
    active = pd.Series(False, index=signal_dates)
    active_until = None
    for d in signal_dates:
        if active_until is not None and d <= active_until:
            active[d] = True
        elif newly_fires.get(d, False):
            active[d] = True
            active_until = d + pd.Timedelta(days=int(126 * 1.45))
    print(f"  CPE crisis signal active on {int(active.sum())} days, {int(newly_fires.sum())} episodes "
          f"(all within the window already established as 2020-only)")

    common_idx = tsmom_ret.index.intersection(active.index)
    tsmom_ret, tsmom_passive = tsmom_ret.reindex(common_idx), tsmom_passive.reindex(common_idx)
    active = active.reindex(common_idx).fillna(False)

    print(f"\n{'='*80}\n  Baseline: TSMOM alone, common eval window {common_idx.min().date()} to {common_idx.max().date()}\n{'='*80}")
    baseline_hac = hac_alpha(tsmom_ret, tsmom_passive, maxlags=63)
    baseline_eq = np.exp(tsmom_ret.fillna(0)).cumprod()
    print(f"  TSMOM alone: Sharpe={sharpe(tsmom_ret):.3f}  MaxDD={max_drawdown(baseline_eq)*100:.2f}%  "
          f"alpha vs passive={baseline_hac['alpha_annualized_pct']:+.2f}%/yr  t_HAC63={baseline_hac['t_hac']:+.2f}  "
          f"[{'SIGNIFICANT' if baseline_hac['significant_95'] else 'not significant'}]")

    for scale in SCALE_FACTORS:
        multiplier = pd.Series(1.0, index=common_idx)
        multiplier[active] = scale
        applied_mult = multiplier.shift(1).fillna(1.0)  # 1-day lag, consistent with every other test
        combo_ret = tsmom_ret * applied_mult
        combo_hac = hac_alpha(combo_ret, tsmom_passive, maxlags=63)
        combo_eq = np.exp(combo_ret.fillna(0)).cumprod()
        print(f"\n  TSMOM x CPE-crisis-scaled ({scale}x on {int(active.sum())} active days): "
              f"Sharpe={sharpe(combo_ret):.3f}  MaxDD={max_drawdown(combo_eq)*100:.2f}%  "
              f"alpha vs passive={combo_hac['alpha_annualized_pct']:+.2f}%/yr  t_HAC63={combo_hac['t_hac']:+.2f}  "
              f"[{'SIGNIFICANT' if combo_hac['significant_95'] else 'not significant'}]")

        # isolate the effect specifically within the 2020 window where the signal actually acted
        mask_2020 = common_idx.year == 2020
        b2020 = hac_alpha(tsmom_ret[mask_2020], tsmom_passive[mask_2020], maxlags=21)
        c2020 = hac_alpha(combo_ret[mask_2020], tsmom_passive[mask_2020], maxlags=21)
        print(f"    2020 only -- TSMOM alone: alpha={b2020['alpha_annualized_pct']:+7.2f}%/yr  Sharpe={sharpe(tsmom_ret[mask_2020]):.2f}  |  "
              f"scaled ({scale}x): alpha={c2020['alpha_annualized_pct']:+7.2f}%/yr  Sharpe={sharpe(combo_ret[mask_2020]):.2f}")

    print(f"\n{'='*80}\n  Year-by-year, 2x scale variant\n{'='*80}")
    multiplier = pd.Series(1.0, index=common_idx)
    multiplier[active] = 2.0
    applied_mult = multiplier.shift(1).fillna(1.0)
    combo_ret = tsmom_ret * applied_mult
    for year in sorted(set(common_idx.year)):
        m = common_idx.year == year
        if m.sum() < 60:
            continue
        b = hac_alpha(tsmom_ret[m], tsmom_passive[m], maxlags=21)
        c = hac_alpha(combo_ret[m], tsmom_passive[m], maxlags=21)
        print(f"  {year}: TSMOM alone alpha={b['alpha_annualized_pct']:+7.2f}%/yr  |  "
              f"combo alpha={c['alpha_annualized_pct']:+7.2f}%/yr  |  active days this year={int(active[m].sum())}")
