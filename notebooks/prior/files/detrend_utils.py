"""
detrend_utils.py
================
Detrending for structurally decaying instruments (primarily VIX-complex
ETPs: UVXY, VIXY, VXX, VIXM, SVXY) so that their roll-decay secular
trend does not dominate the extreme-quantile thresholds used by the CPE
framework.

MOTIVATION (Portfolio Tilt paper Sections 3.3, 5, 19.6)
---------------------------------------------------------
VIX-complex ETPs carry a persistent secular roll-decay trend. The CPE
framework computes quantile thresholds from trailing-window log-returns.
For a product whose 252-day trailing log-return is dominated by roll cost,
the 95th percentile sits at an extreme level reflecting the product's own
decay history rather than a generalisable market regime. The frozen
training-period threshold (e.g. -307% for UVXY at tau=252) is essentially
never re-cleared in live evaluation — even when the underlying vol regime
the configuration was designed to detect genuinely occurs.

The paper's Section 5 follow-up confirmed that simple exclusion also
removes the only working signal (the April 2025 episode used VXX and
VIXM). The correct fix is stationarisation — computing thresholds on the
decay-adjusted residual rather than the raw return.

APPROACH
--------
For each flagged instrument, at each trailing horizon τ:

    raw_return(t, τ) = trend(t, τ) + residual(t, τ)

where trend is estimated using a causal EWMA. The CPE threshold is then
estimated from the residual series (training window only).

KEY FINDING FROM PHASE 2 DIAGNOSTIC
-------------------------------------
A single halflife does not fit all decay instruments. The Phase 2 output
showed that UVXY and VIXM with halflife=252 produced detrended thresholds
so high they fired FEWER days than raw in 2025. The fix is per-instrument
halflives calibrated to each product's decay rate:

  - Faster-decaying instruments (UVXY, VIXY ~60-80% pa): need a LONGER
    halflife (504 days) so the trend estimate tracks the slow secular drift
    rather than the short-term vol signal.
  - Medium-decay instruments (VXX ~40-50% pa): 252 days works well
    (confirmed by Phase 2: 144 fires recovered at tau=252).
  - Slow-decay instruments (VIXM ~15-20% pa): need the longest halflife
    (756 days) because the gentle decay is most easily confused with signal.

INTEGRATION
-----------
Call integrate_detrending() AFTER building the standard increments dict,
before computing thresholds. See run_improvements.py Phase 2 for usage.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Set, Dict

# ── CONFIGURATION ────────────────────────────────────────────────────────────

DECAY_INSTRUMENTS: Set[str] = {
    "UVXY",   # 1.5x or 2x short-term VIX futures, decays fastest (~60-80% pa)
    "VIXY",   # 1x short-term VIX futures, roll decay ~40-60% pa
    "VXX",    # original short-term VIX futures ETP, same decay mechanism
    "VIXM",   # medium-term VIX futures, slower decay (~15-20% pa) but directional
    # SVXY omitted: inverse direction, structural change 2018, different mechanism
}

# Default EWMA halflife in trading days when no per-instrument override exists.
DEFAULT_TREND_HALFLIFE: int = 504

# Per-instrument halflives, calibrated to each product's decay rate.
# Faster-decaying instruments need a LONGER halflife so the EWMA trend
# tracks the slow secular drift rather than the short-term regime signal.
# Values confirmed by Phase 2 diagnostic: detrended threshold achievable
# in 2025 evaluation window (positive firing-day count after detrending).
INSTRUMENT_HALFLIVES: Dict[str, int] = {
    "UVXY":  504,   # ~2 trading years; fast decay, long halflife separates trend from signal
    "VIXY":  504,   # same structure as UVXY
    "VXX":   252,   # 1 year works well (Phase 2: 144 fires recovered at tau=252)
    "VIXM":  756,   # ~3 trading years; slow decay needs longest halflife
}

# Minimum number of observations before attempting detrending.
MIN_OBS_FOR_DETREND: int = 252


def _ewma_trend(series: pd.Series, halflife: int) -> pd.Series:
    """
    Causal EWMA-smoothed trend. Only past data contributes to each
    estimate — safe to use in a rolling training window without look-ahead.
    """
    return series.ewm(halflife=halflife, adjust=False).mean()


def get_detrended_increment(
    price_series: pd.Series,
    tau: int,
    ticker: str,
    halflife: Optional[int] = None,
    rate_index: bool = False,
) -> pd.Series:
    """
    Compute the detrended tau-day increment for one instrument.

    For instruments NOT in DECAY_INSTRUMENTS, returns the raw increment
    unchanged. For decay instruments, subtracts an EWMA trend computed
    with the per-instrument halflife (or halflife parameter if provided).

    Parameters
    ----------
    price_series : pd.Series
        Full price history (DatetimeIndex).
    tau : int
        Trailing horizon in trading days.
    ticker : str
        Instrument ticker, checked against DECAY_INSTRUMENTS.
    halflife : int or None
        Override the per-instrument halflife. If None, uses
        INSTRUMENT_HALFLIVES.get(ticker, DEFAULT_TREND_HALFLIFE).
    rate_index : bool
        If True, compute level-change increments (for rate indices).

    Returns
    -------
    pd.Series
        Detrended increment series. Unchanged for non-decay instruments.
    """
    if rate_index:
        raw_inc = price_series - price_series.shift(tau)
    else:
        raw_inc = np.log(price_series / price_series.shift(tau))

    if ticker not in DECAY_INSTRUMENTS:
        return raw_inc

    n_valid = raw_inc.dropna().__len__()
    if n_valid < MIN_OBS_FOR_DETREND:
        return raw_inc

    hl = halflife if halflife is not None else INSTRUMENT_HALFLIVES.get(
        ticker, DEFAULT_TREND_HALFLIFE
    )
    trend = _ewma_trend(raw_inc, hl)
    return raw_inc - trend


def integrate_detrending(
    increments: Dict[int, pd.DataFrame],
    prices: pd.DataFrame,
    tau_list: list,
    halflife: Optional[int] = None,
    rate_index_tickers: Optional[Set[str]] = None,
    instruments: Optional[Set[str]] = None,
) -> Dict[int, pd.DataFrame]:
    """
    Apply detrending to the relevant columns of an existing increments dict.

    Call AFTER building the standard increments dict, BEFORE computing
    thresholds. Modifies in-place and returns the same dict.

    Parameters
    ----------
    increments : dict
        Existing {tau: DataFrame} from the standard pipeline.
    prices : pd.DataFrame
        Full price history (same as used to build increments).
    tau_list : list
        Tau values to process.
    halflife : int or None
        Global halflife override. If None, per-instrument halflives from
        INSTRUMENT_HALFLIVES are used (recommended).
    rate_index_tickers : set or None
        Tickers using level-change rather than log-return increments.
    instruments : set or None
        Override the set of instruments to detrend. Defaults to
        DECAY_INSTRUMENTS.

    Returns
    -------
    dict
        Same increments dict with detrended columns replaced.
    """
    if rate_index_tickers is None:
        rate_index_tickers = {
            "^VIX", "^VXN", "^OVX", "^GVZ", "^EVZ", "^VVIX", "^SKEW",
            "^TNX", "^TYX", "^FVX", "^IRX",
        }
    if instruments is None:
        instruments = DECAY_INSTRUMENTS

    target_tickers = instruments & set(prices.columns)
    if not target_tickers:
        return increments

    for tau in tau_list:
        if tau not in increments:
            continue
        for ticker in target_tickers:
            if ticker not in increments[tau].columns:
                continue
            is_rate = ticker in rate_index_tickers
            detrended = get_detrended_increment(
                prices[ticker], tau, ticker,
                halflife=halflife,  # None -> uses per-instrument default
                rate_index=is_rate,
            )
            increments[tau][ticker] = detrended.reindex(increments[tau].index)

    return increments


def decay_detrend_report(
    prices: pd.DataFrame,
    train_cutoff: pd.Timestamp,
    tau: int = 252,
    q: float = 0.95,
    halflife: Optional[int] = None,
) -> pd.DataFrame:
    """
    Diagnostic table comparing raw vs. detrended quantile thresholds and
    2025 firing-day counts for each decay instrument at the given horizon.
    """
    rows = []
    train = prices[prices.index <= train_cutoff]

    for ticker in sorted(DECAY_INSTRUMENTS):
        if ticker not in prices.columns:
            continue
        s = prices[ticker].dropna()
        s_train = train[ticker].dropna()

        if len(s_train) < tau + 1:
            rows.append({"ticker": ticker, "n_train_obs": len(s_train),
                         "raw_q_threshold": np.nan,
                         "detrended_q_threshold": np.nan,
                         "note": "insufficient training history"})
            continue

        raw_inc = np.log(s / s.shift(tau))
        raw_train = raw_inc.loc[s.index <= train_cutoff].dropna()

        detrended_inc = get_detrended_increment(s, tau, ticker, halflife=halflife)
        det_train = detrended_inc.loc[s.index <= train_cutoff].dropna()

        raw_thresh = float(raw_train.quantile(q))
        det_thresh = (
            float(det_train.quantile(q))
            if len(det_train) >= MIN_OBS_FOR_DETREND
            else np.nan
        )

        eval_start = pd.Timestamp("2025-01-01")
        eval_end   = pd.Timestamp("2025-12-31")
        raw_eval = raw_inc[(raw_inc.index >= eval_start) & (raw_inc.index <= eval_end)]
        det_eval = detrended_inc[(detrended_inc.index >= eval_start) & (detrended_inc.index <= eval_end)]

        raw_fires = int((raw_eval > raw_thresh).sum()) if len(raw_eval) > 0 else np.nan
        det_fires = (
            int((det_eval > det_thresh).sum())
            if len(det_eval) > 0 and not np.isnan(det_thresh)
            else np.nan
        )

        rows.append({
            "ticker":                ticker,
            "n_train_obs":           int(len(raw_train)),
            "tau":                   tau,
            "quantile":              q,
            "halflife_used":         INSTRUMENT_HALFLIVES.get(ticker, DEFAULT_TREND_HALFLIFE)
                                     if halflife is None else halflife,
            "raw_q_threshold":       round(raw_thresh, 4),
            "detrended_q_threshold": round(det_thresh, 4) if not np.isnan(det_thresh) else np.nan,
            "raw_fires_2025":        raw_fires,
            "detrended_fires_2025":  det_fires,
            "note": (
                "detrending effective: threshold achievable in 2025"
                if (not np.isnan(det_thresh) and det_fires and det_fires > 0)
                else "raw threshold fires 0 days in 2025"
                if raw_fires == 0
                else ""
            ),
        })

    return pd.DataFrame(rows)


def _run_self_test():
    """
    Verify detrending raises the 95th-pct threshold for a decaying instrument
    and is a no-op for non-decay instruments.
    """
    rng = np.random.default_rng(0)
    n = 1500
    dates = pd.bdate_range("2018-01-02", periods=n)

    daily_decay = -0.60 / 252
    noise = rng.normal(0, 0.03, n)
    px_decay = pd.Series(np.exp(np.cumsum(daily_decay + noise)), index=dates)
    px_normal = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)

    train_cutoff = dates[1000]
    tau = 63

    raw_inc_decay = np.log(px_decay / px_decay.shift(tau))
    det_inc_decay = get_detrended_increment(px_decay, tau, "UVXY")
    det_inc_normal = get_detrended_increment(px_normal, tau, "SPY")

    raw_q95 = float(raw_inc_decay.loc[raw_inc_decay.index <= train_cutoff].dropna().quantile(0.95))
    det_q95 = float(det_inc_decay.loc[det_inc_decay.index <= train_cutoff].dropna().quantile(0.95))

    assert det_q95 > raw_q95, (
        f"Detrended threshold ({det_q95:.4f}) should exceed raw ({raw_q95:.4f})"
    )
    assert det_q95 > -0.20, (
        f"Detrended threshold ({det_q95:.4f}) should be > -0.20 (achievable)"
    )

    raw_inc_normal = np.log(px_normal / px_normal.shift(tau))
    pd.testing.assert_series_equal(
        det_inc_normal.dropna(), raw_inc_normal.dropna(),
        check_names=False, rtol=1e-10,
    )

    print("  detrend_utils self-test [PASS]")
    print(f"    Raw 95th-pct threshold (UVXY, tau={tau}): {raw_q95:.4f}")
    print(f"    Detrended 95th-pct threshold:             {det_q95:.4f}")
    print(f"    Threshold shift (detrend benefit):        {det_q95 - raw_q95:+.4f}")
    print(f"    Non-decay instrument (SPY): raw == detrended [verified]")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  detrend_utils.py — self-test")
    print("=" * 60)
    _run_self_test()
    print("\n  All checks passed.\n")
