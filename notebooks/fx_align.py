"""
FX bar alignment for the price panel
=====================================
Finding (2026-10-13): from 2010-08-03 onward every Yahoo FX series ('...=X' tickers) in the stored panels is dated
exactly ONE bar late relative to the US-listed series; before that date FX and US series are aligned.

Evidence (all on the stored panel, which equals raw Yahoo bar for bar):
  * 2011-2026, every year: corr(UUP ret_t, EURUSD=X ret_t) is about 0 and corr(UUP ret_t, EURUSD=X ret_{t+1}) is about
    -0.9 (UDN +0.9; GBPUSD -0.77, CHFUSD -0.76, AUDUSD -0.64; SPY vs AUDJPY next-bar +0.46 to +0.78).
  * 2007-2010Q2: same-bar strong (UUP/EURUSD -0.80 to -0.95, SPY/AUDJPY +0.65 to +0.80), next-bar about 0. 2004-2006,
    anchored on gold/EURUSD (+0.65 / +0.47 / +0.37 same-bar, next-bar about 0), also aligned. 2010Q3 is mixed (-0.40 / -0.65).
  * Changepoint: spliced-correlation search over 7 pairs puts it at 2010-08-02/03 (EURUSD, GBPUSD and CHFUSD all
    2010-08-03; pooled near-optimal range 2010-07-29..2010-08-09, so about +/-4 business days of uncertainty, a few bars).
  * Controls: US/US pairs (SPY/QQQ, GLD/SLV, TLT/IEF) aligned in every year; FX/FX pairs aligned with each other.
Before 2003 only JPYUSD, EURGBP and CNYUSD exist and no US-listed anchor is strong enough to test; they are left as
they are (consistent with the aligned early regime).

Effect if left uncorrected: a rule "US-listed X tail at t -> FX target over the next day" is really a same-day
co-movement. All 76 one-day rows of the old CPE table (100%) had FX targets, dominated by UDN -> EURUSD=X; for
longer horizons the contamination is one bar in 21 to 300.

realign_fx() moves each FX close, from SWITCH_DATE on, onto the label of its own day: within the series' own dates,
value[d] := value[next bar]. At the splice nothing is duplicated or lost (the first offset label repeats the previous
real close and is dropped). The newest FX bar becomes NaN until the following bar exists, so there is never look-ahead.
Apply exactly once per raw download; apply_if_needed() refuses to shift a frame it has already marked.
"""
import pandas as pd

FX_SUFFIX = "=X"
SWITCH_DATE = pd.Timestamp("2010-08-03")   # first label that holds the PREVIOUS day's close
MARK = "fx_aligned_v1"


def fx_columns(prices):
    return [c for c in prices.columns if str(c).endswith(FX_SUFFIX)]


def realign_fx(prices, switch_date=SWITCH_DATE):
    """Return a copy with every FX column, from switch_date on, moved onto the label of its own day."""
    out = prices.copy()
    for c in fx_columns(out):
        s = out[c].dropna()
        shifted = s.shift(-1)
        new = s.copy()
        late = new.index >= pd.Timestamp(switch_date)
        new[late] = shifted[late]
        out[c] = new.reindex(out.index)
    out.attrs[MARK] = True
    return out


def is_aligned(prices):
    return bool(getattr(prices, "attrs", {}).get(MARK, False))


def apply_if_needed(prices):
    return prices if is_aligned(prices) else realign_fx(prices)
