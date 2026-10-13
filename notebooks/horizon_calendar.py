"""
Horizons in each instrument's OWN trading days.

The stored panels use a union calendar: weekdays up to 2013, every calendar day from 2014 on (crypto trades daily, so
weekend rows exist and are NaN for equities, FX, futures and ETFs). Shifting that union frame by `tau` rows therefore meant
`tau` trading days before 2014 but `tau` CALENDAR days from 2014 on (21 rows ~ 15 US trading days, 252 rows ~ 174), and left a
third of the increments undefined. Here every increment is computed on the instrument's own bars (its non-NaN closes) and
only then placed back on the union calendar at the date of its own bar, so
  past increment at t   = log(P[t] / P[t - tau own bars])
  forward increment at t = log(P[t + tau own bars] / P[t])
and a date on which the instrument has no bar stays NaN. A crypto series has a bar every day, so its horizons are calendar days,
which is its own trading-day count.
"""
import numpy as np
import pandas as pd

RATE_TICKERS = {"^VIX", "^VXN", "^OVX", "^GVZ", "^EVZ", "^VVIX", "^SKEW", "^TNX", "^TYX", "^FVX", "^IRX"}


def own_increment(prices, ticker, tau, rate_tickers=RATE_TICKERS):
    s = prices[ticker].dropna()
    inc = (s - s.shift(tau)) if ticker in rate_tickers else np.log(s / s.shift(tau))
    return inc.reindex(prices.index)


def own_forward(prices, ticker, tau):
    s = prices[ticker].dropna()
    return np.log(s.shift(-tau) / s).reindex(prices.index)
