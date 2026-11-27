"""
Form edge and Home PPG for ETFs: the football board's two statistics, rebuilt for finance.

Football                                        Stock board
-----------------------------------------------------------------------------------------------
one game = one match                            one game = one COMPLETED barrier bet on that ETF (entered every day, see
                                                sb_core: target +0.5 sigma before stop -1.5 sigma, entry next close)
points: win 3, draw 1, loss 0                   target hit 3, time limit reached with neither barrier 1, stop hit 0
PPG = points per game, last 30 matches          PPG = mean points of the ETF's last 30 bets that had COMPLETED by that date
form edge = home PPG (any venue)                form edge = ETF's PPG minus the mean PPG of the OTHER ETFs in its asset class
            minus away team's PPG                           (equities, commodities, bonds) on the same date
home-only PPG                                   home PPG = PPG over the ETF's last 30 completed bets whose signal day was a
                                                "home" day = SPY closed up that day
pick when both reach the 90th percentile       pick when both reach the 90th percentile of the statistic over all earlier
of the statistic's historical distribution      ETF-days (frozen each 1 January, so no look-ahead), pooled across ETFs
                                                (home-team win)                                  (long the ETF)

Nothing is fitted: window 30 and percentile 0.90 are the football board's own settings, fixed before any result was seen.
"""
import numpy as np
import pandas as pd

import sb_core as C

N_GAMES = 30
Q_THR = 0.90
MIN_PEERS = 3
ASSET_CLASS = {}
for _g, _ts in C.UNIVERSE.items():
    for _t in _ts:
        ASSET_CLASS[_t] = ("Bonds" if _g == "Bond ETFs" else
                           "Commodities" if _g in ("Metals ETFs", "Other commodity ETFs") else "Equities")


def points(outcome):
    return outcome.map({"TARGET": 3.0, "TIMEOUT": 1.0, "STOP": 0.0})


def trailing_completed_mean(dates, exit_dates, pts, n=N_GAMES):
    """For each date, the mean of the last n values of pts among bets with exit_date <= date (ordered by exit date)."""
    order = np.argsort(exit_dates, kind="stable")
    ex = exit_dates[order]
    p = pts[order]
    cs = np.concatenate([[0.0], np.cumsum(p)])
    k = np.searchsorted(ex, dates, side="right")
    out = np.full(len(dates), np.nan)
    ok = k >= n
    out[ok] = (cs[k[ok]] - cs[k[ok] - n]) / n
    return out


def add_form_columns(T, spy_close, n=N_GAMES):
    """T: bet table for ONE horizon (rows: ticker, date, exit_date, outcome ...). Adds ppg, home_ppg, peer_ppg, form_edge."""
    T = T.copy()
    T["pts"] = points(T["outcome"])
    spy_ret = spy_close.pct_change()
    T["home"] = T["date"].map(spy_ret) > 0
    ppg = np.full(len(T), np.nan)
    hppg = np.full(len(T), np.nan)
    for t, ix in T.groupby("ticker").indices.items():
        d = T.iloc[ix]
        done = d["outcome"].notna().values
        dates = d["date"].values.astype("datetime64[ns]")
        ex = d["exit_date"].values.astype("datetime64[ns]")
        p = d["pts"].values
        home = d["home"].values
        ppg[ix] = trailing_completed_mean(dates, ex[done], p[done], n) if done.sum() > n else np.nan
        hm = done & home
        hppg[ix] = trailing_completed_mean(dates, ex[hm], p[hm], n) if hm.sum() > n else np.nan
    T["ppg"] = ppg
    T["home_ppg"] = hppg
    T["aclass"] = T["ticker"].map(ASSET_CLASS)
    # peer mean of the OTHER ETFs in the same asset class on the same date
    g = T.groupby(["aclass", "date"])["ppg"]
    s = g.transform("sum")
    c = g.transform("count")
    own = T["ppg"].fillna(0.0)
    own_c = T["ppg"].notna().astype(int)
    peer_n = c - own_c
    T["peer_ppg"] = np.where(peer_n >= MIN_PEERS, (s - own) / peer_n.replace(0, np.nan), np.nan)
    T["form_edge"] = T["ppg"] - T["peer_ppg"]
    return T


def frozen_thresholds(T, year, q=Q_THR, min_rows=2000):
    """90th percentile of form edge and home PPG over all rows dated before 1 January of `year` (pooled across ETFs)."""
    past = T[T["date"] < pd.Timestamp(f"{year}-01-01")]
    fe = past["form_edge"].dropna()
    hp = past["home_ppg"].dropna()
    if len(fe) < min_rows or len(hp) < min_rows:
        return None
    return float(fe.quantile(q)), float(hp.quantile(q))


def pick_mask(T, first_year, last_year, q=Q_THR):
    m = np.zeros(len(T), dtype=bool)
    thr = {}
    yrs = T["date"].dt.year.values
    fe = T["form_edge"].values
    hp = T["home_ppg"].values
    for Y in range(first_year, last_year + 1):
        th = frozen_thresholds(T, Y, q)
        thr[Y] = th
        if th is None:
            continue
        sel = yrs == Y
        m[sel] = (fe[sel] >= th[0]) & (hp[sel] >= th[1])
    return m, thr
