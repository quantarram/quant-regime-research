"""
Fixture-list version of the football board for ETFs: every term is built only from match results.

  matchday  : SPY's trading days are cut into consecutive blocks of H days (no overlap). Each ETF plays ONE match per block.
  opponent  : "class"  = equal-weight mean of the other ETFs in its asset class,  "spy" = SPY,  "zero" = cash (absolute result)
  result    : the ETF's block return minus the opponent's;  win 3 if > +band, draw 1 within +-band, loss 0 if < -band
  venue     : the market regime at the START of a match (the ETF's "ground"): SPY above/below its 200-day average,
              VIX above/below its trailing-year median, SPY's previous block up/down, or none
  PPG       : mean points of the ETF's last N completed matches                      (any venue)
  home PPG  : mean points of its last N completed matches played in the SAME venue as the upcoming match
  form edge : PPG minus the mean PPG of the other ETFs in its asset class
  pick      : both >= the q-th percentile of all earlier ETF-matchdays (frozen each 1 January); bet = the ETF wins its next match

Discovery 2008-2014 chooses; the holdout (2015 on) is looked at once for the finalists.
"""
import itertools
import numpy as np
import pandas as pd

import sb_core as C
import sb_form as F


def load_blocks(P, H):
    spy = P["SPY"].dropna()
    spy = spy.loc["2002-01-01":]
    ends = spy.index[H - 1::H]                      # block end dates (every H-th SPY trading day)
    tick = [t for t in C.ALL_TICKERS if t in P.columns]
    close = pd.DataFrame({t: P[t].dropna().reindex(spy.index).ffill(limit=5) for t in tick})
    cb = close.loc[ends]
    ret = cb.pct_change(fill_method=None)
    # regimes known at each block END (these become the venue of the NEXT match)
    spy_ma = spy.rolling(200).mean()
    vix = P["^VIX"].dropna().reindex(spy.index).ffill()
    vix_med = vix.rolling(252).median()
    reg = pd.DataFrame(index=ends)
    reg["trend"] = (spy.loc[ends] > spy_ma.loc[ends]).where(spy_ma.loc[ends].notna())
    reg["vol"] = (vix.loc[ends] > vix_med.loc[ends]).where(vix_med.loc[ends].notna())
    spy_b = spy.loc[ends].pct_change()
    reg["prev"] = (spy_b > 0).where(spy_b.notna())
    reg["none"] = True
    return ends, ret, reg


def opponent_diff(ret, mode):
    cls = pd.Series({t: F.ASSET_CLASS[t] for t in ret.columns})
    if mode == "zero":
        return ret.copy()
    if mode == "spy":
        d = ret.sub(ret["SPY"], axis=0)
        d["SPY"] = 0.0
        return d.where(ret.notna())
    out = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    for c in cls.unique():
        cols = cls[cls == c].index
        for t in cols:
            o = ret[cols.drop(t)]
            peer = o.mean(axis=1).where(o.notna().sum(axis=1) >= 3)
            out[t] = ret[t] - peer
    return out


def build_cell(ret, reg, H, n_years, opp, band_pct, venue):
    N = max(4, int(round(n_years * 252 / H)))
    band = band_pct / 100.0 * np.sqrt(H / 21.0)
    diff = opponent_diff(ret, opp)
    pts = pd.DataFrame(np.where(diff.isna(), np.nan, np.where(diff > band, 3.0, np.where(diff >= -band, 1.0, 0.0))),
                       index=ret.index, columns=ret.columns)
    B = len(pts)
    P_ = pts.values
    ppg = pts.rolling(N, min_periods=N).mean().values
    # venue of completed match k = regime at the START of the match = regime at the end of block k-1
    V = reg[venue].shift(1).values.astype(float)      # V[k]
    U = reg[venue].values.astype(float)               # upcoming venue at signal block b
    hppg = np.full_like(P_, np.nan)
    for b in range(B):
        if np.isnan(U[b]):
            continue
        ks = np.where((V[:b + 1] == U[b]))[0]
        if len(ks) < N:
            continue
        sel = ks[-N:]
        blk = P_[sel]
        ok = np.isfinite(blk).sum(axis=0) == N
        hppg[b, ok] = blk[:, ok].mean(axis=0)
    cls = pd.Series({t: F.ASSET_CLASS[t] for t in ret.columns})
    ppg_df = pd.DataFrame(ppg, index=ret.index, columns=ret.columns)
    peer = pd.DataFrame({t: ppg_df[cls[cls == cls[t]].index.drop(t)].mean(axis=1) for t in ret.columns})
    fe = ppg_df - peer
    nxt_r = ret.shift(-1)
    nxt_d = diff.shift(-1)
    rows = []
    idx = np.arange(B)
    for j, t in enumerate(ret.columns):
        rows.append(pd.DataFrame({"date": ret.index, "ticker": t, "form_edge": fe[t].values, "home_ppg": hppg[:, j],
                                  "ret": nxt_r[t].values, "rel": nxt_d[t].values, "pos": idx}))
    T = pd.concat(rows, ignore_index=True)
    T = T[T["rel"].notna() & T["ret"].notna()].reset_index(drop=True)
    return T, band


def picks(T, q, first_year, last_year, min_rows=800):
    m = np.zeros(len(T), dtype=bool)
    yrs = T["date"].dt.year.values
    fe, hp = T["form_edge"].values, T["home_ppg"].values
    for Y in range(first_year, last_year + 1):
        past = T[T["date"] < pd.Timestamp(f"{Y}-01-01")]
        a, b = past["form_edge"].dropna(), past["home_ppg"].dropna()
        if len(a) < min_rows or len(b) < min_rows:
            continue
        ta, tb = a.quantile(q), b.quantile(q)
        s = yrs == Y
        m[s] = (fe[s] >= ta) & (hp[s] >= tb)
    return m


def evaluate(T, m, y0, y1, kind, band):
    """kind 'rel' (beat the opponent) or 'abs' (return > 0). Compared with the same ETF's own record in the window."""
    W = T[(T["date"].dt.year >= y0) & (T["date"].dt.year <= y1)].copy()
    if kind == "rel":
        W["win"] = (W["rel"] > band).astype(float)
        W["pnl"] = W["rel"]
    else:
        W["win"] = (W["ret"] > 0).astype(float)
        W["pnl"] = W["ret"]
    sel = m[W.index] if False else None
    return W


def stats_for(T, m, y0, y1, kind, band, H):
    W = T[(T["date"].dt.year >= y0) & (T["date"].dt.year <= y1)].copy()
    msk = m[W.index.values]
    if kind == "rel":
        W["win"] = (W["rel"] > band).astype(float); W["pnl"] = W["rel"]
    else:
        W["win"] = (W["ret"] > 0).astype(float); W["pnl"] = W["ret"]
    R_ = W[msk]
    if len(R_) < 5:
        return dict(n=len(R_), hit=np.nan, base=np.nan, lift=np.nan, exc=np.nan, t=np.nan, blocks=0)
    bw = W.groupby("ticker")["win"].mean()
    bp = W.groupby("ticker")["pnl"].mean()
    exc = R_["pnl"].values - R_["ticker"].map(bp).values
    lift = R_["win"].mean() - R_["ticker"].map(bw).mean()
    blk = R_["pos"].values
    em = pd.Series(exc).groupby(blk).mean()
    t = em.mean() / (em.std(ddof=1) / np.sqrt(len(em))) if len(em) > 2 and em.std(ddof=1) > 0 else np.nan
    return dict(n=len(R_), hit=R_["win"].mean(), base=R_["ticker"].map(bw).mean(), lift=lift, exc=exc.mean(), t=t, blocks=len(em))
