"""
Walk-forward research for the stock board.

Every year Y from FIRST_OOS_YEAR on, the rule is chosen using ONLY bets whose outcome was complete (exit date) before
1 January of Y, then applied to Y's signals. Pooled out-of-sample results are the numbers the board reports, the stock
analogue of the football holdout. Rules are chosen separately for each horizon.

Selection score: cluster t-statistic of excess P&L (rule bet P&L minus that instrument's own all-days average P&L in the
same window). Clusters are calendar blocks (one month for the 21-bar bet, one quarter for the 63-bar bet) shared by ALL
instruments, because bets entered in the same block overlap in time and are driven by the same market moves; overlapping
bets are used in full but counted once for the standard error (an analytic effective-sample correction, indicative only).
Football analogue of "fires about 1 in 20-25 matches": a rule may fire on at most MAX_FIRE_RATE of the instrument-days in
the selection window. Requires >= MIN_BLOCKS blocks, positive mean P&L and a hit rate above the instruments' own baseline.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sb_core as C

FIRST_OOS_YEAR = 2010
MIN_EPISODES = 40
MIN_BLOCKS = 30
MAX_FIRE_RATE = 0.06


def build_table(prices, k_t=C.K_TARGET, k_s=C.K_STOP, horizons=C.HORIZONS, universe=None):
    frames = []
    tickers = [t for ts in C.UNIVERSE.values() for t in ts] if universe is None else universe
    for t in tickers:
        if t not in prices.columns:
            continue
        s = prices[t].dropna()
        s = s[s > 0]
        if len(s) < 800:
            continue
        f = C.features_for(s)
        for h in horizons:
            o = C.barrier_outcomes(s, f["sigma_d"], h, k_t, k_s)
            d = pd.concat([f, o.drop(columns=["h"])], axis=1)
            d["ticker"] = t
            d["h"] = h
            d["pos"] = np.arange(len(d))
            d.index.name = "date"
            frames.append(d.reset_index())
    T = pd.concat(frames, ignore_index=True)
    T["group"] = T["ticker"].map(C.TICKER_GROUP)
    return T


def rule_matrix(T, rules):
    M = np.zeros((len(T), len(rules)), dtype=bool)
    cache = {}
    for j, (conds, q) in enumerate(rules):
        m = np.ones(len(T), dtype=bool)
        for c in conds:
            key = (c, q)
            if key not in cache:
                cache[key] = C.condition_mask(T, c, q).fillna(False).values
            m &= cache[key]
        M[:, j] = m
    return M


def _episodes(ticker_codes, pos, h):
    order = np.lexsort((pos, ticker_codes))
    tc, ps = ticker_codes[order], pos[order]
    new = np.ones(len(order), dtype=bool)
    new[1:] = (tc[1:] != tc[:-1]) | ((ps[1:] - ps[:-1]) > h)
    ep = np.cumsum(new) - 1
    out = np.empty(len(order), dtype=np.int64)
    out[order] = ep
    return out


def baseline_maps(W):
    g = W.groupby("ticker")
    return g["pnl"].mean(), g["win"].mean()


def rule_stats(R, W, h, bmaps=None):
    """R = rows where the rule fires (resolved), W = all resolved rows in the same window. Returns dict."""
    if len(R) == 0:
        return dict(n=0, n_ep=0, n_blk=0, hit=np.nan, pnl=np.nan, avg_win=np.nan, avg_loss=np.nan, be=np.nan,
                    base_hit=np.nan, base_pnl=np.nan, exc_pnl=np.nan, t=np.nan, lift=np.nan)
    bp, bw = bmaps if bmaps is not None else baseline_maps(W)
    base_pnl = R["ticker"].map(bp).values
    base_hit = R["ticker"].map(bw).values
    exc = R["pnl"].values - base_pnl
    codes = pd.factorize(R["ticker"])[0]
    ep = _episodes(codes, R["pos"].values, h)
    n_ep = len(np.unique(ep))
    blk_len = 1 if h <= 21 else 3
    d = R["date"]
    blk = ((d.dt.year * 12 + d.dt.month - 1) // blk_len).values
    ep_mean = pd.Series(exc).groupby(blk).mean()
    n_blk = len(ep_mean)
    t = ep_mean.mean() / (ep_mean.std(ddof=1) / np.sqrt(n_blk)) if n_blk > 2 and ep_mean.std(ddof=1) > 0 else np.nan
    wins = R["pnl"][R["win"] == 1]
    loss = R["pnl"][R["win"] == 0]
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = -loss.mean() if len(loss) else np.nan
    be = avg_loss / (avg_win + avg_loss) if len(wins) and len(loss) else np.nan
    hit = R["win"].mean()
    return dict(n=len(R), n_ep=n_ep, n_blk=n_blk, hit=hit, pnl=R["pnl"].mean(), avg_win=avg_win, avg_loss=avg_loss, be=be,
                base_hit=np.nanmean(base_hit), base_pnl=np.nanmean(base_pnl), exc_pnl=np.nanmean(exc), t=t,
                lift=hit - np.nanmean(base_hit))


def select_rule(T, M, rules, h, cutoff):
    """Pick the best rule for horizon h using only rows whose exit_date < cutoff. Returns (rule_idx, stats, n_candidates)."""
    sel = (T["h"].values == h) & T["win"].notna().values & (T["exit_date"].values < np.datetime64(cutoff))
    W = T[sel]
    bm = baseline_maps(W)
    Ms = M[sel]
    best, best_t, best_stats, n_ok = None, -np.inf, None, 0
    for j in range(len(rules)):
        R = W[Ms[:, j]]
        if len(R) < MIN_EPISODES or len(R) / len(W) > MAX_FIRE_RATE:
            continue
        st = rule_stats(R, W, h, bm)
        if st["n_ep"] < MIN_EPISODES or st["n_blk"] < MIN_BLOCKS or not np.isfinite(st["t"]):
            continue
        if st["pnl"] <= 0 or st["lift"] <= 0:
            continue
        n_ok += 1
        if st["t"] > best_t:
            best, best_t, best_stats = j, st["t"], st
    return best, best_stats, n_ok


def walk_forward(T, M, rules, horizons=C.HORIZONS, first_year=FIRST_OOS_YEAR, last_year=None):
    last_year = last_year or int(T["date"].dt.year.max())
    oos, chosen = [], []
    for h in horizons:
        for Y in range(first_year, last_year + 1):
            cutoff = pd.Timestamp(f"{Y}-01-01")
            j, st, n_ok = select_rule(T, M, rules, h, cutoff)
            in_year = (T["h"].values == h) & (T["date"].dt.year.values == Y)
            chosen.append(dict(h=h, year=Y, rule=C.rule_label(rules[j]) if j is not None else None,
                               train_n_ep=st["n_ep"] if st else 0, train_blocks=st["n_blk"] if st else 0, train_hit=st["hit"] if st else np.nan,
                               train_t=st["t"] if st else np.nan, n_candidates_passing=n_ok))
            if j is None:
                continue
            sel = in_year & M[:, j] & T["win"].notna().values
            R = T[sel].copy()
            R["rule"] = C.rule_label(rules[j])
            oos.append(R)
    return (pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()), pd.DataFrame(chosen)


def summarize(R, W, h):
    return rule_stats(R, W, h)
