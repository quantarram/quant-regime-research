"""Walk-forward search for weather-based decisions, following STUDY_PLAN_EVERYTHING.md.

  python edge_search.py signals        # build and cache all signals (observed, forecast, calendar)
  python edge_search.py run L|E|F      # evaluate a universe, walk-forward, write edge_<U>.json
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "climate-energy-dashboard")))
from gfs_archive import METROS, OUT as GFS_NPZ, STEPS  # noqa: E402
from weatherdash.calendar_utils import doy366  # noqa: E402

PRICES = os.path.join(HERE, "data", "universe_all.parquet")
SIGCACHE = os.path.join(HERE, "data", "edge_signals.pkl")
COST = 0.0005
NAMES = [m[0] for m in METROS]
POP = {m[0]: m[4] for m in METROS}
REGIONS = {
    "NAT": NAMES,
    "COLD": ["New York", "Chicago", "Washington", "Philadelphia", "Boston", "Detroit", "Minneapolis", "Denver", "Baltimore", "St. Louis",
             "Pittsburgh", "Cincinnati", "Kansas City", "Columbus", "Indianapolis", "Cleveland"],
    "SOUTH": ["Dallas", "Houston", "Miami", "Atlanta", "Phoenix", "Tampa", "Orlando", "Charlotte", "San Antonio", "Austin", "Las Vegas", "Nashville"],
    "TX": ["Dallas", "Houston", "Austin", "San Antonio"],
    "MW": ["Chicago", "Minneapolis", "Kansas City", "St. Louis", "Indianapolis", "Columbus", "Detroit"],
    "NE": ["New York", "Boston", "Philadelphia", "Washington", "Baltimore", "Pittsburgh"],
    "WEST": ["Los Angeles", "San Francisco", "San Diego", "Sacramento", "Seattle", "Portland", "Phoenix", "Las Vegas", "Denver"],
}
SEASON = {"hdd": [11, 12, 1, 2, 3], "cdd": [6, 7, 8, 9]}
RULES = ["long_short", "long_only", "short_only", "neutral"]
HOLDS = [1, 5, 21]


# ----------------------------------------------------------------------------- signals
def weights(cities):
    w = np.array([POP[c] for c in cities], float)
    return w / w.sum()


def metro_temps():
    return pd.DataFrame({m: pd.read_csv(os.path.join(HERE, "data", "metro_hist", f"{m.replace(' ', '_').replace('.', '')}.csv"),
                                        index_col=0, parse_dates=True)["t"] for m in NAMES}) * 9 / 5 + 32


def region_dd(T, cities):
    w = weights(cities)
    X = T[cities].values
    return {"hdd": pd.Series((np.maximum(65 - X, 0) * w).sum(1), index=T.index), "cdd": pd.Series((np.maximum(X - 65, 0) * w).sum(1), index=T.index)}


def yearly_normals(dd, first_year=2001):
    """Anomaly of a daily DD series against normals computed from years strictly before each year (15-day smoothed, leap-aware)."""
    out = pd.Series(np.nan, index=dd.index)
    doy = doy366(dd.index)
    for Y in sorted(set(dd.index.year)):
        if Y < first_year:
            continue
        past = dd[dd.index.year < Y]
        n = past.groupby(doy366(past.index)).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
        sm = pd.concat([n.iloc[-7:], n, n.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
        sm.index = range(1, 367)
        m = dd.index.year == Y
        out[m] = dd[m].values - sm.loc[doy[m]].values
    return out


def expanding_z(x, months, min_periods=300):
    """x: signal on calendar days (public at the close of that day); in-season only, standardised by the expanding std of earlier in-season values."""
    s = x.where(np.isin(x.index.month, months))
    sd = s.expanding(min_periods=min_periods).std().shift(1)
    return s / sd


def observed_signals():
    T = metro_temps()
    sigs = {}
    for rname, cities in REGIONS.items():
        dd = region_dd(T, cities)
        for q in ("hdd", "cdd"):
            an = yearly_normals(dd[q])
            for w in (3, 7, 14, 28):
                A = an.rolling(w).sum().shift(1)  # weather through the previous day
                sigs[f"obs|{rname}|{q}|anom{w}"] = expanding_z(A, SEASON[q])
                M = A - A.shift(w)
                sigs[f"obs|{rname}|{q}|mom{w}"] = expanding_z(M, SEASON[q])
    return sigs


def calendar_signals(idx):
    """Month-of-year rules (the climatology twin) and the seasonal-ramp signal from day-of-year normals."""
    sigs = {}
    for m in range(1, 13):
        ind = pd.Series((idx.month == m).astype(float), index=idx)
        sigs[f"cal|month{m}|long"] = ind
        sigs[f"cal|month{m}|short"] = -ind
    T = metro_temps()
    for rname, cities in REGIONS.items():
        dd = region_dd(T, cities)
        for q in ("hdd", "cdd"):
            d = dd[q]
            n = d[d.index.year < 2026].groupby(doy366(d.index[d.index.year < 2026])).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
            sm = pd.concat([n.iloc[-7:], n, n.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
            sm.index = range(1, 367)
            nser = pd.Series(sm.loc[doy366(idx)].values, index=idx)
            fut = nser.rolling(7).sum().shift(-7)  # normals for t+1..t+7 (seasonal expectation, known in advance)
            past = nser.rolling(7).sum()  # normals t-6..t
            ramp = fut - past
            sigs[f"cal|{rname}|{q}|ramp7"] = expanding_z(ramp, SEASON[q], 300)
    return sigs


def gfs_metro_cycles():
    """Per cycle: DataFrame (valid local date x metro) of daily mean deg F, complete local days only."""
    import degree_days as dd
    z = np.load(GFS_NPZ, allow_pickle=True)
    out = {}
    for day, arr in zip(z["days"], z["data"]):
        out[pd.Timestamp(str(day))] = dd.metro_daily(arr, str(day))
    return out


def forecast_signals(cycles):
    """GFS true-forecast signals per region, quantity, family and lead window; indexed by cycle date."""
    T = metro_temps()
    normals = {}
    for rname, cities in REGIONS.items():
        d = region_dd(T, cities)
        for q in ("hdd", "cdd"):
            x = d[q][d[q].index.year < 2026]
            n = x.groupby(doy366(x.index)).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
            sm = pd.concat([n.iloc[-7:], n, n.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
            sm.index = range(1, 367)
            normals[(rname, q)] = sm
    cdays = sorted(cycles)
    sigs = {}
    idx_names = {n: i for i, n in enumerate(NAMES)}
    for rname, cities in REGIONS.items():
        cid = [idx_names[c] for c in cities]
        w = weights(cities)
        # region DD per cycle (valid x [hdd, cdd])
        reg = {}
        for c in cdays:
            t = cycles[c].iloc[:, cid].dropna()
            if t.empty:
                continue
            reg[c] = pd.DataFrame({"hdd": (np.maximum(65 - t.values, 0) * w).sum(1), "cdd": (np.maximum(t.values - 65, 0) * w).sum(1)}, index=t.index)
        lead1 = {}
        for c, df in reg.items():
            v = c + pd.Timedelta(days=1)
            if v in df.index:
                lead1[v] = df.loc[v]
        L1 = pd.DataFrame(lead1).T
        for q in ("hdd", "cdd"):
            nm = normals[(rname, q)]
            rows = {}
            for c, df in reg.items():
                c0 = c - pd.Timedelta(days=1)
                rec = {}
                for (a, b) in ((1, 3), (4, 7), (1, 7)):
                    fut = [c + pd.Timedelta(days=k) for k in range(a, b + 1)]
                    if not all(v in df.index for v in fut):
                        continue
                    fs = df.loc[fut, q].sum()
                    nf = nm.loc[[int(doy366([v])[0]) for v in fut]].sum()
                    rec[f"anom{a}{b}"] = fs - nf
                    if c0 in reg and all(v in reg[c0].index for v in fut):
                        rec[f"rev{a}{b}"] = fs - reg[c0].loc[fut, q].sum()
                    if (a, b) in ((1, 3), (1, 7)):
                        k = b
                        past = [c - pd.Timedelta(days=j) for j in range(1, k + 1)]
                        if all(v in L1.index for v in past):
                            rec[f"chg{a}{b}"] = fs - L1.loc[past, q].sum()
                            npst = nm.loc[[int(doy366([v])[0]) for v in past]].sum()
                            rec[f"chgclim{a}{b}"] = nf - npst
                rows[c] = rec
            F = pd.DataFrame(rows).T
            F.index = pd.DatetimeIndex(F.index)
            for col in F.columns:
                name = f"fc|{rname}|{q}|{col}"
                sigs[name] = expanding_z(F[col], SEASON[q], min_periods=120)
    return sigs


def build_signals():
    sigs = observed_signals()
    print("observed", len(sigs), flush=True)
    px = pd.read_parquet(PRICES)
    days = px["SPY"].dropna().index
    sigs.update(calendar_signals(pd.date_range("1996-01-01", "2026-12-31")))
    print("with calendar", len(sigs), flush=True)
    cyc = gfs_metro_cycles()
    print("gfs cycles", len(cyc), flush=True)
    fc = forecast_signals(cyc)
    print("forecast", len(fc), flush=True)
    sigs.update(fc)
    pickle.dump(sigs, open(SIGCACHE, "wb"))
    print("saved", len(sigs))


# ----------------------------------------------------------------------------- evaluation
STATS = ["n_act", "s_act", "ss_act", "s_all", "s_ps", "n_days", "s_spy", "ss_spy", "s_r_act"]


def positions(sig_td, rule, hold):
    s = sig_td.values
    pos = np.where(s > 0.5, 1.0, np.where(s < -0.5, -1.0, 0.0))
    pos[np.isnan(s)] = 0.0
    if rule == "long_only":
        pos = np.clip(pos, 0, None)
    elif rule == "short_only":
        pos = np.clip(pos, None, 0)
    ph = pd.Series(pos).rolling(hold, min_periods=1).mean().shift(1).fillna(0.0).values
    return ph


def config_pnl(ph, R, spy, avail, rule):
    d = np.abs(np.diff(ph, prepend=0.0))
    cost = COST * d * (2.0 if rule == "neutral" else 1.0)
    Rx = R - spy[:, None] if rule == "neutral" else R
    pnl = (ph[:, None] * Rx - cost[:, None]) * avail
    act = ((np.abs(ph) > 0) | (d > 0))[:, None] & (avail > 0)
    return pnl, act


def year_bounds(years):
    ys = np.unique(years)
    starts = np.searchsorted(years, ys)
    return ys, starts


def evaluate(sigs, names, R, spy, avail, years):
    ys, starts = year_bounds(years)
    nS, nR, nH, N, nY = len(names), len(RULES), len(HOLDS), R.shape[1], len(ys)
    S = np.zeros((len(STATS), nS, nR, nH, N, nY), dtype=np.float64)
    nd = np.add.reduceat(avail, starts, axis=0)  # (Y,N) days available
    s_spy = np.add.reduceat(spy[:, None] * avail, starts, axis=0)
    ss_spy = np.add.reduceat((spy ** 2)[:, None] * avail, starts, axis=0)
    for i, name in enumerate(names):
        for ri, rule in enumerate(RULES):
            for hi, hold in enumerate(HOLDS):
                ph = positions(sigs[name], rule, hold)
                pnl, act = config_pnl(ph, R, spy, avail, rule)
                pa = pnl * act
                S[0, i, ri, hi] = np.add.reduceat(act.astype(float), starts, axis=0).T
                S[1, i, ri, hi] = np.add.reduceat(pa, starts, axis=0).T
                S[2, i, ri, hi] = np.add.reduceat(pa ** 2, starts, axis=0).T
                S[3, i, ri, hi] = np.add.reduceat(pnl, starts, axis=0).T
                S[4, i, ri, hi] = np.add.reduceat(pnl * spy[:, None], starts, axis=0).T
                S[8, i, ri, hi] = np.add.reduceat(R * act, starts, axis=0).T
        if i % 20 == 0:
            print("  evaluated", i, "/", nS, flush=True)
    S[5] = nd.T[None, None, None]
    S[6] = s_spy.T[None, None, None]
    S[7] = ss_spy.T[None, None, None]
    return S, ys


def admit(S, ys, Y, min_prior, rules_idx):
    """Admission using years < Y: returns boolean mask (nS,nR,nH,N) and score."""
    prior = ys < Y
    if prior.sum() < min_prior:
        return None, None
    P = S[..., prior].sum(-1)
    n_act, s_act, ss_act, s_all, s_ps, n_d, s_spy, ss_spy, s_r = P
    with np.errstate(all="ignore"):
        mean_act = s_act / n_act
        sd_act = np.sqrt(np.maximum(ss_act / n_act - mean_act ** 2, 0))
        t = mean_act / (sd_act / np.sqrt(n_act))
        mp, ms = s_all / n_d, s_spy / n_d
        cov = s_ps / n_d - mp * ms
        var = ss_spy / n_d - ms ** 2
        beta = cov / var
        alpha = (mp - beta * ms) * 252
        bh_act = s_r / n_act
    ok = (n_act >= 60) & (mean_act > 0) & (alpha > 0)
    lo = RULES.index("long_only")
    ok[:, lo] &= (mean_act[:, lo] > bh_act[:, lo])
    return ok, np.where(ok, t, -np.inf)


def meta_pnl(sigs, names, chosen, R, spy, avail, dates, Y, cols):
    """Daily equal-weighted P&L in year Y of the chosen configurations; chosen: list of (sig_i, rule_i, hold_i, inst_j)."""
    m = dates.year == Y
    tot = np.zeros(m.sum())
    cache = {}
    for (i, ri, hi, j) in chosen:
        key = (i, ri, hi)
        if key not in cache:
            ph = positions(sigs[names[i]], RULES[ri], HOLDS[hi])
            cache[key] = ph
        ph = cache[key]
        d = np.abs(np.diff(ph, prepend=0.0))
        cost = COST * d * (2.0 if RULES[ri] == "neutral" else 1.0)
        r = R[:, j] - (spy if RULES[ri] == "neutral" else 0.0)
        pnl = (ph * r - cost) * avail[:, j]
        tot += pnl[m]
    return tot / max(len(chosen), 1)


def walk_forward(sigs, names, S, ys, R, spy, avail, dates, test_years, min_prior, top, group_mask, label, cols):
    """group_mask: boolean over signals to include."""
    rec = {"label": label, "years": {}, "series": []}
    allp, spys, dts = [], [], []
    for Y in test_years:
        ok, score = admit(S, ys, Y, min_prior, None)
        if ok is None:
            continue
        ok = ok & group_mask[:, None, None, None]
        sc = np.where(ok, score, -np.inf)
        flat = np.argsort(sc.ravel())[::-1]
        n_ok = int(ok.sum())
        k = min(top, n_ok)
        if k == 0:
            rec["years"][int(Y)] = {"n_admitted": 0, "n_chosen": 0}
            allp.append(np.zeros((dates.year == Y).sum()))
        else:
            idx = np.array(np.unravel_index(flat[:k], sc.shape)).T
            chosen = [tuple(int(v) for v in r) for r in idx]
            p = meta_pnl(sigs, names, chosen, R, spy, avail, dates, Y, cols)
            allp.append(p)
            # realised year-Y statistics for ALL admitted configs (family tables)
            yi = int(np.where(ys == Y)[0][0])
            na, sa = S[0][..., yi], S[1][..., yi]
            with np.errstate(all="ignore"):
                realised = np.where(ok & (na > 0), sa / na, np.nan)
            rec["years"][int(Y)] = {"n_admitted": n_ok, "n_chosen": k, "mean_realised_all_admitted_bp": float(np.nanmean(realised) * 1e4),
                                    "share_positive_admitted": float(np.nanmean(realised > 0))}
        mY = dates.year == Y
        spys.append(spy[mY])
        dts.append(dates[mY])
    if not allp:
        return rec
    p = np.concatenate(allp)
    s = np.concatenate(spys)
    d = pd.DatetimeIndex(np.concatenate([np.asarray(x) for x in dts]))
    rec["series"] = {"dates": [str(x.date()) for x in d], "pnl": p.tolist()}
    rec["summary"] = summarise(p, s, d)
    return rec


def summarise(p, s, d):
    p, s = np.asarray(p), np.asarray(s)
    out = {"n_days": int(len(p)), "ann_mean": float(p.mean() * 252), "ann_vol": float(p.std() * np.sqrt(252))}
    out["sharpe"] = out["ann_mean"] / out["ann_vol"] if out["ann_vol"] > 0 else None
    cum = np.cumsum(p)
    out["maxdd"] = float((cum - np.maximum.accumulate(cum)).min())
    var = s.var()
    beta = float(np.cov(p, s)[0, 1] / var) if var > 0 else None
    out["beta"] = beta
    out["alpha_md_ann"] = float((p.mean() - (beta or 0) * s.mean()) * 252)
    out["spy_ann_mean"] = float(s.mean() * 252)
    yrs = pd.Series(p, index=pd.DatetimeIndex(d)).groupby(pd.DatetimeIndex(d).year).sum()
    out["yearly_return"] = {int(k): float(v) for k, v in yrs.items()}
    return out


def run_universe(U):
    sigs_all = pickle.load(open(SIGCACHE, "rb"))
    px = pd.read_parquet(PRICES)
    days = px["SPY"].dropna().index
    px = px.reindex(days)
    if U == "L":
        start, test_years, min_prior = "1998-01-01", list(range(2008, 2026)), 3
        inst = [c for c in px.columns if px[c].first_valid_index() is not None and px[c].first_valid_index() <= pd.Timestamp("2002-12-31")]
        kinds = ("obs", "cal")
        end = "2025-12-31"
    elif U == "E":
        start, test_years, min_prior = "2016-01-01", list(range(2019, 2026)), 3
        inst = list(px.columns)
        kinds = ("obs", "cal")
        end = "2025-12-31"
    else:
        start, test_years, min_prior = "2021-03-23", [2024, 2025, 2026], 2
        inst = list(px.columns)
        kinds = ("fc", "cal")
        end = "2026-12-31"
    px = px.loc[start:end, inst]
    dates = px.index
    R = np.log(px / px.shift(1)).fillna(0.0).values
    avail = (px.notna() & px.shift(1).notna()).astype(float).values
    spy = R[:, inst.index("SPY")]
    names = [n for n in sigs_all if n.split("|")[0] in kinds]
    sigs = {n: sigs_all[n].reindex(dates) for n in names}
    # drop signals that never fire in the window
    names = [n for n in names if (sigs[n].abs() > 0.5).sum() > 30]
    print(U, "instruments", len(inst), "signals", len(names), "days", len(dates), flush=True)
    S, ys = evaluate(sigs, names, R, spy, avail, dates.year.values)
    fam = np.array([n.split("|")[0] for n in names])
    out = {"universe": U, "instruments": inst, "n_signals": len(names), "test_years": test_years}
    for label, mask in (("weather", fam != "cal"), ("calendar", fam == "cal")):
        out[label] = walk_forward(sigs, names, S, ys, R, spy, avail, dates, test_years, min_prior, 50, mask, label, inst)
        print(U, label, out[label].get("summary", {}).get("ann_mean"), out[label].get("summary", {}).get("alpha_md_ann"), flush=True)
    # baselines: equal-weight buy-and-hold of the universe, and the average of all weather configurations (what admission adds)
    ty = dates.year.isin(test_years)
    ew = (R * avail).sum(1) / np.maximum(avail.sum(1), 1)
    out["buy_hold_equal_weight"] = summarise(ew[ty], spy[ty], dates[ty])
    out["spy"] = summarise(spy[ty], spy[ty], dates[ty])
    allavg = {}
    for label, mask in (("weather", fam != "cal"), ("calendar", fam == "cal")):
        yr = {}
        for Y in test_years:
            yi = int(np.where(ys == Y)[0][0])
            sall, nd_ = S[3][..., yi][mask], S[5][..., yi][mask]
            with np.errstate(all="ignore"):
                yr[int(Y)] = float(np.nanmean(sall / nd_) * (dates.year == Y).sum())
        allavg[label] = yr
    out["all_configs_average_yearly_return"] = allavg
    # family-level realised performance of the admitted set, pooled over test years (signal kind, rule, hold)
    fam_tab = {}
    groups = {}
    for Y in test_years:
        ok, score = admit(S, ys, Y, min_prior, None)
        if ok is None:
            continue
        yi = int(np.where(ys == Y)[0][0])
        na, sa = S[0][..., yi], S[1][..., yi]
        for i, n in enumerate(names):
            for ri, rule in enumerate(RULES):
                for hi, hold in enumerate(HOLDS):
                    o = ok[i, ri, hi] & (na[i, ri, hi] > 0)
                    if o.any():
                        for key in (("rule", rule), ("hold", hold), ("kind", n.split("|")[0]), ("region", n.split("|")[1]), ("sig", n.split("|")[-1])):
                            g = groups.setdefault(key, [0.0, 0])
                            g[0] += float((sa[i, ri, hi][o] / na[i, ri, hi][o]).sum()); g[1] += int(o.sum())
    out["family_realised_bp_per_active_day"] = {f"{k[0]}={k[1]}": {"mean_bp": v[0] / v[1] * 1e4, "n": v[1]} for k, v in groups.items() if v[1] >= 50}
    json.dump(out, open(os.path.join(HERE, f"edge_{U}.json"), "w"))
    print("saved", U)


if __name__ == "__main__":
    if sys.argv[1] == "signals":
        build_signals()
    else:
        run_universe(sys.argv[2])
