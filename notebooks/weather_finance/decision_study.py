"""Weather forecast signals -> decision rules vs baselines, following STUDY_PLAN_DECISIONS.md. Writes decision_rules.parquet.

  python decision_study.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "climate-energy-dashboard")))
import ng_regional as nr  # noqa: E402
import ng_study as ng  # noqa: E402
from weatherdash.calendar_utils import doy366  # noqa: E402

PRICES = os.path.join(HERE, "data", "equity_prices.parquet")
SPLIT = ng.SPLIT
FORM = (pd.Timestamp("2021-03-23"), SPLIT)
TEST = (SPLIT + pd.Timedelta(days=1), pd.Timestamp("2030-01-01"))
COST = 0.0005
W = ng.WINDOW
REGIONS = {  # name: (cities, [(quantity, season months)])
    "NAT": (nr.NAMES, [("hdd", [11, 12, 1, 2, 3]), ("cdd", [6, 7, 8, 9])]),
    "TX": (["Dallas", "Houston", "Austin", "San Antonio"], [("cdd", [6, 7, 8, 9])]),
    "COLD": (nr.COLD, [("hdd", [11, 12, 1, 2, 3])]),
    "SOUTH": (nr.SOUTH, [("cdd", [6, 7, 8, 9])]),
}
HOLD = [1, 5]


def region_signals(name, cities):
    """Per cycle: seven-day forecast sums, revision, anomaly and forecast-minus-observed-proxy change, plus their climatological analogues."""
    df, w, idx = nr.regional_df(cities)
    n = nr.normals(cities, idx, w)
    p = df.pivot_table(index=["cycle", "valid"], values=["hdd", "cdd"])
    have = set(p.index.get_level_values(0))
    # observed proxy for day d: forecast for d made by the cycle of d-1 (lead 1)
    lead1 = df[df.lead == 1].set_index("valid")[["hdd", "cdd"]]
    rows = []
    for c in sorted(have):
        fut = [c + pd.Timedelta(days=k) for k in W]
        past = [c - pd.Timedelta(days=k) for k in range(1, 8)]
        cur = p.loc[c].reindex(fut)
        if cur.isna().any().any() or not all(d in lead1.index for d in past):
            continue
        row = {"cycle": c}
        for q in ("hdd", "cdd"):
            fut_sum, obs_sum = cur[q].sum(), lead1.loc[past, q].sum()
            nf = n.loc[[int(doy366([d])[0]) for d in fut], q].sum()
            npst = n.loc[[int(doy366([d])[0]) for d in past], q].sum()
            row[f"anom_{q}"] = fut_sum - nf
            row[f"chg_{q}"] = fut_sum - obs_sum
            row[f"chgclim_{q}"] = nf - npst
            c0 = c - pd.Timedelta(days=1)
            if c0 in have:
                old = p.loc[c0].reindex(fut)
                if old.notna().all().all():
                    row[f"rev_{q}"] = (cur[q] - old[q]).sum()
        rows.append(row)
    return pd.DataFrame(rows).set_index("cycle")


def metrics(pnl, ret, pos, spy):
    pnl, ret, pos, spy = pnl.align(ret, join="inner")[0], ret, pos, spy
    act = pos.abs() > 0
    out = {"n_days": int(len(pnl)), "n_active": int(act.sum())}
    if act.sum() < 20:
        return out
    out["bp_active"] = float(pnl[act].mean() * 1e4)
    out["bh_bp_same_days"] = float(ret[act].mean() * 1e4)
    out["ann_mean"] = float(pnl.mean() * 252)
    out["ann_vol"] = float(pnl.std() * np.sqrt(252))
    out["hit"] = float((pnl[act] > 0).mean())
    cum = pnl.cumsum()
    out["maxdd"] = float((cum - cum.cummax()).min())
    s = spy.reindex(pnl.index).fillna(0.0)
    var = s.var()
    beta = float(pnl.cov(s) / var) if var > 0 else np.nan
    out["beta"] = beta
    out["alpha_md_ann"] = float((pnl.mean() - beta * s.mean()) * 252)
    out["exposure"] = float(act.mean())
    return out


def run_rule(sig, ret, spy, kind, hold, period):
    """sig: standardised signal on cycle dates (any calendar day); returns pnl series on trading days within the period."""
    days = ret.index
    s = sig.reindex(days, method=None)  # keep only trading days that are cycle dates
    s = s.reindex(days)
    pos = pd.Series(np.where(s > 0.5, 1.0, np.where(s < -0.5, -1.0, 0.0)), index=days)
    pos[s.isna()] = 0.0
    if kind == "long_only":
        pos = pos.clip(lower=0)
    pos_h = pos.rolling(hold, min_periods=1).mean().shift(1).fillna(0.0)  # overlapping h-day holds, earned from the next day
    if kind == "neutral":
        pnl_raw = pos_h * (ret - spy.reindex(days).fillna(0.0))
        cost = COST * 2 * pos_h.diff().abs().fillna(0.0)
    else:
        pnl_raw = pos_h * ret
        cost = COST * pos_h.diff().abs().fillna(0.0)
    pnl = pnl_raw - cost
    m = (days >= period[0]) & (days <= period[1])
    return pnl[m], pos_h[m], ret[m]


def main():
    px = pd.read_parquet(PRICES)
    rets = np.log(px / px.shift(1))
    spy = rets["SPY"]
    inst = [c for c in px.columns]
    feats = {}
    for rname, (cities, seasons) in REGIONS.items():
        feats[rname] = region_signals(rname, cities)
        print(rname, len(feats[rname]), flush=True)
    rows = []
    for rname, (cities, seasons) in REGIONS.items():
        f = feats[rname]
        for q, months in seasons:
            for fam in ("anom", "rev", "chg"):
                col = f"{fam}_{q}"
                if col not in f:
                    continue
                m_ = np.isin(f.index.month, months)
                s_raw = f.loc[m_, col]
                form = s_raw[(s_raw.index >= FORM[0]) & (s_raw.index <= FORM[1])]
                sd = form.std()
                sig = s_raw / sd
                clim_sig = None
                if fam == "chg":
                    cc = f.loc[m_, f"chgclim_{q}"]
                    clim_sig = cc / cc[(cc.index >= FORM[0]) & (cc.index <= FORM[1])].std()
                for tk in inst:
                    r = rets[tk].dropna()
                    for kind in ("long_short", "long_only", "neutral"):
                        if kind == "neutral" and tk == "SPY":
                            continue
                        for h in HOLD:
                            rec = {"region": rname, "qty": q, "family": fam, "inst": tk, "rule": kind, "hold": h}
                            for per_name, per in (("form", FORM), ("test", TEST)):
                                pnl, pos, rr = run_rule(sig, r, spy, kind, h, per)
                                met = metrics(pnl, rr if kind != "neutral" else rr - spy.reindex(rr.index).fillna(0.0), pos, spy)
                                for k, v in met.items():
                                    rec[f"{per_name}_{k}"] = v
                                if clim_sig is not None:
                                    pnl_c, pos_c, rr_c = run_rule(clim_sig, r, spy, kind, h, per)
                                    mc = metrics(pnl_c, rr_c if kind != "neutral" else rr_c - spy.reindex(rr_c.index).fillna(0.0), pos_c, spy)
                                    rec[f"{per_name}_clim_bp_active"] = mc.get("bp_active")
                            rows.append(rec)
        print(rname, "done", len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(HERE, "decision_rules.parquet"))
    print(len(df), "rule configs")


if __name__ == "__main__":
    main()
