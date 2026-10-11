"""Crop weather -> grain futures, following STUDY_PLAN_CROPS.md (definitions fixed there). Writes crop_futures_results.json.

  python crop_futures_study.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PRICES = os.path.join(HERE, "..", "multiasset_prices_live_history.parquet")
WEEKLY = os.path.join(HERE, "data", "crop_weekly.csv")
SPLIT_YEAR = 2020
FEATS = ["kdd", "hot_days", "vpd", "rain"]
TICK = {"CORN": "ZC=F", "SOYBEANS": "ZS=F"}


def build(crop, prices):
    w = pd.read_csv(WEEKLY, index_col=0, parse_dates=True)
    w = w[w.crop == crop].copy()
    w["iso"] = w["week"]
    form = w[w.year <= SPLIT_YEAR]
    z = pd.DataFrame(index=w.index)
    for f in FEATS:
        clim = form.groupby("iso")[f].mean()
        anom = w[f] - w["iso"].map(clim)
        sd = (form[f] - form["iso"].map(clim)).std()
        z[f] = anom / sd
    w["S"] = (z["kdd"] + z["hot_days"] + z["vpd"] - z["rain"]) / 4
    px = prices[TICK[crop]].dropna()
    days = px.index
    rows = []
    for d in w.index:  # d = Sunday week end
        t0i = days.searchsorted(d + pd.Timedelta(days=1))  # first trading day on or after Monday
        if t0i + 21 >= len(days) or t0i < 6:
            continue
        t0 = days[t0i]
        fri = days[days.searchsorted(d, side="right") - 1]
        pfri = px.loc[fri]
        prev = days[days.searchsorted(d - pd.Timedelta(days=7), side="right") - 1]
        rows.append({"week_ending": d, "R_wk": np.log(pfri / px.loc[prev]), "R_rel": np.log(px.iloc[t0i + 1] / px.iloc[t0i]),
                     "R_5": np.log(px.iloc[t0i + 5] / px.iloc[t0i]), "R_21": np.log(px.iloc[t0i + 21] / px.iloc[t0i]),
                     "R_pre": np.log(px.iloc[t0i] / pfri)})
    r = pd.DataFrame(rows).set_index("week_ending")
    return w.join(r, how="inner")


def block(d, form):
    out = {"n": int(len(d))}
    for sig in ("S", "p", "e", "dge"):
        for ret in ("R_wk", "R_pre", "R_rel", "R_5", "R_21"):
            out[f"sp_{sig}_{ret}"] = float(d[sig].corr(d[ret], method="spearman"))
            out[f"pe_{sig}_{ret}"] = float(d[sig].corr(d[ret]))
    # tails of S (formation cut-offs), conditional exceedance of the forward return
    for ret in ("R_5", "R_21"):
        hi_s, lo_s = form["S"].quantile(0.8), form["S"].quantile(0.2)
        hi_r, lo_r = form[ret].quantile(0.8), form[ret].quantile(0.2)
        a, b = d[d.S >= hi_s], d[d.S <= lo_s]
        out[f"P_top_{ret}"] = [float((a[ret] >= hi_r).mean()) if len(a) else None, int(len(a))]
        out[f"P_bot_{ret}"] = [float((b[ret] <= lo_r).mean()) if len(b) else None, int(len(b))]
        out[f"mean_{ret}_Stop"] = float(a[ret].mean()) if len(a) else None
        out[f"mean_{ret}_Sbot"] = float(b[ret].mean()) if len(b) else None
        out[f"mean_{ret}_all"] = float(d[ret].mean())
    # release reaction by tercile of the actual rating change and of its parts
    for sig in ("dge", "p", "e"):
        q = form[sig].quantile([1 / 3, 2 / 3]).values
        lo, mid, hi = d[d[sig] <= q[0]], d[(d[sig] > q[0]) & (d[sig] <= q[1])], d[d[sig] > q[1]]
        out[f"rel_by_{sig}_tercile"] = {"low": [float(lo.R_rel.mean()), int(len(lo))], "mid": [float(mid.R_rel.mean()), int(len(mid))],
                                       "high": [float(hi.R_rel.mean()), int(len(hi))]}
    return out


def main():
    prices = pd.read_parquet(PRICES)
    res = {}
    for crop in ("CORN", "SOYBEANS"):
        d = build(crop, prices)
        form = d[d.year <= SPLIT_YEAR]
        res[crop] = {"formation": block(form, form), "test": block(d[d.year > SPLIT_YEAR], form),
                     "summer_formation": block(form[form.week <= 34], form), "summer_test": block(d[(d.year > SPLIT_YEAR) & (d.week <= 34)], form)}
        d.to_csv(os.path.join(HERE, "data", f"crop_futures_{crop.lower()}.csv"))
    json.dump(res, open(os.path.join(HERE, "crop_futures_results.json"), "w"), indent=1)
    for crop, r in res.items():
        for per in ("formation", "test"):
            b = r[per]
            print(f"\n{crop} {per} n={b['n']}")
            for ret in ("R_wk", "R_pre", "R_rel", "R_5", "R_21"):
                print(f"  {ret:6s} spearman  S {b['sp_S_'+ret]:+.2f}  p {b['sp_p_'+ret]:+.2f}  e {b['sp_e_'+ret]:+.2f}  dge {b['sp_dge_'+ret]:+.2f}")
            for ret in ("R_5", "R_21"):
                print(f"  {ret}: P(top20 ret | S top20) {b['P_top_'+ret]}  P(bottom | S bottom) {b['P_bot_'+ret]}   mean ret S-top {b['mean_'+ret+'_Stop']}, S-bottom {b['mean_'+ret+'_Sbot']}, all {b['mean_'+ret+'_all']}")
            print("  release reaction by dge tercile", b["rel_by_dge_tercile"])
            print("  release reaction by e   tercile", b["rel_by_e_tercile"])
            print("  release reaction by p   tercile", b["rel_by_p_tercile"])


if __name__ == "__main__":
    main()
