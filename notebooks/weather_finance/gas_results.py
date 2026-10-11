"""Collects every natural-gas number used in the paper into gas_results.json (definitions from STUDY_PLAN.md, unchanged).

  python gas_results.py

Adds one diagnostic that was not in the plan: the same anomaly built from ERA5 *observed* weather (perfect foresight), to separate
"the forecast is poor" from "the market does not trade this weather quantity this way". ERA5 history in data/metro_hist ends 2025-12-31,
so that diagnostic's test period is 2024-01 .. 2025-12.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "climate-energy-dashboard")))
import degree_days as dd  # noqa: E402
import ng_study as ng  # noqa: E402
from gfs_archive import METROS  # noqa: E402
from weatherdash.calendar_utils import doy366  # noqa: E402

SPLIT = ng.SPLIT
PERIODS = {"formation": (pd.Timestamp("2021-03-23"), SPLIT), "test": (SPLIT + pd.Timedelta(days=1), pd.Timestamp("2030-01-01"))}
W = ng.WINDOW


def observed_anomaly(f):
    w = np.array([m[4] for m in METROS], float)
    w /= w.sum()
    T = pd.DataFrame({m[0]: pd.read_csv(os.path.join(HERE, "data", "metro_hist", f"{m[0].replace(' ', '_').replace('.', '')}.csv"),
                                        index_col=0, parse_dates=True)["t"] for m in METROS}) * 9 / 5 + 32
    R = pd.DataFrame({"hdd": (np.maximum(65 - T.values, 0) * w).sum(1), "cdd": (np.maximum(T.values - 65, 0) * w).sum(1)}, index=T.index)
    n = pd.read_csv(ng.NORMALS, index_col=0)
    out = {}
    for c in f.index:
        v = [c + pd.Timedelta(days=k) for k in W]
        if all(x in R.index for x in v):
            doys = [int(doy366([x])[0]) for x in v]
            out[c] = {"o_hdd": R.loc[v, "hdd"].sum() - n.loc[doys, "hdd"].sum(), "o_cdd": R.loc[v, "cdd"].sum() - n.loc[doys, "cdd"].sum()}
    return pd.DataFrame(out).T


def main():
    df = dd.build()
    f = ng.cycle_features(df)
    f = f.join(observed_anomaly(f), how="left")
    p = pd.read_parquet(ng.PRICES)["NG=F"].dropna()
    ret = pd.DataFrame({"r0": np.log(p / p.shift(1)), "r1": np.log(p.shift(-1) / p), "r5": np.log(p.shift(-5) / p)})
    res = {"meta": {"cycles": int(len(f)), "first_cycle": str(f.index.min().date()), "last_cycle": str(f.index.max().date()),
                    "ng_last": str(p.index.max().date())}, "signals": [], "oracle": []}
    stats_by_kind = {}
    for kind in ("rev", "anom"):
        sig, season = ng.season_signal(f, kind)
        d = pd.DataFrame({"sig": sig, "season": season}).join(ret, how="inner").dropna()
        d = d[d.season != "off"]
        form = d[d.index <= SPLIT]
        st = {}
        for s in ("heat", "cool"):
            x = form[form.season == s]
            st[s] = {"sd": x.sig.std(), "q_hi": x.sig.quantile(0.8), "q_lo": x.sig.quantile(0.2),
                     "r1_hi": x.r1.quantile(0.8), "r1_lo": x.r1.quantile(0.2), "r5_hi": x.r5.quantile(0.8), "r5_lo": x.r5.quantile(0.2)}
        stats_by_kind[kind] = st
        for per, (a, b) in PERIODS.items():
            for s in ("heat", "cool"):
                y = d[(d.index >= a) & (d.index <= b) & (d.season == s)]
                z = y.sig / st[s]["sd"]
                hi = y.sig >= st[s]["q_hi"]
                lo = y.sig <= st[s]["q_lo"]
                pos = np.where(z > 0.5, 1.0, np.where(z < -0.5, -1.0, 0.0))
                pnl = pos * y.r1 - 0.0005 * np.abs(np.diff(np.r_[0, pos]))
                act = pos != 0
                res["signals"].append({
                    "signal": kind, "season": s, "period": per, "n": int(len(y)),
                    "corr_same_day": float(y.sig.corr(y.r0)), "corr_r1": float(y.sig.corr(y.r1)), "corr_r5": float(y.sig.corr(y.r5)),
                    "p_top_r1": float((y.loc[hi, "r1"] >= st[s]["r1_hi"]).mean()), "n_top": int(hi.sum()),
                    "p_bot_r1": float((y.loc[lo, "r1"] <= st[s]["r1_lo"]).mean()), "n_bot": int(lo.sum()),
                    "rule_bp_active": float(pnl[act].mean() * 1e4), "bh_bp": float(y.r1.mean() * 1e4), "active_share": float(act.mean())})
    # perfect-foresight diagnostic, same seasons / periods, windows ending 2025-12-31
    for s, col_o, col_g, months in (("heat", "o_hdd", "a_hdd", [11, 12, 1, 2, 3]), ("cool", "o_cdd", "a_cdd", [6, 7, 8, 9])):
        for per, (a, b) in PERIODS.items():
            b = min(b, pd.Timestamp("2025-12-24"))
            d = f[np.isin(f.index.month, months)][[col_o, col_g]].join(ret, how="inner").dropna()
            y = d[(d.index >= a) & (d.index <= b)]
            res["oracle"].append({"season": s, "period": per, "n": int(len(y)),
                                  "obs_r1": float(y[col_o].corr(y.r1)), "obs_r5": float(y[col_o].corr(y.r5)),
                                  "gfs_r1": float(y[col_g].corr(y.r1)), "gfs_r5": float(y[col_g].corr(y.r5)),
                                  "gfs_vs_obs": float(y[col_g].corr(y[col_o]))})
    with open(os.path.join(HERE, "gas_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for r in res["oracle"]:
        print(r)


if __name__ == "__main__":
    main()
