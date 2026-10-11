"""Natural gas vs regional GFS degree-day signals, following STUDY_PLAN_GAS_REGIONAL.md. Writes gas_regional_results.json.

  python ng_regional.py
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
from gfs_archive import METROS, OUT  # noqa: E402
from weatherdash.calendar_utils import doy366  # noqa: E402

COLD = ["New York", "Chicago", "Washington", "Philadelphia", "Boston", "Detroit", "Minneapolis", "Denver", "Baltimore", "St. Louis",
        "Pittsburgh", "Cincinnati", "Kansas City", "Columbus", "Indianapolis", "Cleveland"]
SOUTH = ["Dallas", "Houston", "Miami", "Atlanta", "Phoenix", "Tampa", "Orlando", "Charlotte", "San Antonio", "Austin", "Las Vegas", "Nashville"]
NAMES = [m[0] for m in METROS]
SPLIT = ng.SPLIT
PERIODS = {"formation": (pd.Timestamp("2021-03-23"), SPLIT), "test": (SPLIT + pd.Timedelta(days=1), pd.Timestamp("2030-01-01"))}


def regional_df(names):
    idx = [NAMES.index(n) for n in names]
    w = np.array([METROS[i][4] for i in idx], float)
    w /= w.sum()
    z = np.load(OUT, allow_pickle=True)
    rows = []
    for day, arr in zip(z["days"], z["data"]):
        t = dd.metro_daily(arr, str(day))[idx].dropna()
        hdd = (np.maximum(65 - t.values, 0) * w).sum(1)
        cdd = (np.maximum(t.values - 65, 0) * w).sum(1)
        n = pd.DataFrame({"hdd": hdd, "cdd": cdd}, index=t.index)
        n["cycle"] = pd.Timestamp(str(day))
        rows.append(n.reset_index().rename(columns={"index": "valid"}))
    df = pd.concat(rows, ignore_index=True)
    df["lead"] = (df["valid"] - df["cycle"]).dt.days
    return df, w, idx


def normals(names, idx, w):
    T = pd.DataFrame({m: pd.read_csv(os.path.join(HERE, "data", "metro_hist", f"{m.replace(' ', '_').replace('.', '')}.csv"),
                                     index_col=0, parse_dates=True)["t"] for m in names}) * 9 / 5 + 32
    hdd = pd.Series((np.maximum(65 - T.values, 0) * w).sum(1), index=T.index)
    cdd = pd.Series((np.maximum(T.values - 65, 0) * w).sum(1), index=T.index)
    d = pd.DataFrame({"hdd": hdd, "cdd": cdd}).groupby(doy366(T.index)).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    sm = pd.concat([d.iloc[-7:], d, d.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
    sm.index = range(1, 367)
    return sm


def features(names):
    df, w, idx = regional_df(names)
    n = normals(names, idx, w)
    ng.NORMALS = "/nonexistent"  # cycle_features would read the national normals from disk; the regional anomaly is computed below
    f = ng.cycle_features(df)
    an_h, an_c = [], []
    for c in f.index:
        doys = [int(doy366([c + pd.Timedelta(days=k)])[0]) for k in ng.WINDOW]
        an_h.append(f.loc[c, "hdd7"] - n.loc[doys, "hdd"].sum())
        an_c.append(f.loc[c, "cdd7"] - n.loc[doys, "cdd"].sum())
    f["a_hdd"], f["a_cdd"] = an_h, an_c
    return f


def run():
    p = pd.read_parquet(ng.PRICES)["NG=F"].dropna()
    ret = pd.DataFrame({"r0": np.log(p / p.shift(1)), "r1": np.log(p.shift(-1) / p), "r5": np.log(p.shift(-5) / p)})
    res = []
    for region, names, season, hcol in (("COLD", COLD, "heat", "hdd"), ("SOUTH", SOUTH, "cool", "cdd")):
        f = features(names)
        months = [11, 12, 1, 2, 3] if season == "heat" else [6, 7, 8, 9]
        for kind, col in (("rev", "r_" + hcol), ("anom", "a_" + hcol)):
            s = f.loc[np.isin(f.index.month, months), col]
            d = pd.DataFrame({"sig": s}).join(ret, how="inner").dropna()
            form = d[d.index <= SPLIT]
            sd, q_hi, q_lo = form.sig.std(), form.sig.quantile(0.8), form.sig.quantile(0.2)
            r1_hi, r1_lo = form.r1.quantile(0.8), form.r1.quantile(0.2)
            abs_hi = form.sig.abs().quantile(0.8)
            for per, (a, b) in PERIODS.items():
                y = d[(d.index >= a) & (d.index <= b)]
                z = y.sig / sd
                pos = np.where(z > 0.5, 1.0, np.where(z < -0.5, -1.0, 0.0))
                pnl = pos * y.r1 - 0.0005 * np.abs(np.diff(np.r_[0, pos]))
                act = pos != 0
                hi, lo = y.sig >= q_hi, y.sig <= q_lo
                big = y.sig.abs() >= abs_hi
                res.append({"region": region, "season": season, "signal": kind, "period": per, "n": int(len(y)),
                            "corr_r0": float(y.sig.corr(y.r0)), "corr_r1": float(y.sig.corr(y.r1)), "corr_r5": float(y.sig.corr(y.r5)),
                            "p_top_r1": float((y.loc[hi, "r1"] >= r1_hi).mean()) if hi.any() else None, "n_top": int(hi.sum()),
                            "p_bot_r1": float((y.loc[lo, "r1"] <= r1_lo).mean()) if lo.any() else None, "n_bot": int(lo.sum()),
                            "rule_bp_active": float(pnl[act].mean() * 1e4) if act.any() else None, "bh_bp": float(y.r1.mean() * 1e4),
                            "abs_sp_r0": float(y.sig.abs().corr(y.r0.abs(), method="spearman")),
                            "abs_sp_r1": float(y.sig.abs().corr(y.r1.abs(), method="spearman")),
                            "absret_big_bp": float(y.loc[big, "r1"].abs().mean() * 1e4) if big.any() else None, "n_big": int(big.sum()),
                            "absret_other_bp": float(y.loc[~big, "r1"].abs().mean() * 1e4)})
    json.dump(res, open(os.path.join(HERE, "gas_regional_results.json"), "w"), indent=1)
    df = pd.DataFrame(res)
    pd.set_option("display.width", 250)
    print(df.round(3).drop(columns=["season"]).to_string(index=False))


if __name__ == "__main__":
    run()
