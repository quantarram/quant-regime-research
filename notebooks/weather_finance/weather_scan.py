"""Weather-impact predictors vs the instrument universe, following STUDY_PLAN_SCAN.md. Writes scan_results.parquet and scan_summary.json.

  python weather_scan.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "climate-energy-dashboard")))
from gfs_archive import METROS  # noqa: E402
from weatherdash.calendar_utils import doy366  # noqa: E402

PRICES = os.path.join(HERE, "..", "multiasset_prices_live_history.parquet")
META = os.path.join(HERE, "..", "multiasset_metadata.parquet")
NAMES = [m[0] for m in METROS]
POP = {m[0]: m[4] for m in METROS}
FORM_END = pd.Timestamp("2012-12-31")
TEST_START = pd.Timestamp("2013-01-01")
COLD = ["New York", "Chicago", "Washington", "Philadelphia", "Boston", "Detroit", "Minneapolis", "Denver", "Baltimore", "St. Louis",
        "Pittsburgh", "Cincinnati", "Kansas City", "Columbus", "Indianapolis", "Cleveland"]
SOUTH = ["Dallas", "Houston", "Miami", "Atlanta", "Phoenix", "Tampa", "Orlando", "Charlotte", "San Antonio", "Austin", "Las Vegas", "Nashville"]
TX = ["Dallas", "Houston", "Austin", "San Antonio"]
MW = ["Chicago", "Minneapolis", "Kansas City", "St. Louis", "Indianapolis", "Columbus", "Detroit"]
HORIZONS = [1, 5, 21, 63]
MIN_EPISODES, MIN_P, GAP = 5, 0.30, 10

# id: (cities, quantity, window, season months)
PREDICTORS = {
    "HDD7_nat": (NAMES, "hdd", 7, [11, 12, 1, 2, 3]), "HDD28_nat": (NAMES, "hdd", 28, [11, 12, 1, 2, 3]),
    "HDD7_cold": (COLD, "hdd", 7, [11, 12, 1, 2, 3]), "HDD28_cold": (COLD, "hdd", 28, [11, 12, 1, 2, 3]),
    "CDD7_nat": (NAMES, "cdd", 7, [6, 7, 8, 9]), "CDD28_nat": (NAMES, "cdd", 28, [6, 7, 8, 9]),
    "CDD7_south": (SOUTH, "cdd", 7, [6, 7, 8, 9]), "CDD28_south": (SOUTH, "cdd", 28, [6, 7, 8, 9]),
    "CDD7_tx": (TX, "cdd", 7, [6, 7, 8, 9]), "CDD7_mw": (MW, "cdd", 7, [6, 7, 8]),
}


def temps():
    return pd.DataFrame({m: pd.read_csv(os.path.join(HERE, "data", "metro_hist", f"{m.replace(' ', '_').replace('.', '')}.csv"),
                                        index_col=0, parse_dates=True)["t"] for m in NAMES}) * 9 / 5 + 32


def predictor(T, cities, q, window):
    w = np.array([POP[c] for c in cities], float)
    w /= w.sum()
    X = T[cities].values
    dd = (np.maximum(65 - X, 0) if q == "hdd" else np.maximum(X - 65, 0)) * w
    s = pd.Series(dd.sum(1), index=T.index)
    form = s[s.index <= FORM_END]
    n = form.groupby(doy366(form.index)).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    sm = pd.concat([n.iloc[-7:], n, n.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
    sm.index = range(1, 367)
    anom = s - pd.Series(sm.loc[doy366(s.index)].values, index=s.index)
    return anom.rolling(window).sum()


def episodes(mask, dates):
    d = dates[mask]
    if len(d) == 0:
        return 0
    gaps = np.diff(np.arange(len(mask))[mask])
    return int(1 + (gaps > GAP).sum())


def main():
    T = temps()
    px = pd.read_parquet(PRICES)
    n_pre = px[px.index <= FORM_END].notna().sum()
    inst = [c for c in px.columns if n_pre[c] >= 1000 and not c.startswith("^")]
    print(len(inst), "instruments with >=1000 obs before 2013")
    rows = []
    for pid, (cities, q, window, months) in PREDICTORS.items():
        sig = predictor(T, cities, q, window).shift(1)  # weather through the previous calendar day
        for tk in inst:
            p = px[tk].dropna()
            p = p[p.index >= "1996-01-01"]
            for h in HORIZONS:
                r = np.log(p.shift(-h) / p).rename("r")
                d = pd.concat([sig.rename("s"), r], axis=1, join="inner").dropna()
                d = d[np.isin(d.index.month, months)]
                form, test = d[d.index <= FORM_END], d[d.index >= TEST_START]
                if len(form) < 500 or len(test) < 500:
                    continue
                s_hi, s_lo = form.s.quantile(0.8), form.s.quantile(0.2)
                r_hi, r_lo = form.r.quantile(0.8), form.r.quantile(0.2)
                for tail, tmask_f, tmask_t in (("top", form.s >= s_hi, test.s >= s_hi), ("bot", form.s <= s_lo, test.s <= s_lo)):
                    for ev in ("up", "down"):
                        ef = (form.r >= r_hi) if ev == "up" else (form.r <= r_lo)
                        et = (test.r >= r_hi) if ev == "up" else (test.r <= r_lo)
                        nf, nt = int(tmask_f.sum()), int(tmask_t.sum())
                        if nf < 100 or nt < 100:
                            continue
                        rows.append({"pred": pid, "inst": tk, "h": h, "tail": tail, "event": ev,
                                     "p_form": float(ef[tmask_f].mean()), "base_form": float(ef.mean()), "n_form": nf,
                                     "eps_form": episodes(tmask_f.values, form.index),
                                     "p_test": float(et[tmask_t].mean()), "base_test": float(et.mean()), "n_test": nt,
                                     "eps_test": episodes(tmask_t.values, test.index)})
        print(pid, len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(HERE, "scan_results.parquet"))
    adm = df[(df.eps_form >= MIN_EPISODES) & (df.p_form >= MIN_P)].copy()
    adm["lift_test"] = adm.p_test - adm.base_test
    adm["lift_form"] = adm.p_form - adm.base_form
    out = {"n_configs": int(len(df)), "n_admitted": int(len(adm)), "n_instruments": len(inst),
           "admitted_mean_form_p": float(adm.p_form.mean()), "admitted_mean_test_p": float(adm.p_test.mean()),
           "admitted_mean_test_base": float(adm.base_test.mean()), "admitted_mean_lift_test": float(adm.lift_test.mean()),
           "admitted_median_lift_test": float(adm.lift_test.median()), "admitted_share_lift_gt0": float((adm.lift_test > 0).mean()),
           "all_mean_lift_test": float((df.p_test - df.base_test).mean())}
    for k in ("pred", "h", "tail", "event"):
        out["by_" + k] = {str(g): {"n": int(len(x)), "mean_lift_test": float(x.lift_test.mean()), "share_gt0": float((x.lift_test > 0).mean())}
                          for g, x in adm.groupby(k)}
    json.dump(out, open(os.path.join(HERE, "scan_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("by_")}, indent=1))
    pd.set_option("display.width", 250)
    print(adm.sort_values("lift_test", ascending=False).head(12)[["pred", "inst", "h", "tail", "event", "p_form", "eps_form", "p_test", "base_test", "n_test", "eps_test"]].round(3).to_string(index=False))
    for k in ("pred", "h", "event"):
        print(pd.DataFrame(out["by_" + k]).T.round(3))


if __name__ == "__main__":
    main()
