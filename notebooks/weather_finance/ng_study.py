"""
Natural-gas study, following STUDY_PLAN.md. Definitions are fixed there; this script only computes them.

  python ng_study.py formation      # report on 2021-03 .. 2023-12 (development period)
  python ng_study.py test           # report on 2024-01 .. present (touch only once the definitions are final)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "climate-energy-dashboard")))
import degree_days as dd  # noqa: E402
from weatherdash.calendar_utils import doy366  # noqa: E402

SPLIT = pd.Timestamp("2023-12-31")
PRICES = os.path.join(HERE, "..", "multiasset_prices_live_history.parquet")
NORMALS = os.path.join(HERE, "data", "era5_degree_day_normals.csv")  # optional: day-of-year HDD/CDD normals
WINDOW = range(1, 8)  # valid days t+1 .. t+7


def cycle_features(df):
    """One row per cycle: forecast degree-day sums over days t+1..t+7, their revision vs the previous day's cycle
    (same valid days), and, if normals are available, the anomaly vs climatology."""
    p = df.pivot_table(index=["cycle", "valid"], values=["hdd", "cdd"])  # every lead, so the previous cycle covers the same valid days
    have = set(p.index.get_level_values(0))
    rows = []
    for c in sorted(have):
        valid = [c + pd.Timedelta(days=k) for k in WINDOW]
        cur = p.loc[c].reindex(valid)
        if cur.isna().any().any():
            continue
        row = {"cycle": c, "hdd7": cur["hdd"].sum(), "cdd7": cur["cdd"].sum()}
        c0 = c - pd.Timedelta(days=1)
        if c0 in have:
            old = p.loc[c0].reindex(valid)
            if old.notna().all().all():
                row["r_hdd"] = (cur["hdd"] - old["hdd"]).sum()
                row["r_cdd"] = (cur["cdd"] - old["cdd"]).sum()
        rows.append(row)
    f = pd.DataFrame(rows).set_index("cycle")
    if os.path.exists(NORMALS):
        n = pd.read_csv(NORMALS, index_col=0)  # day-of-year -> hdd, cdd
        an_h, an_c = [], []
        for c in f.index:
            doys = [int(doy366([c + pd.Timedelta(days=k)])[0]) for k in WINDOW]
            an_h.append(f.loc[c, "hdd7"] - n.loc[doys, "hdd"].sum())
            an_c.append(f.loc[c, "cdd7"] - n.loc[doys, "cdd"].sum())
        f["a_hdd"], f["a_cdd"] = an_h, an_c
    return f


def season_signal(f, kind):
    """Signal oriented so that positive = bullish gas: HDD in the heating season (Nov-Mar), CDD in the cooling season (Jun-Sep)."""
    col = {"rev": ("r_hdd", "r_cdd"), "anom": ("a_hdd", "a_cdd")}[kind]
    m = f.index.month
    s = pd.Series(np.nan, index=f.index)
    heat = np.isin(m, [11, 12, 1, 2, 3])
    cool = np.isin(m, [6, 7, 8, 9])
    if col[0] in f:
        s[heat] = f.loc[heat, col[0]]
        s[cool] = f.loc[cool, col[1]]
    return s, pd.Series(np.where(heat, "heat", np.where(cool, "cool", "off")), index=f.index)


def returns(ticker="NG=F"):
    p = pd.read_parquet(PRICES)[ticker].dropna()
    r1 = np.log(p.shift(-1) / p)
    r5 = np.log(p.shift(-5) / p)
    return pd.DataFrame({"r1": r1, "r5": r5})


def report(kind, start, end, split_stats=None, ticker="NG=F", cost=0.0005):
    df = dd.build()
    f = cycle_features(df)
    sig, season = season_signal(f, kind)
    ret = returns(ticker)
    d = pd.DataFrame({"sig": sig, "season": season}).join(ret, how="inner").dropna()
    d = d[(d.season != "off")]
    form = d[d.index <= SPLIT]
    # normalisation and tail thresholds come from the formation period only
    stats = split_stats or {}
    if not stats:
        for s in ("heat", "cool"):
            x = form[form.season == s]
            stats[s] = {"sd": float(x.sig.std()), "q_hi": float(x.sig.quantile(0.8)), "q_lo": float(x.sig.quantile(0.2)),
                        "r1_hi": float(x.r1.quantile(0.8)), "r1_lo": float(x.r1.quantile(0.2)),
                        "r5_hi": float(x.r5.quantile(0.8)), "r5_lo": float(x.r5.quantile(0.2))}
    x = d[(d.index >= start) & (d.index <= end)].copy()
    print(f"\n=== {kind}  {ticker}  {start.date()} .. {end.date()}   n={len(x)} signal-days "
          f"(heat {int((x.season=='heat').sum())}, cool {int((x.season=='cool').sum())})")
    for s in ("heat", "cool"):
        y = x[x.season == s]
        if len(y) < 30:
            print(f"  {s}: only {len(y)} days"); continue
        st = stats[s]
        z = y.sig / st["sd"]
        out = {"corr_r1": round(float(y.sig.corr(y.r1)), 3), "corr_r5": round(float(y.sig.corr(y.r5)), 3)}
        hi, lo = y.sig >= st["q_hi"], y.sig <= st["q_lo"]
        for h in ("r1", "r5"):
            out[f"P({h}>=q80 | sig top20)"] = (round(float((y.loc[hi, h] >= st[f"{h}_hi"]).mean()), 3), int(hi.sum()))
            out[f"P({h}<=q20 | sig bottom20)"] = (round(float((y.loc[lo, h] <= st[f"{h}_lo"]).mean()), 3), int(lo.sum()))
        pos = np.where(z > 0.5, 1.0, np.where(z < -0.5, -1.0, 0.0))
        pnl = pos * y.r1 - cost * np.abs(np.diff(np.r_[0, pos]))
        act = pos != 0
        out["rule: active share"] = round(float(act.mean()), 3)
        out["rule: mean r1 per active day (bp)"] = round(float(pnl[act].mean() * 1e4), 1) if act.any() else None
        out["buy&hold mean r1 same days (bp)"] = round(float(y.r1.mean() * 1e4), 1)
        out["rule: ann. mean/vol on all days"] = (round(float(pnl.mean() * 252), 3), round(float(pnl.std() * np.sqrt(252)), 3))
        print(f"  {s}: n={len(y)}  ", out)
    return stats


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "formation"
    for kind in ("rev", "anom"):
        try:
            st = report(kind, pd.Timestamp("2021-03-23"), SPLIT)
            if which == "test":
                report(kind, SPLIT + pd.Timedelta(days=1), pd.Timestamp("2030-01-01"), split_stats=st)
        except Exception as e:  # noqa: BLE001
            print(kind, "skipped:", type(e).__name__, e)
