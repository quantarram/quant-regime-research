"""Weekly corn/soy weather drivers and weather-explained condition change (replicating the live dashboard model), no network.

  python crop_weekly.py        # writes data/crop_weekly.csv and prints the replication check against the live model's LOYO error
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "climate-energy-dashboard"))
sys.path.insert(0, DASH)
os.environ["OPEN_METEO_NO_WAIT"] = "1"
from weatherdash import crops, truth  # noqa: E402

CACHE = os.path.join(DASH, ".cache", "era5")
NASS = os.path.join(DASH, "data", "nass_condition.csv")
OUT = os.path.join(HERE, "data", "crop_weekly.csv")
RIDGE = 20.0
FEATS = ["kdd", "hot_days", "vpd", "rain"]


def state_week(st):
    tz, pts = crops.ALL_REGIONS[st]
    daily = []
    for la, lo in pts:
        h = pd.concat([pd.read_csv(os.path.join(CACHE, f"crop_{la}_{lo}_{tz.replace('/', '-')}_{y}.csv"), index_col=0, parse_dates=True)
                       for y in range(2015, 2026)])
        daily.append(crops.daily_from_hourly(h, tz))
    idx = daily[0].index
    for d in daily[1:]:
        idx = idx.intersection(d.index)
    tmax = pd.concat([d.loc[idx, "tmax"] for d in daily], axis=1)
    kdd = np.maximum(tmax - 30.0, 0).mean(axis=1)
    hot = (tmax > 35.0).mean(axis=1)
    vpd = pd.concat([d.loc[idx, "vpd_pm"] for d in daily], axis=1).mean(axis=1)
    rain = pd.concat([d.loc[idx, "rain"] for d in daily], axis=1).mean(axis=1)
    df = pd.DataFrame({"kdd": kdd, "hot": hot, "vpd": vpd, "rain": rain}, index=pd.to_datetime(idx))
    g = df.groupby(df.index.to_period("W-SUN"))
    w = pd.DataFrame({"kdd": g["kdd"].sum(), "hot_days": g["hot"].sum(), "vpd": g["vpd"].mean(), "rain": g["rain"].sum(), "n": g["kdd"].count()})
    w = w[w["n"] == 7].drop(columns="n")
    w.index = w.index.end_time.normalize()  # Sunday
    return w


def ridge_fit(X, y, alpha=RIDGE):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    reg = alpha * np.eye(A.shape[1]); reg[0, 0] = 0
    return np.linalg.solve(A.T @ A + reg, A.T @ y), mu, sd


def ridge_pred(b, mu, sd, X):
    return np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ b


def main():
    wk = {st: state_week(st) for st in crops.CROP_STATES}
    nass = pd.read_csv(NASS)
    nass["week_ending"] = pd.to_datetime(nass["week_ending"])
    res = {}
    for crop in ("CORN", "SOYBEANS"):
        n = nass[(nass.commodity == crop) & (nass["class"] == "ALL CLASSES") & nass.state.isin(crops.CROP_STATES)].copy()
        n["ge"] = n["good"] + n["excellent"]
        rows = []
        for st, g in n.groupby("state"):
            g = g.sort_values("week_ending").set_index("week_ending")["ge"]
            d = g.diff()[g.index.to_series().diff().dt.days == 7]
            rows.append(pd.DataFrame({"dge": d}).join(wk[st], how="inner").dropna())
        D = pd.concat(rows)
        cnt = D.groupby(D.index)["dge"].count()
        A = D.groupby(D.index).mean()[FEATS + ["dge"]]
        A = A[cnt.reindex(A.index) >= 8]
        A["week"] = A.index.isocalendar().week.astype(int).values
        A["year"] = A.index.year
        A = A[(A.week >= 22) & (A.week <= 38)].copy()
        A["season"] = (A["week"] - 30) / 8.0
        feats = FEATS + ["season"]
        pred = pd.Series(np.nan, index=A.index)
        for y in sorted(A.year.unique()):
            tr, te = A[A.year != y], A[A.year == y]
            b, mu, sd = ridge_fit(tr[feats].values, tr["dge"].values)
            pred[te.index] = ridge_pred(b, mu, sd, te[feats].values)
        A["p"] = pred
        A["e"] = A["dge"] - A["p"]
        A["crop"] = crop
        res[crop] = A
        print(f"{crop}: weeks {len(A)}  years {A.year.min()}-{A.year.max()}  LOYO MAE {np.abs(A.e).mean():.3f}  "
              f"zero {np.abs(A.dge).mean():.3f}  corr {np.corrcoef(A.p, A.dge)[0, 1]:.3f}")
    out = pd.concat(res.values())
    out.index.name = "week_ending"
    out.to_csv(OUT)
    live = json.load(open(os.path.join(DASH, "data", "crops_model.json")))["models"]
    print("live model LOYO (all seasons incl. 2026):", {k: (v["loyo_mae_model"], v["n_weeks"]) for k, v in live.items()})


if __name__ == "__main__":
    main()
