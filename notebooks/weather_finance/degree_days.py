"""Population-weighted national HDD/CDD forecasts from the GFS archive, per cycle and valid day."""
import os
import numpy as np
import pandas as pd

from gfs_archive import METROS, OUT, STEPS

BASE_F = 65.0


def metro_daily(arr, day):
    """arr: [step, metro] deg C of the 00z cycle of `day`. Returns DataFrame (valid local date x metro) of daily mean deg F for complete days."""
    init = pd.Timestamp(day, tz="UTC")
    idx = init + pd.to_timedelta(STEPS, unit="h")
    df = pd.DataFrame(arr, index=idx).resample("1h").interpolate("linear")
    out = {}
    for j, m in enumerate(METROS):
        s = df[j].tz_convert(m[3])
        g = s.groupby(s.index.date)
        d = g.mean()[g.count() == 24]
        out[j] = pd.Series(d.values * 9 / 5 + 32, index=pd.to_datetime(d.index))
    return pd.DataFrame(out)


def national(arr, day):
    """Population-weighted HDD and CDD by valid date for one cycle (days where every metro has a complete local day)."""
    t = metro_daily(arr, day).dropna()
    w = np.array([m[4] for m in METROS], float)
    w /= w.sum()
    hdd = (np.maximum(BASE_F - t.values, 0) * w).sum(axis=1)
    cdd = (np.maximum(t.values - BASE_F, 0) * w).sum(axis=1)
    return pd.DataFrame({"hdd": hdd, "cdd": cdd, "tmean": (t.values * w).sum(axis=1)}, index=t.index)


def build():
    z = np.load(OUT, allow_pickle=True)
    rows = []
    for day, arr in zip(z["days"], z["data"]):
        n = national(arr, str(day))
        n["cycle"] = pd.Timestamp(str(day))
        rows.append(n.reset_index().rename(columns={"index": "valid"}))
    df = pd.concat(rows, ignore_index=True)
    df["lead"] = (df["valid"] - df["cycle"]).dt.days
    return df


if __name__ == "__main__":
    df = build()
    print(df.shape, df.cycle.nunique())
    print(df[df.cycle == df.cycle.max()].head(10).round(2))
    df.to_csv(os.path.join(os.path.dirname(OUT), "gfs_national_degree_days.csv"), index=False)
