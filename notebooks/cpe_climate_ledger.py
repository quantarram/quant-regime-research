#!/usr/bin/env python3
"""
============================================================
  CPE CLIMATE SIGNAL LEDGER
  Dr. Arun Ramanathan
============================================================
The 43 climate rows in cpe_climate_table.parquet are the survivors of an engine-faithful pairwise CPE run on climate predictors (ONI, Nino 3.4, SOI, AO, NAO, PDO as slow
state series): rows that cleared the engine's gates and the economic prior, rested on at least four separated formation episodes (through 2012), and held up in a chronological
test (>= 3 test episodes with a hit rate of at least two thirds and a test CPE at least ten points above the test base rate). The survivor filter used data through
2026-10-10, so nothing before the freeze date is out-of-sample for this table; this ledger records and scores only what happens after it, exactly as cpe_signal_ledger.py does
for the main table.

What is stored (cpe_climate_events.csv): one row per (entry_date, X, tau_past, q_X, direction): the first trading day on which the climate state crossed its frozen threshold after
a day on which it had not (plus, for conditions already in the tail on the freeze date, an 'ongoing' entry dated at the freeze). Thresholds for X and Y are frozen in the table
(from formation data through 2012-12-31). Each event fans out to every table row of that condition; scoring uses the engine's definitions: Y's forward tau_future log move beyond
the frozen Y threshold (above for bullish, below for bearish), horizons in trading days, an outcome counts once its closing bar is dated before today (UTC).

Climate indices are refetched from NOAA each run (cached copies in cpe_climate_data/ if the fetch fails); the monthly values are lagged for publication (ONI ~10 days into month m+2,
every other index the 15th of month m+1). Y series missing from multiasset_prices_live_history.parquet (KC=F, RB=F) are kept in cpe_climate_prices.csv.

Usage:
  python cpe_climate_ledger.py              # daily: log new entry events, print summary and watchlist
  python cpe_climate_ledger.py --summary    # scored results only
  python cpe_climate_ledger.py --status     # watchlist: where each condition stands against its threshold
============================================================
"""
import argparse
import hashlib
import os
import re
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(HERE, "cpe_climate_table.parquet")
EVENTS_PATH = os.path.join(HERE, "cpe_climate_events.csv")
DATA_DIR = os.path.join(HERE, "cpe_climate_data")
EXTRA_PATH = os.path.join(HERE, "cpe_climate_prices.csv")
PRICES_PATH = os.path.join(HERE, "multiasset_prices_live_history.parquet")
TABLE_END = pd.Timestamp("2026-10-10")
COND = ["X", "tau_past", "q_X", "direction"]
EVENT_COLS = ["entry_date", "X", "tau_past", "q_X", "direction", "kind", "x_move", "x_threshold", "table_sha"]
URLS = {
    "oni.txt": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "sstoi.txt": "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices",
    "ao.txt": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii",
    "nao.txt": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table",
    "pdo.txt": "https://psl.noaa.gov/data/correlation/pdo.data",
    "soi.txt": "https://www.cpc.ncep.noaa.gov/data/indices/soi",
}
INDEX_FILES = {"ONI": "oni.txt", "NINO34": "sstoi.txt", "AO": "ao.txt", "NAO": "nao.txt", "PDO": "pdo.txt", "SOI": "soi.txt"}


def table_sha():
    with open(TABLE_PATH, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:10]


# ----------------------------------------------------------------------------- climate series
def fetch_indices(names):
    """Refetch the NOAA files needed; keep the cached copy if the fetch fails."""
    import requests
    os.makedirs(DATA_DIR, exist_ok=True)
    for n in names:
        fn = INDEX_FILES[n]
        try:
            r = requests.get(URLS[fn], timeout=40, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and len(r.text) > 2000:
                with open(os.path.join(DATA_DIR, fn), "w") as fh:
                    fh.write(r.text)
        except Exception as e:  # noqa: BLE001
            print(f"  index fetch failed for {fn} ({type(e).__name__}); using cached copy")


def _next15(vals):
    s = pd.Series(vals).dropna()
    idx = [pd.Timestamp(y, m, 1) + pd.DateOffset(months=1) + pd.Timedelta(days=14) for (y, m) in s.index]
    return pd.Series(s.values, index=pd.DatetimeIndex(idx)).sort_index()


def _table_ym(path, missing):
    out = {}
    for line in open(path):
        parts = line.split()
        if len(parts) >= 13 and re.fullmatch(r"\d{4}", parts[0]):
            for m, v in enumerate(parts[1:13], 1):
                try:
                    x = float(v)
                except ValueError:
                    continue
                if abs(x) < missing:
                    out[(int(parts[0]), m)] = x
    return out


def index_series(name):
    p = lambda f: os.path.join(DATA_DIR, f)
    if name == "ONI":
        seas = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
        o = pd.read_csv(p("oni.txt"), sep=r"\s+")
        idx = [pd.Timestamp(int(y), seas[s], 1) + pd.DateOffset(months=2) + pd.Timedelta(days=9) for y, s in zip(o.YR, o.SEAS)]
        return pd.Series(o.ANOM.values, index=pd.DatetimeIndex(idx)).sort_index()
    if name == "NINO34":
        d = pd.read_csv(p("sstoi.txt"), sep=r"\s+")
        d.columns = ["YR", "MON", "N12", "N12A", "N3", "N3A", "N4", "N4A", "N34", "N34A"]
        return _next15({(int(r.YR), int(r.MON)): r.N34A for r in d.itertuples()})
    if name == "AO":
        d = pd.read_csv(p("ao.txt"), sep=r"\s+", header=None)
        return _next15({(int(r[0]), int(r[1])): r[2] for r in d.itertuples(index=False)})
    if name == "NAO":
        return _next15(_table_ym(p("nao.txt"), 99))
    if name == "PDO":
        return _next15(_table_ym(p("pdo.txt"), 9.0))
    if name == "SOI":
        txt = open(p("soi.txt")).read().splitlines()
        start = [i for i, l in enumerate(txt) if "STANDARDIZED" in l][0]
        out = {}
        for l in txt[start:]:
            if re.match(r"^\d{4}", l):
                y = int(l[:4])
                for m in range(12):
                    try:
                        x = float(l[4 + m * 6:4 + (m + 1) * 6].strip())
                    except ValueError:
                        continue
                    if abs(x) < 99:
                        out[(y, m + 1)] = x
        return _next15(out)
    raise KeyError(name)


def climate_series(x, price_index):
    """Daily series for a predictor name like ONI_chg (the level; increment = change over tau) or ONI_mean (cumulative sum; increment = sum over tau days)."""
    idx_name, form = x.rsplit("_", 1)
    s = index_series(idx_name)
    daily = s.reindex(s.index.union(price_index)).ffill().reindex(price_index)
    daily[daily.index < s.index.min()] = np.nan
    return daily if form == "chg" else daily.cumsum()


# ----------------------------------------------------------------------------- prices
def extra_prices(tickers):
    """Y series not in the live history, fetched with yfinance and kept in a small csv."""
    if not tickers:
        return pd.DataFrame()
    old = pd.read_csv(EXTRA_PATH, index_col=0, parse_dates=True) if os.path.exists(EXTRA_PATH) else pd.DataFrame()
    try:
        import yfinance as yf
        d = yf.download(list(tickers), start="2026-08-01", auto_adjust=True, progress=False)["Close"]
        if isinstance(d, pd.Series):
            d = d.to_frame(list(tickers)[0])
        d.index = pd.DatetimeIndex(d.index).tz_localize(None)
        both = pd.concat([old, d])
        both = both[~both.index.duplicated(keep="last")].sort_index()
        both.to_csv(EXTRA_PATH)
        return both
    except Exception as e:  # noqa: BLE001
        print(f"  extra price fetch failed ({type(e).__name__}: {str(e)[:80]}); using the stored file")
        return old


def load_inputs(today_utc=None, asof=None):
    today_utc = today_utc or datetime.now(timezone.utc).date()
    table = pd.read_parquet(TABLE_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    need_y = sorted(set(table["Y"]))
    missing = [y for y in need_y if y not in prices.columns]
    ex = extra_prices(missing)
    for y in missing:
        if y in ex.columns:
            prices[y] = ex[y].reindex(prices.index.union(ex.index)).reindex(prices.index)
    prices = prices[prices.index.date < today_utc]
    if asof is not None:
        prices = prices[prices.index <= pd.Timestamp(asof)]
    return prices, table


def x_increment(x, tau, price_index):
    s = climate_series(x, price_index)
    return s - s.shift(tau)


# ----------------------------------------------------------------------------- events
def find_entry_events(prices, table, after, include_ongoing_at=None):
    conds = table[COND + ["x_threshold"]].drop_duplicates(COND)
    sha = table_sha()
    rows = []
    for r in conds.itertuples():
        inc = x_increment(r.X, int(r.tau_past), prices.index)
        thr = float(r.x_threshold)
        fire = (inc > thr) if r.direction == "bullish" else (inc < thr)
        fire = fire.fillna(False)
        onset = fire & ~fire.shift(1, fill_value=False)
        for dt in onset[(onset) & (onset.index > after)].index:
            rows.append((dt.strftime("%Y-%m-%d"), r.X, int(r.tau_past), float(r.q_X), r.direction, "onset", float(inc.loc[dt]), thr, sha))
        if include_ongoing_at is not None and include_ongoing_at in fire.index and bool(fire.loc[include_ongoing_at]):
            rows.append((include_ongoing_at.strftime("%Y-%m-%d"), r.X, int(r.tau_past), float(r.q_X), r.direction, "ongoing_at_freeze",
                         float(inc.loc[include_ongoing_at]), thr, sha))
    return pd.DataFrame(rows, columns=EVENT_COLS)


def load_events():
    if not os.path.exists(EVENTS_PATH):
        return pd.DataFrame(columns=EVENT_COLS)
    return pd.read_csv(EVENTS_PATH, dtype={"entry_date": str})


def log_new_events():
    prices, table = load_inputs()
    ev = load_events()
    first_run = len(ev) == 0
    last_logged = max(pd.Timestamp(ev["entry_date"].max()), TABLE_END) if len(ev) else TABLE_END
    new = find_entry_events(prices, table, after=last_logged, include_ongoing_at=TABLE_END if first_run else None)
    if len(new):
        ev = pd.concat([ev, new], ignore_index=True).drop_duplicates(COND + ["entry_date"])
        ev = ev.sort_values(["entry_date", "X", "tau_past", "q_X", "direction"])
        ev.to_csv(EVENTS_PATH, index=False)
    print(f"CPE climate ledger: {len(new)} new entry event(s) (after {last_logged.date()}; through {prices.index.max().date()}); {len(ev)} total")
    return ev


# ----------------------------------------------------------------------------- scoring
def score(ev=None, asof=None):
    prices, table = load_inputs(asof=asof)
    ev = load_events() if ev is None else ev
    if asof is not None:
        ev = ev[pd.to_datetime(ev["entry_date"]) <= pd.Timestamp(asof)]
    if ev.empty:
        return pd.DataFrame(), prices.index.max()
    ev = ev.copy()
    ev["entry_date"] = pd.to_datetime(ev["entry_date"])
    rows = ev[["entry_date", "kind"] + COND].merge(table, on=COND, how="inner")
    idx = prices.index
    pos_of = pd.Series(np.arange(len(idx)), index=idx)
    rows["pos"] = pos_of.reindex(rows["entry_date"]).values
    rows["resolve_pos"] = rows["pos"] + rows["tau_future"]
    rows["resolved"] = rows["resolve_pos"] <= len(idx) - 1
    rows["event"], rows["live_base"] = np.nan, np.nan
    rows["outcome_date"] = pd.NaT
    live_start = pos_of[idx > TABLE_END].iloc[0] if (idx > TABLE_END).any() else len(idx)
    for (y, tf), g in rows.groupby(["Y", "tau_future"]):
        if y not in prices.columns:
            continue
        s = prices[y]
        fwd = np.log(s.shift(-int(tf)) / s).values
        for (qy, d, thr), gg in g.groupby(["q_Y", "direction", "y_threshold"]):
            r = gg[gg["resolved"]]
            if r.empty:
                continue
            f = fwd[r["pos"].astype(int).values]
            flag = (f > thr) if d == "bullish" else (f < thr)
            flag = np.where(np.isnan(f), np.nan, flag.astype(float))
            rows.loc[r.index, "event"] = flag
            w = fwd[live_start:len(idx) - int(tf)]
            w = w[~np.isnan(w)]
            rows.loc[r.index, "live_base"] = float(((w > thr) if d == "bullish" else (w < thr)).mean()) if len(w) else np.nan
            rows.loc[r.index, "outcome_date"] = idx[r["resolve_pos"].astype(int).values]
    rows.loc[rows["event"].isna(), "resolved"] = False
    return rows, prices.index.max()


def live_stats(rows=None, last_bar=None, ev=None):
    if rows is None:
        rows, last_bar = score(ev)
    ev = load_events() if ev is None else ev
    out = {"table_end": str(TABLE_END.date()), "table_sha": table_sha(), "n_table_rows": int(len(pd.read_parquet(TABLE_PATH))),
           "n_events": int(len(ev)), "n_rows": int(len(rows)), "last_bar": str(last_bar.date()) if last_bar is not None else None,
           "n_resolved": 0, "n_pending": 0, "by_horizon": []}
    if rows.empty:
        return out
    res = rows[rows["resolved"]]
    out["n_resolved"], out["n_pending"] = int(len(res)), int(len(rows) - len(res))
    for tf, g in rows.groupby("tau_future"):
        r = res[res["tau_future"] == tf]
        row = {"horizon": int(tf), "n_pending": int(len(g) - len(r)), "n": int(len(r))}
        pend = g[~g["resolved"]]
        if len(pend):
            est = pend["entry_date"] + pd.to_timedelta((pend["tau_future"] * 1.4).round().astype(int), unit="D")
            row["first_pending"] = str(est.min().date())
        if len(r):
            row.update({"claimed": round(float(r["CPE"].mean()), 4), "hit": round(float(r["event"].mean()), 4), "base": round(float(r["uncond_prob"].mean()), 4),
                        "live_base": round(float(r["live_base"].mean()), 4), "events": int(r.groupby(["entry_date"] + COND).ngroups), "targets": int(r["Y"].nunique())})
        out["by_horizon"].append(row)
    return out


def status():
    """Watchlist: each distinct condition's current value against its frozen threshold."""
    prices, table = load_inputs()
    conds = table[COND + ["x_threshold"]].drop_duplicates(COND)
    print(f"\nWatchlist as of {prices.index.max().date()} (state of the climate series against the frozen thresholds)")
    print(f"{'condition':<44}{'now':>9}{'threshold':>11}{'gap':>9}  state")
    for r in conds.itertuples():
        inc = x_increment(r.X, int(r.tau_past), prices.index)
        v = float(inc.dropna().iloc[-1])
        thr = float(r.x_threshold)
        firing = (v > thr) if r.direction == "bullish" else (v < thr)
        gap = (thr - v) if r.direction == "bullish" else (v - thr)   # amount the state still has to move to fire
        tl = "bullish: tail above" if r.direction == "bullish" else "bearish: tail below"
        ys = sorted(set(table[(table.X == r.X) & (table.tau_past == r.tau_past) & (table.q_X == r.q_X) & (table.direction == r.direction)]["Y"]))
        print(f"{r.X + ' tau=' + str(int(r.tau_past)) + ' q=' + str(r.q_X) + ' ' + tl:<44}{v:>9.2f}{thr:>11.2f}{(0 if firing else gap):>9.2f}  {'FIRING' if firing else 'waiting'}  -> {', '.join(ys)}")


def summary():
    rows, last_bar = score()
    s = live_stats(rows, last_bar)
    print(f"\nTable: {s['n_table_rows']} frozen rows (sha {s['table_sha']}, frozen {s['table_end']}); entry events {s['n_events']}, scored rows {s['n_rows']}, "
          f"resolved {s['n_resolved']}, pending {s['n_pending']}; last bar {s['last_bar']}.")
    for r in s["by_horizon"]:
        if r["n"]:
            print(f"  tau_future={r['horizon']:>3}: n={r['n']:>4}  claimed {r['claimed']:.3f}  realised {r['hit']:.3f}  table base {r['base']:.3f}  live base {r['live_base']:.3f}  pending {r['n_pending']}")
        else:
            print(f"  tau_future={r['horizon']:>3}: no outcomes yet, {r['n_pending']} pending (first about {r.get('first_pending')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    table = pd.read_parquet(TABLE_PATH)
    fetch_indices(sorted({x.rsplit("_", 1)[0] for x in table["X"]}))
    if not (args.summary or args.status):
        log_new_events()
    summary()
    status()


if __name__ == "__main__":
    sys.exit(main())
