#!/usr/bin/env python3
"""
============================================================
  CPE LIVE SIGNAL LEDGER
  Dr. Arun Ramanathan
============================================================
The CPE dashboard's claim is conditional: when predictor X is in its tail over
the past tau_past days, the probability that target Y exceeds its own tail
threshold over the next tau_future days is CPE (table: cpe_results.parquet,
already gated at CPE >= 0.80, lift >= 1.5, n >= 100). Until now nothing scored
that claim on data the table never saw.

The table was estimated on prices through TABLE_END (the file was saved
2026-06-05). Everything after that date is out-of-sample for it, and this
ledger records and scores exactly that period.

What is stored
--------------
Roughly 13,000 gated signal rows are "firing" on any given day, so logging one
row per signal per day would put millions of rows a year into git. The scoring
of a signal is, however, a deterministic function of three things that are
already fixed: the frozen CPE table, the dated price history, and the date X
entered its tail. So the ledger stores only the entry events:

    cpe_signal_events.csv   one row per (entry_date, X, tau_past, q_X, direction):
                            the first trading day on which X's tau_past move crossed
                            its tail threshold after a day on which it had not.
                            Thresholds are frozen from data through TABLE_END, exactly as
                            in the table. table_sha records which table the events were
                            logged against.

Each event fans out to every gated table row for that predictor condition
(every Y, tau_future and q_Y the table claims something about). Scoring uses
the engine's own definitions (cpe_engine_parallel.py): the event is that Y's
forward tau_future log move (difference, for rate/vol indices) lies beyond the
frozen q_Y quantile of Y's tau_future moves (above for bullish, below for
bearish). Horizons are in trading days on the price index; an outcome counts
once its closing bar is dated before today (UTC).

Usage:
  python cpe_signal_ledger.py              # daily: log new entry events
  python cpe_signal_ledger.py --summary    # print scored results
============================================================
"""

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # log of non-positive prices in a few series -> NaN, as in the engine

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(HERE, "cpe_results.parquet")
PRICES_PATH = os.path.join(HERE, "multiasset_prices_live_history.parquet")
EVENTS_PATH = os.path.join(HERE, "cpe_signal_events.csv")

TABLE_END = pd.Timestamp("2026-06-05")   # last date the frozen CPE table could have seen
RATE_TICKERS = {"^VIX", "^VXN", "^OVX", "^GVZ", "^EVZ", "^VVIX", "^SKEW", "^TNX", "^TYX", "^FVX", "^IRX"}
COND = ["X", "tau_past", "q_X", "direction"]
EVENT_COLS = ["entry_date", "X", "tau_past", "q_X", "direction", "x_move", "x_threshold", "table_sha"]


def table_sha():
    with open(TABLE_PATH, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:10]


def load_inputs(today_utc=None, asof=None):
    """Price history restricted to fully formed bars (dated before today, UTC); with `asof`, only bars through that date."""
    today_utc = today_utc or datetime.now(timezone.utc).date()
    prices = pd.read_parquet(PRICES_PATH)
    prices = prices[prices.index.date < today_utc]
    if asof is not None:
        prices = prices[prices.index <= pd.Timestamp(asof)]
    table = pd.read_parquet(TABLE_PATH)
    return prices, table


def increment(prices, ticker, tau):
    s = prices[ticker]
    return (s - s.shift(tau)) if ticker in RATE_TICKERS else np.log(s / s.shift(tau))


def find_entry_events(prices, table, after):
    """Entry events (first day in tail) for every predictor condition in the table, dated after `after`."""
    combos = table[COND].drop_duplicates()
    sha = table_sha()
    inc_cache, rows = {}, []
    for (x, tp), grp in combos.groupby(["X", "tau_past"]):
        if x not in prices.columns:
            continue
        inc = increment(prices, x, int(tp))
        frozen = inc[inc.index <= TABLE_END].dropna()
        if len(frozen) < 100:
            continue
        for q, d in zip(grp["q_X"], grp["direction"]):
            if d == "bullish":
                thr = frozen.quantile(float(q))
                fire = inc > thr
            else:
                thr = frozen.quantile(1.0 - float(q))
                fire = inc < thr
            onset = fire & ~fire.shift(1, fill_value=False)
            hit = onset[(onset) & (onset.index > after)].index
            for dt in hit:
                rows.append((dt.strftime("%Y-%m-%d"), x, int(tp), float(q), d, float(inc.loc[dt]), float(thr), sha))
    return pd.DataFrame(rows, columns=EVENT_COLS)


def load_events():
    if not os.path.exists(EVENTS_PATH):
        return pd.DataFrame(columns=EVENT_COLS)
    return pd.read_csv(EVENTS_PATH, dtype={"entry_date": str})


def log_new_events():
    prices, table = load_inputs()
    ev = load_events()
    last_logged = pd.Timestamp(ev["entry_date"].max()) if len(ev) else TABLE_END
    if len(ev) and ev["table_sha"].nunique() > 1:
        print("WARNING: events were logged against more than one CPE table version")
    if len(ev) and ev["table_sha"].iloc[-1] != table_sha():
        print("WARNING: cpe_results.parquet has changed since events were logged; claimed CPEs below use the new table")
    new = find_entry_events(prices, table, after=last_logged)
    if len(new):
        ev = pd.concat([ev, new], ignore_index=True).drop_duplicates(COND + ["entry_date"])
        ev = ev.sort_values(["entry_date", "X", "tau_past", "q_X", "direction"])
        ev.to_csv(EVENTS_PATH, index=False)
    print(f"CPE signal ledger: {len(new)} new entry event(s) "
          f"({'after ' + str(last_logged.date())}; through {prices.index.max().date()}); {len(ev)} total")
    return ev


def score(ev=None, asof=None):
    """Fan entry events out to the table's rows and score resolved ones. Returns (rows_df, last_bar_date).
    With `asof`, scores as the ledger stood on that date: only events entered and bars published through it."""
    prices, table = load_inputs(asof=asof)
    ev = load_events() if ev is None else ev
    if asof is not None:
        ev = ev[pd.to_datetime(ev["entry_date"]) <= pd.Timestamp(asof)]
    if ev.empty:
        return pd.DataFrame(), prices.index.max()
    ev = ev.copy()
    ev["entry_date"] = pd.to_datetime(ev["entry_date"])
    rows = ev[["entry_date"] + COND].merge(table, on=COND, how="inner")
    idx = prices.index
    pos_of = pd.Series(np.arange(len(idx)), index=idx)
    rows["pos"] = pos_of.reindex(rows["entry_date"]).values
    rows["resolve_pos"] = rows["pos"] + rows["tau_future"]
    rows["resolved"] = rows["resolve_pos"] <= len(idx) - 1
    rows["event"] = np.nan
    rows["live_base"] = np.nan   # how often Y's event happened on ANY day of the live window (the live control)
    rows["outcome_date"] = pd.NaT
    live_start = pos_of[idx > TABLE_END].iloc[0]

    for (y, tf), g in rows.groupby(["Y", "tau_future"]):
        if y not in prices.columns:
            continue
        inc = increment(prices, y, int(tf))
        frozen = inc[inc.index <= TABLE_END].dropna()
        if len(frozen) < 100:
            continue
        fwd = inc.shift(-int(tf)).values  # forward move from each bar
        for (qy, d), gg in g.groupby(["q_Y", "direction"]):
            r = gg[gg["resolved"]]
            if r.empty:
                continue
            f = fwd[r["pos"].astype(int).values]
            if d == "bullish":
                ev_flag = f > frozen.quantile(float(qy))
            else:
                ev_flag = f < frozen.quantile(1.0 - float(qy))
            ev_flag = np.where(np.isnan(f), np.nan, ev_flag.astype(float))
            rows.loc[r.index, "event"] = ev_flag
            # same event, evaluated on every bar of the live window that has resolved
            w = fwd[live_start:len(idx) - int(tf)]
            w = w[~np.isnan(w)]
            thr = frozen.quantile(float(qy)) if d == "bullish" else frozen.quantile(1.0 - float(qy))
            rows.loc[r.index, "live_base"] = float((w > thr).mean() if d == "bullish" else (w < thr).mean()) if len(w) else np.nan
            rows.loc[r.index, "outcome_date"] = idx[r["resolve_pos"].astype(int).values]
    rows.loc[rows["event"].isna(), "resolved"] = False
    return rows, prices.index.max()


def live_stats(rows=None, last_bar=None, ev=None, asof=None):
    """JSON-safe summary for dashboards."""
    if rows is None:
        rows, last_bar = score(ev, asof)
    ev = load_events() if ev is None else ev
    if asof is not None:
        ev = ev[pd.to_datetime(ev["entry_date"]) <= pd.Timestamp(asof)]
    out = {"table_end": str(TABLE_END.date()), "table_sha": table_sha(),
           "n_events": int(len(ev)), "n_rows": int(len(rows)),
           "first_entry": str(ev["entry_date"].min()) if len(ev) else None,
           "last_entry": str(ev["entry_date"].max()) if len(ev) else None,
           "last_bar": str(last_bar.date()) if last_bar is not None else None,
           "n_resolved": 0, "n_pending": 0, "by_horizon": [], "calibration": []}
    if rows.empty:
        return out
    res = rows[rows["resolved"]]
    out["n_resolved"], out["n_pending"] = int(len(res)), int(len(rows) - len(res))
    for tf, g in rows.groupby("tau_future"):
        r = res[res["tau_future"] == tf]
        row = {"horizon": int(tf), "n_pending": int(len(g) - len(r)), "n": int(len(r)),
               "first_pending": None}
        pend = g[~g["resolved"]]
        if len(pend):
            row["first_pending"] = str(idx_date(pend, prices_index_len=None))
        if len(r):
            row.update({"claimed": round(float(r["CPE"].mean()), 4),
                        "hit": round(float(r["event"].mean()), 4),
                        "base": round(float(r["uncond_prob"].mean()), 4),
                        "live_base": round(float(r["live_base"].mean()), 4),
                        "events": int(r.groupby(["entry_date"] + COND).ngroups),
                        "targets": int(r["Y"].nunique())})
            row["lift"] = round(row["hit"] / row["base"], 3) if row["base"] else None
            for d in ("bullish", "bearish"):
                rd = r[r["direction"] == d]
                row[d] = ({"n": int(len(rd)), "claimed": round(float(rd["CPE"].mean()), 4),
                           "hit": round(float(rd["event"].mean()), 4),
                           "base": round(float(rd["uncond_prob"].mean()), 4),
                           "live_base": round(float(rd["live_base"].mean()), 4)} if len(rd) else None)
        out["by_horizon"].append(row)
    bins = [0.80, 0.85, 0.90, 0.95, 1.0001]
    labels = ["0.80-0.85", "0.85-0.90", "0.90-0.95", "0.95-1.00"]
    if len(res):
        b = pd.cut(res["CPE"], bins=bins, labels=labels, right=False)
        for lab in labels:
            r = res[b == lab]
            if len(r):
                out["calibration"].append({"bin": lab, "n": int(len(r)),
                                           "claimed": round(float(r["CPE"].mean()), 4),
                                           "hit": round(float(r["event"].mean()), 4),
                                           "base": round(float(r["uncond_prob"].mean()), 4),
                                           "live_base": round(float(r["live_base"].mean()), 4)})
    return out


def idx_date(pend, prices_index_len=None):
    """Earliest outcome bar still to come for pending rows (approx. calendar date: entry + horizon * 1.4 days)."""
    est = pend["entry_date"] + pd.to_timedelta((pend["tau_future"] * 1.4).round().astype(int), unit="D")
    return est.min().date()


def summary():
    rows, last_bar = score()
    s = live_stats(rows, last_bar)
    print(f"\nEntry events: {s['n_events']} ({s['first_entry']} to {s['last_entry']}), "
          f"expanding to {s['n_rows']:,} scored signal rows; {s['n_resolved']:,} resolved, {s['n_pending']:,} pending. "
          f"Table frozen at {s['table_end']} (sha {s['table_sha']}); last bar {s['last_bar']}.")
    for r in s["by_horizon"]:
        if r["n"]:
            print(f"  tau_future={r['horizon']:>3}: n={r['n']:>6}  claimed CPE {r['claimed']:.3f}  "
                  f"realised {r['hit']:.3f}  table base {r['base']:.3f}  live base {r['live_base']:.3f}  pending {r['n_pending']}")
        else:
            print(f"  tau_future={r['horizon']:>3}: no outcomes yet, {r['n_pending']:,} pending (first ~{r['first_pending']})")
    if s["calibration"]:
        print("  by claimed-CPE bin:", [(c["bin"], c["n"], c["claimed"], c["hit"]) for c in s["calibration"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if not args.summary:
        log_new_events()
    summary()


if __name__ == "__main__":
    sys.exit(main())
