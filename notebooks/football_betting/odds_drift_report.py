"""
Odds drift report -- do home-win odds get better or worse as kickoff approaches?
================================================================================
Reads output/odds_history.csv (one reading per fixture per day from the daily pipeline run,
see odds_snapshot.py) and, for fixtures that have kicked off and whose last daily reading is
within CLOSE_WITHIN_H hours of kickoff (the nearest-to-close price a once-a-day log can give),
compares the home-win odds seen at each hours-to-kickoff bucket with that last reading:

    ratio = odds_h(at snapshot) / odds_h(closing proxy)

ratio > 1  -> betting at that point beat the closing price (higher decimal odds pay more)
ratio < 1  -> waiting until close would have paid more

Everything is raw levels with n shown; nothing is filtered after the fact except the
completeness rule above, which is counted and printed. Two views: all fixtures, and
short-priced favourites (closing odds <= 1.65, the range the qualifying picks live in).
Also lists the bets actually placed against the latest and closing odds.

Run: python football_betting/odds_drift_report.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "output"
CLOSE_WITHIN_H = 36.0
BUCKETS = [(72, 1e9, ">72h"), (48, 72, "48-72h"), (24, 48, "24-48h"), (0, 24, "<24h")]


def bucket_of(h):
    for lo, hi, name in BUCKETS:
        if lo <= h < hi:
            return name
    return None


def main():
    path = OUT / "odds_history.csv"
    if not path.exists():
        print("No odds_history.csv yet -- run odds_snapshot.py first.")
        return
    h = pd.read_csv(path, parse_dates=["snapshot_utc"])
    h["start_time"] = pd.to_datetime(h["start_time"], utc=True)
    h["snapshot_utc"] = h["snapshot_utc"].dt.tz_convert("UTC") if h["snapshot_utc"].dt.tz else h["snapshot_utc"].dt.tz_localize("UTC")
    now = pd.Timestamp.now(tz="UTC")

    n_snap = h["snapshot_utc"].nunique()
    print(f"Snapshots: {n_snap} | first {h['snapshot_utc'].min():%Y-%m-%d %H:%M} UTC | last {h['snapshot_utc'].max():%Y-%m-%d %H:%M} UTC")
    fx = h.groupby(["fixture", "start_time"])
    n_fix = fx.ngroups
    print(f"Fixtures tracked: {n_fix} | with >=2 snapshots: {int((fx.size() >= 2).sum())}")

    rows = []
    n_done = n_complete = 0
    for (fixture, start), g in fx:
        if start > now:
            continue
        n_done += 1
        g = g.sort_values("snapshot_utc")
        last = g.iloc[-1]
        if last["hours_to_kickoff"] > CLOSE_WITHIN_H or pd.isna(last["odds_h"]):
            continue
        n_complete += 1
        close = last["odds_h"]
        for _, r in g.iterrows():
            b = bucket_of(r["hours_to_kickoff"])
            if b and pd.notna(r["odds_h"]):
                rows.append(dict(fixture=fixture, bucket=b, ratio=r["odds_h"] / close, close=close))
    print(f"Kicked off: {n_done} | with a last daily reading <= {CLOSE_WITHIN_H:.0f}h before kickoff: {n_complete}")

    if rows:
        d = pd.DataFrame(rows)
        order = [b[2] for b in BUCKETS]
        for label, sub in [("ALL fixtures", d), ("Short-priced favourites (last reading <= 1.65)", d[d.close <= 1.65])]:
            print(f"\n{label}: odds at bucket / last daily reading  (>1 = earlier odds were better)")
            t = sub.groupby("bucket").agg(n=("ratio", "size"), fixtures=("fixture", "nunique"),
                                          median=("ratio", "median"), mean=("ratio", "mean"),
                                          share_better=("ratio", lambda x: float((x > 1.0005).mean())),
                                          share_worse=("ratio", lambda x: float((x < 0.9995).mean())))
            print(t.reindex([o for o in order if o in t.index]).round(4).to_string())
    else:
        print("\nNo completed fixtures yet -- the table fills in once tracked fixtures kick off.")

    # bets actually placed vs the odds seen since
    log_path = OUT / "qualifying_log.csv"
    if log_path.exists():
        q = pd.read_csv(log_path)
        q = q[q["bet_placed"].astype(str).str.lower() == "true"]
        if len(q):
            print("\nBets placed vs odds seen in snapshots:")
            for _, b in q.iterrows():
                g = h[(h.fixture == b["fixture"]) & (h.start_time == pd.Timestamp(b["start_time"]))].sort_values("snapshot_utc")
                if g.empty:
                    continue
                first, last = g.iloc[0], g.iloc[-1]
                print(f"  {b['fixture']}: got {b['actual_odds']} | first snapshot {first['odds_h']} ({first['hours_to_kickoff']:.0f}h out) "
                      f"| latest {last['odds_h']} ({last['hours_to_kickoff']:.0f}h out) | {len(g)} snapshots")


if __name__ == "__main__":
    main()
