"""
Odds snapshot tracker -- how do Singapore Pools' 1X2 odds move as kickoff approaches?
=====================================================================================
The daily pipeline sees each fixture's odds at one moment a day, which can't answer
"would I have got better odds betting closer to kickoff?". This script is run every few
hours by its own workflow; it appends one row per (snapshot, fixture) for every fixture
on the board in a validated league that kicks off within the next 10 days, with the
home/draw/away decimal odds and the hours remaining to kickoff, to
output/odds_history.csv. Every snapshot carries its own timestamp, so irregular run
times (GitHub's scheduler is often hours late) don't matter -- the analysis is by
hours-to-kickoff, not by clock time.

Read-only against the odds feed; it never places or touches a bet. The last snapshot
before kickoff is the best available proxy for closing odds.

Run: python football_betting/odds_snapshot.py
Report: python football_betting/odds_drift_report.py
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from daily_dashboard import COMPETITION_MAP, fetch_upcoming  # noqa: E402

OUT = Path(__file__).parent / "output" / "odds_history.csv"
COLS = ["snapshot_utc", "fixture", "league", "start_time", "hours_to_kickoff",
        "odds_h", "odds_d", "odds_a"]
HORIZON_DAYS = 10


def _price(outcomes, code):
    o = outcomes.get(code)
    if not o:
        return None
    try:
        return float(o["prices"][0]["decimal"])
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def snapshot_rows(events, now_utc):
    rows = []
    for ev in events:
        league = COMPETITION_MAP.get((ev["type"]["sportClass"]["name"], ev["type"]["name"]))
        if league is None or not ev.get("startTime"):
            continue
        start = datetime.fromisoformat(ev["startTime"].replace("Z", "+00:00"))
        hours = (start - now_utc).total_seconds() / 3600.0
        if hours <= 0 or hours > HORIZON_DAYS * 24:
            continue
        mkt = next((m for m in ev.get("markets", []) if m.get("name") == "1X2"), None)
        if mkt is None:
            continue
        outcomes = {o["minorCode"]: o for o in mkt.get("outcomes", [])}
        if "H" not in outcomes:
            continue
        rows.append(dict(
            snapshot_utc=now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            fixture=f"{outcomes['H']['name']} vs {outcomes.get('A', {}).get('name', '?')}",
            league=league, start_time=ev["startTime"], hours_to_kickoff=round(hours, 2),
            odds_h=_price(outcomes, "H"), odds_d=_price(outcomes, "D"), odds_a=_price(outcomes, "A"),
        ))
    return pd.DataFrame(rows, columns=COLS)


def main():
    now = datetime.now(timezone.utc)
    events = fetch_upcoming()
    snap = snapshot_rows(events, now)
    print(f"{len(events)} events on the board; {len(snap)} fixtures in validated leagues within {HORIZON_DAYS} days")
    if snap.empty:
        print("Nothing to record.")
        return
    if OUT.exists():
        hist = pd.read_csv(OUT)
        hist = pd.concat([hist, snap], ignore_index=True)
        hist = hist.drop_duplicates(subset=["snapshot_utc", "fixture", "start_time"], keep="first")
    else:
        hist = snap
    hist.to_csv(OUT, index=False)
    print(f"Saved -> {OUT} ({len(hist)} rows, {hist['snapshot_utc'].nunique()} snapshots)")


if __name__ == "__main__":
    main()
