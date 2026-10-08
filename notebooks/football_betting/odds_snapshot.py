"""
Daily odds log -- what did Singapore Pools quote for each fixture on each daily run?
====================================================================================
Called by daily_dashboard.py on every daily pipeline run (nothing runs more often than
that). Appends one row per fixture on the board in a validated league that kicks off
within the next 10 days -- home/draw/away decimal odds and hours to kickoff -- to
output/odds_history.csv. At most ONE reading per fixture per SGT calendar day is kept
(the first of the day), so the history is a day-by-day series, never intraday movement.

odds_chart.py plots it on the football dashboard; odds_drift_report.py summarises it.
Read-only against the odds feed; never places or touches a bet.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

OUT = Path(__file__).parent / "output" / "odds_history.csv"
COLS = ["snapshot_utc", "fixture", "league", "start_time", "hours_to_kickoff",
        "odds_h", "odds_d", "odds_a"]
HORIZON_DAYS = 10
SGT = ZoneInfo("Asia/Singapore")


def _price(outcomes, code):
    o = outcomes.get(code)
    if not o:
        return None
    try:
        return float(o["prices"][0]["decimal"])
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def snapshot_rows(events, now_utc, competition_map):
    rows = []
    for ev in events:
        league = competition_map.get((ev["type"]["sportClass"]["name"], ev["type"]["name"]))
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


def append_history(snap, path=OUT):
    """Append a snapshot, keeping only the first reading per fixture per SGT day. Returns the
    full history frame."""
    path = Path(path)
    hist = pd.concat([pd.read_csv(path), snap], ignore_index=True) if path.exists() else snap.copy()
    if hist.empty:
        return hist
    hist = hist.sort_values("snapshot_utc", kind="stable")
    day = pd.to_datetime(hist["snapshot_utc"], utc=True).dt.tz_convert(SGT).dt.date
    hist = hist[~pd.DataFrame({"f": hist["fixture"].values, "s": hist["start_time"].values, "d": day.values})
                .duplicated(keep="first").values]
    hist.to_csv(path, index=False)
    return hist


def main():
    """Manual one-off: fetch the board and append today's reading (the daily run does this itself)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from daily_dashboard import COMPETITION_MAP, fetch_upcoming
    now = datetime.now(timezone.utc)
    events = fetch_upcoming()
    snap = snapshot_rows(events, now, COMPETITION_MAP)
    hist = append_history(snap)
    print(f"{len(snap)} fixtures in validated leagues within {HORIZON_DAYS} days; "
          f"history now {len(hist)} rows over {hist['snapshot_utc'].str[:10].nunique()} day(s)")


if __name__ == "__main__":
    main()
