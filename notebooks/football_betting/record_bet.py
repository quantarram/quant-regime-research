"""
Marks a logged pick in qualifying_log.csv as an ACTUAL bet placed, with the
real stake used -- distinct from "qualifies", which just means the dashboard
flagged it. Not every qualifying pick necessarily gets a real bet behind it,
so "real hit rate / real ROI" on the dashboard is computed over bet_placed
rows only, not every qualifying signal, to keep the track record honest
about what money was actually on the line.

Usage:
    python record_bet.py "Barcelona vs Athletic Bilbao" 10
    python record_bet.py "Barcelona" 10                    # fuzzy match on part of the fixture name
    python record_bet.py "Barcelona" 100 1.15 --add --placed "2026-11-19 18:54"
                                                             # ADD a further slip to an existing position:
                                                             # totals and blended odds are worked out here,
                                                             # --placed is the slip's time in SGT (default now)
    python record_bet.py "Barcelona" 10 1.17               # optional: the ACTUAL odds you got,
                                                             # if different from the logged snapshot
                                                             # (odds move between dashboard build and
                                                             # bet placement -- this is what your real
                                                             # payout is actually based on)
"""
import sys
from pathlib import Path
from difflib import get_close_matches

import pandas as pd

OUT_DIR = Path(__file__).parent / "output"
LOG_PATH = OUT_DIR / "qualifying_log.csv"


def _append_slip(fixture, kickoff, placed_sgt, stake, odds):
    """One row per real bet slip (bet_slips.csv) so the odds chart can show each price actually obtained."""
    path = OUT_DIR / "bet_slips.csv"
    row = pd.DataFrame([{"fixture": fixture, "kickoff_utc": kickoff,
                         "placed_utc": placed_sgt.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "stake_sgd": stake, "odds": odds}])
    if path.exists():
        row = pd.concat([pd.read_csv(path), row], ignore_index=True)
    row.to_csv(path, index=False)


def main():
    args = sys.argv[1:]
    add = "--add" in args
    placed = pd.Timestamp.now(tz="Asia/Singapore")
    if "--placed" in args:
        i = args.index("--placed")
        placed = pd.Timestamp(args[i + 1]).tz_localize("Asia/Singapore")
        del args[i:i + 2]
    args = [a for a in args if a != "--add"]
    sys.argv = [sys.argv[0]] + args
    if len(sys.argv) < 3:
        print('Usage: python record_bet.py "<fixture or team name>" <stake_sgd>')
        sys.exit(1)
    query = sys.argv[1]
    stake = float(sys.argv[2])
    actual_odds = float(sys.argv[3]) if len(sys.argv) > 3 else None

    if not LOG_PATH.exists():
        print("No qualifying_log.csv found.")
        sys.exit(1)

    log = pd.read_csv(LOG_PATH)
    candidates = log["fixture"].tolist()
    match = query if query in candidates else None
    if match is None:
        hits = get_close_matches(query, candidates, n=1, cutoff=0.3)
        # also try substring match (e.g. "Barcelona" inside "Barcelona vs Athletic Bilbao")
        substr_hits = [c for c in candidates if query.lower() in c.lower()]
        if substr_hits:
            match = substr_hits[-1]  # most recently logged if multiple
        elif hits:
            match = hits[0]

    if match is None:
        print(f'No fixture matching "{query}" found in the log. Candidates:')
        print(log["fixture"].tail(10).to_string(index=False))
        sys.exit(1)

    idx = log[log["fixture"] == match].index[-1]  # most recent if logged more than once
    if "actual_odds" not in log.columns:
        log["actual_odds"] = pd.NA
    slip_odds = actual_odds if actual_odds is not None else float(log.loc[idx, "odds"])
    _append_slip(match, log.loc[idx, "start_time"], placed, stake, slip_odds)
    if add and bool(log.loc[idx, "bet_placed"]) and pd.notna(log.loc[idx, "actual_stake_sgd"]):
        prev_stake = float(log.loc[idx, "actual_stake_sgd"])
        prev_odds = float(log.loc[idx, "actual_odds"])
        total = prev_stake + stake
        price = (prev_stake * prev_odds + stake * slip_odds) / total   # blended = total payout / total stake
        stake = total
    else:
        price = slip_odds
    log.loc[idx, "bet_placed"] = True
    log.loc[idx, "actual_stake_sgd"] = stake
    log.loc[idx, "actual_odds"] = price
    log.to_csv(LOG_PATH, index=False)

    row = log.loc[idx]
    odds_note = f" (dashboard showed {row['odds']:.2f})" if actual_odds is not None and abs(actual_odds - row["odds"]) > 1e-6 else ""
    print(f"Recorded: S${stake:.2f} on {row['pick']} ({row['fixture']}) @ {price:.2f}{odds_note}, "
          f"kickoff {row['start_time']}, status={row['status']}")
    if row["status"] == "RESOLVED":
        actual_pnl = stake * (price - 1) if bool(row["won"]) else -stake
        log.loc[idx, "actual_pnl"] = actual_pnl
        log.to_csv(LOG_PATH, index=False)
        print(f"Already resolved: {'WON' if row['won'] else 'LOST'}, actual P&L = S${actual_pnl:+.2f}")


if __name__ == "__main__":
    main()
