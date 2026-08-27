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


def main():
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
    log.loc[idx, "bet_placed"] = True
    log.loc[idx, "actual_stake_sgd"] = stake
    if "actual_odds" not in log.columns:
        log["actual_odds"] = pd.NA
    price = actual_odds if actual_odds is not None else log.loc[idx, "odds"]
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
