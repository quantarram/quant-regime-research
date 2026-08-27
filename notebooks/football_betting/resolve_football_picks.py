"""
Resolves PENDING rows in output/qualifying_log.csv against real match
results, same pattern as log_predictions.py's resolve_gold()/resolve_metals()/
resolve_portfolio(): a pick logged with a live odds price sits PENDING until
its match has actually been played, then gets checked against what really
happened and locked in as RESOLVED, building a running track record of the
CPE rule's real, not backtested, performance.

Result source: matches.parquet itself (refreshed daily by daily_dashboard.py
via data_download.refresh_core_leagues() before every run) -- once
football-data.co.uk has posted a fixture's final score, it shows up there
under the exact matched team names already stored in qualifying_log.csv
(home_team_matched/away_team_matched), so no separate live-score API is
needed. A pick's match not yet appearing in matches.parquet just means the
source hasn't posted it yet -- it stays PENDING and gets checked again next
run, it is never marked wrong by default.

Usage:
    python resolve_football_picks.py
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
LOG_PATH = OUT_DIR / "qualifying_log.csv"
STAKE_SGD = 50.0


def main():
    if not LOG_PATH.exists():
        print("No qualifying_log.csv found -- nothing to resolve.")
        return

    log = pd.read_csv(LOG_PATH)
    pending = log[log["status"] == "PENDING"].copy()
    if pending.empty:
        print("No pending picks to resolve.")
        return

    matches = pd.read_parquet(DATA_DIR / "matches.parquet")
    matches["date"] = pd.to_datetime(matches["date"])

    resolved_count = 0
    for idx, row in pending.iterrows():
        result = matches[
            (matches["league"] == row["league"]) &
            (matches["home_team"] == row["home_team_matched"]) &
            (matches["away_team"] == row["away_team_matched"]) &
            (matches["date"] >= pd.to_datetime(row["start_time"]).tz_localize(None) - pd.Timedelta(days=1)) &
            (matches["date"] <= pd.to_datetime(row["start_time"]).tz_localize(None) + pd.Timedelta(days=1))
        ]
        if result.empty:
            continue  # not played yet, or source hasn't posted the result yet -- stays PENDING

        r = result.iloc[0]
        won = bool(r["ftr"] == "H")
        # pnl (theoretical, flat S$50 stake): comparable to the 83.8%/+1.90% backtest figures,
        # computed on EVERY qualifying signal regardless of whether a real bet was placed on it.
        pnl = STAKE_SGD * (row["odds"] - 1) if won else -STAKE_SGD

        log.loc[idx, "status"] = "RESOLVED"
        log.loc[idx, "actual_result"] = f"{int(r['fthg'])}-{int(r['ftag'])} ({r['ftr']})"
        log.loc[idx, "won"] = won
        log.loc[idx, "pnl"] = pnl

        # actual_pnl (real money): only set when record_bet.py marked this row bet_placed,
        # using the REAL stake AND the real odds actually obtained (may differ slightly from
        # the "odds" column, which is just a snapshot from when the dashboard was last built).
        if bool(row.get("bet_placed", False)) and pd.notna(row.get("actual_stake_sgd")):
            stake = float(row["actual_stake_sgd"])
            price = row["actual_odds"] if pd.notna(row.get("actual_odds")) else row["odds"]
            log.loc[idx, "actual_pnl"] = stake * (price - 1) if won else -stake

        resolved_count += 1
        outcome_str = "WON " if won else "LOST"
        bet_note = f"  [REAL BET: S${row['actual_stake_sgd']:.2f}]" if bool(row.get("bet_placed", False)) else ""
        print(f"  [{row['fixture']}] {outcome_str}  actual {int(r['fthg'])}-{int(r['ftag'])}  "
              f"pnl(theoretical S$50)=S${pnl:+.2f}{bet_note}")

    log.to_csv(LOG_PATH, index=False)
    print(f"\n{resolved_count} pick(s) resolved this run.")

    resolved = log[log["status"] == "RESOLVED"]
    still_pending = (log["status"] == "PENDING").sum()
    if len(resolved):
        hit_rate = resolved["won"].mean()
        total_pnl = resolved["pnl"].sum()
        roi = total_pnl / (STAKE_SGD * len(resolved))
        print(f"\n=== Signal track record (every qualifying pick, theoretical S$50 stake) ===")
        print(f"Resolved: {len(resolved)}   Still pending: {still_pending}")
        print(f"Hit rate: {hit_rate:.1%}  (backtested: 83.8%)")
        print(f"Total P&L: S${total_pnl:+.2f}   ROI: {roi:+.2%}  (backtested: +1.90%)")
    else:
        print(f"\nNo picks resolved yet ({still_pending} still pending).")

    real_bets = resolved[resolved.get("bet_placed", False) == True] if len(resolved) else resolved
    if len(real_bets):
        real_hit_rate = real_bets["won"].mean()
        real_pnl = real_bets["actual_pnl"].sum()
        real_staked = real_bets["actual_stake_sgd"].sum()
        real_roi = real_pnl / real_staked if real_staked else float("nan")
        print(f"\n=== REAL money track record (actual bets placed, actual stakes) ===")
        print(f"Bets placed: {len(real_bets)}   Total staked: S${real_staked:.2f}")
        print(f"Real hit rate: {real_hit_rate:.1%}")
        print(f"Real total P&L: S${real_pnl:+.2f}   ROI: {real_roi:+.2%}")
    else:
        n_placed_pending = int(((log.get("bet_placed", False) == True) & (log["status"] == "PENDING")).sum())
        print(f"\nNo real bets resolved yet ({n_placed_pending} placed bet(s) still pending).")


if __name__ == "__main__":
    main()
