"""
check_history_weights.py
==========================
Standalone diagnostic: how much does the h(Pi) history-length down-weight
(paper Section 2.3, implemented in backtest_engine.compute_h_for_joint_row)
actually suppress short-history predictors like the spot Bitcoin ETFs
(IBIT, FBTC, BITB -- launched Jan 2024, under 2 years of history as of
the 2024-12-31 training cutoff)?

Run this after generating joint_cpe_results.parquet. No backtest needed.

Usage:
    python check_history_weights.py
    python check_history_weights.py --joint joint_cpe_results.parquet
"""

import argparse
import pandas as pd
from backtest_engine import _ticker_history_obs, _history_weight, TRAIN_CUTOFF

SHORT_HISTORY_WATCHLIST = ["IBIT", "FBTC", "BITB", "GBTC", "ETHE"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", default="joint_cpe_results.parquet")
    parser.add_argument("--prices", default="multiasset_prices.parquet")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)

    print(f"\n{'='*70}")
    print(f"  HISTORY-WEIGHT DIAGNOSTIC  (training cutoff: {TRAIN_CUTOFF.date()})")
    print(f"{'='*70}\n")

    print("  Per-ticker training-period observation count and resulting")
    print("  history weight h(t) -- ramps 0.35 (floor, <=100 obs) linearly")
    print("  to 1.0 (full weight, >=756 obs / ~3 trading years):\n")

    rows = []
    for t in SHORT_HISTORY_WATCHLIST:
        if t not in prices.columns:
            print(f"  {t:<10} not present in {args.prices}, skipping")
            continue
        n = _ticker_history_obs(t, prices)
        w = _history_weight(n)
        rows.append({"ticker": t, "train_obs": n, "history_weight": round(w, 3)})

    diag = pd.DataFrame(rows)
    print(diag.to_string(index=False))

    print(f"\n  For reference: a 'fully seasoned' instrument with >=756 training")
    print(f"  observations gets a weight of 1.0 -- so any ticker above shows")
    print(f"  what fraction of full weight its configurations actually carry")
    print(f"  in the w(Pi) quality-weight formula used for sizing/firing.\n")

    # If a joint screen is available, show how many surviving configurations
    # actually involve these short-history tickers, and at what weight
    try:
        joint = pd.read_parquet(args.joint)
    except FileNotFoundError:
        print(f"  ({args.joint} not found -- skipping joint-screen exposure check.")
        print(f"   Run joint_cpe_engine.py first if you want this section.)")
        return

    print(f"{'='*70}")
    print(f"  EXPOSURE: how many surviving joint configs involve these tickers?")
    print(f"{'='*70}\n")

    watchlist_set = set(SHORT_HISTORY_WATCHLIST)
    joint["involves_short_history"] = joint["predictors"].apply(
        lambda preds: any(p in watchlist_set for p in preds)
    )
    n_total = len(joint)
    n_involved = joint["involves_short_history"].sum()
    print(f"  Total surviving joint configs: {n_total}")
    print(f"  Configs involving a short-history ticker ({SHORT_HISTORY_WATCHLIST}): "
          f"{n_involved} ({100*n_involved/max(n_total,1):.1f}%)")

    if n_involved > 0:
        print(f"\n  Targets (Y) affected, with config counts:")
        affected = joint[joint["involves_short_history"]]
        print(affected.groupby("Y").size().sort_values(ascending=False).to_string())

        print(f"\n  Sample of affected configs (up to 15):")
        cols = ["Y", "direction", "tau_future", "q_Y", "predictors", "joint_CPE", "n_joint"]
        print(affected[cols].head(15).to_string(index=False))

    print(f"\n{'='*70}")
    print("  READING THIS OUTPUT")
    print(f"{'='*70}")
    print("""
  If IBIT/FBTC/BITB show a history_weight well below 1.0 (e.g. ~0.4-0.6)
  AND they still appear in many surviving joint configs that fire often
  in the backtest, that means the down-weighting is doing SOME work but
  not enough to prevent these short-history predictors from dominating
  a sleeve's signal -- the weight reduces conviction/sizing, but does
  NOT prevent the configuration from firing at all (firing is governed
  by CPE/lift/sample-size filters in the screen, which the history
  weight does not touch). If you want short-history predictors to be
  excluded outright rather than merely down-weighted, that requires a
  separate, harder filter -- e.g. a minimum training_obs requirement
  applied at screen-build time, not just a continuous weight applied at
  scoring time.
""")


if __name__ == "__main__":
    main()
