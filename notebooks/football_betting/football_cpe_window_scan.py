"""
The joint CPE signal (ppg_diff + home_ppg_home_only) was validated at a
10-match trailing window -- but that window length was picked by hand, not
derived from evidence, which is exactly the kind of unvalidated parameter
choice this program's CPE discipline exists to avoid. This scans a grid of
window lengths, runs the SAME discovery-only joint search + frozen-threshold
holdout check at each one, and picks the window based on which one actually
survives holdout -- not on which one looks best in discovery, and not on a
guess by Claude or the user.

Windows tested: 3, 5, 8, 10, 15, 20, 25, 30, 38 (38 = a full single-league
season's matches, the natural upper bound before you're mixing in a
different season's squad).
"""
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

from data_download import CORE_LEAGUES

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"

WINDOWS = [3, 5, 8, 10, 15, 20, 25, 30, 38]
SPLIT_DATE = pd.Timestamp("2022-02-01")
Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]
CPE_THRESH = 0.80
MIN_LIFT = 1.5
MIN_N = 100
STAKE_SGD = 50.0


def build_features_for_window(matches, window):
    hist_any, hist_home = {}, {}
    home_ppg_any, away_ppg_any, home_ppg_home_only = [], [], []

    for row in matches.itertuples():
        h, a = row.home_team, row.away_team
        hh, ah = hist_any.get(h, []), hist_any.get(a, [])
        hh_home = hist_home.get(h, [])

        home_ppg_any.append(np.mean(hh[-window:]) if len(hh) >= window else np.nan)
        away_ppg_any.append(np.mean(ah[-window:]) if len(ah) >= window else np.nan)
        home_ppg_home_only.append(np.mean(hh_home[-window:]) if len(hh_home) >= window else np.nan)

        if row.ftr == "H":
            h_pts, a_pts = 3, 0
        elif row.ftr == "A":
            h_pts, a_pts = 0, 3
        else:
            h_pts, a_pts = 1, 1
        hist_any.setdefault(h, []).append(h_pts)
        hist_any.setdefault(a, []).append(a_pts)
        hist_home.setdefault(h, []).append(h_pts)

    feat = matches[["date", "league", "home_team", "away_team", "ftr", "odds_h"]].copy()
    feat["ppg_diff"] = np.array(home_ppg_any) - np.array(away_ppg_any)
    feat["home_ppg_home_only"] = home_ppg_home_only
    feat["outcome_home_win"] = (feat["ftr"] == "H").astype(int)
    return feat.dropna(subset=["ppg_diff", "home_ppg_home_only"])


def joint_search_discovery(disc):
    home_win = disc["outcome_home_win"].values
    uncond = home_win.mean()
    vals = {"ppg_diff": disc["ppg_diff"].values, "home_ppg_home_only": disc["home_ppg_home_only"].values}
    thresh = {p: {q: np.quantile(vals[p], q) for q in Q_GRID} for p in vals}

    results = []
    for qs in itertools.product(Q_GRID, repeat=2):
        mask = (vals["ppg_diff"] > thresh["ppg_diff"][qs[0]]) & (vals["home_ppg_home_only"] > thresh["home_ppg_home_only"][qs[1]])
        n = mask.sum()
        if n < MIN_N:
            continue
        cpe = home_win[mask].mean()
        lift = cpe / uncond if uncond > 0 else np.nan
        if cpe >= CPE_THRESH and lift >= MIN_LIFT:
            results.append(dict(q_ppg_diff=qs[0], q_home_ppg=qs[1],
                                 t_ppg_diff=thresh["ppg_diff"][qs[0]], t_home_ppg=thresh["home_ppg_home_only"][qs[1]],
                                 n=n, cpe=cpe, lift=lift))
    return pd.DataFrame(results)


def evaluate_holdout(hold, t_ppg_diff, t_home_ppg):
    mask = (hold["ppg_diff"] > t_ppg_diff) & (hold["home_ppg_home_only"] > t_home_ppg)
    g = hold[mask].dropna(subset=["odds_h"]).copy()
    n = len(g)
    if n == 0:
        return dict(n=0, hit_rate=np.nan, roi=np.nan, pnl=0.0)
    hit_rate = g["outcome_home_win"].mean()
    g["pnl"] = np.where(g["outcome_home_win"] == 1, STAKE_SGD * (g["odds_h"] - 1), -STAKE_SGD)
    return dict(n=n, hit_rate=hit_rate, roi=g["pnl"].sum() / (STAKE_SGD * n), pnl=g["pnl"].sum())


def main():
    matches = pd.read_parquet(DATA_DIR / "matches.parquet")
    matches = matches[matches["league"].isin(CORE_LEAGUES)].sort_values("date").reset_index(drop=True)

    summary_rows = []
    for window in WINDOWS:
        print(f"\n{'='*70}\n  Window = {window} matches\n{'='*70}")
        feat = build_features_for_window(matches, window)
        disc = feat[feat["date"] <= SPLIT_DATE]
        hold = feat[feat["date"] > SPLIT_DATE]
        print(f"  discovery n={len(disc)}  holdout n={len(hold)}")

        passing = joint_search_discovery(disc)
        if not len(passing):
            print("  No discovery combo cleared CPE>=0.80, lift>=1.5, n>=100 at this window.")
            summary_rows.append(dict(window=window, best_disc_n=0, holdout_n=0, holdout_hit_rate=np.nan, holdout_roi=np.nan))
            continue

        # pick the discovery combo with the LARGEST n among passing (most statistical power,
        # least likely to be a lucky extreme-quantile artifact) rather than the highest raw CPE
        best = passing.sort_values("n", ascending=False).iloc[0]
        res = evaluate_holdout(hold, best["t_ppg_diff"], best["t_home_ppg"])
        print(f"  Best discovery combo (by n): q=({best['q_ppg_diff']:.2f},{best['q_home_ppg']:.2f}) "
              f"n={best['n']:.0f} CPE={best['cpe']:.3f} lift={best['lift']:.2f}")
        print(f"  HOLDOUT: n={res['n']} hit_rate={res['hit_rate']:.3f} roi={res['roi']:+.2%}"
              if res['n'] else "  HOLDOUT: no qualifying matches")

        summary_rows.append(dict(
            window=window, q_ppg_diff=best["q_ppg_diff"], q_home_ppg=best["q_home_ppg"],
            t_ppg_diff=best["t_ppg_diff"], t_home_ppg=best["t_home_ppg"],
            disc_n=best["n"], disc_cpe=best["cpe"], disc_lift=best["lift"],
            holdout_n=res["n"], holdout_hit_rate=res["hit_rate"], holdout_roi=res["roi"], holdout_pnl=res["pnl"],
        ))

    summary = pd.DataFrame(summary_rows)
    print(f"\n{'='*70}\n  SUMMARY ACROSS ALL WINDOWS\n{'='*70}")
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "football_cpe_window_scan.csv", index=False)
    print(f"\nSaved -> {OUT_DIR / 'football_cpe_window_scan.csv'}")


if __name__ == "__main__":
    main()
