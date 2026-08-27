"""
Every joint CPE search run so far (football_joint_cpe.py, football_cpe_widened.py,
football_cpe_window_scan.py) hardcoded outcome_home_win as the target and never
tested the mirrored away-win direction jointly -- a real asymmetry, not just a
framing choice, since the live dashboard's signal was only ever compared against
other home-win combos, never against an equally-searched away-team signal.

This runs the exact mirror of the validated home signal, same window (30, chosen
by football_cpe_window_scan.py), same discovery/holdout discipline:

    away_ppg_diff       = away team's trailing PPG (any venue) - home team's
    away_ppg_away_only  = away team's trailing PPG from ONLY its last 30 away
                            matches (mirrors home_ppg_home_only)

Target: outcome_away_win. Same Q_GRID scan, same CPE>=0.80/lift>=1.5/n>=100
discovery gate, same frozen-threshold holdout check.
"""
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

from data_download import CORE_LEAGUES

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"

WINDOW = 30
SPLIT_DATE = pd.Timestamp("2022-02-01")
Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]
CPE_THRESH = 0.80
MIN_LIFT = 1.5
MIN_N = 100
STAKE_SGD = 50.0


def build_features(matches, window):
    hist_any, hist_away = {}, {}
    away_ppg_diff, away_ppg_away_only = [], []

    for row in matches.itertuples():
        h, a = row.home_team, row.away_team
        hh, ah = hist_any.get(h, []), hist_any.get(a, [])
        aa_away = hist_away.get(a, [])

        if len(hh) >= window and len(ah) >= window:
            away_ppg_diff.append(np.mean(ah[-window:]) - np.mean(hh[-window:]))
        else:
            away_ppg_diff.append(np.nan)
        away_ppg_away_only.append(np.mean(aa_away[-window:]) if len(aa_away) >= window else np.nan)

        if row.ftr == "H":
            h_pts, a_pts = 3, 0
        elif row.ftr == "A":
            h_pts, a_pts = 0, 3
        else:
            h_pts, a_pts = 1, 1
        hist_any.setdefault(h, []).append(h_pts)
        hist_any.setdefault(a, []).append(a_pts)
        hist_away.setdefault(a, []).append(a_pts)

    feat = matches[["date", "league", "home_team", "away_team", "ftr", "odds_a"]].copy()
    feat["away_ppg_diff"] = away_ppg_diff
    feat["away_ppg_away_only"] = away_ppg_away_only
    feat["outcome_away_win"] = (feat["ftr"] == "A").astype(int)
    return feat.dropna(subset=["away_ppg_diff", "away_ppg_away_only"])


def joint_search_discovery(disc):
    away_win = disc["outcome_away_win"].values
    uncond = away_win.mean()
    vals = {"away_ppg_diff": disc["away_ppg_diff"].values, "away_ppg_away_only": disc["away_ppg_away_only"].values}
    thresh = {p: {q: np.quantile(vals[p], q) for q in Q_GRID} for p in vals}

    results = []
    for qs in itertools.product(Q_GRID, repeat=2):
        mask = (vals["away_ppg_diff"] > thresh["away_ppg_diff"][qs[0]]) & \
               (vals["away_ppg_away_only"] > thresh["away_ppg_away_only"][qs[1]])
        n = mask.sum()
        if n < MIN_N:
            continue
        cpe = away_win[mask].mean()
        lift = cpe / uncond if uncond > 0 else np.nan
        if cpe >= CPE_THRESH and lift >= MIN_LIFT:
            results.append(dict(q_diff=qs[0], q_away=qs[1],
                                 t_diff=thresh["away_ppg_diff"][qs[0]], t_away=thresh["away_ppg_away_only"][qs[1]],
                                 n=n, cpe=cpe, lift=lift))
    return pd.DataFrame(results)


def evaluate_holdout(hold, t_diff, t_away):
    mask = (hold["away_ppg_diff"] > t_diff) & (hold["away_ppg_away_only"] > t_away)
    g = hold[mask].dropna(subset=["odds_a"]).copy()
    n = len(g)
    if n == 0:
        return dict(n=0, hit_rate=np.nan, roi=np.nan, pnl=0.0)
    hit_rate = g["outcome_away_win"].mean()
    g["pnl"] = np.where(g["outcome_away_win"] == 1, STAKE_SGD * (g["odds_a"] - 1), -STAKE_SGD)
    return dict(n=n, hit_rate=hit_rate, roi=g["pnl"].sum() / (STAKE_SGD * n), pnl=g["pnl"].sum())


def main():
    matches = pd.read_parquet(DATA_DIR / "matches.parquet")
    matches = matches[matches["league"].isin(CORE_LEAGUES)].sort_values("date").reset_index(drop=True)

    feat = build_features(matches, WINDOW)
    disc = feat[feat["date"] <= SPLIT_DATE]
    hold = feat[feat["date"] > SPLIT_DATE]
    print(f"Away-win mirror signal, window={WINDOW}")
    print(f"discovery n={len(disc)}  holdout n={len(hold)}")
    print(f"Base rate P(away win): {feat['outcome_away_win'].mean():.4f}")

    passing = joint_search_discovery(disc)
    print(f"\nDiscovery combos clearing CPE>=0.80, lift>=1.5, n>=100: {len(passing)}")
    if not len(passing):
        print("NONE. No away-win joint signal at this window clears even the discovery gate.")
        return

    print(passing.sort_values("n", ascending=False).head(10).to_string(index=False))

    best = passing.sort_values("n", ascending=False).iloc[0]
    res = evaluate_holdout(hold, best["t_diff"], best["t_away"])
    print(f"\nBest discovery combo (by n): q=({best['q_diff']:.2f},{best['q_away']:.2f}) "
          f"n={best['n']:.0f} CPE={best['cpe']:.3f} lift={best['lift']:.2f}")
    print(f"HOLDOUT: n={res['n']} hit_rate={res['hit_rate']:.3f} roi={res['roi']:+.2%}"
          if res['n'] else "HOLDOUT: no qualifying matches")

    # also report the single best-CPE combo for comparison
    best_cpe = passing.sort_values("cpe", ascending=False).iloc[0]
    res2 = evaluate_holdout(hold, best_cpe["t_diff"], best_cpe["t_away"])
    print(f"\nBest discovery combo (by CPE): q=({best_cpe['q_diff']:.2f},{best_cpe['q_away']:.2f}) "
          f"n={best_cpe['n']:.0f} CPE={best_cpe['cpe']:.3f} lift={best_cpe['lift']:.2f}")
    print(f"HOLDOUT: n={res2['n']} hit_rate={res2['hit_rate']:.3f} roi={res2['roi']:+.2%}"
          if res2['n'] else "HOLDOUT: no qualifying matches")


if __name__ == "__main__":
    main()
