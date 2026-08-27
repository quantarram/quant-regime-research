"""
Widened predictor set for the football CPE joint search, with the
discovery/holdout discipline applied FROM THE START this time (not bolted
on after finding something that looked good) -- the joint search itself
only ever sees the discovery period; holdout numbers are computed once,
after selection, and reported regardless of how they turn out.

All predictors are "actual events" -- derived only from real match results
and dates, nothing from market odds (market_prob_home is kept only as a
calibration control, as before). New predictors added on top of the
original 3 (ppg_diff, home_ppg_home_only, away_ppg_away_only):

    goal_diff_diff     : trailing avg goal margin (GF-GA), home - away,
                          same 10-match window -- captures dominance/margin,
                          which PPG alone can't (a team grinding out 1-0s
                          and a team winning 4-0s can have the same PPG)
    season_ppg_diff     : CURRENT SEASON cumulative PPG, home - away
                          (resets each season; requires >=5 games played
                          this season for both teams -- distinct from the
                          rolling trailing window, which can span a
                          season boundary early in the year)
    h2h_home_winrate     : the home team's actual win rate in its last up
                          to 6 meetings with THIS SPECIFIC opponent
                          (any venue), requires >=3 prior meetings --
                          literally "how have these two teams' actual past
                          meetings gone"
    rest_days_diff       : (home team's days since last match) - (away
                          team's days since last match), capped at +/-20
                          to exclude summer-break noise -- a fatigue/
                          fixture-congestion proxy from real scheduling
    win_streak_diff      : current active win streak length, home - away
                          (capped at 8) -- momentum, distinct from average
                          form

Discovery/holdout split: matches before 2022-02-01 are DISCOVERY (joint
search runs here only); everything from 2022-02-01 onward is HOLDOUT,
touched only after a combo has already been chosen.
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dixon_coles import implied_prob_devigged

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

FORM_WINDOW = 10
SPLIT_DATE = pd.Timestamp("2022-02-01")
Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]
CPE_THRESH = 0.80
MIN_LIFT = 1.5
MIN_N = 100
STAKE_SGD = 50.0
MIN_SEASON_GAMES = 5
MIN_H2H_MEETINGS = 3
H2H_LOOKBACK = 6
REST_CAP = 20
STREAK_CAP = 8


def build_features():
    matches = pd.read_parquet(DATA_DIR / "matches.parquet").sort_values("date").reset_index(drop=True)

    hist_any, hist_home, hist_away = {}, {}, {}          # team -> [points]
    hist_margin = {}                                       # team -> [goal margin], any venue
    season_pts, season_games, season_key = {}, {}, {}     # team -> cumulative this-season state
    last_date = {}                                         # team -> last match date
    win_streak = {}                                        # team -> current active win streak
    h2h = {}                                                # frozenset({A,B}) -> [(date, winner_or_None)]

    rows = []
    for row in matches.itertuples():
        h, a, lg, seas = row.home_team, row.away_team, row.league, row.season
        hh, ah = hist_any.get(h, []), hist_any.get(a, [])
        hh_home, aa_away = hist_home.get(h, []), hist_away.get(a, [])
        hm, am = hist_margin.get(h, []), hist_margin.get(a, [])

        home_ppg_any = np.mean(hh[-FORM_WINDOW:]) if len(hh) >= FORM_WINDOW else np.nan
        away_ppg_any = np.mean(ah[-FORM_WINDOW:]) if len(ah) >= FORM_WINDOW else np.nan
        home_ppg_home_only = np.mean(hh_home[-FORM_WINDOW:]) if len(hh_home) >= FORM_WINDOW else np.nan
        away_ppg_away_only = np.mean(aa_away[-FORM_WINDOW:]) if len(aa_away) >= FORM_WINDOW else np.nan
        home_margin = np.mean(hm[-FORM_WINDOW:]) if len(hm) >= FORM_WINDOW else np.nan
        away_margin = np.mean(am[-FORM_WINDOW:]) if len(am) >= FORM_WINDOW else np.nan

        # season cumulative PPG (resets when this team's stored season key changes)
        if season_key.get(h) != (lg, seas):
            season_pts[h], season_games[h], season_key[h] = 0, 0, (lg, seas)
        if season_key.get(a) != (lg, seas):
            season_pts[a], season_games[a], season_key[a] = 0, 0, (lg, seas)
        h_season_ppg = season_pts[h] / season_games[h] if season_games[h] >= MIN_SEASON_GAMES else np.nan
        a_season_ppg = season_pts[a] / season_games[a] if season_games[a] >= MIN_SEASON_GAMES else np.nan

        # head-to-head: this specific pair's last meetings, any venue
        pair_key = frozenset((h, a))
        pair_hist = [rec for rec in h2h.get(pair_key, [])]
        recent = pair_hist[-H2H_LOOKBACK:]
        if len(recent) >= MIN_H2H_MEETINGS:
            h2h_home_winrate = np.mean([1.0 if w == h else 0.0 for _, w in recent])
        else:
            h2h_home_winrate = np.nan

        # rest days
        h_rest = (row.date - last_date[h]).days if h in last_date else np.nan
        a_rest = (row.date - last_date[a]).days if a in last_date else np.nan
        rest_diff = np.nan
        if pd.notna(h_rest) and pd.notna(a_rest):
            rest_diff = float(np.clip(h_rest - a_rest, -REST_CAP, REST_CAP))

        # win streaks
        h_streak = min(win_streak.get(h, 0), STREAK_CAP)
        a_streak = min(win_streak.get(a, 0), STREAK_CAP)

        rows.append(dict(
            date=row.date, league=lg, home_team=h, away_team=a, ftr=row.ftr,
            odds_h=row.odds_h, odds_d=row.odds_d, odds_a=row.odds_a,
            ppg_diff=home_ppg_any - away_ppg_any,
            home_ppg_home_only=home_ppg_home_only, away_ppg_away_only=away_ppg_away_only,
            goal_diff_diff=home_margin - away_margin,
            season_ppg_diff=h_season_ppg - a_season_ppg,
            h2h_home_winrate=h2h_home_winrate,
            rest_days_diff=rest_diff,
            win_streak_diff=float(h_streak - a_streak),
        ))

        # -- update all state with this match's REAL result (after prediction) --
        if row.ftr == "H":
            h_pts, a_pts = 3, 0
        elif row.ftr == "A":
            h_pts, a_pts = 0, 3
        else:
            h_pts, a_pts = 1, 1
        hist_any.setdefault(h, []).append(h_pts)
        hist_any.setdefault(a, []).append(a_pts)
        hist_home.setdefault(h, []).append(h_pts)
        hist_away.setdefault(a, []).append(a_pts)
        hist_margin.setdefault(h, []).append(row.fthg - row.ftag)
        hist_margin.setdefault(a, []).append(row.ftag - row.fthg)
        season_pts[h] += h_pts; season_games[h] += 1
        season_pts[a] += a_pts; season_games[a] += 1
        winner = h if row.ftr == "H" else (a if row.ftr == "A" else None)
        h2h.setdefault(pair_key, []).append((row.date, winner))
        last_date[h] = row.date
        last_date[a] = row.date
        win_streak[h] = win_streak.get(h, 0) + 1 if row.ftr == "H" else 0
        win_streak[a] = win_streak.get(a, 0) + 1 if row.ftr == "A" else 0

    feat = pd.DataFrame(rows)
    devig = feat[["odds_h", "odds_d", "odds_a"]].apply(
        lambda r: implied_prob_devigged([r["odds_h"], r["odds_d"], r["odds_a"]])
        if r.notna().all() and (r > 1).all() else pd.Series([np.nan, np.nan, np.nan]),
        axis=1, result_type="expand"
    )
    feat["market_prob_home"] = devig[0].values
    feat["outcome_home_win"] = (feat["ftr"] == "H").astype(int)
    return feat


PREDICTORS = ["ppg_diff", "home_ppg_home_only", "away_ppg_away_only", "goal_diff_diff",
              "season_ppg_diff", "h2h_home_winrate", "rest_days_diff", "win_streak_diff"]


def single_predictor_scan(disc):
    """Mirrors football_cpe_engine.py's single-predictor CPE scan, discovery-only."""
    results = []
    for predictor in PREDICTORS + ["market_prob_home"]:
        sub = disc.dropna(subset=[predictor])
        if len(sub) < MIN_N:
            continue
        vals = sub[predictor].values
        home_win = sub["outcome_home_win"].values
        uncond = home_win.mean()
        full_q_grid = sorted(set(Q_GRID + [round(1 - q, 10) for q in Q_GRID]))
        thresholds = {q: np.quantile(vals, q) for q in full_q_grid}
        best = (0.0, None, None)
        for q in Q_GRID:
            cond = vals > thresholds[q]
            if cond.sum() >= MIN_N:
                cpe = home_win[cond].mean()
                lift = cpe / uncond if uncond > 0 else np.nan
                if cpe >= CPE_THRESH and lift >= MIN_LIFT and cpe > best[0]:
                    best = (cpe, q, lift)
        results.append((predictor, len(sub), round(uncond, 4), *best))
    return pd.DataFrame(results, columns=["predictor", "n_total", "uncond_prob", "best_CPE", "at_q", "lift"])


def joint_search(disc, predictors, max_size):
    """NaN handling is per-combo, not a global upfront dropna across all predictors --
    otherwise requiring e.g. h2h_home_winrate (only defined for pairs with >=3 prior
    meetings) to be non-null for EVERY combo, including ones that don't even use it,
    would needlessly starve every other combo's sample size."""
    home_win_full = disc["outcome_home_win"].values
    uncond = home_win_full.mean()
    valid = {p: disc[p].notna().values for p in predictors}
    vals = {p: disc[p].values for p in predictors}
    thresh = {p: {q: np.nanquantile(vals[p], q) for q in Q_GRID} for p in predictors}

    results = []
    for size in range(2, max_size + 1):
        for combo in itertools.combinations(predictors, size):
            for qs in itertools.product(Q_GRID, repeat=size):
                mask = np.ones(len(disc), dtype=bool)
                for p, q in zip(combo, qs):
                    mask &= valid[p] & (vals[p] > thresh[p][q])
                n = mask.sum()
                if n < MIN_N:
                    continue
                cpe = home_win_full[mask].mean()
                lift = cpe / uncond if uncond > 0 else np.nan
                if cpe >= CPE_THRESH and lift >= MIN_LIFT:
                    results.append(dict(predictors=combo, quantiles=qs, thresholds={p: thresh[p][q] for p, q in zip(combo, qs)},
                                         n=n, CPE=cpe, lift=lift))
    return pd.DataFrame(results)


def evaluate_on_holdout(hold, combo_row):
    mask = np.ones(len(hold), dtype=bool)
    for p, t in combo_row["thresholds"].items():
        mask &= hold[p] > t
    g = hold[mask].dropna(subset=["odds_h"]).copy()
    n = len(g)
    if n == 0:
        return dict(n=0, hit_rate=np.nan, roi=np.nan, pnl=0.0, avg_market_p=np.nan)
    hit_rate = g["outcome_home_win"].mean()
    g["pnl"] = np.where(g["outcome_home_win"] == 1, STAKE_SGD * (g["odds_h"] - 1), -STAKE_SGD)
    pnl = g["pnl"].sum()
    return dict(n=n, hit_rate=hit_rate, roi=pnl / (STAKE_SGD * n), pnl=pnl,
                avg_market_p=(1 / g["odds_h"]).mean())


def main():
    print("Building widened, causal feature set (single pass over all matches)...")
    feat = build_features()
    feat.to_parquet(OUT_DIR / "football_cpe_widened_features.parquet", index=False)

    disc = feat[feat["date"] <= SPLIT_DATE].copy()
    hold = feat[feat["date"] > SPLIT_DATE].copy()
    print(f"Discovery: {len(disc):,} matches (through {SPLIT_DATE.date()})")
    print(f"Holdout:   {len(hold):,} matches ({hold['date'].min().date()} to {hold['date'].max().date()})")

    print("\nCoverage (non-null count) per new predictor, discovery period:")
    for p in PREDICTORS:
        print(f"  {p:22s} n={disc[p].notna().sum():,}")

    print(f"\n{'='*70}\n  SINGLE-PREDICTOR SCAN (discovery only)\n{'='*70}")
    single = single_predictor_scan(disc)
    print(single.to_string(index=False))

    print(f"\n{'='*70}\n  JOINT SEARCH, actual-events predictors only (discovery only)\n{'='*70}")
    joint = joint_search(disc, PREDICTORS, max_size=3)
    print(f"  Passing combos found: {len(joint)}")
    if len(joint):
        joint = joint.sort_values("n", ascending=False).reset_index(drop=True)
        print(joint.drop(columns=["thresholds"]).head(15).to_string(index=False))

        print(f"\n{'='*70}\n  HOLDOUT EVALUATION (thresholds frozen from discovery, never re-fit)\n{'='*70}")
        holdout_rows = []
        for i, r in joint.iterrows():
            res = evaluate_on_holdout(hold, r)
            holdout_rows.append(dict(predictors=r["predictors"], quantiles=r["quantiles"],
                                      disc_n=r["n"], disc_CPE=r["CPE"], disc_lift=r["lift"], **res))
        hres = pd.DataFrame(holdout_rows).sort_values("n", ascending=False)
        pd.set_option("display.width", 160)
        print(hres.to_string(index=False))
        hres.to_csv(OUT_DIR / "football_cpe_widened_holdout.csv", index=False)

        # plot: discovery CPE vs holdout hit rate, and holdout ROI, for the widest-n combos
        top = hres.drop_duplicates(subset=["predictors"]).nlargest(8, "n")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        labels = [" + ".join(p.replace("_diff", "").replace("_only", "") for p in preds) for preds in top["predictors"]]
        x = np.arange(len(top))
        ax = axes[0]
        ax.bar(x - 0.2, top["disc_CPE"], 0.4, label="Discovery CPE", color="#999999")
        ax.bar(x + 0.2, top["hit_rate"], 0.4, label="Holdout hit rate", color="#2166ac")
        ax.axhline(0.80, color="red", linestyle=":")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.legend(fontsize=8)
        ax.set_title("Widened joint CPE: discovery vs holdout hit rate")
        ax = axes[1]
        ax.bar(x, top["roi"] * 100, color="#2166ac")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title("Holdout ROI per bet (real market odds)")
        ax.set_ylabel("ROI (%)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "football_cpe_widened_holdout.png", dpi=150)
        print(f"\nSaved plot -> {OUT_DIR / 'football_cpe_widened_holdout.png'}")
    else:
        print("  No combos cleared CPE>=0.80, lift>=1.5, n>=100 on the discovery period.")


if __name__ == "__main__":
    main()
