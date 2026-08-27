"""
CPE Test on Football Match Outcomes
====================================
Same engine, same gates, same methodology as the rest of this research
program's CPE work (see files/cpe_engine_polymarket_odds.py, the closest
prior application -- discrete, single-resolution events, exactly like a
football match, unlike the overlapping multi-day holding-period strategies
elsewhere in the book that need the separate HAC/effective-N correction).

CPE = P(outcome | predictor in its tail), tested by threshold-scanning each
predictor across a quantile grid, in both directions (bullish tail -> home
win, bearish tail -> away win). A predictor/threshold survives only if it
clears ALL THREE gates used everywhere else in this program:
    CPE   >= 0.80   (the actual question: how often did it happen)
    lift  >= 1.5x   (over that outcome's own unconditional base rate --
                      not over Singapore Pools' odds. A signal only counts
                      if it beats what you'd expect from football in
                      general, e.g. "most home favorites win" doesn't
                      count just because it clears 80% -- it has to clear
                      80% BECAUSE of the specific condition, well beyond
                      the base rate)
    n     >= 100    (enough real matches behind the number to trust it)

market_prob_home (Singapore-Pools-relevant devigged odds) is included as
one predictor among the others, exactly as up_mid_level was in the
Polymarket test -- a calibration control, not the benchmark the others are
measured against. Whether a given predictor's CPE is above or below the
market's own implied probability is incidental; the market is not assumed
to already know everything a predictor captures.

Predictors (each also tested at two trailing-form windows, 5 and 10 matches,
to see whether any survivor is robust to that choice rather than an
artifact of one window):
    market_prob_home   : devigged home-win probability from closing odds
    ppg_diff            : (home trailing PPG) - (away trailing PPG), any venue
    home_ppg_home_only  : home team's PPG computed from ONLY their last N
                           home matches (not mixed with away matches)
    away_ppg_away_only  : away team's PPG computed from ONLY their last N
                           away matches

Usage:
    python football_cpe_engine.py
Output:
    football_cpe_features.parquet
    football_cpe_results.parquet   (survivors, if any)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from dixon_coles import implied_prob_devigged

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

FORM_WINDOWS = [5, 10]
Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]
CPE_THRESH = 0.80
MIN_SAMPLE = 100
MIN_LIFT = 1.5


def build_features():
    matches = pd.read_parquet(DATA_DIR / "matches.parquet")
    matches = matches.sort_values("date").reset_index(drop=True)

    # any-venue trailing PPG for both teams (for ppg_diff)
    home_any = {w: [] for w in FORM_WINDOWS}
    away_any = {w: [] for w in FORM_WINDOWS}
    home_home = {w: [] for w in FORM_WINDOWS}   # home team's PPG from ITS home matches only
    away_away = {w: [] for w in FORM_WINDOWS}   # away team's PPG from ITS away matches only

    hist_any = {}       # team -> list of points, any venue
    hist_home = {}       # team -> list of points, home matches only
    hist_away = {}       # team -> list of points, away matches only

    for row in matches.itertuples():
        h, a = row.home_team, row.away_team
        hh = hist_any.get(h, [])
        ah = hist_any.get(a, [])
        hh_home = hist_home.get(h, [])
        aa_away = hist_away.get(a, [])
        for w in FORM_WINDOWS:
            home_any[w].append(np.mean(hh[-w:]) if len(hh) >= w else np.nan)
            away_any[w].append(np.mean(ah[-w:]) if len(ah) >= w else np.nan)
            home_home[w].append(np.mean(hh_home[-w:]) if len(hh_home) >= w else np.nan)
            away_away[w].append(np.mean(aa_away[-w:]) if len(aa_away) >= w else np.nan)

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

    feat_rows = []
    for w in FORM_WINDOWS:
        f = pd.DataFrame({
            "date": matches["date"], "league": matches["league"],
            "home_team": matches["home_team"], "away_team": matches["away_team"],
            "ftr": matches["ftr"],
            "form_window": w,
            "home_ppg_any": home_any[w], "away_ppg_any": away_any[w],
            "home_ppg_home_only": home_home[w], "away_ppg_away_only": away_away[w],
            "odds_h": matches["odds_h"], "odds_d": matches["odds_d"], "odds_a": matches["odds_a"],
        })
        f["ppg_diff"] = f["home_ppg_any"] - f["away_ppg_any"]

        devig = matches[["odds_h", "odds_d", "odds_a"]].apply(
            lambda r: implied_prob_devigged([r["odds_h"], r["odds_d"], r["odds_a"]])
            if r.notna().all() and (r > 1).all() else pd.Series([np.nan, np.nan, np.nan]),
            axis=1, result_type="expand"
        )
        f["market_prob_home"] = devig[0].values

        f["outcome_home_win"] = (matches["ftr"] == "H").astype(int).values
        f["outcome_away_win"] = (matches["ftr"] == "A").astype(int).values
        feat_rows.append(f)

    return pd.concat(feat_rows, ignore_index=True)


def compute_cpe(feat):
    predictors = ["market_prob_home", "ppg_diff", "home_ppg_home_only", "away_ppg_away_only"]
    results = []

    for w in FORM_WINDOWS:
        sub_all = feat[feat["form_window"] == w]
        for predictor in predictors:
            sub = sub_all.dropna(subset=[predictor])
            if len(sub) < MIN_SAMPLE:
                continue
            vals = sub[predictor].values
            home_win = sub["outcome_home_win"].values
            away_win = sub["outcome_away_win"].values
            uncond_home = home_win.mean()
            uncond_away = away_win.mean()

            full_q_grid = sorted(set(Q_GRID + [round(1 - q, 10) for q in Q_GRID]))
            thresholds = {q: np.quantile(vals, q) for q in full_q_grid}

            for q in Q_GRID:
                thresh_up = thresholds[q]
                thresh_dn = thresholds[round(1 - q, 10)]

                # Bullish: predictor in upper tail -> home win?
                cond_bull = vals > thresh_up
                n_bull = cond_bull.sum()
                if n_bull >= MIN_SAMPLE:
                    cpe_bull = home_win[cond_bull].mean()
                    lift_bull = cpe_bull / uncond_home if uncond_home > 0 else np.nan
                    if cpe_bull >= CPE_THRESH and lift_bull >= MIN_LIFT:
                        results.append((predictor, w, q, round(float(cpe_bull), 4),
                                         round(float(uncond_home), 4), round(float(lift_bull), 4),
                                         int(n_bull), len(sub), "bullish (home win)"))

                # Bearish: predictor in lower tail -> away win?
                cond_bear = vals < thresh_dn
                n_bear = cond_bear.sum()
                if n_bear >= MIN_SAMPLE:
                    cpe_bear = away_win[cond_bear].mean()
                    lift_bear = cpe_bear / uncond_away if uncond_away > 0 else np.nan
                    if cpe_bear >= CPE_THRESH and lift_bear >= MIN_LIFT:
                        results.append((predictor, w, q, round(float(cpe_bear), 4),
                                         round(float(uncond_away), 4), round(float(lift_bear), 4),
                                         int(n_bear), len(sub), "bearish (away win)"))

    cols = ["predictor", "form_window", "q", "CPE", "uncond_prob", "lift", "n_condition", "n_total", "direction"]
    return pd.DataFrame(results, columns=cols)


def diagnostic_max_cpe(feat):
    predictors = ["market_prob_home", "ppg_diff", "home_ppg_home_only", "away_ppg_away_only"]
    rows = []
    for w in FORM_WINDOWS:
        sub_all = feat[feat["form_window"] == w]
        for predictor in predictors:
            sub = sub_all.dropna(subset=[predictor])
            if len(sub) < MIN_SAMPLE:
                continue
            vals = sub[predictor].values
            home_win = sub["outcome_home_win"].values
            away_win = sub["outcome_away_win"].values
            best_cpe, best_q, best_dir, best_lift = 0.0, None, None, None
            uncond_home = home_win.mean()
            uncond_away = away_win.mean()
            for q in Q_GRID:
                thresh_up = np.quantile(vals, q)
                thresh_dn = np.quantile(vals, round(1 - q, 10))
                cond_bull = vals > thresh_up
                if cond_bull.sum() >= MIN_SAMPLE:
                    cpe_b = home_win[cond_bull].mean()
                    if cpe_b > best_cpe:
                        best_cpe, best_q, best_dir = cpe_b, q, "bullish"
                        best_lift = cpe_b / uncond_home if uncond_home > 0 else np.nan
                cond_bear = vals < thresh_dn
                if cond_bear.sum() >= MIN_SAMPLE:
                    cpe_d = away_win[cond_bear].mean()
                    if cpe_d > best_cpe:
                        best_cpe, best_q, best_dir = cpe_d, q, "bearish"
                        best_lift = cpe_d / uncond_away if uncond_away > 0 else np.nan
            rows.append((predictor, w, round(best_cpe, 4), best_q, best_dir,
                         round(best_lift, 4) if best_lift is not None else None, len(sub)))
    return pd.DataFrame(rows, columns=["predictor", "form_window", "best_CPE", "at_q", "direction", "lift_at_best", "n_total"])


def main():
    print(f"\n{'='*70}")
    print(f"  CPE TEST ON FOOTBALL MATCH OUTCOMES")
    print(f"{'='*70}")
    print(f"  Form windows (matches): {FORM_WINDOWS}")
    print(f"  Filter: CPE >= {CPE_THRESH} AND lift >= {MIN_LIFT} AND n >= {MIN_SAMPLE}")

    feat = build_features()
    n_matches = feat[feat["form_window"] == FORM_WINDOWS[0]].dropna(subset=["ppg_diff"]).shape[0]
    print(f"\n  Matches with sufficient trailing history: {n_matches:,}")
    print(f"  Base rate P(home win): {feat['outcome_home_win'].mean():.4f}")
    print(f"  Base rate P(away win): {feat['outcome_away_win'].mean():.4f}")

    feat_path = OUT_DIR / "football_cpe_features.parquet"
    feat.to_parquet(feat_path, index=False)
    print(f"  Saved feature table -> {feat_path}")

    print(f"\n  Running CPE quantile-exceedance test across 4 predictors x "
          f"{len(FORM_WINDOWS)} windows x {len(Q_GRID)} quantiles x 2 directions...")
    results = compute_cpe(feat)

    out_path = OUT_DIR / "football_cpe_results.parquet"
    results.to_parquet(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"  COMPLETE")
    print(f"  Rows surviving all 3 gates: {len(results)}")
    print(f"  Saved -> {out_path}")

    diag = diagnostic_max_cpe(feat)
    print(f"\n  Best CPE achieved per predictor (regardless of gate) -- "
          f"honest characterization even if nothing survives:")
    print(diag.to_string(index=False))

    if results.empty:
        print(f"\n  NO CONFIGURATION CLEARED CPE>={CPE_THRESH}, lift>={MIN_LIFT}, n>={MIN_SAMPLE}.")
    else:
        print(f"\n  Direction breakdown:")
        print(results.groupby("direction")["CPE"].describe().round(3).to_string())
        print(f"\n  All surviving configurations:")
        print(results.sort_values("CPE", ascending=False).to_string(index=False))
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
