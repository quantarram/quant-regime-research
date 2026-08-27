"""
Weekly checklist: read Singapore Pools' public "upcoming football" odds page
(no login required -- it's the same page the site shows to anonymous visitors),
match team names against the historical database, score each match/market with
the fitted Dixon-Coles model, and print only the bets that clear BOTH:
  - model probability >= 90%
  - model probability - devigged market probability >= EDGE_THRESHOLD

Threshold raised from 80% to 90% and league coverage restricted to
CORE_LEAGUES -- both validated by backtest.py: 80% carried a thin, fragile
edge; 90% on the original 13 leagues held up in a discovery/holdout split
(n=183, hit rate 91.8%, ROI +5.22%). Widening to all 37 available leagues
was tested and made things WORSE (the 24 added leagues alone: n=112, hit
rate 58.9%, ROI -13.9%) -- deep/liquid markets calibrate; thin ones don't.

This only outputs a list. It never places a bet, logs into an account, or
touches any payment method -- that stays entirely manual, on the user's side,
by design.

Coverage caveat: Singapore Pools lists many leagues (J-League, Liga MX, MLS,
Brazilian/Argentine/Chinese/Russian leagues, various internationals, K-League,
Saudi Pro League) that either have no free historical-odds source at all, or
turned out (J-League, Liga MX, etc. -- see CORE_LEAGUES) to calibrate too
poorly to trade even though data exists. This script only ever scores matches
in CORE_LEAGUES. Everything else on the SG Pools board is silently skipped,
not treated as a bet.
"""
import json
import re
import subprocess
import sys
from pathlib import Path
from difflib import get_close_matches

import pandas as pd

from dixon_coles import DixonColesModel, implied_prob_devigged
from data_download import CORE_LEAGUES

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"

PROB_THRESHOLD = 0.90
EDGE_THRESHOLD = 0.0

SGPOOLS_JS_URL = "https://online.singaporepools.com/sports/football"


def load_models():
    """Fit one current model per league on all history to date, restricted to
    CORE_LEAGUES. Returns {league: DixonColesModel}."""
    matches = pd.read_parquet(DATA_DIR / "matches.parquet")
    matches = matches[matches["league"].isin(CORE_LEAGUES)]
    models = {}
    team_lists = {}
    for league, df in matches.groupby("league"):
        if len(df) < 300:
            continue
        m = DixonColesModel().fit(df.sort_values("date"))
        models[league] = m
        team_lists[league] = m.teams
    return models, team_lists


def match_team_name(name, candidates, cutoff=0.72):
    if not candidates:
        return None
    hits = get_close_matches(name, candidates, n=1, cutoff=cutoff)
    return hits[0] if hits else None


def print_manual_instructions():
    print(__doc__)
    print("=" * 70)
    print("This script needs the current SG Pools odds pulled from the browser")
    print("session (online.singaporepools.com/sports/football is a JS app with")
    print("no stable public JSON endpoint). Steps:")
    print()
    print("  1. In the Claude browser tool, navigate to:")
    print(f"     {SGPOOLS_JS_URL}")
    print("  2. Click 'Upcoming' and 'Show All Events'.")
    print("  3. Ask Claude to re-run this checklist -- it will read the live")
    print("     odds off the page and score them against the fitted models.")
    print("=" * 70)


def score_fixtures(fixtures, models, team_lists, include_unverified=False):
    """fixtures: list of dicts with home_team, away_team, odds_h, odds_d, odds_a
    (and optionally odds_over25/odds_under25, odds_ah_home/odds_ah_away/ah_line).
    Returns a DataFrame of qualifying bets."""
    rows = []
    for fx in fixtures:
        matched_league, model = None, None
        for league, teams in team_lists.items():
            h = match_team_name(fx["home_team"], teams)
            a = match_team_name(fx["away_team"], teams)
            if h and a:
                matched_league, model = league, models[league]
                home, away = h, a
                break
        if model is None:
            continue  # league/team not in our covered set -- skip, don't guess

        probs = model.match_probs(home, away)
        if probs is None:
            continue

        # Only the 1x2 market is scored by default. The backtest (see output/results_by_market.csv
        # and output/pnl_by_market.png) showed: over_under_2.5 is badly overconfident (66% realized
        # vs >=80% predicted -- a real Poisson-underdispersion problem, not noise) and ah0_dnb has
        # too few historical observations (n=20) to trust. double_chance looked good in the backtest
        # but only because there's no free historical double-chance odds data to price it against --
        # its backtest P&L used a synthetic no-vig price, not a real market price, so it's excluded
        # here rather than shown as if it were a validated signal. Pass include_unverified=True to
        # see it anyway, clearly labeled.
        h, d, a = fx.get("odds_h"), fx.get("odds_d"), fx.get("odds_a")
        if h and d and a and min(h, d, a) > 1:
            mkt = implied_prob_devigged([h, d, a])
            for sel, mp, mo in zip(["H", "D", "A"], mkt, [h, d, a]):
                p = probs["1x2"][sel]
                if p >= PROB_THRESHOLD and (p - mp) >= EDGE_THRESHOLD:
                    rows.append(dict(league=matched_league, fixture=f"{fx['home_team']} vs {fx['away_team']}",
                                      market="1x2", selection=sel, model_p=p, market_p=mp, odds=mo))
            if include_unverified:
                dc_pairs = {"1X": ("H", "D"), "X2": ("D", "A"), "12": ("H", "A")}
                for sel, (o1, o2) in dc_pairs.items():
                    mp_dc = mkt[["H", "D", "A"].index(o1)] + mkt[["H", "D", "A"].index(o2)]
                    p = probs["double_chance"][sel]
                    if p >= PROB_THRESHOLD and (p - mp_dc) >= EDGE_THRESHOLD:
                        rows.append(dict(league=matched_league, fixture=f"{fx['home_team']} vs {fx['away_team']}",
                                          market="double_chance [UNVERIFIED PRICE]", selection=sel,
                                          model_p=p, market_p=mp_dc,
                                          odds=round(1 / mp_dc, 3) if mp_dc > 0 else None))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fixtures-json":
        fixtures = json.loads(Path(sys.argv[2]).read_text())
        models, team_lists = load_models()
        result = score_fixtures(fixtures, models, team_lists)
        if len(result):
            result = result.sort_values("model_p", ascending=False)
            print(result.to_string(index=False))
            result.to_csv(OUT_DIR / "live_checklist.csv", index=False)
        else:
            print(f"No qualifying bets (model P>={PROB_THRESHOLD:.0%} and beating the market) this week.")
    else:
        print_manual_instructions()
