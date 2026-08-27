"""
Daily dashboard generator: pulls Singapore Pools' own public odds API (no
login, no browser needed -- confirmed live to be a plain unauthenticated
JSON endpoint), fits the validated Dixon-Coles model on CORE_LEAGUES history,
and lists every upcoming match where a 1x2 selection clears the validated
rule: model probability >= 90% AND beats the market's own devigged price.

This produces a list only. It never places a bet, logs into an account, or
touches money -- checking availability on Singapore Pools and deciding
whether to bet stays entirely manual, by design.

Usage:
    python daily_dashboard.py            # fetch, score, write dashboard.html
Output:
    output/dashboard.html                 -- the page itself
    output/dashboard_qualifying.json      -- same data, machine-readable
"""
import json
import re
import unicodedata
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path

import requests
import pandas as pd

from dixon_coles import DixonColesModel, implied_prob_devigged
from data_download import CORE_LEAGUES

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

PROB_THRESHOLD = 0.90
EDGE_THRESHOLD = 0.0

SGPOOLS_API = "https://api.singaporepools.com/football/events/v1/upcoming-event"

# (country, competition name) as returned by the SG Pools API -> our league code.
# Confirmed live against the API for the rows marked "confirmed"; the rest
# (Portugal/Turkey/Greece/Belgium/Scotland) weren't on the board on the day this
# was built, so they're best-guess names, marked unconfirmed below.
#
# STRICT ON PURPOSE: a competition string not in this dict is SKIPPED, never
# passed to a fuzzy team-name search across all 13 leagues' rosters. An early
# version did that as a "catch unrecognized names" fallback and it produced a
# live false positive: "Dutch League Div 2" (Eerste Divisie, not a validated
# league) wasn't in the map, fell through to searching all leagues, and both
# teams happened to have played in the Eredivisie at some point in the last 10
# years -- so it got scored with stale top-flight team-strength ratings and
# claimed 93% on a match the real market priced at 22%. Missing a genuine
# match because a competition name isn't mapped yet is a far smaller cost than
# silently scoring the wrong league.
COMPETITION_MAP = {
    ("England", "English Premier"): "E0",           # confirmed
    ("England", "English League Champ"): "E1",       # confirmed
    ("Germany", "German League"): "D1",              # confirmed
    ("Germany", "German League Div 2"): "D2",         # confirmed
    ("Italy", "Italian League"): "I1",               # confirmed
    ("Spain", "Spanish League"): "SP1",              # confirmed
    ("France", "French League"): "F1",               # confirmed
    ("Netherlands", "Dutch League"): "N1",           # confirmed
    ("Portugal", "Portuguese League"): "P1",         # unconfirmed guess
    ("Turkey", "Turkish League"): "T1",              # unconfirmed guess
    ("Greece", "Greek League"): "G1",                # unconfirmed guess
    ("Belgium", "Belgian League"): "B1",             # unconfirmed guess
    ("Scotland", "Scottish Premier"): "SC0",         # unconfirmed guess
    ("Scotland", "Scottish League"): "SC0",          # unconfirmed guess
}

# Competitions confirmed to exist on the board that must NEVER map to a
# CORE_LEAGUES code, even though the country/name looks superficially close
# to one that does -- these are lower tiers of countries we DO trade the top
# flight of, so a naming-drift fix to COMPETITION_MAP must not accidentally
# catch these instead.
KNOWN_EXCLUDED = {
    ("England", "English League One"), ("England", "English League Two"),
    ("Italy", "Italian League Div 2"), ("Spain", "Spanish League Div 2"),
    ("France", "French League Div 2"), ("Netherlands", "Dutch League Div 2"),
}


def load_models():
    matches = pd.read_parquet(DATA_DIR / "matches.parquet")
    matches = matches[matches["league"].isin(CORE_LEAGUES)]
    models, team_lists = {}, {}
    for league, df in matches.groupby("league"):
        if len(df) < 300:
            continue
        models[league] = DixonColesModel().fit(df.sort_values("date"))
        team_lists[league] = models[league].teams
    return models, team_lists


def _norm(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_team(name, candidates, cutoff=0.72):
    if not candidates:
        return None
    norm_map = {_norm(c): c for c in candidates}
    n = _norm(name)
    if n in norm_map:
        return norm_map[n]
    hits = get_close_matches(n, list(norm_map.keys()), n=1, cutoff=cutoff)
    return norm_map[hits[0]] if hits else None


def fetch_upcoming():
    r = requests.get(SGPOOLS_API, params={"lang": "en", "betType": "MR"}, timeout=30)
    r.raise_for_status()
    return r.json().get("events", [])


def score_events(events, models, team_lists):
    rows = []
    for ev in events:
        country = ev["type"]["sportClass"]["name"]
        comp = ev["type"]["name"]
        start_time = ev.get("startTime")

        mkt = next((m for m in ev.get("markets", []) if m.get("name") == "1X2"), None)
        if mkt is None:
            continue
        outcomes = {o["minorCode"]: o for o in mkt.get("outcomes", [])}
        if not all(k in outcomes for k in ("H", "D", "A")):
            continue
        try:
            odds_h = float(outcomes["H"]["prices"][0]["decimal"])
            odds_d = float(outcomes["D"]["prices"][0]["decimal"])
            odds_a = float(outcomes["A"]["prices"][0]["decimal"])
        except (KeyError, IndexError, ValueError):
            continue
        home_name = outcomes["H"]["name"]
        away_name = outcomes["A"]["name"]

        # Strict: only score a fixture whose competition string is explicitly
        # recognized as one of the 13 validated leagues. No fallback search
        # across other leagues' rosters -- see COMPETITION_MAP's docstring
        # for why that's actively dangerous, not just imprecise.
        league_code = COMPETITION_MAP.get((country, comp))
        if league_code is None or league_code not in team_lists:
            continue

        home = match_team(home_name, team_lists[league_code])
        away = match_team(away_name, team_lists[league_code])
        if not home or not away:
            continue
        matched_league = league_code

        probs = models[matched_league].match_probs(home, away)
        if probs is None:
            continue
        mkt_p = implied_prob_devigged([odds_h, odds_d, odds_a])
        for sel, p, mp, odds, disp_name in [
            ("H", probs["1x2"]["H"], mkt_p[0], odds_h, home_name),
            ("D", probs["1x2"]["D"], mkt_p[1], odds_d, "Draw"),
            ("A", probs["1x2"]["A"], mkt_p[2], odds_a, away_name),
        ]:
            rows.append(dict(
                fixture=f"{home_name} vs {away_name}", league=matched_league, country=country,
                competition=comp, start_time=start_time, selection=sel, pick=disp_name,
                model_p=p, market_p=mp, odds=odds,
                qualifies=(p >= PROB_THRESHOLD and (p - mp) >= EDGE_THRESHOLD),
            ))
    cols = ["fixture", "league", "country", "competition", "start_time", "selection",
            "pick", "model_p", "market_p", "odds", "qualifies"]
    return pd.DataFrame(rows, columns=cols)


def render_html(qualifying_df, all_scored_df, generated_at):
    LEAGUE_NAMES = {
        "E0": "England - Premier League", "E1": "England - Championship",
        "D1": "Germany - Bundesliga", "D2": "Germany - 2. Bundesliga",
        "I1": "Italy - Serie A", "SP1": "Spain - La Liga", "F1": "France - Ligue 1",
        "N1": "Netherlands - Eredivisie", "P1": "Portugal - Primeira Liga",
        "SC0": "Scotland - Premiership", "T1": "Turkey - Super Lig",
        "G1": "Greece - Super League", "B1": "Belgium - Pro League",
    }

    def fmt_time(iso_str):
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return dt.strftime("%a %d %b, %H:%M UTC")
        except Exception:
            return iso_str or ""

    def prob_bar(p, hi=1.0):
        pct = max(0, min(100, p / hi * 100))
        return f'<div class="pbar"><div class="pbar-fill" style="width:{pct:.1f}%"></div></div>'

    cards_html = ""
    if len(qualifying_df):
        qualifying_df = qualifying_df.sort_values("model_p", ascending=False)
        for _, r in qualifying_df.iterrows():
            edge = r["model_p"] - r["market_p"]
            cards_html += f"""
        <div class="ticket">
          <div class="ticket-main">
            <div class="ticket-comp">{LEAGUE_NAMES.get(r['league'], r['league'])} &middot; {fmt_time(r['start_time'])}</div>
            <div class="ticket-fixture">{r['fixture']}</div>
            <div class="ticket-pick">Pick: <span>{r['pick']}</span></div>
          </div>
          <div class="ticket-figures">
            <div class="fig">
              <div class="fig-label">Model</div>
              <div class="fig-val model">{r['model_p']:.1%}</div>
              {prob_bar(r['model_p'])}
            </div>
            <div class="fig">
              <div class="fig-label">Market</div>
              <div class="fig-val">{r['market_p']:.1%}</div>
              {prob_bar(r['market_p'])}
            </div>
            <div class="fig">
              <div class="fig-label">Edge</div>
              <div class="fig-val {'pos' if edge >= 0 else 'neg'}">{edge:+.1%}</div>
            </div>
            <div class="fig">
              <div class="fig-label">Odds</div>
              <div class="fig-val odds">{r['odds']:.2f}</div>
            </div>
          </div>
        </div>"""
    else:
        cards_html = """
        <div class="empty">
          <div class="empty-mark">&mdash;</div>
          <div class="empty-title">No pick clears the bar today</div>
          <div class="empty-body">That's the normal state, not a malfunction &mdash; across the
          validated 13 leagues this rule fires roughly once every 3&ndash;4 weeks. The board is
          quiet until a match genuinely clears 90% model probability with real edge over the
          market's own price. Check back tomorrow.</div>
        </div>"""

    n_scanned = all_scored_df["fixture"].nunique() if len(all_scored_df) else 0
    n_qualifying_fixtures = qualifying_df["fixture"].nunique() if len(qualifying_df) else 0

    html = f"""<title>Ninety Percent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;800&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #f2f5f3; --surface: #ffffff; --surface-2: #eaf0ec; --border: #d9e2dc;
  --text: #16221b; --text-dim: #59695f; --text-faint: #8a988e;
  --accent: #9c7a24; --accent-soft: #f4ead0;
  --pos: #2f7d4f; --neg: #a4402f;
  --shadow: 0 1px 2px rgba(22,34,27,0.06), 0 8px 24px -12px rgba(22,34,27,0.12);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #0b1310; --surface: #121c17; --surface-2: #17231d; --border: #263630;
    --text: #e9f1ec; --text-dim: #9fb3a7; --text-faint: #647a6d;
    --accent: #d8ac4a; --accent-soft: #2a2313;
    --pos: #4fae7c; --neg: #d1685c;
    --shadow: none;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #0b1310; --surface: #121c17; --surface-2: #17231d; --border: #263630;
  --text: #e9f1ec; --text-dim: #9fb3a7; --text-faint: #647a6d;
  --accent: #d8ac4a; --accent-soft: #2a2313;
  --pos: #4fae7c; --neg: #d1685c;
  --shadow: none;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg); color: var(--text); margin: 0; padding: 40px 20px 64px;
  font-family: "Public Sans", -apple-system, BlinkMacSystemFont, sans-serif;
}}
.wrap {{ max-width: 760px; margin: 0 auto; display: flex; flex-direction: column; gap: 28px; }}
.masthead {{ display: flex; flex-direction: column; gap: 4px; }}
h1 {{
  font-family: "Archivo", sans-serif; font-weight: 800; font-size: 26px; letter-spacing: -0.01em;
  margin: 0; text-wrap: balance;
}}
h1 .dim {{ color: var(--text-dim); font-weight: 500; }}
.meta {{ color: var(--text-faint); font-size: 13px; font-family: "IBM Plex Mono", monospace; }}

.scoreboard {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
.stat {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px 18px; box-shadow: var(--shadow);
}}
.stat .n {{
  font-family: "IBM Plex Mono", monospace; font-weight: 600; font-size: 28px;
  font-variant-numeric: tabular-nums; line-height: 1;
}}
.stat .l {{ font-size: 12px; color: var(--text-dim); margin-top: 8px; letter-spacing: 0.01em; }}

.rule {{
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px 20px; font-size: 13.5px; color: var(--text-dim); line-height: 1.6;
}}
.rule b {{ color: var(--text); font-weight: 600; }}

.board-label {{
  font-family: "IBM Plex Mono", monospace; font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--text-faint);
}}

.tickets {{ display: flex; flex-direction: column; gap: 10px; }}
.ticket {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  box-shadow: var(--shadow); padding: 18px 20px;
  display: flex; justify-content: space-between; align-items: center; gap: 20px; flex-wrap: wrap;
}}
.ticket-comp {{ font-size: 12px; color: var(--text-faint); font-family: "IBM Plex Mono", monospace; }}
.ticket-fixture {{ font-family: "Archivo", sans-serif; font-weight: 600; font-size: 17px; margin-top: 3px; }}
.ticket-pick {{ font-size: 13px; color: var(--text-dim); margin-top: 6px; }}
.ticket-pick span {{ color: var(--accent); font-weight: 600; }}

.ticket-figures {{ display: flex; gap: 22px; align-items: flex-end; }}
.fig {{ display: flex; flex-direction: column; align-items: flex-end; gap: 4px; min-width: 58px; }}
.fig-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-faint); }}
.fig-val {{ font-family: "IBM Plex Mono", monospace; font-weight: 600; font-size: 16px; font-variant-numeric: tabular-nums; }}
.fig-val.model {{ color: var(--accent); }}
.fig-val.pos {{ color: var(--pos); }}
.fig-val.neg {{ color: var(--neg); }}
.pbar {{ width: 52px; height: 4px; background: var(--surface-2); border-radius: 2px; overflow: hidden; }}
.pbar-fill {{ height: 100%; background: var(--accent); }}

.empty {{
  background: var(--surface); border: 1px dashed var(--border); border-radius: 12px;
  padding: 40px 24px; text-align: center; display: flex; flex-direction: column; align-items: center; gap: 8px;
}}
.empty-mark {{ font-family: "IBM Plex Mono", monospace; font-size: 22px; color: var(--text-faint); }}
.empty-title {{ font-family: "Archivo", sans-serif; font-weight: 600; font-size: 16px; }}
.empty-body {{ color: var(--text-dim); font-size: 13.5px; line-height: 1.6; max-width: 480px; }}

.footer {{ color: var(--text-faint); font-size: 12px; text-align: center; line-height: 1.6; }}
</style>
<div class="wrap">
  <div class="masthead">
    <h1>Ninety Percent <span class="dim">&middot; SG Pools checklist</span></h1>
    <div class="meta">updated {generated_at}</div>
  </div>

  <div class="scoreboard">
    <div class="stat"><div class="n">{n_qualifying_fixtures}</div><div class="l">Qualifying today</div></div>
    <div class="stat"><div class="n">{n_scanned}</div><div class="l">Fixtures scanned</div></div>
    <div class="stat"><div class="n">90%</div><div class="l">Minimum model probability</div></div>
  </div>

  <div class="rule">
    A pick appears only if <b>both</b> hold: the Dixon-Coles model, fit on 10 years of results
    across the 13 validated deep/liquid leagues, puts the outcome at <b>&ge;90% probability</b>,
    and that beats the vig-stripped probability implied by the market's own odds. Backtested hit
    rate at this bar: <b>91.8%</b> on 183 bets, holds up in an out-of-sample check. Confirm the
    exact fixture and price on Singapore Pools yourself &mdash; if it isn't listed there, don't bet it.
  </div>

  <div>
    <div class="board-label" style="margin-bottom:10px;">Today's board</div>
    <div class="tickets">{cards_html}</div>
  </div>

  <div class="footer">Generated from Singapore Pools' own public odds feed.<br>This page never places bets or touches your account.</div>
</div>
"""
    return html


def main():
    print("Fetching Singapore Pools upcoming football odds...")
    events = fetch_upcoming()
    print(f"  {len(events)} events returned")

    print("Loading Dixon-Coles models for the 13 validated leagues...")
    models, team_lists = load_models()
    print(f"  {len(models)} league models loaded")

    seen_competitions = {(ev["type"]["sportClass"]["name"], ev["type"]["name"]) for ev in events}
    unmapped = seen_competitions - set(COMPETITION_MAP.keys()) - KNOWN_EXCLUDED
    if unmapped:
        print(f"  Unrecognized competitions on the board today (skipped, not scored): {sorted(unmapped)}")

    scored = score_events(events, models, team_lists)
    qualifying = scored[scored["qualifies"]].copy() if len(scored) else scored
    print(f"  {scored['fixture'].nunique() if len(scored) else 0} fixtures matched to a validated league")
    print(f"  {len(qualifying)} qualifying (market, selection) rows, "
          f"{qualifying['fixture'].nunique() if len(qualifying) else 0} distinct fixtures")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = render_html(qualifying, scored, generated_at)
    (OUT_DIR / "dashboard.html").write_text(html)

    qualifying.to_json(OUT_DIR / "dashboard_qualifying.json", orient="records", indent=2)
    scored.to_parquet(OUT_DIR / "dashboard_all_scored.parquet", index=False)
    print(f"Saved -> {OUT_DIR / 'dashboard.html'}")

    # Append today's qualifying picks to a running log (same pattern as this repo's other
    # dashboards -- gold_predictions.csv etc. -- so picks can later be checked against what
    # actually happened, not just seen once and forgotten). Dedup on rerun same day.
    log_path = OUT_DIR / "qualifying_log.csv"
    today_log = qualifying.copy()
    today_log.insert(0, "run_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    if log_path.exists():
        prior = pd.read_csv(log_path)
        combined = pd.concat([prior, today_log], ignore_index=True)
        combined = combined.drop_duplicates(subset=["run_date", "fixture", "selection"], keep="last")
    else:
        combined = today_log
    combined.to_csv(log_path, index=False)
    print(f"Saved -> {log_path} ({len(combined)} rows total)")


if __name__ == "__main__":
    main()
