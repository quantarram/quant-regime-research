"""
Daily dashboard generator -- CPE (empirical exceedance) engine, not a fitted
distributional model. Pulls Singapore Pools' own public odds API (no login,
no browser needed -- confirmed live to be a plain unauthenticated JSON
endpoint) and lists every upcoming match where the home team's real trailing
results clear the validated CPE joint condition.

NO Poisson, NO Gaussian, no fitted distribution of any kind anywhere in this
pipeline. Every number here is either a real historical points-per-game
average or a raw percentile cutoff computed directly from real historical
values -- see football_cpe_engine.py / football_cpe_widened.py for how this
was discovered and football_joint_cpe.py for the discovery/holdout check
that validated it:

    ppg_diff            > 0.60   (home team's trailing PPG, any venue, last
                                   10 matches, minus the away team's)
    home_ppg_home_only  > 2.00   (home team's trailing PPG from ONLY its
                                   last 10 home matches)

Both thresholds are the 80th/75th percentile of those statistics' real
historical distribution across the 13 validated leagues -- not fitted, not
assumed, just where the actual numbers happened to fall.

Backtested (this exact rule, chronological discovery/holdout split, real
market odds): 74.0% hit rate, +1.39% ROI on 1,650 holdout bets. This is a
DIRECTIONAL, HOME-WIN-ONLY signal -- it only ever picks the home team, never
draw or away, because that's what the two predictors were built to detect.
It is also a much more frequent signal than a probability-threshold rule
would suggest: about 10% of all matches in these leagues clear it, so expect
multiple qualifying picks most weeks, not one every few weeks.

This produces a list only. It never places a bet, logs into an account, or
touches money.

Usage:
    python daily_dashboard.py
Output:
    output/dashboard.html                 -- the page itself
    output/dashboard_qualifying.json      -- same data, machine-readable
    output/qualifying_log.csv             -- running log, appended daily
"""
import re
import unicodedata
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import requests
import pandas as pd

from data_download import CORE_LEAGUES

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

FORM_WINDOW = 10
PPG_DIFF_THRESHOLD_Q = 0.80
HOME_PPG_THRESHOLD_Q = 0.75
VALIDATED_HIT_RATE = 0.740
VALIDATED_ROI = 0.0139
VALIDATED_HOLDOUT_N = 1650

SGPOOLS_API = "https://api.singaporepools.com/football/events/v1/upcoming-event"

# (country, competition name) as returned by the SG Pools API -> our league code.
# Confirmed live against the API for the rows marked "confirmed"; the rest
# (Portugal/Turkey/Greece/Belgium/Scotland) weren't on the board on the day this
# was built, so they're best-guess names, marked unconfirmed below.
#
# STRICT ON PURPOSE: a competition string not in this dict is SKIPPED, never
# passed to a fuzzy team-name search across all 13 leagues' rosters -- an
# early version did that as a fallback and it produced a live false positive
# (a Dutch second-division match scored with top-flight team history because
# both teams had once played in the Eredivisie). Missing a genuine match
# because a competition name isn't mapped yet is a far smaller cost than
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

KNOWN_EXCLUDED = {
    ("England", "English League One"), ("England", "English League Two"),
    ("Italy", "Italian League Div 2"), ("Spain", "Spanish League Div 2"),
    ("France", "French League Div 2"), ("Netherlands", "Dutch League Div 2"),
}


def compute_current_form_and_thresholds():
    """Pure counting, no model fit: walks all CORE_LEAGUES history chronologically
    and returns each team's CURRENT trailing PPG (any venue, last 10 matches) and
    home-only trailing PPG (last 10 home matches), plus the live percentile
    thresholds computed from the full real historical distribution of both stats."""
    matches = pd.read_parquet(DATA_DIR / "matches.parquet")
    matches = matches[matches["league"].isin(CORE_LEAGUES)].sort_values("date").reset_index(drop=True)

    hist_any, hist_home = {}, {}   # team -> list of real points earned, chronological
    ppg_diff_history, home_ppg_history = [], []  # for threshold computation

    for row in matches.itertuples():
        h, a = row.home_team, row.away_team
        hh, ah = hist_any.get(h, []), hist_any.get(a, [])
        hh_home = hist_home.get(h, [])

        if len(hh) >= FORM_WINDOW and len(ah) >= FORM_WINDOW:
            ppg_diff_history.append(np.mean(hh[-FORM_WINDOW:]) - np.mean(ah[-FORM_WINDOW:]))
        if len(hh_home) >= FORM_WINDOW:
            home_ppg_history.append(np.mean(hh_home[-FORM_WINDOW:]))

        if row.ftr == "H":
            h_pts, a_pts = 3, 0
        elif row.ftr == "A":
            h_pts, a_pts = 0, 3
        else:
            h_pts, a_pts = 1, 1
        hist_any.setdefault(h, []).append(h_pts)
        hist_any.setdefault(a, []).append(a_pts)
        hist_home.setdefault(h, []).append(h_pts)

    current_ppg_any = {t: np.mean(pts[-FORM_WINDOW:]) for t, pts in hist_any.items() if len(pts) >= FORM_WINDOW}
    current_ppg_home_only = {t: np.mean(pts[-FORM_WINDOW:]) for t, pts in hist_home.items() if len(pts) >= FORM_WINDOW}

    ppg_diff_threshold = float(np.quantile(ppg_diff_history, PPG_DIFF_THRESHOLD_Q))
    home_ppg_threshold = float(np.quantile(home_ppg_history, HOME_PPG_THRESHOLD_Q))
    return current_ppg_any, current_ppg_home_only, ppg_diff_threshold, home_ppg_threshold


def _all_team_names():
    matches = pd.read_parquet(DATA_DIR / "matches.parquet")
    matches = matches[matches["league"].isin(CORE_LEAGUES)]
    by_league = {}
    for league, df in matches.groupby("league"):
        by_league[league] = sorted(set(df["home_team"]) | set(df["away_team"]))
    return by_league


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


def score_events(events, team_lists, current_ppg_any, current_ppg_home_only,
                  ppg_diff_threshold, home_ppg_threshold):
    rows = []
    for ev in events:
        country = ev["type"]["sportClass"]["name"]
        comp = ev["type"]["name"]
        start_time = ev.get("startTime")

        mkt = next((m for m in ev.get("markets", []) if m.get("name") == "1X2"), None)
        if mkt is None:
            continue
        outcomes = {o["minorCode"]: o for o in mkt.get("outcomes", [])}
        if "H" not in outcomes:
            continue
        try:
            odds_h = float(outcomes["H"]["prices"][0]["decimal"])
        except (KeyError, IndexError, ValueError):
            odds_h = None
        home_name = outcomes["H"]["name"]
        away_name = outcomes.get("A", {}).get("name", "?")

        league_code = COMPETITION_MAP.get((country, comp))
        if league_code is None or league_code not in team_lists:
            continue
        home = match_team(home_name, team_lists[league_code])
        away = match_team(away_name, team_lists[league_code])
        if not home or not away:
            continue

        ppg_diff = current_ppg_any.get(home)
        away_ppg = current_ppg_any.get(away)
        home_only = current_ppg_home_only.get(home)
        if ppg_diff is None or away_ppg is None or home_only is None:
            continue
        ppg_diff = ppg_diff - away_ppg

        qualifies = (ppg_diff > ppg_diff_threshold) and (home_only > home_ppg_threshold)

        rows.append(dict(
            fixture=f"{home_name} vs {away_name}", league=league_code, country=country,
            competition=comp, start_time=start_time, pick=home_name,
            ppg_diff=ppg_diff, home_ppg_home_only=home_only,
            ppg_diff_threshold=ppg_diff_threshold, home_ppg_threshold=home_ppg_threshold,
            odds=odds_h, qualifies=qualifies,
        ))
    cols = ["fixture", "league", "country", "competition", "start_time", "pick",
            "ppg_diff", "home_ppg_home_only", "ppg_diff_threshold", "home_ppg_threshold",
            "odds", "qualifies"]
    return pd.DataFrame(rows, columns=cols)


def render_html(qualifying_df, all_scored_df, generated_at, ppg_diff_threshold, home_ppg_threshold):
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

    def margin_bar(val, threshold, span):
        pct = max(0, min(100, (val - threshold) / span * 100 + 50))
        return f'<div class="pbar"><div class="pbar-fill" style="width:{pct:.1f}%"></div></div>'

    cards_html = ""
    if len(qualifying_df):
        qualifying_df = qualifying_df.sort_values("ppg_diff", ascending=False)
        for _, r in qualifying_df.iterrows():
            odds_str = f"{r['odds']:.2f}" if pd.notna(r["odds"]) else "&mdash;"
            cards_html += f"""
        <div class="ticket">
          <div class="ticket-main">
            <div class="ticket-comp">{LEAGUE_NAMES.get(r['league'], r['league'])} &middot; {fmt_time(r['start_time'])}</div>
            <div class="ticket-fixture">{r['fixture']}</div>
            <div class="ticket-pick">Pick: <span>{r['pick']}</span> to win</div>
          </div>
          <div class="ticket-figures">
            <div class="fig">
              <div class="fig-label">Form edge</div>
              <div class="fig-val model">{r['ppg_diff']:+.2f}</div>
              {margin_bar(r['ppg_diff'], r['ppg_diff_threshold'], 2.0)}
            </div>
            <div class="fig">
              <div class="fig-label">Home PPG</div>
              <div class="fig-val model">{r['home_ppg_home_only']:.2f}</div>
              {margin_bar(r['home_ppg_home_only'], r['home_ppg_threshold'], 1.5)}
            </div>
            <div class="fig">
              <div class="fig-label">Odds</div>
              <div class="fig-val odds">{odds_str}</div>
            </div>
          </div>
        </div>"""
    else:
        cards_html = """
        <div class="empty">
          <div class="empty-mark">&mdash;</div>
          <div class="empty-title">No pick clears the bar today</div>
          <div class="empty-body">Unusual, but not wrong &mdash; this condition typically clears
          on roughly 1 in 10 matches across the validated 13 leagues, so most days should show
          at least one pick. Check back tomorrow, or verify the data refreshed correctly if this
          persists for several days running.</div>
        </div>"""

    n_scanned = all_scored_df["fixture"].nunique() if len(all_scored_df) else 0
    n_qualifying_fixtures = qualifying_df["fixture"].nunique() if len(qualifying_df) else 0

    html = f"""<title>Form Edge</title>
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
.rule code {{
  font-family: "IBM Plex Mono", monospace; background: var(--surface); border: 1px solid var(--border);
  border-radius: 4px; padding: 1px 5px; font-size: 12.5px; color: var(--text);
}}

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
.fig {{ display: flex; flex-direction: column; align-items: flex-end; gap: 4px; min-width: 62px; }}
.fig-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-faint); }}
.fig-val {{ font-family: "IBM Plex Mono", monospace; font-weight: 600; font-size: 16px; font-variant-numeric: tabular-nums; }}
.fig-val.model {{ color: var(--accent); }}
.pbar {{ width: 56px; height: 4px; background: var(--surface-2); border-radius: 2px; overflow: hidden; }}
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
    <h1>Form Edge <span class="dim">&middot; SG Pools checklist</span></h1>
    <div class="meta">updated {generated_at}</div>
  </div>

  <div class="scoreboard">
    <div class="stat"><div class="n">{n_qualifying_fixtures}</div><div class="l">Qualifying today</div></div>
    <div class="stat"><div class="n">{n_scanned}</div><div class="l">Fixtures scanned</div></div>
    <div class="stat"><div class="n">{VALIDATED_HIT_RATE:.0%}</div><div class="l">Validated historical hit rate</div></div>
  </div>

  <div class="rule">
    Pure empirical counting &mdash; no fitted distribution, Poisson or otherwise. A pick appears
    only when <b>both</b> real historical conditions hold for the home team, over its last 10
    matches: <code>trailing PPG edge &gt; {ppg_diff_threshold:.2f}</code>
    and <code>home-only trailing PPG &gt; {home_ppg_threshold:.2f}</code>
    &mdash; both are the 80th/75th percentile of those stats' real historical distribution, not
    fitted values. Backtested on a real chronological holdout (never seen when the thresholds were
    set): <b>{VALIDATED_HIT_RATE:.1%}</b> hit rate, <b>{VALIDATED_ROI:+.1%}</b> ROI on
    {VALIDATED_HOLDOUT_N:,} bets. This fires far more often than a 90%-confidence rule would
    (~1 in 10 matches) &mdash; expect several picks most weeks, not one every few weeks. Confirm
    the fixture and price on Singapore Pools yourself &mdash; if it isn't listed there, don't bet it.
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
    print("Computing current team form from real historical results (no model fit)...")
    current_ppg_any, current_ppg_home_only, ppg_diff_threshold, home_ppg_threshold = compute_current_form_and_thresholds()
    print(f"  ppg_diff threshold (80th pct, real history): {ppg_diff_threshold:.2f}")
    print(f"  home_ppg_home_only threshold (75th pct, real history): {home_ppg_threshold:.2f}")

    team_lists = _all_team_names()

    print("Fetching Singapore Pools upcoming football odds...")
    events = fetch_upcoming()
    print(f"  {len(events)} events returned")

    seen_competitions = {(ev["type"]["sportClass"]["name"], ev["type"]["name"]) for ev in events}
    unmapped = seen_competitions - set(COMPETITION_MAP.keys()) - KNOWN_EXCLUDED
    if unmapped:
        print(f"  Unrecognized competitions on the board today (skipped, not scored): {sorted(unmapped)}")

    scored = score_events(events, team_lists, current_ppg_any, current_ppg_home_only,
                           ppg_diff_threshold, home_ppg_threshold)
    qualifying = scored[scored["qualifies"]].copy() if len(scored) else scored
    print(f"  {scored['fixture'].nunique() if len(scored) else 0} fixtures matched to a validated league")
    print(f"  {len(qualifying)} qualifying fixtures")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = render_html(qualifying, scored, generated_at, ppg_diff_threshold, home_ppg_threshold)
    (OUT_DIR / "dashboard.html").write_text(html)

    qualifying.to_json(OUT_DIR / "dashboard_qualifying.json", orient="records", indent=2)
    scored.to_parquet(OUT_DIR / "dashboard_all_scored.parquet", index=False)
    print(f"Saved -> {OUT_DIR / 'dashboard.html'}")

    log_path = OUT_DIR / "qualifying_log.csv"
    today_log = qualifying.copy()
    today_log.insert(0, "run_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    if log_path.exists():
        prior = pd.read_csv(log_path)
        combined = pd.concat([prior, today_log], ignore_index=True)
        combined = combined.drop_duplicates(subset=["run_date", "fixture"], keep="last")
    else:
        combined = today_log
    combined.to_csv(log_path, index=False)
    print(f"Saved -> {log_path} ({len(combined)} rows total)")


if __name__ == "__main__":
    main()
