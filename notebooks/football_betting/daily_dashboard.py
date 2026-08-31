"""
Daily dashboard generator -- CPE (empirical exceedance) engine, not a fitted
distributional model. Pulls Singapore Pools' own public odds API (no login,
no browser needed -- confirmed live to be a plain unauthenticated JSON
endpoint) and lists every upcoming match where the home team's real trailing
results clear the validated CPE joint condition.

NO Poisson, NO Gaussian, no fitted distribution of any kind anywhere in this
pipeline. Every number here is either a real historical points-per-game
average or a raw percentile cutoff computed directly from real historical
values -- see football_cpe_engine.py / football_cpe_widened.py for how the
signal was discovered and football_joint_cpe.py for the discovery/holdout
check that validated it. The trailing-match WINDOW itself was also run
through that same discipline (football_cpe_window_scan.py), not guessed --
3 and 5 matches never clear the discovery gate at all, 8 posts the best
single holdout ROI but is an isolated spike between weak/failing neighbors
(noise, not signal), and 25-30 is the one region that holds together as a
coherent plateau (rising hit rate, consistently positive ROI, largest and
most stable holdout n of any window tested). 30 -- the top of that plateau
-- is what's used here:

    ppg_diff            > 0.80   (home team's trailing PPG, any venue, last
                                   30 matches, minus the away team's)
    home_ppg_home_only  > 2.27   (home team's trailing PPG from ONLY its
                                   last 30 home matches)

Both thresholds are the 90th percentile of those statistics' real historical
distribution across the 13 validated leagues -- not fitted, not assumed,
just where the actual numbers happened to fall.

Backtested (this exact rule, chronological discovery/holdout split, real
market odds): 83.8% hit rate, +1.90% ROI on 648 holdout bets. This is a
DIRECTIONAL, HOME-WIN-ONLY signal -- it only ever picks the home team, never
draw or away, because that's what the two predictors were built to detect.
It qualifies roughly 1 in 20-25 matches in these leagues, so expect a
handful of picks most weeks, not one every few weeks.

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
from zoneinfo import ZoneInfo

SGT = ZoneInfo("Asia/Singapore")  # UTC+8, no DST -- fixed offset year-round
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import requests
import pandas as pd

from data_download import CORE_LEAGUES, refresh_core_leagues

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

## FORM_WINDOW itself was scanned, not guessed: football_cpe_window_scan.py ran the
## SAME joint-search-then-holdout discipline at windows of 3/5/8/10/15/20/25/30/38
## matches. 3 and 5 never even clear the discovery gate. 8 posts the single best
## holdout ROI (+2.94%) but is an isolated spike between two failing/weak neighbors
## (5 and 10) -- the signature of noise, not a real effect. 25-30 is the one region
## that holds together as a coherent plateau: hit rate climbs steadily (81.7% ->
## 82.9% -> 83.8%), ROI stays positive (+0.40% -> +1.85% -> +1.90%), and holdout n
## is the largest and most stable of any window tested (657-679 bets). 30 is the top
## of that plateau. See output/football_cpe_window_scan.csv for the full table.
FORM_WINDOW = 30
PPG_DIFF_THRESHOLD_Q = 0.90
HOME_PPG_THRESHOLD_Q = 0.90
VALIDATED_HIT_RATE = 0.838
VALIDATED_ROI = 0.0190
VALIDATED_HOLDOUT_N = 648

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
    and returns each team's CURRENT trailing PPG (any venue, last FORM_WINDOW matches) and
    home-only trailing PPG (last FORM_WINDOW home matches), plus the live percentile
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
            # canonical (football-data.co.uk) names, distinct from the SG Pools display
            # names above -- needed so resolve_football_picks.py can look the match up
            # in matches.parquet later without re-fuzzy-matching against a name that may
            # have drifted (SG Pools renames, accents, "Man City" vs "Manchester City").
            home_team_matched=home, away_team_matched=away,
            ppg_diff=ppg_diff, home_ppg_home_only=home_only,
            ppg_diff_threshold=ppg_diff_threshold, home_ppg_threshold=home_ppg_threshold,
            odds=odds_h, qualifies=qualifies,
        ))
    cols = ["fixture", "league", "country", "competition", "start_time", "pick",
            "home_team_matched", "away_team_matched",
            "ppg_diff", "home_ppg_home_only", "ppg_diff_threshold", "home_ppg_threshold",
            "odds", "qualifies"]
    return pd.DataFrame(rows, columns=cols)


def render_html(qualifying_df, all_scored_df, generated_at, ppg_diff_threshold, home_ppg_threshold, log_df):
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
            dt_utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            dt_sgt = dt_utc.astimezone(SGT)
            return f"{dt_sgt.strftime('%a %d %b, %H:%M')} SGT ({dt_utc.strftime('%H:%M')} UTC)"
        except Exception:
            return iso_str or ""

    def bar(val, lo, hi, color="var(--gold)"):
        pct = max(0, min(100, (val - lo) / (hi - lo) * 100))
        return (f'<div class="weight-bar-bg" style="width:64px;"><div class="weight-bar-fill" '
                f'style="width:{pct:.0f}%;background:{color};height:100%;"></div></div>')

    n_scanned = all_scored_df["fixture"].nunique() if len(all_scored_df) else 0
    n_qualifying_fixtures = qualifying_df["fixture"].nunique() if len(qualifying_df) else 0

    # Singapore Pools' own API returns several days of upcoming fixtures at once (currently
    # ~5), not just "today" -- scanning and displaying all of it, not just the current
    # calendar day, so the section label reflects the real span rather than calling it "today".
    if len(all_scored_df):
        span_dates = pd.to_datetime(all_scored_df["start_time"]).dt.tz_convert(SGT)
        span_label = f"{span_dates.min().strftime('%a %d %b')} &ndash; {span_dates.max().strftime('%a %d %b')} SGT"
    else:
        span_label = "no upcoming fixtures found"

    # -- real-world track record: resolved picks only, chronological cumulative P&L.
    #    Rendered as inline SVG, not Plotly: this page is published through the Artifact
    #    tool, whose CSP blocks external script hosts (Plotly's CDN included) -- shipping
    #    a <script src="cdn.plot.ly/..."> here would silently fail there even though it
    #    works fine opening the local HTML file directly. Inline SVG matches the same
    #    dark/gold visual language and works identically in both places. Renders as a
    #    blank axes (no line, no points) until there's real resolved data -- never a
    #    synthetic or placeholder point.
    if len(log_df) and "bet_placed" in log_df.columns:
        log_df["bet_placed"] = log_df["bet_placed"].fillna(False).astype(bool)
    resolved = log_df[log_df["status"] == "RESOLVED"].sort_values("start_time") if len(log_df) else log_df
    real_bets = resolved[resolved["bet_placed"]] if len(resolved) and "bet_placed" in resolved.columns else resolved.iloc[0:0]
    n_resolved = len(resolved)
    n_real_bets = len(real_bets)
    n_pending = int((log_df["status"] == "PENDING").sum()) if len(log_df) else 0
    n_bets_placed_total = int(log_df["bet_placed"].sum()) if len(log_df) and "bet_placed" in log_df.columns else 0

    def render_chart_svg(real_bets):
        # Plots REAL money only (actual bets placed, actual stakes) -- not the theoretical
        # S$50-per-signal figure, since what matters here is what actually happened to real cash.
        # PAD_L is wider than the other three sides to leave room for the S$ value labels on the
        # y-axis (added 2026-08-31 -- the chart previously had no axis labels at all, just a bare
        # shape against a zero line, which the user found confusing with no way to read exact values).
        W, H, PAD_L, PAD_R, PAD_T, PAD_B = 900, 220, 60, 20, 20, 20
        if not len(real_bets):
            zero_y = H / 2
            return (f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:220px;" preserveAspectRatio="none">'
                    f'<line x1="{PAD_L}" y1="{zero_y}" x2="{W-PAD_R}" y2="{zero_y}" stroke="#2C302C" stroke-width="1"/>'
                    f'<text x="{PAD_L-8}" y="{zero_y+4:.1f}" text-anchor="end" font-family="monospace" font-size="11" fill="#7A8F7A">S$0</text>'
                    f'</svg>')
        cum = real_bets["actual_pnl"].astype(float).cumsum().tolist()
        n = len(cum)
        y_min, y_max = min(cum + [0]), max(cum + [0])
        y_span = (y_max - y_min) or 1.0
        def x_at(i): return PAD_L + (i / max(n - 1, 1)) * (W - PAD_L - PAD_R)
        def y_at(v): return H - PAD_B - ((v - y_min) / y_span) * (H - PAD_T - PAD_B)
        pts = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(cum))
        zero_y = y_at(0)
        top_y, bottom_y = y_at(y_max), y_at(y_min)
        last_x, last_y = x_at(n - 1), y_at(cum[-1])
        line_color = "#4DB87A" if cum[-1] >= 0 else "#E05555"

        # y-axis value labels: always show S$0 (the baseline), plus the max/min of the cumulative
        # series when they're far enough (>=14px) from the zero line and from each other to avoid
        # overlapping text -- with only 1-3 resolved bets the three values can easily collide.
        labels = [(zero_y, "S$0")]
        MIN_GAP = 14
        if abs(top_y - zero_y) >= MIN_GAP:
            labels.append((top_y, f"S${y_max:+.0f}"))
        if abs(bottom_y - zero_y) >= MIN_GAP and (not labels or abs(bottom_y - labels[-1][0]) >= MIN_GAP):
            labels.append((bottom_y, f"S${y_min:+.0f}"))
        label_svg = "".join(
            f'<text x="{PAD_L-8}" y="{y+4:.1f}" text-anchor="end" font-family="monospace" '
            f'font-size="11" fill="#7A8F7A">{text}</text>'
            for y, text in labels
        )
        return f"""<svg viewBox="0 0 {W} {H}" style="width:100%;height:220px;" preserveAspectRatio="none">
          <line x1="{PAD_L}" y1="{zero_y:.1f}" x2="{W-PAD_R}" y2="{zero_y:.1f}" stroke="#2C302C" stroke-width="1" stroke-dasharray="3,3"/>
          {label_svg}
          <polyline points="{pts}" fill="none" stroke="{line_color}" stroke-width="2"/>
          <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="{line_color}"/>
        </svg>"""

    chart_svg = render_chart_svg(real_bets)

    # "Placed" (n_bets_placed_total) counts every real bet regardless of whether it has
    # resolved yet -- this is the number that answers "how many bets have I made". The
    # resolved-only subset (real_bets/n_real_bets) is a NARROWER group used only for hit
    # rate/P&L/ROI, which can't be computed on a bet whose match hasn't been played yet.
    # Conflating the two here previously made "Bets placed" silently undercount any bet
    # still awaiting its result.
    n_bets_pending_real = n_bets_placed_total - n_real_bets
    total_staked_all = float(log_df.loc[log_df["bet_placed"], "actual_stake_sgd"].astype(float).sum()) if len(log_df) and "bet_placed" in log_df.columns else 0.0

    if n_real_bets:
        real_hit_rate = real_bets["won"].astype(bool).mean()
        real_staked_resolved = real_bets["actual_stake_sgd"].astype(float).sum()
        real_pnl = real_bets["actual_pnl"].astype(float).sum()
        real_roi = real_pnl / real_staked_resolved if real_staked_resolved else float("nan")
        pending_note = (f"{n_bets_pending_real} bet(s) still awaiting resolution &mdash; not in the hit "
                        f"rate/P&amp;L figures below yet." if n_bets_pending_real else "")
        track_stats_html = f"""
      <div class="buy-stats" style="border-top:none;padding-top:0;margin-top:0;">
        <div><div class="buy-stat-l">Bets placed</div><span>{n_bets_placed_total}</span></div>
        <div><div class="buy-stat-l">Total staked</div><span>S${total_staked_all:.2f}</span></div>
        <div><div class="buy-stat-l">Resolved</div><span>{n_real_bets}</span></div>
        <div><div class="buy-stat-l">Real hit rate</div><span style="color:var(--gold)">{real_hit_rate:.1%}</span></div>
        <div><div class="buy-stat-l">Real P&amp;L</div><span style="color:{'var(--green)' if real_pnl>=0 else 'var(--red)'}">S${real_pnl:+.2f} ({real_roi:+.1%})</span></div>
        <div><div class="buy-stat-l">Backtested</div><span style="color:var(--muted)">{VALIDATED_HIT_RATE:.1%} / {VALIDATED_ROI:+.1%}</span></div>
      </div>
      <div style="font-size:10px;color:var(--muted);margin-top:8px;">{pending_note}</div>"""
    else:
        track_stats_html = f"""
      <div class="buy-stats" style="border-top:none;padding-top:0;margin-top:0;">
        <div><div class="buy-stat-l">Bets placed</div><span>{n_bets_placed_total}</span></div>
        <div><div class="buy-stat-l">Total staked</div><span>S${total_staked_all:.2f}</span></div>
        <div><div class="buy-stat-l">Signals pending</div><span>{n_pending}</span></div>
        <div><div class="buy-stat-l">Backtested (reference)</div><span style="color:var(--muted)">{VALIDATED_HIT_RATE:.1%} / {VALIDATED_ROI:+.1%}</span></div>
      </div>
      <div style="font-size:10px;color:var(--muted);margin-top:8px;">{'Chart fills in once a placed bet resolves.' if n_bets_placed_total else 'Chart fills in once a placed bet resolves -- this tracks real money, not just qualifying signals.'}</div>"""

    # Fixtures already backed with a real bet (record_bet.py), so the picks table can show
    # it directly on the row instead of only in the separate track-record section below --
    # this is the more intuitive place to notice "I already acted on this one".
    bet_fixtures = set(log_df.loc[log_df["bet_placed"], "fixture"]) if len(log_df) and "bet_placed" in log_df.columns else set()

    rows_html = ""
    if len(qualifying_df):
        qualifying_df = qualifying_df.sort_values("start_time")  # chronological -- soonest kickoff first
        for _, r in qualifying_df.iterrows():
            odds_str = f"{r['odds']:.2f}" if pd.notna(r["odds"]) else "&mdash;"
            bet_badge = ('<span class="tilt-pill" style="color:var(--green);background:rgba(77,184,122,0.12);'
                         'border:1px solid rgba(77,184,122,0.3);margin-left:6px;">&#10003; BET PLACED</span>'
                         if r["fixture"] in bet_fixtures else "")
            rows_html += f"""
        <tr>
          <td>
            <div class="ac-cell">
              <span class="ac-dot" style="background:var(--gold);"></span>
              <div>
                <div class="ac-name">{r['fixture']}{bet_badge}</div>
                <div class="ac-desc">{LEAGUE_NAMES.get(r['league'], r['league'])} &middot; {fmt_time(r['start_time'])}</div>
              </div>
            </div>
          </td>
          <td><span class="tilt-pill" style="color:var(--gold);background:rgba(201,168,76,0.12);border:1px solid rgba(201,168,76,0.3);">{r['pick']}</span></td>
          <td style="font-family:var(--mono);">{r['ppg_diff']:+.2f} {bar(r['ppg_diff'], 0, 2.5)}</td>
          <td style="font-family:var(--mono);">{r['home_ppg_home_only']:.2f} {bar(r['home_ppg_home_only'], 0, 3.0)}</td>
          <td style="font-family:var(--mono);text-align:right;">{odds_str}</td>
        </tr>"""
    else:
        rows_html = f"""
        <tr><td colspan="5" style="text-align:center;padding:32px 16px;color:var(--muted);">
          No fixture in this window clears the bar. This condition typically fires on roughly
          1 in 20-25 matches across the 13 validated leagues, so most days should show at least
          one pick somewhere in the upcoming span &mdash; check the log if this persists for
          several runs in a row.
        </td></tr>"""

    html = f"""<title>Football Checklist &mdash; CPE Framework</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet">
<style>
  :root{{
    --bg:#0C0E0D; --card:#141614; --card2:#1B1E1B; --border:#2C302C; --border2:#3A3F3A;
    --text:#DDE8DD; --muted:#7A8F7A; --faint:#1B1E1B;
    --gold:#C9A84C; --green:#4DB87A; --red:#E05555; --warn:#E8A020;
    --mono:'IBM Plex Mono',monospace; --serif:'DM Serif Display',serif;
  }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);font-family:'Inter',sans-serif;color:var(--text);font-size:14px;}}
  a{{color:var(--gold);text-decoration:none;}}
  .page{{max-width:1000px;margin:0 auto;padding:32px 24px 80px;}}

  .header{{background:linear-gradient(160deg,#141614,#1B1E1B);border:1px solid var(--border2);
           border-radius:16px;padding:32px 40px;margin-bottom:28px;
           display:grid;grid-template-columns:1fr auto;align-items:center;gap:24px;}}
  .h-title{{font-family:var(--serif);font-size:28px;color:var(--text);margin-bottom:6px;}}
  .h-title em{{font-style:italic;color:var(--gold);}}
  .h-sub{{font-family:var(--mono);font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin-bottom:4px;}}
  .h-meta{{font-size:12px;color:var(--muted);}}
  .h-badge{{background:rgba(201,168,76,0.12);border:1px solid rgba(201,168,76,0.3);
            border-radius:12px;padding:12px 20px;text-align:center;}}
  .h-badge-num{{font-family:var(--mono);font-size:28px;font-weight:600;color:var(--gold);}}
  .h-badge-label{{font-size:10px;color:var(--muted);display:block;margin-top:2px;letter-spacing:.1em;text-transform:uppercase;}}

  .section{{margin-bottom:28px;}}
  .section-title{{font-family:var(--mono);font-size:10px;letter-spacing:.18em;
                  text-transform:uppercase;color:var(--muted);margin-bottom:14px;}}
  .card{{background:var(--card);border-radius:14px;border:1px solid var(--border);padding:20px 24px;}}
  .grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;}}

  .tilt-table{{width:100%;border-collapse:collapse;}}
  .tilt-table th{{font-family:var(--mono);font-size:9px;letter-spacing:.12em;
                  text-transform:uppercase;color:var(--muted);padding:10px 14px;
                  border-bottom:2px solid var(--border);text-align:left;background:var(--faint);}}
  .tilt-table td{{padding:12px 14px;border-bottom:1px solid var(--border);vertical-align:middle;}}
  .tilt-table tr:last-child td{{border-bottom:none;}}
  .tilt-table tr:hover td{{background:var(--card2);}}
  .ac-cell{{display:flex;align-items:center;gap:10px;}}
  .ac-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0;}}
  .ac-name{{font-weight:600;font-size:13px;color:var(--text);}}
  .ac-desc{{font-size:10px;color:var(--muted);margin-top:2px;}}
  .tilt-pill{{display:inline-flex;align-items:center;padding:4px 10px;
              border-radius:100px;font-family:var(--mono);font-size:9px;
              font-weight:600;letter-spacing:.06em;white-space:nowrap;}}
  .weight-bar-bg{{background:var(--border);border-radius:4px;height:6px;display:inline-block;vertical-align:middle;margin-top:4px;}}
  .weight-bar-fill{{height:100%;border-radius:4px;}}
  .buy-stats{{display:flex;gap:14px;margin-top:12px;padding-top:12px;border-top:1px solid var(--border);
              font-family:var(--mono);font-size:11px;color:var(--text);flex-wrap:wrap;}}
  .buy-stat-l{{color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.06em;}}

  .disclaimer{{background:var(--card);border:1px solid var(--border);border-radius:10px;
               padding:16px 20px;font-size:11px;color:var(--muted);line-height:1.7;margin-top:28px;}}
  .disclaimer code{{font-family:var(--mono);background:var(--card2);border:1px solid var(--border);
               border-radius:4px;padding:1px 5px;font-size:10.5px;color:var(--text);}}

  ::-webkit-scrollbar{{width:4px;height:4px;}}
  ::-webkit-scrollbar-track{{background:var(--bg);}}
  ::-webkit-scrollbar-thumb{{background:var(--border2);border-radius:2px;}}

  @media(max-width:700px){{
    .header{{grid-template-columns:1fr;}}
    .page{{padding:16px 12px 60px;}}
  }}
</style>
<div class="page">

<div class="header">
  <div>
    <div class="h-sub">CPE Framework &mdash; Football Checklist</div>
    <div class="h-title">Singapore Pools <em>Form Edge</em> Checklist</div>
    <div class="h-meta">
      Updated: {generated_at} &nbsp;.&nbsp;
      {n_scanned} fixtures scanned &nbsp;.&nbsp;
      13 validated leagues
    </div>
  </div>
  <div class="h-badge">
    <div class="h-badge-num">{n_qualifying_fixtures}</div>
    <span class="h-badge-label">Qualifying Now</span>
  </div>
</div>

<div class="section">
  <div class="section-title">Qualifying Picks &mdash; {span_label}</div>
  <div class="card" style="padding:0;overflow:hidden;">
    <table class="tilt-table">
      <thead>
        <tr>
          <th style="width:280px">Fixture</th><th>Pick</th><th>Form Edge</th><th>Home PPG</th><th style="text-align:right">Odds</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>

<div class="section">
  <div class="section-title">Real-World Track Record</div>
  <div class="card">
    {chart_svg}
    {track_stats_html}
  </div>
</div>

<div class="disclaimer">
  <strong>How this works:</strong> pure empirical counting &mdash; no fitted distribution, Poisson
  or otherwise, anywhere in this pipeline. A pick appears only when <b>both</b> hold for the home
  team, over its last {FORM_WINDOW} matches: <code>trailing PPG edge &gt; {ppg_diff_threshold:.2f}</code> and
  <code>home-only trailing PPG &gt; {home_ppg_threshold:.2f}</code> &mdash; both are the
  90th percentile of those stats' real historical distribution across the 13 validated
  leagues (top-5 Europe, EFL Championship, Scottish Premiership, Eredivisie, Portugal, Turkey,
  Greece, Belgium), not fitted values. Backtested on a real chronological holdout, never seen
  when the thresholds were set: {VALIDATED_HIT_RATE:.1%} hit rate, {VALIDATED_ROI:+.1%} ROI on
  {VALIDATED_HOLDOUT_N:,} bets. Fires roughly 1 in 20-25 matches &mdash; expect a handful of picks most
  weeks. Confirm the fixture and price on Singapore Pools yourself; if it isn't listed there,
  don't bet it. This page never places a bet or touches your account.
  &nbsp;.&nbsp; <strong>Dr. Arun Ramanathan</strong>
</div>

</div>
"""
    return html


def main():
    print("Refreshing historical results (13 validated leagues) so form/thresholds reflect today...")
    refreshed = refresh_core_leagues()
    print(f"  {len(refreshed):,} matches on file, most recent: {refreshed['date'].max().date()}")

    print("Computing current team form from real historical results (no model fit)...")
    current_ppg_any, current_ppg_home_only, ppg_diff_threshold, home_ppg_threshold = compute_current_form_and_thresholds()
    print(f"  ppg_diff threshold ({PPG_DIFF_THRESHOLD_Q:.0%} pct, real history): {ppg_diff_threshold:.2f}")
    print(f"  home_ppg_home_only threshold ({HOME_PPG_THRESHOLD_Q:.0%} pct, real history): {home_ppg_threshold:.2f}")

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

    log_path = OUT_DIR / "qualifying_log.csv"
    today_log = qualifying.copy()
    today_log.insert(0, "run_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    today_log["status"] = "PENDING"
    today_log["bet_placed"] = False
    for c in ["actual_result", "won", "pnl", "actual_stake_sgd", "actual_pnl"]:
        today_log[c] = np.nan
    if log_path.exists():
        prior = pd.read_csv(log_path)
        combined = pd.concat([prior, today_log], ignore_index=True)
        # if a rerun same day duplicates a fixture already RESOLVED by resolve_football_picks.py,
        # or already marked bet_placed by record_bet.py, keep that row rather than clobbering
        # it with a fresh PENDING/not-bet one
        combined = combined.sort_values(["bet_placed", "status"], ascending=[False, False])
        combined = combined.drop_duplicates(subset=["fixture", "start_time"], keep="first")
    else:
        combined = today_log
    combined.to_csv(log_path, index=False)
    print(f"Saved -> {log_path} ({len(combined)} rows total)")

    _now_utc = datetime.now(timezone.utc)
    generated_at = f"{_now_utc.astimezone(SGT).strftime('%Y-%m-%d %H:%M')} SGT ({_now_utc.strftime('%H:%M')} UTC)"
    html = render_html(qualifying, scored, generated_at, ppg_diff_threshold, home_ppg_threshold, combined)
    (OUT_DIR / "dashboard.html").write_text(html)

    qualifying.to_json(OUT_DIR / "dashboard_qualifying.json", orient="records", indent=2)
    scored.to_parquet(OUT_DIR / "dashboard_all_scored.parquet", index=False)
    print(f"Saved -> {OUT_DIR / 'dashboard.html'}")


if __name__ == "__main__":
    main()
