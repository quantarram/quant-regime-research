"""
Odds-by-daily-run charts for qualifying football fixtures
==========================================================
One inline-SVG chart per qualifying fixture on the football dashboard: the home-win odds
Singapore Pools quoted on each DAILY pipeline run (odds_history.csv, one reading per
fixture per day -- see odds_snapshot.py), from the first run that saw the fixture up to
1 day before kickoff (CHART_CUTOFF_H; later readings stay in the CSV but are not
charted), with the odds actually obtained drawn as a green dashed line if a bet was placed
(record_bet.py -> actual_odds / actual_stake_sgd in qualifying_log.csv).

Reading it: if the daily-run line sits ABOVE the green line, an earlier or later daily
price paid more than the bet did; below it, the bet beat those prices. Raw levels, the
run-by-run odds, and the number of daily runs are always shown -- with few runs the card
says so instead of implying a trend.

Used by daily_dashboard.render_html via odds_section_html().
"""
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

OUT_DIR = Path(__file__).parent / "output"
SGT = ZoneInfo("Asia/Singapore")
SHOW_AFTER_KICKOFF_H = 36   # keep a finished fixture's chart visible for a day and a half
CHART_CUTOFF_H = 24         # chart runs "until 1 day before the game"


def _fmt_sgt(ts):
    return ts.astimezone(SGT).strftime("%a %d %b %H:%M SGT")


def load_history(path=None):
    path = Path(path) if path else OUT_DIR / "odds_history.csv"
    if not path.exists():
        return pd.DataFrame()
    h = pd.read_csv(path)
    h["snapshot_utc"] = pd.to_datetime(h["snapshot_utc"], utc=True)
    h["start_time"] = pd.to_datetime(h["start_time"], utc=True)
    return h


def fixture_chart_svg(series, cutoff, now, bet_odds=None):
    """series: DataFrame (snapshot_utc tz-aware, odds_h), already limited to <= cutoff."""
    W, H, PL, PR, PT, PB = 560, 200, 46, 16, 14, 30
    pts = series.sort_values("snapshot_utc")
    t0 = pts["snapshot_utc"].min()
    t1 = cutoff
    if (t1 - t0) < timedelta(hours=12):
        t0 = t1 - timedelta(hours=12)
    span = (t1 - t0).total_seconds()

    def x_at(t):
        return PL + (t - t0).total_seconds() / span * (W - PL - PR)

    vals = list(pts["odds_h"]) + ([bet_odds] if bet_odds else [])
    lo, hi = min(vals), max(vals)
    pad = max(0.02, 0.2 * (hi - lo))
    lo, hi = lo - pad, hi + pad
    if hi - lo < 0.06:
        mid = (hi + lo) / 2
        lo, hi = mid - 0.03, mid + 0.03

    def y_at(v):
        return H - PB - (v - lo) / (hi - lo) * (H - PT - PB)

    p = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;max-width:640px;" role="img" '
         f'aria-label="Home-win odds on each daily run">']
    for v in (lo, (lo + hi) / 2, hi):
        y = y_at(v)
        p.append(f'<line x1="{PL}" x2="{W-PR}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--border)" stroke-width="1"/>')
        p.append(f'<text x="{PL-6}" y="{y+3:.1f}" text-anchor="end" font-size="10" fill="var(--muted)" '
                 f'font-family="var(--mono)">{v:.2f}</text>')
    # day ticks at SGT midnights
    d = t0.astimezone(SGT).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    ticks = []
    while d < t1.astimezone(SGT):
        ticks.append(d)
        d += timedelta(days=1)
    step = max(1, -(-len(ticks) // 6))
    for tk in ticks[::step]:
        x = x_at(tk)
        if x > W - PR - 115:  # keep clear of the right-edge label
            continue
        p.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{PT}" y2="{H-PB}" stroke="var(--border)" stroke-width="1" opacity="0.5"/>')
        p.append(f'<text x="{x:.1f}" y="{H-PB+14}" text-anchor="middle" font-size="9" fill="var(--muted)" '
                 f'font-family="var(--mono)">{tk.strftime("%a %d")}</text>')
    p.append(f'<text x="{W-PR}" y="{H-PB+14}" text-anchor="end" font-size="9" fill="var(--warn)" '
             f'font-family="var(--mono)">1 day before kickoff</text>')
    p.append(f'<line x1="{W-PR}" x2="{W-PR}" y1="{PT}" y2="{H-PB}" stroke="var(--warn)" stroke-width="1.2" stroke-dasharray="4 3"/>')
    if t0 < now < t1:
        x = x_at(now)
        p.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{PT}" y2="{H-PB}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="1 3"/>')
        p.append(f'<text x="{x+4:.1f}" y="{H-PB-4}" font-size="9" fill="var(--muted)" font-family="var(--mono)">now</text>')
    if bet_odds:
        y = y_at(bet_odds)
        p.append(f'<line x1="{PL}" x2="{W-PR}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--green)" stroke-width="1.4" stroke-dasharray="6 4"/>')
        p.append(f'<text x="{PL+4}" y="{y-5:.1f}" font-size="10" fill="var(--green)" '
                 f'font-family="var(--mono)">your bet @ {bet_odds:.2f}</text>')
    xy = [(x_at(r.snapshot_utc), y_at(r.odds_h)) for r in pts.itertuples()]
    if len(xy) > 1:
        p.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in xy) +
                 '" fill="none" stroke="var(--gold)" stroke-width="2"/>')
    for x, y in xy:
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="var(--gold)"/>')
    p.append("</svg>")
    return "".join(p)


def fixture_card_html(row, series, now):
    kickoff = pd.Timestamp(row["start_time"])
    kickoff = kickoff.tz_convert("UTC") if kickoff.tzinfo else kickoff.tz_localize("UTC")
    cutoff = kickoff - timedelta(hours=CHART_CUTOFF_H)
    placed = str(row.get("bet_placed", False)).lower() == "true"
    bet_odds = float(row["actual_odds"]) if placed and pd.notna(row.get("actual_odds")) else None
    stake = float(row["actual_stake_sgd"]) if placed and pd.notna(row.get("actual_stake_sgd")) else None

    header = (f'<div class="ac-name">{row["fixture"]}'
              + ('<span class="tilt-pill" style="color:var(--green);background:rgba(77,184,122,0.12);'
                 'border:1px solid rgba(77,184,122,0.3);margin-left:6px;">&#10003; BET PLACED</span>' if placed else "")
              + f'</div><div class="ac-desc">Pick: {row["pick"]} &middot; kickoff {_fmt_sgt(kickoff)}</div>')

    s = None
    if series is not None and len(series):
        s = series.dropna(subset=["odds_h"])
        s = s[s["snapshot_utc"] <= cutoff].sort_values("snapshot_utc")
    if s is None or s.empty:
        msg = ("No daily run saw this fixture more than 1 day before kickoff (or it was logged before odds "
               "tracking started), so there is nothing to chart yet.")
        return (f'<div class="card" style="margin-bottom:14px;">{header}'
                f'<div style="font-size:11px;color:var(--muted);padding:14px 0;">{msg}</div></div>')

    n = len(s)
    last = s.iloc[-1]
    stats = [("Latest charted run", f'{last["odds_h"]:.2f}', _fmt_sgt(last["snapshot_utc"])),
             ("High / low", f'{s["odds_h"].max():.2f} / {s["odds_h"].min():.2f}', f"{n} daily run{'s' if n != 1 else ''}")]
    if bet_odds is not None:
        diff = last["odds_h"] - bet_odds
        word = "run paid more" if diff > 0.0049 else ("run paid less" if diff < -0.0049 else "same as your bet")
        stats.insert(0, ("Your bet", f"{bet_odds:.2f}", f"S${stake:,.2f} staked" if stake else ""))
        stats.append(("Latest run vs bet", f"{diff:+.2f}", word))
    stats_html = "".join(
        f'<div><div class="buy-stat-l">{k}</div><span style="font-family:var(--mono);">{v}</span>'
        f'<div style="font-size:9px;color:var(--muted);">{sub}</div></div>' for k, v, sub in stats)
    runs = " &nbsp;&middot;&nbsp; ".join(
        f'{r.snapshot_utc.astimezone(SGT).strftime("%a %d %b")}: <b style="font-family:var(--mono);">{r.odds_h:.2f}</b>'
        for r in s.itertuples())
    note = ""
    if n < 3:
        note = (f'<div style="font-size:10px;color:var(--muted);margin-top:8px;">Only {n} daily run'
                f'{"s" if n != 1 else ""} so far &mdash; no trend can be read yet.</div>')
    svg = fixture_chart_svg(s, cutoff, now, bet_odds)
    return (f'<div class="card" style="margin-bottom:14px;">{header}<div style="margin-top:10px;">{svg}</div>'
            f'<div style="font-size:11px;color:var(--muted);margin-top:6px;line-height:1.7;">Odds on each daily run: {runs}</div>'
            f'<div class="buy-stats">{stats_html}</div>{note}</div>')


def odds_section_html(log_df, hist=None, now=None):
    """Section of per-fixture charts for qualifying fixtures (upcoming, or finished within
    SHOW_AFTER_KICKOFF_H hours). Returns '' if there is nothing to show."""
    hist = load_history() if hist is None else hist
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if log_df is None or len(log_df) == 0:
        return ""
    q = log_df[log_df["qualifies"].astype(str).str.lower() == "true"].copy()
    if q.empty:
        return ""
    q["_ko"] = pd.to_datetime(q["start_time"], utc=True)
    q = q[q["_ko"] > now - pd.Timedelta(hours=SHOW_AFTER_KICKOFF_H)].sort_values("_ko")
    if q.empty:
        return ""
    cards = []
    for _, r in q.iterrows():
        g = hist[(hist["fixture"] == r["fixture"]) & (hist["start_time"] == r["_ko"])] if len(hist) else None
        cards.append(fixture_card_html(r, g, now))
    return ('<div class="section"><div class="section-title">Odds on each daily run &mdash; qualifying fixtures, '
            'up to 1 day before kickoff</div>'
            '<div style="font-size:11px;color:var(--muted);margin-bottom:12px;line-height:1.6;">'
            'Home-win odds Singapore Pools quoted on each daily run (gold), and where a bet was placed, the odds you '
            'actually got (green dashed). Line above the green one = that day\'s price paid more than the bet did.'
            '</div>' + "".join(cards) + '</div>')
