"""
Odds-movement charts for qualifying football fixtures
=====================================================
One inline-SVG line chart per qualifying fixture: the home-win odds Singapore Pools
quoted at each snapshot (odds_history.csv, written by odds_snapshot.py), from the first
snapshot up to kickoff, with
  - a dashed amber line at "1 day before kickoff",
  - a faint "now" line while the match is still ahead,
  - a dashed green line at the odds actually obtained, if a bet was placed
    (record_bet.py -> actual_odds / actual_stake_sgd in qualifying_log.csv).
Reading it: if the market line sits ABOVE the green line, waiting paid more than the
bet did; below it, the bet beat the market. Raw levels and the number of snapshots are
always shown -- with few snapshots the chart says so instead of implying a trend.

Used two ways:
  * daily_dashboard.render_html embeds odds_section_html() in the football dashboard;
  * `python football_betting/odds_chart.py` writes the standalone output/odds_tracker.html
    (rebuilt by the odds-snapshot workflow every few hours, so it is fresher than the
    once-a-day dashboard).
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

OUT_DIR = Path(__file__).parent / "output"
SGT = ZoneInfo("Asia/Singapore")
SHOW_AFTER_KICKOFF_H = 36  # keep a finished fixture's chart visible for a day and a half


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


def fixture_chart_svg(series, kickoff, now, bet_odds=None):
    """series: DataFrame with snapshot_utc (tz-aware) and odds_h. Returns inline SVG markup."""
    W, H, PL, PR, PT, PB = 560, 210, 46, 16, 14, 30
    pts = series.dropna(subset=["odds_h"]).sort_values("snapshot_utc")
    t0 = pts["snapshot_utc"].min()
    if (kickoff - t0) < timedelta(hours=6):
        t0 = kickoff - timedelta(hours=6)
    t1 = kickoff
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

    parts = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;max-width:640px;" role="img" '
             f'aria-label="Home-win odds over time">']
    # y grid + labels
    for v in (lo, (lo + hi) / 2, hi):
        y = y_at(v)
        parts.append(f'<line x1="{PL}" x2="{W-PR}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--border)" stroke-width="1"/>')
        parts.append(f'<text x="{PL-6}" y="{y+3:.1f}" text-anchor="end" font-size="10" fill="var(--muted)" '
                     f'font-family="var(--mono)">{v:.2f}</text>')
    # x ticks at SGT midnights
    d = t0.astimezone(SGT).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    ticks = []
    while d.astimezone(timezone.utc) < t1:
        ticks.append(d)
        d += timedelta(days=1)
    step = max(1, len(ticks) // 6 + (1 if len(ticks) % 6 else 0))
    for tk in ticks[::step]:
        x = x_at(tk.astimezone(timezone.utc))
        if x > W - PR - 48:  # would collide with the "kickoff" label
            continue
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{PT}" y2="{H-PB}" stroke="var(--border)" stroke-width="1" opacity="0.5"/>')
        parts.append(f'<text x="{x:.1f}" y="{H-PB+14}" text-anchor="middle" font-size="9" fill="var(--muted)" '
                     f'font-family="var(--mono)">{tk.strftime("%a %d")}</text>')
    # one day before kickoff
    t24 = kickoff - timedelta(hours=24)
    if t0 < t24 < t1:
        x = x_at(t24)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{PT}" y2="{H-PB}" stroke="var(--warn)" stroke-width="1.2" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{x-4:.1f}" y="{PT+9}" text-anchor="end" font-size="9" fill="var(--warn)" '
                     f'font-family="var(--mono)">1 day before</text>')
    # kickoff edge label
    parts.append(f'<text x="{W-PR}" y="{H-PB+14}" text-anchor="end" font-size="9" fill="var(--text)" '
                 f'font-family="var(--mono)">kickoff</text>')
    # now
    if t0 < now < t1:
        x = x_at(now)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{PT}" y2="{H-PB}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="1 3"/>')
        parts.append(f'<text x="{x+4:.1f}" y="{H-PB-4}" font-size="9" fill="var(--muted)" font-family="var(--mono)">now</text>')
    # bet line
    if bet_odds:
        y = y_at(bet_odds)
        parts.append(f'<line x1="{PL}" x2="{W-PR}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--green)" stroke-width="1.4" stroke-dasharray="6 4"/>')
        parts.append(f'<text x="{W-PR}" y="{y-5:.1f}" text-anchor="end" font-size="10" fill="var(--green)" '
                     f'font-family="var(--mono)">your bet @ {bet_odds:.2f}</text>')
    # odds line + points
    xy = [(x_at(r.snapshot_utc), y_at(r.odds_h)) for r in pts.itertuples()]
    if len(xy) > 1:
        parts.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in xy) +
                     '" fill="none" stroke="var(--gold)" stroke-width="2"/>')
    for x, y in xy:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="var(--gold)"/>')
    parts.append("</svg>")
    return "".join(parts)


def fixture_card_html(row, series, now):
    kickoff = pd.Timestamp(row["start_time"]).tz_convert("UTC") if pd.Timestamp(row["start_time"]).tzinfo \
        else pd.Timestamp(row["start_time"]).tz_localize("UTC")
    placed = str(row.get("bet_placed", False)).lower() == "true"
    bet_odds = float(row["actual_odds"]) if placed and pd.notna(row.get("actual_odds")) else None
    stake = float(row["actual_stake_sgd"]) if placed and pd.notna(row.get("actual_stake_sgd")) else None

    header = (f'<div class="ac-name">{row["fixture"]}'
              + ('<span class="tilt-pill" style="color:var(--green);background:rgba(77,184,122,0.12);'
                 'border:1px solid rgba(77,184,122,0.3);margin-left:6px;">&#10003; BET PLACED</span>' if placed else "")
              + f'</div><div class="ac-desc">Pick: {row["pick"]} &middot; kickoff {_fmt_sgt(kickoff)}</div>')

    if series is None or series.dropna(subset=["odds_h"]).empty:
        body = ('<div style="font-size:11px;color:var(--muted);padding:14px 0;">No snapshots tracked for this fixture '
                '(it was logged before odds tracking started).</div>')
        return f'<div class="card" style="margin-bottom:14px;">{header}{body}</div>'

    s = series.dropna(subset=["odds_h"]).sort_values("snapshot_utc")
    last = s.iloc[-1]
    n = len(s)
    hours_out = max(0.0, (kickoff - last["snapshot_utc"]).total_seconds() / 3600)
    stats = [("Latest", f'{last["odds_h"]:.2f}', f'{hours_out:.0f}h before KO'),
             ("High / low", f'{s["odds_h"].max():.2f} / {s["odds_h"].min():.2f}', f"{n} snapshots")]
    if bet_odds is not None:
        diff = last["odds_h"] - bet_odds
        word = "market pays more" if diff > 0.0049 else ("market pays less" if diff < -0.0049 else "same as your bet")
        stake_txt = f"S${stake:,.2f} staked" if stake else ""
        stats.insert(0, ("Your bet", f"{bet_odds:.2f}", stake_txt))
        stats.append(("Latest vs bet", f"{diff:+.2f}", word))
    stats_html = "".join(
        f'<div><div class="buy-stat-l">{k}</div><span style="font-family:var(--mono);">{v}</span>'
        f'<div style="font-size:9px;color:var(--muted);">{sub}</div></div>' for k, v, sub in stats)
    note = ""
    if n < 4:
        note = ('<div style="font-size:10px;color:var(--muted);margin-top:8px;">Only '
                f'{n} snapshot{"s" if n != 1 else ""} so far &mdash; the line fills in as the 3-hourly tracker runs; '
                'no trend can be read from this yet.</div>')
    svg = fixture_chart_svg(s, kickoff, now, bet_odds)
    return (f'<div class="card" style="margin-bottom:14px;">{header}<div style="margin-top:10px;">{svg}</div>'
            f'<div class="buy-stats">{stats_html}</div>{note}</div>')


def odds_section_html(log_df, hist=None, now=None):
    """Section of per-fixture odds charts for qualifying fixtures (upcoming, or finished within
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
    return ('<div class="section"><div class="section-title">Odds movement toward kickoff '
            '&mdash; qualifying fixtures</div>'
            '<div style="font-size:11px;color:var(--muted);margin-bottom:12px;line-height:1.6;">'
            'Home-win odds at each tracker snapshot (gold), the 1-day-before-kickoff mark (amber) and, where a bet '
            'was placed, the odds you actually got (green dashed). Market line above the green line = waiting paid '
            'more than the bet did.</div>' + "".join(cards) + '</div>')


PAGE_CSS = """
:root{--bg:#0C0E0D;--card:#141614;--card2:#1B1E1B;--border:#2C302C;--border2:#3A3F3A;--text:#DDE8DD;--muted:#7A8F7A;
--faint:#1B1E1B;--gold:#C9A84C;--green:#4DB87A;--red:#E05555;--warn:#E8A020;--mono:'IBM Plex Mono',monospace;}
@media (prefers-color-scheme: light){:root{--bg:#F6F7F5;--card:#FFFFFF;--card2:#EEF1EC;--border:#D5DAD2;--border2:#C2C9BE;
--text:#1B2A1B;--muted:#5B6E5B;--faint:#EEF1EC;--gold:#8A6D1E;--green:#2E8B57;--red:#C0392B;--warn:#B8750A;}}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);font-family:'Inter',sans-serif;color:var(--text);font-size:14px;}
.page{max-width:900px;margin:0 auto;padding:28px 16px 60px;}
.h-sub{font-family:var(--mono);font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin-bottom:4px;}
.h-title{font-size:26px;font-weight:600;margin-bottom:6px;}
.h-meta{font-size:12px;color:var(--muted);margin-bottom:22px;}
.section-title{font-family:var(--mono);font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin-bottom:12px;}
.card{background:var(--card);border-radius:14px;border:1px solid var(--border);padding:18px 22px;}
.ac-name{font-weight:600;font-size:14px;} .ac-desc{font-size:11px;color:var(--muted);margin-top:2px;}
.tilt-pill{font-family:var(--mono);font-size:10px;padding:2px 8px;border-radius:10px;}
.buy-stats{display:flex;gap:22px;flex-wrap:wrap;margin-top:12px;padding-top:12px;border-top:1px solid var(--border);font-size:13px;}
.buy-stat-l{color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.06em;}
a{color:var(--gold);text-decoration:none;}
"""


def build_standalone():
    log_path = OUT_DIR / "qualifying_log.csv"
    log_df = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    hist = load_history()
    now = pd.Timestamp.now(tz="UTC")
    section = odds_section_html(log_df, hist, now)
    if not section:
        section = ('<div class="card"><div class="ac-desc">No qualifying fixtures to chart right now.</div></div>')
    n_snap = hist["snapshot_utc"].nunique() if len(hist) else 0
    last = f'{_fmt_sgt(hist["snapshot_utc"].max())}' if len(hist) else "n/a"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Football Odds Tracker</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{PAGE_CSS}</style></head><body><div class="page">
<div class="h-sub">CPE Framework &mdash; Football</div>
<div class="h-title">Odds Tracker</div>
<div class="h-meta">Built {_fmt_sgt(now)} &middot; {n_snap} snapshots &middot; latest snapshot {last}
 &middot; refreshed by the tracker workflow every ~3 hours (GitHub may run it late)</div>
{section}
</div></body></html>"""
    (OUT_DIR / "odds_tracker.html").write_text(html)
    print(f"Saved -> {OUT_DIR / 'odds_tracker.html'}")


if __name__ == "__main__":
    build_standalone()
