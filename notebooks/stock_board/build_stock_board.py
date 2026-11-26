"""
Daily stock board: the stock analogue of the football checklist.

Reads the frozen rule (stock_board_rule.json, made by sb_freeze.py), the live price history, and the running log.
  1. finds ETFs whose form clears the frozen rule on the latest bar (only for horizons that passed the walk-forward gate)
  2. logs each new signal once (one live bet per instrument and horizon) and resolves logged bets as their target,
     stop or time limit is reached, on the instrument's own trading days
  3. writes stock_board.html and stock_board_log.csv

It lists candidates only. It never places a trade or touches an account.

Usage:  python stock_board/build_stock_board.py     (run from notebooks/)
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sb_core as C  # noqa: E402

SGT = ZoneInfo("Asia/Singapore")
RULE_PATH = HERE / "stock_board_rule.json"
LOG_PATH = HERE / "stock_board_log.csv"
HTML_PATH = HERE / "stock_board.html"
STALE_DAYS = 6  # an instrument whose last bar is older than this (vs the panel's latest bar) is skipped

LOG_COLS = ["signal_date", "ticker", "name", "group", "horizon", "rule", "ref_close", "sigma_h", "target_pct", "stop_pct",
            "status", "entry_date", "entry_px", "target_px", "stop_px", "exit_date", "exit_px", "outcome", "win", "pnl",
            "bet_placed", "actual_stake", "actual_entry_px", "logged_at"]


def load_log():
    if LOG_PATH.exists():
        d = pd.read_csv(LOG_PATH, dtype={"signal_date": str, "entry_date": str, "exit_date": str})
        for c in LOG_COLS:
            if c not in d.columns:
                d[c] = np.nan
        for c in ("entry_date", "exit_date", "outcome", "status", "rule", "logged_at"):
            d[c] = d[c].astype(object)
        return d[LOG_COLS]
    return pd.DataFrame(columns=LOG_COLS)


def live_signals(prices, rule, panel_end):
    """Return list of dicts: one per (ticker, horizon) whose latest bar clears the frozen rule."""
    sigs, feats = [], {}
    for t in C.ALL_TICKERS:
        if t not in prices.columns:
            continue
        s = prices[t].dropna()
        s = s[s > 0]
        if len(s) < 800 or (panel_end - s.index[-1]).days > STALE_DAYS:
            continue
        f = C.features_for(s)
        feats[t] = (s, f)
        last = f.iloc[-1]
        for h in C.HORIZONS:
            lr = rule["horizons"][str(h)].get("live_rule")
            if not lr:
                continue
            ok = True
            for c in lr["conditions"]:
                feat, side = C.CONDITIONS[c]
                p = last[feat + "_pct"]
                if not np.isfinite(p) or not ((p >= lr["q"]) if side == "hi" else (p <= 1 - lr["q"])):
                    ok = False
                    break
            if ok and np.isfinite(last["sigma_d"]):
                sigs.append(dict(ticker=t, horizon=h, rule=lr["label"], signal_date=s.index[-1], ref_close=float(s.iloc[-1]),
                                 sigma_h=float(last["sigma_d"] * np.sqrt(h)), mom21=float(last["mom21"]),
                                 mom21_pct=float(last["mom21_pct"]), volratio=float(last["volratio"]),
                                 volratio_pct=float(last["volratio_pct"]), trend=float(last["trend"]), dd=float(last["dd"])))
    return sigs, feats


def update_log(log, sigs, feats, rule, now_utc):
    # 1) advance every unresolved bet
    for i, r in log.iterrows():
        if r["status"] == "RESOLVED":
            continue
        t = r["ticker"]
        if t not in feats:
            s = None
        else:
            s = feats[t][0]
        if s is None:
            continue
        sp = s.index.get_loc(pd.Timestamp(r["signal_date"]))
        ev = C.evaluate_bet(s, sp, float(r["sigma_h"]), int(r["horizon"]))
        log.at[i, "status"] = ev["status"]
        if ev["status"] != "PENDING_ENTRY":
            log.at[i, "entry_date"] = str(ev["entry_date"].date())
            for k in ("entry_px", "target_px", "stop_px"):
                log.at[i, k] = ev[k]
        if ev["status"] == "RESOLVED":
            log.at[i, "exit_date"] = str(ev["exit_date"].date())
            for k in ("exit_px", "outcome", "win", "pnl"):
                log.at[i, k] = ev[k]
    # 2) log new signals (one live bet per ticker+horizon; never re-log the same signal date)
    live = set(zip(log.loc[log["status"] != "RESOLVED", "ticker"], log.loc[log["status"] != "RESOLVED", "horizon"].astype(int)))
    seen = set(zip(log["ticker"], log["horizon"].astype(int), log["signal_date"]))
    new_rows = []
    for g in sigs:
        key = (g["ticker"], g["horizon"])
        sd = str(g["signal_date"].date())
        if key in live or (g["ticker"], g["horizon"], sd) in seen:
            continue
        a = C.K_TARGET * g["sigma_h"]
        b = min(C.K_STOP * g["sigma_h"], 0.95)
        new_rows.append({"signal_date": sd, "ticker": g["ticker"], "name": rule["universe"][g["ticker"]]["name"],
                         "group": rule["universe"][g["ticker"]]["group"], "horizon": g["horizon"], "rule": g["rule"],
                         "ref_close": g["ref_close"], "sigma_h": g["sigma_h"], "target_pct": a, "stop_pct": b,
                         "status": "PENDING_ENTRY", "bet_placed": False,
                         "logged_at": now_utc.strftime("%Y-%m-%d %H:%M UTC")})
    if new_rows:
        new = pd.DataFrame(new_rows).reindex(columns=LOG_COLS)
        log = new if log.empty else pd.concat([log, new], ignore_index=True)
    return log


# ----------------------------------------------------------------------------------------------- HTML

CSS = """
  :root{
    --bg:#0C0E0D; --card:#141614; --card2:#1B1E1B; --border:#2C302C; --border2:#3A3F3A;
    --text:#DDE8DD; --muted:#7A8F7A; --faint:#1B1E1B;
    --gold:#C9A84C; --green:#4DB87A; --red:#E05555; --warn:#E8A020;
    --mono:'IBM Plex Mono',monospace; --serif:'DM Serif Display',serif;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);font-family:'Inter',sans-serif;color:var(--text);font-size:14px;}
  a{color:var(--gold);text-decoration:none;}
  .page{max-width:1000px;margin:0 auto;padding:32px 24px 80px;}
  .header{background:linear-gradient(160deg,#141614,#1B1E1B);border:1px solid var(--border2);
           border-radius:16px;padding:32px 40px;margin-bottom:28px;
           display:grid;grid-template-columns:1fr auto;align-items:center;gap:24px;}
  .h-title{font-family:var(--serif);font-size:28px;color:var(--text);margin-bottom:6px;}
  .h-title em{font-style:italic;color:var(--gold);}
  .h-sub{font-family:var(--mono);font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin-bottom:4px;}
  .h-meta{font-size:12px;color:var(--muted);}
  .h-badge{background:rgba(201,168,76,0.12);border:1px solid rgba(201,168,76,0.3);
            border-radius:12px;padding:12px 20px;text-align:center;}
  .h-badge-num{font-family:var(--mono);font-size:28px;font-weight:600;color:var(--gold);}
  .h-badge-label{font-size:10px;color:var(--muted);display:block;margin-top:2px;letter-spacing:.1em;text-transform:uppercase;}
  .section{margin-bottom:28px;}
  .section-title{font-family:var(--mono);font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-bottom:14px;}
  .card{background:var(--card);border-radius:14px;border:1px solid var(--border);padding:20px 24px;}
  .tilt-table{width:100%;border-collapse:collapse;}
  .tilt-table th{font-family:var(--mono);font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
                  padding:10px 14px;border-bottom:2px solid var(--border);text-align:left;background:var(--faint);}
  .tilt-table td{padding:12px 14px;border-bottom:1px solid var(--border);vertical-align:middle;}
  .tilt-table tr:last-child td{border-bottom:none;}
  .tilt-table tr:hover td{background:var(--card2);}
  .ac-cell{display:flex;align-items:center;gap:10px;}
  .ac-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
  .ac-name{font-weight:600;font-size:13px;color:var(--text);}
  .ac-desc{font-size:10px;color:var(--muted);margin-top:2px;}
  .tilt-pill{display:inline-flex;align-items:center;padding:4px 10px;border-radius:100px;font-family:var(--mono);font-size:9px;
              font-weight:600;letter-spacing:.06em;white-space:nowrap;}
  .weight-bar-bg{background:var(--border);border-radius:4px;height:6px;display:inline-block;vertical-align:middle;margin-top:4px;}
  .weight-bar-fill{height:100%;border-radius:4px;}
  .buy-stats{display:flex;gap:18px;margin-top:12px;padding-top:12px;border-top:1px solid var(--border);
              font-family:var(--mono);font-size:11px;color:var(--text);flex-wrap:wrap;}
  .buy-stat-l{color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.06em;}
  .note{font-size:11px;color:var(--muted);line-height:1.7;margin-top:10px;}
  .disclaimer{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px 20px;font-size:11px;
               color:var(--muted);line-height:1.7;margin-top:28px;}
  .disclaimer code{font-family:var(--mono);background:var(--card2);border:1px solid var(--border);border-radius:4px;
               padding:1px 5px;font-size:10.5px;color:var(--text);}
  .mono{font-family:var(--mono);}
  @media(max-width:700px){ .header{grid-template-columns:1fr;} .page{padding:16px 12px 60px;} }
"""


def pct(x, d=1, sign=False):
    if x is None or not np.isfinite(x):
        return "&mdash;"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def bar(val, lo, hi, color="var(--gold)"):
    w = max(0, min(100, (val - lo) / (hi - lo) * 100))
    return (f'<div class="weight-bar-bg" style="width:64px;"><div class="weight-bar-fill" '
            f'style="width:{w:.0f}%;background:{color};"></div></div>')


def expiry_label(entry_ts, h):
    try:
        d = np.busday_offset(np.datetime64(pd.Timestamp(entry_ts).date()), h, roll="forward")
        return pd.Timestamp(d).strftime("%a %d %b %Y")
    except Exception:
        return "&mdash;"


def yearly_svg(yearly):
    """Bars: out-of-sample excess P&L per bet by year (rule bets minus the same ETFs' always-long)."""
    if not yearly:
        return ""
    W, H, PL, PR, PT, PB = 900, 200, 52, 16, 16, 30
    vals = [r["exc"] for r in yearly]
    lim = max(abs(min(vals)), abs(max(vals)), 0.005) * 1.15
    n = len(yearly)
    bw = (W - PL - PR) / n * 0.62
    zero = PT + (H - PT - PB) / 2

    def y(v):
        return zero - v / lim * (H - PT - PB) / 2

    out = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:200px;">',
           f'<line x1="{PL}" y1="{zero}" x2="{W - PR}" y2="{zero}" stroke="#2C302C" stroke-dasharray="3,3"/>',
           f'<text x="{PL - 8}" y="{zero + 4}" text-anchor="end" font-family="monospace" font-size="10" fill="#7A8F7A">0</text>',
           f'<text x="{PL - 8}" y="{y(lim) + 10}" text-anchor="end" font-family="monospace" font-size="10" fill="#7A8F7A">{lim * 100:+.1f}%</text>',
           f'<text x="{PL - 8}" y="{y(-lim)}" text-anchor="end" font-family="monospace" font-size="10" fill="#7A8F7A">{-lim * 100:+.1f}%</text>']
    for i, r in enumerate(yearly):
        cx = PL + (i + 0.5) * (W - PL - PR) / n
        v = r["exc"]
        col = "#4DB87A" if v >= 0 else "#E05555"
        top, h = (y(v), zero - y(v)) if v >= 0 else (zero, y(v) - zero)
        out.append(f'<rect x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{max(h, 0.8):.1f}" fill="{col}"/>')
        out.append(f'<text x="{cx:.1f}" y="{H - 12}" text-anchor="middle" font-family="monospace" font-size="9" fill="#7A8F7A">{str(int(r["year"]))[2:]}</text>')
        out.append(f'<text x="{cx:.1f}" y="{H - 2}" text-anchor="middle" font-family="monospace" font-size="8" fill="#55665A">n={int(r["n"])}</text>')
    out.append("</svg>")
    return "".join(out)


def cum_svg(resolved):
    W, H, PL, PR, PT, PB = 900, 200, 52, 16, 16, 16
    if not len(resolved):
        zero = H / 2
        return (f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:200px;"><line x1="{PL}" y1="{zero}" x2="{W - PR}" y2="{zero}" '
                f'stroke="#2C302C"/><text x="{PL - 8}" y="{zero + 4}" text-anchor="end" font-family="monospace" font-size="10" fill="#7A8F7A">0%</text></svg>')
    cum = (resolved.sort_values("exit_date")["pnl"].astype(float).cumsum() * 100).tolist()
    lo, hi = min(cum + [0]), max(cum + [0])
    span = (hi - lo) or 1
    n = len(cum)
    xs = [PL + (i / max(n - 1, 1)) * (W - PL - PR) for i in range(n)]
    ys = [H - PB - (v - lo) / span * (H - PT - PB) for v in cum]
    z = H - PB - (0 - lo) / span * (H - PT - PB)
    col = "#4DB87A" if cum[-1] >= 0 else "#E05555"
    pts = " ".join(f"{x:.1f},{yy:.1f}" for x, yy in zip(xs, ys))
    return (f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:200px;"><line x1="{PL}" y1="{z:.1f}" x2="{W - PR}" y2="{z:.1f}" '
            f'stroke="#2C302C" stroke-dasharray="3,3"/><text x="{PL - 8}" y="{z + 4:.1f}" text-anchor="end" font-family="monospace" '
            f'font-size="10" fill="#7A8F7A">0%</text><polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>'
            f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="4" fill="{col}"/>'
            f'<text x="{xs[-1] - 6:.1f}" y="{ys[-1] - 8:.1f}" text-anchor="end" font-family="monospace" font-size="10" fill="{col}">{cum[-1]:+.1f}%</text></svg>')


def render(rule, sigs, feats, log, generated_at, panel_end):
    hz21 = rule["horizons"]["21"]
    hz63 = rule["horizons"]["63"]
    oos21, base21 = hz21["oos"], hz21["always_long"]
    grp_stats = {g["group"]: g for g in hz21["by_group"]}
    live_keys = {}
    for _, r in log.iterrows():
        live_keys.setdefault((r["ticker"], int(r["horizon"])), []).append(r)

    rows = ""
    for g in sorted(sigs, key=lambda x: x["mom21"]):
        t, h = g["ticker"], g["horizon"]
        u = rule["universe"][t]
        recs = [r for r in live_keys.get((t, h), []) if r["status"] != "RESOLVED"]
        cur = recs[0] if recs else None
        a, b = C.K_TARGET * g["sigma_h"], min(C.K_STOP * g["sigma_h"], 0.95)
        if cur is not None and cur["status"] == "OPEN":
            tgt, stp, base_px = float(cur["target_px"]), float(cur["stop_px"]), float(cur["entry_px"])
            state = f'<span class="tilt-pill" style="color:var(--green);background:rgba(77,184,122,0.12);border:1px solid rgba(77,184,122,0.3);">IN PLAY since {cur["entry_date"]}</span>'
            levels_note = f"fixed at entry {cur['entry_date']}"
        else:
            base_px = g["ref_close"]
            tgt, stp = base_px * (1 + a), base_px * (1 - b)
            is_new = cur is not None and cur["signal_date"] == str(g["signal_date"].date())
            state = ('<span class="tilt-pill" style="color:var(--gold);background:rgba(201,168,76,0.12);border:1px solid rgba(201,168,76,0.3);">'
                     'NEW &middot; enters next close</span>')
            levels_note = "indicative, reset at next close"
        gs = grp_stats.get(u["group"])
        grp_txt = (f'{u["group"]}: backtest hit {gs["hit"]:.0%} vs {gs["base_hit"]:.0%} always-long, n={int(gs["n"])}'
                   if gs else u["group"])
        exp = expiry_label(cur["entry_date"] if cur is not None and cur["status"] == "OPEN" else g["signal_date"] + pd.Timedelta(days=1), h)
        rows += f"""
        <tr>
          <td><div class="ac-cell"><span class="ac-dot" style="background:var(--gold);"></span>
            <div><div class="ac-name">{t} <span style="color:var(--muted);font-weight:400;">&middot; {u['name']}</span></div>
            <div class="ac-desc">{grp_txt}</div></div></div></td>
          <td>{state}</td>
          <td class="mono">{pct(g['mom21'], 1, True)} {bar(-g['mom21'], 0, 0.15, 'var(--red)')}<div class="ac-desc">bottom {g['mom21_pct'] * 100:.0f}% of its own 5y</div></td>
          <td class="mono">{g['volratio']:.2f}&times; {bar(g['volratio'], 0.5, 2.5)}<div class="ac-desc">top {(1 - g['volratio_pct']) * 100:.0f}% of its own 5y</div></td>
          <td class="mono" style="text-align:right;">{g['ref_close']:.2f}</td>
          <td class="mono" style="text-align:right;color:var(--green);">{tgt:.2f}<div class="ac-desc">{pct(a, 1, True)}</div></td>
          <td class="mono" style="text-align:right;color:var(--red);">{stp:.2f}<div class="ac-desc">{pct(-b, 1, True)}</div></td>
          <td class="mono" style="font-size:11px;">{h} bars<div class="ac-desc">about {exp}</div></td>
        </tr>"""
    if not sigs:
        rows = ("""<tr><td colspan="8" style="text-align:center;padding:32px 16px;color:var(--muted);">
          No ETF clears the bar today. The rule needs a sharp 1-month drop and unusually high recent volatility
          together, so most days show nothing &mdash; that is the design, not a fault.</td></tr>""")

    # open + resolved bets
    open_bets = log[log["status"] != "RESOLVED"]
    resolved = log[log["status"] == "RESOLVED"].copy()
    resolved["pnl"] = resolved["pnl"].astype(float)
    resolved["win"] = resolved["win"].astype(float)
    n_logged, n_res = len(log), len(resolved)
    if n_res:
        live_stats = f"""
      <div class="buy-stats" style="border-top:none;padding-top:0;margin-top:0;">
        <div><div class="buy-stat-l">Signals logged</div><span>{n_logged}</span></div>
        <div><div class="buy-stat-l">Resolved</div><span>{n_res}</span></div>
        <div><div class="buy-stat-l">Live hit rate</div><span style="color:var(--gold)">{resolved['win'].mean():.1%}</span></div>
        <div><div class="buy-stat-l">Avg P&amp;L per bet</div><span style="color:{'var(--green)' if resolved['pnl'].mean() >= 0 else 'var(--red)'}">{resolved['pnl'].mean() * 100:+.2f}%</span></div>
        <div><div class="buy-stat-l">Backtest (out of sample)</div><span style="color:var(--muted)">{oos21['hit']:.1%} / {oos21['pnl'] * 100:+.2f}%</span></div>
      </div>"""
    else:
        live_stats = f"""
      <div class="buy-stats" style="border-top:none;padding-top:0;margin-top:0;">
        <div><div class="buy-stat-l">Signals logged</div><span>{n_logged}</span></div>
        <div><div class="buy-stat-l">Awaiting entry / open</div><span>{len(open_bets)}</span></div>
        <div><div class="buy-stat-l">Resolved</div><span>0</span></div>
        <div><div class="buy-stat-l">Backtest (out of sample)</div><span style="color:var(--muted)">{oos21['hit']:.1%} / {oos21['pnl'] * 100:+.2f}%</span></div>
      </div>"""
    open_rows = ""
    for _, r in open_bets.sort_values("signal_date").iterrows():
        entered = r["status"] == "OPEN"
        open_rows += f"""<tr><td class="mono">{r['ticker']}</td><td class="mono">{r['signal_date']}</td>
          <td class="mono">{r['status'].replace('_', ' ')}</td><td class="mono" style="text-align:right;">{(f"{float(r['entry_px']):.2f}" if entered else '&mdash;')}</td>
          <td class="mono" style="text-align:right;">{(f"{float(r['target_px']):.2f}" if entered else pct(float(r['target_pct']), 1, True))}</td>
          <td class="mono" style="text-align:right;">{(f"{float(r['stop_px']):.2f}" if entered else pct(-float(r['stop_pct']), 1, True))}</td>
          <td class="mono">{int(r['horizon'])} bars</td></tr>"""
    open_html = (f"""<div class="card" style="padding:0;overflow:hidden;margin-top:14px;"><table class="tilt-table"><thead><tr>
        <th>ETF</th><th>Signal bar</th><th>Status</th><th style="text-align:right">Entry</th><th style="text-align:right">Target</th>
        <th style="text-align:right">Stop</th><th>Horizon</th></tr></thead><tbody>{open_rows}</tbody></table></div>""" if open_rows else "")

    grp_rows = "".join(
        f"""<tr><td>{g['group']}</td><td class="mono" style="text-align:right;">{int(g['n'])}</td>
        <td class="mono" style="text-align:right;">{g['hit']:.0%}</td><td class="mono" style="text-align:right;">{g['base_hit']:.0%}</td>
        <td class="mono" style="text-align:right;color:{'var(--green)' if g['exc'] >= 0 else 'var(--red)'};">{g['exc'] * 100:+.2f}%</td></tr>"""
        for g in sorted(hz21["by_group"], key=lambda x: -x["exc"]))

    rule_label = hz21["live_rule"]["label"] if hz21.get("live_rule") else "none"
    p = rule["params"]
    n_pick = len(sigs)
    corr_note = (f"These {n_pick} ETFs mostly move together, so treat them as largely one bet on the same market bounce, not {n_pick} separate ones. "
                 if n_pick >= 3 else "")
    updated = generated_at.astimezone(SGT).strftime("%Y-%m-%d %H:%M SGT")
    html = f"""<title>Stock Dip Board &mdash; CPE Framework</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet">
<style>{CSS}</style>
<div class="page">

<div class="header">
  <div>
    <div class="h-sub">CPE Framework &mdash; Stock Board</div>
    <div class="h-title">ETF <em>Dip &amp; Volatility</em> Checklist</div>
    <div class="h-meta">Updated: {updated} &nbsp;.&nbsp; prices through {panel_end.strftime('%a %d %b %Y')} (US close)
      &nbsp;.&nbsp; {len(rule['universe'])} ETFs scanned &nbsp;.&nbsp; 21-bar bets only</div>
  </div>
  <div class="h-badge"><div class="h-badge-num">{n_pick}</div><span class="h-badge-label">Qualifying Now</span></div>
</div>

<div class="section">
  <div class="section-title">Qualifying ETFs &mdash; 21-bar bet: target +{p['k_target']}&sigma; before stop &minus;{p['k_stop']}&sigma;</div>
  <div class="card" style="padding:0;overflow:hidden;">
    <table class="tilt-table"><thead><tr>
      <th style="width:300px">ETF</th><th>Status</th><th>1-month drop</th><th>Volatility</th>
      <th style="text-align:right">Last close</th><th style="text-align:right">Target</th><th style="text-align:right">Stop</th><th>Horizon</th>
    </tr></thead><tbody>{rows}</tbody></table>
  </div>
  <div class="note">A bet means: buy at the next close after the signal, sell at the target or the stop (checked on daily closes) or after
  21 trading days, whichever comes first. Levels use each ETF's own volatility: &sigma; is the last 63 days' daily standard deviation
  &times; &radic;21. {corr_note}63-bar bets: <b>no rule passed validation</b> ({hz63['oos']['n']:,} out-of-sample bets, hit rate lift
  {hz63['oos']['lift'] * 100:+.1f} pts, t = {hz63['oos']['t']:.2f}), so none are listed.</div>
</div>

<div class="section">
  <div class="section-title">Live Record &mdash; every signal logged from the day this board started</div>
  <div class="card">
    {cum_svg(resolved)}
    {live_stats}
    <div class="note">Cumulative P&amp;L in % of position, one unit per bet, summed in the order bets resolved. A bet is held until its
    target, stop or time limit and is never re-scored on later days. One live bet per ETF at a time.
    The first outcomes arrive within about a month.</div>
  </div>
  {open_html}
</div>

<div class="section">
  <div class="section-title">Walk-Forward Evidence (2010 &ndash; {panel_end.year}, out of sample) &mdash; rule: {rule_label}</div>
  <div class="card">
    <div class="buy-stats" style="border-top:none;padding-top:0;margin-top:0;">
      <div><div class="buy-stat-l">Bets</div><span>{oos21['n']:,}</span></div>
      <div><div class="buy-stat-l">Independent blocks</div><span>{oos21['n_blk']} months</span></div>
      <div><div class="buy-stat-l">Hit rate</div><span style="color:var(--gold)">{oos21['hit']:.1%}</span></div>
      <div><div class="buy-stat-l">Same ETFs always-long</div><span>{oos21['base_hit']:.1%}</span></div>
      <div><div class="buy-stat-l">Break-even hit rate</div><span>{oos21['be']:.1%}</span></div>
      <div><div class="buy-stat-l">Avg P&amp;L / bet</div><span style="color:var(--green)">{oos21['pnl'] * 100:+.2f}%</span></div>
      <div><div class="buy-stat-l">Always-long P&amp;L / bet</div><span>{oos21['base_pnl'] * 100:+.2f}%</span></div>
      <div><div class="buy-stat-l">Cluster t-stat</div><span>{oos21['t']:.2f}</span></div>
    </div>
    <div class="section-title" style="margin-top:22px;">Excess P&amp;L per bet by year &mdash; rule bets minus the same ETFs' always-long bets</div>
    {yearly_svg(hz21['yearly'])}
    <div class="section-title" style="margin-top:22px;">By ETF group (diagnostic only, not used to choose the rule)</div>
    <table class="tilt-table"><thead><tr><th>Group</th><th style="text-align:right">Bets</th><th style="text-align:right">Hit rate</th>
      <th style="text-align:right">Always-long</th><th style="text-align:right">Excess P&amp;L</th></tr></thead><tbody>{grp_rows}</tbody></table>
    <div class="note">Hit rate here counts a win when the target is hit first, or when the 21 days end with a positive return. The break-even
    hit rate is |average loss| &divide; (average win + |average loss|), the stock version of 1 &divide; odds. It sits well below the hit
    rate for any long position because most bets end on the time limit with small gains; the fair comparison is the &ldquo;same ETFs
    always-long&rdquo; column, i.e. what buying every one of these ETFs on every day would have scored.</div>
  </div>
</div>

<div class="disclaimer">
  <strong>How this works:</strong> the same recipe as the football board. A pick appears only when <b>both</b> form conditions hold on the latest
  close: <code>1-month return in the bottom 10% of the ETF's own last 5 years</code> and <code>1-month volatility relative to 1-year
  volatility in the top 10%</code> &mdash; percentile cut-offs of each ETF's own point-in-time history, not fitted values. Of the
  {p['n_rules_tested']} two-condition and one-condition rules tested, the rule is chosen each year using only outcomes already
  complete before 1 January, and the table above pools the following year's bets, never seen when the rule was chosen. A rule may fire on
  at most {p['max_fire_rate']:.0%} of ETF-days; a horizon publishes picks only if its pooled out-of-sample cluster t-statistic is at least {p['t_gate']:.0f}.
  <br><br><b>Read this before using it:</b> the effective sample is the number of separate market episodes, not the number of bets &mdash;
  ETFs dip together, so {oos21['n']:,} bets stand on roughly {oos21['n_blk']} months, and the t-statistic is indicative only. The edge is in
  equity-type ETFs; bond and commodity ETFs show none (table above), and 2026 so far is negative. Single stocks are excluded: the ten in the
  stored price panel were picked after they became large, which inflates any backtest. Costs are 5 bps round trip; entry is assumed at the
  next close after the signal, so there is no look-ahead, but real fills will differ. Validated on past data only; the live record above is the
  real test. This page lists candidates only &mdash; it never places a trade or touches an account, and it is research, not advice.
  &nbsp;.&nbsp; <strong>Dr. Arun Ramanathan</strong>
</div>

</div>
"""
    return html


def main():
    now_utc = datetime.now(timezone.utc)
    rule = json.loads(RULE_PATH.read_text())
    prices = C.load_panel()
    panel_end = prices.index[-1]
    sigs, feats = live_signals(prices, rule, panel_end)
    log = update_log(load_log(), sigs, feats, rule, now_utc)
    log.to_csv(LOG_PATH, index=False)
    html = render(rule, sigs, feats, log, now_utc, panel_end)
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Stock board: {len(sigs)} qualifying ({', '.join(g['ticker'] for g in sigs) or 'none'}); "
          f"log {len(log)} rows ({int((log['status'] == 'RESOLVED').sum())} resolved); prices through {panel_end.date()}")


if __name__ == "__main__":
    main()
