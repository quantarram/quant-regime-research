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
import sb_sim as SIM  # noqa: E402

SGT = ZoneInfo("Asia/Singapore")
RULE_PATH = HERE / "stock_board_rule.json"
LOG_PATH = HERE / "stock_board_log.csv"
HTML_PATH = HERE / "stock_board.html"
SIM_PATH = HERE / "stock_board_sim.json"
STALE_DAYS = 6  # an instrument whose last bar is older than this (vs the panel's latest bar) is skipped

LOG_COLS = ["signal_date", "ticker", "name", "group", "horizon", "rule", "ref_close", "sigma_h", "target_pct", "stop_pct",
            "status", "entry_date", "entry_px", "target_px", "stop_px", "exit_date", "exit_px", "outcome", "win", "pnl",
            "bet_placed", "actual_stake", "actual_entry_px", "logged_at", "mom21"]


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
                         "status": "PENDING_ENTRY", "bet_placed": False, "mom21": g["mom21"],
                         "logged_at": now_utc.strftime("%Y-%m-%d %H:%M UTC")})
    if new_rows:
        new = pd.DataFrame(new_rows).reindex(columns=LOG_COLS)
        log = new if log.empty else pd.concat([log, new], ignore_index=True)
    return log


# ------------------------------------------------------------------------------------------- paper account

def usd(x, d=0, sign=False):
    return f"{'+' if sign and x >= 0 else ''}{'-' if x < 0 else ''}US${abs(x):,.{d}f}"


def live_account(log, prices):
    """Paper account built from the ledger: every bet that has entered, sized by sb_sim's rules."""
    ent = log[log["status"].isin(["OPEN", "RESOLVED"])].copy()
    if ent.empty:
        return None
    ent["rank"] = pd.to_numeric(ent["mom21"], errors="coerce").fillna(0.0)
    tr = ent[["ticker", "entry_date", "entry_px", "exit_date", "pnl", "rank"]].copy()
    tr["entry_px"] = tr["entry_px"].astype(float)
    tr["pnl"] = pd.to_numeric(tr["pnl"], errors="coerce")
    cols = sorted(set(tr["ticker"]) | {"SPY"})
    irx = prices["^IRX"].dropna()
    res = SIM.simulate(tr, prices[cols], cash_rate=irx)
    E = res["equity"]
    spy = prices["SPY"].dropna().reindex(E.index).ffill()
    pos = res["positions"].copy()
    last = prices[cols].ffill().iloc[-1]
    rows = []
    for _, p in pos.iterrows():
        closed = pd.notna(p["exit_date"]) and pd.Timestamp(p["exit_date"]) <= E.index[-1]
        ret = p["pnl"] if closed else last[p["ticker"]] / p["entry_px"] - 1
        rows.append(dict(ticker=p["ticker"], entry_date=p["entry_date"], entry_px=p["entry_px"], last_px=float(last[p["ticker"]]),
                         alloc=p["alloc"], ret=float(ret), dollars=float(p["alloc"] * ret), closed=bool(closed),
                         exit_date=p["exit_date"] if closed else None))
    return dict(E=E, spy=spy, positions=pd.DataFrame(rows), skipped=res["skipped"],
                start=E.index[0], spy_ret=float(spy.iloc[-1] / spy.iloc[0] - 1))


def paper_svg(m):
    """Log-scale lines: paper account, SPY buy-and-hold, SPY at the account's own exposure (+ T-bills)."""
    W, H, PL, PR, PT, PB = 900, 260, 52, 16, 14, 24
    d = pd.to_datetime(m["dates"])
    series = [("strategy", "#C9A84C", "Paper account (rule picks)"), ("spy", "#7A8F7A", "SPY buy and hold"),
              ("matched", "#4D8FB8", "SPY at the account's average exposure + T-bills")]
    allv = np.log(np.array([v for k, _, _ in series for v in m[k]]))
    lo, hi = allv.min(), allv.max()
    span = hi - lo or 1
    x = lambda i: PL + i / max(len(d) - 1, 1) * (W - PL - PR)
    y = lambda v: H - PB - (np.log(v) - lo) / span * (H - PT - PB)
    out = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:260px;">']
    for lvl in (100, 200, 400, 800):
        if np.log(lvl) < lo - 0.05 or np.log(lvl) > hi + 0.05:
            continue
        out.append(f'<line x1="{PL}" y1="{y(lvl):.1f}" x2="{W - PR}" y2="{y(lvl):.1f}" stroke="#2C302C" stroke-dasharray="3,3"/>'
                   f'<text x="{PL - 8}" y="{y(lvl) + 4:.1f}" text-anchor="end" font-family="monospace" font-size="10" fill="#7A8F7A">{lvl}</text>')
    for yr in sorted(set(d.year)):
        if yr % 2:
            continue
        i = int(np.argmax(d.year == yr))
        out.append(f'<text x="{x(i):.1f}" y="{H - 6}" text-anchor="middle" font-family="monospace" font-size="9" fill="#7A8F7A">{yr}</text>')
    for k, col, _ in series:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(m[k]))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        out.append(f'<text x="{W - PR - 2}" y="{y(m[k][-1]) - 5:.1f}" text-anchor="end" font-family="monospace" font-size="10" fill="{col}">{m[k][-1]:.0f}</text>')
    out.append("</svg>")
    legend = "".join(f'<span style="margin-right:16px;"><span style="display:inline-block;width:10px;height:3px;background:{c};vertical-align:middle;margin-right:5px;"></span>{lab}</span>' for _, c, lab in series)
    return "".join(out) + f'<div class="note" style="margin-top:4px;">{legend} &nbsp; (start = 100, log scale, month-end)</div>'


def paper_section(sim, live, rule):
    p = sim["params"] if sim else dict(start_equity=SIM.START_EQUITY, weight=SIM.WEIGHT, max_pos=SIM.MAX_POS)
    start_eq = p["start_equity"]
    # ---- live account
    if live is None:
        live_html = f"""<div class="buy-stats" style="border-top:none;padding-top:0;margin-top:0;">
          <div><div class="buy-stat-l">Paper account</div><span>{usd(start_eq)}</span></div>
          <div><div class="buy-stat-l">Status</div><span style="color:var(--gold)">waiting for the first entry (next close after the first signal)</span></div></div>"""
    else:
        E, pos = live["E"], live["positions"]
        eq = float(E["equity"].iloc[-1])
        inv = float(E["exposure"].iloc[-1])
        n_open = int((~pos["closed"]).sum()) if len(pos) else 0
        col = "var(--green)" if eq >= start_eq else "var(--red)"
        live_html = f"""<div class="buy-stats" style="border-top:none;padding-top:0;margin-top:0;">
          <div><div class="buy-stat-l">Account value</div><span style="color:{col}">{usd(eq)}</span></div>
          <div><div class="buy-stat-l">Return since {live['start'].strftime('%d %b %Y')}</div><span style="color:{col}">{(eq / start_eq - 1) * 100:+.2f}%</span></div>
          <div><div class="buy-stat-l">SPY over the same days</div><span>{live['spy_ret'] * 100:+.2f}%</span></div>
          <div><div class="buy-stat-l">Invested</div><span>{inv * 100:.0f}%</span></div>
          <div><div class="buy-stat-l">Open positions</div><span>{n_open}</span></div>
          <div><div class="buy-stat-l">Picks skipped (account full)</div><span>{live['skipped']}</span></div></div>"""
        rows = ""
        for _, r in pos.sort_values("entry_date").iterrows():
            c = "var(--green)" if r["dollars"] >= 0 else "var(--red)"
            rows += (f'<tr><td class="mono">{r["ticker"]}</td><td class="mono">{pd.Timestamp(r["entry_date"]).strftime("%d %b %Y")}</td>'
                     f'<td class="mono" style="text-align:right;">{r["entry_px"]:.2f}</td><td class="mono" style="text-align:right;">{r["last_px"]:.2f}</td>'
                     f'<td class="mono" style="text-align:right;">{usd(r["alloc"])}</td>'
                     f'<td class="mono" style="text-align:right;color:{c};">{usd(r["dollars"], 0, True)} ({r["ret"] * 100:+.1f}%)</td>'
                     f'<td class="mono">{"closed " + pd.Timestamp(r["exit_date"]).strftime("%d %b") if r["closed"] else "open"}</td></tr>')
        live_html += f"""<div class="card" style="padding:0;overflow:hidden;margin-top:14px;"><table class="tilt-table"><thead><tr>
          <th>ETF</th><th>Bought</th><th style="text-align:right">Entry</th><th style="text-align:right">Last</th><th style="text-align:right">Size</th>
          <th style="text-align:right">P&amp;L</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table></div>"""
    # ---- backtest
    bt = ""
    if sim:
        s = sim["stats"]
        w = sim["worst"]
        yrs = sim["yearly"]
        yrow = "".join(f'<td class="mono" style="text-align:right;color:{"var(--green)" if v["strategy"] >= 0 else "var(--red)"};">{v["strategy"] * 100:+.0f}%</td>' for v in yrs.values())
        yrow_s = "".join(f'<td class="mono" style="text-align:right;color:var(--muted);">{v["spy"] * 100:+.0f}%</td>' for v in yrs.values())
        yhead = "".join(f'<th style="text-align:right">{y[2:]}</th>' for y in yrs)
        bt = f"""
    <div class="section-title" style="margin-top:26px;">What following every pick would have done, 2010 &ndash; {sim['data_through'][:4]} (walk-forward rules, same sizing)</div>
    {paper_svg(sim['monthly'])}
    <table class="tilt-table" style="margin-top:12px;"><thead><tr><th></th><th style="text-align:right">Total return</th><th style="text-align:right">Per year</th>
      <th style="text-align:right">Worst drop</th></tr></thead><tbody>
      <tr><td>Paper account (rule picks)</td><td class="mono" style="text-align:right;">{s['total_return'] * 100:+.0f}%</td><td class="mono" style="text-align:right;">{s['cagr'] * 100:.1f}%</td><td class="mono" style="text-align:right;color:var(--red);">{s['max_drawdown'] * 100:.0f}%</td></tr>
      <tr><td>SPY at the account's average exposure ({s['avg_exposure'] * 100:.0f}%) + T-bills</td><td class="mono" style="text-align:right;">{s['matched_total'] * 100:+.0f}%</td><td class="mono" style="text-align:right;">{s['matched_cagr'] * 100:.1f}%</td><td class="mono" style="text-align:right;color:var(--red);">{s['matched_dd'] * 100:.0f}%</td></tr>
      <tr><td>SPY buy and hold (fully invested)</td><td class="mono" style="text-align:right;">{s['bench_total'] * 100:+.0f}%</td><td class="mono" style="text-align:right;">{s['bench_cagr'] * 100:.1f}%</td><td class="mono" style="text-align:right;color:var(--red);">{s['bench_dd'] * 100:.0f}%</td></tr>
    </tbody></table>
    <div style="overflow-x:auto;margin-top:12px;"><table class="tilt-table"><thead><tr><th>Year</th>{yhead}</tr></thead><tbody>
      <tr><td>Paper account</td>{yrow}</tr><tr><td style="color:var(--muted);">SPY</td>{yrow_s}</tr></tbody></table></div>
    <div class="note"><b>Read this plainly:</b> following these picks with 10% per ETF would have returned about {s['cagr'] * 100:.1f}% a year, less than SPY at the same average
    exposure ({s['matched_cagr'] * 100:.1f}%) and far less than holding SPY ({s['bench_cagr'] * 100:.1f}%). The account was invested only {s['avg_exposure'] * 100:.0f}% of the time on average,
    then went from {w['exposure_peak'] * 100:.0f}% to {w['exposure_trough'] * 100:.0f}% invested as dips piled up in the February&ndash;March 2020 fall and lost {abs(w['dd']) * 100:.0f}%
    ({w['peak']} to {w['trough']}). The picks that were funded won {sim['funded_hit'] * 100:.0f}% of the time for an average {sim['funded_mean_pnl'] * 100:+.2f}% per pick
    ({sim['n_funded']} funded, {sim['n_skipped']} skipped because the account was full or already held the ETF), far below the {rule['horizons']['21']['oos']['pnl'] * 100:+.2f}% per bet in the
    evidence table, because that table counts every qualifying ETF on every day, mostly the same market bounce many times over. The rule's per-bet statistics are real;
    they do not add up to a strategy that beats holding SPY.</div>"""
    return f"""<div class="section">
  <div class="section-title">Paper Account &mdash; {usd(start_eq)} of simulated money, {p['weight'] * 100:.0f}% per pick, at most {p['max_pos']} positions</div>
  <div class="card">
    {live_html}
    <div class="note">Rules: each pick is bought at its entry close with {p['weight'] * 100:.0f}% of the account's value, no leverage; at most {p['max_pos']} positions at once and one per ETF
    (the deepest 1-month drop is funded first); sold at the target, stop or time limit; 5 bps round-trip cost; idle cash earns the 13-week T-bill yield. Prices are in US$ with no currency conversion.
    Nothing here is real money.</div>
    {bt}
  </div>
</div>
"""


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


def render(rule, sigs, feats, log, generated_at, panel_end, paper_html=""):
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

{paper_html}

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
    sim = json.loads(SIM_PATH.read_text()) if SIM_PATH.exists() else None
    paper_html = paper_section(sim, live_account(log, prices), rule)
    html = render(rule, sigs, feats, log, now_utc, panel_end, paper_html)
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Stock board: {len(sigs)} qualifying ({', '.join(g['ticker'] for g in sigs) or 'none'}); "
          f"log {len(log)} rows ({int((log['status'] == 'RESOLVED').sum())} resolved); prices through {panel_end.date()}")


if __name__ == "__main__":
    main()
