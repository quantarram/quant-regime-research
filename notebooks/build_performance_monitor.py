"""
Performance Monitor -- one page that scores every other dashboard
=================================================================
Reads the prediction ledgers the daily pipeline already keeps and shows, for
each dashboard, what it called and what then happened:

  Gold dashboard      gold_predictions.csv        direction calls, 21d/63d scored
  Metals dashboard    metals_predictions.csv      Silver + Platinum, same scoring
  Portfolio tilt      portfolio_predictions.csv   tilt P&L vs neutral weights, resolved windows
                      ibkr_paper_ledger.csv       simulated NAV paths, tilt / neutral / hold-to-horizon
  Predictor           predictor_forecasts.csv     price forecasts vs realised closes
  Football checklist  football_betting/output/qualifying_log.csv  qualifying picks
  CPE dashboard       cpe_signal_events.csv       entry events scored against the frozen CPE table

Every panel shows raw levels next to the relevant baseline (an always-long
call for direction signals, the neutral portfolio for the tilt, a no-change
forecast for the price forecasts, the odds-implied win rate for football, the live unconditional frequency for CPE) and
always prints n. Nothing is filtered after the fact: every resolved row counts.

History: each run first writes the latest complete day's snapshot to monitor_history.csv
(see monitor_history.py); the trend charts read it back.

Run: python build_performance_monitor.py      Output: performance_monitor.html
"""
import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import predictor_ledger as pled  # noqa: E402
import cpe_signal_ledger as cled  # noqa: E402
import monitor_history as mh  # noqa: E402

SGT = ZoneInfo("Asia/Singapore")
NOW_SGT = datetime.now(timezone.utc).astimezone(SGT)


def _csv(name):
    return pd.read_csv(os.path.join(HERE, name))


def _r(x, d=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


# ---------------------------------------------------------------------------
# direction-call dashboards (gold, metals)
# ---------------------------------------------------------------------------
def direction_block(df, label):
    """df: resolved rows with direction, horizon_days, actual_return_pct, prediction_correct, outcome_date."""
    df = df.copy()
    df["prediction_correct"] = df["prediction_correct"].astype(float)
    df["actual_return_pct"] = df["actual_return_pct"].astype(float)
    df["outcome_date"] = pd.to_datetime(df["outcome_date"])
    df["rose"] = df["actual_return_pct"] > 0
    df["within3"] = df["actual_return_pct"].abs() < 3
    out = {"label": label, "n_resolved": int(len(df)), "rows": [], "series": {}}
    for (h, d), g in df.groupby(["horizon_days", "direction"]):
        base_all = df[df["horizon_days"] == h]
        base = base_all["rose"].mean() if d == "BULLISH" else base_all["within3"].mean()
        out["rows"].append({
            "horizon": int(h), "direction": d, "n": int(len(g)),
            "hit": _r(g["prediction_correct"].mean(), 4),
            "baseline": _r(base, 4),
            "baseline_n": int(len(base_all)),
            "mean_return": _r(g["actual_return_pct"].mean(), 2),
        })
    out["rows"].sort(key=lambda r: (r["horizon"], r["direction"]))
    # BULLISH calls only: cumulative hit rate by outcome date, next to the cumulative share of ALL resolved
    # windows where the price rose (the always-long baseline). NEUTRAL calls use a different rule (within +/-3%)
    # and are reported in the table instead of being mixed into one line.
    bull = df[df["direction"] == "BULLISH"].sort_values("outcome_date")
    allw = df.sort_values("outcome_date")
    out["series"] = {
        "hit_dates": [d.strftime("%Y-%m-%d") for d in bull["outcome_date"]],
        "hit": [_r(v, 4) for v in bull["prediction_correct"].expanding(min_periods=10).mean()],
        "rose_dates": [d.strftime("%Y-%m-%d") for d in allw["outcome_date"]],
        "rose": [_r(v, 4) for v in allw["rose"].expanding(min_periods=10).mean()],
    }
    out["bull_n"] = int(len(bull))
    out["bull_hit"] = _r(bull["prediction_correct"].mean(), 4) if len(bull) else None
    out["overall_rose"] = _r(allw["rose"].mean(), 4)
    out["last_outcome"] = allw["outcome_date"].max().strftime("%Y-%m-%d") if len(allw) else None
    return out


def gold_block():
    g = _csv("gold_predictions.csv")
    res = g[g["status"] == "RESOLVED"]
    b = direction_block(res, "Gold")
    b["n_pending"] = int((g["status"] == "PENDING").sum())
    return b


def metals_block():
    m = _csv("metals_predictions.csv")
    out = {}
    for metal in ("Silver", "Platinum"):
        sub = m[m["metal"] == metal]
        b = direction_block(sub[sub["status"] == "RESOLVED"], metal)
        b["n_pending"] = int((sub["status"] == "PENDING").sum())
        out[metal] = b
    return out


# ---------------------------------------------------------------------------
# portfolio tilt + IBKR paper ledger
# ---------------------------------------------------------------------------
def portfolio_block():
    p = _csv("portfolio_predictions.csv")
    res = p[p["status"] == "RESOLVED"].copy()
    for c in ("tilt_pnl", "neutral_pnl", "tilt_beat_neutral"):
        res[c] = res[c].astype(float)
    rows = []
    for h, g in res.groupby("dominant_horizon_days"):
        rows.append({"horizon": int(h), "n": int(len(g)),
                     "tilt": _r(g["tilt_pnl"].mean(), 3), "neutral": _r(g["neutral_pnl"].mean(), 3),
                     "diff": _r((g["tilt_pnl"] - g["neutral_pnl"]).mean(), 3),
                     "beat": _r(g["tilt_beat_neutral"].mean(), 4)})
    allr = {"n": int(len(res)), "tilt": _r(res["tilt_pnl"].mean(), 3), "neutral": _r(res["neutral_pnl"].mean(), 3),
            "diff": _r((res["tilt_pnl"] - res["neutral_pnl"]).mean(), 3),
            "beat": _r(res["tilt_beat_neutral"].mean(), 4)}
    led = _csv("ibkr_paper_ledger.csv")
    nav = {"dates": led["date"].tolist()}
    for k in ("tilt", "neutral", "hth"):
        nav[k] = [_r(v, 3) for v in led[f"cum_return_{k}"]]
    return {"rows": rows, "all": allr, "n_pending": int((p["status"] == "PENDING").sum()),
            "nav": nav, "ledger_start": led["date"].iloc[0], "ledger_last": led["date"].iloc[-1],
            "ledger_days": int(len(led))}


# ---------------------------------------------------------------------------
# football checklist
# ---------------------------------------------------------------------------
def football_block():
    q = _csv(os.path.join("football_betting", "output", "qualifying_log.csv"))
    res = q[q["status"] == "RESOLVED"].copy()
    res["won"] = res["won"].astype(float)
    res["start"] = pd.to_datetime(res["start_time"])
    res = res.sort_values("start")
    placed = res[res["bet_placed"].astype(str).str.lower() == "true"].copy()
    placed["actual_pnl"] = placed["actual_pnl"].astype(float)
    return {
        "n_picks": int(len(q)), "n_resolved": int(len(res)), "n_pending": int((q["status"] == "PENDING").sum()),
        "wins": int(res["won"].sum()),
        "hit": _r(res["won"].mean(), 4) if len(res) else None,
        "implied": _r((1 / res["odds"].astype(float)).mean(), 4) if len(res) else None,
        "odds_min": _r(res["odds"].min(), 2), "odds_max": _r(res["odds"].max(), 2),
        "n_placed_resolved": int(len(placed)),
        "stake": _r(placed["actual_stake_sgd"].astype(float).sum(), 2),
        "pnl": _r(placed["actual_pnl"].sum(), 2),
        "series": {"dates": [d.strftime("%Y-%m-%d") for d in placed["start"]],
                   "cum_pnl": [_r(v, 2) for v in placed["actual_pnl"].cumsum()],
                   "labels": placed["fixture"].tolist()},
    }


def history_block():
    """monitor_history.csv -> {"dashboard.metric": {dates, values, n}} for the trend charts."""
    h = mh.load_history()
    out = {}
    for (d, m), g in h.groupby(["dashboard", "metric"]):
        g = g.sort_values("date")
        out[f"{d}.{m}"] = {"dates": g["date"].tolist(), "values": [round(float(v), 5) for v in g["value"]],
                           "n": [int(n) for n in g["n"]]}
    return out


def build_bundle():
    mh.append_today()
    return {
        "generated": NOW_SGT.strftime("%Y-%m-%d %H:%M:%S"),
        "gold": gold_block(),
        "metals": metals_block(),
        "portfolio": portfolio_block(),
        "predictor": pled.live_stats(pled.load_ledger()),
        "football": football_block(),
        "cpe": cled.live_stats(),
        "history": history_block(),
    }


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Performance Monitor — Dashboards Scored</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{--accent:#5B8DBE;--accent2:#7FAAD1;--bg:#0C0E0D;--s1:#141614;--s2:#1B1E1B;--s3:#222522;
--bdr:#2C302C;--bdr2:#3A3F3A;--text:#DDE8DD;--text2:#7A8F7A;--text3:#4A5A4A;--warn:#C8A84A;--rose:#B5726A;--ok:#7FB08A;
--mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;--r:10px;--gap:16px;--pad:20px;}
*{box-sizing:border-box;margin:0;padding:0;}
html{font-size:15px;}
body{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.6;min-height:100vh;}
header{background:linear-gradient(160deg,#0C0E0D,#141a1e,#0C0E0D);border-bottom:1px solid var(--bdr);padding:18px var(--pad);
  display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;}
.h-title{font-family:var(--mono);font-size:clamp(13px,2vw,17px);font-weight:600;color:var(--accent2);}
.h-sub{font-family:var(--mono);font-size:11px;color:var(--text2);letter-spacing:.06em;text-transform:uppercase;margin-top:2px;}
.h-meta{font-family:var(--mono);font-size:11px;color:var(--text2);text-align:right;line-height:1.8;}
main{max-width:1180px;margin:0 auto;padding:20px var(--pad) 10px;}
.intro{font-size:13px;color:var(--text2);line-height:1.75;margin-bottom:18px;max-width:900px;}
.intro b{color:var(--text);}
.overview{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:var(--gap);margin-bottom:26px;}
.ov{background:var(--s1);border:1px solid var(--bdr);border-radius:var(--r);padding:16px;display:block;text-decoration:none;color:inherit;}
.ov:hover{border-color:var(--bdr2);}
.ov-name{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--accent2);text-transform:uppercase;letter-spacing:.05em;}
.ov-big{font-family:var(--mono);font-size:24px;font-weight:600;margin-top:8px;}
.ov-big small{font-size:11px;color:var(--text2);font-weight:400;}
.ov-line{font-family:var(--mono);font-size:11px;color:var(--text2);margin-top:4px;line-height:1.6;}
.pill{display:inline-block;font-family:var(--mono);font-size:9px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
  padding:2px 7px;border-radius:99px;border:1px solid var(--bdr2);color:var(--text2);margin-left:8px;vertical-align:1px;}
.pill.live{color:var(--ok);border-color:#2F4A36;}
.pill.gap{color:var(--warn);border-color:#5A4D22;}
section{background:var(--s1);border:1px solid var(--bdr);border-radius:var(--r);padding:var(--pad);margin-bottom:20px;}
.s-title{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--accent2);text-transform:uppercase;letter-spacing:.06em;}
.s-sub{font-family:var(--mono);font-size:11px;color:var(--text2);margin:4px 0 14px;line-height:1.7;}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:var(--gap);}
.c-label{font-family:var(--mono);font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:.06em;margin-bottom:2px;}
table{width:100%;border-collapse:collapse;margin-top:14px;font-size:12px;}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--bdr);font-family:var(--mono);}
th:first-child,td:first-child{text-align:left;}
th{color:var(--text2);font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:.05em;}
.note{font-size:12px;color:var(--text2);margin-top:12px;line-height:1.7;}
.table-wrap{overflow-x:auto;}
footer{padding:24px var(--pad) 40px;text-align:center;font-family:var(--mono);font-size:10px;color:var(--text3);}
</style>
</head>
<body>
<header>
  <div><div class="h-title">Performance Monitor</div>
  <div class="h-sub">Every dashboard, scored against what happened</div></div>
  <div class="h-meta" id="hmeta"></div>
</header>
<main>
<p class="intro">Each panel below takes one dashboard's own prediction ledger and reports <b>what it called and what then happened</b>,
next to the baseline that gives the number meaning: an always-long call for the direction signals, the neutral-weight
portfolio for the tilt, a no-change forecast for the price forecasts, the odds-implied win rate for the football picks.
Sample sizes are printed everywhere; every resolved row counts.</p>
<div class="overview" id="overview"></div>
<div id="panels"></div>
</main>
<footer>performance monitor &mdash; quantarram/quant-regime-research &mdash; research record, not investment advice</footer>
<script>
const D = __DATA__;
const pct = (v, d) => (v === null || v === undefined) ? '&ndash;' : (v * 100).toFixed(d === undefined ? 0 : d) + '%';
const num = (v, d) => (v === null || v === undefined) ? '&ndash;' : v.toFixed(d === undefined ? 2 : d);
const sgn = (v, d) => (v === null || v === undefined) ? '&ndash;' : (v >= 0 ? '+' : '') + v.toFixed(d === undefined ? 2 : d);
document.getElementById('hmeta').innerHTML = 'Generated ' + D.generated + ' SGT';
const BASE = {paper_bgcolor:'transparent', plot_bgcolor:'transparent', height:240,
  margin:{l:42,r:10,t:6,b:44}, font:{family:'IBM Plex Mono', size:10, color:'#7A8F7A'},
  xaxis:{tickfont:{size:9}, gridcolor:'#1B1E1B'}, yaxis:{gridcolor:'#1B1E1B', zeroline:false},
  legend:{orientation:'h', y:-0.3, font:{size:9}}};
const CFG = {displayModeBar:false, responsive:true};
const merge = (o) => Object.assign({}, BASE, o);
const panels = document.getElementById('panels'), overview = document.getElementById('overview');
// trend helpers: history is keyed "dashboard.metric" -> {dates, values, n}
const H = (k) => D.history[k];
function diffSeries(a, b, minN) {   // a - b on shared dates, points with n >= minN only
  const A = H(a), B = H(b); if (!A || !B) return null;
  const bi = {}; B.dates.forEach((d, i) => bi[d] = i);
  const x = [], y = [];
  A.dates.forEach((d, i) => { if (d in bi && A.n[i] >= minN) { x.push(d); y.push(A.values[i] - B.values[bi[d]]); } });
  return x.length ? {x: x, y: y} : null;
}
function lineSeries(k, minN, scale) {
  const A = H(k); if (!A) return null;
  const x = [], y = [];
  A.dates.forEach((d, i) => { if (A.n[i] >= minN) { x.push(d); y.push(A.values[i] * (scale || 1)); } });
  return x.length ? {x: x, y: y} : null;
}
const ZERO = [{type:'line', xref:'paper', x0:0, x1:1, y0:0, y1:0, line:{color:'#3A3F3A', width:1}}];

function addOverview(id, name, status, big, small, lines) {
  const a = document.createElement('a'); a.className = 'ov'; a.href = '#' + id;
  a.innerHTML = '<div class="ov-name">' + name + '<span class="pill ' + (status === 'gap' ? 'gap' : 'live') + '">' +
    (status === 'gap' ? 'not tracked live' : 'tracked live') + '</span></div>' +
    '<div class="ov-big">' + big + ' <small>' + small + '</small></div>' +
    lines.map(l => '<div class="ov-line">' + l + '</div>').join('');
  overview.appendChild(a);
}
function addPanel(id, title, sub, bodyHtml) {
  const s = document.createElement('section'); s.id = id;
  s.innerHTML = '<div class="s-title">' + title + '</div><div class="s-sub">' + sub + '</div>' + bodyHtml;
  panels.appendChild(s); return s;
}

// ------------------------------------------------------------ direction signals (gold, metals)
function directionTable(b) {
  let t = '<div class="table-wrap"><table><tr><th>Horizon</th><th>Call</th><th>Resolved</th><th>Hit rate</th>' +
    '<th>Baseline</th><th>Mean realised move</th></tr>';
  b.rows.forEach(r => {
    const bl = r.direction === 'BULLISH' ? 'price rose, all ' + r.baseline_n + ' windows' : 'within &plusmn;3%, all ' + r.baseline_n + ' windows';
    t += '<tr><td>' + r.horizon + 'd</td><td>' + r.direction + '</td><td>' + r.n + '</td><td>' + pct(r.hit) + '</td>' +
      '<td>' + pct(r.baseline) + ' <span style="color:var(--text3)">(' + bl + ')</span></td><td>' + sgn(r.mean_return) + '%</td></tr>';
  });
  return t + '</table></div>';
}
function directionPanel(id, title, blocks, scoring) {
  // blocks: list of {label, b}
  let body = '<div class="charts">';
  blocks.forEach((x, i) => { body += '<div><div class="c-label">' + x.label + ' &mdash; BULLISH-call hit rate vs share of all windows where price rose, cumulative by outcome date, shown once 10 outcomes are in</div><div id="' + id + 'c' + i + '"></div></div>'; });
  body += '</div>';
  blocks.forEach(x => { body += '<div class="c-label" style="margin-top:14px">' + x.label + '</div>' + directionTable(x.b); });
  body += '<div class="note">' + scoring + '</div>';
  const nres = blocks.reduce((a, x) => a + x.b.n_resolved, 0), npend = blocks.reduce((a, x) => a + x.b.n_pending, 0);
  addPanel(id, title, nres + ' resolved &middot; ' + npend + ' pending', body);
  blocks.forEach((x, i) => {
    const s = x.b.series;
    Plotly.newPlot(id + 'c' + i, [
      {x:s.hit_dates, y:s.hit.map(v => v*100), name:'Bullish hit rate', mode:'lines', line:{color:'#5B8DBE', width:2}},
      {x:s.rose_dates, y:s.rose.map(v => v*100), name:'Price rose (all)', mode:'lines', line:{color:'#B5726A', width:1.5, dash:'dot'}}
    ], merge({yaxis:{gridcolor:'#1B1E1B', zeroline:false, ticksuffix:'%', range:[0,100]}}), CFG);
  });
}
const SCORING = 'Scoring (as logged): a BULLISH call is correct if the price is higher at the horizon; a NEUTRAL call is correct if the move stays within &plusmn;3%. ' +
  'The baseline for a BULLISH call is how often the price rose over all resolved windows of that horizon, whatever the signal said; ' +
  'for NEUTRAL it is how often the move stayed within &plusmn;3%. Hit rate above its baseline is the signal adding something beyond the market&rsquo;s own drift.';

addOverview('gold', 'Gold', 'live', pct(D.gold.bull_hit), 'bullish-call hit', [D.gold.bull_n + ' bullish of ' + D.gold.n_resolved + ' resolved (' + D.gold.n_pending + ' pending)', 'price rose in ' + pct(D.gold.overall_rose) + ' of all windows']);
const metN = D.metals.Silver.n_resolved + D.metals.Platinum.n_resolved;
const metBull = D.metals.Silver.bull_n + D.metals.Platinum.bull_n;
const metHit = (D.metals.Silver.bull_hit * D.metals.Silver.bull_n + D.metals.Platinum.bull_hit * D.metals.Platinum.bull_n) / metBull;
const metRose = (D.metals.Silver.overall_rose * D.metals.Silver.n_resolved + D.metals.Platinum.overall_rose * D.metals.Platinum.n_resolved) / metN;
addOverview('metals', 'Precious metals', 'live', pct(metHit), 'bullish-call hit', [metBull + ' bullish of ' + metN + ' resolved (Silver, Platinum)', 'price rose in ' + pct(metRose) + ' of all windows']);
const P = D.portfolio;
addOverview('portfolio', 'Portfolio tilt', 'live', sgn(P.all.diff, 2) + ' <small>pts vs neutral</small>', '', [P.all.n + ' resolved windows, tilt avg ' + sgn(P.all.tilt, 2) + '%', 'neutral avg ' + sgn(P.all.neutral, 2) + '%']);
const PR = D.predictor;
addOverview('predictor', 'Predictor (22 instruments)', 'live', PR.n_resolved + ' <small>of ' + PR.n_logged.toLocaleString() + ' resolved</small>', '', [PR.n_instruments_resolved + ' of ' + PR.n_instruments + ' instruments have an outcome', 'long horizons resolve from 2027']);
const F = D.football;
addOverview('football', 'Football checklist', 'live', F.wins + '/' + F.n_resolved, '<small>picks won</small>', ['odds-implied win rate ' + pct(F.implied), 'S$' + sgn(F.pnl, 2) + ' on S$' + num(F.stake, 2) + ' staked']);
const C = D.cpe, c63 = C.by_horizon.find(r => r.horizon === 63);
addOverview('cpe', 'CPE dashboard (161 instruments)', 'live', c63 ? pct(c63.hit) : '&ndash;', '<small>63d realised</small>',
  [c63 ? 'claimed ' + pct(c63.claimed) + ', live base ' + pct(c63.live_base) + ' (n=' + c63.n.toLocaleString() + ')' : 'no 63d outcomes yet',
   C.n_resolved.toLocaleString() + ' of ' + C.n_rows.toLocaleString() + ' signal rows resolved']);

directionPanel('gold', 'Gold dashboard', [{label:'Gold', b:D.gold}], SCORING);
directionPanel('metals', 'Precious metals dashboard', [{label:'Silver', b:D.metals.Silver}, {label:'Platinum', b:D.metals.Platinum}], SCORING);

// ------------------------------------------------------------ portfolio
(function () {
  let t = '<div class="table-wrap"><table><tr><th>Dominant horizon</th><th>Resolved</th><th>Tilt avg P&amp;L</th><th>Neutral avg P&amp;L</th><th>Tilt &minus; neutral</th><th>Tilt ahead in</th></tr>';
  P.rows.forEach(r => { t += '<tr><td>' + r.horizon + 'd</td><td>' + r.n + '</td><td>' + sgn(r.tilt) + '%</td><td>' + sgn(r.neutral) + '%</td><td>' + sgn(r.diff, 3) + ' pts</td><td>' + pct(r.beat) + '</td></tr>'; });
  t += '<tr><td><b>All</b></td><td>' + P.all.n + '</td><td>' + sgn(P.all.tilt) + '%</td><td>' + sgn(P.all.neutral) + '%</td><td>' + sgn(P.all.diff, 3) + ' pts</td><td>' + pct(P.all.beat) + '</td></tr></table></div>';
  const body = '<div class="charts"><div><div class="c-label">Average resolved-window P&amp;L (%), tilt vs neutral, by horizon</div><div id="pfBars"></div></div>' +
    '<div><div class="c-label">Simulated paper portfolio, cumulative return since ' + P.ledger_start + ' (%)</div><div id="pfNav"></div></div>' +
    '<div><div class="c-label">Tilt minus neutral, average resolved-window P&amp;L, as the record built up (points; shown from 10 windows)</div><div id="pfTrend"></div></div></div>' + t +
    '<div class="note">The tilt portfolio re-weights daily to the dashboard&rsquo;s bullish tilts; the neutral portfolio keeps constant weights; hold-to-horizon opens a position when a sleeve fires and holds it until the shortest firing horizon elapses. ' +
    'The paper ledger is simulated (no broker), runs from ' + P.ledger_start + ' (' + P.ledger_days + ' daily rows, last ' + P.ledger_last + '), includes estimated trading-cost drag for the tilt and hold-to-horizon tracks, and its bullish-only tilt is a different rule from the resolved-window tilt in the table.</div>';
  addPanel('portfolio', 'Portfolio tilt dashboard', P.all.n + ' resolved windows &middot; ' + P.n_pending + ' pending', body);
  Plotly.newPlot('pfBars', [
    {x:P.rows.map(r => r.horizon + 'd (n=' + r.n + ')'), y:P.rows.map(r => r.tilt), name:'Tilt', type:'bar', marker:{color:'#5B8DBE'}, text:P.rows.map(r => r.tilt.toFixed(2)), textposition:'outside'},
    {x:P.rows.map(r => r.horizon + 'd (n=' + r.n + ')'), y:P.rows.map(r => r.neutral), name:'Neutral', type:'bar', marker:{color:'#B5726A'}, text:P.rows.map(r => r.neutral.toFixed(2)), textposition:'outside'}
  ], merge({barmode:'group', xaxis:{type:'category', tickfont:{size:9}}, yaxis:{gridcolor:'#1B1E1B', zeroline:false, ticksuffix:'%', range:[0, Math.max(...P.rows.map(r => Math.max(r.tilt, r.neutral))) * 1.2]}}), CFG);
  const pt = diffSeries('portfolio.tilt_pnl_avg', 'portfolio.neutral_pnl_avg', 10);
  if (pt) Plotly.newPlot('pfTrend', [{x:pt.x, y:pt.y, mode:'lines', line:{color:'#5B8DBE', width:2}, name:'Tilt minus neutral'}],
    merge({shapes:ZERO, yaxis:{gridcolor:'#1B1E1B', zeroline:false}, showlegend:false}), CFG);
  Plotly.newPlot('pfNav', [
    {x:P.nav.dates, y:P.nav.tilt, name:'Tilt', mode:'lines', line:{color:'#5B8DBE', width:2}},
    {x:P.nav.dates, y:P.nav.neutral, name:'Neutral', mode:'lines', line:{color:'#B5726A', width:1.5, dash:'dot'}},
    {x:P.nav.dates, y:P.nav.hth, name:'Hold-to-horizon', mode:'lines', line:{color:'#C8A84A', width:1.5}}
  ], merge({yaxis:{gridcolor:'#1B1E1B', zeroline:false, ticksuffix:'%'}}), CFG);
})();

// ------------------------------------------------------------ predictor
(function () {
  const done = PR.by_horizon.filter(r => r.n > 0), lab = done.map(r => r.horizon + 'd (n=' + r.n + ')');
  let t = '<div class="table-wrap"><table><tr><th>Horizon</th><th>Instruments</th><th>Resolved</th><th>Model MAPE</th><th>No-change MAPE</th><th>In 80% band</th><th>In 50% band</th><th>Direction</th><th>Pending</th><th>First resolves</th></tr>';
  PR.by_horizon.forEach(r => {
    const h = r.n > 0;
    t += '<tr><td>' + r.horizon + 'd</td><td>' + r.tickers.join(', ') + '</td><td>' + r.n + '</td><td>' + (h ? r.mape.toFixed(2) + '%' : '&ndash;') + '</td><td>' + (h ? r.naive_mape.toFixed(2) + '%' : '&ndash;') +
      '</td><td>' + (h ? pct(r.cov80) : '&ndash;') + '</td><td>' + (h ? pct(r.cov50) : '&ndash;') + '</td><td>' + (h ? pct(r.direction) : '&ndash;') + '</td><td>' + r.n_pending + '</td><td>' + (r.first_pending_target || '&ndash;') + '</td></tr>';
  });
  t += '</table></div>';
  const body = '<div class="charts"><div><div class="c-label">Median-forecast error vs no-change, by horizon (MAPE, %)</div><div id="prErr"></div></div>' +
    '<div><div class="c-label">Share of outcomes inside the forecast band (%)</div><div id="prCov"></div></div>' +
    '<div><div class="c-label">Forecast error minus no-change error, as the record built up (points; below zero = forecaster ahead; from 20 outcomes)</div><div id="prTrend"></div></div></div>' + t +
    '<div class="note">Forecasts logged since ' + PR.first_forecast + ', scored on the close each forecast&rsquo;s own horizon (in trading days) after it was made, exactly as first published. ' +
    'Each instrument has one fixed horizon, so the 126d, 189d and 252d instruments show no outcome until 2027. Full instrument-level detail sits on the <a href="predictor_dashboard.html" style="color:var(--accent2)">predictor dashboard</a>.</div>';
  addPanel('predictor', 'Predictor dashboard', PR.n_resolved + ' resolved &middot; ' + PR.n_pending.toLocaleString() + ' pending &middot; ' + PR.n_instruments_resolved + ' of ' + PR.n_instruments + ' instruments with an outcome', body);
  if (!done.length) return;
  const ptr = [1, 5, 21].map(h => ({h: h, s: diffSeries('predictor.mape_' + h + 'd', 'predictor.no_change_mape_' + h + 'd', 20)})).filter(o => o.s);
  if (ptr.length) Plotly.newPlot('prTrend', ptr.map((o, i) => ({x:o.s.x, y:o.s.y, mode:'lines', name:o.h + 'd', line:{width:2, color:['#5B8DBE', '#C8A84A', '#7FB08A'][i]}})),
    merge({shapes:ZERO, yaxis:{gridcolor:'#1B1E1B', zeroline:false}}), CFG);
  Plotly.newPlot('prErr', [
    {x:lab, y:done.map(r => r.mape), name:'Model median', type:'bar', marker:{color:'#5B8DBE'}, text:done.map(r => r.mape.toFixed(2)), textposition:'outside'},
    {x:lab, y:done.map(r => r.naive_mape), name:'No change', type:'bar', marker:{color:'#B5726A'}, text:done.map(r => r.naive_mape.toFixed(2)), textposition:'outside'}
  ], merge({barmode:'group', xaxis:{type:'category', tickfont:{size:9}}, yaxis:{gridcolor:'#1B1E1B', zeroline:false, ticksuffix:'%', range:[0, Math.max(...done.map(r => Math.max(r.mape, r.naive_mape))) * 1.25]}}), CFG);
  Plotly.newPlot('prCov', [
    {x:lab, y:done.map(r => r.cov80*100), name:'q10-q90 (nominal 80%)', type:'bar', marker:{color:'#5B8DBE'}, text:done.map(r => (r.cov80*100).toFixed(0)), textposition:'outside'},
    {x:lab, y:done.map(r => r.cov50*100), name:'q25-q75 (nominal 50%)', type:'bar', marker:{color:'#7FAAD1'}, text:done.map(r => (r.cov50*100).toFixed(0)), textposition:'outside'}
  ], merge({barmode:'group', xaxis:{type:'category', tickfont:{size:9}}, yaxis:{gridcolor:'#1B1E1B', zeroline:false, ticksuffix:'%', range:[0,115]},
    shapes:[80,50].map(v => ({type:'line', xref:'paper', x0:0, x1:1, y0:v, y1:v, line:{color:'#C8A84A', width:1, dash:'dot'}}))}), CFG);
})();

// ------------------------------------------------------------ football
(function () {
  const s = F.series;
  const body = '<div class="charts"><div><div class="c-label">Win rate vs odds-implied win rate (%)</div><div id="fbBars"></div></div>' +
    '<div><div class="c-label">Cumulative profit on bets actually placed (S$), by match date</div><div id="fbPnl"></div></div></div>' +
    '<div class="table-wrap"><table><tr><th>Resolved picks</th><th>Won</th><th>Win rate</th><th>Odds-implied win rate</th><th>Odds range</th><th>Bets placed (resolved)</th><th>Staked</th><th>Profit</th></tr>' +
    '<tr><td>' + F.n_resolved + '</td><td>' + F.wins + '</td><td>' + pct(F.hit) + '</td><td>' + pct(F.implied, 1) + '</td><td>' + num(F.odds_min) + '&ndash;' + num(F.odds_max) + '</td><td>' + F.n_placed_resolved + '</td><td>S$' + num(F.stake) + '</td><td>S$' + sgn(F.pnl) + '</td></tr></table></div>' +
    '<div class="note">The checklist flags short-priced home favourites that clear its form thresholds; at these odds a bet wins small and loses large, so the win rate has to be read against the odds-implied rate, and the sample is still small (' + F.n_resolved + ' resolved picks, ' + F.n_pending + ' pending).</div>';
  addPanel('football', 'Football checklist', F.n_resolved + ' resolved &middot; ' + F.n_pending + ' pending', body);
  Plotly.newPlot('fbBars', [{x:['Actual win rate', 'Odds-implied'], y:[F.hit*100, F.implied*100], type:'bar', marker:{color:['#5B8DBE', '#B5726A']},
    text:[(F.hit*100).toFixed(0) + '%', (F.implied*100).toFixed(1) + '%'], textposition:'outside'}], merge({xaxis:{type:'category'}, yaxis:{gridcolor:'#1B1E1B', zeroline:false, ticksuffix:'%', range:[0,115]}}), CFG);
  Plotly.newPlot('fbPnl', [{x:s.dates, y:s.cum_pnl, text:s.labels, mode:'lines+markers', line:{color:'#7FB08A', width:2}, marker:{size:6}, name:'Cumulative profit', hovertemplate:'%{text}<br>S$%{y}<extra></extra>'}],
    merge({yaxis:{gridcolor:'#1B1E1B', zeroline:true, zerolinecolor:'#3A3F3A', tickprefix:'S$'}}), CFG);
})();

// ------------------------------------------------------------ CPE
(function () {
  const done = C.by_horizon.filter(r => r.n > 0), lab = done.map(r => r.horizon + 'd (n=' + r.n.toLocaleString() + ')');
  let t = '<div class="table-wrap"><table><tr><th>Horizon</th><th>Call</th><th>Resolved rows</th><th>Claimed CPE</th><th>Realised</th><th>Table base</th><th>Live base</th><th>Pending</th><th>First resolves</th></tr>';
  C.by_horizon.forEach(r => {
    if (!r.n) { t += '<tr><td>' + r.horizon + 'd</td><td>all</td><td>0</td><td colspan="4" style="text-align:center;color:var(--text3)">no outcomes yet</td><td>' + r.n_pending.toLocaleString() + '</td><td>~' + r.first_pending + '</td></tr>'; return; }
    ['bullish', 'bearish'].forEach(d => {
      const x = r[d]; if (!x) return;
      t += '<tr><td>' + r.horizon + 'd</td><td>' + d + '</td><td>' + x.n.toLocaleString() + '</td><td>' + pct(x.claimed, 1) + '</td><td>' + pct(x.hit, 1) + '</td><td>' + pct(x.base, 1) + '</td><td>' + pct(x.live_base, 1) + '</td><td>' + (d === 'bullish' ? r.n_pending.toLocaleString() : '') + '</td><td>' + (d === 'bullish' && r.first_pending ? '~' + r.first_pending : '') + '</td></tr>';
    });
  });
  t += '</table></div>';
  const body = '<div class="charts"><div><div class="c-label">Claimed CPE vs realised exceedance vs live base rate, by horizon (%)</div><div id="cpeBars"></div></div>' +
    '<div><div class="c-label">By claimed-CPE band, resolved rows (%)</div><div id="cpeCal"></div></div>' +
    '<div><div class="c-label">63-day rows as the record built up: claimed, realised, live base (%; from 100 rows)</div><div id="cpeTrend63"></div></div>' +
    '<div><div class="c-label">Realised minus live base, as the record built up (points; above zero = signal adds beyond the period&rsquo;s drift; from 100 rows)</div><div id="cpeEdge"></div></div></div>' + t +
    '<div class="note">Scores the CPE table (frozen at ' + C.table_end + ', sha ' + C.table_sha + ') on the period after it, from ' + C.first_entry + ' to ' + C.last_entry + '. ' +
    'Each event is a predictor entering its tail for the first time after a day outside it (' + C.n_events.toLocaleString() + ' events); it fans out to every gated table row for that predictor, ' + C.n_rows.toLocaleString() + ' signal rows in all. ' +
    'A row scores 1 if the target&rsquo;s forward move beyond its horizon landed past the frozen q<sub>Y</sub> threshold, as the table defines it. ' +
    '<b>Table base</b> is the unconditional frequency the table assumed (1 &minus; q<sub>Y</sub>); <b>live base</b> is how often that same event happened on <i>any</i> day of the live window, so realised above live base is what the signal adds beyond the period&rsquo;s own drift. ' +
    'Most of the table&rsquo;s claims sit at 126d to 300d, which cannot resolve before December 2026 to 2027; the 21d and 63d rows resolved so far are a small slice of the claim, from a handful of entry dates, and rows from one event are not independent of each other. ' +
    'Trend charts reconstruct the record as it stood on each date, using only outcomes known by then. Last price bar used: ' + C.last_bar + '.</div>';
  addPanel('cpe', 'CPE dashboard (161 instruments)', C.n_resolved.toLocaleString() + ' signal rows resolved &middot; ' + C.n_pending.toLocaleString() + ' pending', body);
  if (!done.length) return;
  Plotly.newPlot('cpeBars', [
    {x:lab, y:done.map(r => r.claimed*100), name:'Claimed CPE', type:'bar', marker:{color:'#5B8DBE'}, text:done.map(r => (r.claimed*100).toFixed(0)), textposition:'outside'},
    {x:lab, y:done.map(r => r.hit*100), name:'Realised', type:'bar', marker:{color:'#7FB08A'}, text:done.map(r => (r.hit*100).toFixed(0)), textposition:'outside'},
    {x:lab, y:done.map(r => r.live_base*100), name:'Live base', type:'bar', marker:{color:'#B5726A'}, text:done.map(r => (r.live_base*100).toFixed(0)), textposition:'outside'}
  ], merge({barmode:'group', xaxis:{type:'category', tickfont:{size:9}}, yaxis:{gridcolor:'#1B1E1B', zeroline:false, ticksuffix:'%', range:[0,115]}}), CFG);
  const tr = [['cpe.claimed_63d', 'Claimed', '#5B8DBE'], ['cpe.realised_63d', 'Realised', '#7FB08A'], ['cpe.live_base_63d', 'Live base', '#B5726A']]
    .map(a => ({s: lineSeries(a[0], 100, 100), name: a[1], c: a[2]})).filter(o => o.s);
  if (tr.length) Plotly.newPlot('cpeTrend63', tr.map(o => ({x:o.s.x, y:o.s.y, mode:'lines', name:o.name, line:{color:o.c, width:2}})),
    merge({yaxis:{gridcolor:'#1B1E1B', zeroline:false, ticksuffix:'%', range:[0,100]}}), CFG);
  const ed = [21, 63].map(h => ({h: h, s: diffSeries('cpe.realised_' + h + 'd', 'cpe.live_base_' + h + 'd', 100)})).filter(o => o.s);
  if (ed.length) Plotly.newPlot('cpeEdge', ed.map((o, i) => ({x:o.s.x, y:o.s.y.map(v => v * 100), mode:'lines', name:o.h + 'd', line:{width:2, color:['#C8A84A', '#5B8DBE'][i]}})),
    merge({shapes:ZERO, yaxis:{gridcolor:'#1B1E1B', zeroline:false, ticksuffix:' pts'}}), CFG);
  const cb = C.calibration;
  Plotly.newPlot('cpeCal', [
    {x:cb.map(r => r.bin + ' (n=' + r.n.toLocaleString() + ')'), y:cb.map(r => r.claimed*100), name:'Claimed CPE', type:'bar', marker:{color:'#5B8DBE'}, text:cb.map(r => (r.claimed*100).toFixed(0)), textposition:'outside'},
    {x:cb.map(r => r.bin + ' (n=' + r.n.toLocaleString() + ')'), y:cb.map(r => r.hit*100), name:'Realised', type:'bar', marker:{color:'#7FB08A'}, text:cb.map(r => (r.hit*100).toFixed(0)), textposition:'outside'},
    {x:cb.map(r => r.bin + ' (n=' + r.n.toLocaleString() + ')'), y:cb.map(r => r.live_base*100), name:'Live base', type:'bar', marker:{color:'#B5726A'}, text:cb.map(r => (r.live_base*100).toFixed(0)), textposition:'outside'}
  ], merge({barmode:'group', xaxis:{type:'category', tickfont:{size:9}}, yaxis:{gridcolor:'#1B1E1B', zeroline:false, ticksuffix:'%', range:[0,115]}}), CFG);
})();
</script>
</body>
</html>"""


def main():
    bundle = build_bundle()
    data_json = json.dumps(bundle, allow_nan=False)
    html = HTML.replace("__DATA__", data_json)
    out = os.path.join(HERE, "performance_monitor.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out} ({os.path.getsize(out)/1e3:.0f} KB)")
    g, m, p, fb = bundle["gold"], bundle["metals"], bundle["portfolio"], bundle["football"]
    print(f"CPE: {bundle['cpe']['n_resolved']} signal rows resolved; Gold: {g['n_resolved']} resolved, bullish hit {g['bull_hit']}; Metals: "
          f"{m['Silver']['n_resolved'] + m['Platinum']['n_resolved']} resolved; "
          f"Portfolio: {p['all']['n']} windows; Football: {fb['wins']}/{fb['n_resolved']}")


if __name__ == "__main__":
    main()
