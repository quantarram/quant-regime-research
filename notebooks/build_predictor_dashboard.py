"""
Live Price-Forecast Dashboard -- predictor_v1
===============================================
Forecast-only, no buy/sell signals: rigorous testing this research program
(5 trading-strategy designs, 5 post-processing designs, formal alpha
significance tests) found no instrument with statistically demonstrated
tradeable alpha. This dashboard shows predicted price + quantile band +
honest backtest accuracy (MAPE) for all 22 instruments -- nothing that
could be read as a trading recommendation.

Run: python build_predictor_dashboard.py
Requires: predictor_v1/{master_model_final_decision.json,
  final_deployed_pipeline.json, oos_predictions_all.parquet,
  features_daily_panel.parquet, features_new_tickers_baseline_cache.parquet,
  sector_proxy_cache.parquet}, multiasset_prices.parquet, cpe_results.parquet
Output: predictor_dashboard.html
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.abspath(__file__))  # notebooks/
PV1_DIR = os.path.join(REPO_DIR, "predictor_v1")
sys.path.insert(0, PV1_DIR)

import live_data as ld  # noqa: E402
import live_features as lfeat  # noqa: E402
import live_train_predict as ltp  # noqa: E402
import feature_lib as fl  # noqa: E402

print("=" * 60)
print("  PREDICTOR_V1 LIVE DASHBOARD BUILDER")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

FORBIDDEN_TOKENS = ["STRONG BUY", "BUY ZONE", "SELL SIGNAL", ">>BUY<<", ">>SELL<<"]

import time as _time
_t0 = _time.time()


def _log(msg):
    print(f"[{_time.time()-_t0:6.1f}s] {msg}", flush=True)


_log("Refreshing live data...")
prices, main_meta = ld.refresh_main_prices()
_log("  main prices done")
prices_proxy, proxy_meta = ld.refresh_sector_proxy_cache()
_log("  sector proxy done")
pair = pd.read_parquet(os.path.join(REPO_DIR, "cpe_results.parquet"))
_log("  cpe_results loaded")

_log("Refreshing feature panels (incremental)...")
orig_panel = lfeat.refresh_orig_panel(prices, pair)
_log("  orig panel done")
new_panel = lfeat.refresh_new_panel(prices, prices_proxy)
_log("  new panel done")
oos_all = pd.read_parquet(os.path.join(PV1_DIR, "oos_predictions_all.parquet"))
_log("  oos_all loaded")

_log("Training final models and predicting...")
results = ltp.predict_all(prices, prices_proxy, orig_panel, new_panel, oos_all)
_log(f"  {len(results)} / 22 instruments predicted")

banner_text = ld.market_status_banner(main_meta["latest_date"])

MODEL_TYPE_LABEL = {"climatology": "Climatology", "credit_only": "Credit-Regime", "vix_only": "VIX-Regime", "both": "Credit+VIX"}
# Feature counts by ticker group + variant, matching live_train_predict.py's
# _build_training_frame exactly: orig-group baseline = 11 z-scored multifractal
# cols + 12 ctx_* cols + self_ref_score = 24; new-group baseline = 11 z-scored
# multifractal cols only. Every informed variant adds 3 interaction/regime
# columns (Eq. 3 in the paper draft). Climatology uses none of this -- a
# frozen day-of-year empirical-quantile lookup, no LightGBM model at all.
N_BASELINE_FEATURES = {True: 24, False: 11}  # keyed by (ticker in fl.ORIG_GROUP_TICKERS)
N_VARIANT_EXTRA = {"credit_only": 3, "vix_only": 3, "both": 6, "climatology": 0}


def get_hist_series(ticker):
    return (prices_proxy[ticker] if ticker in ("IYR", "VOX") else prices[ticker]).dropna()


def model_detail(ticker, winner):
    if winner == "climatology":
        return "Frozen day-of-year empirical quantile table (no ML model) -- the calendar baseline itself won this instrument's model-selection competition."
    n_base = N_BASELINE_FEATURES[ticker in fl.ORIG_GROUP_TICKERS]
    n_feat = n_base + N_VARIANT_EXTRA[winner]
    return f"LightGBM quantile regression, 5 models (q0.10-q0.90), {n_feat} input features."


instruments = []
for _, r in results.sort_values("mape_deployed").iterrows():
    hist = get_hist_series(r["ticker"]).tail(90)
    instruments.append({
        "ticker": r["ticker"],
        "model_type": MODEL_TYPE_LABEL.get(r["winner"], r["winner"]),
        "model_detail": model_detail(r["ticker"], r["winner"]),
        "horizon": int(r["horizon"]),
        "post_processed": bool(r["correction_applied"]),
        "correction_n_pairs": int(r["correction_n_resolved_pairs"]),
        "as_of_date": r["as_of_date"],
        "target_date_est": r["target_date_est"],
        "price_now": round(float(r["price_now"]), 4),
        "q10": round(float(r["price_q0.1"]), 4),
        "q25": round(float(r["price_q0.25"]), 4),
        "q50": round(float(r["price_q0.5"]), 4),
        "q75": round(float(r["price_q0.75"]), 4),
        "q90": round(float(r["price_q0.9"]), 4),
        "mape_deployed": round(float(r["mape_deployed"]), 2),
        "hist_dates": [d.strftime("%Y-%m-%d") for d in hist.index],
        "hist_prices": [round(float(v), 4) for v in hist.values],
    })

data = {
    "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "latest_date": str(main_meta["latest_date"].date()),
    "data_age_days": main_meta["age_days"],
    "data_stale": main_meta["stale"],
    "n_instruments": len(instruments),
    "instruments": instruments,
}
data_json = json.dumps(data, allow_nan=False)
print(f"  Data bundle: {len(data_json)/1e3:.1f} KB")

banner_html = ""
if banner_text:
    banner_html = (f'<div class="status-banner">{banner_text}</div>')
if main_meta["stale"]:
    banner_html += (f'<div class="status-banner warn">Data is {main_meta["age_days"]} days old '
                     f'(latest: {main_meta["latest_date"].date()}). Yahoo Finance may be lagging.</div>')

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predictor Dashboard — Price Forecasts</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --accent:#5B8DBE; --accent2:#7FAAD1;
  --bg:#0C0E0D; --s1:#141614; --s2:#1B1E1B; --s3:#222522;
  --bdr:#2C302C; --bdr2:#3A3F3A;
  --text:#DDE8DD; --text2:#7A8F7A; --text3:#4A5A4A;
  --mono:'IBM Plex Mono',monospace; --sans:'IBM Plex Sans',sans-serif;
  --r:10px; --r-sm:6px; --gap:16px; --pad:20px;
  --warn:#C8A84A;
}
*{box-sizing:border-box;margin:0;padding:0;}
html{font-size:15px;}
body{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.6;min-height:100vh;}
header{background:linear-gradient(160deg,#0C0E0D,#141a1e,#0C0E0D);border-bottom:1px solid var(--bdr);
  padding:18px var(--pad);display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;}
.h-title{font-family:var(--mono);font-size:clamp(13px,2vw,17px);font-weight:600;color:var(--accent2);}
.h-sub{font-family:var(--mono);font-size:clamp(9px,1.5vw,11px);color:var(--text2);letter-spacing:.06em;text-transform:uppercase;margin-top:2px;}
.h-meta{font-family:var(--mono);font-size:11px;color:var(--text2);text-align:right;line-height:1.8;flex-shrink:0;}
main{padding:var(--pad);max-width:1440px;margin:0 auto;}
.status-banner{background:#141a1e;border:1px solid #2C4050;border-radius:8px;padding:14px 24px;
  font-family:var(--sans);font-size:13px;color:var(--accent2);text-align:center;margin-bottom:16px;line-height:1.7;}
.status-banner.warn{background:#1C1A0E;border-color:#4A4020;color:var(--warn);}
.framing-banner{background:linear-gradient(135deg,#141614,#171b17);border:1px solid var(--bdr2);border-radius:var(--r);
  padding:var(--pad);margin-bottom:20px;font-size:13px;color:var(--text2);line-height:1.7;}
.framing-banner b{color:var(--text);}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:var(--gap);}
.card{background:var(--s1);border:1px solid var(--bdr);border-radius:var(--r);padding:var(--pad);}
.card-head{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:10px;}
.ticker{font-family:var(--mono);font-size:18px;font-weight:600;color:var(--text);}
.horizon{font-family:var(--mono);font-size:11px;color:var(--text2);}
.badges{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;}
.badge{font-family:var(--mono);font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;
  padding:3px 8px;border-radius:20px;background:var(--s3);color:var(--text2);border:1px solid var(--bdr2);}
.badge.model{color:var(--accent2);border-color:#2C4050;}
.badge.pp{color:var(--warn);border-color:#4A4020;}
.price-row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;}
.price-now{font-family:var(--mono);font-size:22px;font-weight:600;}
.price-label{font-family:var(--mono);font-size:9px;color:var(--text2);text-transform:uppercase;letter-spacing:.08em;}
.pred-range{font-family:var(--mono);font-size:13px;color:var(--text2);margin-top:2px;}
.pred-median{color:var(--accent2);font-weight:600;}
.mape-row{margin-top:10px;padding-top:10px;border-top:1px solid var(--bdr);
  font-family:var(--mono);font-size:11px;color:var(--text2);display:flex;justify-content:space-between;}
.model-detail{font-family:var(--mono);font-size:10.5px;color:var(--text3);margin-top:8px;line-height:1.5;}
.chart{height:120px;margin-top:10px;}
footer{padding:24px var(--pad) 40px;text-align:center;font-family:var(--mono);font-size:10px;color:var(--text3);}
.methodology{background:var(--s1);border:1px solid var(--bdr);border-radius:var(--r);padding:var(--pad);margin-bottom:20px;}
.methodology summary{cursor:pointer;font-family:var(--mono);font-size:12px;font-weight:600;color:var(--accent2);
  text-transform:uppercase;letter-spacing:.06em;list-style:none;display:flex;align-items:center;gap:8px;}
.methodology summary::-webkit-details-marker{display:none;}
.methodology summary::before{content:'+';font-size:14px;width:14px;}
.methodology[open] summary::before{content:'\\2212';}
.methodology .body{margin-top:16px;font-size:13px;color:var(--text2);line-height:1.75;}
.methodology .body h4{font-family:var(--mono);font-size:11px;color:var(--text);text-transform:uppercase;
  letter-spacing:.05em;margin:18px 0 6px;}
.methodology .body h4:first-child{margin-top:0;}
.methodology .body code{font-family:var(--mono);font-size:12px;background:var(--s3);padding:1px 5px;border-radius:4px;color:var(--accent2);}
.methodology .body table{width:100%;border-collapse:collapse;margin-top:8px;font-size:12px;}
.methodology .body th, .methodology .body td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--bdr);font-family:var(--mono);}
.methodology .body th{color:var(--text2);font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:.05em;}
.methodology .body a{color:var(--accent2);}
</style>
</head>
<body>
<header>
  <div>
    <div class="h-title">Predictor Dashboard</div>
    <div class="h-sub">22-instrument price forecast, updated on demand</div>
  </div>
  <div class="h-meta" id="hmeta"></div>
</header>
<main>
""" + banner_html + """
  <div class="framing-banner">
    <b>This is a price-forecasting research tool, not a trading signal.</b> Rigorous testing on this
    program (five different trading-strategy designs, five post-processing designs, and formal
    statistical alpha-significance tests) found <b>no instrument with statistically demonstrated
    tradeable alpha</b>. The forecasts below have real but modest accuracy value for most instruments
    &mdash; shown here as an honest, continuously-tracked record, not a signal to act on. “Backtest MAPE”
    is the mean absolute percentage error of this exact model on genuinely held-out data (2022 onward),
    never touched during model selection.
  </div>
  <details class="methodology">
    <summary>What AI/ML models generate these forecasts?</summary>
    <div class="body">
      <h4>Model family</h4>
      Gradient-boosted <b>quantile regression</b> (LightGBM). For each instrument, 5 independent regressors are
      trained &mdash; one per quantile level (<code>q0.10, q0.25, q0.50, q0.75, q0.90</code>) &mdash; each minimizing
      the pinball loss for its own quantile, jointly producing a full predicted price distribution rather than a
      single point forecast. Hyperparameters: <code>n_estimators=200, max_depth=4, learning_rate=0.05,
      subsample=0.8, colsample_bytree=0.8</code>.
      <h4>Input features</h4>
      Two families of engineered features, multiplicatively interacted:
      <ol style="margin-left:18px;margin-top:6px;">
        <li><b>Each instrument's own multifractal price dynamics</b> &mdash; trace-moment parameters, structure-function
        exponents, and correlated/decorrelated structure-function gaps, computed on a trailing 512-trading-day rolling
        window and cross-sectionally z-scored against a peer group of comparable instruments (reused from this
        program's Paper 11 multifractal-predictability research).</li>
        <li><b>Two forward-looking market regime signals</b>: a credit-spread regime (HYG/LQD high-yield-to-investment-grade
        bond ratio, 200-day rolling z-score) and a VIX-term-structure regime (VIXM/VIXY medium-vs-short-term
        volatility-ETP ratio, 200-day rolling z-score) &mdash; both independently Granger-causality-validated against
        realized volatility before use, not included on correlation alone.</li>
      </ol>
      <h4>Model selection: four candidates compete, per instrument</h4>
      <table>
        <tr><th>Candidate</th><th>What it is</th></tr>
        <tr><td>Climatology</td><td>A frozen day-of-year empirical quantile table &mdash; no machine learning at all, the pure calendar baseline</td></tr>
        <tr><td>Credit-Regime</td><td>LightGBM on multifractal features &times; credit-spread regime</td></tr>
        <tr><td>VIX-Regime</td><td>LightGBM on multifractal features &times; VIX-term-structure regime</td></tr>
        <tr><td>Credit+VIX</td><td>LightGBM using both regime interactions together</td></tr>
      </table>
      The winner for each instrument is whichever candidate has the lowest mean absolute percentage error (MAPE)
      on a genuinely held-out period (2022 onward) never touched while choosing between candidates &mdash; climatology
      is not a baseline to beat, it is a real candidate, and it wins outright for 12 of the 22 instruments shown below.
      <h4>Training: research vs. live</h4>
      The backtested research behind this dashboard used strict walk-forward validation (each model trained only on
      data before the period it was tested on). The live forecasts shown here instead retrain a single <i>final</i>
      model on all available history through today, each time this page is rebuilt, and predict forward from today's
      freshest feature values &mdash; the one genuinely new mode of operation versus the backtested research.
      <h4>Post-processing (2 of 22 instruments only)</h4>
      For <b>GLD and JPM specifically</b> &mdash; the only two instruments, out of five independently designed
      correction techniques tested, where a real and repeatable forecast bias was found &mdash; an ongoing rolling
      bias correction (moment-matching then quantile-mapping, refit from a trailing window of already-resolved past
      predictions) is layered on top of the raw model output. The other 20 instruments use the raw model forecast,
      uncorrected, because correction was tested and found to make them worse.
      <h4>Full methodology</h4>
      Every equation, all five trading-strategy tests, all five post-processing designs, and the honest economic
      results (including why none of this is shown as a trading signal) are documented in the accompanying
      <a href="predictor_v1_paper_draft.md">draft preprint</a>. Code: <code>notebooks/predictor_v1/</code>.
    </div>
  </details>
  <div class="grid" id="cards"></div>
</main>
<footer>predictor_v1 &mdash; quantarram/quant-regime-research</footer>
<script>
const D = """ + data_json + """;

document.getElementById('hmeta').innerHTML =
  'Generated ' + D.generated + '<br>Data as of ' + D.latest_date +
  (D.data_stale ? ' (STALE, ' + D.data_age_days + 'd old)' : '');

const cardsEl = document.getElementById('cards');
D.instruments.forEach((inst, i) => {
  const div = document.createElement('div');
  div.className = 'card';
  const ppBadge = inst.post_processed ? '<span class="badge pp">Post-processed</span>' : '';
  const ppDetail = inst.post_processed
    ? ' Rolling bias correction active, refit from ' + inst.correction_n_pairs + ' recent resolved predictions.'
    : '';
  div.innerHTML =
    '<div class="card-head"><span class="ticker">' + inst.ticker + '</span>' +
    '<span class="horizon">' + inst.horizon + 'd horizon</span></div>' +
    '<div class="badges"><span class="badge model">' + inst.model_type + '</span>' + ppBadge + '</div>' +
    '<div class="price-row"><span class="price-now">$' + inst.price_now.toLocaleString(undefined,{maximumFractionDigits:2}) + '</span>' +
    '<span class="price-label">current</span></div>' +
    '<div class="pred-range">Forecast (~' + inst.target_date_est + '): $' + inst.q10.toFixed(2) + ' – $' + inst.q90.toFixed(2) +
    ' &nbsp;(median <span class="pred-median">$' + inst.q50.toFixed(2) + '</span>)</div>' +
    '<div class="chart" id="chart' + i + '"></div>' +
    '<div class="mape-row"><span>Backtest MAPE (2022+ holdout)</span><span>' + inst.mape_deployed.toFixed(1) + '%</span></div>' +
    '<div class="model-detail">' + inst.model_detail + ppDetail + '</div>';
  cardsEl.appendChild(div);

  const histX = inst.hist_dates, histY = inst.hist_prices;
  const lastDate = histX[histX.length - 1];
  const futX = [lastDate, inst.target_date_est];
  const traces = [
    {x: histX, y: histY, type: 'scatter', mode: 'lines', line: {color: '#B5726A', width: 1.2}, hoverinfo: 'skip', showlegend: false},
    {x: futX, y: [inst.price_now, inst.q90], type: 'scatter', mode: 'lines', line: {color: 'rgba(91,141,190,0.25)', width: 0}, showlegend: false, hoverinfo: 'skip'},
    {x: futX, y: [inst.price_now, inst.q10], type: 'scatter', mode: 'lines', fill: 'tonexty', fillcolor: 'rgba(91,141,190,0.18)', line: {color: 'rgba(91,141,190,0.25)', width: 0}, showlegend: false, hoverinfo: 'skip'},
    {x: futX, y: [inst.price_now, inst.q50], type: 'scatter', mode: 'lines', line: {color: '#7FAAD1', width: 1.6, dash: 'dot'}, showlegend: false, hoverinfo: 'skip'},
  ];
  Plotly.newPlot('chart' + i, traces, {
    margin: {l: 0, r: 0, t: 4, b: 18},
    height: 120,
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    xaxis: {showgrid: false, tickfont: {size: 8, color: '#4A5A4A'}, nticks: 4},
    yaxis: {showgrid: true, gridcolor: '#1B1E1B', tickfont: {size: 8, color: '#4A5A4A'}},
  }, {displayModeBar: false, responsive: true});
});
</script>
</body>
</html>"""

for tok in FORBIDDEN_TOKENS:
    assert tok not in html, f"FRAMING GUARDRAIL FAILED: forbidden token '{tok}' found in generated HTML"

out_path = os.path.join(REPO_DIR, "predictor_dashboard.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\nSaved: {out_path} ({os.path.getsize(out_path)/1e3:.0f} KB)")
print("Framing guardrail passed: no buy/sell tokens in output.")
