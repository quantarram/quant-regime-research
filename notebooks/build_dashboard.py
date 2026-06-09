"""
CPE Dashboard Builder
Reads parquet files, generates cpe_dashboard.html
Run: python build_dashboard.py
"""
import pandas as pd
import numpy as np
import json, os
from datetime import datetime

print("Loading data...")
pairwise = pd.read_parquet("cpe_results.parquet")
joint    = pd.read_parquet("joint_cpe_results.parquet")

ASSET_CLASS_MAP = {
    "equities":    ["SPY","QQQ","IWM","DIA","VTI","VT","EFA","EEM","VEA","VWO",
                    "EWJ","EWZ","FXI","INDA","EWY","XLK","XLF","XLE","XLV","XLI",
                    "XLP","XLY","XLU","XLRE","XLB","XLC","VTV","VUG","MTUM","USMV",
                    "QUAL","SIZE","ARKK","ICLN","ITB","XBI","SOXX",
                    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","BRK-B","XOM"],
    "crypto":      ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD",
                    "AVAX-USD","DOGE-USD","DOT-USD","LINK-USD","MATIC-USD","LTC-USD",
                    "BCH-USD","UNI-USD","ATOM-USD","IBIT","FBTC","GBTC","ETHE","BITB"],
    "volatility":  ["^VIX","^VXN","^OVX","^GVZ","^EVZ","^VVIX","^SKEW"],
    "commodities": ["GLD","IAU","SLV","PPLT","PALL","USO","BNO","UNG","UGA",
                    "CORN","WEAT","SOYB","CANE","NIB","JO","DJP","PDBC","DBC",
                    "GSG","CPER","DBB","GC=F","SI=F","CL=F","BZ=F","NG=F",
                    "HG=F","ZC=F","ZW=F","ZS=F"],
    "rates":       ["SHY","IEI","IEF","TLH","TLT","ZROZ","EDV","TIP","SCHP",
                    "LQD","HYG","JNK","EMB","AGG","BND","VCSH","VCIT","VCLT",
                    "MUB","MBB","^TNX","^TYX","^FVX","^IRX"],
    "fx":          ["EURUSD=X","GBPUSD=X","JPYUSD=X","CHFUSD=X","CADUSD=X",
                    "AUDUSD=X","NZDUSD=X","SGDUSD=X","INRUSD=X","BRLUSD=X",
                    "MXNUSD=X","ZARUSD=X","UUP","UDN",
                    "EURJPY=X","EURGBP=X","GBPJPY=X","AUDJPY=X","EURCHF=X"],
}

def get_ac(ticker):
    for ac, tickers in ASSET_CLASS_MAP.items():
        if ticker in tickers: return ac
    return "other"

print("Preparing pairwise data...")
pw = pairwise.copy()
pw["Y_class"] = pw["Y"].apply(get_ac)
pw["X_class"] = pw["X"].apply(get_ac)

bull_sample = pw[pw["direction"]=="bullish"].nlargest(3000, "CPE")
bear_sample = pw[pw["direction"]=="bearish"].nlargest(3000, "CPE")
pairwise_sample = pd.concat([bull_sample, bear_sample]).reset_index(drop=True)

tau_curve_data = []
for (y, x, qx, qy, direction, tp), grp in pw.groupby(["Y","X","q_X","q_Y","direction","tau_past"]):
    if len(grp) < 2: continue
    row = {"Y":y,"X":x,"q_X":qx,"q_Y":qy,"direction":direction,"tau_past":int(tp),
           "Y_class":get_ac(y),"X_class":get_ac(x)}
    for _, r in grp.iterrows():
        row["tf_%d" % int(r["tau_future"])] = round(float(r["CPE"]),4)
    tau_curve_data.append(row)

heatmap_params = [
    (63,  63,  0.90, 0.60, "bullish"),
    (63,  63,  0.75, 0.50, "bullish"),
    (252, 252, 0.75, 0.75, "bullish"),
    (252, 252, 0.90, 0.95, "bullish"),
    (63,  63,  0.90, 0.60, "bearish"),
    (63,  63,  0.75, 0.50, "bearish"),
    (252, 252, 0.75, 0.75, "bearish"),
]

heatmaps = []
for (tp, tf, qx, qy, direction) in heatmap_params:
    sub = pw[(pw["tau_past"]==tp)&(pw["tau_future"]==tf)&
             (pw["q_X"]==qx)&(pw["q_Y"]==qy)&(pw["direction"]==direction)]
    if sub.empty: continue
    pivot = sub.pivot_table(index="Y", columns="X", values="CPE", aggfunc="mean")
    pivot = pivot.loc[pivot.notna().sum(axis=1).nlargest(30).index,
                      pivot.notna().sum(axis=0).nlargest(30).index]
    heatmaps.append({
        "label": "%s tp=%d tf=%d qX=%s qY=%s" % (direction, tp, tf, qx, qy),
        "direction": direction,
        "tau_past": tp, "tau_future": tf, "q_X": qx, "q_Y": qy,
        "Y_labels": list(pivot.index),
        "X_labels": list(pivot.columns),
        "z": [[None if np.isnan(v) else round(v,4) for v in row] for row in pivot.values],
    })

print("Preparing joint data...")
jt = joint.copy()
jt["Y_class"] = jt["Y"].apply(get_ac)
jt["predictors_str"] = jt.apply(
    lambda r: " n ".join(["%s(t=%d,q=%s)" % (x, tp, qx)
                           for x,tp,qx in zip(r["predictors"],r["tau_pasts"],r["q_Xs"])]), axis=1)

joint_records = []
for _, r in jt.iterrows():
    joint_records.append({
        "Y": r["Y"], "direction": r["direction"],
        "tau_future": int(r["tau_future"]), "q_Y": float(r["q_Y"]),
        "n_predictors": int(r["n_predictors"]),
        "joint_CPE": float(r["joint_CPE"]),
        "n_joint": int(r["n_joint"]),
        "Y_class": r["Y_class"],
        "predictors_str": r["predictors_str"],
        "predictors": list(r["predictors"]),
        "tau_pasts": [int(x) for x in r["tau_pasts"]],
        "q_Xs": [float(x) for x in r["q_Xs"]],
        "pairwise_CPEs": [float(x) for x in r["pairwise_CPEs"]],
    })

cpe_stats = {}
for direction in ["bullish","bearish"]:
    sub = jt[jt["direction"]==direction]
    stats = sub.groupby("n_predictors")["joint_CPE"].agg(["mean","std","count"]).reset_index()
    cpe_stats[direction] = stats.rename(columns={"mean":"mean_cpe","std":"std_cpe","count":"n_signals"}).to_dict("records")

freq_predictors = {}
for direction in ["bullish","bearish"]:
    sub = jt[(jt["direction"]==direction) & (jt["n_predictors"]==2)]
    all_pred = []
    for _, r in sub.iterrows():
        all_pred.extend(r["predictors"])
    freq = pd.Series(all_pred).value_counts().head(20)
    freq_predictors[direction] = [{"ticker":k,"count":int(v),"ac":get_ac(k)} for k,v in freq.items()]

data_bundle = {
    "pairwise_sample": pairwise_sample.to_dict("records"),
    "tau_curves": tau_curve_data[:2000],
    "heatmaps": heatmaps,
    "joint": joint_records,
    "cpe_stats": cpe_stats,
    "freq_predictors": freq_predictors,
    "all_Y": sorted(pw["Y"].unique().tolist()),
    "all_X": sorted(pw["X"].unique().tolist()),
    "all_tau": [int(x) for x in sorted(pw["tau_past"].unique())],
    "all_qX":  [float(x) for x in sorted(pw["q_X"].unique())],
    "all_qY":  [float(x) for x in sorted(pw["q_Y"].unique())],
    "ac_colors": {
        "equities":    "#3B82F6",
        "crypto":      "#F59E0B",
        "volatility":  "#8B5CF6",
        "commodities": "#10B981",
        "rates":       "#EF4444",
        "fx":          "#06B6D4",
        "other":       "#6B7280",
    },
    "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
}

data_json = json.dumps(data_bundle, allow_nan=False)
print("  Data bundle: %.2f MB" % (len(data_json)/1e6))

# ── WRITE HTML ────────────────────────────────────────────────────────────────
out_path = "cpe_dashboard.html"

with open(out_path, "w", encoding="utf-8") as f:
    f.write("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-Asset CPE Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#0A0E1A; --surface:#111827; --surface2:#1F2937; --border:#2D3748;
    --accent:#38BDF8; --accent2:#818CF8; --bull:#34D399; --bear:#F87171;
    --text:#E2E8F0; --text2:#94A3B8;
    --font-mono:'Space Mono',monospace; --font-sans:'DM Sans',sans-serif;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:var(--font-sans); }
  header { background:linear-gradient(135deg,#0A0E1A,#0F172A,#0A0E1A);
    border-bottom:1px solid var(--border); padding:24px 40px;
    display:flex; align-items:center; justify-content:space-between;
    position:sticky; top:0; z-index:100; }
  .logo { display:flex; align-items:center; gap:14px; }
  .logo-icon { width:40px; height:40px;
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    border-radius:10px; display:flex; align-items:center; justify-content:center;
    font-family:var(--font-mono); font-size:16px; font-weight:700; color:var(--bg); }
  .logo-text { font-family:var(--font-mono); font-size:15px; font-weight:700; letter-spacing:.05em; }
  .logo-sub { font-size:11px; color:var(--text2); letter-spacing:.08em; text-transform:uppercase; margin-top:2px; }
  .header-meta { font-family:var(--font-mono); font-size:11px; color:var(--text2); text-align:right; line-height:1.8; }
  nav { background:var(--surface); border-bottom:1px solid var(--border); padding:0 40px; display:flex; }
  .tab { padding:14px 20px; font-family:var(--font-mono); font-size:11px; font-weight:700;
    color:var(--text2); cursor:pointer; border:none; background:none;
    border-bottom:2px solid transparent; transition:all .2s; text-transform:uppercase; letter-spacing:.05em; }
  .tab:hover { color:var(--text); }
  .tab.active { color:var(--accent); border-bottom-color:var(--accent); }
  main { padding:32px 40px; max-width:1600px; margin:0 auto; }
  .panel { display:none; }
  .panel.active { display:block; }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:24px; margin-bottom:24px; }
  .card-title { font-family:var(--font-mono); font-size:12px; font-weight:700;
    color:var(--accent); text-transform:uppercase; letter-spacing:.1em; margin-bottom:20px;
    display:flex; align-items:center; gap:10px; }
  .card-title::before { content:''; display:block; width:3px; height:14px; background:var(--accent); border-radius:2px; }
  .controls { display:flex; flex-wrap:wrap; gap:16px; margin-bottom:24px; align-items:flex-end; }
  .ctrl-group { display:flex; flex-direction:column; gap:6px; }
  .ctrl-group label { font-family:var(--font-mono); font-size:10px; font-weight:700;
    color:var(--text2); text-transform:uppercase; letter-spacing:.08em; }
  select, input { background:var(--surface2); border:1px solid var(--border); color:var(--text);
    padding:8px 12px; border-radius:8px; font-family:var(--font-mono); font-size:12px;
    outline:none; cursor:pointer; transition:border-color .2s; min-width:130px; }
  select:hover, input:hover, select:focus, input:focus { border-color:var(--accent); }
  .btn { background:var(--accent); color:var(--bg); border:none; border-radius:8px;
    padding:9px 20px; font-family:var(--font-mono); font-size:12px; font-weight:700;
    cursor:pointer; transition:all .2s; text-transform:uppercase; letter-spacing:.05em; }
  .btn:hover { background:#7DD3FC; transform:translateY(-1px); }
  .stats-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:16px; margin-bottom:24px; }
  .stat-card { background:var(--surface2); border:1px solid var(--border); border-radius:10px; padding:16px; }
  .stat-label { font-family:var(--font-mono); font-size:10px; color:var(--text2); text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px; }
  .stat-value { font-family:var(--font-mono); font-size:22px; font-weight:700; color:var(--accent); }
  .stat-sub { font-size:11px; color:var(--text2); margin-top:4px; }
  .dir-pill { display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:20px;
    font-family:var(--font-mono); font-size:11px; font-weight:700; }
  .dir-pill.bullish { background:#34D39922; color:var(--bull); border:1px solid #34D39944; }
  .dir-pill.bearish { background:#F8717122; color:var(--bear); border:1px solid #F8717144; }
  .signal-table { width:100%; border-collapse:collapse; font-size:12px; }
  .signal-table th { font-family:var(--font-mono); font-size:10px; font-weight:700;
    color:var(--text2); text-transform:uppercase; letter-spacing:.08em;
    padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; white-space:nowrap; }
  .signal-table td { padding:9px 12px; border-bottom:1px solid var(--border); vertical-align:middle; }
  .signal-table tr:hover td { background:var(--surface2); }
  .cpe-bar { display:flex; align-items:center; gap:8px; }
  .cpe-fill { height:6px; border-radius:3px; flex-shrink:0; }
  .cpe-val { font-family:var(--font-mono); font-size:12px; font-weight:700; color:var(--text); white-space:nowrap; }
  .ac-badge { display:inline-block; padding:2px 8px; border-radius:4px;
    font-family:var(--font-mono); font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
  .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
  .plot-box { width:100%; border-radius:8px; overflow:hidden; }
  .pred-chip { display:inline-block; background:var(--surface2); border:1px solid var(--border);
    border-radius:4px; padding:1px 6px; margin:1px; font-size:10px; color:var(--text); font-family:var(--font-mono); }
  .pred-sep { color:var(--accent); font-weight:700; margin:0 3px; font-family:var(--font-mono); }
  ::-webkit-scrollbar { width:6px; height:6px; }
  ::-webkit-scrollbar-track { background:var(--bg); }
  ::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
  .ticker { font-family:var(--font-mono); font-weight:700; }
  .empty { text-align:center; padding:60px; color:var(--text2); font-family:var(--font-mono); font-size:13px; }
  .overview-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; margin-bottom:24px; }
  .sig-card { background:var(--surface2); border:1px solid var(--border); border-radius:10px; padding:16px; transition:border-color .2s; }
  .sig-card:hover { border-color:var(--accent); }
  .sig-y { font-family:var(--font-mono); font-size:15px; font-weight:700; }
  .sig-cpe { font-family:var(--font-mono); font-size:19px; font-weight:700; }
  .sig-detail { font-size:12px; color:var(--text2); line-height:1.7; margin-top:6px; }
  .sig-preds { margin-top:8px; font-family:var(--font-mono); font-size:10px; color:var(--accent2); }
  /* methodology */
  .mth-section { font-size:13px; line-height:1.9; color:var(--text2); max-width:860px; }
  .mth-section p { margin-bottom:10px; }
  .mth-section ul { margin-left:24px; margin-bottom:10px; }
  .mth-section li { margin-bottom:4px; }
  .formula-box { background:var(--surface2); border:1px solid var(--border); border-radius:8px;
    padding:18px 24px; font-family:var(--font-mono); font-size:13px; margin:12px 0; }
  .formula-bull { color:var(--accent); }
  .formula-bear { color:var(--bear); }
  .formula-neutral { color:var(--accent2); }
  strong.hl { color:var(--text); }
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-icon">CPE</div>
    <div>
      <div class="logo-text">Multi-Asset CPE Dashboard</div>
      <div class="logo-sub">Conditional Probability of Exceedance &mdash; Cross-Asset Analysis</div>
    </div>
  </div>
  <div class="header-meta">
    <div>Generated: <span id="gen-date"></span></div>
    <div>Universe: 136 assets &middot; 6 classes &middot; 169K pairwise signals</div>
  </div>
</header>
<nav>
  <button class="tab active" onclick="showPanel('overview',this)">Overview</button>
  <button class="tab" onclick="showPanel('pairwise',this)">Pairwise CPE</button>
  <button class="tab" onclick="showPanel('heatmap',this)">Heatmaps</button>
  <button class="tab" onclick="showPanel('taucurves',this)">&#964; Curves</button>
  <button class="tab" onclick="showPanel('joint',this)">Joint CPE</button>
  <button class="tab" onclick="showPanel('network',this)">Signal Network</button>
  <button class="tab" onclick="showPanel('methodology',this)">Methodology</button>
</nav>
<main>

<!-- OVERVIEW -->
<div id="panel-overview" class="panel active">
  <div class="stats-row" id="overview-stats"></div>
  <div class="grid-2">
    <div class="card"><div class="card-title">Top Bullish Joint Signals (size 2)</div><div id="ov-bull"></div></div>
    <div class="card"><div class="card-title">Top Bearish Joint Signals (size 2)</div><div id="ov-bear"></div></div>
  </div>
  <div class="grid-2">
    <div class="card"><div class="card-title">Joint CPE vs Predictor Set Size</div><div id="ov-cpe-plot" class="plot-box" style="height:300px"></div></div>
    <div class="card"><div class="card-title">Most Frequent Predictors in Joint Sets (size 2)</div><div id="ov-freq-plot" class="plot-box" style="height:300px"></div></div>
  </div>
</div>

<!-- PAIRWISE -->
<div id="panel-pairwise" class="panel">
  <div class="card">
    <div class="card-title">Pairwise CPE Explorer</div>
    <div class="controls">
      <div class="ctrl-group"><label>Predicted Y</label><select id="pw-Y"><option value="">All</option></select></div>
      <div class="ctrl-group"><label>Predictor X</label><select id="pw-X"><option value="">All</option></select></div>
      <div class="ctrl-group"><label>Direction</label><select id="pw-dir"><option value="">Both</option><option value="bullish">Bullish</option><option value="bearish">Bearish</option></select></div>
      <div class="ctrl-group"><label>&#964;_future</label><select id="pw-tf"><option value="">All</option></select></div>
      <div class="ctrl-group"><label>q_Y</label><select id="pw-qy"><option value="">All</option></select></div>
      <div class="ctrl-group"><label>Min CPE</label><input type="number" id="pw-mincpe" value="0.80" min="0" max="1" step="0.05" style="width:90px"></div>
      <div class="ctrl-group"><label>&nbsp;</label><button class="btn" onclick="filterPairwise()">Apply</button></div>
    </div>
    <div style="overflow-x:auto">
      <table class="signal-table">
        <thead><tr><th>Y</th><th>Y Class</th><th>X</th><th>X Class</th><th>Dir</th><th>&#964;_past</th><th>&#964;_future</th><th>q_X</th><th>q_Y</th><th>CPE</th><th>Lift</th><th>n</th></tr></thead>
        <tbody id="pw-tbody"></tbody>
      </table>
    </div>
    <div id="pw-count" style="margin-top:12px;font-family:var(--font-mono);font-size:11px;color:var(--text2)"></div>
  </div>
</div>

<!-- HEATMAPS -->
<div id="panel-heatmap" class="panel">
  <div class="card">
    <div class="card-title">CPE Heatmap &mdash; Y &times; X</div>
    <div class="controls">
      <div class="ctrl-group"><label>Select Heatmap</label><select id="hm-select" onchange="renderHeatmap()"></select></div>
    </div>
    <div id="heatmap-plot" class="plot-box" style="height:600px"></div>
  </div>
</div>

<!-- TAU CURVES -->
<div id="panel-taucurves" class="panel">
  <div class="card">
    <div class="card-title">CPE vs &#964;_future Curves</div>
    <div class="controls">
      <div class="ctrl-group"><label>Predicted Y</label><select id="tc-Y"></select></div>
      <div class="ctrl-group"><label>Predictor X</label><select id="tc-X"></select></div>
      <div class="ctrl-group"><label>Direction</label><select id="tc-dir"><option value="bullish">Bullish</option><option value="bearish">Bearish</option></select></div>
      <div class="ctrl-group"><label>q_Y</label><select id="tc-qy"></select></div>
      <div class="ctrl-group"><label>&nbsp;</label><button class="btn" onclick="renderTauCurves()">Plot</button></div>
    </div>
    <div id="tau-plot" class="plot-box" style="height:450px"></div>
    <div id="tau-info" style="margin-top:12px;font-family:var(--font-mono);font-size:11px;color:var(--text2)"></div>
  </div>
</div>

<!-- JOINT CPE -->
<div id="panel-joint" class="panel">
  <div class="card">
    <div class="card-title">Joint CPE Signal Browser</div>
    <div class="controls">
      <div class="ctrl-group"><label>Predicted Y</label><select id="jt-Y"><option value="">All</option></select></div>
      <div class="ctrl-group"><label>Direction</label><select id="jt-dir"><option value="">Both</option><option value="bullish">Bullish</option><option value="bearish">Bearish</option></select></div>
      <div class="ctrl-group"><label>n_predictors</label><select id="jt-np"><option value="">All</option><option>2</option><option>3</option><option>4</option><option>5</option><option>6</option><option>7</option><option>8</option><option>9</option><option>10</option></select></div>
      <div class="ctrl-group"><label>q_Y</label><select id="jt-qy"><option value="">All</option></select></div>
      <div class="ctrl-group"><label>Min CPE</label><input type="number" id="jt-mincpe" value="0.90" min="0" max="1" step="0.05" style="width:90px"></div>
      <div class="ctrl-group"><label>&nbsp;</label><button class="btn" onclick="filterJoint()">Apply</button></div>
    </div>
    <div style="overflow-x:auto">
      <table class="signal-table">
        <thead><tr><th>Y</th><th>Class</th><th>Dir</th><th>&#964;_f</th><th>q_Y</th><th>n_pred</th><th>Joint CPE</th><th>n_joint</th><th>Predictor Set</th></tr></thead>
        <tbody id="jt-tbody"></tbody>
      </table>
    </div>
    <div id="jt-count" style="margin-top:12px;font-family:var(--font-mono);font-size:11px;color:var(--text2)"></div>
  </div>
  <div class="grid-2">
    <div class="card"><div class="card-title">Joint CPE Distribution &mdash; Bullish</div><div id="jt-dist-bull" class="plot-box" style="height:280px"></div></div>
    <div class="card"><div class="card-title">Joint CPE Distribution &mdash; Bearish</div><div id="jt-dist-bear" class="plot-box" style="height:280px"></div></div>
  </div>
</div>

<!-- NETWORK -->
<div id="panel-network" class="panel">
  <div class="card">
    <div class="card-title">Signal Network &mdash; Top Pairwise Links</div>
    <div class="controls">
      <div class="ctrl-group"><label>Direction</label><select id="net-dir" onchange="renderNetwork()"><option value="bullish">Bullish</option><option value="bearish">Bearish</option></select></div>
      <div class="ctrl-group"><label>Min CPE</label><input type="number" id="net-mincpe" value="0.95" min="0.80" max="1" step="0.01" style="width:90px" onchange="renderNetwork()"></div>
      <div class="ctrl-group"><label>Max Links</label><input type="number" id="net-maxlinks" value="80" min="10" max="200" step="10" style="width:90px" onchange="renderNetwork()"></div>
    </div>
    <div id="network-plot" class="plot-box" style="height:600px"></div>
    <div id="network-info" style="margin-top:12px;font-family:var(--font-mono);font-size:11px;color:var(--text2)"></div>
  </div>
</div>

<!-- METHODOLOGY -->
<div id="panel-methodology" class="panel">

  <div class="card">
    <div class="card-title">Framework Overview</div>
    <div class="mth-section">
      <p>This dashboard presents a purely <strong class="hl">descriptive empirical analysis</strong>
      of historical joint tail behaviour across a universe of 136 financial assets spanning
      equities, crypto, fixed income, commodities, FX, and volatility indices.
      No parametric model is fitted; all quantities are computed directly from the empirical
      distribution of observed price increments over overlapping windows.
      The analysis makes no out-of-sample predictions &mdash; it characterises
      what has historically co-occurred in the data.</p>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Notation</div>
    <div class="mth-section" style="margin-bottom:20px">
      <p>Symbols used throughout the framework. Note that q<sub>Y</sub> applies uniformly to all 136 predicted assets regardless of asset class &mdash; it is a quantile of that asset's own empirical increment distribution, not a fixed price level.</p>
    </div>
    <table class="signal-table" style="max-width:860px;margin-bottom:32px">
      <thead><tr><th>Symbol</th><th>Definition</th></tr></thead>
      <tbody>
        <tr><td class="ticker" style="color:var(--accent)">Y</td><td>Predicted asset. Any of the 136 price-based assets (equities, crypto, commodities, fixed income ETFs, FX). Volatility and yield indices are excluded from Y.</td></tr>
        <tr><td class="ticker" style="color:var(--accent)">X</td><td>Predictor asset. Any of the 158 assets including all Y candidates plus VIX-family and Treasury yield indices.</td></tr>
        <tr><td class="ticker" style="color:var(--accent)">&tau;<sub>past</sub></td><td>Look-back window in trading days over which X's past increment is computed. Grid: {1, 5, 10, 21, 63, 126, 252, 300}. Different X predictors in a joint set may have different &tau;<sub>past</sub> values.</td></tr>
        <tr><td class="ticker" style="color:var(--accent)">&tau;<sub>future</sub></td><td>Forward window in trading days over which Y's future increment is computed. Same grid as &tau;<sub>past</sub> but chosen independently.</td></tr>
        <tr><td class="ticker" style="color:var(--accent)">q<sub>X</sub></td><td>Quantile threshold for the conditioning event on X. Defines the tail of X's own empirical increment distribution that must be exceeded. Grid: {0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99}.</td></tr>
        <tr><td class="ticker" style="color:var(--accent)">q<sub>Y</sub></td><td>Quantile threshold for the exceedance event on Y. Defines what counts as a large move in Y's own future increment distribution. Applies equally to all 136 Y assets. Same grid as q<sub>X</sub>.</td></tr>
        <tr><td class="ticker" style="color:var(--accent)">r<sub>q</sub>(&middot;)</td><td>Empirical q-th quantile of the full-sample increment distribution for a given (asset, &tau;) pair. Estimated once from all available history.</td></tr>
        <tr><td class="ticker" style="color:var(--accent)">n</td><td>Number of historical overlapping windows in which the conditioning event on X was satisfied. Minimum enforced: n &ge; 100 for both pairwise and joint CPE.</td></tr>
        <tr><td class="ticker" style="color:var(--accent)">Lift</td><td>CPE / (1 &minus; q<sub>Y</sub>). Amplification of exceedance probability over the unconditional base rate. Minimum enforced: Lift &ge; 1.5.</td></tr>
      </tbody>
    </table>
    <div class="mth-section" style="margin-bottom:12px">
      <p>How the &tau;-day increment is defined for each asset class, and whether that class is used as a predicted variable Y:</p>
    </div>
    <table class="signal-table" style="max-width:860px">
      <thead><tr><th>Asset Class</th><th>Increment &Delta;<sup>(&tau;)</sup></th><th>Used as Y?</th><th>Notes</th></tr></thead>
      <tbody>
        <tr><td><span class="ac-badge" style="background:#3B82F622;color:#3B82F6;border:1px solid #3B82F644">equities</span></td><td class="ticker">ln(P<sub>t</sub> / P<sub>t&minus;&tau;</sub>)</td><td style="color:var(--bull)">&#10003; Yes</td><td>Adjusted close (split &amp; dividend adjusted)</td></tr>
        <tr><td><span class="ac-badge" style="background:#F59E0B22;color:#F59E0B;border:1px solid #F59E0B44">crypto</span></td><td class="ticker">ln(P<sub>t</sub> / P<sub>t&minus;&tau;</sub>)</td><td style="color:var(--bull)">&#10003; Yes</td><td>USD spot prices and ETF adjusted close</td></tr>
        <tr><td><span class="ac-badge" style="background:#10B98122;color:#10B981;border:1px solid #10B98144">commodities</span></td><td class="ticker">ln(P<sub>t</sub> / P<sub>t&minus;&tau;</sub>)</td><td style="color:var(--bull)">&#10003; Yes</td><td>ETF adjusted close; futures use front-month contract price</td></tr>
        <tr><td><span class="ac-badge" style="background:#EF444422;color:#EF4444;border:1px solid #EF444444">rates (ETFs)</span></td><td class="ticker">ln(P<sub>t</sub> / P<sub>t&minus;&tau;</sub>)</td><td style="color:var(--bull)">&#10003; Yes</td><td>Price-based ETFs only (SHY, IEF, TLT, HYG etc.)</td></tr>
        <tr><td><span class="ac-badge" style="background:#06B6D422;color:#06B6D4;border:1px solid #06B6D444">fx</span></td><td class="ticker">ln(P<sub>t</sub> / P<sub>t&minus;&tau;</sub>)</td><td style="color:var(--bull)">&#10003; Yes</td><td>USD per unit of foreign currency (Yahoo Finance mid-price)</td></tr>
        <tr><td><span class="ac-badge" style="background:#8B5CF622;color:#8B5CF6;border:1px solid #8B5CF644">vol indices</span></td><td class="ticker">P<sub>t</sub> &minus; P<sub>t&minus;&tau;</sub></td><td style="color:var(--bear)">&#10007; X only</td><td>^VIX, ^VXN, ^OVX, ^GVZ, ^EVZ, ^VVIX, ^SKEW &mdash; not price series; level changes used</td></tr>
        <tr><td><span class="ac-badge" style="background:#EF444422;color:#EF4444;border:1px solid #EF444444">yield indices</span></td><td class="ticker">P<sub>t</sub> &minus; P<sub>t&minus;&tau;</sub></td><td style="color:var(--bear)">&#10007; X only</td><td>^TNX, ^TYX, ^FVX, ^IRX &mdash; yield levels in %; level changes used</td></tr>
      </tbody>
    </table>
  </div>

  <div class="card">
    <div class="card-title">Pairwise CPE &mdash; Definition</div>
    <div class="mth-section">
      <p>For predicted asset <em>Y</em>, predictor <em>X</em>, look-back window
      &tau;<sub>p</sub>, forward window &tau;<sub>f</sub>, and quantile thresholds
      q<sub>X</sub>, q<sub>Y</sub>, the <strong class="hl">pairwise Conditional
      Probability of Exceedance (CPE)</strong> is:</p>
      <div class="formula-box formula-bull">
        CPE(Y, X, &tau;<sub>p</sub>, &tau;<sub>f</sub>, q<sub>X</sub>, q<sub>Y</sub>)
        &nbsp;=&nbsp;
        P(&Delta;<sup>(&tau;<sub>f</sub>)</sup>Y<sub>t</sub> &gt; r<sub>q<sub>Y</sub></sub>(Y)
        &nbsp;|&nbsp;
        &Delta;<sup>(&tau;<sub>p</sub>)</sup>X<sub>t</sub> &gt; r<sub>q<sub>X</sub></sub>(X))
      </div>
      <p>where the increments are:</p>
      <ul>
        <li><span style="font-family:var(--font-mono);color:var(--accent)">&Delta;<sup>(&tau;)</sup>Y<sub>t</sub> = ln(P<sup>Y</sup><sub>t+&tau;</sub> / P<sup>Y</sup><sub>t</sub>)</span>
            &mdash; &tau;-day log return of Y from t (forward-looking)</li>
        <li><span style="font-family:var(--font-mono);color:var(--accent)">&Delta;<sup>(&tau;)</sup>X<sub>t</sub> = ln(P<sup>X</sup><sub>t</sub> / P<sup>X</sup><sub>t&minus;&tau;</sub>)</span>
            &mdash; &tau;-day log return of X ending at t (backward-looking)</li>
      </ul>
      <p style="margin-top:10px">For the <strong class="hl">bearish direction</strong>, both inequalities are reversed
      (lower tail conditioning predicts lower tail outcome):</p>
      <div class="formula-box formula-bear">
        CPE<sub>bear</sub>
        &nbsp;=&nbsp;
        P(&Delta;<sup>(&tau;<sub>f</sub>)</sup>Y<sub>t</sub> &lt; r<sub>1&minus;q<sub>Y</sub></sub>(Y)
        &nbsp;|&nbsp;
        &Delta;<sup>(&tau;<sub>p</sub>)</sup>X<sub>t</sub> &lt; r<sub>1&minus;q<sub>X</sub></sub>(X))
      </div>
      <p>The empirical estimator counts co-exceedance events over all overlapping windows:</p>
      <div class="formula-box formula-neutral">
        <span style="color:var(--accent)">CPE&#770;</span>
        &nbsp;=&nbsp;
        |{t : &Delta;<sup>(&tau;<sub>p</sub>)</sup>X<sub>t</sub> &gt; r<sub>q<sub>X</sub></sub>(X)
        &nbsp;&and;&nbsp;
        &Delta;<sup>(&tau;<sub>f</sub>)</sup>Y<sub>t</sub> &gt; r<sub>q<sub>Y</sub></sub>(Y)}|
        &nbsp;/&nbsp;
        |{t : &Delta;<sup>(&tau;<sub>p</sub>)</sup>X<sub>t</sub> &gt; r<sub>q<sub>X</sub></sub>(X)}|
      </div>
      <p><strong class="hl">Overlapping windows</strong> are used throughout. Each calendar date t
      constitutes one observation regardless of window overlap with adjacent dates, maximising the
      number of conditioning events especially at large &tau; values.</p>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Lift &mdash; Amplification over Base Rate</div>
    <div class="mth-section">
      <p>Lift measures how much the conditioning event amplifies the exceedance probability
      relative to its unconditional value:</p>
      <div class="formula-box formula-bull">
        Lift &nbsp;=&nbsp; CPE&#770; / P(&Delta;<sup>(&tau;<sub>f</sub>)</sup>Y<sub>t</sub>
        &gt; r<sub>q<sub>Y</sub></sub>(Y)) &nbsp;=&nbsp; CPE&#770; / (1 &minus; q<sub>Y</sub>)
      </div>
      <ul>
        <li><span style="font-family:var(--font-mono);color:var(--accent)">Lift = 1.0</span>
            &mdash; conditioning provides no information; CPE equals base rate</li>
        <li><span style="font-family:var(--font-mono);color:var(--accent)">Lift = 1.5</span>
            &mdash; conditioning increases exceedance probability by 50%</li>
        <li><span style="font-family:var(--font-mono);color:var(--accent)">Lift = 2.0</span>
            &mdash; conditioning doubles the exceedance probability</li>
      </ul>
      <p style="margin-top:10px">All signals satisfy <strong class="hl">CPE &ge; 0.80</strong>,
      <strong class="hl">Lift &ge; 1.5</strong>, and <strong class="hl">n &ge; 100</strong>.</p>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Joint CPE &mdash; Definition</div>
    <div class="mth-section">
      <p>Joint CPE generalises pairwise CPE to a set of K predictors
      {X<sub>1</sub>, &hellip;, X<sub>K</sub>}, each with their own look-back window
      &tau;<sub>p,k</sub> and quantile threshold q<sub>X<sub>k</sub></sub>:</p>
      <div class="formula-box formula-bull">
        CPE<sub>joint</sub> &nbsp;=&nbsp;
        P(&Delta;<sup>(&tau;<sub>f</sub>)</sup>Y<sub>t</sub> &gt; r<sub>q<sub>Y</sub></sub>(Y)
        &nbsp;|&nbsp;
        &bigcap;<sub>k=1</sub><sup>K</sup>
        {&Delta;<sup>(&tau;<sub>p,k</sub>)</sup>X<sub>k,t</sub>
        &gt; r<sub>q<sub>X<sub>k</sub></sub></sub>(X<sub>k</sub>)})
      </div>
      <p>The joint conditioning event is the <strong class="hl">intersection</strong> of all K
      individual conditioning events &mdash; all predictors must simultaneously exceed their
      respective thresholds. Different predictors may have different look-back windows.</p>
      <p style="margin-top:10px"><strong class="hl">Greedy predictor selection:</strong>
      Starting from the single predictor with the highest pairwise CPE, at each step the predictor
      is added that maximises joint CPE while maintaining n<sub>joint</sub> &ge; 100 and
      Lift &ge; 1.5. Each ticker appears at most once (deduplication enforced). The algorithm
      terminates when no further predictor can be added satisfying the constraints,
      or when K = K<sub>max</sub> = 10.</p>
      <p style="margin-top:10px"><strong class="hl">Statistical discipline:</strong>
      The minimum joint conditioning sample size n<sub>joint</sub> &ge; 100 is enforced at all
      set sizes identically to the pairwise minimum and is never relaxed for larger K.
      If no predictor can be added while maintaining n<sub>joint</sub> &ge; 100,
      the algorithm stops. This ensures all reported joint CPE values have genuine
      statistical support.</p>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Parameter Grid</div>
    <div class="grid-2">
      <table class="signal-table">
        <thead><tr><th>&tau; value</th><th>Calendar interpretation</th></tr></thead>
        <tbody>
          <tr><td class="ticker" style="color:var(--accent)">1</td><td>1 trading day</td></tr>
          <tr><td class="ticker" style="color:var(--accent)">5</td><td>1 calendar week</td></tr>
          <tr><td class="ticker" style="color:var(--accent)">10</td><td>2 calendar weeks</td></tr>
          <tr><td class="ticker" style="color:var(--accent)">21</td><td>~1 calendar month</td></tr>
          <tr><td class="ticker" style="color:var(--accent)">63</td><td>~1 fiscal quarter</td></tr>
          <tr><td class="ticker" style="color:var(--accent)">126</td><td>~6 calendar months</td></tr>
          <tr><td class="ticker" style="color:var(--accent)">252</td><td>~1 calendar year</td></tr>
          <tr><td class="ticker" style="color:var(--accent)">300</td><td>~15 calendar months</td></tr>
        </tbody>
      </table>
      <table class="signal-table">
        <thead><tr><th>q value</th><th>Upper tail probability</th><th>Approx. frequency</th></tr></thead>
        <tbody>
          <tr><td class="ticker" style="color:var(--accent)">0.50</td><td>50%</td><td>~126 days/year</td></tr>
          <tr><td class="ticker" style="color:var(--accent)">0.60</td><td>40%</td><td>~101 days/year</td></tr>
          <tr><td class="ticker" style="color:var(--accent)">0.70</td><td>30%</td><td>~76 days/year</td></tr>
          <tr><td class="ticker" style="color:var(--accent)">0.75</td><td>25%</td><td>~63 days/year</td></tr>
          <tr><td class="ticker" style="color:var(--accent)">0.80</td><td>20%</td><td>~50 days/year</td></tr>
          <tr><td class="ticker" style="color:var(--accent)">0.90</td><td>10%</td><td>~25 days/year</td></tr>
          <tr><td class="ticker" style="color:var(--accent)">0.95</td><td>5%</td><td>~13 days/year</td></tr>
          <tr><td class="ticker" style="color:var(--accent)">0.99</td><td>1%</td><td>~2&ndash;3 days/year</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Data Universe &amp; Sources</div>
    <div class="mth-section">
      <p>All price data sourced from <strong class="hl">Yahoo Finance</strong> via the
      <code style="font-family:var(--font-mono);color:var(--accent)">yfinance</code> Python library.
      Adjusted closing prices are used for all equity and commodity ETFs (split- and dividend-adjusted).
      Futures series use front-month contracts. FX rates are mid-price quotes.</p>
      <p>The following assets are <strong class="hl">excluded from the predicted set (Y)</strong>
      to avoid mechanically implied signals:</p>
      <ul>
        <li><strong class="hl">Leveraged/inverse ETFs:</strong>
            SSO, SDS, TQQQ, TMF, TBT, TBF, UVXY, SVXY, VIXY, VIXM, VXX &mdash;
            returns are deterministic functions of their underlying assets</li>
        <li><strong class="hl">Managed/pegged currencies:</strong>
            THBUSD=X, CNYUSD=X, KRWUSD=X &mdash;
            heavy central bank intervention compresses increment distributions,
            generating spurious quantile exceedances</li>
      </ul>
      <p>VIX-family and Treasury yield indices are included as
      <strong class="hl">predictors only</strong>, using level changes (not log returns).</p>
      <p>Quantile thresholds r<sub>q</sub> are estimated using the
      <strong class="hl">full historical sample</strong> for each (asset, &tau;) pair.
      This is appropriate for a purely descriptive analysis &mdash; no out-of-sample
      forecasting is claimed.</p>
    </div>
  </div>

</div>

</main>
""")

    # Write the JS separately to avoid f-string issues
    f.write("""<script>
const D = """ + data_json + """;
const AC = D.ac_colors;

function showPanel(name, btn) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  if (btn) btn.classList.add('active');
}

const PL = {
  paper_bgcolor:'transparent', plot_bgcolor:'#111827',
  font:{family:'Space Mono,monospace',color:'#94A3B8',size:11},
  margin:{l:60,r:20,t:30,b:60},
  xaxis:{gridcolor:'#2D3748',linecolor:'#2D3748'},
  yaxis:{gridcolor:'#2D3748',linecolor:'#2D3748'},
  legend:{bgcolor:'#1F2937',bordercolor:'#2D3748',borderwidth:1},
};

function pl(id, traces, layout) {
  Plotly.newPlot(id, traces, Object.assign({}, PL, layout), {responsive:true, displayModeBar:false});
}

function acBadge(ac) {
  const c = AC[ac]||'#6B7280';
  return '<span class="ac-badge" style="background:'+c+'22;color:'+c+';border:1px solid '+c+'44">'+ac+'</span>';
}

function dirPill(d) {
  return '<span class="dir-pill '+d+'">'+(d==='bullish'?'&#9650;':'&#9660;')+' '+d+'</span>';
}

function cpeBar(cpe, dir) {
  const c = dir==='bullish'?'var(--bull)':'var(--bear)';
  const w = Math.round(cpe*100);
  return '<div class="cpe-bar"><div class="cpe-fill" style="width:'+w+'px;max-width:100px;background:'+c+'"></div><span class="cpe-val">'+cpe.toFixed(4)+'</span></div>';
}

// OVERVIEW
function initOverview() {
  document.getElementById('gen-date').textContent = D.generated;
  const bull = D.joint.filter(r=>r.direction==='bullish').length;
  const bear = D.joint.filter(r=>r.direction==='bearish').length;
  document.getElementById('overview-stats').innerHTML = [
    ['Pairwise Signals','169K','CPE >= 0.80 & lift >= 1.5 & n >= 100'],
    ['Joint Bullish', bull.toLocaleString(),'up to 10 predictors'],
    ['Joint Bearish', bear.toLocaleString(),'up to 10 predictors'],
    ['Asset Classes','6','equities, crypto, fx, rates, commodities, vol'],
    ['Max tau window','300d','~15 calendar months'],
    ['Min n_joint','100','identical to pairwise minimum'],
  ].map(function(x){return '<div class="stat-card"><div class="stat-label">'+x[0]+'</div><div class="stat-value">'+x[1]+'</div><div class="stat-sub">'+x[2]+'</div></div>';}).join('');

  const s2 = D.joint.filter(r=>r.n_predictors===2);
  function sigCard(r) {
    const c = r.direction==='bullish'?'var(--bull)':'var(--bear)';
    const preds = r.predictors.map(function(x,i){return '<span class="pred-chip">'+x+' t='+r.tau_pasts[i]+' q='+r.q_Xs[i]+'</span>';}).join('<span class="pred-sep">n</span>');
    return '<div class="sig-card"><div style="display:flex;justify-content:space-between;align-items:center"><span class="sig-y ticker" style="color:'+AC[r.Y_class]+'">'+r.Y+'</span><span class="sig-cpe" style="color:'+c+'">'+r.joint_CPE.toFixed(4)+'</span></div><div class="sig-detail">'+dirPill(r.direction)+' &nbsp; tf='+r.tau_future+'d &nbsp; qY='+r.q_Y+' &nbsp; n='+r.n_joint+'</div><div class="sig-preds" style="margin-top:8px">'+preds+'</div></div>';
  }
  const topBull = s2.filter(r=>r.direction==='bullish').sort(function(a,b){return b.joint_CPE-a.joint_CPE||b.n_joint-a.n_joint;}).slice(0,8);
  const topBear = s2.filter(r=>r.direction==='bearish').sort(function(a,b){return b.joint_CPE-a.joint_CPE||b.n_joint-a.n_joint;}).slice(0,8);
  document.getElementById('ov-bull').innerHTML = '<div class="overview-grid">'+topBull.map(sigCard).join('')+'</div>';
  document.getElementById('ov-bear').innerHTML = '<div class="overview-grid">'+topBear.map(sigCard).join('')+'</div>';

  var traces = ['bullish','bearish'].map(function(dir){
    var s = D.cpe_stats[dir];
    return {x:s.map(function(x){return x.n_predictors;}), y:s.map(function(x){return x.mean_cpe;}),
      error_y:{type:'data',array:s.map(function(x){return x.std_cpe;}),visible:true,color:dir==='bullish'?'#34D399':'#F87171',thickness:1.5,width:4},
      mode:'lines+markers', name:dir,
      line:{color:dir==='bullish'?'#34D399':'#F87171',width:2}, marker:{size:6,color:dir==='bullish'?'#34D399':'#F87171'}};
  });
  pl('ov-cpe-plot', traces, {xaxis:{title:'n_predictors',dtick:1}, yaxis:{title:'Mean Joint CPE',range:[0.94,1.02]},
    shapes:[{type:'line',x0:2,x1:10,y0:1,y1:1,line:{color:'#475569',dash:'dot',width:1}}], showlegend:true});

  var fb = D.freq_predictors['bullish'].slice(0,15);
  var fbe = D.freq_predictors['bearish'].slice(0,15);
  pl('ov-freq-plot', [
    {x:fb.map(function(r){return r.count;}), y:fb.map(function(r){return r.ticker;}), type:'bar', orientation:'h', name:'bullish', marker:{color:fb.map(function(r){return AC[r.ac]||'#6B7280';})}},
    {x:fbe.map(function(r){return -r.count;}), y:fbe.map(function(r){return r.ticker;}), type:'bar', orientation:'h', name:'bearish', marker:{color:fbe.map(function(r){return AC[r.ac]||'#6B7280';})}}
  ], {barmode:'overlay', xaxis:{title:'Frequency in joint sets (size 2)'}, yaxis:{title:''}, showlegend:true, margin:{l:100,r:20,t:20,b:50}});
}

// PAIRWISE
function initPairwise() {
  var ysel=document.getElementById('pw-Y'), xsel=document.getElementById('pw-X'),
      tfsel=document.getElementById('pw-tf'), qysel=document.getElementById('pw-qy');
  D.all_Y.forEach(function(y){ysel.add(new Option(y,y));});
  D.all_X.forEach(function(x){xsel.add(new Option(x,x));});
  D.all_tau.forEach(function(t){tfsel.add(new Option(t+'d',t));});
  D.all_qY.forEach(function(q){qysel.add(new Option('q='+q,q));});
  filterPairwise();
}

function filterPairwise() {
  var Y=document.getElementById('pw-Y').value, X=document.getElementById('pw-X').value,
      dir=document.getElementById('pw-dir').value, tf=document.getElementById('pw-tf').value,
      qy=document.getElementById('pw-qy').value, minc=parseFloat(document.getElementById('pw-mincpe').value)||0.80;
  var rows=D.pairwise_sample.filter(function(r){
    return r.CPE>=minc && (!Y||r.Y===Y) && (!X||r.X===X) && (!dir||r.direction===dir) && (!tf||r.tau_future==tf) && (!qy||r.q_Y==qy);
  }).sort(function(a,b){return b.CPE-a.CPE;}).slice(0,200);
  document.getElementById('pw-tbody').innerHTML = rows.map(function(r){
    return '<tr><td><span class="ticker" style="color:'+AC[r.Y_class]+'">'+r.Y+'</span></td><td>'+acBadge(r.Y_class)+'</td><td><span class="ticker" style="color:'+AC[r.X_class]+'">'+r.X+'</span></td><td>'+acBadge(r.X_class)+'</td><td>'+dirPill(r.direction)+'</td><td style="font-family:var(--font-mono)">'+r.tau_past+'d</td><td style="font-family:var(--font-mono)">'+r.tau_future+'d</td><td style="font-family:var(--font-mono)">'+r.q_X+'</td><td style="font-family:var(--font-mono)">'+r.q_Y+'</td><td>'+cpeBar(r.CPE,r.direction)+'</td><td style="font-family:var(--font-mono);color:var(--accent2)">'+r.lift.toFixed(2)+'x</td><td style="font-family:var(--font-mono);color:var(--text2)">'+r.n_condition+'</td></tr>';
  }).join('');
  document.getElementById('pw-count').textContent = 'Showing '+rows.length+' signals (capped at 200).';
}

// HEATMAP
function initHeatmap() {
  var sel=document.getElementById('hm-select');
  D.heatmaps.forEach(function(hm,i){sel.add(new Option(hm.label,i));});
  renderHeatmap();
}

function renderHeatmap() {
  var idx=parseInt(document.getElementById('hm-select').value)||0, hm=D.heatmaps[idx];
  if(!hm) return;
  var cs = hm.direction==='bullish'
    ? [['0','#064e3b'],['0.5','#059669'],['1','#34d399']]
    : [['0','#7f1d1d'],['0.5','#dc2626'],['1','#fca5a5']];
  pl('heatmap-plot', [{type:'heatmap',z:hm.z,x:hm.X_labels,y:hm.Y_labels,colorscale:cs,zmin:0.80,zmax:1.0,hoverongaps:false,
    colorbar:{title:'CPE',tickfont:{family:'Space Mono',size:10}}}],
    {xaxis:{title:'Predictor X',tickangle:-45,tickfont:{size:9}}, yaxis:{title:'Predicted Y',tickfont:{size:9}}, margin:{l:100,r:80,t:20,b:120}});
}

// TAU CURVES
function initTauCurves() {
  var ysel=document.getElementById('tc-Y'), xsel=document.getElementById('tc-X'), qysel=document.getElementById('tc-qy');
  var ys=[...new Set(D.tau_curves.map(function(r){return r.Y;}))].sort();
  var xs=[...new Set(D.tau_curves.map(function(r){return r.X;}))].sort();
  var qs=[...new Set(D.tau_curves.map(function(r){return r.q_Y;}))].sort();
  ys.forEach(function(y){ysel.add(new Option(y,y));});
  xs.forEach(function(x){xsel.add(new Option(x,x));});
  qs.forEach(function(q){qysel.add(new Option('q='+q,q));});
}

function renderTauCurves() {
  var Y=document.getElementById('tc-Y').value, X=document.getElementById('tc-X').value,
      dir=document.getElementById('tc-dir').value, qy=parseFloat(document.getElementById('tc-qy').value);
  var tfs=[1,5,10,21,63,126,252,300];
  var curves=D.tau_curves.filter(function(r){return r.Y===Y&&r.X===X&&r.direction===dir&&r.q_Y==qy;});
  if(!curves.length){document.getElementById('tau-plot').innerHTML='<div class="empty">No curves found. Try different Y, X, or q_Y.</div>';return;}
  var pal=['#38BDF8','#818CF8','#34D399','#F59E0B','#F87171','#A78BFA','#06B6D4','#EC4899'];
  var traces=curves.map(function(c,i){
    var x=[],y=[];
    tfs.forEach(function(tf){var v=c['tf_'+tf];if(v!==undefined){x.push(tf);y.push(v);}});
    return {x:x,y:y,mode:'lines+markers',name:'tp='+c.tau_past+' qX='+c.q_X,
      line:{color:pal[i%pal.length],width:2},marker:{size:5}};
  });
  traces.push({x:[1,300],y:[0.5,0.5],mode:'lines',name:'CPE=0.5 (random)',line:{color:'#475569',dash:'dot',width:1}});
  pl('tau-plot', traces, {
    xaxis:{title:'tau_future (trading days)',type:'log',tickvals:tfs,ticktext:tfs.map(String)},
    yaxis:{title:'CPE',range:[0.45,1.05]}, title:Y+' | '+X+' | '+dir+' | qY='+qy});
  document.getElementById('tau-info').textContent = curves.length+' curves shown. Each line = one (tau_past, q_X) combination.';
}

// JOINT
function initJoint() {
  var ysel=document.getElementById('jt-Y'), qysel=document.getElementById('jt-qy');
  var ys=[...new Set(D.joint.map(function(r){return r.Y;}))].sort();
  var qs=[...new Set(D.joint.map(function(r){return r.q_Y;}))].sort();
  ys.forEach(function(y){ysel.add(new Option(y,y));});
  qs.forEach(function(q){qysel.add(new Option('q='+q,q));});
  filterJoint();
  ['bullish','bearish'].forEach(function(dir){
    var sub=D.joint.filter(function(r){return r.direction===dir;});
    var col=dir==='bullish'?'#34D399':'#F87171';
    pl('jt-dist-'+(dir==='bullish'?'bull':'bear'),
      [{x:sub.map(function(r){return r.joint_CPE;}),type:'histogram',nbinsx:40,marker:{color:col,opacity:0.8},name:dir}],
      {xaxis:{title:'Joint CPE'},yaxis:{title:'Count'},bargap:0.05,margin:{l:50,r:20,t:20,b:50}});
  });
}

function filterJoint() {
  var Y=document.getElementById('jt-Y').value, dir=document.getElementById('jt-dir').value,
      np=document.getElementById('jt-np').value, qy=document.getElementById('jt-qy').value,
      minc=parseFloat(document.getElementById('jt-mincpe').value)||0.90;
  var rows=D.joint.filter(function(r){
    return r.joint_CPE>=minc && (!Y||r.Y===Y) && (!dir||r.direction===dir) && (!np||r.n_predictors==np) && (!qy||r.q_Y==qy);
  }).sort(function(a,b){return b.joint_CPE-a.joint_CPE||b.n_joint-a.n_joint;}).slice(0,150);
  document.getElementById('jt-tbody').innerHTML = rows.map(function(r){
    var chips=r.predictors.map(function(x,i){return '<span class="pred-chip">'+x+' t='+r.tau_pasts[i]+' q='+r.q_Xs[i]+'</span>';}).join('<span class="pred-sep">&#8745;</span>');
    return '<tr><td><span class="ticker" style="color:'+AC[r.Y_class]+'">'+r.Y+'</span></td><td>'+acBadge(r.Y_class)+'</td><td>'+dirPill(r.direction)+'</td><td style="font-family:var(--font-mono)">'+r.tau_future+'d</td><td style="font-family:var(--font-mono)">'+r.q_Y+'</td><td style="font-family:var(--font-mono);color:var(--accent)">'+r.n_predictors+'</td><td>'+cpeBar(r.joint_CPE,r.direction)+'</td><td style="font-family:var(--font-mono);color:var(--text2)">'+r.n_joint+'</td><td><div style="font-family:var(--font-mono);font-size:10px;line-height:1.8">'+chips+'</div></td></tr>';
  }).join('');
  document.getElementById('jt-count').textContent = 'Showing '+rows.length+' joint signals (capped at 150).';
}

// NETWORK
function renderNetwork() {
  var dir=document.getElementById('net-dir').value;
  var minCPE=parseFloat(document.getElementById('net-mincpe').value)||0.95;
  var maxLinks=parseInt(document.getElementById('net-maxlinks').value)||80;
  var linkMap={};
  D.pairwise_sample.filter(function(r){return r.direction===dir&&r.CPE>=minCPE;}).forEach(function(r){
    var key=r.Y+'||'+r.X;
    if(!linkMap[key]||r.CPE>linkMap[key].cpe) linkMap[key]={Y:r.Y,X:r.X,cpe:r.CPE,Y_class:r.Y_class,X_class:r.X_class};
  });
  var links=Object.values(linkMap).sort(function(a,b){return b.cpe-a.cpe;}).slice(0,maxLinks);
  if(!links.length){document.getElementById('network-plot').innerHTML='<div class="empty">No links at this CPE threshold. Lower Min CPE.</div>';return;}
  var nodeSet={};
  links.forEach(function(l){nodeSet[l.Y]=l.Y_class;nodeSet[l.X]=l.X_class;});
  var nodeList=Object.entries(nodeSet), nodeIdx={};
  nodeList.forEach(function(e,i){nodeIdx[e[0]]=i;});
  var n=nodeList.length;
  var nx=nodeList.map(function(_,i){return Math.cos(2*Math.PI*i/n);});
  var ny=nodeList.map(function(_,i){return Math.sin(2*Math.PI*i/n);});
  var ex=[],ey=[];
  links.forEach(function(l){var i=nodeIdx[l.Y],j=nodeIdx[l.X];ex.push(nx[i],nx[j],null);ey.push(ny[i],ny[j],null);});
  pl('network-plot',
    [{x:ex,y:ey,mode:'lines',line:{color:dir==='bullish'?'#34D39944':'#F8717144',width:1},hoverinfo:'none',showlegend:false},
     {x:nx,y:ny,mode:'markers+text',marker:{size:14,color:nodeList.map(function(e){return AC[e[1]]||'#6B7280';}),line:{color:'#0A0E1A',width:2}},
      text:nodeList.map(function(e){return e[0];}),textposition:'top center',textfont:{family:'Space Mono',size:8,color:'#94A3B8'},
      hovertext:nodeList.map(function(e){return e[0]+' ('+e[1]+')';}),hoverinfo:'text',showlegend:false}],
    {xaxis:{showgrid:false,zeroline:false,showticklabels:false},yaxis:{showgrid:false,zeroline:false,showticklabels:false},
     plot_bgcolor:'#0A0E1A',margin:{l:20,r:20,t:20,b:20}});
  document.getElementById('network-info').textContent = nodeList.length+' nodes · '+links.length+' links · dir='+dir+' · min CPE='+minCPE;
}

document.addEventListener('DOMContentLoaded', function() {
  initOverview();
  initPairwise();
  initHeatmap();
  initTauCurves();
  initJoint();
  renderNetwork();
});
</script>
</body>
</html>""")

size_mb = os.path.getsize(out_path) / 1e6
print("Saved: %s  (%.1f MB)" % (out_path, size_mb))
print("Open in any browser - no server needed.")
