import base64
import json
import os

PLOT_DIR = "pnl_plots"
RESULTS_PATH = "pnl_backtest_results.json"
OUT_PATH = "/private/tmp/claude-501/-Users-arrams-Documents-personal-work-quant-quant-regime-research-notebooks/8f3c07e9-8023-4678-bd30-533b4dda9c52/scratchpad/pnl_gallery.html"

TICKER_LABELS = {"EURUSDX": "EUR/USD", "BTC-USD": "BTC-USD", "VIX": "^VIX"}


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def img_tag(path, alt):
    return f'<img src="data:image/png;base64,{b64(path)}" alt="{alt}" loading="lazy">'


results = json.load(open(RESULTS_PATH))
alpha_results = json.load(open("alpha_test_results.json"))
alpha_own = json.load(open("alpha_test_own_benchmark_results.json"))
tickers = sorted(results.keys(), key=lambda t: -results[t]["strategy_net_total_return_pct"])
n_beat = sum(1 for t in tickers if results[t]["beats_buy_hold_net"])
n_alpha_spy = sum(1 for r in alpha_results.values() if r["significant_95"] and r["alpha_annualized_pct"] > 0)
n_alpha_own = sum(1 for r in alpha_own.values() if r["significant_vs_own_95"])
alpha_spy_tickers = {t for t, r in alpha_results.items() if r["significant_95"] and r["alpha_annualized_pct"] > 0}

nav_items, section_blocks = [], []
for tkr in tickers:
    r = results[tkr]
    ao = alpha_own.get(tkr)
    label = TICKER_LABELS.get(tkr, tkr)
    beats = r["beats_buy_hold_net"]
    nav_items.append(f'<a class="nav-link{"" if beats else " no"}" href="#tkr-{tkr.lower()}">{label}</a>')
    safe_tkr = tkr.replace('=', '').replace('^', '')
    badge = '<span class="badge">beats buy &amp; hold</span>' if beats else '<span class="badge no">does not beat buy &amp; hold</span>'
    alpha_line = ""
    if ao is not None:
        alpha_line = (f'<p class="subtitle">Alpha vs own buy-and-hold (the appropriate test of timing skill): '
                      f'{ao["alpha_vs_own_benchmark_pct"]:+.2f}%/yr '
                      f'(95% CI {ao["alpha_vs_own_ci_lo_pct"]:+.1f}% to {ao["alpha_vs_own_ci_hi_pct"]:+.1f}%), '
                      f't={ao["t_vs_own_benchmark"]:+.2f}, beta={ao["beta_vs_own_benchmark"]:.2f} &mdash; not distinguishable from zero</p>')
    section_blocks.append(f"""
<section class="plot-section" id="tkr-{tkr.lower()}">
  <div class="section-head">
    <h2>{label} {badge}</h2>
    <p class="subtitle">Master model: {r['winner']} @ {r['horizon']}d &mdash; strategy net {r['strategy_net_total_return_pct']:+.1f}% vs buy&amp;hold {r['buy_hold_total_return_pct']:+.1f}% &mdash; Sharpe {r['strategy_net_sharpe']:+.2f} (buy&amp;hold {r['buy_hold_sharpe']:+.2f}) &mdash; max drawdown {r['strategy_net_max_dd_pct']:.1f}% (buy&amp;hold {r['buy_hold_max_dd_pct']:.1f}%) &mdash; {r['n_position_changes']} position changes over {r['n_days']} days</p>
    {alpha_line}
  </div>
  <div class="plot-frame">{img_tag(os.path.join(PLOT_DIR, f"{safe_tkr}.png"), f"{label} P&L")}</div>
</section>
""")

nav_html = "\n".join(nav_items)
section_html = "\n".join(section_blocks)
summary_img = img_tag(os.path.join(PLOT_DIR, "_SUMMARY_all_instruments.png"), "P&L summary")
alpha_img = img_tag(os.path.join(PLOT_DIR, "_ALPHA_test_all.png"), "Alpha test vs SPY, all instruments")
alpha_own_img = img_tag(os.path.join(PLOT_DIR, "_ALPHA_test_own_benchmark.png"), "Corrected alpha test vs own instrument, all instruments")

target_results = json.load(open("target_price_strategy_results.json"))
portfolio_stats = json.load(open("portfolio_backtest_results.json"))
n_target_beat = sum(1 for r in target_results.values() if r["beats_buy_hold_net"])
n_target_alpha = sum(1 for r in target_results.values() if r["significant_vs_own_95"])
target_summary_img = img_tag(os.path.join(PLOT_DIR, "_TARGET_SUMMARY_all_instruments.png"), "Target-price strategy summary")
portfolio_img = img_tag(os.path.join(PLOT_DIR, "_PORTFOLIO_backtest.png"), "Portfolio backtest")

target_nav = "\n".join(
    f'<a class="nav-link{"" if target_results[t]["beats_buy_hold_net"] else " no"}" href="#target-{t.lower()}">{TICKER_LABELS.get(t, t)}</a>'
    for t in sorted(target_results, key=lambda t: -target_results[t]["strategy_net_total_return_pct"])
)
target_sections = []
for tkr in sorted(target_results, key=lambda t: -target_results[t]["strategy_net_total_return_pct"]):
    r = target_results[tkr]
    label = TICKER_LABELS.get(tkr, tkr)
    safe_tkr = tkr.replace('=', '').replace('^', '')
    badge = '<span class="badge">beats buy &amp; hold</span>' if r["beats_buy_hold_net"] else '<span class="badge no">does not beat buy &amp; hold</span>'
    target_sections.append(f"""
<section class="plot-section" id="target-{tkr.lower()}">
  <div class="section-head">
    <h2>{label} {badge}</h2>
    <p class="subtitle">Master model: {r['winner']} @ {r['horizon']}d &mdash; strategy net {r['strategy_net_total_return_pct']:+.1f}% vs buy&amp;hold {r['buy_hold_total_return_pct']:+.1f}% &mdash; {r['n_trades']} trades ({r['n_buys']} buys, {r['n_sells']} sells) over {r['n_days']} days &mdash; alpha vs own buy&amp;hold {r['alpha_vs_own_pct']:+.2f}%/yr (t={r['t_vs_own']:+.2f}, not significant)</p>
  </div>
  <div class="plot-frame">{img_tag(os.path.join(PLOT_DIR, f"_TARGET_{safe_tkr}.png"), f"{label} target-price strategy")}</div>
</section>
""")
target_section_html = "\n".join(target_sections)

final_pipeline = json.load(open("final_deployed_pipeline.json"))
final_pipeline_img = img_tag(os.path.join(PLOT_DIR, "_FINAL_DEPLOYED_PIPELINE.png"), "Final deployed forecasting pipeline")

biweekly_results = json.load(open("biweekly_postprocess_results.json"))
biweekly_cols = ["mape_moment_continuous", "mape_moment_firstraw", "mape_quantile_continuous", "mape_quantile_firstraw"]
biweekly_n_improved = {c: sum(1 for r in biweekly_results.values() if r[c] < r["mape_raw"] - 0.01) for c in biweekly_cols}
biweekly_heatmap_img = img_tag(os.path.join(PLOT_DIR, "_BIWEEKLY_heatmap.png"), "Bi-weekly post-processing heatmap")
postproc_final_summary_img = img_tag(os.path.join(PLOT_DIR, "_POSTPROC_FINAL_SUMMARY.png"), "Post-processing final summary across all 5 designs")

rolling_results = json.load(open("rolling_postprocess_results.json"))
n_rolling_improved = sum(1 for r in rolling_results.values() if r["improved"])
n_rolling_late_improved = sum(1 for r in rolling_results.values() if r["improved_late"])
n_rolling_has_late = sum(1 for r in rolling_results.values() if r["improved_late"] is not None)
rolling_summary_img = img_tag(os.path.join(PLOT_DIR, "_ROLLING_SUMMARY.png"), "Rolling post-processing summary")
rolling_earlylate_img = img_tag(os.path.join(PLOT_DIR, "_ROLLING_EARLY_LATE.png"), "Rolling early vs late")
rolling_jpm_img = img_tag(os.path.join(PLOT_DIR, "_ROLLING_JPM_detail.png"), "JPM rolling detail")
rolling_gld_img = img_tag(os.path.join(PLOT_DIR, "_ROLLING_GLD_detail.png"), "GLD rolling detail")

postproc_results = json.load(open("post_processing_results.json"))
n_postproc_improved = sum(1 for r in postproc_results.values() if r["selected_method"] != "raw" and r["mape_price_selected"] < r["mape_price_raw"] - 0.01)
postproc_summary_img = img_tag(os.path.join(PLOT_DIR, "_POSTPROC_SUMMARY.png"), "Post-processing summary")
postproc_jpm_img = img_tag(os.path.join(PLOT_DIR, "_POSTPROC_JPM_detail.png"), "JPM post-processing detail")
postproc_jpm_alpha_img = img_tag(os.path.join(PLOT_DIR, "_POSTPROC_JPM_alpha_retest.png"), "JPM post-processing alpha retest")
postproc_jpm_bias_img = img_tag(os.path.join(PLOT_DIR, "_POSTPROC_JPM_bias_stability.png"), "JPM bias stability over time")

stacked_results = json.load(open("stacked_postprocess_results.json"))
n_stacked_improved = sum(1 for r in stacked_results.values() if r["mape_stacked_postprocessed"] < r["mape_old_single_winner"] - 0.01)
stacked_summary_img = img_tag(os.path.join(PLOT_DIR, "_STACKED_SUMMARY.png"), "Stacked regression summary")

percand_results = json.load(open("per_candidate_postprocess_results.json"))
n_percand_cells_improved = sum(1 for r in percand_results.values() for c in r["candidates"].values() if c["mape_postprocessed"] < c["mape_raw"] - 0.01)
n_percand_total_cells = sum(len(r["candidates"]) for r in percand_results.values())
n_percand_any_beats_old = sum(1 for r in percand_results.values() if min(c["mape_postprocessed"] for c in r["candidates"].values()) < (r["old_single_winner_mape"] or 999) - 0.01)
percand_heatmap_img = img_tag(os.path.join(PLOT_DIR, "_PERCAND_heatmap.png"), "Per-candidate post-processing heatmap")

relval_stats = json.load(open("relative_value_results.json"))
relval_img = img_tag(os.path.join(PLOT_DIR, "_RELVAL_long_short.png"), "Relative-value long/short")

kelly_results = json.load(open("kelly_strategy_results.json"))
kelly_portfolio_stats = json.load(open("kelly_portfolio_backtest_results.json"))
n_kelly_beat = sum(1 for r in kelly_results.values() if r["beats_buy_hold_net"])
n_kelly_alpha = sum(1 for r in kelly_results.values() if r["significant_vs_own_95"])
kelly_summary_img = img_tag(os.path.join(PLOT_DIR, "_KELLY_SUMMARY_all_instruments.png"), "Kelly-sized strategy summary")
kelly_portfolio_img = img_tag(os.path.join(PLOT_DIR, "_KELLY_PORTFOLIO_backtest.png"), "Kelly-sized portfolio backtest")

kelly_nav = "\n".join(
    f'<a class="nav-link{"" if kelly_results[t]["beats_buy_hold_net"] else " no"}" href="#kelly-{t.lower()}">{TICKER_LABELS.get(t, t)}</a>'
    for t in sorted(kelly_results, key=lambda t: -kelly_results[t]["strategy_net_total_return_pct"])
)
kelly_sections = []
for tkr in sorted(kelly_results, key=lambda t: -kelly_results[t]["strategy_net_total_return_pct"]):
    r = kelly_results[tkr]
    label = TICKER_LABELS.get(tkr, tkr)
    safe_tkr = tkr.replace('=', '').replace('^', '')
    badge = '<span class="badge">beats buy &amp; hold</span>' if r["beats_buy_hold_net"] else '<span class="badge no">does not beat buy &amp; hold</span>'
    kelly_sections.append(f"""
<section class="plot-section" id="kelly-{tkr.lower()}">
  <div class="section-head">
    <h2>{label} {badge}</h2>
    <p class="subtitle">Master model: {r['winner']} @ {r['horizon']}d &mdash; strategy net {r['strategy_net_total_return_pct']:+.1f}% vs buy&amp;hold {r['buy_hold_total_return_pct']:+.1f}% &mdash; Sharpe {r['strategy_net_sharpe']:+.2f} (buy&amp;hold {r['buy_hold_sharpe']:+.2f}) &mdash; max DD {r['strategy_net_max_dd_pct']:.1f}% (buy&amp;hold {r['buy_hold_max_dd_pct']:.1f}%) &mdash; avg size when in {r['avg_kelly_fraction_when_in']:.2f}x &mdash; alpha vs own buy&amp;hold {r['alpha_vs_own_pct']:+.2f}%/yr (t={r['t_vs_own']:+.2f}, not significant)</p>
  </div>
  <div class="plot-frame">{img_tag(os.path.join(PLOT_DIR, f"_KELLY_{safe_tkr}.png"), f"{label} Kelly-sized strategy")}</div>
</section>
""")
kelly_section_html = "\n".join(kelly_sections)

html = f"""<title>P&amp;L Backtest — master model vs. buy &amp; hold</title>
<style>
:root {{
  --bg: #f6f5f2; --surface: #ffffff; --text: #1a1d22; --text-muted: #5b6270;
  --accent: #3d6a99; --border: #e2e1dd; --shadow: 0 1px 2px rgba(20, 23, 28, 0.06);
  --good: #2f8a4e; --bad: #b0492f;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #14171c; --surface: #1c2028; --text: #e8e9ec; --text-muted: #9aa1ad;
    --accent: #7aa8d4; --border: #2a2f38; --shadow: 0 1px 3px rgba(0, 0, 0, 0.4); --good: #4fbf74; --bad: #e0805f;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #14171c; --surface: #1c2028; --text: #e8e9ec; --text-muted: #9aa1ad;
  --accent: #7aa8d4; --border: #2a2f38; --shadow: 0 1px 3px rgba(0, 0, 0, 0.4); --good: #4fbf74; --bad: #e0805f;
}}
:root[data-theme="light"] {{
  --bg: #f6f5f2; --surface: #ffffff; --text: #1a1d22; --text-muted: #5b6270;
  --accent: #3d6a99; --border: #e2e1dd; --shadow: 0 1px 2px rgba(20, 23, 28, 0.06); --good: #2f8a4e; --bad: #b0492f;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--text);
  font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-variant-numeric: tabular-nums;
}}
.layout {{ display: grid; grid-template-columns: 220px 1fr; min-height: 100vh; }}
.sidebar {{
  position: sticky; top: 0; height: 100vh; overflow-y: auto;
  border-right: 1px solid var(--border); padding: 24px 16px; background: var(--surface);
}}
.brand {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13px; letter-spacing: 0.04em; text-transform: uppercase;
  color: var(--text-muted); margin: 0 0 4px 4px;
}}
.brand-title {{ font-size: 17px; font-weight: 650; margin: 0 0 20px 4px; text-wrap: balance; }}
nav {{ display: flex; flex-direction: column; }}
.nav-link {{
  display: block; padding: 5px 8px; border-radius: 6px; color: var(--good);
  text-decoration: none; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13px;
}}
.nav-link.no {{ color: var(--text-muted); }}
.nav-link:hover {{ background: var(--bg); }}
main {{ padding: 40px 48px 96px; max-width: 1200px; }}
.page-head {{ margin-bottom: 32px; }}
.page-head h1 {{ font-size: 24px; font-weight: 650; margin: 0 0 8px; text-wrap: balance; }}
.page-head p {{ color: var(--text-muted); max-width: 78ch; line-height: 1.55; margin: 0 0 8px; }}
.callout {{
  background: color-mix(in srgb, var(--accent) 8%, var(--surface)); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 16px; font-size: 13px; margin-top: 12px; max-width: 78ch;
}}
.summary-frame {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 8px; box-shadow: var(--shadow); margin-top: 20px; max-width: 720px;
}}
.summary-frame img {{ display: block; width: 100%; height: auto; border-radius: 4px; }}
.plot-section {{ padding-top: 24px; margin-top: 24px; border-top: 1px solid var(--border); scroll-margin-top: 20px; }}
.plot-section:first-of-type {{ border-top: none; margin-top: 0; }}
.section-head h2 {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 19px; font-weight: 650; margin: 0 0 4px; display: flex; align-items: center; gap: 10px;
}}
.section-head .subtitle {{ color: var(--text-muted); font-size: 13px; margin: 0 0 12px; }}
.badge {{
  font-family: ui-sans-serif, sans-serif; font-size: 11px; font-weight: 600; padding: 2px 8px;
  border-radius: 20px; background: color-mix(in srgb, var(--good) 18%, transparent); color: var(--good);
  text-transform: uppercase; letter-spacing: 0.03em;
}}
.badge.no {{ background: color-mix(in srgb, var(--bad) 15%, transparent); color: var(--bad); }}
.badge.alpha {{ background: color-mix(in srgb, var(--good) 22%, transparent); color: var(--good); }}
.plot-frame {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 8px; box-shadow: var(--shadow); overflow-x: auto; max-width: 760px;
}}
.plot-frame img {{ display: block; width: 100%; height: auto; border-radius: 4px; }}
</style>

<div class="layout">
  <aside class="sidebar">
    <p class="brand">predictor_v1</p>
    <p class="brand-title">P&amp;L backtest</p>
    <nav>
      <a class="nav-link" href="#final-pipeline-top" style="font-weight:700;">Final deployed pipeline</a>
      <a class="nav-link" href="#postproc-top" style="font-weight:700; margin-top:8px;">Post-processing / bias correction</a>
      <a class="nav-link" href="#relval-top" style="font-weight:700; margin-top:8px;">Relative-value long/short</a>
      <a class="nav-link" href="#target-price-top" style="font-weight:700; margin-top:8px;">Target-price strategy</a>
      {target_nav}
      <a class="nav-link" href="#portfolio-top" style="font-weight:700; margin-top:8px;">Portfolio backtest</a>
      <a class="nav-link" href="#kelly-top" style="font-weight:700; margin-top:8px;">Kelly-sized strategy</a>
      {kelly_nav}
      <a class="nav-link" href="#kelly-portfolio-top" style="font-weight:700; margin-top:8px;">Kelly portfolio</a>
      <a class="nav-link" href="#baseline-top" style="font-weight:700; margin-top:8px;">Earlier baseline (superseded)</a>
      {nav_html}
    </nav>
  </aside>
  <main>
    <div class="page-head" id="final-pipeline-top">
      <h1>Final deployed forecasting pipeline</h1>
      <p>The end state of the whole post-processing investigation: keep the correction only where it's actually been validated, use the raw master-model forecast everywhere else. Only <b>GLD and JPM</b> get a correction applied &mdash; the rolling/adaptive design (refit every 5 trading days on a trailing, horizon-scaled window of already-resolved outcomes), the only one of the five designs tried that's a genuinely sustainable, ongoing walk-forward process rather than a one-time static fit that would go stale over time. All other 20 instruments use the plain master-model forecast, uncorrected.</p>
      <p class="callout">One nuance worth being explicit about: JPM's rolling correction only paid off once its trailing window had matured (2024 onward) &mdash; the whole test period's average was actually flat-to-worse, dragged down by an early "still catching up" phase. Since deployment starts from today (mid-2026), that transient is already more than two years behind us, so the mature-phase performance (JPM: 13.4%&rarr;9.6% MAPE; GLD: 20.2%&rarr;12.0% MAPE, both shown here) is the honestly relevant number for what to expect going forward, not the whole-period average. Separately, already established: JPM's MAPE improvement does NOT translate into any additional trading alpha (its raw signal already saturates every trading rule tested) &mdash; this pipeline improves its forecast accuracy, not necessarily its economic value.</p>
    </div>
    <div class="summary-frame">{final_pipeline_img}</div>

    <div class="page-head" id="postproc-top" style="margin-top:56px; padding-top:24px; border-top:2px solid var(--border);">
      <h1>Final verdict: was any post-processing design actually useful?</h1>
      <p>Five structurally different post-processing designs were tried this session (single-model with a recency-based calibration/verification split, stacked regression blending all four candidates, per-candidate independent correction, rolling/adaptive daily-refit correction, and bi-weekly continuous correction with two separate techniques). Each instrument was processed entirely independently throughout -- its own horizon, its own FSS-selected winning candidate, its own fitting windows, its own corrections, its own final MAPE, with no pooling or sharing across instruments anywhere in any design.</p>
      <p class="callout"><b>18 of 22 instruments were never improved by any of the 5 designs</b> -- every attempt made them worse or had no effect. <b>GLD is the one instrument with a real, repeatable signal</b> -- 4 of 5 designs improved it, across genuinely different correction mechanisms (blended regression, per-candidate correction, rolling, and bi-weekly quantile mapping), which is strong evidence its forecast has an actual correctable bias, not a fluke. <b>JPM shows a real but more fragile signal</b> -- 3 of 5 designs helped, but not the more aggressive short-window ones, suggesting its bias needs a longer, more stable calibration window to capture. IWM, XLK, and XLI each improved under exactly one design (the very first one) and never replicated under any of the four subsequent, more careful tests -- the kind of isolated, non-repeating result that's more likely noise than signal, consistent with this project's standing rule against trusting single unreplicated positives.</p>
    </div>
    <div class="summary-frame">{postproc_final_summary_img}</div>

    <div class="page-head" style="margin-top:40px;">
      <h1>Bi-weekly continuous post-processing -- fifth design, precisely specified</h1>
      <p>The user's exact, detailed correction of every earlier attempt: model selection returns to the original FSS-skill-based single winner (<code>best_config_selection_holdout.json</code>, not the later MAPE-overridden version), with climatology included as a genuine candidate (it wins whenever the best informed variant's own selection-period skill is &le;0). Post-processing in the test period runs on a strict bi-weekly cadence: every month's SECOND half is always corrected using a fit from that same month's FIRST half's newly-resolved (prediction, actual) pairs; every month's FIRST half has two variations -- keep the correction running continuously using the previous half-month's fit, or leave it completely raw. Two techniques tested separately (not chained): PDF/moment matching, and frequency-corrected quantile mapping. 2 techniques &times; 2 first-half variations = 4 combinations, all on real, no-look-ahead data (a prediction's outcome is only used once its own horizon has actually elapsed).</p>
      <p><b>A real bug caught and fixed before trusting any result:</b> with half-month (often just 5-10 row) fitting windows, the moment-matching method's std estimate can land very close to zero by pure sampling chance, and the original numerical floor was far too small relative to return-scale data -- confirmed directly: MAPEs of 43 million percent (AAPL) and infinite (GLD) came out of the first run. Fixed by capping the rescale ratio to a sane range (no legitimate correction needs more than roughly 5x rescaling) rather than only flooring the denominator.</p>
      <p class="callout">Honest result after the fix: <b>{biweekly_n_improved['mape_moment_continuous']}/22</b> improved with moment-matching (continuous), <b>{biweekly_n_improved['mape_moment_firstraw']}/22</b> with moment-matching (first-half-raw), <b>{biweekly_n_improved['mape_quantile_continuous']}/22</b> with quantile-mapping (continuous), <b>{biweekly_n_improved['mape_quantile_firstraw']}/22</b> with quantile-mapping (first-half-raw) -- GLD is the one improvement, and only under quantile-mapping. Quantile mapping is consistently far less damaging than moment matching across the board (order-statistic mapping is inherently more robust to a handful of noisy points than a mean/std rescale). The deciding factor is sample size: a two-week fitting window (as few as 5-10 resolved pairs) is simply too small to estimate a stable correction for almost every instrument -- consistent with, and a sharper version of, the earlier finding that the ~1-year rolling window needed real data to mature before it helped even JPM and GLD.</p>
    </div>
    <div class="summary-frame">{biweekly_heatmap_img}</div>

    <div class="page-head" style="margin-top:56px; padding-top:24px; border-top:2px solid var(--border);">
      <h1>Rolling / adaptive post-processing -- fourth design, the most informative one</h1>
      <p>Given three static (fit-once-on-distant-data, freeze-forever) designs all failed for the same reason -- the correction needed drifts over time -- the natural next step: keep updating it. Genuine walk-forward, no look-ahead: refit every 5 trading days using only a trailing window of already-RESOLVED (prediction, actual) pairs -- a prediction made for a 252-day horizon doesn't have a known outcome until 252 days later, so the fitting window can only reach back that far before it runs dry. (Caught and fixed a real bug here first: the original window size didn't account for this, so every 252-day-horizon instrument silently fell back to zero corrections, 100% of the time -- confirmed and fixed before trusting any result.)</p>
      <p class="callout">Whole-period result: still mostly negative, {n_rolling_improved} of 22 improved. But splitting the test period in half reveals something the aggregate hides: early on (2022-2024) the rolling window is still mostly stale, pre-test data and performs worse than raw for nearly everyone -- expected, a genuine "burn-in" period. Late (2024+), once the window is genuinely fresh, <b>JPM and GLD -- the two most credible, statistically real relationships in this entire program -- both show real, out-of-sample improvement</b> (JPM: 13.4%&rarr;9.6% MAPE; GLD: 20.2%&rarr;12.0% MAPE). Every other instrument stays worse in both halves. The pattern makes sense: adaptive recalibration can only track a REAL, drifting bias -- for instruments where the underlying relationship was mostly noise to begin with, refitting more often just chases that noise and makes things worse; for JPM and GLD, where a real (if time-varying) structural relationship exists, tracking it adaptively pays off.</p>
    </div>
    <div class="summary-frame">{rolling_summary_img}</div>
    <div class="summary-frame" style="margin-top:16px;">{rolling_earlylate_img}</div>
    <div class="summary-frame" style="margin-top:16px;">{rolling_jpm_img}</div>
    <div class="summary-frame" style="margin-top:16px;">{rolling_gld_img}</div>

    <div class="page-head" style="margin-top:56px; padding-top:24px; border-top:2px solid var(--border);">
      <h1>Per-candidate post-processing (no combining) -- third design, still negative</h1>
      <p>User's further correction: don't combine the four candidates via any regression at all. Instead, post-process EACH of the four (climatology, credit_only, vix_only, both) independently -- moment/PDF matching, then quantile mapping with frequency correction, exactly the same two-stage recipe, fit entirely within the selection period, applied once to test -- so there are simply four separately-corrected predictions to compare against actual prices in the pure test period. No collinearity risk this time (each candidate is corrected using only its own series vs actual, never regressed against the others).</p>
      <p class="callout">Still broadly negative: only <b>{n_percand_cells_improved} of {n_percand_total_cells}</b> individual (instrument, candidate) corrections beat their own raw MAPE, and only <b>{n_percand_any_beats_old}</b> instruments have even one post-processed candidate beating the existing single-winner MAPE. The heatmap below is overwhelmingly red. GLD is the one clear exception -- all four of its candidates improved -- and JPM's credit_only candidate also improved (15.4%&rarr;13.0%, a smaller gain than the earlier single-model version found with a different correction technique). <b>Three independent designs now (single-model+recency-split, stacked regression, per-candidate correction) all point to the same root cause: it's not about how the correction is structured (blended, single, or per-candidate) -- it's that fitting on the distant selection period and freezing the result doesn't survive the trip to a much-later, different-regime test period.</b> The one design that worked at all (JPM, single-model post-processing) calibrated on RECENT data immediately preceding its test window, not on distant selection-period data -- recency, not correction technique, appears to be the deciding factor.</p>
    </div>
    <div class="summary-frame">{percand_heatmap_img}</div>

    <div class="page-head" style="margin-top:56px; padding-top:24px; border-top:2px solid var(--border);">
      <h1>Stacked regression + post-processing, fit entirely on the selection period</h1>
      <p>The user's corrected design, distinct from the version below: instead of picking one winning candidate (climatology / credit_only / vix_only / both) and post-processing it against a further split of the holdout, this fits EVERYTHING -- a stacking regression that blends all four candidates into one output, then sequential post-processing (moment/PDF matching, then quantile mapping with frequency correction) on top -- entirely within the selection period (pre-2022, already used elsewhere to choose horizons). The test period (2022 onward) is used exactly once, with no further splitting, to evaluate the fully-frozen pipeline.</p>
      <p><b>Two real bugs found and fixed before trusting any result, both disclosed:</b> (1) the four candidates are strongly collinear ("both" is literally built from the same ingredients as credit_only and vix_only), so plain OLS assigned huge, unstable, opposite-signed weights (up to &plusmn;3.8) that fit in-sample and blew up out-of-sample -- fixed with ridge (L2-regularized) regression, still a genuine regression/curve-fit, with the penalty strength chosen via an internal split inside the selection period. (2) the quantile-mapping stage's tail extrapolation was unbounded, and a handful of out-of-range test values blew up to nonsensical corrections (one instrument's MAPE hit 193%) -- fixed with a more robust, bounded extrapolation.</p>
      <p class="callout">Even after both fixes, the honest result is negative: <b>only {n_stacked_improved} of 22</b> instruments improved on the existing single-winner master model's test-period MAPE; the other 21 got worse, several dramatically. The likely cause, consistent with the JPM bias-stability finding elsewhere on this page: fitting a static blend and correction on a long, distant selection period (spanning years to decades for some instruments) and freezing it for application to a much later, different-regime test period assumes the *relationship itself* -- which combination of models works best, what correction is needed -- is stable over that whole gap. It isn't. This result directly supports using recent, adjacent-in-time data for any calibration step, over distant historical data, even though the distant-data approach preserves more of the final test period for evaluation.</p>
    </div>
    <div class="summary-frame">{stacked_summary_img}</div>

    <div class="page-head" style="margin-top:56px; padding-top:24px; border-top:2px solid var(--border);">
      <h1>Earlier post-processing (single-winner + recency-based split)</h1>
      <p>Kept for comparison -- this is the version described further below, which picks one winning candidate first (as everywhere else in this project) and post-processes it using a further split inside the holdout, calibrating on the period immediately before the one it's tested on.</p>
    </div>
      <h1>Post-processing / bias correction</h1>
      <p>A different question from everything above: not "how should we trade the forecast" but "can the forecast itself be made more accurate," using standard model-output-statistics (MOS) techniques from numerical weather prediction, applied to the raw (predicted, actual) pairs already on hand. The existing holdout (&ge;2022-01-01) is split again, chronologically: a <b>calibration period</b> (2022-01-01 to 2024-01-01) to fit each correction, and a <b>verification period</b> (2024-01-01 onward) the correction never sees, used only once to report the result &mdash; the same discipline as every selection/holdout split in this project. Three methods: (1) OLS linear recalibration (actual &asymp; a + b&times;raw), (2) moment/PDF matching (rescale to match the observed distribution's mean and std, no regression-toward-the-mean attenuation), (3) empirical quantile mapping with linear tail extrapolation.</p>
      <p><b>Important correction made mid-analysis:</b> picking whichever of the 3 methods had the lowest error <i>on the verification set itself</i> is exactly the selection-bias mistake already caught once this session (choosing a config on the same data used to report its performance). Fixed: the method is chosen via a further internal split <i>inside</i> the calibration period only (train/validate), then the winning method is refit on the full calibration period and applied to verification exactly once. This changed the headline number substantially &mdash; the naive (uncorrected) version of this analysis showed 11/22 instruments improving, including a dramatic-looking GLD result (20.2% &rarr; 8.1% MAPE); the honest version shows only <b>{n_postproc_improved} of 22</b>. GLD's apparent win didn't survive: its OLS slope came out at -1.91 (the calibration fit concluded raw predictions were <i>inversely</i> related to outcomes and told the correction to flip sign and amplify it) &mdash; the signature of an unstable fit on a weak relationship, and it made verification-period MAPE worse (28.4%), not better.</p>
      <p class="callout"><b>JPM is the one credible case.</b> Its OLS correction (a=+0.14, b=1.32 &mdash; a sensible amplification of a systematically under-scaled raw signal, not a sign-flip) was chosen by the internal calibration split and cut verification-period MAPE from 13.4% to 7.7%, entirely out-of-sample. The plot below shows why: the raw model persistently under-predicted JPM's price throughout the verification period (error consistently -10% to -25%), and the correction fixes that systematic bias, oscillating around zero instead. This is the strongest single piece of evidence in the whole program that a real, fixable, non-tautological bias exists somewhere in this pipeline.</p>
    </div>
    <div class="summary-frame">{postproc_summary_img}</div>
    <div class="summary-frame" style="margin-top:16px;">{postproc_jpm_img}</div>

    <div class="page-head" style="margin-top:40px;">
      <h1>Does JPM's correction produce real alpha?</h1>
      <p>Rebuilt the trading strategy using the corrected quantiles instead of raw, restricted to the verification period only (2024-01-01 onward, identical dates for both, so the comparison isn't confounded by sample period). Result: <b>identical</b> P&amp;L and alpha for raw and corrected, twice in a row, for two different reasons found by checking rather than assuming. First, the buy-low/sell-high entry logic: JPM's raw q0.25 predicted return was positive on 100% of days in this window, so the "wait for a dip" trigger fires almost immediately for both raw and corrected -- the strategy degenerates to plain buy-and-hold regardless of the correction. Second, tried Kelly sizing instead (sensitive to magnitude, not just sign) -- still identical, because the raw signal's implied position size already saturates the 2.0x leverage cap (avg size 1.99x for both); the correction roughly doubles the predicted magnitude, but doubling an already-capped number doesn't change the capped output. Uncapped (a pure diagnostic, not a real tradeable rule -- no sane risk limit would allow this): raw Kelly fraction averages 21x, corrected averages 29x -- both absurd, confirming the raw signal was already maximally confident before any correction.</p>
      <p class="callout">The honest conclusion: this MAPE improvement is real and out-of-sample-validated, but it is invisible to every trading rule built in this program, for a structural reason, not a data problem -- it corrects the <i>magnitude</i> of an already-maximally-confident, always-bullish signal, and every sane (risk-limited) way of turning a forecast into a position is blind to magnitude once a signal is already saturating the entry logic and the leverage cap. A real accuracy fix does not automatically mean there was any economic value left on the table to capture.</p>
    </div>
    <div class="summary-frame">{postproc_jpm_alpha_img}</div>

    <div class="page-head" style="margin-top:40px;">
      <h1>Would fitting the correction on the selection period have worked better?</h1>
      <p>The obvious alternative design: fit the correction using the pre-2022 selection period (already untouched by the final report) instead of carving the holdout into calibration/verification -- preserving the full holdout for the final test. Checked directly rather than argued in the abstract: does the SAME correction (fit on 2022-2024) also reduce error if applied backward to the selection period? It does not -- selection-period MAE goes from 0.238 (raw) to 0.279 (corrected), i.e. worse.</p>
      <p class="callout">The plot below shows why: JPM's forecast bias is not a stable property of the model -- it swings from about -0.5 to +0.8 (log-return units) across different multi-year regimes (the COVID crash, the recovery, the 2022+ tightening cycle). A correction fit on distant selection-period data would have captured whichever bias happened to exist back then -- quite possibly the opposite sign of what needed fixing in 2022-2024. The recency-based split (calibrating on the period immediately preceding verification, not old selection data) is what let this correction work at all, and the same logic implies any deployed correction would need periodic refitting, not a one-time fit -- standard practice in real meteorological MOS systems, for the same reason.</p>
    </div>
    <div class="summary-frame">{postproc_jpm_bias_img}</div>

    <div class="page-head" id="relval-top" style="margin-top:56px; padding-top:24px; border-top:2px solid var(--border);">
      <h1>Cross-sectional relative-value long/short</h1>
      <p>A genuinely different mechanism from everything above, all of which was long-vs-cash on one instrument at a time. This asks a relative question instead: is the master model more bullish on instrument A than instrument B <i>right now</i>, regardless of the broad market? Each rebalance, all {relval_stats['n_instruments_universe']} instruments are ranked by an annualized, risk-adjusted score &mdash; (predicted median return / predicted dispersion) &times; &radic;(252/horizon), correcting for the fact that different instruments use different winning horizons (1&ndash;252 days) and so aren't comparable on raw predicted return alone. Long the top tercile (avg {relval_stats['avg_n_long']:.1f} names), short the bottom tercile (avg {relval_stats['avg_n_short']:.1f} names), equal-weighted, dollar-neutral, rebalanced every 21 trading days (an earlier daily-rebalance version reshuffled the entire book every day and let transaction costs alone swamp the result &mdash; corrected). Same 5bps cost convention as everywhere else; no borrow cost modeled for the short leg (disclosed simplification).</p>
      <p class="callout">Realized beta to SPY: {relval_stats['beta_vs_spy']:+.2f} &mdash; confirms the construction is genuinely close to market-neutral. For a dollar-neutral book the correct null is zero mean return (there's no single buy-and-hold counterfactual), tested with a plain one-sample t-test: net annualized return {relval_stats['annualized_mean_return_pct']:+.2f}%/yr, t={relval_stats['t_stat_vs_zero']:+.2f} &mdash; <b>not significant</b> (gross, before costs: {relval_stats['gross_annualized_mean_return_pct']:+.2f}%/yr, t={relval_stats['gross_t_stat_vs_zero']:+.2f}, also not significant). A genuinely different, market-neutral way of using the same forecasts, and it lands on the same conclusion as every directional version: no statistically demonstrated edge.</p>
    </div>
    <div class="summary-frame">{relval_img}</div>

    <div class="page-head" id="target-price-top" style="margin-top:56px; padding-top:24px; border-top:2px solid var(--border);">
      <h1>Buy low, sell high: a real trading strategy, not just a sign rule</h1>
      <p>The earlier version of this backtest (kept below for comparison) went long/flat purely on the SIGN of the predicted median return &mdash; it never used the fact that the master model outputs a full predicted price distribution, and it never modeled how a trader would actually use it: be prepared to buy once the model signals a low is coming, buy once the actual price genuinely reaches it; be prepared to sell once holding, sell once price reaches the model's predicted high.</p>
      <p>Strategy: each day, the winning config's q0.25/q0.75 predicted H-day-ahead quantiles become resting price targets &mdash; buy_target = price &times; exp(q0.25 return), sell_target = price &times; exp(q0.75 return). While flat, buy once price reaches buy_target; while long, sell once price reaches sell_target. Targets refresh daily with the freshest forecast until filled. Same disclosed 5bps cost per trade, no look-ahead (yesterday's forecast triggers today's fill).</p>
      <p class="callout"><b>{n_target_beat} of {len(target_results)}</b> instruments beat buy-and-hold net of costs this way (up from {n_beat}/{len(tickers)} under the naive sign rule) &mdash; but <b>{n_target_alpha} of {len(target_results)}</b> show alpha statistically distinguishable from zero vs their own buy-and-hold. One disclosed mechanical quirk: for several climatology winners with a mildly bullish median forecast, q0.25 sits above today's price, so the buy trigger fires almost immediately (1 trade total) rather than waiting for a real dip &mdash; those cases are close to plain buy-and-hold with one extra cost charged. Even accounting for that, the headline result matches the naive-rule finding: no instrument demonstrates statistically real timing skill, regardless of how the signal is converted into trades.</p>
    </div>
    <div class="summary-frame">{target_summary_img}</div>

    <div class="page-head" id="portfolio-top" style="margin-top:40px;">
      <h1>Multi-instrument portfolio</h1>
      <p>Combining the target-price strategy across all {portfolio_stats['n_instruments']} evaluable instruments, each contributing its own master-model-selected winner &mdash; climatology included, not filtered out. (An earlier version of this backtest restricted the universe to non-climatology winners only, on the theory that a climatology win meant "no real skill" there; that contradicted this project's own master-model framing, where climatology winning means seasonal patterns are genuinely the best available predictor for that instrument, not that there's nothing to trade. Corrected.) Equal-weight, daily-rebalanced combination of each instrument's individual strategy, benchmarked against an equal-weight buy-and-hold of the same basket (the asset-matched benchmark) and SPY (context only).</p>
      <p class="callout">Portfolio net total return {portfolio_stats['portfolio_net_total_return_pct']:+.1f}% vs basket buy&amp;hold {portfolio_stats['basket_buy_hold_total_return_pct']:+.1f}% &mdash; {'beats' if portfolio_stats['beats_basket_buy_hold'] else 'does <b>not</b> beat'} its own basket. Alpha vs basket buy&amp;hold: {portfolio_stats['alpha_vs_basket_pct']:+.2f}%/yr (t={portfolio_stats['t_vs_basket']:+.2f}, not significant). Diversifying across instruments did not turn individually-insignificant edges into a collectively significant one.</p>
    </div>
    <div class="summary-frame">{portfolio_img}</div>
    {target_section_html}

    <div class="page-head" id="kelly-top" style="margin-top:56px; padding-top:24px; border-top:2px solid var(--border);">
      <h1>Kelly-style position sizing: does conviction-weighted exposure help?</h1>
      <p>The target-price strategy above is binary: fully invested or flat. This version sizes exposure by conviction instead &mdash; the Kelly criterion, f* = &mu;/&sigma;&sup2;, using the model's own forecast each day: &mu; = predicted median H-day return (q0.5), &sigma; = predicted dispersion from the interquartile range (q0.75&minus;q0.25)/1.349. Because both &mu; and &sigma;&sup2; scale linearly with the horizon under a random-walk assumption, this ratio is time-scale invariant &mdash; no separate rescaling needed per horizon. Two standard, disclosed practitioner modifications to raw Kelly (well known to be too aggressive under real parameter uncertainty): <b>half-Kelly</b> (0.5&times; multiplier) and a <b>hard cap at 2.0&times;</b> leverage. Long-only, same buy-low/sell-high entry/exit logic as above &mdash; sizing only scales exposure while "in," it does not create a new signal.</p>
      <p class="callout"><b>{n_kelly_beat} of {len(kelly_results)}</b> instruments beat buy-and-hold net of costs (up from {n_target_beat}/{len(target_results)} unsized) &mdash; but still <b>{n_kelly_alpha} of {len(kelly_results)}</b> show significant alpha. Sizing cannot manufacture edge that isn't there: it only reallocates risk given the same underlying (statistically insignificant) signal. Leverage cuts both ways &mdash; several instruments got meaningfully worse (IYR, XLV, AAPL all move further negative) since amplifying a noisy signal amplifies its noise, not just its edge.</p>
    </div>
    <div class="summary-frame">{kelly_summary_img}</div>

    <div class="page-head" id="kelly-portfolio-top" style="margin-top:40px;">
      <h1>Kelly-sized portfolio</h1>
      <p>Same 22-instrument equal-weight portfolio construction as above, using the Kelly-sized per-instrument returns instead of the flat 100%-when-in version.</p>
      <p class="callout">Portfolio net total return {kelly_portfolio_stats['portfolio_net_total_return_pct']:+.1f}% vs basket buy&amp;hold {kelly_portfolio_stats['basket_buy_hold_total_return_pct']:+.1f}% &mdash; {'beats' if kelly_portfolio_stats['beats_basket_buy_hold'] else 'does <b>not</b> beat'} its own basket in raw return. But look at the risk-adjusted picture: Sharpe {kelly_portfolio_stats['portfolio_sharpe']:+.2f} vs buy&amp;hold's {kelly_portfolio_stats['basket_buy_hold_sharpe']:+.2f} (<b>lower</b>), max drawdown {kelly_portfolio_stats['portfolio_max_dd_pct']:.1f}% vs buy&amp;hold's {kelly_portfolio_stats['basket_buy_hold_max_dd_pct']:.1f}% (<b>worse</b>), and alpha vs basket {kelly_portfolio_stats['alpha_vs_basket_pct']:+.2f}%/yr (t={kelly_portfolio_stats['t_vs_basket']:+.2f}, not significant, and actually negative in point estimate). This is leverage doing what leverage does &mdash; amplifying the ride in both directions &mdash; not new skill. Higher raw return with a worse Sharpe and a deeper drawdown is the signature of added risk, not added edge.</p>
    </div>
    <div class="summary-frame">{kelly_portfolio_img}</div>
    {kelly_section_html}

    <div class="page-head" id="baseline-top" style="margin-top:56px; padding-top:24px; border-top:2px solid var(--border);">
      <h1>Earlier baseline (superseded): sign-of-median-return rule</h1>
      <p>Kept for comparison. Each instrument uses its own master-model winner to drive a simple long/flat rule: long when that day's predicted median return is positive, flat otherwise, rebalanced daily on the freshest forecast, one day at a time. Position decided using day t's forecast is applied to day t+1's realized return &mdash; no look-ahead. Net figures include the same disclosed 5bps transaction cost per position change.</p>
      <p class="callout"><b>{n_beat} of {len(tickers)}</b> instruments' strategy beats buy-and-hold net of costs. XLE and JPM are the only cases where both the statistical edge over climatology AND this economic result hold up together.</p>
    </div>
    <div class="summary-frame">{summary_img}</div>

    <div class="page-head" style="margin-top:40px;">
      <h1>Is any of this real alpha?</h1>
      <p>First attempt: a market-model (Jensen's alpha) regression of each strategy's net daily return on SPY's daily return over the holdout period, for every instrument's master-model winner (climatology included). The intercept is alpha; significance is a plain analytic OLS t-test, not a resampling/bootstrap test.</p>
      <p class="callout"><b>{n_alpha_spy} of {len(tickers)}</b> instrument showed alpha distinguishable from zero at 95% this way: <b>{", ".join(TICKER_LABELS.get(t, t) for t in sorted(alpha_spy_tickers))}</b>.</p>
    </div>
    <div class="summary-frame">{alpha_img}</div>

    <div class="page-head" style="margin-top:40px;">
      <h1>Correction: SPY is the wrong benchmark for a non-equity asset</h1>
      <p>GLD's "alpha vs SPY" turned out to be an artifact. Gold is structurally uncorrelated with equities (that's its whole appeal as a diversifier) &mdash; so during a window when gold rallied independent of stocks, <b>simply buying and holding GLD, with zero model, also shows significant alpha vs SPY</b> (+18.6%/yr, t=2.34). The test wasn't measuring the model's skill; it was measuring "gold went up." The properly-specified test of timing skill regresses each strategy against <b>that same instrument's own buy-and-hold return</b>, not SPY &mdash; isolating whether the model's timing added anything beyond simply holding the asset throughout.</p>
      <p class="callout"><b>{n_alpha_own} of {len(tickers)}</b> instruments show significant alpha this way. Under the appropriate benchmark, GLD's alpha drops to +2.1%/yr (t=0.61, not significant) &mdash; its timing added nothing measurable beyond holding gold outright. <b>No instrument in the panel shows a statistically demonstrated edge from the model's own timing decisions</b>, once tested against the correct counterfactual.</p>
    </div>
    <div class="summary-frame">{alpha_own_img}</div>

    {section_html}
  </main>
</div>
"""

with open(OUT_PATH, "w") as f:
    f.write(html)

print(f"Wrote {OUT_PATH}, {os.path.getsize(OUT_PATH)/1024/1024:.2f} MB, {len(tickers)} instruments, {n_beat} beat buy&hold")
