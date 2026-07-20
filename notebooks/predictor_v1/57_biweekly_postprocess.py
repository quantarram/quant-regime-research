"""
Bi-weekly continuous post-processing, per the user's explicit, detailed
correction of every earlier post-processing attempt. Exact design:

Model selection (period 2, "like we did earlier"): use
best_config_selection_holdout.json -- the ORIGINAL FSS-skill-based
selection (horizon + variant chosen by selection_skill on the selection
period only), NOT master_model_final_decision.json (which later
overrode the variant choice with MAPE -- a different thing the user is
not asking for here). Climatology is included as a genuine 4th candidate:
if the best informed variant's own selection_skill (skill ABOVE
climatology) is <= 0, climatology itself is the selected winner, since
that's what a non-positive skill-above-climatology means.

Post-processing (period 3, the test period), exact mechanics:
  - Every month is split into two halves: 1st-15th, and 16th-end of month.
  - The SECOND half of every month is always post-processed, using a
    correction fit ONLY from resolved (prediction, actual) pairs whose
    TARGET date (origination + horizon) falls in that SAME month's FIRST
    half -- regardless of how long ago the prediction was originally made.
  - The FIRST half of every month has two variations:
      X ("continuous"): still post-processed, using a correction fit from
        the immediately PRECEDING half-month's resolved pairs (i.e. last
        month's second half) -- the correction keeps running uninterrupted.
      Y ("first-half raw"): left completely uncorrected -- only second
        halves ever get post-processed.
  - Two separate correction techniques, tested independently (not chained
    this time, specifically so they can be compared against each other):
      1. PDF/moment matching (2 parameters: rescale to match the fitting
         window's observed mean and std).
      2. Frequency-corrected quantile mapping (empirical order-statistic
         mapping between the fitting window's raw and actual values, with
         bounded tail extrapolation for out-of-range applications).
  - 2 techniques x 2 first-half variations = 4 combinations, run on every
    instrument, plus RAW (no correction at all) for reference.

No look-ahead: a period's fitting window only ever uses pairs that were
ALREADY RESOLVED (target date has passed) by the time that period's own
predictions are made and corrected.

Run: python 57_biweekly_postprocess.py
Output: biweekly_postprocess_results.json, pnl_plots/_BIWEEKLY_heatmap.png
"""
import pandas as pd
import numpy as np
import json
import os
import calendar

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUT_DIR, "pnl_plots")
TEST_START = pd.Timestamp("2022-01-01")
MIN_FIT_ROWS = 5

best_config = json.load(open(os.path.join(OUT_DIR, "best_config_selection_holdout.json")))
oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))
PROXY_TICKERS = ("IYR", "VOX")


def get_series(tkr):
    return (prices_proxy[tkr] if tkr in PROXY_TICKERS else prices[tkr]).dropna()


# --- model selection: FSS-skill-based, climatology included as a genuine candidate ---
decisions = {}
for tkr, cfg in best_config.items():
    winner = cfg["variant"] if cfg["selection_skill"] > 0 else "climatology"
    decisions[tkr] = {"horizon": cfg["horizon"], "winner": winner, "selection_skill": cfg["selection_skill"]}


def half_month_periods(start, end):
    """(period_start, period_end, month_key, half) tuples covering [start, end]."""
    periods = []
    cur = pd.Timestamp(year=start.year, month=start.month, day=1)
    while cur <= end:
        last_day = calendar.monthrange(cur.year, cur.month)[1]
        mid = pd.Timestamp(year=cur.year, month=cur.month, day=15)
        month_end = pd.Timestamp(year=cur.year, month=cur.month, day=last_day)
        month_key = f"{cur.year}-{cur.month:02d}"
        periods.append((cur, mid, month_key, "first"))
        periods.append((mid + pd.Timedelta(days=1), month_end, month_key, "second"))
        cur = month_end + pd.Timedelta(days=1)
    return periods


def moment_match_fit_apply(fit_raw, fit_actual, apply_raw, max_scale_ratio=5.0):
    """With half-month (often ~5-10 row) fitting windows, sample std can be
    very close to zero purely by chance, and 1e-8 is far too small a floor
    relative to return-scale data -- confirmed directly: an unguarded version
    of this function produced MAPEs of 43 MILLION percent (AAPL) and inf
    (GLD) from exactly this. Fixed by capping the rescale ratio (astd/rs) to
    a sane range instead of only flooring the denominator -- no legitimate
    bias correction should need more than a ~5x rescaling; larger ratios are
    the signature of a degenerate small-sample std estimate, not a real
    correction to apply."""
    rm, rs = fit_raw.mean(), fit_raw.std()
    am, astd = fit_actual.mean(), fit_actual.std()
    if rs < 1e-6:
        return apply_raw
    ratio = np.clip(astd / rs, 1.0 / max_scale_ratio, max_scale_ratio)
    return am + (apply_raw - rm) * ratio


def quantile_map_fit_apply(fit_raw, fit_actual, apply_raw, max_extrap_multiple=2.0):
    order = np.argsort(fit_raw)
    raw_q, act_q = fit_raw[order], fit_actual[order]
    raw_q_u, idx = np.unique(raw_q, return_index=True)
    act_q_u = act_q[idx]
    if len(raw_q_u) < 2:
        return apply_raw
    corrected = np.interp(apply_raw, raw_q_u, act_q_u)
    n_tail = max(1, len(raw_q_u) // 4)
    lo_slope = np.polyfit(raw_q_u[:max(2, n_tail)], act_q_u[:max(2, n_tail)], 1)[0] if len(raw_q_u) >= 2 else 0.0
    hi_slope = np.polyfit(raw_q_u[-max(2, n_tail):], act_q_u[-max(2, n_tail):], 1)[0] if len(raw_q_u) >= 2 else 0.0
    act_range = act_q_u[-1] - act_q_u[0]
    lo_cap, hi_cap = act_q_u[0] - max_extrap_multiple * max(act_range, 1e-6), act_q_u[-1] + max_extrap_multiple * max(act_range, 1e-6)
    below, above = apply_raw < raw_q_u[0], apply_raw > raw_q_u[-1]
    ext_lo = np.clip(act_q_u[0] + (apply_raw - raw_q_u[0]) * lo_slope, lo_cap, act_q_u[0])
    ext_hi = np.clip(act_q_u[-1] + (apply_raw - raw_q_u[-1]) * hi_slope, act_q_u[-1], hi_cap)
    corrected = np.where(below, ext_lo, corrected)
    corrected = np.where(above, ext_hi, corrected)
    return corrected


results = {}
for tkr in sorted(decisions.keys()):
    horizon, winner = decisions[tkr]["horizon"], decisions[tkr]["winner"]
    sub_th = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon)]
    variants_present = sub_th["variant"].unique().tolist()
    if winner == "climatology":
        vsrc = sub_th[sub_th["variant"] == ("both" if "both" in variants_present else variants_present[0])]
        pred_col = "clim_q0.5"
    else:
        vsrc = sub_th[sub_th["variant"] == winner]
        pred_col = "q0.5"
    df = vsrc[["date", pred_col, "y_true"]].dropna().rename(columns={pred_col: "raw"}).sort_values("date").reset_index(drop=True)

    series = get_series(tkr)
    series_pos = {d: i for i, d in enumerate(series.index)}
    df = df[df["date"].isin(series_pos)].copy()
    df["date_idx"] = df["date"].map(series_pos)
    tidx = df["date_idx"] + horizon
    valid = tidx < len(series)
    df = df[valid.values].copy()
    df["target_date"] = series.index[tidx[valid].values]
    if len(df) < 200:
        continue

    last_date = df["date"].max()
    periods = half_month_periods(TEST_START - pd.Timedelta(days=75), last_date)

    # precompute resolved-pairs (fitting data) per period, and own-predictions (rows to correct) per period
    resolved_by_period, own_by_period = {}, {}
    for i, (ps, pe, mk, half) in enumerate(periods):
        resolved_by_period[i] = df[(df["target_date"] >= ps) & (df["target_date"] <= pe)]
        own_by_period[i] = df[(df["date"] >= ps) & (df["date"] <= pe)]

    # map month_key -> index of its "first" half period, for same-month second-half lookups
    first_half_idx = {mk: i for i, (ps, pe, mk, half) in enumerate(periods) if half == "first"}

    n_rows = len(df)
    variant_series = {"raw": df["raw"].values.copy()}
    for variation in ["continuous", "firstraw"]:
        for method_name, method_fn in [("moment", moment_match_fit_apply), ("quantile", quantile_map_fit_apply)]:
            out = df["raw"].values.copy().astype(float)
            n_corrected_periods, n_fallback_periods = 0, 0
            for i, (ps, pe, mk, half) in enumerate(periods):
                own = own_by_period[i]
                if len(own) == 0:
                    continue
                if half == "second":
                    src_idx = first_half_idx.get(mk)
                elif variation == "continuous":
                    src_idx = i - 1 if i > 0 else None
                else:  # firstraw: first halves never corrected
                    src_idx = None
                if src_idx is None:
                    continue  # leave as raw (already copied)
                fit = resolved_by_period.get(src_idx)
                if fit is None or len(fit) < MIN_FIT_ROWS:
                    n_fallback_periods += 1
                    continue
                own_mask = df.index.isin(own.index)
                out[own_mask] = method_fn(fit["raw"].values, fit["y_true"].values, df.loc[own_mask, "raw"].values)
                n_corrected_periods += 1
            variant_series[f"{method_name}_{variation}"] = out

    # evaluate on TEST period only
    test_mask = (df["date"] >= TEST_START).values
    price_now = series.reindex(df["date"].values[test_mask]).values
    tgt_price = series.reindex(df["target_date"].values[test_mask]).values

    def mape(pred_ret):
        pred_price = price_now * np.exp(pred_ret[test_mask])
        return float(np.mean(np.abs(pred_price / tgt_price - 1)) * 100)

    entry = {"horizon": horizon, "winner": winner, "n_test": int(test_mask.sum())}
    for k, v in variant_series.items():
        entry[f"mape_{k}"] = mape(v)
    results[tkr] = entry

with open(os.path.join(OUT_DIR, "biweekly_postprocess_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)

cols = ["mape_raw", "mape_moment_continuous", "mape_moment_firstraw", "mape_quantile_continuous", "mape_quantile_firstraw"]
print(f"=== Bi-weekly continuous post-processing, {len(results)} instruments ===")
for tkr in sorted(results, key=lambda t: results[t]["mape_raw"]):
    r = results[tkr]
    print(f"  {tkr} ({r['winner']}@{r['horizon']}d): " + "  ".join(f"{c.replace('mape_','')}={r[c]:.2f}%" for c in cols))

for c in cols[1:]:
    n_imp = sum(1 for r in results.values() if r[c] < r["mape_raw"] - 0.01)
    print(f"{c}: {n_imp}/{len(results)} beat raw")
print("\nDone.")
