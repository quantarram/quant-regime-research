"""
Second pass on "post-processing as a regime-change indicator" (see memory),
per the user's refinement: use each instrument's OWN empirically-measured
Paper 11 predictability limit (top-tradeable-lag from the correlated/
decorrelated structure-function decomposition, q=2 specifically -- Paper 11
itself flags q=4 as noisier) as BOTH the fitting-window size and the
test-segment size, instead of the arbitrary 5-day-step/max(252,4*horizon)
convention from 58_regime_transfer_test.py. Testing transfer onto a segment
further out than an instrument's own known correlated range isn't a
meaningful regime-change test -- decorrelation there is expected anyway.

Predictability limits read directly from
predictability_paper/results_correlated_decorrelated.json (already
computed, no new Paper 11 work): GLD=22 trading days, JPM=23 -- NOT
hardcoded, loaded programmatically below.

Deliberate deviation from postprocess_lib.py's fit_correction: these
windows (22-23 rows) are smaller than MIN_WINDOW_ROWS=60, the floor that
exists because moment-matching's std estimate destabilizes at small
samples (see 57_biweekly_postprocess.py's 43-million-percent MAPE bug).
Rather than lowering that production floor (which would risk the live
GLD/JPM dashboard), this script uses quantile-mapping ONLY, skipping
moment-matching -- quantile-mapping already degraded far more gracefully
at small windows in the bi-weekly design. quantile_map_apply itself is
reused verbatim from postprocess_lib.py.
"""
import json
import os

import numpy as np
import pandas as pd

import postprocess_lib as pl

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTABILITY_JSON = os.path.join(os.path.dirname(OUT_DIR), "predictability_paper", "results_correlated_decorrelated.json")

INSTRUMENTS = {
    "GLD": {"horizon": 189, "winner": "vix_only"},
    "JPM": {"horizon": 252, "winner": "credit_only"},
}


def load_predictability_limits():
    with open(PREDICTABILITY_JSON) as f:
        d = json.load(f)
    return {t: d[t]["2"]["top5_tradeable"][0][0] for t in INSTRUMENTS}  # q=2, top lag


def load_pairs(ticker, horizon, winner, oos_all):
    sub_th = oos_all[(oos_all["ticker"] == ticker) & (oos_all["horizon"] == horizon)]
    variants_present = sub_th["variant"].unique().tolist()
    if winner == "climatology":
        vsrc = sub_th[sub_th["variant"] == ("both" if "both" in variants_present else variants_present[0])]
        pred_col = "clim_q0.5"
    else:
        vsrc = sub_th[sub_th["variant"] == winner]
        pred_col = "q0.5"
    df = vsrc[["date", pred_col, "y_true"]].dropna().rename(columns={pred_col: "raw"}).sort_values("date")
    return df.reset_index(drop=True)


def fit_quantile_map_only(trail_df, n_q=10):
    """Quantile-mapping only, no moment-matching -- small-window variant."""
    if len(trail_df) < 8:  # need at least enough points for a meaningful quantile grid
        return None
    qs = np.linspace(0, 1, n_q)
    raw_q, act_q = np.quantile(trail_df["raw"].values, qs), np.quantile(trail_df["y_true"].values, qs)
    raw_q_u, idx = np.unique(raw_q, return_index=True)
    act_q_u = act_q[idx]
    return {"raw_q_u": raw_q_u, "act_q_u": act_q_u, "n_resolved_pairs": len(trail_df)}


def apply_quantile_map_only(raw_value, params):
    raw_arr = np.atleast_1d(raw_value).astype(float)
    if len(params["raw_q_u"]) >= 2:
        corrected = pl.quantile_map_apply(raw_arr, params["raw_q_u"], params["act_q_u"])
    else:
        corrected = raw_arr
    return corrected


def run_transfer_test(ticker, horizon, winner, oos_all, window):
    df = load_pairs(ticker, horizon, winner, oos_all)
    results = []
    i = window
    while i + window <= len(df):
        trail = df.iloc[i - window:i]
        params = fit_quantile_map_only(trail)
        if params is None:
            i += window
            continue
        seg = df.iloc[i:i + window]
        raw_mae = float(np.mean(np.abs(seg["raw"].values - seg["y_true"].values)))
        corrected = apply_quantile_map_only(seg["raw"].values, params)
        corr_mae = float(np.mean(np.abs(corrected - seg["y_true"].values)))
        results.append({
            "seg_start": str(seg["date"].iloc[0].date()),
            "seg_end": str(seg["date"].iloc[-1].date()),
            "raw_mae": raw_mae,
            "corrected_mae": corr_mae,
            "transferred_ok": bool(corr_mae < raw_mae),
            "delta": corr_mae - raw_mae,
        })
        i += window  # non-overlapping consecutive blocks, not the old 5-day step
    return pd.DataFrame(results)


if __name__ == "__main__":
    limits = load_predictability_limits()
    print(f"Predictability limits (q=2, top tradeable lag): {limits}")
    oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
    all_results = {}
    for ticker, cfg in INSTRUMENTS.items():
        window = limits[ticker]
        res = run_transfer_test(ticker, cfg["horizon"], cfg["winner"], oos_all, window)
        all_results[ticker] = res
        pct_ok = 100 * res["transferred_ok"].mean() if len(res) else float("nan")
        print(f"{ticker}: window={window}d, {len(res)} segments, {pct_ok:.1f}% transferred OK")
        res.to_csv(os.path.join(OUT_DIR, f"59_predictability_limit_transfer_{ticker}.csv"), index=False)

    with open(os.path.join(OUT_DIR, "59_predictability_limit_transfer_summary.json"), "w") as f:
        json.dump({t: {"window_days": limits[t], "n_segments": len(r),
                        "pct_transferred_ok": float(100 * r["transferred_ok"].mean()) if len(r) else None}
                   for t, r in all_results.items()}, f, indent=2)
    print("Saved: 59_predictability_limit_transfer_{GLD,JPM}.csv, 59_predictability_limit_transfer_summary.json")
