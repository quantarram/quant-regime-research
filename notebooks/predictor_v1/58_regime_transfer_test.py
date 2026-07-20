"""
Quick first-pass exploration of "post-processing as a regime-change
indicator" (see memory: fit a correction on one segment, apply it UN-REFIT
to the next segment, read transfer success/failure as the regime signal).

Reuses postprocess_lib.py's already-deployed fit_correction/apply_correction
verbatim (moment-match + quantile-map, chained) -- no new correction
technique built here, this is purely a different way of reading the same
walk-forward machinery 56_rolling_postprocess.py already runs. GLD and JPM
only, the two instruments with any real correctable signal at all. Uses
FULL available history (2014+, not holdout-only) deliberately, to see
whether known regime events (COVID crash, 2022+ tightening) show up --
this is exploratory diagnostics, not a performance claim, so the usual
selection/holdout discipline doesn't apply here.

Works in log-return space directly (MAE of q0.5 vs y_true) rather than
reconstructing price-level MAPE -- a reasonable proxy for "did the
correction help" since that's the space the correction itself operates in,
and price series aren't needed for it.
"""
import json
import os

import numpy as np
import pandas as pd

import postprocess_lib as pl

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP_DAYS = 5  # matches design 4's rolling-refit cadence

INSTRUMENTS = {
    "GLD": {"horizon": 189, "winner": "vix_only"},
    "JPM": {"horizon": 252, "winner": "credit_only"},
}


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


def run_transfer_test(ticker, horizon, winner, oos_all):
    df = load_pairs(ticker, horizon, winner, oos_all)
    effective_window = max(pl.WINDOW_DAYS, 4 * horizon)
    results = []
    i = effective_window
    while i + STEP_DAYS <= len(df):
        trail = df.iloc[i - effective_window:i]
        params = pl.fit_correction(trail)
        if params is None:
            i += STEP_DAYS
            continue
        seg = df.iloc[i:i + STEP_DAYS]
        raw_mae = float(np.mean(np.abs(seg["raw"].values - seg["y_true"].values)))
        corrected = pl.apply_correction(seg["raw"].values, params)
        corr_mae = float(np.mean(np.abs(corrected - seg["y_true"].values)))
        results.append({
            "seg_start": str(seg["date"].iloc[0].date()),
            "seg_end": str(seg["date"].iloc[-1].date()),
            "raw_mae": raw_mae,
            "corrected_mae": corr_mae,
            "transferred_ok": bool(corr_mae < raw_mae),
            "delta": corr_mae - raw_mae,  # negative = correction helped
        })
        i += STEP_DAYS
    return pd.DataFrame(results)


if __name__ == "__main__":
    oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
    all_results = {}
    for ticker, cfg in INSTRUMENTS.items():
        res = run_transfer_test(ticker, cfg["horizon"], cfg["winner"], oos_all)
        all_results[ticker] = res
        pct_ok = 100 * res["transferred_ok"].mean()
        print(f"{ticker}: {len(res)} segments, {pct_ok:.1f}% transferred OK "
              f"(correction still helped on the next un-refit segment)")
        res.to_csv(os.path.join(OUT_DIR, f"58_regime_transfer_{ticker}.csv"), index=False)

    with open(os.path.join(OUT_DIR, "58_regime_transfer_summary.json"), "w") as f:
        json.dump({t: {"n_segments": len(r), "pct_transferred_ok": float(100 * r["transferred_ok"].mean())}
                   for t, r in all_results.items()}, f, indent=2)
    print("Saved: 58_regime_transfer_{GLD,JPM}.csv, 58_regime_transfer_summary.json")
