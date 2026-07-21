"""
Exhaustive validation of the "post-processing transfer as regime-change
detector" idea (see memory), per the user's explicit request to run both
open checks before any paper is written:

  1. A REAL, analytic significance test for whether the observed failure
     clustering (59_predictability_limit_transfer_test.py) is distinguishable
     from chance -- the Wald-Wolfowitz runs test, a classical closed-form
     test for randomness in a binary sequence. NOT shuffle/permutation-based
     (per feedback-no-randomization-testing): given n1 successes and n2
     failures, the expected number of runs and its variance under the null
     of random ordering are known analytically, giving a z-score directly.
     A significantly NEGATIVE z (fewer runs than expected) means values
     cluster into streaks more than chance would produce.

  2. Whether the phenomenon (higher transfer-rate + clustering) generalizes
     beyond GLD/JPM, or is an artifact of testing only the two instruments
     already known to be structurally special (the only ones with ANY
     correctable post-processing signal in the whole 22-instrument panel).

Honest scope limitation, disclosed rather than silently worked around:
Paper 11's predictability-limit analysis (results_correlated_decorrelated.json)
covers a 15-instrument sample, not all 22 predictor_v1 instruments. Only 12
of the 22 master-model instruments have both a decision AND a real,
already-computed Paper 11 predictability limit -- this script tests exactly
those 12, with NO new Paper-11-side computation for the other 10 (that
would be a separate, bigger undertaking, not done here).

Reuses 59_'s exact transfer-test design (log-return MAE, quantile-mapping
only, predictability-limit-sized window AND step) generalized across all
12 available instruments instead of just GLD/JPM.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import postprocess_lib as pl

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTABILITY_JSON = os.path.join(os.path.dirname(OUT_DIR), "predictability_paper", "results_correlated_decorrelated.json")


def load_predictability_limits():
    with open(PREDICTABILITY_JSON) as f:
        return json.load(f)


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
    if len(trail_df) < 8:
        return None
    qs = np.linspace(0, 1, n_q)
    raw_q, act_q = np.quantile(trail_df["raw"].values, qs), np.quantile(trail_df["y_true"].values, qs)
    raw_q_u, idx = np.unique(raw_q, return_index=True)
    act_q_u = act_q[idx]
    return {"raw_q_u": raw_q_u, "act_q_u": act_q_u}


def run_transfer_test(ticker, horizon, winner, oos_all, window, with_dates=False):
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
        corrected = pl.quantile_map_apply(seg["raw"].values, params["raw_q_u"], params["act_q_u"]) \
            if len(params["raw_q_u"]) >= 2 else seg["raw"].values
        corr_mae = float(np.mean(np.abs(corrected - seg["y_true"].values)))
        ok = bool(corr_mae < raw_mae)
        if with_dates:
            results.append({"seg_start": seg["date"].iloc[0], "seg_end": seg["date"].iloc[-1], "transferred_ok": ok})
        else:
            results.append(ok)
        i += window
    return results


def runs_test(sequence):
    """Wald-Wolfowitz runs test for randomness in a binary sequence.
    Returns (n_runs, expected_runs, z_score, p_value_two_sided).
    z << 0 (fewer runs than expected) means values cluster into streaks."""
    seq = np.asarray(sequence, dtype=bool)
    n1 = int(seq.sum())
    n2 = int((~seq).sum())
    n = n1 + n2
    if n1 == 0 or n2 == 0 or n < 2:
        return None
    n_runs = 1 + int(np.sum(seq[1:] != seq[:-1]))
    expected = 1 + (2 * n1 * n2) / n
    variance = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n ** 2 * (n - 1))
    if variance <= 0:
        return {"n_runs": n_runs, "expected_runs": expected, "z": None, "p_value": None, "n1": n1, "n2": n2}
    z = (n_runs - expected) / np.sqrt(variance)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {"n_runs": n_runs, "expected_runs": float(expected), "z": float(z), "p_value": p, "n1": n1, "n2": n2}


if __name__ == "__main__":
    decisions = json.load(open(os.path.join(OUT_DIR, "master_model_final_decision.json")))
    pred_data = load_predictability_limits()
    common = sorted(set(decisions.keys()) & set(pred_data.keys()))
    print(f"Testing {len(common)} instruments with both a master-model decision "
          f"and a real Paper 11 predictability limit (of 22 total; the other 10 "
          f"have no Paper 11 result and are NOT estimated here): {common}\n")

    oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
    summary = {}
    for tkr in common:
        horizon = decisions[tkr]["horizon"]
        winner = decisions[tkr]["price_based_winner"]
        window = pred_data[tkr]["2"]["top5_tradeable"][0][0]
        seq = run_transfer_test(tkr, horizon, winner, oos_all, window)
        if len(seq) < 10:
            print(f"{tkr}: only {len(seq)} segments, too few to test meaningfully -- skipped")
            continue
        pct_ok = 100 * np.mean(seq)
        rt = runs_test(seq)
        summary[tkr] = {
            "winner": winner, "horizon": horizon, "predictability_limit_days": int(window),
            "n_segments": len(seq), "pct_transferred_ok": float(pct_ok),
            "runs_test": rt,
        }
        z_str = f"z={rt['z']:.2f}, p={rt['p_value']:.4f}" if rt and rt["z"] is not None else "n/a"
        clustering = "CLUSTERED (fewer runs than chance)" if (rt and rt["z"] is not None and rt["z"] < -1.96) else \
                     ("not distinguishable from random" if rt and rt["z"] is not None else "n/a")
        print(f"{tkr:10s} winner={winner:12s} window={window:3d}d  n_seg={len(seq):4d}  "
              f"transfer_ok={pct_ok:5.1f}%  runs_test: {z_str}  -> {clustering}")

    with open(os.path.join(OUT_DIR, "61_exhaustive_regime_detector_validation.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: 61_exhaustive_regime_detector_validation.json")
