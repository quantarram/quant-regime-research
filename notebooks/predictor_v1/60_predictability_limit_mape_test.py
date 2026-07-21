"""
Does a predictability-limit-windowed rolling correction actually beat the
DEPLOYED correction's real MAPE, not just show a higher walk-forward
transfer-success rate (59_predictability_limit_transfer_test.py)? Those are
different claims -- this tests the second one directly.

Mirrors 56_rolling_postprocess.py's EXACT evaluation protocol (same
TEST_START=2022-01-01 holdout, same early/late 2024-01-01 split, same
price-level MAPE reconstruction: price_now * exp(pred_ret) vs actual target
price) so the comparison is apples-to-apples against the numbers already
in final_deployed_pipeline.json / rolling_postprocess_results.json
(GLD: raw_late=20.23%, deployed_late=12.03%; JPM: raw_late=13.37%,
deployed_late=9.57%).

Only two things changed from 56_'s deployed design:
  1. effective_window and REFIT_EVERY both set to each instrument's own
     Paper 11 predictability limit (22d GLD, 23d JPM; q=2 top-tradeable-lag,
     loaded programmatically) instead of max(252, 4*horizon) / a fixed 5-day
     cadence -- matching 59_'s validated "fit and test on the instrument's
     own predictability-limit-sized blocks" design, applied here to the
     REAL deployed evaluation protocol instead of the diagnostic MAE test.
  2. Quantile-mapping ONLY, no moment-matching -- same reasoning as 59_:
     a ~22-23 row window is below postprocess_lib.py's MIN_WINDOW_ROWS=60
     stability floor, and moment-matching is the technique known to
     destabilize at small samples (57_biweekly_postprocess.py's bug).
"""
import json
import os

import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTABILITY_JSON = os.path.join(os.path.dirname(OUT_DIR), "predictability_paper", "results_correlated_decorrelated.json")

TEST_START = pd.Timestamp("2022-01-01")
MID = pd.Timestamp("2024-01-01")
MIN_WINDOW_ROWS = 8  # quantile-map-only floor, matching 59_'s choice

INSTRUMENTS = {
    "GLD": {"horizon": 189, "winner": "vix_only"},
    "JPM": {"horizon": 252, "winner": "credit_only"},
}
PROXY_TICKERS = ("IYR", "VOX")


def load_predictability_limits():
    with open(PREDICTABILITY_JSON) as f:
        d = json.load(f)
    return {t: d[t]["2"]["top5_tradeable"][0][0] for t in INSTRUMENTS}


def quantile_map_apply(raw, raw_q_u, act_q_u, max_extrap_multiple=2.0):
    corrected = np.interp(raw, raw_q_u, act_q_u)
    if len(raw_q_u) >= 4:
        n_tail = max(2, len(raw_q_u) // 5)
        lo_slope = np.polyfit(raw_q_u[:n_tail], act_q_u[:n_tail], 1)[0]
        hi_slope = np.polyfit(raw_q_u[-n_tail:], act_q_u[-n_tail:], 1)[0]
        act_range = act_q_u[-1] - act_q_u[0]
        lo_cap, hi_cap = act_q_u[0] - max_extrap_multiple * act_range, act_q_u[-1] + max_extrap_multiple * act_range
        below, above = raw < raw_q_u[0], raw > raw_q_u[-1]
        ext_lo = np.clip(act_q_u[0] + (raw - raw_q_u[0]) * lo_slope, lo_cap, act_q_u[0])
        ext_hi = np.clip(act_q_u[-1] + (raw - raw_q_u[-1]) * hi_slope, act_q_u[-1], hi_cap)
        corrected = np.where(below, ext_lo, corrected)
        corrected = np.where(above, ext_hi, corrected)
    return corrected


def get_series(tkr, prices, prices_proxy):
    return (prices_proxy[tkr] if tkr in PROXY_TICKERS else prices[tkr]).dropna()


if __name__ == "__main__":
    limits = load_predictability_limits()
    oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
    prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
    prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))

    deployed = json.load(open(os.path.join(OUT_DIR, "final_deployed_pipeline.json")))
    results = {}

    for tkr, cfg in INSTRUMENTS.items():
        horizon, winner = cfg["horizon"], cfg["winner"]
        window = limits[tkr]
        sub_th = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon)]
        vsrc = sub_th[sub_th["variant"] == winner]
        df = vsrc[["date", "q0.5", "y_true"]].dropna().rename(columns={"q0.5": "raw"}).sort_values("date").reset_index(drop=True)

        series = get_series(tkr, prices, prices_proxy)
        series_pos = {d: i for i, d in enumerate(series.index)}
        df = df[df["date"].isin(series_pos)].copy()
        df["date_idx"] = df["date"].map(series_pos)
        target_idx = df["date_idx"] + horizon
        valid = target_idx < len(series)
        df = df[valid.values].copy()
        df["target_date"] = series.index[target_idx[valid].values]

        test_df = df[df["date"] >= TEST_START].reset_index(drop=True)

        corrected = np.full(len(test_df), np.nan)
        n_refits, n_fallback_raw = 0, 0
        for start in range(0, len(test_df), window):
            end = min(start + window, len(test_df))
            checkpoint_date = test_df["date"].iloc[start]
            window_start = checkpoint_date - pd.Timedelta(days=int((window + horizon) * 1.5))
            trail = df[(df["date"] >= window_start) & (df["date"] < checkpoint_date) & (df["target_date"] < checkpoint_date)]
            trail = trail.tail(window)
            raw_chunk = test_df["raw"].values[start:end]
            if len(trail) < MIN_WINDOW_ROWS:
                corrected[start:end] = raw_chunk
                n_fallback_raw += (end - start)
                continue
            qs = np.linspace(0, 1, 10)
            raw_q, act_q = np.quantile(trail["raw"].values, qs), np.quantile(trail["y_true"].values, qs)
            raw_q_u, idx = np.unique(raw_q, return_index=True)
            act_q_u = act_q[idx]
            corrected[start:end] = quantile_map_apply(raw_chunk, raw_q_u, act_q_u) if len(raw_q_u) >= 2 else raw_chunk
            n_refits += 1

        price_now = series.reindex(test_df["date"].values).values
        tgt_price = series.reindex(test_df["target_date"].values).values

        def mape(pred_ret, mask):
            pred_price = price_now[mask] * np.exp(pred_ret[mask])
            return float(np.mean(np.abs(pred_price / tgt_price[mask] - 1)) * 100)

        late_mask = (test_df["date"] >= MID).values
        mape_raw_late = mape(test_df["raw"].values, late_mask)
        mape_predlim_late = mape(corrected, late_mask)

        results[tkr] = {
            "window_days": int(window), "n_test": int(len(test_df)), "n_refits": n_refits,
            "n_fallback_raw_days": n_fallback_raw,
            "mape_raw_late": mape_raw_late,
            "mape_predictability_limit_corrected_late": mape_predlim_late,
            "mape_deployed_late": deployed[tkr]["mape_deployed"],
            "beats_deployed": bool(mape_predlim_late < deployed[tkr]["mape_deployed"]),
        }
        print(f"{tkr} (window={window}d): raw_late={mape_raw_late:.2f}%  "
              f"predictability-limit-corrected_late={mape_predlim_late:.2f}%  "
              f"vs deployed_late={deployed[tkr]['mape_deployed']:.2f}%  "
              f"-> {'BEATS' if results[tkr]['beats_deployed'] else 'does NOT beat'} deployed")

    with open(os.path.join(OUT_DIR, "60_predictability_limit_mape_test.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: 60_predictability_limit_mape_test.json")
