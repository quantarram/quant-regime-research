"""
Pure post-processing functions for GLD/JPM, extracted verbatim from
`56_rolling_postprocess.py` (moment-matching + bounded quantile-mapping),
plus a new `get_resolved_pairs` for live use.

Phase 1 has no live prediction log yet, so `get_resolved_pairs` draws only
from the historical backtest pairs already saved in
`oos_predictions_all.parquet` -- confirmed sufficient at cold start (GLD:
2,187 historical rows available, needs a trailing window of 756; JPM: 2,870
available, needs 1,008). Written so a Phase 2 live-log source can later be
concatenated in without changing the calling code in `live_train_predict.py`.
"""
import numpy as np
import pandas as pd

WINDOW_DAYS = 252
MIN_WINDOW_ROWS = 60


def moment_match_fit(raw, actual):
    raw_mean, raw_std = raw.mean(), max(raw.std(), 1e-8)
    act_mean, act_std = actual.mean(), actual.std()
    return raw_mean, raw_std, act_mean, act_std


def quantile_map_apply(raw, raw_q_u, act_q_u, max_extrap_multiple=2.0):
    corrected = np.interp(raw, raw_q_u, act_q_u)
    if len(raw_q_u) >= 4:
        n_tail = max(2, len(raw_q_u) // 5)
        lo_slope = np.polyfit(raw_q_u[:n_tail], act_q_u[:n_tail], 1)[0]
        hi_slope = np.polyfit(raw_q_u[-n_tail:], act_q_u[-n_tail:], 1)[0]
        act_range = act_q_u[-1] - act_q_u[0]
        lo_cap = act_q_u[0] - max_extrap_multiple * act_range
        hi_cap = act_q_u[-1] + max_extrap_multiple * act_range
        below, above = raw < raw_q_u[0], raw > raw_q_u[-1]
        ext_lo = np.clip(act_q_u[0] + (raw - raw_q_u[0]) * lo_slope, lo_cap, act_q_u[0])
        ext_hi = np.clip(act_q_u[-1] + (raw - raw_q_u[-1]) * hi_slope, act_q_u[-1], hi_cap)
        corrected = np.where(below, ext_lo, corrected)
        corrected = np.where(above, ext_hi, corrected)
    return corrected


def get_resolved_pairs(ticker, horizon, winner, oos_all_df, as_of_date, live_log_df=None):
    """Trailing effective_window rows of already-RESOLVED (raw q0.5, actual)
    pairs, using the same no-lookahead 'target_date < as_of_date' filter as
    56_rolling_postprocess.py. live_log_df (Phase 2, not yet built) would be
    concatenated in here alongside the historical oos_all_df rows -- accepted
    as an optional param now so live_train_predict.py's call site doesn't
    need to change later."""
    sub_th = oos_all_df[(oos_all_df["ticker"] == ticker) & (oos_all_df["horizon"] == horizon)]
    variants_present = sub_th["variant"].unique().tolist()
    if winner == "climatology":
        vsrc = sub_th[sub_th["variant"] == ("both" if "both" in variants_present else variants_present[0])]
        pred_col = "clim_q0.5"
    else:
        vsrc = sub_th[sub_th["variant"] == winner]
        pred_col = "q0.5"
    df = vsrc[["date", pred_col, "y_true"]].dropna().rename(columns={pred_col: "raw"}).sort_values("date")

    if live_log_df is not None and len(live_log_df) > 0:
        live_sub = live_log_df[(live_log_df["ticker"] == ticker) & (live_log_df["status"] == "RESOLVED")]
        if len(live_sub) > 0:
            live_pairs = live_sub[["origination_date", "raw_q0.5", "resolved_log_return"]].rename(
                columns={"origination_date": "date", "raw_q0.5": "raw", "resolved_log_return": "y_true"})
            df = pd.concat([df, live_pairs], ignore_index=True).sort_values("date")

    effective_window = max(WINDOW_DAYS, 4 * horizon)
    trail = df[df["date"] < pd.Timestamp(as_of_date)].tail(effective_window)
    return trail


def fit_correction(trail_df):
    """Fits moment-match + quantile-map params from a trailing-pairs
    DataFrame (columns: raw, y_true). Returns None if too few rows (caller
    should fall back to raw, exactly as 56_rolling_postprocess.py does)."""
    if len(trail_df) < MIN_WINDOW_ROWS:
        return None
    raw_mean, raw_std, act_mean, act_std = moment_match_fit(trail_df["raw"].values, trail_df["y_true"].values)
    mm_trail = act_mean + (trail_df["raw"].values - raw_mean) * (act_std / raw_std)
    qs = np.linspace(0, 1, 20)
    raw_q, act_q = np.quantile(mm_trail, qs), np.quantile(trail_df["y_true"].values, qs)
    raw_q_u, idx = np.unique(raw_q, return_index=True)
    act_q_u = act_q[idx]
    return {"raw_mean": raw_mean, "raw_std": raw_std, "act_mean": act_mean, "act_std": act_std,
            "raw_q_u": raw_q_u, "act_q_u": act_q_u, "n_resolved_pairs": len(trail_df)}


def apply_correction(raw_value, params):
    """raw_value: scalar or array of raw log-return quantile(s) to correct."""
    raw_arr = np.atleast_1d(raw_value).astype(float)
    mm = params["act_mean"] + (raw_arr - params["raw_mean"]) * (params["act_std"] / params["raw_std"])
    if len(params["raw_q_u"]) >= 2:
        corrected = quantile_map_apply(mm, params["raw_q_u"], params["act_q_u"])
    else:
        corrected = mm
    return corrected if np.ndim(raw_value) else float(corrected[0])
