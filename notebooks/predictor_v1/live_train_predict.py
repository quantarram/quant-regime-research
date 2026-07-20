"""
Live model training and prediction -- the one genuinely new mode of
operation in this whole codebase. Every existing script only ever does
walk-forward backtesting (train strictly before each historical test fold);
this trains ONE final model per instrument on ALL available history through
today and predicts a single forward-looking forecast.

Per-ticker logic driven entirely by `master_model_final_decision.json`
(read programmatically, never hardcoded):
  - winner == "climatology": no LightGBM. A frozen day-of-year empirical
    quantile table built from all history (feature_lib.climatology_quantiles_by_day).
  - winner in ("credit_only", "vix_only"): 5 independent LGBMRegressor
    quantile models (alpha in [0.1,0.25,0.5,0.75,0.9]), same hyperparameters
    as `38_fss_selection_holdout_split.py`, trained on every historical row
    with a computable forward-return label, predicting on today's single
    labelless feature row.
  - ticker in (GLD, JPM): additionally corrected via postprocess_lib, using
    the SAME correction transform (fit from q0.5 resolved pairs) applied
    uniformly across all 5 quantiles -- a monotonic transform (moment-match
    rescale + quantile map are both order-preserving), so applying it
    uniformly preserves q0.1 < q0.25 < ... < q0.9. This mirrors the uniform-
    quantile-correction approach already used earlier this research program
    (53_jpm_corrected_alpha_retest.py), not a new invention.
  - "both" never wins for any of the 22 tickers (confirmed from the
    decision file), so it's handled but never actually exercised live.

mape_deployed is read directly from `final_deployed_pipeline.json` (the
already-validated backtest number) -- never recomputed live, since a live
run has no held-out future to score against.
"""
import json
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

import feature_lib as fl
import postprocess_lib as pl

warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(OUT_DIR)
ALPHAS = fl.ALPHAS
LGB_BASE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
POST_PROCESSED_TICKERS = ("GLD", "JPM")


def load_decisions():
    with open(os.path.join(OUT_DIR, "master_model_final_decision.json")) as f:
        return json.load(f)


def load_deployed_mape():
    with open(os.path.join(OUT_DIR, "final_deployed_pipeline.json")) as f:
        return json.load(f)


def _get_price_series(ticker, prices, prices_proxy):
    return (prices_proxy[ticker] if ticker in ("IYR", "VOX") else prices[ticker]).dropna()


def _build_training_frame(ticker, horizon, variant, feature_panel, price_series, regimes):
    """feature_panel: the ticker's rows from the live orig/new feature cache
    (already z-scored). regimes: DataFrame with date, credit_spread_regime,
    vix_term_slope. Returns (X_train, y_train, X_today) or None if
    insufficient history."""
    df = feature_panel[feature_panel["ticker"] == ticker].merge(regimes, on="date", how="left")
    df["interact_gap21q4_credit"] = df["gap_tau21_q4_z"] * df["credit_spread_regime"]
    df["interact_xiq4_credit"] = df["xi_q4_z"] * df["credit_spread_regime"]
    df["interact_gap21q4_vix"] = df["gap_tau21_q4_z"] * df["vix_term_slope"]
    df["interact_xiq4_vix"] = df["xi_q4_z"] * df["vix_term_slope"]

    baseline_cols = [c for c in feature_panel.columns if c.endswith("_z")]
    if ticker in fl.ORIG_GROUP_TICKERS:
        baseline_cols = baseline_cols + [c for c in feature_panel.columns if c.startswith("ctx_")] + ["self_ref_score"]
    feature_cols = baseline_cols + fl.VARIANT_EXTRA_COLS[variant]

    date_pos = {d: i for i, d in enumerate(price_series.index)}
    df["pos"] = df["date"].map(date_pos)
    df = df.dropna(subset=["pos"]).copy()
    df["pos"] = df["pos"].astype(int)

    series_vals = price_series.values
    n = len(series_vals)
    fwd_ret = np.full(len(df), np.nan)
    has_label = df["pos"].values + horizon < n
    fwd_ret[has_label] = np.log(series_vals[df["pos"].values[has_label] + horizon] / series_vals[df["pos"].values[has_label]])
    df["fwd_ret"] = fwd_ret

    df = df.sort_values("date").reset_index(drop=True)
    # Real edge case, confirmed directly: FX instruments (EURUSD=X) trade on
    # a different calendar than the US-equity-ETF-derived regime signals
    # (credit_spread_regime/vix_term_slope come from HYG/LQD/VIXM/VIXY) --
    # e.g. a Sunday FX bar exists with no corresponding Friday-close-only
    # regime value yet, leaving literally the last row NaN in the regime
    # columns even though plenty of earlier, fully-valid rows exist. Predict
    # from the latest row where the FULL feature set is actually valid, not
    # unconditionally the instrument's own latest price bar.
    valid_rows = df.dropna(subset=feature_cols)
    if len(valid_rows) == 0:
        return None
    today_row = valid_rows.iloc[[-1]]
    train_df = df.dropna(subset=feature_cols + ["fwd_ret"])
    if len(train_df) < 500:
        return None

    X_train, y_train = train_df[feature_cols], train_df["fwd_ret"]
    X_today = today_row[feature_cols]
    return X_train, y_train, X_today, today_row["date"].iloc[0]


def fit_and_predict_quantiles(ticker, horizon, variant, feature_panel, price_series, regimes):
    built = _build_training_frame(ticker, horizon, variant, feature_panel, price_series, regimes)
    if built is None:
        return None
    X_train, y_train, X_today, as_of_date = built
    preds = {}
    for a in ALPHAS:
        m = lgb.LGBMRegressor(**LGB_BASE, objective="quantile", alpha=a)
        m.fit(X_train, y_train)
        preds[a] = float(m.predict(X_today)[0])
    # Each alpha is an independently-fit model with no constraint that they stay
    # ordered ("quantile crossing") -- monotone rearrangement (Chernozhukov,
    # Fernandez-Val & Galichon, 2010): sort the predicted values and reassign to
    # the (already-ascending) alpha grid. Confirmed occurring for a handful of
    # long-horizon informed models (e.g. XLE@252d) via direct inspection of
    # dashboard output.
    sorted_vals = sorted(preds[a] for a in ALPHAS)
    preds = dict(zip(ALPHAS, sorted_vals))
    return preds, as_of_date


def climatology_predict(ticker, horizon, price_series):
    """Builds the day-of-year table from ALL history (not 38_'s fixed
    6-year slice) and looks up the calendar day approximately `horizon`
    trading days from today (today + horizon*(7/5) calendar days, matching
    the trading-to-calendar-day ratio used for display estimates
    elsewhere)."""
    series = price_series.dropna()
    n = len(series)
    fwd_ret = np.log(series.values[horizon:] / series.values[:-horizon])
    dates = series.index[:-horizon]
    table = fl.climatology_quantiles_by_day(dates, fwd_ret)
    as_of_date = series.index[-1]
    target_date_est = as_of_date + pd.Timedelta(days=int(horizon * 1.4))
    key = (target_date_est.month, target_date_est.day)
    if key not in table:
        key = (target_date_est.month, min(target_date_est.day, 28))
    preds = table[key]
    return preds, as_of_date


def apply_gld_jpm_correction(ticker, horizon, winner, raw_preds, oos_all, as_of_date):
    trail = pl.get_resolved_pairs(ticker, horizon, winner, oos_all, as_of_date)
    params = pl.fit_correction(trail)
    if params is None:
        return raw_preds, {"applied": False, "n_resolved_pairs": len(trail)}
    corrected = {a: pl.apply_correction(v, params) for a, v in raw_preds.items()}
    return corrected, {"applied": True, "n_resolved_pairs": params["n_resolved_pairs"]}


def predict_all(prices, prices_proxy, orig_panel, new_panel, oos_all):
    decisions = load_decisions()
    deployed = load_deployed_mape()

    hyg, lqd = prices["HYG"], prices["LQD"]
    vixm, vixy = prices["VIXM"], prices["VIXY"]
    credit = fl.credit_spread_regime(hyg, lqd).rename("credit_spread_regime")
    vix = fl.vix_term_slope(vixm, vixy).rename("vix_term_slope")
    regimes = pd.concat([credit, vix], axis=1).reset_index().rename(columns={"index": "date"})

    import time as _time
    results = []
    for ticker in sorted(decisions.keys()):
        _tk0 = _time.time()
        horizon = decisions[ticker]["horizon"]
        winner = decisions[ticker]["price_based_winner"]
        price_series = _get_price_series(ticker, prices, prices_proxy)
        price_now = float(price_series.iloc[-1])

        correction_meta = {"applied": False, "n_resolved_pairs": 0}
        if winner == "climatology":
            preds, as_of_date = climatology_predict(ticker, horizon, price_series)
        else:
            panel = orig_panel if ticker in fl.ORIG_GROUP_TICKERS else new_panel
            out = fit_and_predict_quantiles(ticker, horizon, winner, panel, price_series, regimes)
            if out is None:
                print(f"  {ticker}: insufficient data for live model, skipping")
                continue
            preds, as_of_date = out
            if ticker in POST_PROCESSED_TICKERS:
                preds, correction_meta = apply_gld_jpm_correction(ticker, horizon, winner, preds, oos_all, as_of_date)

        target_date_est = as_of_date + pd.Timedelta(days=int(horizon * 1.4))
        row = {
            "ticker": ticker, "winner": winner, "horizon": horizon,
            "as_of_date": str(as_of_date.date()), "target_date_est": str(target_date_est.date()),
            "price_now": price_now,
            "method": deployed.get(ticker, {}).get("method", "raw"),
            "mape_deployed": deployed.get(ticker, {}).get("mape_deployed"),
            "correction_applied": correction_meta["applied"],
            "correction_n_resolved_pairs": correction_meta["n_resolved_pairs"],
        }
        for a in ALPHAS:
            row[f"q{a}"] = preds[a]
            row[f"price_q{a}"] = price_now * float(np.exp(preds[a]))
        results.append(row)
        print(f"    {ticker} ({winner}): {_time.time()-_tk0:.1f}s", flush=True)

    return pd.DataFrame(results)
