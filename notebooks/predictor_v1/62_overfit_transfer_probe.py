"""
Independent cross-check of the regime-detector idea, per the user's own
design: deliberately fit an EXTREMELY overfit model (no regularization --
memorizes training-segment idiosyncrasies rather than learning general
structure) on one predictability-limit-sized segment, then test it on the
NEXT segment. If the two segments are genuinely the same regime, an overfit
model should still transfer reasonably; if the regime has genuinely
changed, it should fail sharply -- more sharply than a well-regularized
model would, since regularization is designed to generalize and can mask a
real shift behind uniformly-mediocre performance everywhere.

This is checked against the EXISTING quantile-mapping-based transfer test
(59_predictability_limit_transfer_test.py)'s own segment flags: does the
overfit model's cross-period error come out higher specifically on
segment-pairs the original test already flagged as failed? If yes, that's
real, independent, convergent evidence from a completely different
technique -- the same "independent convergent evidence" pattern this
program already trusts elsewhere (Paper 11 vs CPE, GLD/JPM flagged by two
unrelated methods in Paper 12).

Per the user's explicit instruction, BOTH variants are tested separately:
  (a) features-based: same multifractal + regime-interaction features the
      real model uses (features_daily_panel.parquet + credit/vix regime
      signals), fit with zero regularization.
  (b) pure memorization: ONLY the instrument's own lagged daily returns as
      features, nothing exogenous -- tests whether segment B's raw price
      behavior even resembles what was memorized from segment A, with no
      information about *why*.

Comparison uses the Mann-Whitney U test (real, analytic, non-parametric --
NOT shuffle-based, per feedback-no-randomization-testing) to check whether
overfit cross-period MAE differs between the original test's OK-flagged
and FAILED-flagged segment-pairs.

Extended to all 4 instruments that showed statistically significant
transfer-failure clustering in 61_exhaustive_regime_detector_validation.py
(JPM, AAPL, GLD, XLK -- not just the original GLD/JPM pair), to see whether
the runs-test significance from that step actually holds up under this
independent, different-technique check, or whether (like a plain
significance test can) it doesn't survive a second, unrelated form of
scrutiny. GLD/JPM reuse the already-generated 59_ segment CSVs; AAPL/XLK's
segments are generated fresh here using the identical method (predictability-
limit window + quantile-mapping-only, non-overlapping consecutive blocks).
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeRegressor

import feature_lib as fl
import postprocess_lib as pl

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.dirname(OUT_DIR)
PREDICTABILITY_JSON = os.path.join(NB_DIR, "predictability_paper", "results_correlated_decorrelated.json")

INSTRUMENTS = {
    "GLD": {"horizon": 189, "winner": "vix_only"},
    "JPM": {"horizon": 252, "winner": "credit_only"},
    "AAPL": {"horizon": 252, "winner": "credit_only"},
    "XLK": {"horizon": 189, "winner": "climatology"},
}
N_LAG_RETURNS = 10  # pure-memorization feature count for option (b)


def load_pairs_for_segments(ticker, horizon, winner, oos_all):
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
    return {"raw_q_u": raw_q_u, "act_q_u": act_q[idx]}


def generate_segments(ticker, horizon, winner, oos_all, window):
    """Reproduces 59_'s exact method -- predictability-limit window/step,
    quantile-mapping only -- returning a DataFrame with dates, for
    instruments that don't already have a 59_ CSV on disk."""
    df = load_pairs_for_segments(ticker, horizon, winner, oos_all)
    rows = []
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
        rows.append({"seg_start": seg["date"].iloc[0], "seg_end": seg["date"].iloc[-1],
                      "transferred_ok": bool(corr_mae < raw_mae)})
        i += window
    return pd.DataFrame(rows)


def build_regime_series(prices):
    hyg, lqd = prices["HYG"], prices["LQD"]
    vixm, vixy = prices["VIXM"], prices["VIXY"]
    credit = fl.credit_spread_regime(hyg, lqd).rename("credit_spread_regime")
    vix = fl.vix_term_slope(vixm, vixy).rename("vix_term_slope")
    return pd.concat([credit, vix], axis=1).reset_index().rename(columns={"index": "date"})


def build_option_a_frame(ticker, horizon, winner, panel, price_series, regimes):
    df = panel[panel["ticker"] == ticker].merge(regimes, on="date", how="left").copy()
    df["interact_gap21q4_credit"] = df["gap_tau21_q4_z"] * df["credit_spread_regime"]
    df["interact_xiq4_credit"] = df["xi_q4_z"] * df["credit_spread_regime"]
    df["interact_gap21q4_vix"] = df["gap_tau21_q4_z"] * df["vix_term_slope"]
    df["interact_xiq4_vix"] = df["xi_q4_z"] * df["vix_term_slope"]

    baseline_cols = [c for c in panel.columns if c.endswith("_z")]
    if ticker in fl.ORIG_GROUP_TICKERS:
        baseline_cols = baseline_cols + [c for c in panel.columns if c.startswith("ctx_")] + ["self_ref_score"]
    # This overfit probe is a diagnostic independent of the instrument's own
    # deployed model type -- a climatology winner has no natural "variant" of
    # interaction terms, so use the full set (both credit and vix) rather
    # than picking one arbitrarily.
    variant_key = "both" if winner == "climatology" else winner
    feature_cols = baseline_cols + fl.VARIANT_EXTRA_COLS[variant_key]

    date_pos = {d: i for i, d in enumerate(price_series.index)}
    df["pos"] = df["date"].map(date_pos)
    df = df.dropna(subset=["pos"]).copy()
    df["pos"] = df["pos"].astype(int)

    vals = price_series.values
    n = len(vals)
    has_label = df["pos"].values + horizon < n
    fwd_ret = np.full(len(df), np.nan)
    fwd_ret[has_label] = np.log(vals[df["pos"].values[has_label] + horizon] / vals[df["pos"].values[has_label]])
    df["fwd_ret"] = fwd_ret
    return df[["date", "fwd_ret"] + feature_cols].dropna().sort_values("date").reset_index(drop=True), feature_cols


def build_option_b_frame(horizon, price_series, n_lags=N_LAG_RETURNS):
    rets = np.log(price_series / price_series.shift(1))
    df = pd.DataFrame({"date": price_series.index})
    for lag in range(1, n_lags + 1):
        df[f"lag_ret_{lag}"] = rets.shift(lag - 1).values
    vals = price_series.values
    n = len(vals)
    fwd_ret = np.full(len(df), np.nan)
    valid_idx = np.arange(n - horizon)
    fwd_ret[valid_idx] = np.log(vals[valid_idx + horizon] / vals[valid_idx])
    df["fwd_ret"] = fwd_ret
    feature_cols = [f"lag_ret_{lag}" for lag in range(1, n_lags + 1)]
    return df[["date", "fwd_ret"] + feature_cols].dropna().sort_values("date").reset_index(drop=True), feature_cols


def overfit_cross_period_mae(train_df, test_df, feature_cols):
    if len(train_df) < 4 or len(test_df) < 1:
        return None
    model = DecisionTreeRegressor(max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=0)
    model.fit(train_df[feature_cols].values, train_df["fwd_ret"].values)
    pred = model.predict(test_df[feature_cols].values)
    return float(np.mean(np.abs(pred - test_df["fwd_ret"].values)))


if __name__ == "__main__":
    prices = pd.read_parquet(os.path.join(NB_DIR, "multiasset_prices.parquet"))
    panel = pd.read_parquet(os.path.join(OUT_DIR, "features_daily_panel.parquet"))
    regimes = build_regime_series(prices)

    with open(PREDICTABILITY_JSON) as f:
        pred_data = json.load(f)
    oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))

    all_rows = []
    for tkr, cfg in INSTRUMENTS.items():
        horizon, winner = cfg["horizon"], cfg["winner"]
        series = prices[tkr].dropna()
        existing_csv = os.path.join(OUT_DIR, f"59_predictability_limit_transfer_{tkr}.csv")
        if os.path.exists(existing_csv):
            seg_df = pd.read_csv(existing_csv, parse_dates=["seg_start", "seg_end"])
        else:
            window = pred_data[tkr]["2"]["top5_tradeable"][0][0]
            seg_df = generate_segments(tkr, horizon, winner, oos_all, window)
            print(f"({tkr}: generated {len(seg_df)} fresh segments, window={window}d -- "
                  f"no existing 59_ CSV on disk)")

        frame_a, cols_a = build_option_a_frame(tkr, horizon, winner, panel, series, regimes)
        frame_b, cols_b = build_option_b_frame(horizon, series)

        for i in range(len(seg_df) - 1):
            train_start, train_end = seg_df["seg_start"].iloc[i], seg_df["seg_end"].iloc[i]
            test_start, test_end = seg_df["seg_start"].iloc[i + 1], seg_df["seg_end"].iloc[i + 1]
            flagged_ok = bool(seg_df["transferred_ok"].iloc[i + 1])

            tr_a = frame_a[(frame_a["date"] >= train_start) & (frame_a["date"] <= train_end)]
            te_a = frame_a[(frame_a["date"] >= test_start) & (frame_a["date"] <= test_end)]
            mae_a = overfit_cross_period_mae(tr_a, te_a, cols_a)

            tr_b = frame_b[(frame_b["date"] >= train_start) & (frame_b["date"] <= train_end)]
            te_b = frame_b[(frame_b["date"] >= test_start) & (frame_b["date"] <= test_end)]
            mae_b = overfit_cross_period_mae(tr_b, te_b, cols_b)

            all_rows.append({"ticker": tkr, "seg_idx": i, "test_seg_start": str(test_start.date()),
                              "quantile_map_flagged_ok": flagged_ok,
                              "overfit_mae_features": mae_a, "overfit_mae_priceonly": mae_b})

    results = pd.DataFrame(all_rows)
    results.to_csv(os.path.join(OUT_DIR, "62_overfit_transfer_probe.csv"), index=False)

    print(f"{'Ticker':6s} {'Variant':16s} {'OK n':>5s} {'OK MAE':>8s} {'FAIL n':>6s} {'FAIL MAE':>9s} {'U-test p':>9s}")
    summary = {}
    for tkr in INSTRUMENTS:
        sub = results[results["ticker"] == tkr]
        summary[tkr] = {}
        for col, label in [("overfit_mae_features", "features-based"), ("overfit_mae_priceonly", "price-only")]:
            ok_vals = sub.loc[sub["quantile_map_flagged_ok"], col].dropna()
            fail_vals = sub.loc[~sub["quantile_map_flagged_ok"], col].dropna()
            if len(ok_vals) < 3 or len(fail_vals) < 3:
                print(f"{tkr:6s} {label:16s} insufficient data (ok={len(ok_vals)}, fail={len(fail_vals)})")
                continue
            u_stat, p = stats.mannwhitneyu(fail_vals, ok_vals, alternative="greater")
            summary[tkr][label] = {"ok_n": int(len(ok_vals)), "ok_mean_mae": float(ok_vals.mean()),
                                    "fail_n": int(len(fail_vals)), "fail_mean_mae": float(fail_vals.mean()),
                                    "mannwhitney_p_fail_greater": float(p)}
            print(f"{tkr:6s} {label:16s} {len(ok_vals):5d} {ok_vals.mean():8.4f} {len(fail_vals):6d} "
                  f"{fail_vals.mean():9.4f} {p:9.4f}")

    import json
    with open(os.path.join(OUT_DIR, "62_overfit_transfer_probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: 62_overfit_transfer_probe.csv, 62_overfit_transfer_probe_summary.json")
