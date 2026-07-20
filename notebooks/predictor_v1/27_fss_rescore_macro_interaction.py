"""
FSS re-score of macro_interaction, extended to ALL instruments tested in this
program (13 original panel + 10 OOS-replication + 2 longer-history sector
proxies = 25) and to MORE horizons than the original 21d-only validation,
per explicit user request following the same multi-scale/multi-threshold
treatment already applied to the SPY distribution model (26_fss_rescore.py).

Differences from 26_fss_rescore.py, both deliberate:
  1. macro_interaction is a POINT forecast (LightGBM regression on fwd_ret),
     not a quantile forecast -- FSS here compares a hard exceedance
     indicator (y_pred > threshold) against the same threshold applied to
     y_true, rather than a quantile-interpolated probability.
  2. Thresholds are PERCENTILE-based (of each instrument's own fixed
     reference-period return distribution), not fixed return-magnitude
     categories -- this panel spans BTC-USD to GLD to individual equities,
     wildly different volatility scales, so a flat +-5%/10% cutoff (as used
     for SPY alone) would not mean the same thing across instruments.

Climatology reference: exactly the corrected definition from
26_fss_rescore.py (Wilks; Jolliffe & Stephenson) -- a FIXED empirical
exceedance rate computed once from each instrument's own initial
pre-sample training window, held constant (never re-estimated) for the
entire OOS period, per instrument and per horizon.

Horizons: 1, 5, 21d (matching the original Wave-1/2 grid's HORIZONS) plus
63d (new, longer horizon).
FSS windows (smoothing scale, distinct from prediction horizon): 21, 63, 126d.
Thresholds: instrument's own fixed-reference-period 80th/90th/95th
percentile (upper) and 20th/10th/5th percentile (lower).

Run: python 27_fss_rescore_macro_interaction.py
Output: results_fss_macro_interaction.json
"""
import pandas as pd
import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb

from loss_functions import fractional_skill_score, fss_from_quantiles

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

HORIZONS = [1, 5, 21, 63]
WINDOWS = [21, 63, 126]
UPPER_PCTS = [0.80, 0.90, 0.95]
LOWER_PCTS = [0.20, 0.10, 0.05]
INITIAL_TRAIN_YEARS = 6
STEP_YEARS = 1
LGB_BASE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
GLOBAL_START = pd.Timestamp("2007-12-12")
MIN_OOS_ROWS = 300  # below this, per-ticker/horizon grid is too noisy to trust (matches prior "XLC unusable" precedent)

t0 = time.time()
print("=" * 60)
print("  FSS RE-SCORE: macro_interaction, all instruments x multiple horizons")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))  # IYR, VOX

# ── Part A: original 13-instrument panel, features already cached ──────────
baseline_orig = pd.read_parquet(os.path.join(OUT_DIR, "features_daily_panel.parquet"))
baseline_orig["date"] = pd.to_datetime(baseline_orig["date"])
ORIG_TICKERS = sorted(baseline_orig["ticker"].unique())
z_cols_orig = [c for c in baseline_orig.columns if c.endswith("_z")]
ctx_cols = [c for c in baseline_orig.columns if c.startswith("ctx_")]
for c in ctx_cols:
    baseline_orig[c] = np.sign(baseline_orig[c]) * np.log1p(np.abs(baseline_orig[c]))
ORIG_BASELINE_COLS = z_cols_orig + ctx_cols + ["self_ref_score"]

macro_orig = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro_orig["date"] = pd.to_datetime(macro_orig["date"])
MACRO_COLS = [c for c in macro_orig.columns if c not in ("ticker", "date")]
regimes = macro_orig[["date", "uup_trend_regime", "credit_spread_regime"]].drop_duplicates("date")

print(f"Original panel: {len(ORIG_TICKERS)} tickers, {ORIG_TICKERS}")

# ── Part B: baseline multifractal features for the 12 replication tickers ──
# (copied verbatim from 23_out_of_sample_replication.py / 24_sector_proxy_replication.py)
NEW_TICKERS_MAIN = ["XLI", "XLB", "XLY", "XLP", "XLU", "XLV", "XLC", "XLRE", "DIA", "VTI"]
NEW_TICKERS_PROXY = ["IYR", "VOX"]
NEW_TICKERS = NEW_TICKERS_MAIN + NEW_TICKERS_PROXY
LAGS = [1, 5, 21]
QS = [2.0, 4.0]
FIT_TAUS = [1, 2, 4, 8, 16, 32, 64]
LOOKBACK = 512
STRIDE = 1


def build_pyramid(field):
    pyramid = {}
    cur = field.copy()
    lam = len(cur)
    pyramid[lam] = cur.copy()
    while len(cur) > 1:
        cur = cur.reshape(-1, 2).mean(axis=1)
        lam = len(cur)
        pyramid[lam] = cur.copy()
    return pyramid


def _Kq_of(pyramid, fit_lambdas, q):
    logT, logL = [], []
    for lam in fit_lambdas:
        vals = pyramid[lam]
        mean_q = np.mean(vals ** q)
        mean_1_q = np.mean(vals) ** q
        if mean_q <= 0 or mean_1_q <= 0:
            continue
        TM = mean_q / mean_1_q
        if TM <= 0 or not np.isfinite(TM):
            continue
        logT.append(np.log(TM))
        logL.append(np.log(lam))
    if len(logL) < 3:
        return np.nan
    b, _ = np.polyfit(logL, logT, 1)
    return b


def rolling_dtm(window_prices):
    N = len(window_prices)
    n_levels = int(np.floor(np.log2(N)))
    N_use = 2 ** n_levels
    flux = window_prices[-N_use:]
    if np.any(flux <= 0):
        return np.nan, np.nan, np.nan
    pyramid = build_pyramid(flux)
    fit_lambdas = [l for l in sorted(pyramid.keys()) if l >= 4]
    if len(fit_lambdas) < 3:
        return np.nan, np.nan, np.nan
    q_ref = 2.0
    K_qref = _Kq_of(pyramid, fit_lambdas, q_ref)
    logT, logL = [], []
    for lam in fit_lambdas:
        m1 = np.mean(pyramid[lam])
        if m1 <= 0:
            continue
        logT.append(np.log(m1))
        logL.append(np.log(lam))
    H_est = -np.polyfit(logL, logT, 1)[0] if len(logL) >= 3 else np.nan
    etas = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
    valid = []
    for eta in etas:
        pyr_eta = build_pyramid(flux ** eta)
        k = _Kq_of(pyr_eta, fit_lambdas, q_ref)
        if np.isfinite(k) and k > 0:
            valid.append((eta, k))
    if len(valid) < 3 or not np.isfinite(K_qref):
        return np.nan, np.nan, H_est
    logE = np.log([v[0] for v in valid])
    logK = np.log([v[1] for v in valid])
    alpha_est, _ = np.polyfit(logE, logK, 1)
    if abs(alpha_est - 1.0) > 1e-6:
        C1_est = K_qref * (alpha_est - 1) / (q_ref ** alpha_est - q_ref)
    else:
        C1_est = K_qref / (q_ref * np.log(q_ref))
    return alpha_est, C1_est, H_est


def rolling_xi(window_prices, q, fit_taus):
    s = pd.Series(window_prices)
    logS, logT = [], []
    for tau in fit_taus:
        if tau >= len(s):
            continue
        inc = (s - s.shift(tau)).dropna()
        if len(inc) < 10:
            continue
        Sq = np.mean(np.abs(inc.values) ** q)
        if Sq <= 0 or not np.isfinite(Sq):
            continue
        logS.append(np.log(Sq))
        logT.append(np.log(tau))
    if len(logT) < 3:
        return np.nan
    b, _ = np.polyfit(logT, logS, 1)
    return b


def rolling_gap(abs_incr_window, tau, q):
    f = abs_incr_window
    n = len(f)
    if n <= tau + 10:
        return np.nan
    f_t = f[:-tau]
    f_ttau = f[tau:]
    q = int(q)
    D = np.mean((f_ttau - f_t) ** q)
    C = 0.0
    for k in range(1, q):
        C += ((-1) ** (k + 1)) * __import__("math").comb(q, k) * np.mean((f_ttau ** (q - k)) * (f_t ** k))
    return C - D


def compute_instrument_features(ticker, series, master_dates):
    idx = series.index
    n = len(series)
    if n < LOOKBACK + max(LAGS) + 10:
        print(f"  {ticker}: insufficient history ({n} obs), skipping")
        return pd.DataFrame()
    abs_incr = np.concatenate([[np.nan], np.abs(np.diff(series.values))])
    pair_dates = [d for d in master_dates if d in set(idx)]
    idx_pos = idx.searchsorted(pair_dates, side="right") - 1
    rows = []
    for d, pos in zip(pair_dates, idx_pos):
        if pos < LOOKBACK - 1 or pos >= n:
            continue
        window_prices = series.values[pos - LOOKBACK + 1: pos + 1]
        alpha, C1, H = rolling_dtm(window_prices)
        xi2 = rolling_xi(window_prices, 2.0, FIT_TAUS)
        xi4 = rolling_xi(window_prices, 4.0, FIT_TAUS)
        window_abs_incr = abs_incr[max(0, pos - LOOKBACK + 1): pos + 1]
        window_abs_incr = window_abs_incr[~np.isnan(window_abs_incr)]
        feat = {"ticker": ticker, "date": d, "alpha": alpha, "C1": C1, "H": H, "xi_q2": xi2, "xi_q4": xi4}
        for tau in LAGS:
            for q in QS:
                feat[f"gap_tau{tau}_q{int(q)}"] = rolling_gap(window_abs_incr, tau, q)
        rows.append(feat)
    return pd.DataFrame(rows)


master_dates_main = prices.index[::STRIDE]
master_dates_proxy = prices_proxy.index[::STRIDE]
series_map = {tkr: (prices_proxy[tkr] if tkr in NEW_TICKERS_PROXY else prices[tkr]).dropna() for tkr in NEW_TICKERS}

NEW_TICKER_FEATURE_CACHE = os.path.join(OUT_DIR, "features_new_tickers_baseline_cache.parquet")
if os.path.exists(NEW_TICKER_FEATURE_CACHE):
    frames_raw = pd.read_parquet(NEW_TICKER_FEATURE_CACHE)
    print(f"\nLoaded cached new-ticker baseline features: {len(frames_raw)} rows, "
          f"tickers={sorted(frames_raw['ticker'].unique())}, {time.time()-t0:.1f}s elapsed")
    frames = [frames_raw[frames_raw["ticker"] == t] for t in NEW_TICKERS if t in frames_raw["ticker"].unique()]
else:
    print(f"\nComputing baseline multifractal features for {len(NEW_TICKERS)} replication tickers "
          f"(expensive step, ~1-2min/ticker, cached afterward so this never needs repeating)...")
    frames = []
    for tkr in NEW_TICKERS:
        master_dates = master_dates_proxy if tkr in NEW_TICKERS_PROXY else master_dates_main
        dft = compute_instrument_features(tkr, series_map[tkr], master_dates)
        if not dft.empty:
            frames.append(dft)
        print(f"  {tkr}: {len(dft)} rows, {time.time()-t0:.1f}s elapsed")
    pd.concat(frames, ignore_index=True).to_parquet(NEW_TICKER_FEATURE_CACHE)
    print(f"Cached to {NEW_TICKER_FEATURE_CACHE}")

new_baseline = pd.concat(frames, ignore_index=True)
zscore_cols = ["alpha", "C1", "H", "xi_q2", "xi_q4"] + [f"gap_tau{t}_q{int(q)}" for t in LAGS for q in QS]


def zscore_group(g):
    for c in zscore_cols:
        mu, sd = g[c].mean(), g[c].std()
        g[c + "_z"] = (g[c] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
    return g


new_baseline = new_baseline.groupby("date", group_keys=False).apply(zscore_group)
new_baseline["date"] = pd.to_datetime(new_baseline["date"])
new_baseline = new_baseline.merge(regimes, on="date", how="left")
new_baseline["interact_gap_tau21_q4_uup"] = new_baseline["gap_tau21_q4_z"] * new_baseline["uup_trend_regime"]
new_baseline["interact_gap_tau21_q4_credit"] = new_baseline["gap_tau21_q4_z"] * new_baseline["credit_spread_regime"]
new_baseline["interact_xi_q4_uup"] = new_baseline["xi_q4_z"] * new_baseline["uup_trend_regime"]
new_baseline["interact_xi_q4_credit"] = new_baseline["xi_q4_z"] * new_baseline["credit_spread_regime"]
NEW_BASELINE_COLS = [c for c in new_baseline.columns if c.endswith("_z")]
print(f"\nNew-ticker baseline panel: {len(new_baseline)} rows, {time.time()-t0:.1f}s elapsed")

# ── Part C: unify all 25 tickers with (df, baseline_cols, price series) ────
ticker_registry = {}
for tkr in ORIG_TICKERS:
    ticker_registry[tkr] = {
        "df": baseline_orig[baseline_orig["ticker"] == tkr].merge(macro_orig, on=["ticker", "date"], how="left"),
        "baseline_cols": ORIG_BASELINE_COLS, "macro_cols": MACRO_COLS, "series": prices[tkr].dropna(),
    }
for tkr in NEW_TICKERS:
    ticker_registry[tkr] = {
        "df": new_baseline[new_baseline["ticker"] == tkr],
        "baseline_cols": NEW_BASELINE_COLS, "macro_cols": MACRO_COLS, "series": series_map[tkr],
    }

ALL_TICKERS = sorted(ticker_registry.keys())
print(f"\nAll {len(ALL_TICKERS)} tickers: {ALL_TICKERS}")


ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]


def run_walkforward(dd, feature_cols, horizon, date_pos):
    """LightGBM QUANTILE regression (5 alphas, same grid as the SPY distribution
    model) rather than a single point forecast -- a hard 0/1 threshold
    indicator from a point prediction is a much noisier "predicted fraction"
    than climatology's smooth constant, which unfairly penalizes the model in
    the FSS comparison (caught via smoke test before the full run). Quantile
    forecasts let climatology, baseline and macro_interaction all go through
    the identical fss_from_quantiles machinery used for SPY."""
    dd = dd.dropna(subset=feature_cols + ["fwd_ret"]).copy()
    if len(dd) < 500:
        return None, None
    min_year, max_year = dd["date"].dt.year.min(), dd["date"].dt.year.max()
    first_test_year = min_year + INITIAL_TRAIN_YEARS
    first_test_start = pd.Timestamp(f"{first_test_year}-01-01")
    cands0 = [p for dt, p in date_pos.items() if dt >= first_test_start]
    if not cands0:
        return None, None
    first_test_start_pos = min(cands0)
    initial_train_mask = (dd["date"] < first_test_start) & (dd["pos"] + horizon < first_test_start_pos)
    if initial_train_mask.sum() < 150:
        return None, None
    # True climatology: day-of-year granularity, one bucket per calendar
    # (month, day) from the initial reference period only (never
    # re-estimated), applied to each OOS day's own calendar day -- computed
    # as-is from whatever raw observations exist per bucket (typically ~5-6
    # with INITIAL_TRAIN_YEARS=6), no small-n fallback or correction. The
    # only fallback is for calendar days with ZERO observations across every
    # reference year (market holidays that never appear in a trading price
    # series at all) -- a data-availability necessity, not a distrust of
    # small samples. The pooled version is kept separately, only to define
    # stable/comparable percentile thresholds across the whole panel
    # (thresholds themselves aren't seasonal).
    initial_train = dd.loc[initial_train_mask, ["date", "fwd_ret"]].copy()
    initial_train["mmdd"] = list(zip(initial_train["date"].dt.month, initial_train["date"].dt.day))
    pooled_vals = initial_train["fwd_ret"].values
    climatology_quantiles_pooled = {a: float(np.quantile(pooled_vals, a)) for a in ALPHAS}
    climatology_quantiles_by_day = {}
    for mo in range(1, 13):
        days_in_month = 29 if mo == 2 else (30 if mo in (4, 6, 9, 11) else 31)
        for da in range(1, days_in_month + 1):
            vals = initial_train.loc[initial_train["mmdd"] == (mo, da), "fwd_ret"].values
            if len(vals) == 0:
                vals = pooled_vals
            climatology_quantiles_by_day[(mo, da)] = {a: float(np.quantile(vals, a)) for a in ALPHAS}

    oos_rows = []
    test_year = first_test_year
    while test_year <= max_year:
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year + STEP_YEARS}-01-01")
        test_mask = (dd["date"] >= test_start) & (dd["date"] < test_end)
        if test_mask.sum() < 20:
            test_year += STEP_YEARS
            continue
        cands = [p for dt, p in date_pos.items() if dt >= test_start]
        if not cands:
            test_year += STEP_YEARS
            continue
        test_start_pos = min(cands)
        train_mask = (dd["date"] < test_start) & (dd["pos"] + horizon < test_start_pos)
        if train_mask.sum() < 150:
            test_year += STEP_YEARS
            continue
        Xtr, ytr = dd.loc[train_mask, feature_cols], dd.loc[train_mask, "fwd_ret"]
        Xte, yte = dd.loc[test_mask, feature_cols], dd.loc[test_mask, "fwd_ret"]
        rec = dd.loc[test_mask, ["date"]].copy()
        rec["y_true"] = yte.values
        for a in ALPHAS:
            m = lgb.LGBMRegressor(**LGB_BASE, objective="quantile", alpha=a)
            m.fit(Xtr, ytr)
            rec[f"q{a}"] = m.predict(Xte)
        oos_rows.append(rec)
        test_year += STEP_YEARS
    if not oos_rows:
        return None, None
    oos = pd.concat(oos_rows, ignore_index=True).sort_values("date").reset_index(drop=True)
    return oos, (climatology_quantiles_by_day, climatology_quantiles_pooled)


def fss_grid_for_series(y_true, quantile_preds_or_none, climatology_info, dates=None):
    """Returns {window: {'above'/'below': {pct: fss}}}. quantile_preds_or_none
    is {alpha: array} for a real model, or None for climatology -- which uses
    each row's OWN calendar day's fixed reference quantiles (dates required
    in that case), day-of-year granularity, not one flat number applied
    year-round."""
    climatology_by_day, climatology_pooled = climatology_info
    n = len(y_true)
    if quantile_preds_or_none is None:
        dt_index = pd.DatetimeIndex(dates)
        mmdd = list(zip(dt_index.month, dt_index.day))
        qp = {a: np.array([climatology_by_day[k][a] for k in mmdd]) for a in ALPHAS}
    else:
        qp = quantile_preds_or_none
    alpha_values = [climatology_pooled[a] for a in ALPHAS]
    out = {}
    for w in WINDOWS:
        out[w] = {"above": {}, "below": {}}
        for pct in UPPER_PCTS:
            thr = float(np.interp(pct, ALPHAS, alpha_values))
            out[w]["above"][pct] = fss_from_quantiles(y_true, qp, ALPHAS, thr, direction="above", window=w, min_periods=20)
        for pct in LOWER_PCTS:
            thr = float(np.interp(pct, ALPHAS, alpha_values))
            out[w]["below"][pct] = fss_from_quantiles(y_true, qp, ALPHAS, thr, direction="below", window=w, min_periods=20)
    return out


def grid_average(grid):
    vals = [v for w in WINDOWS for d in ("above", "below") for v in grid[w][d].values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


all_results = {}
for tkr in ALL_TICKERS:
    reg = ticker_registry[tkr]
    all_results[tkr] = {}
    series = reg["series"]
    date_pos_tkr = {d: i for i, d in enumerate(series.index)}
    df_tkr = reg["df"]

    for horizon in HORIZONS:
        fwd_ret = np.log(series.shift(-horizon) / series).rename("fwd_ret")
        d_tkr = df_tkr.merge(fwd_ret, left_on="date", right_index=True, how="left")
        d_tkr = d_tkr[d_tkr["date"] >= GLOBAL_START].copy()
        d_tkr["pos"] = d_tkr["date"].map(date_pos_tkr)

        oos_base, clim_base = run_walkforward(d_tkr, reg["baseline_cols"], horizon, date_pos_tkr)
        oos_macro, clim_macro = run_walkforward(d_tkr, reg["baseline_cols"] + reg["macro_cols"], horizon, date_pos_tkr)
        if oos_base is None or oos_macro is None or len(oos_macro) < MIN_OOS_ROWS:
            continue

        qp_base = {a: oos_base[f"q{a}"].values for a in ALPHAS}
        qp_macro = {a: oos_macro[f"q{a}"].values for a in ALPHAS}
        grid_clim = fss_grid_for_series(oos_macro["y_true"].values, None, clim_macro, dates=oos_macro["date"].values)
        grid_base = fss_grid_for_series(oos_base["y_true"].values, qp_base, clim_base)
        grid_macro = fss_grid_for_series(oos_macro["y_true"].values, qp_macro, clim_macro)

        avg_clim, avg_base, avg_macro = grid_average(grid_clim), grid_average(grid_base), grid_average(grid_macro)
        all_results[tkr][horizon] = {
            "n_oos": int(len(oos_macro)),
            "grid_avg_fss_climatology": avg_clim,
            "grid_avg_fss_baseline": avg_base,
            "grid_avg_fss_macro_interaction": avg_macro,
            "skill_above_climatology_baseline": avg_base - avg_clim,
            "skill_above_climatology_macro": avg_macro - avg_clim,
            "macro_minus_baseline": avg_macro - avg_base,
            # Full per-cell grid (window x direction x threshold), for plotting --
            # not just the scalar average, so FSS-vs-threshold-by-window curves
            # can be drawn per instrument without re-running the walk-forward.
            "grid_climatology": grid_clim,
            "grid_baseline": grid_base,
            "grid_macro_interaction": grid_macro,
        }
        print(f"  {tkr} @ {horizon}d: n={len(oos_macro)}, FSS clim={avg_clim:.4f}, base={avg_base:.4f} "
              f"({avg_base-avg_clim:+.4f}), macro={avg_macro:.4f} ({avg_macro-avg_clim:+.4f}), "
              f"macro-base={avg_macro-avg_base:+.4f}  [{time.time()-t0:.0f}s elapsed]")

out_path = os.path.join(OUT_DIR, "results_fss_macro_interaction.json")
with open(out_path, "w") as f:
    json.dump({"horizons": HORIZONS, "windows": WINDOWS, "upper_pcts": UPPER_PCTS, "lower_pcts": LOWER_PCTS,
               "results": all_results}, f, indent=2, default=float)
print(f"\nSaved results to {out_path}")
print(f"Total time: {time.time()-t0:.1f}s")
print("\nDone.")
