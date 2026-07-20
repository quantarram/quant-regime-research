"""
Longer-history proxies for the two sectors that were underpowered in
23_out_of_sample_replication.py (XLC n=83, XLRE n=835 -- too little data
for a fair read):
  Real Estate:            IYR (iShares US Real Estate, since 2000-06-19,
                           vs XLRE's 2015 start -- 26 years instead of 8)
  Communication Services: VOX (Vanguard Communication Services ETF, since
                           2004-09-29 -- this is the SAME fund that was
                           "Vanguard Telecommunication Services ETF" until
                           the 2018 GICS reclassification, so it's a
                           genuinely continuous history of "the sector now
                           called Communication Services," not a mismatched
                           substitute -- vs XLC's 2018 start, so 22 years
                           instead of 8. IYZ (old, narrow Telecom Services
                           only, never reclassified) was considered and
                           rejected as apples-to-oranges with modern
                           Communication Services.

Same methodology as 23_*.py: real walk-forward OOS, baseline-only vs
baseline+macro_interaction, no bootstrap/resampling.

Run: python 24_sector_proxy_replication.py
Output: results_sector_proxy_replication.json
"""
import pandas as pd
import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import lightgbm as lgb

from loss_functions import extreme_hit_rate

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

TICKERS = {"IYR": "real_estate_proxy", "VOX": "communication_services_proxy"}
LAGS = [1, 5, 21]
QS = [2.0, 4.0]
FIT_TAUS = [1, 2, 4, 8, 16, 32, 64]
LOOKBACK = 512
STRIDE = 1
HORIZON = 21
INITIAL_TRAIN_YEARS = 6
STEP_YEARS = 1
LGB_BASE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)

print("=" * 60)
print("  LONGER-HISTORY SECTOR PROXIES: IYR (real estate), VOX (comm services)")
print("=" * 60)

cache_path = os.path.join(OUT_DIR, "sector_proxy_cache.parquet")
if os.path.exists(cache_path):
    prices_new = pd.read_parquet(cache_path)
    print(f"Loaded cached proxy prices: {prices_new.shape}")
else:
    frames = {}
    for t in TICKERS:
        d = yf.download(t, period="max", progress=False, auto_adjust=True)
        close = d["Close"][t] if isinstance(d.columns, pd.MultiIndex) else d["Close"]
        frames[t] = close
        print(f"  Fetched {t}: {len(close)} rows, {close.index.min().date()} .. {close.index.max().date()}")
    prices_new = pd.DataFrame(frames)
    prices_new.to_parquet(cache_path)
    print(f"Cached to {cache_path}")

date_pos = {d: i for i, d in enumerate(prices_new.index)}


# ── Copied from 01_rolling_features.py / 23_*.py ─────────────────────────
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


def compute_instrument_features(ticker, master_dates):
    series = prices_new[ticker].dropna()
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


t0 = time.time()
master_dates = prices_new.index[::STRIDE]
frames = []
for tkr in TICKERS:
    print(f"Computing baseline features for {tkr}...")
    dft = compute_instrument_features(tkr, master_dates)
    if not dft.empty:
        frames.append(dft)
    print(f"  {tkr}: {len(dft)} rows, {time.time()-t0:.1f}s elapsed")

new_baseline = pd.concat(frames, ignore_index=True)
zscore_cols = ["alpha", "C1", "H", "xi_q2", "xi_q4"] + [f"gap_tau{t}_q{int(q)}" for t in LAGS for q in QS]


def zscore_group(g):
    for c in zscore_cols:
        mu, sd = g[c].mean(), g[c].std()
        g[c + "_z"] = (g[c] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
    return g


new_baseline = new_baseline.groupby("date", group_keys=False).apply(zscore_group)
print(f"\nNew baseline panel: {len(new_baseline)} rows, {time.time()-t0:.1f}s")

macro_existing = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro_existing["date"] = pd.to_datetime(macro_existing["date"])
regimes = macro_existing[["date", "uup_trend_regime", "credit_spread_regime"]].drop_duplicates("date")

new_baseline["date"] = pd.to_datetime(new_baseline["date"])
merged_regime = new_baseline.merge(regimes, on="date", how="left")
merged_regime["interact_gap21q4_uup"] = merged_regime["gap_tau21_q4_z"] * merged_regime["uup_trend_regime"]
merged_regime["interact_gap21q4_credit"] = merged_regime["gap_tau21_q4_z"] * merged_regime["credit_spread_regime"]
merged_regime["interact_xiq4_uup"] = merged_regime["xi_q4_z"] * merged_regime["uup_trend_regime"]
merged_regime["interact_xiq4_credit"] = merged_regime["xi_q4_z"] * merged_regime["credit_spread_regime"]

BASELINE_COLS = [c for c in new_baseline.columns if c.endswith("_z")]
MACRO_COLS = ["uup_trend_regime", "credit_spread_regime", "interact_gap21q4_uup",
              "interact_gap21q4_credit", "interact_xiq4_uup", "interact_xiq4_credit"]

ret_frames = []
for t in TICKERS:
    s = prices_new[t].dropna()
    df = pd.DataFrame({"date": s.index, "ticker": t})
    df["fwd_ret"] = np.log(s.shift(-HORIZON).values / s.values)
    ret_frames.append(df)
ret_df = pd.concat(ret_frames, ignore_index=True)
d_all = merged_regime.merge(ret_df, on=["ticker", "date"], how="left")
GLOBAL_START = pd.Timestamp("2007-12-12")
d_all = d_all[d_all["date"] >= GLOBAL_START].copy()
d_all["pos"] = d_all["date"].map(date_pos)


def run_walkforward(dd, feature_cols):
    dd = dd.dropna(subset=feature_cols + ["fwd_ret"]).copy()
    if len(dd) < 500:
        return None
    min_year, max_year = dd["date"].dt.year.min(), dd["date"].dt.year.max()
    first_test_year = min_year + INITIAL_TRAIN_YEARS
    oos_rows = []
    test_year = first_test_year
    while test_year <= max_year:
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year + STEP_YEARS}-01-01")
        test_mask = (dd["date"] >= test_start) & (dd["date"] < test_end)
        if test_mask.sum() < 20:
            test_year += STEP_YEARS
            continue
        test_start_pos = date_pos.get(test_start, None)
        if test_start_pos is None:
            candidates = [p for dt, p in date_pos.items() if dt >= test_start]
            test_start_pos = min(candidates) if candidates else None
        train_mask = (dd["date"] < test_start) & (dd["pos"] + HORIZON < test_start_pos)
        if train_mask.sum() < 150:
            test_year += STEP_YEARS
            continue
        Xtr, ytr = dd.loc[train_mask, feature_cols], dd.loc[train_mask, "fwd_ret"]
        Xte, yte = dd.loc[test_mask, feature_cols], dd.loc[test_mask, "fwd_ret"]
        m = lgb.LGBMRegressor(**LGB_BASE, objective="regression")
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        rec = dd.loc[test_mask, ["ticker", "date"]].copy()
        rec["y_true"] = yte.values
        rec["y_pred"] = pred
        oos_rows.append(rec)
        test_year += STEP_YEARS
    if not oos_rows:
        return None
    return pd.concat(oos_rows, ignore_index=True)


print("\n--- Per-instrument: baseline-only vs baseline+macro_interaction, real walk-forward OOS ---")
results = {}
for tkr in TICKERS:
    d_tkr = d_all[d_all["ticker"] == tkr]
    oos_base = run_walkforward(d_tkr, BASELINE_COLS)
    oos_macro = run_walkforward(d_tkr, BASELINE_COLS + MACRO_COLS)
    if oos_base is None or oos_macro is None:
        print(f"  {tkr}: insufficient data, skipped")
        continue
    hit_base = extreme_hit_rate(oos_base["y_true"].values, oos_base["y_pred"].values)
    hit_macro = extreme_hit_rate(oos_macro["y_true"].values, oos_macro["y_pred"].values)
    results[tkr] = {"n": int(len(oos_macro)), "hit_baseline": float(hit_base), "hit_macro_interaction": float(hit_macro)}
    print(f"  {tkr} ({TICKERS[tkr]}): n={len(oos_macro)}, baseline_hit={hit_base:.3f}, "
          f"macro_interaction_hit={hit_macro:.3f}, delta={hit_macro-hit_base:+.3f}")

out_path = os.path.join(OUT_DIR, "results_sector_proxy_replication.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved results to {out_path}")
print(f"Total time: {time.time()-t0:.1f}s")
print("\nDone.")
