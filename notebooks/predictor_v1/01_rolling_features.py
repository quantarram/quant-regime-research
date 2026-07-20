"""
Rolling, point-in-time multifractal / structure-function feature engine.

Generalizes the whole-history, single-instrument math already proven in
predictability_paper/01_dtm_multifractal_analysis.py (TM/DTM: alpha, C1, H)
and predictability_paper/02_structure_function_scan.py (structure-function
exponent xi(q)), plus the correlated/decorrelated structure-function gap
G(tau,q) defined in cpe_paper_multifractal_predictability_draft.md Section 4
(Eq. 8-9), into a rolling trailing-window feature panel across multiple
instruments. Every feature value at date t is computed using only data with
index <= t (no lookahead) -- this is what makes it usable for a live/
walk-forward prediction model rather than a static diagnostic.

Run: python 01_rolling_features.py
Requires: ../multiasset_prices.parquet, ../cpe_results.parquet
Output: features_daily_panel.parquet
"""
import pandas as pd
import numpy as np
import os
import time
from math import comb

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # notebooks/
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

INSTRUMENTS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "AAPL", "MSFT", "JPM", "XOM",
               "GLD", "BTC-USD", "TLT", "EURUSD=X", "^VIX"]
CONTEXT_TICKERS = ["^VIX", "TLT"]          # cross-asset regime features attached to every row
RATE_LIKE = {"^VIX"}                        # level-diff convention, matches build_gold_dashboard.py

LAGS = [1, 5, 21]                           # target horizon grid (trading days)
QS = [2.0, 4.0]                             # moment orders, matches Paper 9
FIT_TAUS = [1, 2, 4, 8, 16, 32, 64]         # lags used to fit the xi(q) slope
LOOKBACK = 512                              # trailing window, ~2 trading years, power of 2
STRIDE = 1                                  # refresh every trading day -- true daily prediction

print("=" * 60)
print("  ROLLING MULTIFRACTAL FEATURE ENGINE (daily)")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
pair = pd.read_parquet(os.path.join(REPO_DIR, "cpe_results.parquet"))


# ── Core math, generalized from predictability_paper/ scripts ──────────────

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
    """TM/DTM analysis on a trailing window of raw price. Returns alpha, C1, H."""
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
    if len(logL) >= 3:
        b, _ = np.polyfit(logL, logT, 1)
        H_est = -b
    else:
        H_est = np.nan

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
    """Correlated/decorrelated structure-function gap G(tau,q) = C(tau,q) - D(tau,q),
    on the raw absolute price-increment field f=|Delta p|, matching Eq. 8-9 of the
    multifractal predictability draft (Section 4.1-4.2)."""
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
        C += ((-1) ** (k + 1)) * comb(q, k) * np.mean((f_ttau ** (q - k)) * (f_t ** k))
    return C - D


# ── Self-referential CPE regime score, made point-in-time (rolling threshold) ──
# Reuses cpe_results.parquet's own pre-gated (X==Y, tau_past, q_X, direction,
# tau_future) combinations -- same discipline build_gold_dashboard.py's
# compute_self_ref_score uses -- but recomputes the firing threshold from a
# trailing window at each date instead of the full history, so it is safe to
# use inside a walk-forward panel. Caveat: which combinations exist in
# cpe_results.parquet was itself selected using the full historical sample,
# so this one feature carries a mild combo-selection look-ahead bias even
# though whether a given combo *fires* on a given day is computed point-in-time.

def get_self_ref_rows(ticker):
    return pair[(pair["X"] == ticker) & (pair["Y"] == ticker)]


def rolling_self_ref_score(ticker, self_ref_rows, ret_series_by_tau, level_series, is_rate_like, pos):
    if len(self_ref_rows) == 0:
        return 0.5
    bull, bear = [], []
    for _, row in self_ref_rows.iterrows():
        tp = int(row["tau_past"])
        qx = float(row["q_X"])
        direction = row["direction"]
        s = ret_series_by_tau.get(tp)
        if s is None:
            continue
        window = s[max(0, pos - LOOKBACK):pos]
        window = window[~np.isnan(window)]
        if len(window) < 100:
            continue
        curr = s[pos] if pos < len(s) else np.nan
        if not np.isfinite(curr):
            continue
        if direction == "bullish":
            th = np.quantile(window, qx)
            fires = curr > th
        else:
            th = np.quantile(window, 1 - qx)
            fires = curr < th
        if fires:
            (bull if direction == "bullish" else bear).append(float(row["CPE"]))
    if bull:
        return float(np.mean(bull))
    elif bear:
        return 1.0 - float(np.mean(bear))
    return 0.5


def build_ret_series_by_tau(price_series, is_rate_like, taus):
    out = {}
    vals = price_series.values
    for tp in taus:
        if is_rate_like:
            r = np.concatenate([[np.nan] * tp, vals[tp:] - vals[:-tp]])
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.concatenate([[np.nan] * tp, np.log(vals[tp:] / vals[:-tp])])
        out[tp] = r
    return out


# ── Per-instrument feature computation ──────────────────────────────────────

def compute_instrument_features(ticker, master_dates):
    series = prices[ticker].dropna()
    series = series[series.index <= prices.index.max()]
    idx = series.index
    n = len(series)
    if n < LOOKBACK + max(LAGS) + 10:
        print(f"  {ticker}: insufficient history ({n} obs), skipping")
        return pd.DataFrame()

    abs_incr = np.concatenate([[np.nan], np.abs(np.diff(series.values))])
    is_rate_like = ticker in RATE_LIKE
    self_ref_rows = get_self_ref_rows(ticker)
    self_ref_taus = sorted(set(int(t) for t in self_ref_rows["tau_past"].unique())) if len(self_ref_rows) else []
    ret_by_tau = build_ret_series_by_tau(series, is_rate_like, self_ref_taus) if self_ref_taus else {}

    rows = []
    idx_pos_lookup = idx.searchsorted(master_dates, side="right") - 1  # last obs <= date
    for d, pos in zip(master_dates, idx_pos_lookup):
        if pos < LOOKBACK - 1 or pos >= n:
            continue
        window_prices = series.values[pos - LOOKBACK + 1: pos + 1]
        alpha, C1, H = rolling_dtm(window_prices)
        xi2 = rolling_xi(window_prices, 2.0, FIT_TAUS)
        xi4 = rolling_xi(window_prices, 4.0, FIT_TAUS)
        window_abs_incr = abs_incr[max(0, pos - LOOKBACK + 1): pos + 1]
        window_abs_incr = window_abs_incr[~np.isnan(window_abs_incr)]

        feat = {"ticker": ticker, "date": d, "alpha": alpha, "C1": C1, "H": H,
                "xi_q2": xi2, "xi_q4": xi4}
        for tau in LAGS:
            for q in QS:
                feat[f"gap_tau{tau}_q{int(q)}"] = rolling_gap(window_abs_incr, tau, q)

        feat["self_ref_score"] = rolling_self_ref_score(
            ticker, self_ref_rows, ret_by_tau, series, is_rate_like, pos
        )
        rows.append(feat)
    return pd.DataFrame(rows)


t0 = time.time()
master_dates = prices.index[::STRIDE]
print(f"Master refresh grid: {len(master_dates)} dates, stride={STRIDE} trading days")

context_frames = {}
for ct in CONTEXT_TICKERS:
    print(f"Computing context features for {ct}...")
    dfc = compute_instrument_features(ct, master_dates)
    gap_cols = [c for c in dfc.columns if c.startswith("gap_")]
    dfc = dfc[["date"] + gap_cols].rename(columns={c: f"ctx_{ct.strip('^')}_{c}" for c in gap_cols})
    context_frames[ct] = dfc
    print(f"  {ct}: {len(dfc)} rows, {time.time()-t0:.1f}s elapsed")

all_frames = []
for tkr in INSTRUMENTS:
    print(f"Computing features for {tkr}...")
    dft = compute_instrument_features(tkr, master_dates)
    if dft.empty:
        continue
    for ct in CONTEXT_TICKERS:
        if tkr == ct:
            continue
        dft = dft.merge(context_frames[ct], on="date", how="left")
    all_frames.append(dft)
    print(f"  {tkr}: {len(dft)} rows, {time.time()-t0:.1f}s elapsed")

panel = pd.concat(all_frames, ignore_index=True)

# Cross-sectional z-score of the raw (non-context) features, per date, so the
# pooled model can share weights sensibly across instruments with different
# price/volatility scales. Context (^VIX/TLT) and self_ref_score are already
# on a comparable scale across instruments and are left as-is.
zscore_cols = ["alpha", "C1", "H", "xi_q2", "xi_q4"] + \
    [f"gap_tau{t}_q{int(q)}" for t in LAGS for q in QS]

def zscore_group(g):
    for c in zscore_cols:
        mu = g[c].mean()
        sd = g[c].std()
        g[c + "_z"] = (g[c] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
    return g

panel = panel.groupby("date", group_keys=False).apply(zscore_group)

out_path = os.path.join(OUT_DIR, "features_daily_panel.parquet")
panel.to_parquet(out_path)
print(f"\nSaved {len(panel)} rows x {len(panel.columns)} cols to {out_path}")
print(f"Total time: {time.time()-t0:.1f}s")
print(panel.groupby("ticker").size())
