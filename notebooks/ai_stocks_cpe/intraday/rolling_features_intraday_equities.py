"""
rolling_features_intraday_equities.py
========================================
Equities counterpart to predictor_v1/01b_rolling_features_intraday.py --
identical rolling multifractal/structure-function feature math (same
functions, copied unchanged), applied to real NVDA/AMD/SPY 1-minute SIP
bars (regular trading session only) instead of 24/7 crypto bars.

This is the second of two intraday-feasibility methods already run on
crypto (both negative): the first (intraday_cpe_feasibility_test.py, this
directory) found zero of 41,472 naive-CPE configurations cleared the bar,
an even more decisive miss than crypto's. This script tests whether
higher-order multifractal features fare any better for equities than they
did for crypto (also negative there: R^2 < 0, ~49-50% directional
accuracy across 5 model families and 3 horizons).

LOOKBACK=4096 bars is kept the same as the crypto version, but since this
is a regular-session-only series (~390 bars/trading day vs. 1,440/day for
24/7 crypto), 4096 bars now spans ~10.5 trading days rather than ~2.8
calendar days -- a deliberate choice to keep the lookback measured in
real trading observations (consistent with how a live system would see
the data) rather than recomputing crypto's exact calendar-time window.

Run: python rolling_features_intraday_equities.py
Requires: intraday_{nvda,amd,spy}_1m.parquet (already fetched)
Output: features_intraday_equities_panel.parquet
"""
import pandas as pd
import numpy as np
import time
from math import comb

SYMBOLS = {"NVDA": "intraday_nvda_1m.parquet", "AMD": "intraday_amd_1m.parquet", "SPY": "intraday_spy_1m.parquet"}

LAGS = [5, 15, 60]
QS = [2.0, 4.0]
FIT_TAUS = [1, 2, 4, 8, 16, 32, 64, 128]
LOOKBACK = 4096
STRIDE = 60

print("=" * 60)
print("  ROLLING MULTIFRACTAL FEATURE ENGINE (intraday equities)")
print("=" * 60)


def load_regular_session(fname: str) -> pd.Series:
    df = pd.read_parquet(fname)
    et = df.index.tz_convert("US/Eastern")
    mask = (et.time >= pd.Timestamp("09:30").time()) & (et.time <= pd.Timestamp("16:00").time())
    close = df["close"][mask].astype(float)
    close.index = close.index.tz_localize(None)
    return close.sort_index()


# ---- identical math to predictor_v1/01b_rolling_features_intraday.py ----
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
        C += ((-1) ** (k + 1)) * comb(q, k) * np.mean((f_ttau ** (q - k)) * (f_t ** k))
    return C - D
# ---- end identical math ----


def compute_symbol_features(label, fname):
    series = load_regular_session(fname)
    n = len(series)
    print(f"  {label}: {n} bars, {series.index.min()} .. {series.index.max()}")

    abs_incr = np.concatenate([[np.nan], np.abs(np.diff(series.values))])
    positions = np.arange(LOOKBACK - 1, n, STRIDE)
    rows = []
    for pos in positions:
        window_prices = series.values[pos - LOOKBACK + 1: pos + 1]
        alpha, C1, H = rolling_dtm(window_prices)
        xi2 = rolling_xi(window_prices, 2.0, FIT_TAUS)
        xi4 = rolling_xi(window_prices, 4.0, FIT_TAUS)
        window_abs_incr = abs_incr[max(0, pos - LOOKBACK + 1): pos + 1]
        window_abs_incr = window_abs_incr[~np.isnan(window_abs_incr)]
        feat = {"symbol": label, "ts": series.index[pos], "alpha": alpha, "C1": C1, "H": H,
                "xi_q2": xi2, "xi_q4": xi4}
        for tau in LAGS:
            for q in QS:
                feat[f"gap_tau{tau}_q{int(q)}"] = rolling_gap(window_abs_incr, tau, q)
        rows.append(feat)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    frames = []
    for label, fname in SYMBOLS.items():
        print(f"Computing features for {label}...")
        dfi = compute_symbol_features(label, fname)
        frames.append(dfi)
        print(f"  {label}: {len(dfi)} rows, {time.time()-t0:.1f}s elapsed")

    panel = pd.concat(frames, ignore_index=True)
    out_path = "features_intraday_equities_panel.parquet"
    panel.to_parquet(out_path)
    print(f"\nSaved {len(panel)} rows x {len(panel.columns)} cols to {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")
    print(panel.groupby("symbol").size())
