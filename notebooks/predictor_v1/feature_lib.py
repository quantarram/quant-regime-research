"""
Pure, side-effect-free feature-engineering library for the live predictor
pipeline. Math extracted VERBATIM from `01_rolling_features.py` (multifractal
block, self-referential CPE score) and `38_fss_selection_holdout_split.py`
(credit_spread_regime, vix_term_slope, interaction terms, climatology
quantile-by-day construction) -- those are standalone research scripts that
execute expensive, disk-writing computation at import time and must never be
imported directly into a daily dashboard build. This module contains no
top-level execution and no file I/O; every function takes its inputs
explicitly and returns a value.

Two ticker groups exist and are cross-sectionally z-scored SEPARATELY, never
mixed (established throughout the research session):
  - "orig" group (12 tickers: SPY,QQQ,IWM,XLK,XLF,XLE,AAPL,MSFT,JPM,XOM,GLD,
    EURUSD=X) -- gets ctx_VIX_*/ctx_TLT_* columns and self_ref_score.
  - "new" group (10 tickers: XLI,XLB,XLY,XLP,XLU,XLV,DIA,VTI,IYR,VOX) --
    baseline multifractal z-scores only, no ctx_*, no self_ref_score.
"""
import numpy as np
import pandas as pd
from math import comb

LAGS = [1, 5, 21]
QS = [2.0, 4.0]
FIT_TAUS = [1, 2, 4, 8, 16, 32, 64]
LOOKBACK = 512
ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]

ORIG_GROUP_TICKERS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "AAPL", "MSFT", "JPM", "XOM", "GLD", "EURUSD=X"]
NEW_GROUP_TICKERS = ["XLI", "XLB", "XLY", "XLP", "XLU", "XLV", "DIA", "VTI", "IYR", "VOX"]
CONTEXT_TICKERS = ["^VIX", "TLT"]  # attached to every ticker in the orig group only
RATE_LIKE = {"^VIX"}

ZSCORE_COLS = ["alpha", "C1", "H", "xi_q2", "xi_q4"] + \
    [f"gap_tau{t}_q{int(q)}" for t in LAGS for q in QS]


# ── Multifractal math (verbatim from 01_rolling_features.py) ───────────────

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
    """Correlated/decorrelated structure-function gap G(tau,q) = C(tau,q) - D(tau,q)."""
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


# ── Self-referential CPE regime score (orig group only) ────────────────────

def get_self_ref_rows(pair_df, ticker):
    return pair_df[(pair_df["X"] == ticker) & (pair_df["Y"] == ticker)]


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


def rolling_self_ref_score(self_ref_rows, ret_series_by_tau, pos, lookback=LOOKBACK):
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
        window = s[max(0, pos - lookback):pos]
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


# ── Per-instrument rolling feature computation for a SINGLE date (the most
# recent one) -- generalized from 01_rolling_features.py's per-master-date
# loop body, which iterated over all history; live mode only ever needs the
# latest point-in-time value. ──────────────────────────────────────────────

def compute_instrument_features_latest(ticker, series, self_ref_rows=None):
    """series: pd.Series of prices, index=dates, already sorted ascending.
    self_ref_rows: pre-filtered rows from cpe_results.parquet for this
    ticker (orig group only); pass None/empty for the new-ticker group.
    Returns a dict of raw (pre-z-score) feature values for the LAST date in
    `series`, or None if there isn't enough history yet."""
    series = series.dropna()
    n = len(series)
    if n < LOOKBACK + max(LAGS) + 10:
        return None

    vals = series.values
    pos = n - 1
    window_prices = vals[pos - LOOKBACK + 1: pos + 1]
    alpha, C1, H = rolling_dtm(window_prices)
    xi2 = rolling_xi(window_prices, 2.0, FIT_TAUS)
    xi4 = rolling_xi(window_prices, 4.0, FIT_TAUS)

    abs_incr = np.concatenate([[np.nan], np.abs(np.diff(vals))])
    window_abs_incr = abs_incr[max(0, pos - LOOKBACK + 1): pos + 1]
    window_abs_incr = window_abs_incr[~np.isnan(window_abs_incr)]

    feat = {"date": series.index[-1], "alpha": alpha, "C1": C1, "H": H, "xi_q2": xi2, "xi_q4": xi4}
    for tau in LAGS:
        for q in QS:
            feat[f"gap_tau{tau}_q{int(q)}"] = rolling_gap(window_abs_incr, tau, q)

    if self_ref_rows is not None and len(self_ref_rows) > 0:
        is_rate_like = ticker in RATE_LIKE
        self_ref_taus = sorted(set(int(t) for t in self_ref_rows["tau_past"].unique()))
        ret_by_tau = build_ret_series_by_tau(series, is_rate_like, self_ref_taus)
        feat["self_ref_score"] = rolling_self_ref_score(self_ref_rows, ret_by_tau, pos)
    else:
        feat["self_ref_score"] = 0.5

    return feat


def zscore_group_df(df, cols=ZSCORE_COLS):
    """Batch version of zscore_group for a historical DataFrame with a
    'date' column -- per-date cross-sectional z-score, matching
    38_fss_selection_holdout_split.py's `new_baseline_raw.groupby('date').
    apply(zscore_group)` used to z-score the "new" ticker group's seed
    cache (which, unlike the "orig" group's cache, is saved WITHOUT
    z-scores pre-computed)."""
    def _z(g):
        for c in cols:
            mu, sd = g[c].mean(), g[c].std()
            g[c + "_z"] = (g[c] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
        return g
    return df.groupby("date", group_keys=False).apply(_z)


def zscore_group(feature_rows, cols=ZSCORE_COLS):
    """feature_rows: dict of {ticker: feat_dict} all sharing the same 'as of'
    date -- cross-sectional z-score across this group only (never mixed
    across the orig/new groups, per 01_rolling_features.py's own
    groupby('date') convention)."""
    df = pd.DataFrame.from_dict(feature_rows, orient="index")
    out = {}
    for c in cols:
        mu, sd = df[c].mean(), df[c].std()
        df[c + "_z"] = (df[c] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
    for tkr in df.index:
        out[tkr] = df.loc[tkr].to_dict()
    return out


def ctx_columns(gap_feat_dict, ctx_name):
    """Log-compressed (not z-scored) gap_* columns from ^VIX/TLT's own feature
    dict, attached to every orig-group ticker, matching 01_rolling_features.py."""
    out = {}
    for tau in LAGS:
        for q in QS:
            key = f"gap_tau{tau}_q{int(q)}"
            v = gap_feat_dict.get(key, np.nan)
            out[f"ctx_{ctx_name.strip('^')}_{key}"] = float(np.sign(v) * np.log1p(abs(v))) if np.isfinite(v) else np.nan
    return out


# ── Macro regime features (verbatim from 38_fss_selection_holdout_split.py) ─

def credit_spread_regime(hyg, lqd):
    """Plain rolling(200), NO min_periods -- needs a full 200 trading days.
    This asymmetry vs vix_term_slope is real, existing behavior -- preserve
    it exactly, don't 'fix' it."""
    ratio = (hyg / lqd).dropna()
    return (ratio - ratio.rolling(200).mean()) / ratio.rolling(200).std()


def vix_term_slope(vixm, vixy):
    """rolling(200, min_periods=100) -- deliberately asymmetric vs
    credit_spread_regime's no-min_periods version above."""
    vixy, vixm = vixy.dropna(), vixm.dropna()
    common = vixy.index.intersection(vixm.index)
    raw_ratio = np.log(vixm.reindex(common) / vixy.reindex(common))
    return (raw_ratio - raw_ratio.rolling(200, min_periods=100).mean()) / \
        raw_ratio.rolling(200, min_periods=100).std()


CREDIT_TERMS = ["interact_gap21q4_credit", "interact_xiq4_credit", "credit_spread_regime"]
VIX_TERMS = ["interact_gap21q4_vix", "interact_xiq4_vix", "vix_term_slope"]
VARIANT_EXTRA_COLS = {"credit_only": CREDIT_TERMS, "vix_only": VIX_TERMS, "both": CREDIT_TERMS + VIX_TERMS}


def add_interaction_terms(feat, credit_val, vix_val):
    """feat must already contain gap_tau21_q4_z and xi_q4_z (post z-score)."""
    feat = dict(feat)
    feat["credit_spread_regime"] = credit_val
    feat["vix_term_slope"] = vix_val
    feat["interact_gap21q4_credit"] = feat.get("gap_tau21_q4_z", np.nan) * credit_val
    feat["interact_xiq4_credit"] = feat.get("xi_q4_z", np.nan) * credit_val
    feat["interact_gap21q4_vix"] = feat.get("gap_tau21_q4_z", np.nan) * vix_val
    feat["interact_xiq4_vix"] = feat.get("xi_q4_z", np.nan) * vix_val
    return feat


# ── Climatology (verbatim construction from 38_fss_selection_holdout_split.py,
# generalized to use ALL available history for a "final" live model instead
# of that script's fixed 6-year initial-training slice). ───────────────────

def climatology_quantiles_by_day(fwd_ret_dates, fwd_ret_values, alphas=ALPHAS):
    """fwd_ret_dates/fwd_ret_values: full historical (date, forward-return)
    pairs with a resolved label. Builds a frozen day-of-year -> quantile
    lookup, with the same pooled-fallback for calendar days with zero
    historical observations (e.g. Feb 29) as the original script."""
    df = pd.DataFrame({"date": pd.to_datetime(fwd_ret_dates), "fwd_ret": fwd_ret_values}).dropna()
    df["mmdd"] = list(zip(df["date"].dt.month, df["date"].dt.day))
    pooled_vals = df["fwd_ret"].values
    table = {}
    for m in range(1, 13):
        days_in_month = 29 if m == 2 else (30 if m in (4, 6, 9, 11) else 31)
        for da in range(1, days_in_month + 1):
            vals = df.loc[df["mmdd"] == (m, da), "fwd_ret"].values
            if len(vals) == 0:
                vals = pooled_vals
            table[(m, da)] = {a: float(np.quantile(vals, a)) for a in alphas}
    return table
