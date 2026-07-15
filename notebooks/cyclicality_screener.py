"""
Cyclicality / mean-reversion screener.

Extends the CPE framework's self-referential idea (does X predict X?) with a
spectral lens: for every instrument in the existing 161-ticker universe, test
whether it behaves like a trendless, periodic oscillator rather than a random
walk or a trending asset.

Five independent tests per ticker, all on log price:
  1. Trend strength   - OLS slope of log(price) on time; t-stat / R^2.
                         Low |t-stat| = no significant secular drift.
  2. Hurst exponent    - aggregated-variance method on log returns.
                         H < 0.5 = mean-reverting, H = 0.5 = random walk,
                         H > 0.5 = trending/persistent.
  3. ADF stationarity  - Augmented Dickey-Fuller on the *detrended* log price
                         (residual after removing the linear trend from #1).
                         Reject the unit-root null -> series oscillates
                         around a fixed level rather than wandering.
  4. Dominant cycle    - Lomb-Scargle periodogram on the detrended series;
                         peak period, tested against an AR(1) red-noise null
                         (Monte Carlo) instead of trusting the raw peak.
  5. ACF confirmation  - autocorrelation of the detrended series at the
                         periodogram's implied lag, checked against the
                         standard +-1.96/sqrt(N) significance band. This is
                         the direct time-domain check on the same peak the
                         periodogram found in the frequency domain -- same
                         signal, dual representation (Wiener-Khinchin).

Composite score ranks tickers that are simultaneously trendless, mean-
reverting, stationary, and have a statistically real (not just visually
apparent) dominant cycle -- i.e. candidates for a periodic buy-low/sell-high
approach, as opposed to a trending or pure-noise instrument.

Usage:
    ../.venv/bin/python cyclicality_screener.py
    ../.venv/bin/python cyclicality_screener.py --lookback-years 5
    ../.venv/bin/python cyclicality_screener.py --tickers AAPL,MSFT,GLD
"""

from __future__ import annotations

import argparse
import datetime as dt
import warnings

import numpy as np
import pandas as pd
from scipy.signal import lombscargle
from statsmodels.tsa.stattools import adfuller, acf

warnings.filterwarnings("ignore")

MIN_OBS = 500          # ~2 trading years minimum to test at all
N_MC = 500              # Monte Carlo draws for periodogram significance
MAX_ACF_LAG_DAYS = 756  # cap ACF search at ~3 trading years


# ── DATA LOADING ────────────────────────────────────────────────
def load_universe(lookback_years: float | None) -> pd.DataFrame:
    print("Loading local price panel...")
    prices = pd.read_parquet("multiasset_prices.parquet")
    meta = pd.read_parquet("multiasset_metadata.parquet")

    print("Fetching fresh prices from Yahoo Finance to top up staleness...")
    try:
        import yfinance as yf
        tickers = list(prices.columns)
        raw = yf.download(tickers, period="730d", auto_adjust=True,
                           progress=False)["Close"]
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        if len(raw.index) and raw.index.max().date() >= dt.date.today():
            raw = raw[raw.index.date < dt.date.today()]
        for col in raw.columns:
            if col not in prices.columns:
                continue
            new = raw[[col]].loc[raw.index > prices.index.max()]
            if not new.empty:
                prices = pd.concat([prices, new])
            fresh_col = raw[[col]].reindex(prices.index)
            prices[col] = prices[col].fillna(fresh_col[col])
        prices = prices.sort_index().loc[~prices.index.duplicated(keep="last")]
        print(f"  Latest date after refresh: {prices.index.max().date()}")
    except Exception as e:
        print(f"  yfinance top-up failed ({e}) — proceeding with cached panel only.")

    if lookback_years:
        cutoff = prices.index.max() - pd.Timedelta(days=int(365.25 * lookback_years))
        prices = prices[prices.index >= cutoff]

    return prices, meta


# ── PER-TICKER TESTS ────────────────────────────────────────────
def hurst_exponent(returns: np.ndarray) -> float:
    """Aggregated-variance method. H<0.5 mean-reverting, 0.5 random walk, >0.5 trending."""
    n = len(returns)
    lags = np.unique(np.logspace(0.7, np.log10(n // 4), 20).astype(int))
    lags = lags[lags >= 2]
    if len(lags) < 4:
        return np.nan
    tau = []
    for lag in lags:
        diffs = returns[lag:] - returns[:-lag] if lag < n else np.array([])
        # aggregated-variance: std of lagged cumulative sums
        m = n // lag
        if m < 2:
            continue
        agg = np.array([np.sum(returns[i * lag:(i + 1) * lag]) for i in range(m)])
        tau.append(np.std(agg))
    valid_lags = lags[:len(tau)]
    tau = np.array(tau)
    mask = tau > 0
    if mask.sum() < 4:
        return np.nan
    slope, _ = np.polyfit(np.log(valid_lags[mask]), np.log(tau[mask]), 1)
    return slope / 1.0  # slope of log(std) vs log(lag) ~ H (std scales as lag^H)


def ar1_halflife(x: np.ndarray) -> float:
    """Fit x_t = phi*x_{t-1} + eps to a detrended series; return half-life in days."""
    x0, x1 = x[:-1], x[1:]
    phi = np.dot(x0, x1) / np.dot(x0, x0)
    if not (0 < phi < 1):
        return np.nan
    return -np.log(2) / np.log(phi)


MIN_CYCLE_REPEATS = 5   # require at least this many full cycles in-sample to trust a period


def lombscargle_peak(t: np.ndarray, x: np.ndarray, min_period=10, max_period=None):
    """Dominant period via Lomb-Scargle, with AR(1) red-noise Monte Carlo significance.

    max_period is capped so a candidate period repeats at least MIN_CYCLE_REPEATS
    times in-sample -- an "11-year cycle" seen 3 times in 35 years of data is a
    couple of regime episodes dressed up as periodicity, not a tradeable cycle.
    """
    n = len(x)
    if max_period is None:
        max_period = n / MIN_CYCLE_REPEATS
    periods = np.geomspace(min_period, max_period, 300)
    freqs = 2 * np.pi / periods
    power = lombscargle(t, x - x.mean(), freqs, normalize=True)
    peak_idx = np.argmax(power)
    peak_period = periods[peak_idx]
    peak_power = power[peak_idx]

    # AR(1) surrogate null: fit AR(1) to x, simulate N_MC red-noise series of same
    # length/variance, compute their max periodogram power, get empirical p-value.
    x0, x1 = x[:-1], x[1:]
    phi = np.clip(np.dot(x0, x1) / np.dot(x0, x0), -0.99, 0.99)
    resid_std = np.std(x1 - phi * x0)
    rng = np.random.default_rng(42)
    null_max_power = np.empty(N_MC)
    for i in range(N_MC):
        sim = np.empty(n)
        sim[0] = x[0]
        eps = rng.normal(0, resid_std, n - 1)
        for j in range(1, n):
            sim[j] = phi * sim[j - 1] + eps[j - 1]
        sim_power = lombscargle(t, sim - sim.mean(), freqs, normalize=True)
        null_max_power[i] = sim_power.max()
    p_value = (null_max_power >= peak_power).mean()
    return peak_period, peak_power, p_value


def acf_confirmation(x: np.ndarray, target_lag: int):
    """Check autocorrelation near the periodogram's implied lag against the
    standard significance band."""
    n = len(x)
    max_lag = min(MAX_ACF_LAG_DAYS, n // 2, int(target_lag * 1.5) + 5)
    if max_lag < 2:
        return np.nan, np.nan
    vals = acf(x, nlags=max_lag, fft=True)
    band = 1.96 / np.sqrt(n)
    lo = max(1, int(target_lag * 0.85))
    hi = min(max_lag, int(target_lag * 1.15) + 1)
    if lo >= hi:
        return vals[min(int(target_lag), max_lag)], band
    window = vals[lo:hi]
    best = window[np.argmax(np.abs(window))]
    return best, band


def screen_ticker(prices: pd.Series) -> dict | None:
    s = prices.dropna()
    if len(s) < MIN_OBS:
        return None
    logp = np.log(s.values)
    t_idx = np.arange(len(logp), dtype=float)

    # 1. Trend strength
    slope, intercept = np.polyfit(t_idx, logp, 1)
    fitted = slope * t_idx + intercept
    resid = logp - fitted
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((logp - logp.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    se_slope = np.sqrt(ss_res / (len(t_idx) - 2) / np.sum((t_idx - t_idx.mean()) ** 2))
    trend_tstat = slope / se_slope if se_slope > 0 else np.nan
    ann_drift = slope * 252  # annualized log-return drift implied by the trend

    # 2. Hurst
    log_ret = np.diff(logp)
    hurst = hurst_exponent(log_ret)

    # 3. ADF on detrended series
    try:
        adf_stat, adf_p, *_ = adfuller(resid, autolag="AIC")
    except Exception:
        adf_p = np.nan

    # 4. Lomb-Scargle dominant cycle vs AR(1) null
    try:
        peak_period, peak_power, cycle_p = lombscargle_peak(t_idx, resid)
    except Exception:
        peak_period, peak_power, cycle_p = np.nan, np.nan, np.nan

    # 5. ACF confirmation at that lag
    if np.isfinite(peak_period):
        acf_val, acf_band = acf_confirmation(resid, int(round(peak_period)))
    else:
        acf_val, acf_band = np.nan, np.nan

    # Mean-reversion half-life (AR(1) on detrended series)
    halflife = ar1_halflife(resid)

    # ── Composite cyclicality score (0-100) ──
    # Reward: low |trend t-stat|, low Hurst, low ADF p, significant cycle
    # (low cycle_p), ACF confirming beyond its band.
    trend_score = max(0.0, 1 - min(abs(trend_tstat) / 3.0, 1.0)) if np.isfinite(trend_tstat) else 0.0
    hurst_score = max(0.0, min(1.0, (0.55 - hurst) / 0.3)) if np.isfinite(hurst) else 0.0
    adf_score = max(0.0, 1 - min(adf_p / 0.10, 1.0)) if np.isfinite(adf_p) else 0.0
    cycle_score = max(0.0, 1 - min(cycle_p / 0.10, 1.0)) if np.isfinite(cycle_p) else 0.0
    acf_score = max(0.0, min(1.0, abs(acf_val) / (3 * acf_band))) if np.isfinite(acf_val) and acf_band else 0.0

    composite = 100 * (0.20 * trend_score + 0.20 * hurst_score + 0.20 * adf_score
                        + 0.25 * cycle_score + 0.15 * acf_score)

    n_cycles = len(s) / peak_period if np.isfinite(peak_period) and peak_period > 0 else np.nan

    return dict(
        n_obs=len(s), trend_tstat=trend_tstat, trend_r2=r2, ann_drift_pct=ann_drift * 100,
        hurst=hurst, adf_p=adf_p, peak_period_days=peak_period, n_cycles_insample=n_cycles,
        cycle_p=cycle_p, acf_at_peak=acf_val, acf_band=acf_band, halflife_days=halflife,
        composite_score=composite,
    )


# ── MAIN ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-years", type=float, default=None,
                     help="Restrict to the trailing N years (default: full history)")
    ap.add_argument("--tickers", type=str, default=None,
                     help="Comma-separated subset of tickers (default: full 161-instrument universe)")
    ap.add_argument("--top", type=int, default=25, help="How many top candidates to print")
    ap.add_argument("--out", type=str, default="cyclicality_screen.csv")
    args = ap.parse_args()

    prices, meta = load_universe(args.lookback_years)
    tickers = args.tickers.split(",") if args.tickers else list(prices.columns)
    tickers = [t for t in tickers if t in prices.columns]

    print(f"\nScreening {len(tickers)} tickers "
          f"(Hurst + ADF + Lomb-Scargle + ACF, {N_MC} MC draws each)...")
    rows = []
    for i, tkr in enumerate(tickers, 1):
        res = screen_ticker(prices[tkr])
        if res is None:
            continue
        res["ticker"] = tkr
        rows.append(res)
        if i % 20 == 0 or i == len(tickers):
            print(f"  {i}/{len(tickers)} done")

    df = pd.DataFrame(rows).set_index("ticker")
    df = df.join(meta.set_index("ticker")[["asset_class", "full_name"]])
    df = df.sort_values("composite_score", ascending=False)

    cols = ["asset_class", "full_name", "composite_score", "trend_tstat", "ann_drift_pct",
            "hurst", "adf_p", "peak_period_days", "n_cycles_insample", "cycle_p",
            "acf_at_peak", "halflife_days", "n_obs"]
    df[cols].to_csv(args.out)
    print(f"\nSaved full results to {args.out}")

    print(f"\nTop {args.top} candidates by cyclicality score:\n")
    top = df[cols].head(args.top).copy()
    top["composite_score"] = top["composite_score"].round(1)
    top["trend_tstat"] = top["trend_tstat"].round(2)
    top["ann_drift_pct"] = top["ann_drift_pct"].round(2)
    top["hurst"] = top["hurst"].round(3)
    top["adf_p"] = top["adf_p"].round(3)
    top["peak_period_days"] = top["peak_period_days"].round(0)
    top["n_cycles_insample"] = top["n_cycles_insample"].round(1)
    top["cycle_p"] = top["cycle_p"].round(3)
    top["acf_at_peak"] = top["acf_at_peak"].round(3)
    top["halflife_days"] = top["halflife_days"].round(0)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(top.to_string())


if __name__ == "__main__":
    main()
