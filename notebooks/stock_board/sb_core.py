"""
Stock board core: barrier-bet outcomes, form features, rule conditions.

A "bet" is the stock analogue of a football bet with a posted price:
  - signal at the close of bar i (features use data through bar i only)
  - entry at the NEXT bar's close, e = i+1 (a one-day delay, since the board is read after the close)
  - target +a and stop -b, both set at entry from the instrument's own trailing volatility:
        sigma_h = std of the last 63 daily log returns (at bar i) * sqrt(h)
        a = K_TARGET * sigma_h,  b = K_STOP * sigma_h
  - monitored on closes of bars e+1 .. e+h (the instrument's OWN trading days, see horizon_calendar.py)
  - WIN  = target hit first; LOSS = stop hit first; TIMEOUT = neither within h bars, scored WIN if the
    return at bar e+h is positive, else LOSS
  - realised P&L = return at the exit close (so overshoot past a barrier is counted, not idealised),
    minus COST_RT round-trip cost

Football comparison: break-even hit rate = |avg loss| / (avg win + |avg loss|), the stock version of 1 / odds.
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parent

K_TARGET = 0.5      # target = 0.5 * sigma_h   (a high-hit-rate, small-win / larger-loss bet, like football's 1.1-1.3 odds)
K_STOP = 1.5        # stop   = 1.5 * sigma_h   (zero-drift break-even hit rate = 1.5/2.0 = 75%)
COST_RT = 0.0005    # 5 bps round trip
HORIZONS = (21, 63)
VOL_WIN = 63
PCT_WIN = 1260      # percentile rank is taken against the instrument's own trailing 1260 bars (point in time)
PCT_MIN = 504

# Tradable, non-leveraged, non-volatility instruments with a long own-bar history.
UNIVERSE = {
    "US equity ETFs": ["SPY", "QQQ", "IWM", "DIA", "VTI"],
    "World / regional ETFs": ["VT", "EFA", "EEM", "VEA", "VWO", "EWJ", "EWZ", "FXI", "INDA", "EWY"],
    "Sector ETFs": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLRE", "XLB", "XLC",
                    "XBI", "SOXX", "ITB", "ICLN", "ARKK"],
    "Factor ETFs": ["VTV", "VUG", "MTUM", "USMV", "QUAL", "SIZE"],
    "Metals ETFs": ["GLD", "SLV", "PPLT", "PALL"],
    "Other commodity ETFs": ["USO", "DBC", "DBB", "CPER"],
    "Bond ETFs": ["TLT", "IEF", "LQD", "HYG", "EMB", "AGG", "TIP"],
}
# Single stocks are NOT in the validated universe: the ten names in the stored panel were picked after they became
# very large, so their history is survivor-selected (with them included the 63-bar result looked significant; without
# them it does not). Validating individual stocks needs a point-in-time constituent list.
EXCLUDED_SINGLE_STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "BRK-B", "XOM"]
TICKER_GROUP = {t: g for g, ts in UNIVERSE.items() for t in ts}
ALL_TICKERS = [t for ts in UNIVERSE.values() for t in ts]

# condition name -> (feature, side).  side "hi": percentile >= q ;  side "lo": percentile <= 1 - q
CONDITIONS = {
    "mom126_hi": ("mom126", "hi"),     # strong 6-month momentum
    "mom21_hi": ("mom21", "hi"),       # strong 1-month momentum
    "mom21_lo": ("mom21", "lo"),       # 1-month pullback
    "trend_hi": ("trend", "hi"),       # far above the 200-day average
    "dd_near_high": ("dd", "hi"),      # close to the 252-day high
    "dd_deep": ("dd", "lo"),           # deep below the 252-day high
    "vol_calm": ("volratio", "lo"),    # 1m vol low relative to 1y vol
    "vol_stressed": ("volratio", "hi"),
}
FEATURES = ["mom126", "mom21", "trend", "dd", "volratio"]
Q_GRID = (0.7, 0.8, 0.9)


def load_panel(path=None):
    path = Path(path) if path else NB / "multiasset_prices_live_history.parquet"
    return pd.read_parquet(path)


def _trailing_pct_rank(x, win=PCT_WIN, min_obs=PCT_MIN):
    """Point-in-time percentile rank of x[t] within x[t-win+1..t] (own history, includes t)."""
    a = np.asarray(x, dtype=float)
    n = len(a)
    out = np.full(n, np.nan)
    for t in range(n):
        if np.isnan(a[t]):
            continue
        lo = max(0, t - win + 1)
        w = a[lo:t + 1]
        w = w[~np.isnan(w)]
        if len(w) < min_obs:
            continue
        out[t] = (w <= a[t]).mean()
    return out


def features_for(close):
    """close: Series on the instrument's own bars. Returns DataFrame of raw features, sigma_d, and percentile ranks."""
    r = np.log(close / close.shift(1))
    f = pd.DataFrame(index=close.index)
    f["close"] = close
    f["sigma_d"] = r.rolling(VOL_WIN).std()
    f["mom126"] = close / close.shift(126) - 1
    f["mom21"] = close / close.shift(21) - 1
    f["trend"] = close / close.rolling(200).mean() - 1
    f["dd"] = close / close.rolling(252).max() - 1
    f["volratio"] = r.rolling(21).std() / r.rolling(252).std()
    for c in FEATURES:
        f[c + "_pct"] = _trailing_pct_rank(f[c].values)
    return f


def barrier_outcomes(close, sigma_d, h, k_t=K_TARGET, k_s=K_STOP, cost=COST_RT):
    """
    For every signal bar i, entry at bar i+1, barriers from sigma_d[i]. Returns DataFrame indexed by signal date with
    entry_date, entry_px, target_px, stop_px, exit_date, exit_px, outcome in {TARGET, STOP, TIMEOUT}, win, pnl.
    Rows whose horizon is not yet complete have outcome NaN.
    """
    P = close.values.astype(float)
    n = len(P)
    idx = close.index
    sig = sigma_d.values
    cols = {k: np.full(n, np.nan) for k in ("entry_px", "target_px", "stop_px", "exit_px", "pnl")}
    exit_pos = np.full(n, -1)
    outcome = np.array([None] * n, dtype=object)
    win = np.full(n, np.nan)
    for i in range(n - 1):
        s = sig[i]
        if not np.isfinite(s):
            continue
        e = i + 1
        sh = s * np.sqrt(h)
        a = k_t * sh
        b = min(k_s * sh, 0.95)
        pe = P[e]
        cols["entry_px"][i] = pe
        cols["target_px"][i] = pe * (1 + a)
        cols["stop_px"][i] = pe * (1 - b)
        end = e + h
        if end > n - 1:
            # horizon incomplete: still check whether a barrier has already been hit (open bet)
            path = P[e + 1:] / pe - 1
        else:
            path = P[e + 1:end + 1] / pe - 1
        hit_t = np.where(path >= a)[0]
        hit_s = np.where(path <= -b)[0]
        jt = hit_t[0] if len(hit_t) else 10**9
        js = hit_s[0] if len(hit_s) else 10**9
        if min(jt, js) < 10**9:
            if jt < js:
                outcome[i] = "TARGET"; win[i] = 1.0; j = jt
            else:
                outcome[i] = "STOP"; win[i] = 0.0; j = js
            exit_pos[i] = e + 1 + j
            cols["exit_px"][i] = P[e + 1 + j]
            cols["pnl"][i] = path[j] - cost
        elif end <= n - 1:
            outcome[i] = "TIMEOUT"
            exit_pos[i] = end
            cols["exit_px"][i] = P[end]
            cols["pnl"][i] = path[-1] - cost
            win[i] = 1.0 if path[-1] > 0 else 0.0
    out = pd.DataFrame(cols, index=idx)
    out["entry_date"] = pd.Series(idx, index=idx).shift(-1)
    out["exit_date"] = [idx[p] if p >= 0 else pd.NaT for p in exit_pos]
    out["outcome"] = outcome
    out["win"] = win
    out["h"] = h
    return out


def condition_mask(pct_df, name, q):
    feat, side = CONDITIONS[name]
    p = pct_df[feat + "_pct"]
    return (p >= q) if side == "hi" else (p <= 1 - q)


def build_rules():
    names = list(CONDITIONS)
    rules = []
    for q in Q_GRID:
        for a in names:
            rules.append(((a,), q))
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if CONDITIONS[a][0] == CONDITIONS[b][0]:
                    continue
                rules.append(((a, b), q))
    return rules


def rule_label(rule):
    conds, q = rule
    return " + ".join(conds) + f" @{int(q * 100)}th pct"


def evaluate_bet(close, sig_pos, sigma_h, h, k_t=K_TARGET, k_s=K_STOP, cost=COST_RT):
    """
    Live version of barrier_outcomes for ONE signal at own-bar position sig_pos. Returns a dict with status
    PENDING_ENTRY (entry bar not yet available), OPEN (entered, no barrier hit, horizon not complete) or RESOLVED.
    """
    P = close.values.astype(float)
    idx = close.index
    n = len(P)
    e = sig_pos + 1
    a = k_t * sigma_h
    b = min(k_s * sigma_h, 0.95)
    if e > n - 1:
        return dict(status="PENDING_ENTRY", a=a, b=b)
    pe = P[e]
    res = dict(status="OPEN", entry_date=idx[e], entry_px=pe, target_px=pe * (1 + a), stop_px=pe * (1 - b),
               a=a, b=b, bars_elapsed=n - 1 - e, last_px=P[-1], last_date=idx[-1])
    end = e + h
    path = P[e + 1:min(end, n - 1) + 1] / pe - 1
    ht = np.where(path >= a)[0]
    hs = np.where(path <= -b)[0]
    jt = ht[0] if len(ht) else 10**9
    js = hs[0] if len(hs) else 10**9
    if min(jt, js) < 10**9:
        j = jt if jt < js else js
        res.update(status="RESOLVED", outcome="TARGET" if jt < js else "STOP", win=1 if jt < js else 0,
                   exit_date=idx[e + 1 + j], exit_px=P[e + 1 + j], pnl=path[j] - cost)
    elif end <= n - 1:
        res.update(status="RESOLVED", outcome="TIMEOUT", win=1 if path[-1] > 0 else 0,
                   exit_date=idx[end], exit_px=P[end], pnl=path[-1] - cost)
    return res
