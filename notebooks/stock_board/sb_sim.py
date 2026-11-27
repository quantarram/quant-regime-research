"""
Paper-account simulation of the stock board: "what if I had followed every pick".

Rules (fixed, stated on the page):
  - Start with START_EQUITY (US$, paper money; prices are in US$, no currency conversion).
  - Each pick is bought at its entry close with WEIGHT (10%) of the account's value that day, no leverage, no margin.
  - At most MAX_POS (10) positions open at once and one position per ETF; a pick that finds the account full, or the
    ETF already held, is skipped. When several picks enter on the same day, the deepest 1-month drop is funded first.
  - A position is sold at its target, stop or time limit exactly as in the bet definition; the 5 bps round-trip cost
    is already inside each bet's P&L. Unfunded cash earns nothing.
  - The account is marked to market on every trading day with each ETF's own closing price.
"""
import numpy as np
import pandas as pd

START_EQUITY = 100_000.0
WEIGHT = 0.10
MAX_POS = 10


def simulate(trades, prices, start_equity=START_EQUITY, weight=WEIGHT, max_pos=MAX_POS, start=None, end=None, cash_rate=None):
    """
    trades: DataFrame with ticker, entry_date, entry_px, exit_date (NaT if still open), pnl (NaN if open), rank (lower funded first).
    prices: DataFrame of closes (union calendar).
    Returns dict(equity=Series, positions=DataFrame of funded trades, skipped=int).
    """
    tr = trades.copy()
    tr["entry_date"] = pd.to_datetime(tr["entry_date"])
    tr["exit_date"] = pd.to_datetime(tr["exit_date"])
    start = pd.Timestamp(start) if start is not None else tr["entry_date"].min()
    end = pd.Timestamp(end) if end is not None else prices.index[-1]
    days = prices.loc[start:end].index
    px = prices.loc[start:end].ffill()
    cr = (cash_rate.reindex(days).ffill().fillna(0.0) / 100.0 / 252.0) if cash_rate is not None else pd.Series(0.0, index=days)
    tr = tr.sort_values(["entry_date", "rank"])
    by_entry = {d: g for d, g in tr.groupby("entry_date")}
    cash = start_equity
    open_pos = {}  # id -> dict(ticker, alloc, entry_px, exit_date, pnl)
    eq, funded, skipped = [], [], 0
    nid = 0
    for d in days:
        cash *= 1 + cr.at[d]
        # 1) exits at today's close
        for k in [k for k, p in open_pos.items() if pd.notna(p["exit_date"]) and p["exit_date"] <= d]:
            p = open_pos.pop(k)
            cash += p["alloc"] * (1 + p["pnl"])
        # 2) value of what is still held (before today's new entries)
        held = sum(p["alloc"] * px.at[d, p["ticker"]] / p["entry_px"] for p in open_pos.values())
        equity = cash + held
        # 3) entries at today's close
        if d in by_entry:
            for _, t in by_entry[d].iterrows():
                if len(open_pos) >= max_pos or any(p["ticker"] == t["ticker"] for p in open_pos.values()):
                    skipped += 1
                    continue
                alloc = min(weight * equity, cash)
                if alloc <= 0:
                    skipped += 1
                    continue
                cash -= alloc
                open_pos[nid] = dict(ticker=t["ticker"], alloc=alloc, entry_px=t["entry_px"], exit_date=t["exit_date"], pnl=t["pnl"])
                funded.append(dict(ticker=t["ticker"], entry_date=d, entry_px=t["entry_px"], exit_date=t["exit_date"],
                                   pnl=t["pnl"], alloc=alloc, equity_at_entry=equity))
                nid += 1
        held = sum(p["alloc"] * px.at[d, p["ticker"]] / p["entry_px"] for p in open_pos.values())
        eq.append((d, cash + held, held / (cash + held) if (cash + held) > 0 else 0.0, len(open_pos)))
    E = pd.DataFrame(eq, columns=["date", "equity", "exposure", "n_open"]).set_index("date")
    return dict(equity=E, positions=pd.DataFrame(funded), skipped=skipped, open_positions=open_pos)


def stats(E, bench=None, cash_rate=None):
    eq = E["equity"]
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0.05 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    out = dict(total_return=eq.iloc[-1] / eq.iloc[0] - 1, cagr=cagr, max_drawdown=dd,
               vol=r.std() * np.sqrt(252), sharpe=(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
               avg_exposure=E["exposure"].mean(), years=yrs)
    if bench is not None:
        b = bench.reindex(eq.index).ffill()
        br = b.pct_change().dropna()
        out["bench_total"] = b.iloc[-1] / b.iloc[0] - 1
        out["bench_cagr"] = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1 if yrs > 0.05 else np.nan
        out["bench_dd"] = (b / b.cummax() - 1).min()
        # exposure-matched benchmark: hold the benchmark with the strategy's own daily exposure, rest in cash
        ex = E["exposure"].shift(1).reindex(br.index).fillna(0.0)
        cr = (cash_rate.reindex(br.index).ffill().fillna(0.0) / 100.0 / 252.0) if cash_rate is not None else 0.0
        m = (1 + ex * br + (1 - ex) * cr).cumprod()
        out["matched_total"] = m.iloc[-1] - 1
        out["matched_cagr"] = m.iloc[-1] ** (1 / yrs) - 1 if yrs > 0.05 else np.nan
        out["matched_dd"] = (m / m.cummax() - 1).min()
    return out


def backtest_payload(oos21, prices, start="2010-01-04"):
    """Walk-forward picks -> paper account, summarised for the page (month-end points keep the file small)."""
    tr = oos21[["ticker", "entry_date", "entry_px", "exit_date", "pnl"]].copy()
    tr["rank"] = oos21["mom21"].values
    cols = [t for t in prices.columns if t in set(oos21["ticker"]) | {"SPY"}]
    irx = prices["^IRX"].dropna()
    res = simulate(tr, prices[cols], start=start, cash_rate=irx)
    E = res["equity"]
    spy = prices["SPY"].dropna().reindex(E.index).ffill()
    st = stats(E, spy, irx)
    ex = E["exposure"].shift(1).fillna(0.0)
    br = spy.pct_change().fillna(0.0)
    cr = (irx.reindex(E.index).ffill().fillna(0.0) / 100.0 / 252.0)
    matched = (1 + ex * br + (1 - ex) * cr).cumprod()
    norm = lambda s: (s / s.iloc[0] * 100.0)
    M = pd.DataFrame({"strategy": norm(E["equity"]), "spy": norm(spy), "matched": norm(matched)}).resample("ME").last()
    yearly = pd.DataFrame({"strategy": E["equity"].resample("YE").last().pct_change(),
                           "spy": spy.resample("YE").last().pct_change()})
    yearly.iloc[0] = [E["equity"].resample("YE").last().iloc[0] / E["equity"].iloc[0] - 1,
                      spy.resample("YE").last().iloc[0] / spy.iloc[0] - 1]
    dd = E["equity"] / E["equity"].cummax() - 1
    trough = dd.idxmin()
    peak = E["equity"][:trough].idxmax()
    pos = res["positions"]
    return {
        "params": dict(start_equity=START_EQUITY, weight=WEIGHT, max_pos=MAX_POS, start=start),
        "stats": {k: (None if not np.isfinite(v) else float(v)) for k, v in st.items()},
        "n_funded": int(len(pos)), "n_skipped": int(res["skipped"]),
        "funded_hit": float((pos["pnl"] > 0).mean()), "funded_mean_pnl": float(pos["pnl"].mean()),
        "avg_open": float(E["n_open"].mean()), "days_at_cap": float((E["n_open"] >= MAX_POS).mean()),
        "worst": dict(peak=str(peak.date()), trough=str(trough.date()), dd=float(dd.min()),
                      exposure_peak=float(E["exposure"][peak]), exposure_trough=float(E["exposure"][trough])),
        "monthly": dict(dates=[str(d.date()) for d in M.index], strategy=[round(float(v), 3) for v in M["strategy"]],
                        spy=[round(float(v), 3) for v in M["spy"]], matched=[round(float(v), 3) for v in M["matched"]]),
        "yearly": {str(d.year): {"strategy": float(r["strategy"]), "spy": float(r["spy"])} for d, r in yearly.iterrows()},
    }
