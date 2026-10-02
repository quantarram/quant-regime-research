"""
intraday_cpe_feasibility_test.py
========================================
Applies the exact same naive-CPE feasibility test already run on crypto
intraday data (files/cpe_engine_intraday_btc.py: CPE >= 0.80, lift >= 1.5x,
n >= 100, horizons 1/5/15/30/60/240 one-minute bars, self-referential
pairs included) to real NVDA/AMD/SPY 1-minute SIP bars instead of BTC/ETH.

That crypto test found zero of 73,728 configurations cleared the bar
(max CPE ~0.55 vs the 0.80 threshold) across BTC/ETH/SOL/BNB over a full
year. This is the direct equities analogue -- same gate, same horizons,
different asset class and different market microstructure (scheduled
regular-session hours instead of 24/7 trading).

Restricted to regular trading hours (9:30-16:00 US/Eastern) -- extended
hours trading is thin and gappy and would confound a short-horizon test
with a liquidity artifact rather than a genuine predictability question.

Run: python intraday_cpe_feasibility_test.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import json

HORIZONS_MIN = [1, 5, 15, 30, 60, 240]
Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]
CPE_THRESH = 0.80
MIN_LIFT = 1.5
MIN_N = 100
SYMBOLS = ["NVDA", "AMD", "SPY"]


def load_regular_session(symbol: str) -> pd.Series:
    df = pd.read_parquet(f"intraday_{symbol.lower()}_1m.parquet")
    et = df.index.tz_convert("US/Eastern")
    mask = (et.time >= pd.Timestamp("09:30").time()) & (et.time <= pd.Timestamp("16:00").time())
    close = df["close"][mask]
    close.index = close.index.tz_localize(None)
    return close


if __name__ == "__main__":
    prices = {}
    for sym in SYMBOLS:
        s = load_regular_session(sym)
        prices[sym] = s
        print(f"{sym}: {len(s)} regular-session 1-min bars, {s.index.min()} to {s.index.max()}")

    df_prices = pd.DataFrame(prices).dropna(how="all")
    print(f"\nCombined panel: {df_prices.shape}")

    # increments at each horizon, using bar-COUNT shifts within the
    # regular-session-only series (so a 60-bar shift means 60 real
    # trading minutes, not 60 calendar minutes spanning an overnight gap)
    increments = {h: np.log(df_prices / df_prices.shift(h)) for h in HORIZONS_MIN}
    future_inc = {h: increments[h].shift(-h) for h in HORIZONS_MIN}

    full_q_grid = sorted(set(Q_GRID + [round(1 - q, 10) for q in Q_GRID]))
    thresholds = {(h, q): increments[h].quantile(q) for h in HORIZONS_MIN for q in full_q_grid}

    results = []
    total_combos = 0
    for y in SYMBOLS:
        for h_f in HORIZONS_MIN:
            fy = future_inc[h_f][y].dropna()
            for h_p in HORIZONS_MIN:
                px_all = increments[h_p]
                common_idx = fy.index.intersection(px_all.dropna(how="all").index)
                if len(common_idx) < MIN_N:
                    continue
                fy_vals = fy.loc[common_idx].values
                px_aligned = px_all.loc[common_idx]
                for q_y in Q_GRID:
                    thresh_y_up = thresholds[(h_f, q_y)].get(y, np.nan)
                    thresh_y_dn = thresholds[(h_f, round(1 - q_y, 10))].get(y, np.nan)
                    if np.isnan(thresh_y_up) or np.isnan(thresh_y_dn):
                        continue
                    uncond = 1.0 - q_y
                    event_bull = fy_vals > thresh_y_up
                    event_bear = fy_vals < thresh_y_dn
                    for x in SYMBOLS:
                        px_vals = px_aligned[x].values
                        valid = ~np.isnan(px_vals)
                        if valid.sum() < MIN_N:
                            continue
                        for q_x in Q_GRID:
                            thresh_x_up = thresholds[(h_p, q_x)].get(x, np.nan)
                            thresh_x_dn = thresholds[(h_p, round(1 - q_x, 10))].get(x, np.nan)
                            if np.isnan(thresh_x_up) or np.isnan(thresh_x_dn):
                                continue
                            total_combos += 2
                            cond_bull = valid & (px_vals > thresh_x_up)
                            if cond_bull.sum() >= MIN_N:
                                cpe = event_bull[cond_bull].mean()
                                lift = cpe / uncond if uncond > 0 else np.nan
                                if cpe >= CPE_THRESH and lift >= MIN_LIFT:
                                    results.append((y, x, h_p, h_f, q_x, q_y, round(float(cpe),4), round(float(lift),4), int(cond_bull.sum()), "bullish"))
                            cond_bear = valid & (px_vals < thresh_x_dn)
                            if cond_bear.sum() >= MIN_N:
                                cpe = event_bear[cond_bear].mean()
                                lift = cpe / uncond if uncond > 0 else np.nan
                                if cpe >= CPE_THRESH and lift >= MIN_LIFT:
                                    results.append((y, x, h_p, h_f, q_x, q_y, round(float(cpe),4), round(float(lift),4), int(cond_bear.sum()), "bearish"))

    cols = ["Y","X","horizon_past_min","horizon_future_min","q_X","q_Y","CPE","lift","n_condition","direction"]
    df_results = pd.DataFrame(results, columns=cols)
    print(f"\nTotal configurations tested: {total_combos:,}")
    print(f"Surviving CPE>={CPE_THRESH}/lift>={MIN_LIFT}/n>={MIN_N}: {len(df_results)}")

    # what's the best CPE achieved anywhere, even if it doesn't clear the bar?
    all_cpes = []
    for y in SYMBOLS:
        for h_f in HORIZONS_MIN:
            fy = future_inc[h_f][y].dropna()
            for h_p in HORIZONS_MIN:
                px_all = increments[h_p]
                common_idx = fy.index.intersection(px_all.dropna(how="all").index)
                if len(common_idx) < MIN_N:
                    continue
                fy_vals = fy.loc[common_idx].values
                px_aligned = px_all.loc[common_idx]
                for q_y in [0.80, 0.90, 0.95, 0.99]:
                    thresh_y_up = thresholds[(h_f, q_y)].get(y, np.nan)
                    if np.isnan(thresh_y_up):
                        continue
                    uncond = 1.0 - q_y
                    event_bull = fy_vals > thresh_y_up
                    for x in SYMBOLS:
                        px_vals = px_aligned[x].values
                        valid = ~np.isnan(px_vals)
                        if valid.sum() < MIN_N:
                            continue
                        for q_x in [0.80, 0.90, 0.95, 0.99]:
                            thresh_x_up = thresholds[(h_p, q_x)].get(x, np.nan)
                            if np.isnan(thresh_x_up):
                                continue
                            cond = valid & (px_vals > thresh_x_up)
                            if cond.sum() >= MIN_N:
                                all_cpes.append(float(event_bull[cond].mean()))

    print(f"\nMax CPE achieved anywhere (q_X,q_Y both >=0.80, any horizon pair): {max(all_cpes):.4f}" if all_cpes else "no valid combos")
    print(f"Median CPE across all tested tail combos: {np.median(all_cpes):.4f}" if all_cpes else "")

    df_results.to_parquet("intraday_equities_cpe_results.parquet")
    summary = {
        "total_combos_tested": total_combos,
        "n_surviving": len(df_results),
        "max_cpe_tail_only": float(max(all_cpes)) if all_cpes else None,
        "median_cpe_tail_only": float(np.median(all_cpes)) if all_cpes else None,
        "symbols": SYMBOLS, "horizons_min": HORIZONS_MIN,
    }
    json.dump(summary, open("intraday_equities_cpe_summary.json", "w"), indent=2)
    print("\n" + json.dumps(summary, indent=2))
