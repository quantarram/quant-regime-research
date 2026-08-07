"""
tsmom_all22_vs_own_methods.py
================================
Completes the added-value check properly: the previous test only ran
TSMOM on the 6 instruments already selected for showing convergent
positive alpha under the user's own methods -- a real selection issue,
flagged before running this. This runs the identical, unfitted TSMOM
spec (252d lookback, 63d vol window) against the OTHER 16 instruments in
Paper 12's 22-instrument panel too, so the full picture is visible: does
TSMOM specifically struggle where the user's own methods show signal, or
does TSMOM struggle broadly across single instruments in this 2022+
period regardless of which ones the user's own methods like?

Same holdout (2022+), same own-buy-and-hold benchmark, same everything,
for both groups -- a genuinely complete, non-selected comparison.

No significance/randomisation-test games -- real point estimates only.

Run: python tsmom_all22_vs_own_methods.py
Output: tsmom_all22_results.json, tsmom_all22_plot.png
"""
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOOKBACK, VOL_WINDOW, TARGET_VOL, MAX_LEV, COST_BPS = 252, 63, 0.10, 2.0, 5
HOLDOUT_START = pd.Timestamp("2022-01-01")

CONVERGENT_POSITIVE = {
    "JPM": {"Kelly-sized": 11.64, "Master model": 2.78},
    "GLD": {"Kelly-sized": 6.84, "Master model": 2.08},
    "XLE": {"Kelly-sized": 6.55, "Master model": 2.88},
    "XLB": {"Kelly-sized": 2.53, "Master model": 1.49},
    "XLU": {"Kelly-sized": 1.16, "Master model": 0.91},
    "MSFT": {"Kelly-sized": 0.48, "Master model": 0.12},
}
OTHER_16 = {
    "QQQ": {"Kelly-sized": 3.09, "Master model": -3.56},
    "XLK": {"Kelly-sized": 2.74, "Master model": -1.62},
    "DIA": {"Kelly-sized": 1.38, "Master model": -0.83},
    "XLY": {"Kelly-sized": 0.33, "Master model": -0.84},
    "VTI": {"Kelly-sized": 0.05, "Master model": -0.18},
    "SPY": {"Kelly-sized": -0.06, "Master model": -0.17},
    "XLP": {"Kelly-sized": -0.32, "Master model": -0.01},
    "XLF": {"Kelly-sized": -0.43, "Master model": -0.85},
    "VOX": {"Kelly-sized": -0.63, "Master model": -1.50},
    "EURUSD=X": {"Kelly-sized": -2.83, "Master model": -1.35},
    "IWM": {"Kelly-sized": -3.40, "Master model": -7.13},
    "XOM": {"Kelly-sized": -5.65, "Master model": 0.25},
    "AAPL": {"Kelly-sized": -6.00, "Master model": -3.80},
    "XLV": {"Kelly-sized": -7.56, "Master model": -2.34},
    "XLI": {"Kelly-sized": -7.70, "Master model": -2.86},
    "IYR": {"Kelly-sized": -7.73, "Master model": None},
}
PROXY_TICKERS = {"VOX", "IYR"}


def get_series(tkr, prices, proxy):
    return (proxy[tkr] if tkr in PROXY_TICKERS else prices[tkr]).dropna()


def tsmom_alpha(tkr, prices, proxy):
    s = get_series(tkr, prices, proxy)
    daily_ret = np.log(s).diff()
    eval_dates = s.index[s.index >= HOLDOUT_START]
    if len(eval_dates) < 60:
        return None
    rows = {}
    for d in eval_dates[::21]:
        hist = s.loc[:d]
        if len(hist) < LOOKBACK + VOL_WINDOW + 5:
            rows[d] = np.nan
            continue
        trail = np.log(hist.iloc[-1] / hist.iloc[-LOOKBACK])
        r = daily_ret.loc[:d].tail(VOL_WINDOW)
        vol = r.std() * np.sqrt(252)
        rows[d] = float(np.clip(np.sign(trail) * (TARGET_VOL / vol), -MAX_LEV, MAX_LEV)) if vol > 1e-6 else np.nan
    w = pd.Series(rows).reindex(eval_dates, method="ffill").shift(1)
    strat_ret = w * daily_ret.reindex(eval_dates)
    turnover = w.diff().abs().fillna(0.0)
    strat_ret = (strat_ret - turnover * (COST_BPS / 10000.0)).dropna()
    bh_ret = daily_ret.reindex(strat_ret.index)
    if len(strat_ret) < 60:
        return None
    return float((strat_ret.mean() - bh_ret.mean()) * 252 * 100)


if __name__ == "__main__":
    prices = pd.read_parquet("multiasset_prices.parquet")
    proxy = pd.read_parquet("predictor_v1/sector_proxy_cache.parquet")

    results = {}
    print("Convergent-positive group (own methods showed real signal here):")
    for tkr, own in CONVERGENT_POSITIVE.items():
        a = tsmom_alpha(tkr, prices, proxy)
        results[tkr] = {"group": "convergent_positive", "tsmom_alpha_pct": a, **own}
        print(f"  {tkr:<10} TSMOM={a:+7.2f}%   Kelly={own['Kelly-sized']:+7.2f}%   Master={own['Master model']:+7.2f}%")

    print("\nOther 16 (own methods showed negative or mixed signal here):")
    for tkr, own in OTHER_16.items():
        a = tsmom_alpha(tkr, prices, proxy)
        results[tkr] = {"group": "other_16", "tsmom_alpha_pct": a, **own}
        master_str = f"{own['Master model']:+.2f}%" if own["Master model"] is not None else "n/a"
        print(f"  {tkr:<10} TSMOM={a:+7.2f}%   Kelly={own['Kelly-sized']:+7.2f}%   Master={master_str}")

    tsmom_vals_cp = [r["tsmom_alpha_pct"] for r in results.values() if r["group"] == "convergent_positive" and r["tsmom_alpha_pct"] is not None]
    tsmom_vals_o16 = [r["tsmom_alpha_pct"] for r in results.values() if r["group"] == "other_16" and r["tsmom_alpha_pct"] is not None]
    n_neg_cp = sum(1 for v in tsmom_vals_cp if v < 0)
    n_neg_o16 = sum(1 for v in tsmom_vals_o16 if v < 0)
    print(f"\nTSMOM negative on {n_neg_cp}/{len(tsmom_vals_cp)} convergent-positive instruments "
          f"(median {np.median(tsmom_vals_cp):+.2f}%)")
    print(f"TSMOM negative on {n_neg_o16}/{len(tsmom_vals_o16)} other instruments "
          f"(median {np.median(tsmom_vals_o16):+.2f}%)")

    with open("tsmom_all22_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(13, 9))
    order = sorted(results.items(), key=lambda kv: (kv[1]["tsmom_alpha_pct"] is None, kv[1]["tsmom_alpha_pct"] or 0))
    labels = [f"{t} ({'own-signal' if r['group']=='convergent_positive' else 'other'})" for t, r in order]
    vals = [r["tsmom_alpha_pct"] for _, r in order]
    colors = ["#2E6DA4" if r["group"] == "convergent_positive" else "#9AA1AD" for _, r in order]
    y = np.arange(len(order))
    ax.barh(y, [v if v is not None else 0 for v in vals], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Standard TSMOM alpha vs. own buy-and-hold, 2022+ holdout (%/yr)")
    ax.set_title("TSMOM applied to ALL 22 Paper 12 instruments, not just the 6 selected for own-method signal\n"
                  "Blue = instruments where this program's own methods showed convergent positive alpha; grey = the other 16")
    fig.tight_layout()
    fig.savefig("tsmom_all22_plot.png", dpi=140)
    plt.close(fig)
    print("\nSaved: tsmom_all22_results.json, tsmom_all22_plot.png")
