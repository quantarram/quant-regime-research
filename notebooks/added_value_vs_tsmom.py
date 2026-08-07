"""
added_value_vs_tsmom.py
=========================
For the instruments where Paper 12's own methods (Kelly-sized, master
model) show real, convergent positive alpha across two independently-
designed approaches on the same 2022+ holdout -- JPM, GLD, XLE, XLB, XLU,
MSFT -- this asks the direct "added value" question: does a standard,
unfitted TSMOM signal (252d lookback, 63d vol window, the exact same
spec used all session) applied to that SAME instrument, over the SAME
holdout period, do as well? If the user's own methods add real value
over an existing quant-shop tool, it should show up directly here, on
the same instrument, the same period, the same benchmark.

No significance/randomisation-test games -- real point estimates, the
same 2022+ holdout Paper 12 itself used, broken down by real year for
replication context, reported directly.

Run: python added_value_vs_tsmom.py
Output: added_value_vs_tsmom_results.json, added_value_vs_tsmom_plot.png
"""
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INSTRUMENTS = ["JPM", "GLD", "XLE", "XLB", "XLU", "MSFT"]
LOOKBACK, VOL_WINDOW, TARGET_VOL, MAX_LEV, COST_BPS = 252, 63, 0.10, 2.0, 5
HOLDOUT_START = pd.Timestamp("2022-01-01")

OWN_METHOD_ALPHAS = {
    "JPM": {"Kelly-sized": 11.64, "Master model": 2.78},
    "GLD": {"Kelly-sized": 6.84, "Master model": 2.08},
    "XLE": {"Kelly-sized": 6.55, "Master model": 2.88},
    "XLB": {"Kelly-sized": 2.53, "Master model": 1.49},
    "XLU": {"Kelly-sized": 1.16, "Master model": 0.91},
    "MSFT": {"Kelly-sized": 0.48, "Master model": 0.12},
}


def sharpe(ret):
    r = ret.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")


if __name__ == "__main__":
    prices = pd.read_parquet("multiasset_prices.parquet")

    results = {}
    for tkr in INSTRUMENTS:
        s = prices[tkr].dropna()
        daily_ret = np.log(s).diff()
        eval_dates = s.index[s.index >= HOLDOUT_START]

        rows = {}
        for d in eval_dates[::21]:  # monthly-ish rebalance, matching TSMOM cadence
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

        alpha_full = (strat_ret.mean() - bh_ret.mean()) * 252 * 100
        by_year = {}
        for yr in [2022, 2023, 2024, 2025]:
            m = strat_ret.index.year == yr
            if m.sum() < 20:
                continue
            by_year[yr] = float((strat_ret[m].mean() - bh_ret[m].mean()) * 252 * 100)

        print(f"{tkr}: TSMOM alpha vs own buy-hold, 2022+ = {alpha_full:+.2f}%  "
              f"(Kelly-sized={OWN_METHOD_ALPHAS[tkr]['Kelly-sized']:+.2f}%, Master model={OWN_METHOD_ALPHAS[tkr]['Master model']:+.2f}%)")
        print(f"    by year: {by_year}")
        results[tkr] = {"tsmom_alpha_full_pct": float(alpha_full), "tsmom_by_year": by_year, **OWN_METHOD_ALPHAS[tkr]}

    with open("added_value_vs_tsmom_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(INSTRUMENTS))
    w_ = 0.25
    ax.bar(x - w_, [OWN_METHOD_ALPHAS[t]["Kelly-sized"] for t in INSTRUMENTS], width=w_, label="Kelly-sized (own method)", color="#2f8a4e")
    ax.bar(x, [OWN_METHOD_ALPHAS[t]["Master model"] for t in INSTRUMENTS], width=w_, label="Master model (own method)", color="#5B9E6F")
    ax.bar(x + w_, [results[t]["tsmom_alpha_full_pct"] for t in INSTRUMENTS], width=w_, label="Standard TSMOM (existing quant-shop tool)", color="#9AA1AD")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(INSTRUMENTS)
    ax.set_ylabel("Alpha vs. own buy-and-hold, 2022+ holdout (%/yr)")
    ax.set_title("Instruments where this program's own methods show convergent positive alpha:\nsame instrument, same holdout, standard TSMOM applied for direct comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig("added_value_vs_tsmom_plot.png", dpi=140)
    plt.close(fig)
    print("\nSaved: added_value_vs_tsmom_results.json, added_value_vs_tsmom_plot.png")
