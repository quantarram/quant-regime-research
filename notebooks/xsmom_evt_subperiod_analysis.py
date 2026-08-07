"""
xsmom_evt_subperiod_analysis.py
==================================
Gives XSMOM the same treatment TSMOM and SPY momentum just got --
extreme-value tail comparison (peaks-over-threshold GPD fit) and real
non-overlapping sub-period replication -- rather than the HAC t-test
framing xsmom_benchmark.py originally used. Completes a fair, identical
treatment of both published control strategies before comparing anything
else against them.

No significance/randomisation-test games, no p-values, no t-statistics --
real tail shape and real historical-block replication, reported directly.

Run: python xsmom_evt_subperiod_analysis.py
Output: xsmom_evt_subperiod_results.json, xsmom_evt_subperiod_plot.png
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xsmom_benchmark as _xm
from tsmom_benchmark import month_end_dates

THRESHOLD_PCTL = 90
N_BLOCKS = 5


def sharpe(ret):
    r = ret.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")


def fit_gpd_tail(ret: pd.Series, label: str) -> dict:
    losses = -ret.dropna().values
    threshold = np.percentile(losses, THRESHOLD_PCTL)
    excesses = losses[losses > threshold] - threshold
    shape, loc, scale = stats.genpareto.fit(excesses, floc=0)
    rl100 = threshold + (scale / shape) * (100 ** shape - 1) if abs(shape) > 1e-6 else threshold + scale * np.log(100)
    print(f"  [{label}] n_exceedances={len(excesses)}, GPD shape xi={shape:+.3f}, "
          f"1-in-100 loss={rl100*100:.2f}%")
    return {"label": label, "n_exceedances": len(excesses), "gpd_shape_xi": float(shape),
            "gpd_scale": float(scale), "return_level_1_in_100_pct": float(rl100 * 100),
            "excesses": excesses.tolist()}


if __name__ == "__main__":
    prices = pd.read_parquet("multiasset_prices.parquet")
    first_valid = min(prices[t].dropna().index.min() for t in _xm.UNIVERSE)
    start = first_valid + pd.Timedelta(days=int(1.5 * _xm.MIN_HISTORY_DAYS))
    rebal_dates = month_end_dates(prices.index[prices.index >= start])
    weight_df = _xm.build_monthly_positions(prices, rebal_dates)
    trading_index = prices.index[prices.index >= rebal_dates[0]]
    xsmom_ret = _xm.simulate_xsmom(weight_df, prices, trading_index)
    passive_ret = _xm.simulate_passive(prices, trading_index, weight_df)

    print("Fitting Generalized Pareto tails via peaks-over-threshold:")
    tail_results = {"xsmom": fit_gpd_tail(xsmom_ret, "XSMOM"), "passive": fit_gpd_tail(passive_ret, "Passive (equity universe)")}

    print(f"\nXSMOM full sample: Sharpe={sharpe(xsmom_ret):.3f}, ann_ret={xsmom_ret.mean()*252*100:+.2f}%")
    print(f"Passive full sample: Sharpe={sharpe(passive_ret):.3f}, ann_ret={passive_ret.mean()*252*100:+.2f}%")

    block_edges = pd.date_range(trading_index.min(), trading_index.max(), periods=N_BLOCKS + 1)
    block_results = []
    print(f"\n{N_BLOCKS} non-overlapping real historical blocks:")
    for i in range(N_BLOCKS):
        b_start, b_end = block_edges[i], block_edges[i + 1]
        mask = (trading_index >= b_start) & (trading_index < b_end if i < N_BLOCKS - 1 else trading_index <= b_end)
        idx = trading_index[mask]
        x_block, p_block = xsmom_ret.reindex(idx), passive_ret.reindex(idx)
        excess = (x_block.mean() - p_block.mean()) * 252 * 100
        print(f"  {b_start.date()} to {b_end.date()}: XSMOM ann_ret={x_block.mean()*252*100:+6.2f}%  |  "
              f"Passive ann_ret={p_block.mean()*252*100:+6.2f}%  |  excess={excess:+6.2f}%/yr")
        block_results.append({"block_start": str(b_start.date()), "block_end": str(b_end.date()),
                               "xsmom_ann_ret_pct": float(x_block.mean() * 252 * 100),
                               "passive_ann_ret_pct": float(p_block.mean() * 252 * 100),
                               "excess_pct": float(excess)})
    n_positive = sum(1 for r in block_results if r["excess_pct"] > 0)
    print(f"\nXSMOM beat passive in {n_positive} of {N_BLOCKS} real, non-overlapping historical blocks")

    with open("xsmom_evt_subperiod_results.json", "w") as f:
        json.dump({"tail": {k: {kk: vv for kk, vv in v.items() if kk != "excesses"} for k, v in tail_results.items()},
                    "blocks": block_results, "n_blocks_positive": n_positive}, f, indent=2, default=float)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    labels_short = [f"{r['block_start'][:4]}-{r['block_end'][:4]}" for r in block_results]
    x = np.arange(N_BLOCKS)
    axes[0].bar(x - 0.2, [r["xsmom_ann_ret_pct"] for r in block_results], width=0.4, label="XSMOM", color="#2E6DA4")
    axes[0].bar(x + 0.2, [r["passive_ann_ret_pct"] for r in block_results], width=0.4, label="Passive", color="#9AA1AD")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels_short)
    axes[0].set_ylabel("Annualized return (%)")
    axes[0].set_title(f"XSMOM beat passive in {n_positive}/{N_BLOCKS} real blocks")
    axes[0].legend()

    exc_x = np.array(tail_results["xsmom"]["excesses"]) * 100
    exc_p = np.array(tail_results["passive"]["excesses"]) * 100
    axes[1].hist(exc_p, bins=25, color="#9AA1AD", alpha=0.6, density=True, label="Passive")
    axes[1].hist(exc_x, bins=25, color="#2E6DA4", alpha=0.6, density=True, label="XSMOM")
    axes[1].set_title(f"Tail excess: XSMOM xi={tail_results['xsmom']['gpd_shape_xi']:+.2f} vs "
                        f"Passive xi={tail_results['passive']['gpd_shape_xi']:+.2f}")
    axes[1].set_xlabel("Loss excess over 90th-pctile threshold (%)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("xsmom_evt_subperiod_plot.png", dpi=140)
    plt.close(fig)
    print("\nSaved: xsmom_evt_subperiod_results.json, xsmom_evt_subperiod_plot.png")
