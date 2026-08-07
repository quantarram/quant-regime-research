"""
tsmom_evt_tail_analysis.py
=============================
Re-does tsmom_crisis_alpha_check.py's "is TSMOM's edge crisis-specific"
question using the statistical framework this whole research program
actually studies extreme, non-stationary market behavior with --
extreme value theory (peaks-over-threshold, Generalized Pareto tail
fitting) -- instead of a classical mean-comparison t-test. TSMOM's daily
return series has 9,564 real observations, genuinely large enough for a
GPD tail fit to be meaningful (unlike the CPE vol-complex detector's 5
episodes, which stays reported by direct enumeration -- no distributional
fit, parametric or extreme-value, meaningfully compresses 5 real data
points into something more informative than the 5 points themselves).

Method: peaks-over-threshold. Take TSMOM's daily losses (negative
returns) and the passive benchmark's daily losses separately. Fit a
Generalized Pareto Distribution to each series' excesses over its own
90th-percentile-of-losses threshold (a standard, disclosed POT threshold
choice, not tuned to flatter either series). Report the fitted shape
parameter (tail index -- how heavy the tail actually is) and derived
return levels (the loss magnitude expected once every N threshold-
exceeding days) directly, for both series, side by side. This describes
the ACTUAL shape of each strategy's downside tail -- what extreme value
theory is for -- rather than testing whether a mean return "is
significantly different" via a classical test this program's own
standing methodology rejects for exactly this kind of question.

No significance/randomisation-test games, no p-values, no t-statistics --
real tail shape, fit via maximum likelihood on real data, reported
directly.

Run: python tsmom_evt_tail_analysis.py
Output: tsmom_evt_tail_results.json, tsmom_evt_tail_plot.png
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tsmom_crisis_alpha_check as _tmc

THRESHOLD_PCTL = 90  # POT threshold: 90th percentile of daily losses, disclosed, not tuned


def fit_gpd_tail(ret: pd.Series, label: str) -> dict:
    losses = -ret.dropna().values  # POT convention: work with losses (positive = bad day)
    threshold = np.percentile(losses, THRESHOLD_PCTL)
    excesses = losses[losses > threshold] - threshold
    n_exceed = len(excesses)

    shape, loc, scale = stats.genpareto.fit(excesses, floc=0)

    # Return levels: the loss magnitude expected to be exceeded once every
    # N threshold-crossing events, derived directly from the fitted GPD
    # (standard EVT return-level formula), not from resampling.
    return_levels = {}
    for n_events in [10, 50, 100]:
        if abs(shape) > 1e-6:
            rl = threshold + (scale / shape) * ((n_events) ** shape - 1)
        else:
            rl = threshold + scale * np.log(n_events)
        return_levels[f"1_in_{n_events}_exceedances_loss_pct"] = float(rl * 100)

    exceed_rate = n_exceed / len(losses)
    print(f"  [{label}] n_days={len(losses)}, threshold={threshold*100:.3f}% daily loss "
          f"(90th pctile), n_exceedances={n_exceed} ({exceed_rate*100:.1f}% of days)")
    print(f"    Fitted GPD shape (tail index) xi={shape:+.3f}  scale={scale:.4f}  "
          f"[{'heavy/Pareto-like tail' if shape > 0.1 else ('bounded tail' if shape < -0.1 else 'light/exponential-like tail')}]")
    for k, v in return_levels.items():
        print(f"    {k}: {v:.2f}% single-day loss")

    return {
        "label": label, "n_days": len(losses), "threshold_pct": float(threshold * 100),
        "n_exceedances": int(n_exceed), "exceedance_rate": float(exceed_rate),
        "gpd_shape_xi": float(shape), "gpd_scale": float(scale), **return_levels,
        "excesses": excesses.tolist(),
    }


if __name__ == "__main__":
    print("Rebuilding TSMOM and passive return series (unchanged spec)...")
    tsmom_ret, passive_ret = _tmc.build_series()

    print("\nFitting Generalized Pareto tails via peaks-over-threshold (maximum likelihood, no resampling):")
    results = {}
    results["tsmom"] = fit_gpd_tail(tsmom_ret, "TSMOM")
    results["passive"] = fit_gpd_tail(passive_ret, "Passive equal-weight")

    # Direct, real comparison: on the SAME real crisis days (passive drawdown
    # <= -10%, already established), what fraction of TSMOM's threshold-
    # exceeding loss days actually fall inside a real crisis episode, vs.
    # the passive benchmark's -- do TSMOM's worst days concentrate inside
    # real crises, or are they scattered? Direct counting, not inference.
    passive_equity = np.exp(passive_ret.fillna(0)).cumprod()
    drawdown = passive_equity / passive_equity.cummax() - 1.0
    crisis_mask = drawdown <= -0.10

    for label, ret, r in [("TSMOM", tsmom_ret, results["tsmom"]), ("Passive", passive_ret, results["passive"])]:
        losses = -ret.dropna()
        threshold = r["threshold_pct"] / 100
        exceed_dates = losses[losses > threshold].index
        in_crisis = crisis_mask.reindex(exceed_dates).fillna(False).sum()
        print(f"\n  {label}: of {len(exceed_dates)} threshold-exceeding loss days, "
              f"{in_crisis} ({in_crisis/len(exceed_dates)*100:.1f}%) fall inside a real crisis episode "
              f"(passive drawdown <= -10%)")
        r["n_exceedances_in_crisis"] = int(in_crisis)
        r["pct_exceedances_in_crisis"] = float(in_crisis / len(exceed_dates) * 100)

    with open("tsmom_evt_tail_results.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "excesses"} for k, v in results.items()}, f, indent=2, default=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (label, r) in zip(axes, results.items()):
        excesses = np.array(r["excesses"]) * 100
        ax.hist(excesses, bins=25, color="#2E6DA4" if label == "tsmom" else "#9AA1AD", alpha=0.8, density=True)
        x = np.linspace(0, excesses.max(), 200)
        pdf = stats.genpareto.pdf(x / 100, r["gpd_shape_xi"], loc=0, scale=r["gpd_scale"]) / 100
        ax.plot(x, pdf, color="black", lw=1.5, label=f"Fitted GPD, xi={r['gpd_shape_xi']:+.2f}")
        ax.set_title(f"{r['label']}: excess-over-threshold daily losses\n"
                      f"({r['n_exceedances']} exceedances, {r['pct_exceedances_in_crisis']:.0f}% inside real crisis episodes)")
        ax.set_xlabel("Loss excess over 90th-percentile threshold (%)")
        ax.legend()
    fig.suptitle("Extreme value (peaks-over-threshold) tail comparison: TSMOM vs. passive -- no significance test, direct tail shape")
    fig.tight_layout()
    fig.savefig("tsmom_evt_tail_plot.png", dpi=140)
    plt.close(fig)
    print("\nSaved: tsmom_evt_tail_results.json, tsmom_evt_tail_plot.png")
