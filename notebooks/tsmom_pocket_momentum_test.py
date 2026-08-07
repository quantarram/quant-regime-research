"""
tsmom_pocket_momentum_test.py
================================
Extends the corrected pocket-vs-momentum check (does an instrument's own
documented predictability POCKET structure near 252 days, Ramanathan
2026a Section 5.3, correlate with how well standard 12-month momentum
works for it) to all 12 instruments with real confidence intervals, not
just point-estimate Sharpe ratios.

Per instrument: Jensen's alpha of the standard TSMOM signal (252d
lookback, 63d vol window, unchanged from tonight's baseline) against that
SAME instrument's own buy-and-hold, with a Newey-West HAC-corrected
standard error and 95% CI -- analytic, not resampling/bootstrap-based
(this program's standing rule: no randomization-test shortcuts; real
OOS return series, analytic correction only, the same discipline as
every HAC test run tonight).

Group comparison (HAS a genuine 200-300d pocket per Ramanathan 2026a's
own top5_tradeable lists at q=2 or q=4, vs. does not): a two-sample
Welch's t-test on the two groups' alpha estimates -- a classic, closed-
form parametric test, not a permutation/bootstrap significance game --
with n=5 vs n=7, small enough that the result should be read as
suggestive, not confirmatory, and is reported as such.

Run: python tsmom_pocket_momentum_test.py
Output: tsmom_pocket_momentum_results.json, tsmom_pocket_momentum_ci.png
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INSTR = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "AAPL", "MSFT", "JPM", "XOM", "GLD", "EURUSD=X"]
HAS_POCKET = {"SPY", "IWM", "AAPL", "MSFT", "GLD"}  # confirmed directly from results_correlated_decorrelated.json
LOOKBACK, VOL_WINDOW, TARGET_VOL, MAX_LEV, COST_BPS = 252, 63, 0.10, 2.0, 5


def month_end_dates(index):
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).last().values)


def hac_alpha_ci(strat_ret: pd.Series, bench_ret: pd.Series, maxlags: int) -> dict:
    common = strat_ret.index.intersection(bench_ret.index)
    y = strat_ret.reindex(common).fillna(0.0).values
    x = bench_ret.reindex(common).fillna(0.0).values
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    alpha_daily, beta = model.params
    se_daily = model.bse[0]
    alpha_ann, se_ann = alpha_daily * 252 * 100, se_daily * 252 * 100
    return {
        "alpha_pct": float(alpha_ann), "se_pct": float(se_ann),
        "ci_lo": float(alpha_ann - 1.96 * se_ann), "ci_hi": float(alpha_ann + 1.96 * se_ann),
        "t": float(model.tvalues[0]), "beta": float(beta), "n": int(len(y)),
        "significant_95": bool(abs(model.tvalues[0]) >= 1.96),
    }


def sharpe_with_ci(ret: pd.Series) -> dict:
    r = ret.dropna()
    sh = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")
    # Lo (2002) asymptotic Sharpe SE for iid returns, annualized: SE(SR_ann) ~= sqrt((1+0.5*SR_daily^2)/T) * sqrt(252)
    sr_daily = r.mean() / r.std() if r.std() > 0 else float("nan")
    se = float(np.sqrt((1 + 0.5 * sr_daily ** 2) / len(r)) * np.sqrt(252)) if np.isfinite(sr_daily) else float("nan")
    return {"sharpe": sh, "se": se, "ci_lo": sh - 1.96 * se, "ci_hi": sh + 1.96 * se, "n": len(r)}


if __name__ == "__main__":
    prices = pd.read_parquet("multiasset_prices.parquet")
    daily_index = prices.index
    first_valid = min(prices[t].dropna().index.min() for t in INSTR)
    start = first_valid + pd.Timedelta(days=int(1.5 * (LOOKBACK + VOL_WINDOW)))
    rebal = month_end_dates(daily_index[daily_index >= start])
    trading_index = daily_index[daily_index >= rebal[0]]
    log_px = np.log(prices[INSTR])
    daily_ret = log_px.diff()

    results = {}
    for tkr in INSTR:
        rows = {}
        for d in rebal:
            s = prices[tkr].loc[:d].dropna()
            if len(s) < LOOKBACK + VOL_WINDOW + 5:
                rows[d] = np.nan
                continue
            trail = np.log(s.iloc[-1] / s.iloc[-LOOKBACK])
            r = daily_ret[tkr].dropna().loc[:d].tail(VOL_WINDOW)
            vol = r.std() * np.sqrt(252)
            if not np.isfinite(vol) or vol <= 1e-6:
                rows[d] = np.nan
                continue
            rows[d] = float(np.clip(np.sign(trail) * (TARGET_VOL / vol), -MAX_LEV, MAX_LEV))
        w = pd.Series(rows).reindex(trading_index, method="ffill").shift(1)
        strat_ret = w * daily_ret[tkr].reindex(trading_index)
        turnover = w.diff().abs().fillna(0.0)
        strat_ret = strat_ret - turnover * (COST_BPS / 10000.0)
        bench_ret = daily_ret[tkr].reindex(trading_index)

        alpha_ci = hac_alpha_ci(strat_ret, bench_ret, maxlags=63)
        sh_ci = sharpe_with_ci(strat_ret)
        results[tkr] = {"pocket": tkr in HAS_POCKET, **alpha_ci, "sharpe": sh_ci}

    print(f"{'Ticker':<10}{'Group':<18}{'Sharpe [95% CI]':<28}{'Alpha vs own B&H [95% CI]':<36}{'Sig?'}")
    for tkr in sorted(results, key=lambda t: -results[t]["sharpe"]["sharpe"]):
        r = results[tkr]
        grp = "HAS pocket" if r["pocket"] else "no pocket"
        sh, sh_lo, sh_hi = r["sharpe"]["sharpe"], r["sharpe"]["ci_lo"], r["sharpe"]["ci_hi"]
        a, a_lo, a_hi = r["alpha_pct"], r["ci_lo"], r["ci_hi"]
        sig = "SIG" if r["significant_95"] else ""
        print(f"{tkr:<10}{grp:<18}{f'{sh:+.2f} [{sh_lo:+.2f}, {sh_hi:+.2f}]':<28}"
              f"{f'{a:+.2f}% [{a_lo:+.2f}, {a_hi:+.2f}]':<36}{sig}")

    pocket_alphas = np.array([results[t]["alpha_pct"] for t in INSTR if t in HAS_POCKET])
    nopocket_alphas = np.array([results[t]["alpha_pct"] for t in INSTR if t not in HAS_POCKET])
    pocket_sharpes = np.array([results[t]["sharpe"]["sharpe"] for t in INSTR if t in HAS_POCKET])
    nopocket_sharpes = np.array([results[t]["sharpe"]["sharpe"] for t in INSTR if t not in HAS_POCKET])

    t_stat, p_val = stats.ttest_ind(pocket_alphas, nopocket_alphas, equal_var=False)
    diff = pocket_alphas.mean() - nopocket_alphas.mean()
    se_diff = np.sqrt(pocket_alphas.var(ddof=1) / len(pocket_alphas) + nopocket_alphas.var(ddof=1) / len(nopocket_alphas))
    dof = (pocket_alphas.var(ddof=1) / len(pocket_alphas) + nopocket_alphas.var(ddof=1) / len(nopocket_alphas)) ** 2 / (
        (pocket_alphas.var(ddof=1) / len(pocket_alphas)) ** 2 / (len(pocket_alphas) - 1)
        + (nopocket_alphas.var(ddof=1) / len(nopocket_alphas)) ** 2 / (len(nopocket_alphas) - 1)
    )
    tcrit = stats.t.ppf(0.975, dof)
    ci_lo, ci_hi = diff - tcrit * se_diff, diff + tcrit * se_diff

    print(f"\n{'='*80}\nGroup comparison (Welch's t-test, analytic, n=5 vs n=7 -- NOT a resampling/permutation test)\n{'='*80}")
    print(f"  HAS pocket (n={len(pocket_alphas)}): mean alpha={pocket_alphas.mean():+.2f}%/yr, "
          f"mean Sharpe={pocket_sharpes.mean():+.3f}")
    print(f"  no pocket  (n={len(nopocket_alphas)}): mean alpha={nopocket_alphas.mean():+.2f}%/yr, "
          f"mean Sharpe={nopocket_sharpes.mean():+.3f}")
    print(f"  Difference in mean alpha: {diff:+.2f}%/yr, 95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}], "
          f"t={t_stat:+.2f}, p={p_val:.3f}, dof={dof:.1f}")
    print(f"  {'SIGNIFICANT at 95% (CI excludes 0)' if ci_lo > 0 or ci_hi < 0 else 'NOT significant -- 95% CI includes 0, small-sample result, suggestive only'}")

    results["group_comparison"] = {
        "pocket_mean_alpha_pct": float(pocket_alphas.mean()), "nopocket_mean_alpha_pct": float(nopocket_alphas.mean()),
        "diff_pct": float(diff), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "t_stat": float(t_stat), "p_value": float(p_val), "dof": float(dof),
        "significant_95": bool(ci_lo > 0 or ci_hi < 0),
    }
    with open("tsmom_pocket_momentum_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    order = sorted(INSTR, key=lambda t: results[t]["alpha_pct"])
    colors = ["#2f8a4e" if t in HAS_POCKET else "#9AA1AD" for t in order]
    y = np.arange(len(order))
    alphas = [results[t]["alpha_pct"] for t in order]
    los = [results[t]["ci_lo"] for t in order]
    his = [results[t]["ci_hi"] for t in order]
    ax1.barh(y, alphas, color=colors, height=0.6, zorder=3)
    for i, (lo, hi) in enumerate(zip(los, his)):
        ax1.plot([lo, hi], [i, i], color="black", lw=1.0, zorder=4)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_yticks(y); ax1.set_yticklabels(order, fontsize=9)
    ax1.set_xlabel("Momentum alpha vs. own buy-and-hold, %/yr (95% HAC CI)")
    ax1.set_title("Per-instrument, green = has a real ~252d pocket (Ramanathan 2026a)")

    ax2.bar(["HAS pocket\n(n=5)", "no pocket\n(n=7)"], [pocket_alphas.mean(), nopocket_alphas.mean()],
            color=["#2f8a4e", "#9AA1AD"], width=0.5, zorder=3)
    ax2.errorbar([0], [pocket_alphas.mean()], yerr=[[pocket_alphas.mean() - (pocket_alphas.mean() - 1.96*pocket_alphas.std(ddof=1)/np.sqrt(len(pocket_alphas)))],
                                                       [1.96*pocket_alphas.std(ddof=1)/np.sqrt(len(pocket_alphas))]], color="black", capsize=6, zorder=4)
    ax2.errorbar([1], [nopocket_alphas.mean()], yerr=[[nopocket_alphas.mean() - (nopocket_alphas.mean() - 1.96*nopocket_alphas.std(ddof=1)/np.sqrt(len(nopocket_alphas)))],
                                                          [1.96*nopocket_alphas.std(ddof=1)/np.sqrt(len(nopocket_alphas))]], color="black", capsize=6, zorder=4)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Mean momentum alpha, %/yr (95% CI on the group mean)")
    ax2.set_title(f"Group difference: {diff:+.2f}%/yr, 95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]\n"
                   f"p={p_val:.2f} (Welch's t-test, n=5 vs n=7 -- suggestive, not confirmatory)")
    fig.tight_layout()
    fig.savefig("tsmom_pocket_momentum_ci.png", dpi=140)
    plt.close(fig)
    print("\nSaved: tsmom_pocket_momentum_results.json, tsmom_pocket_momentum_ci.png")
