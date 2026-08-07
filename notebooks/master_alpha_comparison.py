"""
master_alpha_comparison.py
============================
Pulls together every strategy this research program has tested for real,
economic (not just statistical-skill) alpha, onto one page, so it's
possible to actually answer "where do we stand as a quant shop" rather
than judging each result in isolation.

Two published, industry-standard, completely unfitted control strategies
(TSMOM, Moskowitz-Ooi-Pedersen 2012; XSMOM, Jegadeesh-Titman 1993) anchor
the comparison -- they're what "a real quant shop's basic toolkit" looks
like on this exact data, so every one of our own strategies is judged
against the same standard, not just against "no significance at all."

Everything is scored the same way: Jensen's alpha (OLS intercept vs. the
correct benchmark -- an instrument's own buy-and-hold for single-
instrument strategies, the same universe's passive combination for
portfolio strategies), annualized, with a t-statistic and a 95%
significance flag. No strategy here is re-tuned or cherry-picked for this
comparison; every number is pulled directly from each strategy's own
already-completed, already-reported test.

Honest limitation, stated up front rather than hidden in a footnote: the
evaluation windows are NOT perfectly aligned. The CPE portfolio-tilt
engine's thresholds are frozen at 2024-12-31 with a hardcoded 2025-01-01
to 2025-12-31 evaluation window (backtest_engine.py's own TRAIN_CUTOFF/
EVAL_START/EVAL_END) and was not re-run for this comparison to preserve
that frozen-threshold discipline; Paper 12 and the predictor_v1 RL sizing
test both use a 2022-01-01 holdout. TSMOM and XSMOM are reported at BOTH
windows directly, so at least one axis of the comparison is exactly
apples-to-apples in each panel; the rest are the best available honest
alignment, not a perfect one.

Run: python master_alpha_comparison.py
Output: master_alpha_comparison.png
"""
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tsmom_benchmark as _tm
import tsmom_crisis_alpha_check as _tmc  # reuse its build_series() (identical spec, no re-run/edit needed)


def jensen_alpha(strat_ret: pd.Series, bench_ret: pd.Series) -> dict:
    common = strat_ret.index.intersection(bench_ret.index)
    y = strat_ret.reindex(common).fillna(0.0).values
    x = bench_ret.reindex(common).fillna(0.0).values
    X = np.column_stack([np.ones_like(x), x])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_daily, beta = beta_hat
    resid = y - X @ beta_hat
    n, k = len(y), 2
    sigma2 = (resid @ resid) / max(n - k, 1)
    se_alpha = float(np.sqrt(max(sigma2 * np.linalg.inv(X.T @ X)[0, 0], 0)))
    t_alpha = alpha_daily / se_alpha if se_alpha > 0 else float("nan")
    ann_ret = float(y.mean() * 252 * 100)
    sharpe = float(y.mean() / y.std() * np.sqrt(252)) if y.std() > 0 else float("nan")
    return {"alpha_pct": float(alpha_daily * 252 * 100), "t_alpha": float(t_alpha),
            "sharpe": sharpe, "ann_ret_pct": ann_ret, "n": int(n),
            "significant": bool(abs(t_alpha) >= 1.96) if np.isfinite(t_alpha) else False}


def cpe_variant_stats(csv_path: str, strat_col: str, bench_col: str = "no_tilt") -> dict:
    eq = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    strat_ret = np.log(eq[strat_col]).diff()
    bench_ret = np.log(eq[bench_col]).diff()
    return jensen_alpha(strat_ret, bench_ret)


def per_instrument_summary(json_path: str, alpha_field: str, t_field: str, sig_field: str) -> dict:
    d = json.load(open(json_path))
    alphas, sig = [], []
    for tkr, row in d.items():
        if isinstance(row, dict) and alpha_field in row and row[alpha_field] is not None:
            a = row[alpha_field]
            if isinstance(a, (int, float)) and np.isfinite(a):
                alphas.append(a)
                sig.append(bool(row.get(sig_field, False)))
    alphas = np.array(alphas)
    return {
        "n_instruments": len(alphas), "median_alpha_pct": float(np.median(alphas)) if len(alphas) else float("nan"),
        "mean_alpha_pct": float(np.mean(alphas)) if len(alphas) else float("nan"),
        "p25_alpha_pct": float(np.percentile(alphas, 25)) if len(alphas) else float("nan"),
        "p75_alpha_pct": float(np.percentile(alphas, 75)) if len(alphas) else float("nan"),
        "n_significant_positive": int(sum(1 for a, s in zip(alphas, sig) if s and a > 0)),
        "n_significant_negative": int(sum(1 for a, s in zip(alphas, sig) if s and a < 0)),
    }


if __name__ == "__main__":
    rows_2025 = []   # (name, kind, alpha_pct, t_or_range, n, sig_summary)
    rows_2022 = []

    # -- CPE portfolio-tilt engine: fixed at 2025-01-01..2025-12-31 by its own frozen thresholds --
    static_orig = cpe_variant_stats("files/backtest_result_tau_aware_horizon_weights.csv", "static_tilt_original")
    static_tau = cpe_variant_stats("files/backtest_result_tau_aware_horizon_weights.csv", "static_tilt_tau_aware")
    hth_orig = cpe_variant_stats("files/backtest_result_tau_aware_hth.csv", "hth_original")
    hth_tau = cpe_variant_stats("files/backtest_result_tau_aware_hth.csv", "hth_tau_aware")
    for name, r in [("CPE static tilt", static_orig), ("CPE static tilt (tau*-aware)", static_tau),
                     ("CPE hold-to-horizon", hth_orig), ("CPE hold-to-horizon (tau*-aware)", hth_tau)]:
        rows_2025.append((name, "portfolio", r["alpha_pct"], f"t={r['t_alpha']:+.2f}", r["n"], r["significant"], r["sharpe"]))

    # -- TSMOM / XSMOM, both windows directly from their own already-computed sub-periods --
    tsmom = json.load(open("tsmom_benchmark_results.json"))
    xsmom = json.load(open("xsmom_benchmark_results.json"))
    for name, d in [("TSMOM (Moskowitz-Ooi-Pedersen)", tsmom), ("XSMOM (Jegadeesh-Titman)", xsmom)]:
        r25 = d.get("2025_only")
        if r25:
            rows_2025.append((name, "portfolio", r25["alpha_annualized_pct"], f"t={r25['t_alpha']:+.2f}",
                               r25["n_days"], r25["significant_95"], r25.get("tsmom_sharpe", r25.get("xsmom_sharpe"))))
        r22 = d.get("since_2022")
        if r22:
            rows_2022.append((name, "portfolio", r22["alpha_annualized_pct"], f"t={r22['t_alpha']:+.2f}",
                               r22["n_days"], r22["significant_95"], r22.get("tsmom_sharpe", r22.get("xsmom_sharpe"))))

    # tsmom_benchmark.py's own run never computed a since-2022 sub-period
    # (only since_2010/2020, 2025_only) -- add it here from the exact same
    # already-validated TSMOM return series, no re-run/edit of that script.
    tsmom_ret, tsmom_passive_ret = _tmc.build_series()
    mask_2022 = tsmom_ret.index >= pd.Timestamp("2022-01-01")
    r22_tsmom = jensen_alpha(tsmom_ret[mask_2022], tsmom_passive_ret[mask_2022])
    rows_2022.append(("TSMOM (Moskowitz-Ooi-Pedersen)", "portfolio", r22_tsmom["alpha_pct"],
                       f"t={r22_tsmom['t_alpha']:+.2f}", r22_tsmom["n"], r22_tsmom["significant"], r22_tsmom["sharpe"]))

    # -- predictor_v1 / Paper 12 + RL sizing: 2022-01-01 holdout, per-instrument --
    kelly = per_instrument_summary("predictor_v1/kelly_strategy_results.json", "alpha_vs_own_pct", "t_vs_own", "significant_vs_own_95")
    alpha_test = per_instrument_summary("predictor_v1/alpha_test_own_benchmark_results.json", "alpha_vs_own_benchmark_pct", "t_vs_own_benchmark", "significant_vs_own_95")
    rl = json.load(open("predictor_v1/81_rl_sizing_results.json"))
    rl_alphas, rl_sig = [], []
    for tkr, r in rl.items():
        if r.get("fresh"):
            rl_alphas.append(r["fresh"]["alpha_annualized_pct"])
            rl_sig.append(r["fresh"]["significant_95"])
    rl_summary = {
        "n_instruments": len(rl_alphas), "median_alpha_pct": float(np.median(rl_alphas)),
        "p25_alpha_pct": float(np.percentile(rl_alphas, 25)), "p75_alpha_pct": float(np.percentile(rl_alphas, 75)),
        "n_significant_positive": sum(1 for a, s in zip(rl_alphas, rl_sig) if s and a > 0),
        "n_significant_negative": sum(1 for a, s in zip(rl_alphas, rl_sig) if s and a < 0),
    }
    for name, r in [("Paper 12: Kelly-sized (22 instr.)", kelly),
                     ("Paper 12: master model (22 instr.)", alpha_test),
                     ("Predictability-aware RL sizing (12 instr.)", rl_summary)]:
        sig_note = f"{r['n_significant_positive']}+/{r['n_significant_negative']}- sig. of {r['n_instruments']}"
        rows_2022.append((name, "per-instrument", r["median_alpha_pct"],
                           f"IQR [{r['p25_alpha_pct']:+.1f}, {r['p75_alpha_pct']:+.1f}]", r["n_instruments"], sig_note, None))

    print("=" * 100)
    print("2025-ONLY COMPARISON (CPE portfolio-tilt engine's fixed evaluation window)")
    print("=" * 100)
    for name, kind, alpha, detail, n, sig, sharpe in sorted(rows_2025, key=lambda r: -r[2]):
        sharpe_str = f"Sharpe={sharpe:.2f}  " if sharpe is not None else ""
        print(f"  {name:<38} alpha={alpha:+7.2f}%/yr  {sharpe_str}{detail:<14} n={n:<6} "
              f"{'SIGNIFICANT' if sig is True else ('not significant' if sig is False else sig)}")

    print("\n" + "=" * 100)
    print("2022-01-01 ONWARD COMPARISON (Paper 12 / RL sizing holdout window)")
    print("=" * 100)
    for name, kind, alpha, detail, n, sig, sharpe in sorted(rows_2022, key=lambda r: -r[2]):
        sharpe_str = f"Sharpe={sharpe:.2f}  " if sharpe is not None else ""
        sig_str = sig if isinstance(sig, str) else ('SIGNIFICANT' if sig is True else 'not significant')
        print(f"  {name:<45} alpha={alpha:+7.2f}%/yr  {sharpe_str}{detail:<22} n={n:<6} {sig_str}")

    # -- combined chart --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))
    for ax, rows, title in [(ax1, rows_2025, "2025 only\n(CPE engine's fixed window)"),
                             (ax2, rows_2022, "2022-01-01 onward\n(Paper 12 / RL sizing holdout)")]:
        rows_sorted = sorted(rows, key=lambda r: r[2])
        names = [r[0] for r in rows_sorted]
        alphas = [r[2] for r in rows_sorted]
        is_control = ["TSMOM" in n or "XSMOM" in n for n in names]
        sig_flags = [r[5] is True for r in rows_sorted]
        colors = []
        for ic, sig, a in zip(is_control, sig_flags, alphas):
            if ic:
                colors.append("#C0392B" if a >= 0 else "#8B2E20")  # controls in red family
            elif sig and a > 0:
                colors.append("#2f8a4e")
            elif sig and a < 0:
                colors.append("#8B2E20")
            else:
                colors.append("#9AA1AD")
        y = np.arange(len(names))
        ax.barh(y, alphas, color=colors, height=0.6)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlabel("Annualized alpha vs. correct benchmark (%/yr)\n(portfolio: vs. same-universe passive | per-instrument: median vs. own buy-and-hold)")
        ax.set_title(title, fontsize=10)
    fig.suptitle("Every strategy this program has tested, one comparable scale --\n"
                  "red = published control strategy (TSMOM/XSMOM) or significant loss, green = significant real alpha, grey = not significant",
                  fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("master_alpha_comparison.png", dpi=140)
    plt.close(fig)
    print("\nSaved: master_alpha_comparison.png")
