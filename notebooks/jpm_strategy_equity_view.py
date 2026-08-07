"""
jpm_strategy_equity_view.py
=============================
Equity-curve view of JPM specifically -- the single strongest convergent-
signal case in Section 5.3 (positive across all three independently-built
own-signal methods) -- shown against standard TSMOM applied to the same
instrument and period, and against plain buy-and-hold.

The Kelly-sized own-signal series is loaded directly from
predictor_v1/kelly_strategy_returns.parquet (JPM column), the exact same
daily return series that produces the cited 11.64%/yr Jensen's alpha in
predictor_v1/kelly_strategy_results.json and predictor_v1/47_kelly_sized_
strategy.py's own _KELLY_JPM.png -- reused directly, not re-derived, so
there is no risk of a second benchmark-formula mismatch. TSMOM's JPM
series is rebuilt with the exact same unfitted spec used throughout this
paper (252d lookback, 63d vol window, 10% vol target, 2x leverage cap).

Reports both the raw cumulative-return picture AND the beta-adjusted
Jensen's alpha explicitly in the title, so the chart cannot be misread as
"3x the return" without the leverage context that actually produces most
of that gap -- the true, beta-adjusted excess return is the smaller,
already-cited 11.64%/yr number.

Run: python jpm_strategy_equity_view.py
Output: jpm_strategy_equity_view.png
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOOKBACK, VOL_WINDOW, TARGET_VOL, MAX_LEV, COST_BPS = 252, 63, 0.10, 2.0, 5
HOLDOUT_START = pd.Timestamp("2022-01-01")


def jensen_alpha(y_series, x_series):
    common = y_series.index.intersection(x_series.index)
    y, x = y_series.reindex(common).values, x_series.reindex(common).values
    X = np.column_stack([np.ones_like(x), x])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_daily, beta = beta_hat
    resid = y - X @ beta_hat
    n, k = len(y), 2
    sigma2 = (resid @ resid) / max(n - k, 1)
    se_alpha = float(np.sqrt(max(sigma2 * np.linalg.inv(X.T @ X)[0, 0], 0)))
    t_alpha = alpha_daily / se_alpha if se_alpha > 0 else float("nan")
    return float(alpha_daily * 252 * 100), float(t_alpha), float(beta)


if __name__ == "__main__":
    kelly = pd.read_parquet("predictor_v1/kelly_strategy_returns.parquet")
    jpm_kelly = kelly["JPM"].dropna()  # simple daily returns, exact series behind the cited 11.64% figure

    prices = pd.read_parquet("multiasset_prices.parquet")
    jpm_series = prices["JPM"].dropna()
    jpm_bh = jpm_series.pct_change(fill_method=None).reindex(jpm_kelly.index)

    # Standard TSMOM on JPM alone, identical spec used throughout this paper
    daily_ret = np.log(jpm_series).diff()
    eval_dates = jpm_series.index[jpm_series.index >= HOLDOUT_START]
    rows = {}
    for d in eval_dates[::21]:
        hist = jpm_series.loc[:d]
        if len(hist) < LOOKBACK + VOL_WINDOW + 5:
            rows[d] = np.nan
            continue
        trail = np.log(hist.iloc[-1] / hist.iloc[-LOOKBACK])
        r = daily_ret.loc[:d].tail(VOL_WINDOW)
        vol = r.std() * np.sqrt(252)
        rows[d] = float(np.clip(np.sign(trail) * (TARGET_VOL / vol), -MAX_LEV, MAX_LEV)) if vol > 1e-6 else np.nan
    w = pd.Series(rows).reindex(eval_dates, method="ffill").shift(1)
    tsmom_log_ret = w * daily_ret.reindex(eval_dates)
    turnover = w.diff().abs().fillna(0.0)
    tsmom_log_ret = (tsmom_log_ret - turnover * (COST_BPS / 10000.0)).dropna()
    tsmom_simple_ret = (np.exp(tsmom_log_ret) - 1).reindex(jpm_kelly.index)

    a_kelly, t_kelly, b_kelly = jensen_alpha(jpm_kelly, jpm_bh)
    tsmom_common = tsmom_simple_ret.dropna()
    bh_for_tsmom = jpm_bh.reindex(tsmom_common.index)
    a_tsmom, t_tsmom, b_tsmom = jensen_alpha(tsmom_common, bh_for_tsmom)

    print(f"JPM Kelly-sized: Jensen alpha={a_kelly:+.2f}%/yr (t={t_kelly:+.2f}, beta={b_kelly:.2f}) "
          f"vs cited 11.64%/yr -- reused series, should match exactly")
    print(f"JPM standard TSMOM: Jensen alpha={a_tsmom:+.2f}%/yr (t={t_tsmom:+.2f}, beta={b_tsmom:.2f})")

    fig, ax = plt.subplots(figsize=(11, 6))
    bh_cum = (1 + jpm_bh.fillna(0)).cumprod()
    kelly_cum = (1 + jpm_kelly).cumprod()
    tsmom_cum = (1 + tsmom_simple_ret.fillna(0)).cumprod()
    ax.plot(bh_cum.index, (bh_cum - 1) * 100, color="black", lw=1.2, label="JPM buy & hold")
    ax.plot(tsmom_cum.index, (tsmom_cum - 1) * 100, color="#B0492F", lw=1.2, label=f"Standard TSMOM on JPM (Jensen alpha {a_tsmom:+.1f}%/yr)")
    ax.plot(kelly_cum.index, (kelly_cum - 1) * 100, color="#2f8a4e", lw=1.4, label=f"Kelly-sized own-signal method (Jensen alpha {a_kelly:+.1f}%/yr)")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("JPM: own-signal method vs. standard TSMOM vs. buy-and-hold, 2022+ holdout\n"
                  f"Own-signal method runs ~2x average leverage when active (beta={b_kelly:.2f} vs. own buy-hold) -- "
                  "the beta-adjusted Jensen's alpha, not raw cumulative return, is the honest comparison")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig("jpm_strategy_equity_view.png", dpi=140)
    plt.close(fig)
    print("\nSaved: jpm_strategy_equity_view.png")
