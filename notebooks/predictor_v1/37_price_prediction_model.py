"""
Final deliverable: an actual raw-price predictive model per instrument,
built directly from the FSS re-scoring work. For each instrument, find the
single (horizon, variant) combination with the highest skill above
climatology across the whole grid already computed (best_config_per_instrument.json,
derived from results_fss_true_unified.json), refit ONLY that winning
combination's walk-forward LightGBM quantile model (same methodology as
32_fss_true_unified.py, just for one variant instead of all four), and
convert its predicted return-quantiles directly into a predicted PRICE:
    predicted_price(t+H) = price(t) * exp(predicted_log_return_quantile(t))
Plotted as actual price vs. predicted median price with an 80% band
(q0.1-q0.9), at the target date the prediction is FOR (t+H), over the full
walk-forward OOS period.

Honesty check built in: instruments whose best skill above climatology is
near zero (SPY, XLV) get flagged in the plot title -- their "best" config
is noise-level, not a real edge, and the predicted band should not be
trusted more than climatology would be.

Run: python 37_price_prediction_model.py
Output: price_predictions/<TICKER>_price_prediction.png, results_price_predictions.json
"""
import pandas as pd
import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUT_DIR, "price_predictions")
os.makedirs(PLOT_DIR, exist_ok=True)

INITIAL_TRAIN_YEARS = 6
STEP_YEARS = 1
ALPHAS = [0.1, 0.25, 0.5, 0.75, 0.9]
LGB_BASE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
WEAK_SKILL_THRESHOLD = 0.01  # below this, flag the instrument as noise-level, not a real edge

t0 = time.time()
print("=" * 60)
print("  PRICE PREDICTION MODEL: best (horizon, variant) per instrument")
print("=" * 60)

best_config = json.load(open(os.path.join(OUT_DIR, "best_config_per_instrument.json")))

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))
date_pos_main = {d: i for i, d in enumerate(prices.index)}
date_pos_proxy = {d: i for i, d in enumerate(prices_proxy.index)}

macro_existing = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro_existing["date"] = pd.to_datetime(macro_existing["date"])
credit_spread_regime = macro_existing[["date", "credit_spread_regime"]].drop_duplicates("date").set_index("date")["credit_spread_regime"]

vixy = prices["VIXY"].dropna()
vixm = prices["VIXM"].dropna()
common = vixy.index.intersection(vixm.index)
raw_ratio = np.log(vixm.reindex(common) / vixy.reindex(common))
vix_term_slope = ((raw_ratio - raw_ratio.rolling(200, min_periods=100).mean())
                   / raw_ratio.rolling(200, min_periods=100).std()).rename("vix_term_slope")
regimes = pd.concat([credit_spread_regime, vix_term_slope], axis=1).reset_index().rename(columns={"index": "date"})

orig_baseline = pd.read_parquet(os.path.join(OUT_DIR, "features_daily_panel.parquet"))
orig_baseline["date"] = pd.to_datetime(orig_baseline["date"])
ORIG_TICKERS = sorted(orig_baseline["ticker"].unique())

new_baseline_raw = pd.read_parquet(os.path.join(OUT_DIR, "features_new_tickers_baseline_cache.parquet"))
new_baseline_raw["date"] = pd.to_datetime(new_baseline_raw["date"])
NEW_TICKERS = sorted(new_baseline_raw["ticker"].unique())
zscore_cols = ["alpha", "C1", "H", "xi_q2", "xi_q4", "gap_tau1_q2", "gap_tau1_q4",
               "gap_tau5_q2", "gap_tau5_q4", "gap_tau21_q2", "gap_tau21_q4"]


def zscore_group(g):
    for c in zscore_cols:
        mu, sd = g[c].mean(), g[c].std()
        g[c + "_z"] = (g[c] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
    return g


new_baseline = new_baseline_raw.groupby("date", group_keys=False).apply(zscore_group)

ticker_registry = {}
for tkr in ORIG_TICKERS:
    df = orig_baseline[orig_baseline["ticker"] == tkr].merge(regimes, on="date", how="left")
    baseline_cols = [c for c in orig_baseline.columns if c.endswith("_z")] + \
        [c for c in orig_baseline.columns if c.startswith("ctx_")] + ["self_ref_score"]
    ticker_registry[tkr] = {"df": df, "baseline_cols": baseline_cols, "series": prices[tkr].dropna(),
                             "date_pos": date_pos_main}
for tkr in NEW_TICKERS:
    df = new_baseline[new_baseline["ticker"] == tkr].merge(regimes, on="date", how="left")
    baseline_cols = [c for c in new_baseline.columns if c.endswith("_z")]
    is_proxy = tkr in ("IYR", "VOX")
    ticker_registry[tkr] = {
        "df": df, "baseline_cols": baseline_cols,
        "series": (prices_proxy[tkr] if is_proxy else prices[tkr]).dropna(),
        "date_pos": date_pos_proxy if is_proxy else date_pos_main,
    }

for tkr, reg in ticker_registry.items():
    df = reg["df"]
    df["interact_gap21q4_credit"] = df["gap_tau21_q4_z"] * df["credit_spread_regime"]
    df["interact_xiq4_credit"] = df["xi_q4_z"] * df["credit_spread_regime"]
    df["interact_gap21q4_vix"] = df["gap_tau21_q4_z"] * df["vix_term_slope"]
    df["interact_xiq4_vix"] = df["xi_q4_z"] * df["vix_term_slope"]

CREDIT_TERMS = ["interact_gap21q4_credit", "interact_xiq4_credit", "credit_spread_regime"]
VIX_TERMS = ["interact_gap21q4_vix", "interact_xiq4_vix", "vix_term_slope"]
VARIANT_EXTRA_COLS = {"credit_only": CREDIT_TERMS, "vix_only": VIX_TERMS, "both": CREDIT_TERMS + VIX_TERMS}


def fit_and_predict(tkr, horizon, variant):
    """Returns OOS predictions for BOTH the winning variant (LightGBM) and
    climatology (fixed day-of-year reference, same construction as
    32_fss_true_unified.py), computed on the identical walk-forward folds
    and identical test dates -- so the two are directly comparable on the
    same plot, not just asserted to be comparable."""
    reg = ticker_registry[tkr]
    series, date_pos = reg["series"], reg["date_pos"]
    series_pos = {d: i for i, d in enumerate(series.index)}  # this ticker's OWN index positions,
    # not the shared multi-ticker date_pos -- needed for an exact (not calendar-approximated) target date
    fwd_ret = np.log(series.shift(-horizon) / series).rename("fwd_ret")
    d_full = reg["df"].merge(fwd_ret, left_on="date", right_index=True, how="left")
    d_full["pos"] = d_full["date"].map(date_pos)
    d_full = d_full.dropna(subset=["pos"]).copy()
    d_full["pos"] = d_full["pos"].astype(int)

    feature_cols = reg["baseline_cols"] + VARIANT_EXTRA_COLS[variant]
    dd = d_full.dropna(subset=feature_cols + ["fwd_ret"]).reset_index(drop=True)

    min_year, max_year = dd["date"].dt.year.min(), dd["date"].dt.year.max()
    first_test_year = min_year + INITIAL_TRAIN_YEARS
    first_test_start = pd.Timestamp(f"{first_test_year}-01-01")
    cands0 = [p for dt, p in date_pos.items() if dt >= first_test_start]
    if not cands0:
        return None
    first_test_start_pos = min(cands0)
    initial_train_mask = (dd["date"] < first_test_start) & (dd["pos"] + horizon < first_test_start_pos)
    if initial_train_mask.sum() < 150:
        return None
    initial_train = dd.loc[initial_train_mask, ["date", "fwd_ret"]].copy()
    initial_train["mmdd"] = list(zip(initial_train["date"].dt.month, initial_train["date"].dt.day))
    pooled_vals = initial_train["fwd_ret"].values
    climatology_quantiles_by_day = {}
    for m in range(1, 13):
        days_in_month = 29 if m == 2 else (30 if m in (4, 6, 9, 11) else 31)
        for da in range(1, days_in_month + 1):
            vals = initial_train.loc[initial_train["mmdd"] == (m, da), "fwd_ret"].values
            if len(vals) == 0:
                vals = pooled_vals
            climatology_quantiles_by_day[(m, da)] = {a: float(np.quantile(vals, a)) for a in ALPHAS}

    oos_rows = []
    test_year = first_test_year
    while test_year <= max_year:
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year + STEP_YEARS}-01-01")
        test_mask = (dd["date"] >= test_start) & (dd["date"] < test_end)
        if test_mask.sum() < 20:
            test_year += STEP_YEARS
            continue
        cands = [p for dt, p in date_pos.items() if dt >= test_start]
        if not cands:
            test_year += STEP_YEARS
            continue
        test_start_pos = min(cands)
        train_mask = (dd["date"] < test_start) & (dd["pos"] + horizon < test_start_pos)
        if train_mask.sum() < 200:
            test_year += STEP_YEARS
            continue

        Xtr, ytr = dd.loc[train_mask, feature_cols], dd.loc[train_mask, "fwd_ret"]
        Xte, yte = dd.loc[test_mask, feature_cols], dd.loc[test_mask, "fwd_ret"]
        row = dd.loc[test_mask, ["date"]].copy()
        row["y_true"] = yte.values
        row["price_now"] = series.reindex(row["date"]).values
        for a in ALPHAS:
            m = lgb.LGBMRegressor(**LGB_BASE, objective="quantile", alpha=a)
            m.fit(Xtr, ytr)
            row[f"q{a}"] = m.predict(Xte)
        row_mmdd = list(zip(row["date"].dt.month, row["date"].dt.day))
        for a in ALPHAS:
            row[f"clim_q{a}"] = [climatology_quantiles_by_day[k][a] for k in row_mmdd]
        oos_rows.append(row)
        test_year += STEP_YEARS

    if not oos_rows:
        return None
    oos = pd.concat(oos_rows, ignore_index=True).sort_values("date").reset_index(drop=True)
    target_idx = (oos["date"].map(series_pos) + horizon).astype(int)
    oos["target_date"] = series.index[target_idx.values]
    oos["actual_price_target"] = oos["price_now"] * np.exp(oos["y_true"])
    for a in ALPHAS:
        oos[f"pred_price_q{a}"] = oos["price_now"] * np.exp(oos[f"q{a}"])
        oos[f"clim_price_q{a}"] = oos["price_now"] * np.exp(oos[f"clim_q{a}"])
    return oos


results_summary = {}
for tkr in sorted(best_config.keys(), key=lambda t: -best_config[t]["skill"]):
    cfg = best_config[tkr]
    horizon, variant, skill = cfg["horizon"], cfg["variant"], cfg["skill"]
    oos = fit_and_predict(tkr, horizon, variant)
    if oos is None or len(oos) < 100:
        print(f"  {tkr}: skipped, insufficient OOS predictions  [{time.time()-t0:.0f}s]")
        continue

    is_weak = skill < WEAK_SKILL_THRESHOLD
    flag = " [credit/vix regime adds ~nothing beyond climatology]" if is_weak else ""
    print(f"  {tkr}: {variant} @ {horizon}d (skill={skill:+.4f}){flag}, n_oos={len(oos)}  [{time.time()-t0:.0f}s]")

    full_price = ticker_registry[tkr]["series"]
    # Zoom to the OOS prediction window (+ ~6mo lead-in for context) rather
    # than the full multi-decade history -- at full-history zoom the actual
    # comparison this plot exists to show becomes an unreadable sliver.
    window_start = oos["target_date"].min() - pd.Timedelta(days=180)
    context_price = full_price[full_price.index >= window_start]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(context_price.index, context_price.values, color="black", lw=1.4, label="Actual price", zorder=4)
    ax.plot(oos["target_date"], oos["clim_price_q0.5"], color="tab:blue", lw=1.3, ls="--",
             label="Climatology median (calendar/seasonal baseline, same dates)", zorder=3, alpha=0.9)
    ax.plot(oos["target_date"], oos["pred_price_q0.5"], color="tab:red", lw=1.3,
             label=f"Predicted median ({variant} @ {horizon}d)", zorder=5)
    ax.fill_between(oos["target_date"], oos["pred_price_q0.1"], oos["pred_price_q0.9"],
                     color="tab:red", alpha=0.15, label="Predicted 10-90% band", zorder=2)
    edge_note = "credit/vix regime adds ~nothing beyond climatology" if is_weak else \
        "credit/vix regime beats climatology"
    ax.set_title(f"{tkr}: actual vs. predicted price -- best config: {variant} @ {horizon}d "
                 f"(skill above climatology {skill:+.4f} -- {edge_note})")
    ax.set_ylabel("Price")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    safe_tkr = tkr.replace('=', '').replace('^', '')
    out_path = os.path.join(PLOT_DIR, f"{safe_tkr}_price_prediction.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    results_summary[tkr] = {"horizon": horizon, "variant": variant, "skill_above_climatology": skill,
                             "n_oos": int(len(oos)), "weak_signal": is_weak}

with open(os.path.join(OUT_DIR, "results_price_predictions.json"), "w") as f:
    json.dump(results_summary, f, indent=2, default=float)
print(f"\nSaved {len(results_summary)} price-prediction plots to {PLOT_DIR}/")
print(f"Total time: {time.time()-t0:.1f}s")
print("\nDone.")
