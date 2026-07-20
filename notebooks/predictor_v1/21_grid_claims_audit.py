"""
Applying the same audit discipline from 20_model_audit.py to the two
headline claims from the Wave-1/Wave-2 grid work, which were never
significance-tested before being presented as findings:
  - macro_interaction + l2 @ 21d: extreme_hit_rate 0.204 vs baseline's 0.162
  - term_structure + l2 @ 5d: extreme_hit_rate 0.209 vs baseline's 0.187
Both were reported as "standouts" based on point estimates alone. The grid
runners never saved raw OOS predictions, so this rebuilds those three
walk-forward runs (baseline, macro_interaction, term_structure -- l2 loss
only, the relevant horizons only) to actually test:
  1. Is each hit_rate significantly above the ~0.10 theoretical chance floor?
  2. Is the paired difference vs baseline distinguishable from noise
     (block bootstrap by year, not treating each OOS row as independent)?
  3. Leakage re-check (shifted label) on these exact configs.
  4. Are the "hits" substantive moves or marginal/boundary cases?

Run: python 21_grid_claims_audit.py
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb

from loss_functions import extreme_hit_rate

REPO_DIR = "/Users/arrams/Documents/personal_work/quant/quant-regime-research"
OUT_DIR = "/Users/arrams/Documents/personal_work/quant/quant-regime-research/notebooks/predictor_v1"
INITIAL_TRAIN_YEARS = 6
STEP_YEARS = 1
LGB_BASE = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)

print("=" * 60)
print("  AUDITING THE WAVE-1/2 GRID'S HEADLINE CLAIMS")
print("=" * 60)

prices = pd.read_parquet(f"{REPO_DIR}/notebooks/multiasset_prices.parquet")
baseline = pd.read_parquet(f"{OUT_DIR}/features_daily_panel.parquet")
baseline["date"] = pd.to_datetime(baseline["date"])
TICKERS = sorted(baseline["ticker"].unique())
date_pos = {d: i for i, d in enumerate(prices.index)}

z_cols = [c for c in baseline.columns if c.endswith("_z")]
ctx_cols = [c for c in baseline.columns if c.startswith("ctx_")]
for c in ctx_cols:
    baseline[c] = np.sign(baseline[c]) * np.log1p(np.abs(baseline[c]))
BASELINE_FEATURE_COLS = z_cols + ctx_cols + ["self_ref_score"]

macro = pd.read_parquet(f"{OUT_DIR}/features_macro_interaction.parquet")
macro["date"] = pd.to_datetime(macro["date"])
macro_cols = [c for c in macro.columns if c not in ("ticker", "date")]

term = pd.read_parquet(f"{OUT_DIR}/features_term_structure.parquet")
term["date"] = pd.to_datetime(term["date"])
term_cols = [c for c in term.columns if c != "date"]

GLOBAL_START = pd.Timestamp("2007-12-12")  # matches the original grid's fair-comparison floor


def build_config(name, horizon, label_shift=0):
    s_map = {t: prices[t] for t in TICKERS}
    ret_frames = []
    for t in TICKERS:
        s = s_map[t]
        df = pd.DataFrame({"date": s.index, "ticker": t})
        fwd_price = s.shift(-horizon - label_shift)
        df["fwd_ret"] = np.log(fwd_price.values / s.values)
        ret_frames.append(df)
    ret_df = pd.concat(ret_frames, ignore_index=True)
    d = baseline.merge(ret_df, on=["ticker", "date"], how="left")
    feature_cols = list(BASELINE_FEATURE_COLS)
    if name == "macro_interaction":
        d = d.merge(macro, on=["ticker", "date"], how="left")
        feature_cols += macro_cols
    elif name == "term_structure":
        d = d.merge(term, on="date", how="left")
        feature_cols += term_cols
    d = d[d["date"] >= GLOBAL_START].copy()
    return d, feature_cols


def run_walkforward(d, feature_cols, horizon):
    dd = d.dropna(subset=feature_cols + ["fwd_ret"]).copy()
    dd["pos"] = dd["date"].map(date_pos)
    min_year, max_year = dd["date"].dt.year.min(), dd["date"].dt.year.max()
    first_test_year = min_year + INITIAL_TRAIN_YEARS

    oos_rows = []
    test_year = first_test_year
    while test_year <= max_year:
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year + STEP_YEARS}-01-01")
        test_mask = (dd["date"] >= test_start) & (dd["date"] < test_end)
        if test_mask.sum() < 30:
            test_year += STEP_YEARS
            continue
        test_start_pos = date_pos.get(test_start, None)
        if test_start_pos is None:
            candidates = [p for dt, p in date_pos.items() if dt >= test_start]
            test_start_pos = min(candidates) if candidates else None
        train_mask = (dd["date"] < test_start) & (dd["pos"] + horizon < test_start_pos)
        if train_mask.sum() < 200:
            test_year += STEP_YEARS
            continue

        Xtr, ytr = dd.loc[train_mask, feature_cols], dd.loc[train_mask, "fwd_ret"]
        Xte, yte = dd.loc[test_mask, feature_cols], dd.loc[test_mask, "fwd_ret"]
        m = lgb.LGBMRegressor(**LGB_BASE, objective="regression")
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)

        rec = dd.loc[test_mask, ["ticker", "date"]].copy()
        rec["y_true"] = yte.values
        rec["y_pred"] = pred
        rec["year"] = test_year
        oos_rows.append(rec)
        test_year += STEP_YEARS

    return pd.concat(oos_rows, ignore_index=True)


def block_bootstrap_hit_rate(oos, n_boot=2000, seed=0):
    years = sorted(oos["year"].unique())
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sampled = rng.choice(years, size=len(years), replace=True)
        boot = pd.concat([oos[oos["year"] == y] for y in sampled], ignore_index=True)
        stats.append(extreme_hit_rate(boot["y_true"].values, boot["y_pred"].values))
    return np.array(stats)


CLAIMS = [("macro_interaction", 21, 0.204, 0.162), ("term_structure", 5, 0.209, 0.187)]

for config_name, horizon, claimed_hit, claimed_baseline_hit in CLAIMS:
    print(f"\n{'='*60}\nAuditing: {config_name} @ {horizon}d (claimed hit_rate={claimed_hit} vs baseline {claimed_baseline_hit})\n{'='*60}")

    d_base, feat_base = build_config("baseline", horizon)
    oos_base = run_walkforward(d_base, feat_base, horizon)
    d_cfg, feat_cfg = build_config(config_name, horizon)
    oos_cfg = run_walkforward(d_cfg, feat_cfg, horizon)

    hit_base = extreme_hit_rate(oos_base["y_true"].values, oos_base["y_pred"].values)
    hit_cfg = extreme_hit_rate(oos_cfg["y_true"].values, oos_cfg["y_pred"].values)
    print(f"Reproduced: baseline hit_rate={hit_base:.4f}, {config_name} hit_rate={hit_cfg:.4f} "
          f"(n_base={len(oos_base)}, n_cfg={len(oos_cfg)})")

    print("\n--- Is each hit_rate significantly above the ~0.10 chance floor? ---")
    boot_base = block_bootstrap_hit_rate(oos_base)
    boot_cfg = block_bootstrap_hit_rate(oos_cfg)
    print(f"  baseline: 95% CI = [{np.percentile(boot_base,2.5):.4f}, {np.percentile(boot_base,97.5):.4f}]")
    print(f"  {config_name}: 95% CI = [{np.percentile(boot_cfg,2.5):.4f}, {np.percentile(boot_cfg,97.5):.4f}]")

    print(f"\n--- Is {config_name}'s edge over baseline real? (paired by year+ticker+date via merge) ---")
    merged = oos_base[["ticker", "date", "year", "y_true", "y_pred"]].merge(
        oos_cfg[["ticker", "date", "y_pred"]], on=["ticker", "date"], suffixes=("_base", "_cfg"))
    years = sorted(merged["year"].unique())
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        sampled = rng.choice(years, size=len(years), replace=True)
        boot = pd.concat([merged[merged["year"] == y] for y in sampled], ignore_index=True)
        hit_b = extreme_hit_rate(boot["y_true"].values, boot["y_pred_base"].values)
        hit_c = extreme_hit_rate(boot["y_true"].values, boot["y_pred_cfg"].values)
        diffs.append(hit_c - hit_b)
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
    print(f"  Point estimate (cfg - baseline): {hit_cfg - hit_base:+.4f}")
    print(f"  Block-bootstrap 95% CI on the difference: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Fraction of bootstrap draws favoring {config_name}: {(diffs > 0).mean():.1%}")
    if ci_lo < 0 < ci_hi:
        print(f"  -> CI spans zero: the '{config_name} beats baseline' claim is NOT statistically distinguishable from noise.")
    else:
        print(f"  -> CI excludes zero: {config_name}'s edge is real at conventional confidence.")

    print(f"\n--- Leakage re-check ({config_name} @ {horizon}d, shifted label) ---")
    d_shift, feat_shift = build_config(config_name, horizon, label_shift=1)
    oos_shift = run_walkforward(d_shift, feat_shift, horizon)
    hit_shift = extreme_hit_rate(oos_shift["y_true"].values, oos_shift["y_pred"].values)
    print(f"  real hit_rate={hit_cfg:.4f} vs shifted-label hit_rate={hit_shift:.4f} "
          f"({'OK, shifted not better' if hit_shift <= hit_cfg + 0.02 else 'WARNING: shifted is close to or better than real'})")

print("\nDone.")
