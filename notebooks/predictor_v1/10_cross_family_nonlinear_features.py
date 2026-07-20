"""
Explicit nonlinear cross-family interaction features -- the genuine
combination Phase C (06_phase_c_combinations.py) didn't actually build.
Phase C concatenated whole feature families side by side and let LightGBM's
trees implicitly approximate any relationship between them; with shallow
trees (depth 4) and ~2,500-3,000 rows per fold, that's a weak substitute for
an explicit product term, the same distinction already drawn for
macro_interaction's own within-family interaction terms.

Cross-family products, all market-wide (one series per date, applies
uniformly across instruments, same as the underlying components):
  curvature_x_credit        = curve_curvature_10ybelly * credit_spread_regime
  curvature_x_uup           = curve_curvature_10ybelly * uup_trend_regime
  hmm_x_credit               = hmm_prob_high_vol * credit_spread_regime
  hmm_x_uup                  = hmm_prob_high_vol * uup_trend_regime
  hmm_x_curvature            = hmm_prob_high_vol * curve_curvature_10ybelly
  hmm_x_credit_x_curvature   = hmm_prob_high_vol * credit_spread_regime * curve_curvature_10ybelly
    (the explicit-product analogue of Phase C's "macro_term_regime" concatenation)

Run: python 10_cross_family_nonlinear_features.py
Output: features_cross_family_nonlinear.parquet
"""
import pandas as pd
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

macro = pd.read_parquet(os.path.join(OUT_DIR, "features_macro_interaction.parquet"))
macro = macro[["date", "uup_trend_regime", "credit_spread_regime"]].drop_duplicates("date")
term = pd.read_parquet(os.path.join(OUT_DIR, "features_term_structure.parquet"))
term = term[["date", "curve_curvature_10ybelly"]]
regime = pd.read_parquet(os.path.join(OUT_DIR, "features_regime_switching.parquet"))
regime = regime[["date", "hmm_prob_high_vol"]]

merged = macro.merge(term, on="date", how="outer").merge(regime, on="date", how="outer").sort_values("date")

merged["curvature_x_credit"] = merged["curve_curvature_10ybelly"] * merged["credit_spread_regime"]
merged["curvature_x_uup"] = merged["curve_curvature_10ybelly"] * merged["uup_trend_regime"]
merged["hmm_x_credit"] = merged["hmm_prob_high_vol"] * merged["credit_spread_regime"]
merged["hmm_x_uup"] = merged["hmm_prob_high_vol"] * merged["uup_trend_regime"]
merged["hmm_x_curvature"] = merged["hmm_prob_high_vol"] * merged["curve_curvature_10ybelly"]
merged["hmm_x_credit_x_curvature"] = (merged["hmm_prob_high_vol"] * merged["credit_spread_regime"]
                                       * merged["curve_curvature_10ybelly"])

out_cols = ["date", "curvature_x_credit", "curvature_x_uup", "hmm_x_credit", "hmm_x_uup",
            "hmm_x_curvature", "hmm_x_credit_x_curvature"]
feat = merged[out_cols]

out_path = os.path.join(OUT_DIR, "features_cross_family_nonlinear.parquet")
feat.to_parquet(out_path)
print(f"Saved {len(feat)} rows x {len(feat.columns)} cols to {out_path}")
print(feat.dropna().tail(5))
print(f"\nCoverage: {feat.dropna().shape[0]} / {len(feat)} rows fully populated")
