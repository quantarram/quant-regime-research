"""
Wave-1 predictor panel: macro regime as a MULTIPLICATIVE interaction term.

The point (per the plan's explicit correction to v1's naive approach): not a
parallel column. v1 attached ^VIX's/TLT's own gap value as a separate feature
next to each instrument's own statistics -- two columns the model has to
discover a relationship between on its own. This instead builds the
interaction directly: each instrument's own dynamical-trigger feature
(already in features_daily_panel.parquet) MULTIPLIED BY a slow macro state
variable, the way steering flow doesn't just co-occur with a storm, it
modulates its motion.

Macro state variables (both "slow," 200-day-smoothed regime indicators, not
noisy daily values):
  uup_trend_regime    = z-scored deviation of UUP (dollar) from its own
                         200-day moving average -- is the dollar in a
                         persistent up- or down-regime right now
  credit_spread_regime = z-scored deviation of the HYG/LQD ratio from its own
                         200-day moving average -- is credit risk appetite
                         currently expanding or contracting

Interaction features, per instrument already in the baseline panel:
  interact_gap21q4_{uup,credit} = gap_tau21_q4_z * {regime variable}
  interact_xiq4_{uup,credit}    = xi_q4_z * {regime variable}
(gap_tau21_q4 and xi_q4 chosen as the representative "extreme dynamics"
features, matching Paper 9's own finding that the q=4 / 21-day combination
carries the cleanest signal)

Run: python 04c_macro_interaction_features.py
Requires: ../multiasset_prices.parquet, features_daily_panel.parquet
Output: features_macro_interaction.parquet
"""
import pandas as pd
import numpy as np
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("  WAVE-1 PREDICTOR PANEL: MACRO REGIME INTERACTION")
print("=" * 60)

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
panel = pd.read_parquet(os.path.join(OUT_DIR, "features_daily_panel.parquet"))
panel["date"] = pd.to_datetime(panel["date"])

uup = prices["UUP"].dropna()
uup_regime = (uup - uup.rolling(200).mean()) / uup.rolling(200).std()
uup_regime.name = "uup_trend_regime"

ratio = (prices["HYG"] / prices["LQD"]).dropna()
credit_regime = (ratio - ratio.rolling(200).mean()) / ratio.rolling(200).std()
credit_regime.name = "credit_spread_regime"

macro = pd.concat([uup_regime, credit_regime], axis=1).reset_index().rename(columns={"index": "date"})
panel = panel.merge(macro, on="date", how="left")

for base_col in ["gap_tau21_q4_z", "xi_q4_z"]:
    for macro_col, macro_short in [("uup_trend_regime", "uup"), ("credit_spread_regime", "credit")]:
        panel[f"interact_{base_col.replace('_z','')}_{macro_short}"] = panel[base_col] * panel[macro_col]

interact_cols = [c for c in panel.columns if c.startswith("interact_")] + ["uup_trend_regime", "credit_spread_regime"]
out = panel[["ticker", "date"] + interact_cols]

out_path = os.path.join(OUT_DIR, "features_macro_interaction.parquet")
out.to_parquet(out_path)
print(f"Saved {len(out)} rows x {len(out.columns)} cols to {out_path}")
print(f"Non-null interaction coverage:\n{out[interact_cols].notna().mean()}")
