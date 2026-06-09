"""
CPE-Weighted Signal Score
=========================
For each predicted asset Y, computes a current signal score at the latest
available date using the joint CPE results (n_predictors <= 6).

Score formula for asset Y at time t:
    S(Y,t) = sum_{k in bullish sets} w_k * I_k(t)
           - sum_{k in bearish sets}  w_k * I_k(t)

where:
    w_k   = CPE_k * lift_k * log(n_joint_k)   (quality weight)
    I_k(t) = 1 if ALL predictors in joint set k are currently in
              their respective tails at their respective tau_past windows
            = 0 otherwise

Output:
    cpe_signal_scores.parquet  — one row per (Y, tau_future) with:
        score, score_norm, direction, n_bullish_firing, n_bearish_firing,
        top_firing_conditions, latest_date
    cpe_signal_scores.csv      — same, human-readable
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MAX_PREDICTORS = 6   # only use joint sets of size <= 6
MIN_CPE        = 0.80
MIN_LIFT       = 1.5
MIN_N          = 100

RATE_INDEX_TICKERS = {
    "^VIX","^VXN","^OVX","^GVZ","^EVZ","^VVIX","^SKEW",
    "^TNX","^TYX","^FVX","^IRX"
}

print(f"\n{'='*65}")
print(f"  CPE SIGNAL SCORE  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*65}")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("\n  Loading data...")
prices   = pd.read_parquet("multiasset_prices.parquet")
joint    = pd.read_parquet("joint_cpe_results.parquet")

all_tickers   = list(prices.columns)
price_tickers = [t for t in all_tickers if t not in RATE_INDEX_TICKERS]
rate_tickers  = [t for t in all_tickers if t in RATE_INDEX_TICKERS]

print(f"  Joint CPE rows total  : {len(joint):,}")

# Filter to n_predictors <= 6
joint = joint[joint["n_predictors"] <= MAX_PREDICTORS].copy()
print(f"  After n_pred <= 6     : {len(joint):,}")
print(f"  Directions            : {joint['direction'].value_counts().to_dict()}")
print(f"  Unique Y assets       : {joint['Y'].nunique()}")

# ── COMPUTE CURRENT INCREMENTS ────────────────────────────────────────────────
print("\n  Computing current increments at latest date...")

# Latest date with data
latest_date = prices.index.max()
print(f"  Latest date: {latest_date.date()}")

TAU_LIST = [1, 5, 10, 21, 63, 126, 252, 300]

# For each tau, compute increment at latest date for all tickers
current_inc = {}
for tau in TAU_LIST:
    inc_row = {}
    for t in price_tickers:
        s = prices[t].dropna()
        if len(s) < tau + 1:
            continue
        # log return over past tau days
        inc_row[t] = np.log(s.iloc[-1] / s.iloc[-1 - tau])
    for t in rate_tickers:
        s = prices[t].dropna()
        if len(s) < tau + 1:
            continue
        # level change for rates/indices
        inc_row[t] = s.iloc[-1] - s.iloc[-1 - tau]
    current_inc[tau] = inc_row
    print(f"    tau={tau:>3}  computed for {len(inc_row)} tickers")

# ── COMPUTE FULL-SAMPLE QUANTILE THRESHOLDS ───────────────────────────────────
print("\n  Computing quantile thresholds...")

Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]
full_q = sorted(set(Q_GRID + [round(1-q,10) for q in Q_GRID]))

thresholds = {}
for tau in TAU_LIST:
    # Build increment series for all tickers
    inc_df = pd.DataFrame(index=prices.index)
    for t in price_tickers:
        s = prices[t]
        inc_df[t] = np.log(s / s.shift(tau))
    for t in rate_tickers:
        s = prices[t]
        inc_df[t] = s - s.shift(tau)
    for q in full_q:
        thresholds[(tau, q)] = inc_df.quantile(q, numeric_only=True).to_dict()

print(f"  Done. {len(thresholds)} (tau,q) threshold dicts.")

# ── CHECK IF CONDITION FIRES ──────────────────────────────────────────────────
def condition_fires(predictors, tau_pasts, q_Xs, direction):
    """
    Returns True if ALL predictors in the joint set are currently
    in their respective tails.
    """
    for x, tau_p, q_x in zip(predictors, tau_pasts, q_Xs):
        tau_p = int(tau_p)
        q_x   = float(q_x)

        # Get current increment of X
        curr = current_inc.get(tau_p, {}).get(x, None)
        if curr is None or np.isnan(curr):
            return False

        # Get threshold
        if direction == "bullish":
            thresh = thresholds.get((tau_p, q_x), {}).get(x, np.nan)
            if np.isnan(thresh):
                return False
            if not (curr > thresh):
                return False
        else:
            thresh = thresholds.get((tau_p, round(1-q_x,10)), {}).get(x, np.nan)
            if np.isnan(thresh):
                return False
            if not (curr < thresh):
                return False
    return True

# ── COMPUTE SIGNAL SCORES ─────────────────────────────────────────────────────
print("\n  Computing signal scores for each (Y, tau_future)...")

results = []

# Group by (Y, tau_future)
for (y, tau_f), grp in joint.groupby(["Y", "tau_future"]):
    bull_sets = grp[grp["direction"] == "bullish"]
    bear_sets = grp[grp["direction"] == "bearish"]

    bull_score    = 0.0
    bear_score    = 0.0
    n_bull_firing = 0
    n_bear_firing = 0
    top_bull      = []
    top_bear      = []
    total_bull_w  = 0.0
    total_bear_w  = 0.0

    for _, row in bull_sets.iterrows():
        w = float(row["joint_CPE"]) * float(row["lift"]) * np.log(max(row["n_joint"], 1))
        total_bull_w += w
        fires = condition_fires(row["predictors"], row["tau_pasts"],
                                row["q_Xs"], "bullish")
        if fires:
            bull_score    += w
            n_bull_firing += 1
            top_bull.append({
                "predictors": list(row["predictors"]),
                "tau_pasts":  [int(x) for x in row["tau_pasts"]],
                "q_Xs":       [float(x) for x in row["q_Xs"]],
                "q_Y":        float(row["q_Y"]),
                "joint_CPE":  float(row["joint_CPE"]),
                "lift":       float(row["lift"]),
                "n_joint":    int(row["n_joint"]),
                "weight":     round(w, 4),
            })

    for _, row in bear_sets.iterrows():
        w = float(row["joint_CPE"]) * float(row["lift"]) * np.log(max(row["n_joint"], 1))
        total_bear_w += w
        fires = condition_fires(row["predictors"], row["tau_pasts"],
                                row["q_Xs"], "bearish")
        if fires:
            bear_score    += w
            n_bear_firing += 1
            top_bear.append({
                "predictors": list(row["predictors"]),
                "tau_pasts":  [int(x) for x in row["tau_pasts"]],
                "q_Xs":       [float(x) for x in row["q_Xs"]],
                "q_Y":        float(row["q_Y"]),
                "joint_CPE":  float(row["joint_CPE"]),
                "lift":       float(row["lift"]),
                "n_joint":    int(row["n_joint"]),
                "weight":     round(w, 4),
            })

    raw_score = bull_score - bear_score

    # Normalise: divide by total possible weight so score is in [-1, +1]
    total_w = total_bull_w + total_bear_w
    score_norm = raw_score / total_w if total_w > 0 else 0.0

    # Direction label
    if score_norm > 0.05:
        direction_label = "BULLISH"
    elif score_norm < -0.05:
        direction_label = "BEARISH"
    else:
        direction_label = "NEUTRAL"

    # Sort top conditions by weight descending
    top_bull.sort(key=lambda x: x["weight"], reverse=True)
    top_bear.sort(key=lambda x: x["weight"], reverse=True)

    results.append({
        "Y":                y,
        "tau_future":       int(tau_f),
        "score_raw":        round(raw_score, 4),
        "score_norm":       round(score_norm, 4),
        "direction":        direction_label,
        "n_bull_sets":      len(bull_sets),
        "n_bear_sets":      len(bear_sets),
        "n_bull_firing":    n_bull_firing,
        "n_bear_firing":    n_bear_firing,
        "bull_weight_total":round(total_bull_w, 4),
        "bear_weight_total":round(total_bear_w, 4),
        "bull_weight_fired":round(bull_score, 4),
        "bear_weight_fired":round(bear_score, 4),
        "top_bull_conditions": top_bull[:3],  # top 3 firing bullish
        "top_bear_conditions": top_bear[:3],  # top 3 firing bearish
        "latest_date":      str(latest_date.date()),
    })

df = pd.DataFrame(results)
df = df.sort_values(["Y","tau_future"]).reset_index(drop=True)

print(f"  Computed scores for {df['Y'].nunique()} assets "
      f"x {df['tau_future'].nunique()} tau_future values "
      f"= {len(df):,} rows")

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  SIGNAL SCORES AT {latest_date.date()}")
print(f"{'='*65}")

# Show the strongest signals — top 20 by abs(score_norm), tau_future in [21,63,252]
summary = df[df["tau_future"].isin([21, 63, 252])].copy()
summary["abs_score"] = summary["score_norm"].abs()
top = summary.nlargest(30, "abs_score")

print(f"\n  Top 30 signals (tau_future in [21,63,252], ranked by |score_norm|):")
print(f"\n  {'Y':<14} {'tf':>4}  {'score':>8}  {'dir':<8}  "
      f"{'bull_fire':>9}  {'bear_fire':>9}  {'q_Y range':<12}")
print(f"  {'-'*80}")
for _, r in top.iterrows():
    print(f"  {r['Y']:<14} {r['tau_future']:>4}  {r['score_norm']:>8.4f}  "
          f"{r['direction']:<8}  {r['n_bull_firing']:>9}  "
          f"{r['n_bear_firing']:>9}  "
          f"bull:{r['bull_weight_fired']:.1f}/bear:{r['bear_weight_fired']:.1f}")

# Direction breakdown
print(f"\n  Direction summary across all (Y, tau_future):")
print(df["direction"].value_counts().to_string())

print(f"\n  By tau_future:")
for tf in sorted(df["tau_future"].unique()):
    sub = df[df["tau_future"]==tf]
    print(f"    tau={tf:>3}:  BULLISH={( sub['direction']=='BULLISH').sum():>3}  "
          f"NEUTRAL={(sub['direction']=='NEUTRAL').sum():>3}  "
          f"BEARISH={(sub['direction']=='BEARISH').sum():>3}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
# Parquet (with list columns)
df.to_parquet("cpe_signal_scores.parquet", engine="pyarrow",
              compression="snappy", index=False)

# CSV (flatten list columns to strings)
df_csv = df.copy()
for col in ["top_bull_conditions","top_bear_conditions"]:
    df_csv[col] = df_csv[col].apply(
        lambda lst: " | ".join([
            " & ".join([f"{x}(t={tp},q={qx})"
                        for x,tp,qx in zip(c["predictors"],c["tau_pasts"],c["q_Xs"])])
            + f" CPE={c['joint_CPE']:.3f} lift={c['lift']:.2f} n={c['n_joint']}"
            for c in lst
        ])
    )
df_csv.to_csv("cpe_signal_scores.csv", index=False)

print(f"\n  Saved: cpe_signal_scores.parquet")
print(f"  Saved: cpe_signal_scores.csv")
print(f"\n{'='*65}\n")

# ── DETAILED PRINTOUT FOR TOP ASSETS ─────────────────────────────────────────
print("  DETAILED SCORES FOR STRONGEST ASSETS")
print("  (assets with |score_norm| > 0.10 at any tau_future)")
print(f"{'='*65}")

strong = df[df["score_norm"].abs() > 0.10].sort_values(
    "score_norm", ascending=False)

for y in strong["Y"].unique():
    sub = df[df["Y"]==y].sort_values("tau_future")
    print(f"\n  {y}:")
    for _, r in sub.iterrows():
        if r["score_norm"] == 0 and r["n_bull_firing"] == 0 and r["n_bear_firing"] == 0:
            continue
        col = "+" if r["direction"]=="BULLISH" else ("-" if r["direction"]=="BEARISH" else " ")
        print(f"    tau_f={r['tau_future']:>3}  score={r['score_norm']:>+7.4f}  "
              f"{col} {r['direction']:<8}  "
              f"firing: {r['n_bull_firing']} bull / {r['n_bear_firing']} bear")
        # Top firing conditions
        for cond in r["top_bull_conditions"][:2]:
            preds = " & ".join([f"{x}(t={tp},q={qx})"
                                for x,tp,qx in zip(cond["predictors"],
                                                    cond["tau_pasts"],
                                                    cond["q_Xs"])])
            print(f"      [BULL] {preds}  CPE={cond['joint_CPE']:.3f}  "
                  f"lift={cond['lift']:.2f}  w={cond['weight']:.2f}")
        for cond in r["top_bear_conditions"][:2]:
            preds = " & ".join([f"{x}(t={tp},q={qx})"
                                for x,tp,qx in zip(cond["predictors"],
                                                    cond["tau_pasts"],
                                                    cond["q_Xs"])])
            print(f"      [BEAR] {preds}  CPE={cond['joint_CPE']:.3f}  "
                  f"lift={cond['lift']:.2f}  w={cond['weight']:.2f}")

print(f"\n  Done.\n")
