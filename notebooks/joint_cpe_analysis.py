"""
Joint CPE Analysis & Plotting
==============================
Analyses joint_cpe_results.parquet across predictor set sizes 2-10.

Outputs:
  - Console tables: top signals by size, direction, tier
  - Plots:
      1. CPE vs n_predictors (mean ± std) by direction
      2. n_joint vs n_predictors by direction
      3. Heatmap: Y × predictor_set for top size-2 signals
      4. Bubble chart: joint_CPE vs n_joint, sized by n_predictors
      5. Signal cards: detailed view of top joint signals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ast, warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TOP_N = 20

TIERS = {
    "Tier1_Directional": {0.50, 0.60},
    "Tier2_Moderate":    {0.70, 0.75, 0.80},
    "Tier3_Extreme":     {0.90, 0.95, 0.99},
}

ASSET_CLASS_MAP = {
    "equities":    ["SPY","QQQ","IWM","DIA","VTI","VT","EFA","EEM","VEA","VWO",
                    "EWJ","EWZ","FXI","INDA","EWY","XLK","XLF","XLE","XLV","XLI",
                    "XLP","XLY","XLU","XLRE","XLB","XLC","VTV","VUG","MTUM","USMV",
                    "QUAL","SIZE","ARKK","ICLN","ITB","XBI","SOXX",
                    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","BRK-B","XOM"],
    "crypto":      ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD",
                    "AVAX-USD","DOGE-USD","DOT-USD","LINK-USD","MATIC-USD","LTC-USD",
                    "BCH-USD","UNI-USD","ATOM-USD","IBIT","FBTC","GBTC","ETHE","BITB"],
    "volatility":  ["^VIX","^VXN","^OVX","^GVZ","^EVZ","^VVIX","^SKEW",
                    "UVXY","SVXY","VXX","VIXY","VIXM"],
    "commodities": ["GLD","IAU","SLV","PPLT","PALL","USO","BNO","UNG","UGA",
                    "CORN","WEAT","SOYB","CANE","NIB","JO","DJP","PDBC","DBC",
                    "GSG","CPER","DBB","GC=F","SI=F","CL=F","BZ=F","NG=F",
                    "HG=F","ZC=F","ZW=F","ZS=F"],
    "rates":       ["SHY","IEI","IEF","TLH","TLT","ZROZ","EDV","TIP","SCHP",
                    "LQD","HYG","JNK","EMB","AGG","BND","VCSH","VCIT","VCLT",
                    "MUB","MBB","^TNX","^TYX","^FVX","^IRX","TMF","TBT","TBF"],
    "fx":          ["EURUSD=X","GBPUSD=X","JPYUSD=X","CHFUSD=X","CADUSD=X",
                    "AUDUSD=X","NZDUSD=X","SGDUSD=X","INRUSD=X","BRLUSD=X",
                    "MXNUSD=X","ZARUSD=X","UUP","UDN",
                    "EURJPY=X","EURGBP=X","GBPJPY=X","AUDJPY=X","EURCHF=X"],
}

AC_COLORS = {
    "equities":    "#2196F3",
    "crypto":      "#FF9800",
    "volatility":  "#9C27B0",
    "commodities": "#4CAF50",
    "rates":       "#F44336",
    "fx":          "#00BCD4",
    "other":       "#9E9E9E",
}

def get_ac(ticker):
    for ac, tickers in ASSET_CLASS_MAP.items():
        if ticker in tickers:
            return ac
    return "other"

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading joint_cpe_results.parquet...")
df = pd.read_parquet("joint_cpe_results.parquet")
print(f"  Total rows    : {len(df):,}")
print(f"  Directions    : {df['direction'].value_counts().to_dict()}")
print(f"  n_predictors  : {sorted(df['n_predictors'].unique())}")
print(f"  q_Y values    : {sorted(df['q_Y'].unique())}")

df["Y_class"] = df["Y"].apply(get_ac)

# Tier label
def get_tier(q_y):
    for tier, qs in TIERS.items():
        if q_y in qs:
            return tier
    return "other"
df["tier"] = df["q_Y"].apply(get_tier)

# ── PLOT 1: CPE vs n_predictors ────────────────────────────────────────────────
print("\n  Plot 1: CPE vs n_predictors...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, direction in zip(axes, ["bullish", "bearish"]):
    sub = df[df["direction"] == direction]
    stats = sub.groupby("n_predictors")["joint_CPE"].agg(["mean","std","count"])
    ax.errorbar(stats.index, stats["mean"], yerr=stats["std"],
                marker="o", linewidth=2, capsize=4, color="steelblue" if direction=="bullish" else "firebrick")
    ax.fill_between(stats.index,
                    stats["mean"] - stats["std"],
                    stats["mean"] + stats["std"], alpha=0.15,
                    color="steelblue" if direction=="bullish" else "firebrick")
    for x, (m, s, c) in stats.iterrows():
        ax.annotate(f"n={int(c)}", (x, m + s + 0.003), ha="center", fontsize=7)
    ax.set_xlabel("Number of joint predictors", fontsize=10)
    ax.set_ylabel("Joint CPE", fontsize=10)
    ax.set_title(f"Joint CPE vs n_predictors — {direction.capitalize()}", fontsize=11, fontweight="bold")
    ax.set_ylim(0.94, 1.02)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig("joint_cpe_vs_npredictors.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: joint_cpe_vs_npredictors.png")

# ── PLOT 2: n_joint vs n_predictors ───────────────────────────────────────────
print("  Plot 2: n_joint vs n_predictors...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, direction in zip(axes, ["bullish", "bearish"]):
    sub = df[df["direction"] == direction]
    stats = sub.groupby("n_predictors")["n_joint"].agg(["median","mean"])
    ax.bar(stats.index, stats["median"], color="steelblue" if direction=="bullish" else "firebrick",
           alpha=0.7, label="median n_joint")
    ax.plot(stats.index, stats["mean"], marker="o", color="black",
            linewidth=1.5, label="mean n_joint")
    ax.axhline(100, color="red", linestyle="--", linewidth=1, label="MIN_N=100")
    ax.set_xlabel("Number of joint predictors", fontsize=10)
    ax.set_ylabel("Joint conditioning sample size (n_joint)", fontsize=10)
    ax.set_title(f"Sample size vs n_predictors — {direction.capitalize()}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("joint_cpe_sample_size.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: joint_cpe_sample_size.png")

# ── PLOT 3: Bubble chart — joint_CPE vs n_joint, by direction ─────────────────
print("  Plot 3: Bubble chart...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, direction in zip(axes, ["bullish", "bearish"]):
    sub = df[df["direction"] == direction].copy()
    colors = sub["Y_class"].map(AC_COLORS)
    sc = ax.scatter(sub["n_joint"], sub["joint_CPE"],
                    s=sub["n_predictors"] * 20,
                    c=colors, alpha=0.5, edgecolors="none")
    ax.set_xlabel("n_joint (conditioning sample size)", fontsize=10)
    ax.set_ylabel("Joint CPE", fontsize=10)
    ax.set_title(f"Joint CPE vs Sample Size — {direction.capitalize()}\n(bubble size = n_predictors)",
                 fontsize=11, fontweight="bold")
    ax.axhline(0.80, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="CPE=0.80")
    ax.axvline(100,  color="orange", linestyle="--", linewidth=0.8, alpha=0.5, label="n=100")
    ax.grid(True, alpha=0.2)
    # Legend for asset classes
    patches = [mpatches.Patch(color=c, label=ac) for ac, c in AC_COLORS.items() if ac != "other"]
    ax.legend(handles=patches, fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig("joint_cpe_bubble.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: joint_cpe_bubble.png")

# ── PLOT 4: Top signals by tier and direction — signal cards ───────────────────
def format_predictors(row):
    parts = []
    for x, tp, qx in zip(row["predictors"], row["tau_pasts"], row["q_Xs"]):
        parts.append(f"{x}(τ={tp},q={qx})")
    return " ∩ ".join(parts)

print("  Plot 4: Signal cards per tier...")
for tier_name, tier_qs in TIERS.items():
    df_tier = df[df["q_Y"].isin(tier_qs)]
    if df_tier.empty:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f"Top Joint CPE Signals — {tier_name.replace('_',' ')}", fontsize=13, fontweight="bold")

    for ax, direction in zip(axes, ["bullish", "bearish"]):
        sub = (df_tier[df_tier["direction"] == direction]
               .sort_values(["joint_CPE","n_joint"], ascending=[False,False])
               .head(TOP_N)
               .reset_index(drop=True))

        if sub.empty:
            ax.set_visible(False)
            continue

        # Text table
        ax.axis("off")
        col_labels = ["Y", "τf", "qY", "n_pred", "CPE", "n_joint", "Predictors (X, τp, qX)"]
        rows = []
        for _, r in sub.iterrows():
            pred_str = " ∩\n".join([f"{x}(τ={tp},q={qx})"
                                     for x, tp, qx in zip(r["predictors"],
                                                           r["tau_pasts"],
                                                           r["q_Xs"])])
            rows.append([r["Y"], int(r["tau_future"]), r["q_Y"],
                         int(r["n_predictors"]), f"{r['joint_CPE']:.4f}",
                         int(r["n_joint"]), pred_str])

        table = ax.table(cellText=rows, colLabels=col_labels,
                         loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width(col=list(range(len(col_labels))))

        # Colour header
        for j in range(len(col_labels)):
            table[(0, j)].set_facecolor("#37474F")
            table[(0, j)].set_text_props(color="white", fontweight="bold")

        # Colour rows by Y asset class
        for i, (_, r) in enumerate(sub.iterrows(), start=1):
            ac  = get_ac(r["Y"])
            col = AC_COLORS.get(ac, "#FFFFFF")
            for j in range(len(col_labels)):
                table[(i, j)].set_facecolor(col + "33")  # 20% opacity

        ax.set_title(f"{direction.capitalize()} signals  (n={len(sub)})",
                     fontsize=10, fontweight="bold", pad=10)

    plt.tight_layout()
    fname = f"joint_signal_cards_{tier_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

# ── CONSOLE TABLES ─────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("  JOINT CPE SUMMARY BY TIER AND DIRECTION")
print(f"{'='*90}")

for tier_name, tier_qs in TIERS.items():
    df_tier = df[df["q_Y"].isin(tier_qs)]
    print(f"\n  {'─'*80}")
    print(f"  {tier_name}  ({len(df_tier):,} joint signals)")
    print(f"  {'─'*80}")
    for direction in ["bullish","bearish"]:
        sub = (df_tier[df_tier["direction"] == direction]
               .sort_values(["joint_CPE","n_joint"], ascending=[False,False]))
        print(f"\n  {direction.upper()}  ({len(sub):,} signals)")
        # Show top by size 2 and size 3
        for size in [2, 3, 4]:
            top = sub[sub["n_predictors"] == size].head(5)
            if top.empty:
                continue
            print(f"\n    Size {size}:")
            for _, r in top.iterrows():
                pred_str = " ∩ ".join([f"{x}(τ={tp},q={qx})"
                                        for x, tp, qx in zip(r["predictors"],
                                                              r["tau_pasts"],
                                                              r["q_Xs"])])
                print(f"      Y={r['Y']:<12} τf={r['tau_future']:>3}  "
                      f"qY={r['q_Y']}  CPE={r['joint_CPE']:.4f}  "
                      f"n={r['n_joint']:>4}  [{pred_str}]")

# ── SUMMARY STATS ──────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("  SUMMARY: BEST PREDICTED ASSETS (by mean joint CPE at size 2)")
print(f"{'='*90}")
for direction in ["bullish","bearish"]:
    sub = df[(df["direction"]==direction) & (df["n_predictors"]==2)]
    stats = (sub.groupby("Y")
               .agg(n_signals=("joint_CPE","count"),
                    mean_CPE=("joint_CPE","mean"),
                    max_CPE=("joint_CPE","max"),
                    mean_n=("n_joint","mean"))
               .sort_values("mean_CPE", ascending=False)
               .head(15))
    print(f"\n  {direction.upper()} — top 15 predicted assets:")
    print(stats.round(4).to_string())

print(f"\n{'='*90}")
print("  SUMMARY: MOST FREQUENT PREDICTORS IN JOINT SETS (size 2)")
print(f"{'='*90}")
for direction in ["bullish","bearish"]:
    sub = df[(df["direction"]==direction) & (df["n_predictors"]==2)]
    all_predictors = []
    for _, r in sub.iterrows():
        all_predictors.extend(r["predictors"])
    freq = pd.Series(all_predictors).value_counts().head(15)
    print(f"\n  {direction.upper()} — top 15 most frequent predictors:")
    print(freq.to_string())

print("\nDone.")
