"""
CPE Analysis & Plotting — Three Tiers
======================================
Tier 1 — Directional  : q_Y in {0.50, 0.60}
Tier 2 — Moderate     : q_Y in {0.70, 0.75, 0.80}
Tier 3 — Extreme      : q_Y in {0.90, 0.95, 0.99}

For each tier x direction (bullish/bearish):
  - Top N signals table (ranked by CPE then lift then n_condition)
  - CPE heatmap (Y x X) for representative (tau_past, tau_future, q_X, q_Y)
  - CPE vs tau_future curves for top signals
  - Summary stats by asset class
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TOP_N = 30

TIERS = {
    "Tier1_Directional": {"q_Y": {0.50, 0.60},          "label": "Directional (q_Y = 0.50, 0.60)"},
    "Tier2_Moderate":    {"q_Y": {0.70, 0.75, 0.80},    "label": "Moderate Large Move (q_Y = 0.70–0.80)"},
    "Tier3_Extreme":     {"q_Y": {0.90, 0.95, 0.99},    "label": "Extreme Large Move (q_Y = 0.90–0.99)"},
}

ASSET_CLASS_MAP = {
    "equities":    ["SPY","QQQ","IWM","DIA","VTI","VT","EFA","EEM","VEA","VWO",
                    "EWJ","EWZ","FXI","INDA","EWY","XLK","XLF","XLE","XLV","XLI",
                    "XLP","XLY","XLU","XLRE","XLB","XLC","VTV","VUG","MTUM","USMV",
                    "QUAL","SIZE","SSO","SDS","TQQQ","ARKK","ICLN","ITB","XBI","SOXX",
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
                    "AUDUSD=X","NZDUSD=X","SGDUSD=X","CNYUSD=X","INRUSD=X",
                    "BRLUSD=X","MXNUSD=X","ZARUSD=X","KRWUSD=X","THBUSD=X",
                    "UUP","UDN","EURJPY=X","EURGBP=X","GBPJPY=X","AUDJPY=X","EURCHF=X"],
}

def get_asset_class(ticker):
    for ac, tickers in ASSET_CLASS_MAP.items():
        if ticker in tickers:
            return ac
    return "other"

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading cpe_results.parquet...")
df = pd.read_parquet("cpe_results.parquet")
print(f"  Total rows : {len(df):,}")
print(f"  Directions : {df['direction'].value_counts().to_dict()}")
print(f"  q_Y values : {sorted(df['q_Y'].unique())}")

df["Y_class"] = df["Y"].apply(get_asset_class)
df["X_class"] = df["X"].apply(get_asset_class)

# ── HELPER: TOP SIGNALS TABLE ─────────────────────────────────────────────────
display_cols = ["Y","X","tau_past","tau_future","q_X","q_Y",
                "CPE","uncond_prob","lift","n_condition","Y_class","X_class"]

def print_top(df_in, direction, tier_label, n=TOP_N):
    sub = df_in[df_in["direction"] == direction].copy()
    if sub.empty:
        print(f"  No {direction} signals in {tier_label}")
        return
    top = sub.sort_values(["CPE","lift","n_condition"],
                          ascending=False).head(n)
    print(f"\n{'='*90}")
    print(f"  TOP {n} {direction.upper()} — {tier_label}")
    print(f"  Total signals in tier: {len(sub):,}")
    print(f"{'='*90}")
    print(top[display_cols].to_string(index=False))

# ── HELPER: HEATMAP ───────────────────────────────────────────────────────────
def plot_heatmap(df_in, direction, tau_p, tau_f, q_x, q_y, tier_name, top_n=40):
    sub = df_in[
        (df_in["direction"]  == direction) &
        (df_in["tau_past"]   == tau_p) &
        (df_in["tau_future"] == tau_f) &
        (df_in["q_X"]        == q_x)  &
        (df_in["q_Y"]        == q_y)
    ].copy()

    if sub.empty:
        return

    pivot = sub.pivot_table(index="Y", columns="X", values="CPE", aggfunc="mean")
    pivot = pivot.loc[
        pivot.notna().sum(axis=1).nlargest(top_n).index,
        pivot.notna().sum(axis=0).nlargest(top_n).index
    ]
    if pivot.empty:
        return

    cmap  = "Greens" if direction == "bullish" else "Reds"
    vmin  = 0.80
    vmax  = 1.00

    fig, ax = plt.subplots(figsize=(
        max(12, len(pivot.columns) * 0.32),
        max(8,  len(pivot.index)   * 0.28)
    ))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Predictor X", fontsize=10)
    ax.set_ylabel("Predicted Y", fontsize=10)
    ax.set_title(
        f"CPE Heatmap [{direction.capitalize()}]  "
        f"τ_past={tau_p}d  τ_future={tau_f}d  q_X={q_x}  q_Y={q_y}\n"
        f"{tier_name}",
        fontsize=10, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, label="CPE")
    plt.tight_layout()
    fname = f"heatmap_{direction}_{tier_name}_tp{tau_p}_tf{tau_f}_qx{q_x}_qy{q_y}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

# ── HELPER: CPE vs TAU_FUTURE CURVES ─────────────────────────────────────────
def plot_tau_curves(df_in, direction, tier_name, top_n=10):
    sub = df_in[df_in["direction"] == direction].copy()
    if sub.empty:
        return

    # Pick top_n unique (Y, X, tau_past, q_X, q_Y) combos by CPE
    top_combos = (sub.sort_values(["CPE","lift","n_condition"], ascending=False)
                     [["Y","X","tau_past","q_X","q_Y"]]
                     .drop_duplicates()
                     .head(top_n))

    fig, ax = plt.subplots(figsize=(14, 6))
    plotted = 0
    for _, row in top_combos.iterrows():
        curve = df_in[
            (df_in["direction"]  == direction) &
            (df_in["Y"]          == row["Y"]) &
            (df_in["X"]          == row["X"]) &
            (df_in["tau_past"]   == row["tau_past"]) &
            (df_in["q_X"]        == row["q_X"]) &
            (df_in["q_Y"]        == row["q_Y"])
        ].sort_values("tau_future")

        if len(curve) > 1:
            ax.plot(curve["tau_future"], curve["CPE"],
                    marker="o", linewidth=1.5, markersize=4,
                    label=f"Y={row['Y']} X={row['X']} τp={int(row['tau_past'])} "
                          f"qX={row['q_X']} qY={row['q_Y']}")
            plotted += 1

    if plotted == 0:
        plt.close()
        return

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="CPE=0.5 (random)")
    ax.set_xlabel("τ_future (trading days)", fontsize=10)
    ax.set_ylabel("CPE", fontsize=10)
    ax.set_ylim(0.75, 1.05)
    ax.set_title(
        f"CPE vs Future Horizon — {direction.capitalize()} — {tier_name}",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"tau_curves_{direction}_{tier_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

# ── HELPER: SUMMARY BY ASSET CLASS ───────────────────────────────────────────
def print_summary(df_in, direction, tier_label):
    sub = df_in[df_in["direction"] == direction]
    if sub.empty:
        return
    print(f"\n  --- {direction.upper()} summary by Y asset class [{tier_label}] ---")
    print(sub.groupby("Y_class").agg(
        n_signals   =("CPE","count"),
        mean_CPE    =("CPE","mean"),
        mean_lift   =("lift","mean"),
        max_lift    =("lift","max"),
        mean_n      =("n_condition","mean"),
    ).round(3).sort_values("mean_lift", ascending=False).to_string())

    print(f"\n  --- {direction.upper()} best predictor classes (X) [{tier_label}] ---")
    print(sub.groupby("X_class").agg(
        n_signals   =("CPE","count"),
        mean_CPE    =("CPE","mean"),
        mean_lift   =("lift","mean"),
    ).round(3).sort_values("mean_lift", ascending=False).to_string())

# ── REPRESENTATIVE HEATMAP PARAMS PER TIER ───────────────────────────────────
HEATMAP_PARAMS = {
    "Tier1_Directional": [
        (21,  21,  0.75, 0.50),
        (63,  63,  0.75, 0.50),
        (63,  63,  0.90, 0.60),
    ],
    "Tier2_Moderate": [
        (21,  21,  0.75, 0.75),
        (63,  63,  0.75, 0.75),
        (63,  63,  0.90, 0.80),
        (252, 252, 0.75, 0.75),
    ],
    "Tier3_Extreme": [
        (21,  21,  0.90, 0.90),
        (63,  63,  0.90, 0.90),
        (252, 252, 0.90, 0.95),
    ],
}

# ── MAIN LOOP OVER TIERS ──────────────────────────────────────────────────────
for tier_name, tier_cfg in TIERS.items():
    tier_label = tier_cfg["label"]
    df_tier = df[df["q_Y"].isin(tier_cfg["q_Y"])].copy()

    print(f"\n\n{'#'*90}")
    print(f"  {tier_name.upper()}  —  {tier_label}")
    print(f"  Total rows: {len(df_tier):,}")
    print(f"{'#'*90}")

    for direction in ["bullish", "bearish"]:
        # Tables
        print_top(df_tier, direction, tier_label)
        # Summary stats
        print_summary(df_tier, direction, tier_label)
        # CPE vs tau curves
        plot_tau_curves(df_tier, direction, tier_name)

    # Heatmaps
    print(f"\n  Plotting heatmaps for {tier_name}...")
    for direction in ["bullish", "bearish"]:
        for (tp, tf, qx, qy) in HEATMAP_PARAMS.get(tier_name, []):
            if qy in tier_cfg["q_Y"]:
                plot_heatmap(df_tier, direction, tp, tf, qx, qy, tier_name)

print("\n\nDone.")
