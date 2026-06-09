"""
CPE Signal Score Plots
======================
Reads cpe_signal_scores.csv and produces publication-quality figures:
  1. signal_heatmap.png       — score_norm heatmap: Y x tau_future
  2. signal_bars_126.png      — ranked bar chart at tau_future=126
  3. signal_bars_252.png      — ranked bar chart at tau_future=252
  4. signal_summary.png       — combined 2-panel figure for manuscript

Run: python plot_signal_scores.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── ASSET CLASS MAP ───────────────────────────────────────────────────────────
ASSET_CLASS_MAP = {
    "equities":    ["SPY","QQQ","IWM","DIA","VTI","VT","EFA","EEM","VEA","VWO",
                    "EWJ","EWZ","FXI","INDA","EWY","XLK","XLF","XLE","XLV","XLI",
                    "XLP","XLY","XLU","XLRE","XLB","XLC","VTV","VUG","MTUM","USMV",
                    "QUAL","SIZE","ARKK","ICLN","ITB","XBI","SOXX",
                    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","BRK-B","XOM"],
    "crypto":      ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD",
                    "AVAX-USD","DOGE-USD","DOT-USD","LINK-USD","MATIC-USD","LTC-USD",
                    "BCH-USD","UNI-USD","ATOM-USD","IBIT","FBTC","GBTC","ETHE","BITB"],
    "commodities": ["GLD","IAU","SLV","PPLT","PALL","USO","BNO","UNG","UGA",
                    "CORN","WEAT","SOYB","CANE","NIB","JO","DJP","PDBC","DBC",
                    "GSG","CPER","DBB","GC=F","SI=F","CL=F","BZ=F","NG=F",
                    "HG=F","ZC=F","ZW=F","ZS=F"],
    "rates":       ["SHY","IEI","IEF","TLH","TLT","ZROZ","EDV","TIP","SCHP",
                    "LQD","HYG","JNK","EMB","AGG","BND","VCSH","VCIT","VCLT",
                    "MUB","MBB","^TNX","^TYX","^FVX","^IRX"],
    "fx":          ["EURUSD=X","GBPUSD=X","JPYUSD=X","CHFUSD=X","CADUSD=X",
                    "AUDUSD=X","NZDUSD=X","SGDUSD=X","INRUSD=X","BRLUSD=X",
                    "MXNUSD=X","ZARUSD=X","UUP","UDN",
                    "EURJPY=X","EURGBP=X","GBPJPY=X","AUDJPY=X","EURCHF=X"],
}
AC_COLORS = {
    "equities":    "#3B82F6",
    "crypto":      "#F59E0B",
    "commodities": "#10B981",
    "rates":       "#EF4444",
    "fx":          "#06B6D4",
    "other":       "#9E9E9E",
}
def get_ac(t):
    for ac, tl in ASSET_CLASS_MAP.items():
        if t in tl: return ac
    return "other"

# ── STYLE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#CCCCCC",
    "axes.grid":         True,
    "grid.color":        "#EEEEEE",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
})

BULL_COLOR = "#16A34A"
BEAR_COLOR = "#DC2626"
NEUT_COLOR = "#94A3B8"

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading cpe_signal_scores.csv...")
df = pd.read_csv("cpe_signal_scores.csv")
df["Y_class"] = df["Y"].apply(get_ac)
latest = df["latest_date"].iloc[0]
print(f"  Rows: {len(df)}  Assets: {df['Y'].nunique()}  Date: {latest}")

TAU_LIST = sorted(df["tau_future"].unique())

# ── PLOT 1: HEATMAP ───────────────────────────────────────────────────────────
print("Plot 1: Signal heatmap...")

# Keep only assets that have at least one non-neutral signal
active = df[df["score_norm"].abs() > 0.04]["Y"].unique()
df_h = df[df["Y"].isin(active)].copy()

# Pivot
pivot = df_h.pivot_table(index="Y", columns="tau_future",
                          values="score_norm", aggfunc="mean")
pivot = pivot.reindex(columns=sorted(pivot.columns))

# Sort rows: by asset class then by mean score
pivot["_class"] = pivot.index.map(get_ac)
class_order = ["crypto","equities","rates","commodities","fx","other"]
pivot["_class_order"] = pivot["_class"].map(
    {c: i for i, c in enumerate(class_order)})
pivot["_mean"] = pivot[[c for c in pivot.columns
                         if isinstance(c, (int, float))]].mean(axis=1)
pivot = pivot.sort_values(["_class_order","_mean"], ascending=[True, False])
pivot = pivot.drop(columns=["_class","_class_order","_mean"])

# Colormap: red-white-green
cmap = mcolors.LinearSegmentedColormap.from_list(
    "rw g", ["#DC2626","#FCA5A5","#F8FAFC","#86EFAC","#16A34A"])

fig, ax = plt.subplots(figsize=(10, max(8, len(pivot)*0.32)))
im = ax.imshow(pivot.values, cmap=cmap, vmin=-1, vmax=1,
               aspect="auto", interpolation="nearest")

# Axes
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"τ={c}d" for c in pivot.columns], fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=8.5)

# Colour y-tick labels by asset class
for tick, label in zip(ax.get_yticklabels(), pivot.index):
    tick.set_color(AC_COLORS.get(get_ac(label), "#555555"))
    tick.set_fontweight("bold")

# Annotate cells
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if np.isnan(val) or abs(val) < 0.04:
            continue
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                fontsize=7, fontweight="bold",
                color="white" if abs(val) > 0.35 else "#1F2937")

# Asset class separators
current_class, last_i = get_ac(pivot.index[0]), 0
for i, y in enumerate(pivot.index[1:], 1):
    nc = get_ac(y)
    if nc != current_class:
        ax.axhline(i - 0.5, color="#AAAAAA", linewidth=1.2, linestyle="--")
        current_class = nc

# Asset class legend on right
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks([])
patches = [mpatches.Patch(color=c, label=ac)
           for ac, c in AC_COLORS.items() if ac != "other"]
ax.legend(handles=patches, loc="upper right", fontsize=8,
          framealpha=0.9, title="Asset class", title_fontsize=8)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
cbar.set_label("Normalised signal score", fontsize=9)
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

ax.set_title(f"CPE Signal Score Heatmap  |  Latest date: {latest}\n"
             f"Score ∈ [−1, +1]: green = bullish, red = bearish, white = neutral",
             pad=12)
ax.set_xlabel("Forward horizon τ_future")
ax.set_ylabel("Predicted asset Y")

plt.tight_layout()
plt.savefig("signal_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: signal_heatmap.png")

# ── PLOT 2 & 3: RANKED BARS AT tau=126 AND tau=252 ────────────────────────────
print("Plots 2 & 3: Ranked bar charts...")

def ranked_bar(ax, tau, min_abs=0.04, title_suffix=""):
    sub = df[df["tau_future"] == tau].copy()
    sub = sub[sub["score_norm"].abs() >= min_abs].copy()
    sub = sub.sort_values("score_norm", ascending=True)

    if sub.empty:
        ax.text(0.5, 0.5, "No signals above threshold",
                ha="center", va="center", transform=ax.transAxes,
                color=NEUT_COLOR, fontsize=10)
        ax.set_title(f"τ_future = {tau}d  {title_suffix}")
        return

    colors = [BULL_COLOR if s > 0 else BEAR_COLOR
              for s in sub["score_norm"]]
    bars = ax.barh(range(len(sub)), sub["score_norm"],
                   color=colors, alpha=0.82, edgecolor="white",
                   linewidth=0.5, height=0.72)

    # Y-tick labels coloured by asset class
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["Y"], fontsize=8.5)
    for tick, y in zip(ax.get_yticklabels(), sub["Y"]):
        tick.set_color(AC_COLORS.get(get_ac(y), "#555555"))
        tick.set_fontweight("bold")

    # Value labels
    for i, (_, row) in enumerate(sub.iterrows()):
        s = row["score_norm"]
        x_off = 0.02 if s >= 0 else -0.02
        ha = "left" if s >= 0 else "right"
        ax.text(s + x_off, i, f"{s:+.3f}", va="center", ha=ha,
                fontsize=7.5, fontweight="bold",
                color=BULL_COLOR if s > 0 else BEAR_COLOR)

    ax.axvline(0, color="#374151", linewidth=1.2)
    ax.set_xlim(-1.15, 1.15)
    ax.set_xlabel("Normalised signal score")
    ax.set_title(f"τ_future = {tau}d  {title_suffix}", pad=8)
    ax.grid(axis="x", alpha=0.5)
    ax.grid(axis="y", alpha=0)

    # Firing counts annotation
    for i, (_, row) in enumerate(sub.iterrows()):
        n_b = int(row["n_bull_firing"])
        n_r = int(row["n_bear_firing"])
        if n_b + n_r > 0:
            ann = f"↑{n_b} ↓{n_r}"
            ax.text(1.12, i, ann, va="center", ha="right",
                    fontsize=7, color="#6B7280")

for tau, fname in [(126, "signal_bars_126.png"),
                   (252, "signal_bars_252.png")]:
    sub = df[df["tau_future"] == tau]
    n_sig = (sub["score_norm"].abs() >= 0.04).sum()
    fig_h = max(5, n_sig * 0.42 + 2)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ranked_bar(ax, tau,
               title_suffix=f"  ({n_sig} assets with |score| ≥ 0.04)")
    fig.suptitle(
        f"CPE-Weighted Signal Scores at {latest}  |  Forward horizon: {tau} trading days",
        fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

# ── PLOT 4: COMBINED SUMMARY (for manuscript) ─────────────────────────────────
print("Plot 4: Combined summary figure...")

# Pick the 2 most informative taus: 126 and 252
fig = plt.figure(figsize=(18, 10))
gs  = GridSpec(1, 3, figure=fig, width_ratios=[2, 1.3, 1.3],
               wspace=0.38)

# Left: heatmap (subset of most active assets)
ax0 = fig.add_subplot(gs[0])
# Top active assets at any horizon
scores_abs = df.groupby("Y")["score_norm"].apply(lambda x: x.abs().max())
top_active = scores_abs[scores_abs >= 0.08].index
df_sub = df[df["Y"].isin(top_active)]
piv2 = df_sub.pivot_table(index="Y", columns="tau_future",
                            values="score_norm", aggfunc="mean")
piv2 = piv2.reindex(columns=sorted(piv2.columns))
piv2["_class_order"] = piv2.index.map(get_ac).map(
    {c: i for i, c in enumerate(class_order)})
piv2["_mean"] = piv2[[c for c in piv2.columns
                       if isinstance(c,(int,float))]].mean(axis=1)
piv2 = piv2.sort_values(["_class_order","_mean"],
                          ascending=[True, False])
piv2 = piv2.drop(columns=["_class_order","_mean"])

im2 = ax0.imshow(piv2.values, cmap=cmap, vmin=-1, vmax=1,
                 aspect="auto", interpolation="nearest")
ax0.set_xticks(range(len(piv2.columns)))
ax0.set_xticklabels([f"τ={c}d" for c in piv2.columns], fontsize=8)
ax0.set_yticks(range(len(piv2.index)))
ax0.set_yticklabels(piv2.index, fontsize=8)
for tick, label in zip(ax0.get_yticklabels(), piv2.index):
    tick.set_color(AC_COLORS.get(get_ac(label), "#555555"))
    tick.set_fontweight("bold")
for i in range(len(piv2.index)):
    for j in range(len(piv2.columns)):
        val = piv2.values[i, j]
        if np.isnan(val) or abs(val) < 0.06:
            continue
        ax0.text(j, i, f"{val:+.2f}", ha="center", va="center",
                 fontsize=6.5, fontweight="bold",
                 color="white" if abs(val) > 0.35 else "#1F2937")
current_class = get_ac(piv2.index[0])
for i, y in enumerate(piv2.index[1:], 1):
    nc = get_ac(y)
    if nc != current_class:
        ax0.axhline(i - 0.5, color="#AAAAAA", linewidth=1.0,
                    linestyle="--")
        current_class = nc
cb = fig.colorbar(im2, ax=ax0, fraction=0.04, pad=0.02, shrink=0.8)
cb.set_label("Score", fontsize=8)
cb.set_ticks([-1, -0.5, 0, 0.5, 1])
cb.ax.tick_params(labelsize=7)
patches = [mpatches.Patch(color=c, label=ac)
           for ac, c in AC_COLORS.items() if ac != "other"]
ax0.legend(handles=patches, loc="upper right", fontsize=7,
           framealpha=0.9, title="Class", title_fontsize=7)
ax0.set_title("(a)  Signal score by asset and horizon", pad=8)
ax0.set_xlabel("Forward horizon")
ax0.set_ylabel("Predicted asset Y")

# Middle: ranked bars tau=126
ax1 = fig.add_subplot(gs[1])
ranked_bar(ax1, 126, min_abs=0.08,
           title_suffix="")
ax1.set_title("(b)  τ_future = 126 days", pad=8)

# Right: ranked bars tau=252
ax2 = fig.add_subplot(gs[2])
ranked_bar(ax2, 252, min_abs=0.08,
           title_suffix="")
ax2.set_title("(c)  τ_future = 252 days", pad=8)

fig.suptitle(
    f"Figure — CPE-Weighted Signal Scores at {latest}\n"
    r"$S_{Y,\tau_f} = \left(\sum_{k \in \mathcal{B}} w_k \cdot \mathbf{1}_k(t) - "
    r"\sum_{k \in \mathcal{R}} w_k \cdot \mathbf{1}_k(t)\right) / "
    r"\left(\sum_{k \in \mathcal{B}} w_k + \sum_{k \in \mathcal{R}} w_k\right)$"
    r",  $w_k = \mathrm{CPE}_k \times \mathrm{Lift}_k \times \ln(n_k)$",
    fontsize=10, fontweight="bold", y=1.03
)

plt.savefig("signal_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: signal_summary.png")

# ── PLOT 5: BUBBLE CHART ──────────────────────────────────────────────────────
print("Plot 5: Bubble chart (firing counts)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, tau in zip(axes, [126, 252]):
    sub = df[(df["tau_future"] == tau) &
             ((df["n_bull_firing"] > 0) | (df["n_bear_firing"] > 0))].copy()
    if sub.empty:
        ax.set_visible(False)
        continue

    sub["color"] = sub["Y_class"].map(AC_COLORS)
    sub["abs_score"] = sub["score_norm"].abs()

    sc = ax.scatter(sub["n_bull_firing"], sub["n_bear_firing"],
                    s=sub["abs_score"] * 800 + 40,
                    c=sub["color"], alpha=0.75, edgecolors="#374151",
                    linewidths=0.6)

    # Label points
    for _, row in sub.iterrows():
        if row["abs_score"] > 0.08:
            ax.annotate(row["Y"],
                        (row["n_bull_firing"], row["n_bear_firing"]),
                        xytext=(5, 3), textcoords="offset points",
                        fontsize=7.5, color="#1F2937")

    ax.set_xlabel("Number of bullish joint sets firing", fontsize=10)
    ax.set_ylabel("Number of bearish joint sets firing", fontsize=10)
    ax.set_title(f"τ_future = {tau}d  —  bubble size = |score_norm|",
                 fontsize=10, fontweight="bold")

    # Diagonal reference
    mx = max(sub["n_bull_firing"].max(), sub["n_bear_firing"].max()) + 1
    ax.plot([0, mx], [0, mx], color="#CCCCCC",
            linestyle="--", linewidth=1, label="equal bull/bear")

    patches = [mpatches.Patch(color=c, label=ac)
               for ac, c in AC_COLORS.items()
               if ac != "other" and ac in sub["Y_class"].values]
    ax.legend(handles=patches, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.4)

fig.suptitle(
    f"CPE Firing Conditions at {latest}\n"
    "Position = (bullish conditions firing, bearish conditions firing); "
    "assets above diagonal are net bullish, below are net bearish",
    fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig("signal_bubble.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: signal_bubble.png")

print("\nAll plots saved. Done.")
