"""
Same single-panel-per-instrument plot as 31_fss_unified_plots.py, pointed at
the corrected results (32_fss_true_unified.py: own multifractal dynamics x
credit/vix regime, LightGBM, applied identically to all 27 instruments).

Run: python 33_fss_true_unified_plots.py
Output: fss_plots_true_unified/<TICKER>.png
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUT_DIR, "fss_plots_true_unified")
os.makedirs(PLOT_DIR, exist_ok=True)

WINDOW_STYLES = {21: "-", 63: "--", 126: ":", 252: "-."}
MODEL_COLORS = {"climatology": "gray", "credit_only": "tab:green", "vix_only": "tab:blue", "both": "tab:red"}
MODEL_ORDER = ["climatology", "credit_only", "vix_only", "both"]

data = json.load(open(os.path.join(OUT_DIR, "results_fss_true_unified.json")))
windows = data["windows"]
lower_thr = sorted(data["lower_thresholds"])
upper_thr = sorted(data["upper_thresholds"])
results = data["results"]


def grid_to_xy(grid, windows, lower_keys, upper_keys):
    out = {}
    for w in windows:
        wk = str(w)
        if wk not in grid:
            continue
        below, above = grid[wk]["below"], grid[wk]["above"]
        xs, ys = [], []
        for k in lower_keys:
            kk = str(k)
            if kk in below:
                xs.append(f"{k}\n(down)")
                ys.append(below[kk])
        for k in upper_keys:
            kk = str(k)
            if kk in above:
                xs.append(f"{k}\n(up)")
                ys.append(above[kk])
        out[w] = (xs, ys)
    return out


n_saved = 0
for tkr in sorted(results.keys()):
    grid = results[tkr]["grid"]
    n_oos = results[tkr]["n_oos"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for model in MODEL_ORDER:
        xy_by_window = grid_to_xy(grid[model], windows, lower_thr, upper_thr)
        color = MODEL_COLORS[model]
        for w, (xs, ys) in xy_by_window.items():
            ax.plot(xs, ys, color=color, linestyle=WINDOW_STYLES.get(w, "-"), marker="o", markersize=3,
                    label=f"{model}, window={w}d", alpha=0.85)
    ax.set_xlabel("Threshold (21d log-return magnitude)")
    ax.set_ylabel("FSS")
    ax.set_title(f"{tkr}: FSS vs. threshold, by model and window (n={n_oos['both']})")
    ax.axhline(0.5, color="black", lw=0.5, ls=":", alpha=0.4)
    ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"{tkr.replace('=', '').replace('^', '')}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    n_saved += 1

print(f"Saved {n_saved} plots to {PLOT_DIR}/")
