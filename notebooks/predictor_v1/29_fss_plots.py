"""
FSS plots per instrument, per explicit user request: y-axis = FSS, x-axis =
threshold (spanning both the downside and upside thresholds on one ordered
axis), one line per window size (legend), all models overlaid on the same
axes so they're directly comparable.

Two plot families, since they come from two different scripts/grids:
  - macro_interaction (24 instruments x 4 horizons): one PNG per instrument,
    4 subplots (one per horizon: 1/5/21/63d), each with climatology/
    baseline/macro_interaction as colored lines, window size as linestyle.
  - SPY distribution model (its own separate analysis, fixed 21d horizon,
    4 windows already the finest grain): one PNG, climatology/credit_only/
    vix_only/both as colored lines, window size as linestyle.

Run: python 29_fss_plots.py
Output: fss_plots/<TICKER>_macro_interaction.png (x24), fss_plots/SPY_distribution_model.png
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUT_DIR, "fss_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

WINDOW_STYLES = {21: "-", 63: "--", 126: ":", 252: "-."}
MODEL_COLORS_MACRO = {"climatology": "gray", "baseline": "tab:orange", "macro_interaction": "tab:blue"}
MODEL_COLORS_SPY = {"climatology": "gray", "credit_only": "tab:green", "vix_only": "tab:blue", "both": "tab:red"}


def grid_to_xy(grid, windows, lower_keys, upper_keys, key_fmt=str):
    """grid: {window(str): {'above'/'below': {threshold(str): fss}}} (as loaded
    from JSON, all keys stringified). Returns {window: (x_labels, y_values)}
    with x ordered from most-severe-downside to most-severe-upside."""
    out = {}
    for w in windows:
        wk = str(w)
        if wk not in grid:
            continue
        below = grid[wk]["below"]
        above = grid[wk]["above"]
        xs, ys = [], []
        for k in lower_keys:  # already ordered most-severe-first by caller
            kk = key_fmt(k)
            if kk in below:
                xs.append(f"{k}\n(down)")
                ys.append(below[kk])
        for k in upper_keys:  # ordered least-severe-first
            kk = key_fmt(k)
            if kk in above:
                xs.append(f"{k}\n(up)")
                ys.append(above[kk])
        out[w] = (xs, ys)
    return out


# ── SPY distribution model ──────────────────────────────────────────────
spy_path = os.path.join(OUT_DIR, "results_fss_rescore.json")
if os.path.exists(spy_path):
    spy = json.load(open(spy_path))
    windows = spy["windows"]
    lower_thr = sorted(spy.get("lower_thresholds", [-0.10, -0.075, -0.05]))  # most negative first
    upper_thr = sorted(spy.get("upper_thresholds", [0.05, 0.075, 0.10]))
    fig, ax = plt.subplots(figsize=(11, 6))
    for model, grid_holder in spy["results"].items():
        if "grid" not in grid_holder:
            continue
        xy_by_window = grid_to_xy(grid_holder["grid"], windows, lower_thr, upper_thr,
                                   key_fmt=lambda v: str(v))
        color = MODEL_COLORS_SPY.get(model, "black")
        for w, (xs, ys) in xy_by_window.items():
            ax.plot(xs, ys, color=color, linestyle=WINDOW_STYLES.get(w, "-"), marker="o", markersize=3,
                    label=f"{model}, window={w}d", alpha=0.85)
    ax.set_xlabel("Threshold (21d log-return magnitude)")
    ax.set_ylabel("FSS")
    ax.set_title("SPY distribution model: FSS vs. threshold, by model and window")
    ax.axhline(0.5, color="black", lw=0.5, ls=":", alpha=0.4)
    ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, "SPY_distribution_model.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
else:
    print(f"Skipping SPY plot -- {spy_path} not found")

# ── macro_interaction, one PNG per instrument, 4 subplots (horizons) ──────
macro_path = os.path.join(OUT_DIR, "results_fss_macro_interaction.json")
if os.path.exists(macro_path):
    macro = json.load(open(macro_path))
    windows = macro["windows"]
    lower_pcts = sorted(macro["lower_pcts"])   # e.g. [0.05, 0.10, 0.20] -- most severe (lowest pct) first
    upper_pcts = sorted(macro["upper_pcts"])   # e.g. [0.80, 0.90, 0.95] -- least severe first
    horizons = macro["horizons"]
    results = macro["results"]

    n_saved = 0
    for tkr in sorted(results.keys()):
        tkr_results = results[tkr]
        if not tkr_results:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
        any_panel = False
        for ax, h in zip(axes.flat, horizons):
            hk = str(h)
            if hk not in tkr_results:
                ax.set_visible(False)
                continue
            any_panel = True
            cell = tkr_results[hk]
            for model_key, label in [("grid_climatology", "climatology"),
                                      ("grid_baseline", "baseline"),
                                      ("grid_macro_interaction", "macro_interaction")]:
                grid = cell[model_key]
                xy_by_window = grid_to_xy(grid, windows, lower_pcts, upper_pcts,
                                           key_fmt=lambda v: str(v))
                color = MODEL_COLORS_MACRO[label]
                for w, (xs, ys) in xy_by_window.items():
                    ax.plot(xs, ys, color=color, linestyle=WINDOW_STYLES.get(w, "-"), marker="o", markersize=3,
                            label=f"{label}, w={w}d", alpha=0.85)
            ax.set_title(f"{tkr} @ {h}d horizon (n={cell['n_oos']})")
            ax.axhline(0.5, color="black", lw=0.5, ls=":", alpha=0.4)
            ax.tick_params(axis="x", labelsize=7)
        if not any_panel:
            plt.close(fig)
            continue
        axes[0, 0].set_ylabel("FSS")
        axes[1, 0].set_ylabel("FSS")
        for ax in axes[1, :]:
            ax.set_xlabel("Percentile threshold (down / up)")
        handles, labels = axes.flat[0].get_legend_handles_labels() if axes.flat[0].get_visible() else ([], [])
        if not handles:
            for a in axes.flat:
                if a.get_visible():
                    handles, labels = a.get_legend_handles_labels()
                    break
        fig.legend(handles, labels, fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle(f"{tkr}: FSS vs. threshold, by model and window, across horizons", fontsize=13)
        fig.tight_layout(rect=[0, 0, 0.86, 0.96])
        out_path = os.path.join(PLOT_DIR, f"{tkr.replace('=', '').replace('^', '')}_macro_interaction.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        n_saved += 1
    print(f"Saved {n_saved} per-instrument macro_interaction plots to {PLOT_DIR}/")
else:
    print(f"Skipping macro_interaction plots -- {macro_path} not found")

print("\nDone.")
