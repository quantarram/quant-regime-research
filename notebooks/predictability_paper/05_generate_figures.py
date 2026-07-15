"""
Generates all figures for the predictability paper: three conceptual
schematics (Figs. 1-3) and six data-driven figures (Figs. 4-9), reading
from the JSON result files produced by scripts 01-04.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

C_CORR = "#1B7837"    # correlated -- green
C_DECORR = "#C0392B"  # decorrelated -- red
C_TOTAL = "#333333"
C_ACCENT = "#2166AC"  # blue accent
C_ACCENT2 = "#B8860B" # amber/gold accent


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


# ============================================================
# Figure 1 (schematic): atmospheric correlated/decorrelated
# energy decomposition and the predictability-limit crossing
# ============================================================
def fig1_schematic_crossing():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    t = np.linspace(0, 4, 400)
    ET = np.ones_like(t)
    Ec = np.exp(-0.6 * t)
    Ed = 1 - Ec

    ax.plot(t, ET, color=C_TOTAL, lw=1.4, ls="--", label=r"total, $E_T$")
    ax.plot(t, Ec, color=C_CORR, lw=2, label=r"correlated, $E_c(\Delta t)$")
    ax.plot(t, Ed, color=C_DECORR, lw=2, label=r"decorrelated, $E_D(\Delta t)$")

    mu = 0.5
    t_cross = -np.log(mu) / 0.6
    ax.axhline(mu, color="gray", lw=0.8, ls=":")
    ax.axvline(t_cross, color="gray", lw=0.8, ls=":")
    ax.plot([t_cross], [mu], "o", color="black", ms=5, zorder=5)
    ax.annotate(r"predictability limit $\Delta t_p$" "\n" r"$E_c/E_T = \mu$",
                xy=(t_cross, mu), xytext=(t_cross + 0.55, mu + 0.28),
                fontsize=8.5, ha="left",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.7))

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(r"lead time / lag, $\Delta t$")
    ax.set_ylabel("fraction of total energy")
    ax.set_title("Atmospheric predictability: monotonic\ncorrelated" r"$\to$" "decorrelated cascade", fontsize=10)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_schematic_atmospheric_crossing.png"))
    plt.close(fig)


# ============================================================
# Figure 2 (schematic): dyadic multifractal cascade / trace
# moment illustration
# ============================================================
def fig2_schematic_cascade():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    n_levels = 4
    rng = np.random.default_rng(7)

    base = np.abs(rng.normal(1, 0.6, 2 ** n_levels)) + 0.15
    levels = [base]
    cur = base
    for _ in range(n_levels):
        cur = cur.reshape(-1, 2).mean(axis=1)
        levels.append(cur)
    levels = levels[::-1]  # coarsest first

    y_positions = np.linspace(0.88, 0.12, n_levels + 1)
    max_val = max(v.max() for v in levels)

    for li, (vals, y) in enumerate(zip(levels, y_positions)):
        n = len(vals)
        xs = (np.arange(n) + 0.5) / n
        widths = 0.9 / n
        for x, v in zip(xs, vals):
            h = 0.09 * (v / max_val)
            rect = Rectangle((x - widths / 2, y - h / 2), widths * 0.85, h,
                              facecolor=C_ACCENT, edgecolor="white", alpha=0.55 + 0.35 * (v / max_val))
            ax.add_patch(rect)
        lam = 2 ** li
        ax.text(-0.03, y, rf"$\lambda={lam}$", fontsize=8.5, va="center", ha="right")

    for li in range(n_levels):
        n_parent = 2 ** li
        n_child = 2 ** (li + 1)
        y0, y1 = y_positions[li], y_positions[li + 1]
        for j in range(n_parent):
            xp = (j + 0.5) / n_parent
            for k in [2 * j, 2 * j + 1]:
                xc = (k + 0.5) / n_child
                ax.plot([xp, xc], [y0 - 0.02, y1 + 0.02], color="gray", lw=0.4, alpha=0.5)

    ax.set_xlim(-0.1, 1.02)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(r"Dyadic cascade: block-averaging a conservative flux $\varepsilon$" "\n"
                 r"$\langle \varepsilon_\lambda^q \rangle \sim \lambda^{K(q)}$", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_schematic_cascade.png"))
    plt.close(fig)


# ============================================================
# Figure 3 (schematic): K(q) [one-point] vs xi(q) [two-point]
# ============================================================
def fig3_schematic_kq_vs_xiq():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))

    ax = axes[0]
    rng = np.random.default_rng(3)
    t = np.arange(60)
    field = np.abs(rng.normal(1, 0.5, 60)) + 0.2
    ax.vlines(t, 0, field, color=C_ACCENT, lw=2, alpha=0.75)
    ax.axhline(np.mean(field), color=C_TOTAL, lw=1, ls="--", label=r"$\langle \varepsilon_\lambda \rangle$")
    hi = t[field > np.quantile(field, 0.85)]
    ax.scatter(hi, field[field > np.quantile(field, 0.85)], color=C_DECORR, zorder=5, s=18,
               label="rare high values\n(drive $K(q)$ at large $q$)")
    ax.set_title(r"$K(q)$: one-point moment" "\n" r"of $\varepsilon$ at fixed resolution $\lambda$", fontsize=9.5)
    ax.set_xlabel("position (single resolution)")
    ax.set_ylabel(r"$\varepsilon_\lambda$")
    ax.legend(fontsize=7.5, loc="upper right")

    ax = axes[1]
    x = np.linspace(0, 60, 400)
    walk = np.cumsum(rng.normal(0, 1, 400)) * 0.15
    walk -= walk[0]
    ax.plot(x, walk, color=C_ACCENT, lw=1.2)
    i0, i1 = 90, 260
    ax.plot([x[i0], x[i1]], [walk[i0], walk[i1]], color=C_DECORR, lw=1.8, marker="o", ms=5)
    ax.annotate(r"$\Delta f(\Delta t) = f(t{+}\Delta t) - f(t)$", xy=((x[i0] + x[i1]) / 2, (walk[i0] + walk[i1]) / 2),
                xytext=(6, walk.max() * 0.85), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.7))
    ax.set_title(r"$\xi(q)$: two-point moment" "\n" r"of the lag-$\Delta t$ fluctuation", fontsize=9.5)
    ax.set_xlabel(r"time, $t$")
    ax.set_ylabel(r"$f(t)$")

    fig.suptitle(r"$K(q)$ and $\xi(q)$ are distinct statistics, related by "
                 r"$\xi(q) = qH - K(q\eta) + qK(\eta)$", fontsize=9.5, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_schematic_Kq_vs_xiq.png"))
    plt.close(fig)


# ============================================================
# Figure 4 (data): DTM/TM analysis of raw SPY price
# ============================================================
def fig4_dtm_analysis():
    d = load("results_dtm.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3))

    qs = sorted(float(q) for q in d["K_q"].keys())
    Kvals = [d["K_q"][str(q)] for q in qs]
    ax = axes[0]
    ax.plot(qs, Kvals, "o-", color=C_ACCENT, ms=5, lw=1.3)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel(r"moment order, $q$")
    ax.set_ylabel(r"$K(q)$")
    ax.set_title("(a) Trace-moment scaling function", fontsize=9.5)

    etas = sorted(float(e) for e in d["K_of_eta"].keys())
    Ketas = [d["K_of_eta"][str(e)] for e in etas]
    ax = axes[1]
    ax.loglog(etas, Ketas, "o", color=C_ACCENT2, ms=6, zorder=5)
    log_e = np.log(etas)
    fit = d["alpha"] * log_e + (np.log(Ketas[4]) - d["alpha"] * log_e[4])
    ax.loglog(etas, np.exp(fit), "--", color=C_TOTAL, lw=1.2,
              label=rf"slope $=\alpha={d['alpha']:.2f}$" "\n" rf"$R^2={d['alpha_r2']:.4f}$")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$K(q_{\rm ref}{=}2,\ \eta)$")
    ax.set_title("(b) Double trace moment", fontsize=9.5)
    ax.legend(loc="upper left")

    fig.suptitle(rf"SPY raw price: $\alpha={d['alpha']:.2f}$, $C_1={d['C1']:.3f}$, "
                 rf"$H\approx{d['H_estimate']:.4f}$ (conservative)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_dtm_analysis_spy.png"))
    plt.close(fig)


# ============================================================
# Figure 5 (data): structure-function xi(q) for SPY
# ============================================================
def fig5_structure_function():
    d = load("results_structure_function.json")
    taus = d["taus"]
    qs_show = [0.5, 1.0, 2.0, 4.0, 6.0]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3))
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(qs_show)))
    for q, c in zip(qs_show, cmap):
        xi = d["xi_q"][str(q)]
        S0 = 1.0
        Svals = [S0 * (t ** xi) for t in taus]
        ax.loglog(taus, Svals, "o-", color=c, ms=3.5, lw=1, label=rf"$q={q:g}$")
    ax.set_xlabel(r"lag, $\tau$ (trading days)")
    ax.set_ylabel(r"$\langle|\Delta f(\tau)|^q\rangle$ (arb. units)")
    ax.set_title("(a) Structure functions", fontsize=9.5)
    ax.legend(fontsize=7.5)

    qs_all = sorted(float(q) for q in d["xi_q"].keys())
    xis = [d["xi_q"][str(q)] for q in qs_all]
    ax = axes[1]
    ax.plot(qs_all, xis, "o-", color=C_DECORR, ms=5, lw=1.3, label=r"$\xi(q)$ (measured)")
    ax.plot(qs_all, [qs_all[i] * xis[2] / qs_all[2] for i in range(len(qs_all))],
            "--", color="gray", lw=1, label="linear (monofractal) reference")
    ax.set_xlabel(r"moment order, $q$")
    ax.set_ylabel(r"$\xi(q)$")
    ax.set_title("(b) Concave scaling = multifractal", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("SPY raw price: two-point structure-function scaling exponent", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_structure_function_spy.png"))
    plt.close(fig)


# ============================================================
# Figure 6 (data): correlated/decorrelated curves, multi-panel
# ============================================================
def fig6_correlated_decorrelated_panels():
    d = load("results_correlated_decorrelated.json")
    tickers = ["SPY", "MSFT", "GLD", "^VIX"]
    labels = ["SPY", "MSFT", "GLD", "VIX"]

    fig, axes = plt.subplots(2, 4, figsize=(11, 5))
    for col, (tk, lab) in enumerate(zip(tickers, labels)):
        for row, q in enumerate(["2", "4"]):
            ax = axes[row, col]
            entry = d[tk][q]
            taus = np.arange(1, len(entry["D"]) + 1)
            C, D = np.array(entry["C"]), np.array(entry["D"])
            ax.fill_between(taus, C, D, where=(C >= D), color=C_CORR, alpha=0.15, interpolate=True)
            ax.fill_between(taus, C, D, where=(C < D), color=C_DECORR, alpha=0.12, interpolate=True)
            ax.plot(taus, C, color=C_CORR, lw=1.1, label="correlated")
            ax.plot(taus, D, color=C_DECORR, lw=1.1, label="decorrelated")

            top5 = entry["top5_all"][:5]
            if top5:
                tx = [p[0] for p in top5]
                ty = [max(C[t - 1], D[t - 1]) * 1.05 for t in tx]
                ax.scatter(tx, ty, marker="*", s=45, color=C_ACCENT2, edgecolor="black",
                           linewidth=0.4, zorder=6)

            ax.set_xscale("log")
            ax.set_xlim(1, 300)
            if row == 0:
                ax.set_title(lab, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"q={q}\nmoment value", fontsize=8.5)
            if row == 1:
                ax.set_xlabel("lag (trading days)", fontsize=8.5)
            ax.tick_params(labelsize=7.5)

    handles = [plt.Line2D([0], [0], color=C_CORR, lw=1.5, label="correlated $C(\\tau)$"),
               plt.Line2D([0], [0], color=C_DECORR, lw=1.5, label="decorrelated $D(\\tau)$"),
               plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=C_ACCENT2,
                          markeredgecolor="black", markersize=9, label="top-5 predictability lag")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Correlated and decorrelated structure functions, |$\\Delta$ price| field", fontsize=10.5)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "fig6_correlated_decorrelated_panels.png"))
    plt.close(fig)


# ============================================================
# Figure 7 (data): fine-grid predictability pockets vs lag,
# with CPE horizons marked
# ============================================================
def fig7_predictability_pockets():
    d = load("results_correlated_decorrelated.json")
    cpe_horizons = [21, 63, 126, 252]

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    for ax, tk, lab, color in zip(axes, ["SPY", "^VIX"], ["SPY", "VIX"], [C_ACCENT, "#6A3D9A"]):
        entry4 = d[tk]["4"]
        ratio = entry4["ratio"]
        taus = np.arange(1, len(ratio) + 1)
        ax.plot(taus, ratio, color=color, lw=1.1)
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        for h in cpe_horizons:
            ax.axvline(h, color=C_DECORR, lw=0.7, ls="--", alpha=0.6)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{lab}\ncorrelated fraction, $q{{=}}4$", fontsize=8.5)
        ax.tick_params(labelsize=8)

    for h in cpe_horizons:
        axes[0].text(h, 1.03, str(h), fontsize=7, color=C_DECORR, ha="center")
    axes[1].set_xlabel("lag (trading days)")
    fig.suptitle("Predictability pockets vs. lag, with CPE's own horizon grid (21/63/126/252d)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig7_predictability_pockets.png"))
    plt.close(fig)


# ============================================================
# Figure 8 (data): CPE cross-validation for SPY
# ============================================================
def fig8_cpe_cross_validation():
    d = load("results_cpe_cross_validation.json")
    counts = d["spy_signal_count_by_horizon"]
    horizons = sorted(counts.keys(), key=lambda x: int(x))
    vals = [counts[h] for h in horizons]
    colors = [C_DECORR if h == "252" else C_ACCENT for h in horizons]

    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar(horizons, vals, color=colors, width=0.6, edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center", fontsize=8.5)
    ax.set_xlabel("CPE horizon, $\\tau_{future}$ (trading days)")
    ax.set_ylabel("validated CPE signals for SPY")
    ax.set_title("SPY signal density by horizon (CPE framework)\n"
                  "vs. structure-function pocket at $\\tau\\approx241$d", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig8_cpe_cross_validation.png"))
    plt.close(fig)


# ============================================================
# Figure 9 (data): cross-instrument summary
# ============================================================
def fig9_cross_instrument_summary():
    d = load("results_correlated_decorrelated.json")
    tickers = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "AAPL", "MSFT", "JPM", "XOM",
               "GLD", "BTC-USD", "TLT", "EURUSD=X", "^VIX"]
    asset_class = ["Equity"] * 10 + ["Commodity", "Crypto", "Rates", "FX", "Volatility"]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    y_labels = []
    for i, tk in enumerate(tickers):
        e2 = d[tk]["2"]
        e4 = d[tk]["4"]
        lo, hi = e2["ratio_min"], e2["ratio_max"]
        # "meaningful" pocket: top tradeable gap must exceed 10% of the mean
        # decorrelated moment, not just be nominally positive -- excludes
        # numerically negligible cases like EUR/USD (see paper Sec 5.4/5.6)
        mean_D4 = np.mean(e4["D"]) if e4["D"] else 0
        top_gap4 = e4["top5_tradeable"][0][1] if e4["top5_tradeable"] else 0
        has_pocket = mean_D4 > 0 and (top_gap4 / mean_D4) > 0.1
        color = C_ACCENT if has_pocket else "#999999"
        ax.plot([lo, hi], [i, i], color=color, lw=4, solid_capstyle="butt", alpha=0.8)
        ax.plot([0.5], [i], "|", color="black", ms=8, mew=1.2)
        y_labels.append(f"{tk}  ({asset_class[i]})")

    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(y_labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("correlated fraction $C/(C+D)$ range across 300-day lag window, $q=2$")
    ax.set_title("Bounded predictability range by instrument\n"
                  "(blue = has tradeable pocket at $\\tau\\geq21$d, q=4; gray = none)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig9_cross_instrument_summary.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig1_schematic_crossing()
    fig2_schematic_cascade()
    fig3_schematic_kq_vs_xiq()
    fig4_dtm_analysis()
    fig5_structure_function()
    fig6_correlated_decorrelated_panels()
    fig7_predictability_pockets()
    fig8_cpe_cross_validation()
    fig9_cross_instrument_summary()
    print("All figures written to", FIG_DIR)
