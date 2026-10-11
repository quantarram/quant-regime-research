"""
paper17_figures.py
Figures for the Weather Impacts Dashboard paper (weather_impacts_paper_draft.md). Every number comes from the dashboard's own
ledgers (via its scoring code) or from weather_finance/gas_results.json; nothing is typed in by hand.

Run (from notebooks/), using the dashboard's virtualenv:
    ../../climate-energy-dashboard/.venv/bin/python paper17_figures.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.abspath(os.path.join(HERE, "..", "..", "climate-energy-dashboard"))
sys.path.insert(0, DASH)
os.chdir(DASH)
from weatherdash import ledger  # noqa: E402

OUT = os.path.join(HERE, "paper17_figs")
os.makedirs(OUT, exist_ok=True)
GREEN, GREY, RED, BLUE, AMBER = "#2e7d4f", "#8a94a3", "#c0392b", "#2f5f8f", "#c98a1b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
NUM = {}


def crossing(lead, fc, base):
    """First lead at which the forecast error is no longer below the baseline's."""
    for l, a, b in zip(lead, fc, base):
        if a >= b:
            return int(l)
    return None


def fig_demand():
    s = ledger.stats()
    bl = pd.DataFrame(s["by_lead"])
    br = pd.DataFrame(s["by_region"])
    NUM["demand"] = {"overall": s["overall"], "n_scored": s["n_scored"], "first_issue": s["first_issue"], "last_issue": s["last_issue"],
                     "crossover_lead": crossing(bl.lead, bl.mape_fc, bl.mape_clim), "by_region": s["by_region"]}
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    a = ax[0]
    a.plot(bl.lead, bl.mape_clim_raw, color=GREY, lw=1.6, ls=":", label="Climatology (1996-2025), raw")
    a.plot(bl.lead, bl.mape_clim, color=GREY, lw=2, label="Climatology (1996-2025), level-anchored")
    a.plot(bl.lead, bl.mape_fc, color=GREEN, lw=2.4, label="IFS forecast through the demand model")
    a.plot(bl.lead, bl.mape_oracle, color=BLUE, lw=1.6, ls="--", label="Same model on observed weather (floor)")
    a.set_xlabel("Forecast lead (days)")
    a.set_ylabel("Daily demand error, MAPE (%)")
    a.set_title(f"Six US grids, {s['n_scored']:,} scored forecast-days", fontsize=11)
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    a.set_ylim(0, 11.5)
    b = ax[1]
    br = br.sort_values("mape_fc")
    y = np.arange(len(br))
    b.barh(y + 0.27, br.mape_clim_raw, height=0.26, color="#c9ced6", label="Climatology, raw")
    b.barh(y, br.mape_clim, height=0.26, color=GREY, label="Climatology, anchored")
    b.barh(y - 0.27, br.mape_fc, height=0.26, color=GREEN, label="Forecast")
    b.set_yticks(y)
    b.set_yticklabels(br.region)
    b.invert_yaxis()
    b.set_xlabel("MAPE over all leads (%)")
    b.set_title("By region", fontsize=11)
    b.set_xlim(0, 13)
    b.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_demand.png"), dpi=170)
    plt.close(fig)


def fig_renewables():
    r = ledger.rstats_calibrated()
    bl = pd.DataFrame(r["by_lead"])
    NUM["renewables"] = {"n_scored": r["n_scored"],
                         "crossover": {k: crossing(g.lead, g.nmae_fc, g.nmae_clim) for k, g in bl.groupby("resource")},
                         "lead0": {k: g[g.lead == 0].iloc[0].to_dict() for k, g in bl.groupby("resource")}}
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=False)
    for a, res, col in zip(ax, ("wind", "solar"), (BLUE, AMBER)):
        g = bl[bl.resource == res]
        a.plot(g.lead, g.nmae_clim, color=GREY, lw=2, label="Climatology (2015-2025)")
        a.plot(g.lead, g.nmae_fc, color=col, lw=2.4, label="IFS forecast through the learned map")
        a.plot(g.lead, g.nmae_oracle, color="black", lw=1.4, ls="--", label="Same map on observed weather (floor)")
        a.set_title(res.capitalize(), fontsize=11)
        a.set_xlabel("Forecast lead (days)")
        a.set_ylabel("Error, % of mean output (NMAE)")
        a.set_ylim(0, g[["nmae_fc", "nmae_clim"]].values.max() * 1.32)
        a.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_renewables.png"), dpi=170)
    plt.close(fig)


def fig_gas():
    g = json.load(open(os.path.join(HERE, "weather_finance", "gas_results.json")))
    NUM["gas"] = g
    sg = pd.DataFrame(g["signals"])
    orc = pd.DataFrame(g["oracle"])
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    # left: tail hit rate, the CPE-style number, formation vs test
    a = ax[0]
    lab, vals, cols, ns = [], [], [], []
    for sig in ("rev", "anom"):
        for season in ("heat", "cool"):
            for per in ("formation", "test"):
                r = sg[(sg.signal == sig) & (sg.season == season) & (sg.period == per)].iloc[0]
                lab.append(f"{'Revision' if sig == 'rev' else 'Anomaly'}\n{'winter' if season == 'heat' else 'summer'}")
                vals.append(r.p_top_r1)
                cols.append(GREEN if per == "formation" else RED)
                ns.append(int(r.n_top))
    x = np.arange(len(vals))
    a.bar(x, vals, color=cols, width=0.75)
    a.axhline(0.2, color="black", lw=1, ls="--")
    a.text(len(vals) - 0.5, 0.205, "chance, 20%", ha="right", va="bottom", fontsize=8.5)
    for xi, v, n in zip(x, vals, ns):
        a.text(xi, v + 0.01, f"{v:.0%}\nn={n}", ha="center", fontsize=7.5, bbox=dict(fc="white", ec="none", pad=0.6))
    a.set_xticks(x[::2] + 0.5)
    a.set_xticklabels(lab[::2], fontsize=9)
    a.set_ylim(0, 0.45)
    a.set_ylabel("P(next-day gas return in its top 20% | signal in its top 20%)", fontsize=8.5)
    a.set_title("Pre-specified signals: formation (green) vs test (red)", fontsize=10.5)
    # right: perfect-foresight diagnostic, 5-day correlation
    b = ax[1]
    labs = ["Winter\nformation", "Winter\ntest", "Summer\nformation", "Summer\ntest"]
    order = [("heat", "formation"), ("heat", "test"), ("cool", "formation"), ("cool", "test")]
    x = np.arange(4)
    ob = [orc[(orc.season == s) & (orc.period == p)].iloc[0].obs_r5 for s, p in order]
    gf = [orc[(orc.season == s) & (orc.period == p)].iloc[0].gfs_r5 for s, p in order]
    b.bar(x - 0.2, ob, width=0.38, color="black", label="Observed weather (perfect foresight)")
    b.bar(x + 0.2, gf, width=0.38, color=BLUE, label="GFS 7-day forecast")
    b.axhline(0, color="black", lw=0.8)
    b.set_xticks(x)
    b.set_xticklabels(labs, fontsize=9)
    b.set_ylabel("Correlation of the 7-day degree-day anomaly\nwith the next 5-day gas return", fontsize=8.5)
    b.set_title("Even exact weather does not predict winter gas;\nsummer heat points the other way", fontsize=10.5)
    b.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_gas.png"), dpi=170)
    plt.close(fig)


def fig_further():
    cf = json.load(open(os.path.join(HERE, "weather_finance", "crop_futures_results.json")))
    gr = pd.DataFrame(json.load(open(os.path.join(HERE, "weather_finance", "gas_regional_results.json"))))
    sc = pd.read_parquet(os.path.join(HERE, "weather_finance", "scan_results.parquet"))
    sc["lift_f"] = sc.p_form - sc.base_form
    sc["lift_t"] = sc.p_test - sc.base_test
    adm = sc[(sc.eps_form >= 5) & (sc.p_form >= 0.30)]
    NUM["further"] = {"scan_configs": int(len(sc)), "scan_admitted": int(len(adm)), "scan_corr_lift": float(sc.lift_f.corr(sc.lift_t)),
                      "adm_p_form": float(adm.p_form.mean()), "adm_p_test": float(adm.p_test.mean()), "adm_base_test": float(adm.base_test.mean())}
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    a = ax[0]
    labs, x0 = ["Condition\nfell most", "Middle", "Condition\nrose most"], np.arange(3)
    for k, (crop, per, col, hatch) in enumerate([("CORN", "formation", GREEN, None), ("CORN", "test", GREEN, "//"),
                                                  ("SOYBEANS", "formation", AMBER, None), ("SOYBEANS", "test", AMBER, "//")]):
        t = cf[crop][per]["rel_by_dge_tercile"]
        v = [t["low"][0] * 1e4, t["mid"][0] * 1e4, t["high"][0] * 1e4]
        a.bar(x0 + (k - 1.5) * 0.2, v, width=0.19, color=col, hatch=hatch, edgecolor="white" if hatch is None else "black", lw=0.4,
              label=f"{crop.capitalize()} {'2015-20' if per == 'formation' else '2021-25'}")
    a.axhline(0, color="black", lw=0.8)
    a.set_xticks(x0)
    a.set_xticklabels(labs, fontsize=9)
    a.set_ylabel("Next-session return after the Monday release (bp)")
    a.set_title("Prices react to the weekly rating change", fontsize=10.5)
    a.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)
    b = ax[1]
    b.scatter(sc.lift_f * 100, sc.lift_t * 100, s=3, color=GREY, alpha=0.25, linewidths=0)
    b.scatter(adm.lift_f * 100, adm.lift_t * 100, s=7, color=RED, alpha=0.7, linewidths=0, label=f"{len(adm)} admitted in formation")
    b.axhline(0, color="black", lw=0.8)
    b.axvline(0, color="black", lw=0.8)
    b.set_xlabel("Formation lift in tail probability, 1996-2012 (points)")
    b.set_ylabel("Test lift, 2013-2025 (points)")
    b.set_title(f"{len(sc):,} weather-tail configurations,\nformation vs test: correlation {sc.lift_f.corr(sc.lift_t):+.2f}", fontsize=10.5)
    b.legend(frameon=False, fontsize=8, loc="upper right")
    c = ax[2]
    sel = [("COLD", "rev", "Cold\nrevision"), ("COLD", "anom", "Cold\nanomaly"), ("SOUTH", "rev", "South\nrevision"),
           ("SOUTH", "anom", "South\nanomaly")]
    xx = np.arange(len(sel))
    for k, (per, col) in enumerate([("formation", GREEN), ("test", RED)]):
        vals = [gr[(gr.region == r) & (gr.signal == sg) & (gr.period == per)].iloc[0].p_top_r1 for r, sg, _ in sel]
        ns = [int(gr[(gr.region == r) & (gr.signal == sg) & (gr.period == per)].iloc[0].n_top) for r, sg, _ in sel]
        c.bar(xx + (k - 0.5) * 0.38, vals, width=0.36, color=col, label="Formation 2021-23" if k == 0 else "Test 2024-26")
        for xi, v, n in zip(xx + (k - 0.5) * 0.38, vals, ns):
            c.text(xi, v + 0.008, f"{v:.0%}", ha="center", fontsize=7.5)
    c.axhline(0.2, color="black", lw=1, ls="--")
    c.text(len(sel) - 0.5, 0.205, "chance, 20%", ha="right", va="bottom", fontsize=8)
    c.set_xticks(xx)
    c.set_xticklabels([s_[2] for s_ in sel], fontsize=8.5)
    c.set_ylim(0, 0.42)
    c.set_ylabel("P(next-day gas in top 20% | signal in top 20%)", fontsize=9)
    c.set_title("Gas, regional degree-days: cold metros\n(winter heating), southern metros (summer cooling)", fontsize=10.5)
    c.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_further.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_demand()
    fig_renewables()
    fig_gas()
    fig_further()
    json.dump(NUM, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in NUM.items() if k != "gas"}, indent=1, default=float)[:3000])
