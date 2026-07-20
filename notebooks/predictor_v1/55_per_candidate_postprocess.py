"""
Third design, per the user's explicit correction of 54's approach: do NOT
combine the four candidate models (climatology, credit_only, vix_only,
both) via any regression/blending. Instead, post-process EACH candidate
INDEPENDENTLY -- PDF/moment matching (2 parameters), then sequential
frequency-corrected quantile mapping on top -- fit entirely within the
selection period (pre-2022, never touching test). This produces four
separately post-processed prediction series per instrument. The test
period (2022 onward) is then used exactly once, purely to compare all
four post-processed candidates against actual prices -- no further
splitting, no forced "pick a winner" step (which would reintroduce the
selection-bias pattern already caught twice this session -- see
feedback-selection-bias-caught) -- just an honest, one-time report of how
each independently-corrected candidate performed.

Avoids both problems found in the (now abandoned) stacked-regression
version: no collinearity (each candidate is corrected using only its own
single time series vs actual, never regressed against the others), and
reuses the same robust/bounded quantile-mapping extrapolation already
fixed there.

Run: python 55_per_candidate_postprocess.py
Output: per_candidate_postprocess_results.json,
        pnl_plots/_PERCAND_heatmap.png
"""
import pandas as pd
import numpy as np
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUT_DIR, "pnl_plots")
SELECTION_END = pd.Timestamp("2022-01-01")
MIN_ROWS = 60

decisions = json.load(open(os.path.join(OUT_DIR, "master_model_final_decision.json")))
oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))
PROXY_TICKERS = ("IYR", "VOX")


def get_series(tkr):
    return (prices_proxy[tkr] if tkr in PROXY_TICKERS else prices[tkr]).dropna()


def moment_match_fit(raw, actual):
    raw_mean, raw_std = raw.mean(), max(raw.std(), 1e-8)
    act_mean, act_std = actual.mean(), actual.std()
    return {"raw_mean": float(raw_mean), "raw_std": float(raw_std), "act_mean": float(act_mean), "act_std": float(act_std)}


def moment_match_apply(raw, p):
    return p["act_mean"] + (raw - p["raw_mean"]) * (p["act_std"] / p["raw_std"])


def quantile_map_fit(raw, actual, n_quantiles=50):
    qs = np.linspace(0, 1, n_quantiles)
    raw_q, act_q = np.quantile(raw, qs), np.quantile(actual, qs)
    raw_q_u, idx = np.unique(raw_q, return_index=True)
    return {"raw_q": raw_q_u.tolist(), "act_q": act_q[idx].tolist()}


def quantile_map_apply(raw, params, max_extrap_multiple=2.0):
    raw_q_u, act_q_u = np.array(params["raw_q"]), np.array(params["act_q"])
    corrected = np.interp(raw, raw_q_u, act_q_u)
    if len(raw_q_u) >= 4:
        n_tail = max(2, len(raw_q_u) // 7)
        lo_slope = np.polyfit(raw_q_u[:n_tail], act_q_u[:n_tail], 1)[0]
        hi_slope = np.polyfit(raw_q_u[-n_tail:], act_q_u[-n_tail:], 1)[0]
        act_range = act_q_u[-1] - act_q_u[0]
        lo_cap, hi_cap = act_q_u[0] - max_extrap_multiple * act_range, act_q_u[-1] + max_extrap_multiple * act_range
        below, above = raw < raw_q_u[0], raw > raw_q_u[-1]
        ext_lo = np.clip(act_q_u[0] + (raw - raw_q_u[0]) * lo_slope, lo_cap, act_q_u[0])
        ext_hi = np.clip(act_q_u[-1] + (raw - raw_q_u[-1]) * hi_slope, act_q_u[-1], hi_cap)
        corrected = np.where(below, ext_lo, corrected)
        corrected = np.where(above, ext_hi, corrected)
    return corrected


def postprocess_candidate(sel_raw, sel_actual, test_raw):
    mm = moment_match_fit(sel_raw, sel_actual)
    mm_sel = moment_match_apply(sel_raw, mm)
    qm = quantile_map_fit(mm_sel, sel_actual)
    mm_test = moment_match_apply(test_raw, mm)
    return quantile_map_apply(mm_test, qm)


CANDIDATES = ["clim", "credit_only", "vix_only", "both"]
results = {}

for tkr in sorted(decisions.keys()):
    horizon = decisions[tkr]["horizon"]
    old_winner = decisions[tkr]["price_based_winner"]
    old_mape = None  # filled below once we know test alignment
    sub_th = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon)]

    series = get_series(tkr)
    series_pos = {d: i for i, d in enumerate(series.index)}

    def mape_from(dates, raw_returns):
        price_now = series.reindex(dates).values
        tidx = np.array([series_pos.get(pd.Timestamp(d), -1) for d in dates])
        valid = (tidx >= 0) & (tidx + horizon < len(series))
        tgt_price = series.values[tidx[valid] + horizon]
        pred_price = price_now[valid] * np.exp(np.asarray(raw_returns)[valid])
        return float(np.mean(np.abs(pred_price / tgt_price - 1)) * 100)

    cand_results = {}
    for cand in CANDIDATES:
        if cand == "clim":
            variant_source = "both" if "both" in sub_th["variant"].unique() else sub_th["variant"].iloc[0]
            vs = sub_th[sub_th["variant"] == variant_source][["date", "clim_q0.5", "y_true"]].dropna()
            vs = vs.rename(columns={"clim_q0.5": "raw"})
        else:
            vs = sub_th[sub_th["variant"] == cand][["date", "q0.5", "y_true"]].dropna()
            vs = vs.rename(columns={"q0.5": "raw"})
        if len(vs) == 0:
            continue
        vs = vs.sort_values("date")
        sel = vs[vs["date"] < SELECTION_END]
        test = vs[vs["date"] >= SELECTION_END]
        if len(sel) < MIN_ROWS or len(test) < MIN_ROWS:
            continue

        pp_test = postprocess_candidate(sel["raw"].values, sel["y_true"].values, test["raw"].values)
        cand_results[cand] = {
            "n_selection": int(len(sel)), "n_test": int(len(test)),
            "mape_raw": mape_from(test["date"].values, test["raw"].values),
            "mape_postprocessed": mape_from(test["date"].values, pp_test),
        }

    if len(cand_results) < 2:
        continue
    old_col = "clim" if old_winner == "climatology" else old_winner
    results[tkr] = {
        "horizon": horizon, "old_winner": old_winner,
        "old_single_winner_mape": cand_results.get(old_col, {}).get("mape_raw"),
        "candidates": cand_results,
    }

with open(os.path.join(OUT_DIR, "per_candidate_postprocess_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)

print(f"=== Per-candidate post-processing (fit entirely on selection, applied once to test), {len(results)} instruments ===")
for tkr in sorted(results, key=lambda t: results[t]["old_single_winner_mape"] or 999):
    r = results[tkr]
    line = f"  {tkr} (old_winner={r['old_winner']}@{r['horizon']}d, old_mape={r['old_single_winner_mape']:.2f}%): "
    parts = []
    for c in CANDIDATES:
        if c in r["candidates"]:
            cr = r["candidates"][c]
            flag = "+" if cr["mape_postprocessed"] < cr["mape_raw"] - 0.01 else ("-" if cr["mape_postprocessed"] > cr["mape_raw"] + 0.01 else "=")
            parts.append(f"{c}: {cr['mape_raw']:.1f}%->{cr['mape_postprocessed']:.1f}%[{flag}]")
    print(line + "  ".join(parts))

n_any_improved_vs_own_raw = sum(
    1 for r in results.values() for c in r["candidates"].values() if c["mape_postprocessed"] < c["mape_raw"] - 0.01
)
n_total_candidates = sum(len(r["candidates"]) for r in results.values())
n_best_pp_beats_old = sum(
    1 for r in results.values()
    if min(c["mape_postprocessed"] for c in r["candidates"].values()) < (r["old_single_winner_mape"] or 999) - 0.01
)
print(f"\n{n_any_improved_vs_own_raw}/{n_total_candidates} individual (instrument, candidate) post-processing corrections beat their own raw MAPE")
print(f"{n_best_pp_beats_old}/{len(results)} instruments have AT LEAST ONE post-processed candidate beating the old single-winner MAPE")
print("(NOTE: 'at least one of four beats the old number' is a descriptive observation, not a new selection rule -- ")
print(" picking whichever looks best here and redeploying it would reintroduce the exact selection-bias pattern already caught twice this session)")

# heatmap: rows=instruments (sorted by old MAPE), cols=candidates, cell=post-processed MAPE,
# color = improvement vs that candidate's own raw MAPE
tickers_sorted = sorted(results, key=lambda t: results[t]["old_single_winner_mape"] or 999)
fig, ax = plt.subplots(figsize=(8, 10))
grid = np.full((len(tickers_sorted), len(CANDIDATES)), np.nan)
delta = np.full((len(tickers_sorted), len(CANDIDATES)), np.nan)
for i, tkr in enumerate(tickers_sorted):
    for j, c in enumerate(CANDIDATES):
        if c in results[tkr]["candidates"]:
            cr = results[tkr]["candidates"][c]
            grid[i, j] = cr["mape_postprocessed"]
            delta[i, j] = cr["mape_postprocessed"] - cr["mape_raw"]

im = ax.imshow(np.clip(delta, -10, 10), cmap="RdYlGn_r", aspect="auto", vmin=-10, vmax=10)
for i in range(len(tickers_sorted)):
    for j in range(len(CANDIDATES)):
        if not np.isnan(grid[i, j]):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=8)
ax.set_xticks(range(len(CANDIDATES)))
ax.set_xticklabels(CANDIDATES)
ax.set_yticks(range(len(tickers_sorted)))
ax.set_yticklabels(tickers_sorted, fontsize=9)
ax.set_title("Per-candidate post-processing: test-period MAPE (%) after correction\n"
             "cell color = change vs that candidate's own raw MAPE (green=improved, red=worsened); text = final MAPE %")
fig.colorbar(im, ax=ax, label="MAPE change from post-processing (pp)", shrink=0.6)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "_PERCAND_heatmap.png"), dpi=115, bbox_inches="tight")
plt.close(fig)
print("\nDone.")
