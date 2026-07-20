"""
Final decision layer: treats climatology, credit_only, vix_only, and both
as four "slave" models under one master model, per instrument -- not
climatology-as-baseline-to-beat plus three challengers, four genuine
candidates on equal footing, exactly as instructed. Adds an absolute-
price-difference metric (MAE and MAPE, mean absolute percentage error) on
top of the existing FSS-based skill, both computed ONLY on the holdout
period (2022-01-01 onward), and uses the price-difference metric as the
deciding criterion for the master model's final per-instrument choice --
FSS is reported alongside for context, not as the decider.

Horizon is NOT re-selected here -- it stays anchored to the horizon chosen
in 38_fss_selection_holdout_split.py's selection-period FSS search (a
choice already made on data prior to 2022, i.e. legitimately unbiased).
Re-opening horizon selection using this new metric would just reintroduce
the same kind of multiple-comparisons bias this whole correction exists to
remove. What DOES change here: which of the four candidate MODELS (not
horizons) is picked for that horizon, now including climatology as an
eligible winner, decided by holdout-period price error.

No new model fits -- pure post-processing of oos_predictions_all.parquet
and selection_holdout_skill.csv, both already computed.

Run: python 41_final_decision_master_model.py
Output: master_model_final_decision.json, price_diff_plots/<TICKER>.png
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
PLOT_DIR = os.path.join(OUT_DIR, "price_diff_plots")
os.makedirs(PLOT_DIR, exist_ok=True)
HOLDOUT_START = pd.Timestamp("2022-01-01")

best_config = json.load(open(os.path.join(OUT_DIR, "best_config_selection_holdout.json")))
oos_all = pd.read_parquet(os.path.join(OUT_DIR, "oos_predictions_all.parquet"))
skill_df = pd.read_csv(os.path.join(OUT_DIR, "selection_holdout_skill.csv"))

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))
prices_proxy = pd.read_parquet(os.path.join(OUT_DIR, "sector_proxy_cache.parquet"))
PROXY_TICKERS = ("IYR", "VOX")


def get_series(tkr):
    return (prices_proxy[tkr] if tkr in PROXY_TICKERS else prices[tkr]).dropna()


def price_error_stats(sub, price_now, actual_target, quantile_col):
    pred_price = price_now * np.exp(sub[quantile_col].values)
    abs_err = np.abs(actual_target - pred_price)
    mape = np.abs(abs_err / actual_target) * 100
    return float(np.mean(abs_err)), float(np.mean(mape))


final_decisions = {}
for tkr in sorted(best_config.keys()):
    horizon = best_config[tkr]["horizon"]
    series = get_series(tkr)
    series_pos = {d: i for i, d in enumerate(series.index)}

    sub_th = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon)]
    if sub_th.empty:
        continue
    variants_present = sub_th["variant"].unique().tolist()

    candidates = {}  # name -> (mae, mape, fss_holdout_skill)
    clim_source = sub_th[sub_th["variant"] == "both"] if "both" in variants_present else \
        sub_th[sub_th["variant"] == variants_present[0]]
    clim_hold = clim_source[clim_source["date"] >= HOLDOUT_START].copy()
    if len(clim_hold) >= 30:
        price_now = series.reindex(clim_hold["date"]).values
        target_idx = (clim_hold["date"].map(series_pos) + horizon).astype(int)
        actual_target = series.values[target_idx.values]
        mae, mape = price_error_stats(clim_hold, price_now, actual_target, "clim_q0.5")
        candidates["climatology"] = (mae, mape, 0.0)  # skill above itself is 0 by definition

    for variant in ["credit_only", "vix_only", "both"]:
        vsub = sub_th[sub_th["variant"] == variant]
        vhold = vsub[vsub["date"] >= HOLDOUT_START].copy()
        if len(vhold) < 30:
            continue
        price_now = series.reindex(vhold["date"]).values
        target_idx = (vhold["date"].map(series_pos) + horizon).astype(int)
        actual_target = series.values[target_idx.values]
        mae, mape = price_error_stats(vhold, price_now, actual_target, "q0.5")
        skill_row = skill_df[(skill_df["ticker"] == tkr) & (skill_df["horizon"] == horizon) & (skill_df["variant"] == variant)]
        fss_skill = float(skill_row["holdout_skill"].iloc[0]) if len(skill_row) else float("nan")
        candidates[variant] = (mae, mape, fss_skill)

    if len(candidates) < 2:
        continue

    price_winner = min(candidates, key=lambda k: candidates[k][1])  # lowest MAPE
    fss_winner = max(candidates, key=lambda k: candidates[k][2])    # highest holdout skill

    final_decisions[tkr] = {
        "horizon": horizon,
        "candidates": {k: {"mae": v[0], "mape_pct": v[1], "fss_holdout_skill": v[2]} for k, v in candidates.items()},
        "price_based_winner": price_winner,
        "fss_based_winner": fss_winner,
        "agree": price_winner == fss_winner,
    }

    # bar chart: MAPE per candidate
    names = list(candidates.keys())
    mapes = [candidates[n][1] for n in names]
    colors = {"climatology": "gray", "credit_only": "tab:green", "vix_only": "tab:blue", "both": "tab:red"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, mapes, color=[colors[n] for n in names])
    winner_idx = names.index(price_winner)
    bars[winner_idx].set_edgecolor("black")
    bars[winner_idx].set_linewidth(2.5)
    for i, m in enumerate(mapes):
        ax.text(i, m, f"{m:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("MAPE on holdout period (%, lower is better)")
    ax.set_title(f"{tkr} @ {horizon}d -- price-error comparison, holdout period only\n"
                 f"master model picks: {price_winner} (black outline)")
    fig.tight_layout()
    safe_tkr = tkr.replace('=', '').replace('^', '')
    fig.savefig(os.path.join(PLOT_DIR, f"{safe_tkr}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

with open(os.path.join(OUT_DIR, "master_model_final_decision.json"), "w") as f:
    json.dump(final_decisions, f, indent=2, default=float)

print("=== Master model final decision per instrument (price-based, FSS shown for comparison) ===")
n_agree = sum(1 for d in final_decisions.values() if d["agree"])
n_clim_wins = sum(1 for d in final_decisions.values() if d["price_based_winner"] == "climatology")
for tkr in sorted(final_decisions, key=lambda t: final_decisions[t]["candidates"][final_decisions[t]["price_based_winner"]]["mape_pct"]):
    d = final_decisions[tkr]
    pw, fw = d["price_based_winner"], d["fss_based_winner"]
    pc = d["candidates"][pw]
    agree_str = "" if d["agree"] else f"  (FSS-based would have picked {fw})"
    print(f"  {tkr} @ {d['horizon']}d: MASTER PICKS {pw} -- MAPE={pc['mape_pct']:.2f}%, MAE={pc['mae']:.4f}{agree_str}")

print(f"\n{n_agree}/{len(final_decisions)} instruments: price-based and FSS-based choice agree")
print(f"{n_clim_wins}/{len(final_decisions)} instruments: climatology itself is the master model's best choice")
print("\nDone.")
