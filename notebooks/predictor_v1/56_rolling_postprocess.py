"""
Rolling / adaptive post-processing, per the user's follow-up: given that
three static (fit-once-on-distant-selection-period, freeze-forever)
designs all failed for the same underlying reason -- the correction needed
drifts over time -- keep updating the correction periodically using only
recent, already-resolved (prediction, actual) pairs, instead of freezing
one fit for the whole multi-year test period.

Genuine walk-forward, no look-ahead: a prediction made on day d for
horizon H doesn't have a known outcome until day d+H has actually passed.
The trailing fitting window at each refit checkpoint only includes rows
whose TARGET date (d+H) is already before that checkpoint -- never rows
whose outcome isn't resolved yet. The window is free to pull from data
before 2022 at the start of the test period (there's no cold-start
problem) and gradually rolls entirely into test-period data as the walk
forward proceeds.

Design choices, disclosed:
  WINDOW_DAYS   = 252 (~1 trading year of resolved raw-vs-actual pairs) --
                  short enough to track drift (per the user's "based on the
                  past few days" emphasis on recency), long enough to fit a
                  2-parameter moment-match and a (smaller, 20-point)
                  quantile map without degenerating into noise.
  REFIT_EVERY   = 5 trading days -- "every few days," cheap to do since
                  these are closed-form fits (not model retraining).
  Correction    = same two-stage recipe as every other post-processing
                  script here: moment/PDF matching, then quantile mapping
                  with the same robust/bounded tail extrapolation already
                  fixed in 54/55.
Applied to each instrument's OWN currently-established winning candidate
(master_model_final_decision.json) -- not re-testing all 4 candidates x
rolling, to keep this run tractable; extending to all 4 is a natural
follow-up if this looks promising.

Run: python 56_rolling_postprocess.py
Output: rolling_postprocess_results.json, pnl_plots/_ROLLING_SUMMARY.png,
        pnl_plots/_ROLLING_JPM_detail.png, pnl_plots/_ROLLING_GLD_detail.png
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
TEST_START = pd.Timestamp("2022-01-01")
WINDOW_DAYS = 252
REFIT_EVERY = 5
N_QUANTILES = 20
MIN_WINDOW_ROWS = 60

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
    return raw_mean, raw_std, act_mean, act_std


def quantile_map_apply(raw, raw_q_u, act_q_u, max_extrap_multiple=2.0):
    corrected = np.interp(raw, raw_q_u, act_q_u)
    if len(raw_q_u) >= 4:
        n_tail = max(2, len(raw_q_u) // 5)
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


results = {}
detail_series = {}

for tkr in sorted(decisions.keys()):
    horizon = decisions[tkr]["horizon"]
    winner = decisions[tkr]["price_based_winner"]
    sub_th = oos_all[(oos_all["ticker"] == tkr) & (oos_all["horizon"] == horizon)]
    variants_present = sub_th["variant"].unique().tolist()
    if winner == "climatology":
        vsrc = sub_th[sub_th["variant"] == ("both" if "both" in variants_present else variants_present[0])]
        pred_col = "clim_q0.5"
    else:
        vsrc = sub_th[sub_th["variant"] == winner]
        pred_col = "q0.5"
    df = vsrc[["date", pred_col, "y_true"]].dropna().rename(columns={pred_col: "raw"}).sort_values("date").reset_index(drop=True)

    series = get_series(tkr)
    series_pos = {d: i for i, d in enumerate(series.index)}
    df = df[df["date"].isin(series_pos)].copy()
    df["date_idx"] = df["date"].map(series_pos)
    target_idx = df["date_idx"] + horizon
    valid = target_idx < len(series)
    df = df[valid.values].copy()
    df["target_date"] = series.index[target_idx[valid].values]

    test_df = df[df["date"] >= TEST_START].reset_index(drop=True)
    if len(test_df) < 100:
        continue

    # A row's outcome (y_true) is only resolved once its TARGET date (its own
    # date + horizon) has passed -- so the pool of usable trailing rows only
    # extends back to roughly (checkpoint - horizon) at the near edge, and the
    # window must reach further back than WINDOW_DAYS + horizon to actually
    # contain WINDOW_DAYS-worth of resolved prediction dates. Missing the
    # "+ horizon" term here was a real bug: for horizon=252d instruments it
    # left a window only ~8 trading days wide, well under MIN_WINDOW_ROWS,
    # so the correction silently never fired (confirmed: 0 refits, 100%
    # fallback-to-raw for every 252d instrument in the first run). Also scale
    # the window itself by horizon -- a "252-day window" at a 252-day horizon
    # contains only ~1 non-overlapping independent observation (the same
    # effective-N issue noted elsewhere in this project), nowhere near enough
    # to fit a stable quantile map.
    effective_window = max(WINDOW_DAYS, 4 * horizon)
    corrected = np.full(len(test_df), np.nan)
    n_refits, n_fallback_raw = 0, 0
    for start in range(0, len(test_df), REFIT_EVERY):
        end = min(start + REFIT_EVERY, len(test_df))
        checkpoint_date = test_df["date"].iloc[start]
        window_start = checkpoint_date - pd.Timedelta(days=int((effective_window + horizon) * 1.5))
        trail = df[(df["date"] >= window_start) & (df["date"] < checkpoint_date) & (df["target_date"] < checkpoint_date)]
        trail = trail.tail(effective_window)
        raw_chunk = test_df["raw"].values[start:end]
        if len(trail) < MIN_WINDOW_ROWS:
            corrected[start:end] = raw_chunk
            n_fallback_raw += (end - start)
            continue
        raw_mean, raw_std, act_mean, act_std = moment_match_fit(trail["raw"].values, trail["y_true"].values)
        mm_trail = act_mean + (trail["raw"].values - raw_mean) * (act_std / raw_std)
        qs = np.linspace(0, 1, N_QUANTILES)
        raw_q, act_q = np.quantile(mm_trail, qs), np.quantile(trail["y_true"].values, qs)
        raw_q_u, idx = np.unique(raw_q, return_index=True)
        act_q_u = act_q[idx]
        mm_chunk = act_mean + (raw_chunk - raw_mean) * (act_std / raw_std)
        corrected[start:end] = quantile_map_apply(mm_chunk, raw_q_u, act_q_u) if len(raw_q_u) >= 2 else mm_chunk
        n_refits += 1

    price_now = series.reindex(test_df["date"].values).values
    tgt_price = series.reindex(test_df["target_date"].values).values

    def mape(pred_ret, mask=None):
        m = mask if mask is not None else np.ones(len(pred_ret), dtype=bool)
        pred_price = price_now[m] * np.exp(pred_ret[m])
        return float(np.mean(np.abs(pred_price / tgt_price[m] - 1)) * 100)

    mid = pd.Timestamp("2024-01-01")
    early_mask = (test_df["date"] < mid).values
    late_mask = ~early_mask

    mape_raw = mape(test_df["raw"].values)
    mape_rolling = mape(corrected)
    results[tkr] = {
        "horizon": horizon, "winner": winner, "n_test": int(len(test_df)),
        "n_refits": n_refits, "n_fallback_raw_days": n_fallback_raw,
        "mape_raw": mape_raw, "mape_rolling_corrected": mape_rolling,
        "improved": bool(mape_rolling < mape_raw - 0.01),
        "mape_raw_early": mape(test_df["raw"].values, early_mask) if early_mask.sum() > 20 else None,
        "mape_rolling_early": mape(corrected, early_mask) if early_mask.sum() > 20 else None,
        "mape_raw_late": mape(test_df["raw"].values, late_mask) if late_mask.sum() > 20 else None,
        "mape_rolling_late": mape(corrected, late_mask) if late_mask.sum() > 20 else None,
    }
    if results[tkr]["mape_rolling_late"] is not None and results[tkr]["mape_raw_late"] is not None:
        results[tkr]["improved_late"] = bool(results[tkr]["mape_rolling_late"] < results[tkr]["mape_raw_late"] - 0.01)
    else:
        results[tkr]["improved_late"] = None
    if tkr in ("JPM", "GLD"):
        detail_series[tkr] = {
            "dates": test_df["date"].astype(str).tolist(),
            "price_now": price_now.tolist(), "target_date": test_df["target_date"].astype(str).tolist(),
            "actual_target_price": tgt_price.tolist(),
            "raw_price": (price_now * np.exp(test_df["raw"].values)).tolist(),
            "rolling_price": (price_now * np.exp(corrected)).tolist(),
        }

with open(os.path.join(OUT_DIR, "rolling_postprocess_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)

n_improved = sum(1 for r in results.values() if r["improved"])
n_improved_late = sum(1 for r in results.values() if r["improved_late"])
n_has_late = sum(1 for r in results.values() if r["improved_late"] is not None)
print(f"=== Rolling post-processing (refit every {REFIT_EVERY}d, trailing window scaled to horizon), {len(results)} instruments ===")
for tkr in sorted(results, key=lambda t: results[t]["mape_raw"]):
    r = results[tkr]
    flag = "IMPROVED" if r["improved"] else ""
    late_flag = "LATE-IMPROVED" if r["improved_late"] else ("late-worse" if r["improved_late"] is not None else "")
    print(f"  {tkr} ({r['winner']}@{r['horizon']}d): whole-period raw={r['mape_raw']:.2f}% rolling={r['mape_rolling_corrected']:.2f}% [{flag}]  |  "
          f"early(<24) raw={r['mape_raw_early']:.1f}% roll={r['mape_rolling_early']:.1f}%  "
          f"late(24+) raw={r['mape_raw_late']:.1f}% roll={r['mape_rolling_late']:.1f}% [{late_flag}]")
print(f"\n{n_improved}/{len(results)} improved whole-period; {n_improved_late}/{n_has_late} improved in the LATE (2024+, more genuinely rolling) sub-period")

tickers_sorted = sorted(results, key=lambda t: results[t]["mape_raw"] - results[t]["mape_rolling_corrected"], reverse=True)
fig, ax = plt.subplots(figsize=(9, 8))
y_pos = np.arange(len(tickers_sorted))
raw_vals = [results[t]["mape_raw"] for t in tickers_sorted]
roll_vals = [results[t]["mape_rolling_corrected"] for t in tickers_sorted]
ax.barh(y_pos - 0.18, raw_vals, height=0.35, color="#9aa1ad", label="Raw (uncorrected)")
ax.barh(y_pos + 0.18, roll_vals, height=0.35,
        color=["#2f8a4e" if r < o - 0.01 else "#b0492f" for o, r in zip(raw_vals, roll_vals)],
        label="Rolling post-processed")
ax.set_yticks(y_pos)
ax.set_yticklabels(tickers_sorted, fontsize=9)
ax.set_xlabel("Test-period MAPE (price, %) -- lower is better")
ax.set_title(f"Rolling post-processing (refit every {REFIT_EVERY}d, trailing {WINDOW_DAYS}d window) vs raw\n{n_improved}/{len(results)} improved")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "_ROLLING_SUMMARY.png"), dpi=115, bbox_inches="tight")
plt.close(fig)

# early vs late: does rolling improve once the window is genuinely fresh?
late_tickers = [t for t in results if results[t]["improved_late"] is not None]
late_tickers.sort(key=lambda t: (results[t]["mape_raw_late"] - results[t]["mape_rolling_late"]), reverse=True)
fig, ax = plt.subplots(figsize=(9, 8))
y_pos = np.arange(len(late_tickers))
early_delta = [results[t]["mape_rolling_early"] - results[t]["mape_raw_early"] for t in late_tickers]
late_delta = [results[t]["mape_rolling_late"] - results[t]["mape_raw_late"] for t in late_tickers]
ax.barh(y_pos - 0.18, early_delta, height=0.35, color=["#b0492f" if d > 0 else "#2f8a4e" for d in early_delta], alpha=0.5, label="Early (2022-2024): rolling window still mostly stale data")
ax.barh(y_pos + 0.18, late_delta, height=0.35, color=["#b0492f" if d > 0 else "#2f8a4e" for d in late_delta], label="Late (2024+): rolling window is genuinely fresh")
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(late_tickers, fontsize=9)
ax.set_xlabel("MAPE change from rolling correction (pp) -- negative = improved, left of zero")
ax.set_title("Does rolling post-processing improve once the window matures?\nEarly (faded) vs late (solid) sub-period, same instruments")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "_ROLLING_EARLY_LATE.png"), dpi=115, bbox_inches="tight")
plt.close(fig)

for tkr in detail_series:
    d = detail_series[tkr]
    dates = pd.to_datetime(d["dates"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, d["actual_target_price"], color="black", lw=1.3, label="Actual price")
    ax.plot(dates, d["raw_price"], color="tab:red", lw=1.0, ls="--", label="Raw prediction")
    ax.plot(dates, d["rolling_price"], color="tab:blue", lw=1.3, label=f"Rolling post-processed (refit every {REFIT_EVERY}d)")
    r = results[tkr]
    ax.set_title(f"{tkr}: rolling post-processing, test period\nMAPE: raw={r['mape_raw']:.2f}%, rolling={r['mape_rolling_corrected']:.2f}%")
    ax.set_ylabel("Price")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"_ROLLING_{tkr}_detail.png"), dpi=115, bbox_inches="tight")
    plt.close(fig)

print("\nDone.")
