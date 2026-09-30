"""
elnino_sugar_analysis.py
========================================
Rebuild of the El Nino monsoon -> global sugar price analysis behind
"Does the 2026 El Nino Move Sugar Prices?" (Substack, 2026-07-02) and the
Paper9_Sugar_Price_El_Nino_2026 working draft, with three real changes:

1. Historical El Nino years are re-derived from NOAA's current ONI v6
   (ERSSTv6) table, not the older vintage used in the original draft.
   Under a consistent (JJA+JAS)/2 >= 0.5 threshold, 2004 (0.45) no longer
   clears the bar that it cleared under the older ONI vintage -- dropped,
   rather than mixed with a different data vintage to preserve a count.

2. All evidence is reported as real point estimates (hit rate, lift) and
   genuine leave-one-year-out replication -- no permutation p-values, no
   block-bootstrap confidence intervals.

3. 2026 is added as a real, resolved-where-possible ninth event: June's
   3-month-forward window (-> September) is fully resolved using real
   sugar price data; July's is not (the window doesn't close until
   October 31); August/September's are not yet flagged-and-resolved.
   FRED's official monthly sugar benchmark (PSUGAISAUSDM) has not yet
   published August/September 2026 -- ICE Sugar futures (SB=F) fill that
   gap, validated at 0.9996 level correlation / 0.9962 return correlation
   with the official benchmark over the prior 119 overlapping months.

Run: python elnino_sugar_analysis.py
"""
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ONI_THRESHOLD = 0.5


def load_oni():
    df = pd.read_csv("oni_v6_raw.csv")
    df["jjas_mean"] = df[["JJA", "JAS"]].mean(axis=1)
    return df.set_index("year")


def load_sugar_price():
    """FRED PSUGAISAUSDM through 2026-07, extended with SB=F monthly
    averages for any month FRED hasn't published yet (validated proxy)."""
    fred = pd.read_csv("sugar_fred_psugaisausdm.csv", header=None,
                        names=["date", "price"], skiprows=1)
    fred["date"] = pd.to_datetime(fred["date"])
    fred = fred.set_index("date")["price"].dropna()

    sb = yf.Ticker("SB=F").history(period="15y")["Close"]
    sb.index = sb.index.tz_localize(None)
    sb_m = sb.resample("MS").mean()

    common = sb_m.index.intersection(fred.index)
    level_corr = np.corrcoef(sb_m.reindex(common), fred.reindex(common))[0, 1]
    ret_corr = np.corrcoef(sb_m.reindex(common).pct_change().dropna(),
                            fred.reindex(common).pct_change().dropna())[0, 1]
    print(f"SB=F vs FRED PSUGAISAUSDM validation: level corr={level_corr:.4f}, "
          f"month-over-month return corr={ret_corr:.4f}, n={len(common)} months")

    last_fred = fred.index.max()
    fill_dates = sb_m.index[sb_m.index > last_fred]
    filled = pd.concat([fred, sb_m.reindex(fill_dates)])
    print(f"FRED published through {last_fred.date()}; filled {len(fill_dates)} "
          f"month(s) with SB=F: {[d.date().isoformat() for d in fill_dates]}")
    return filled.sort_index(), last_fred, level_corr, ret_corr


def forward_return(price: pd.Series, month: pd.Timestamp, k: int):
    """% price change from `month` to `month + k months`, using real
    published/filled monthly average prices. Returns None if unresolved."""
    target = month + pd.DateOffset(months=k)
    if month not in price.index or target not in price.index:
        return None
    return float((price[target] / price[month] - 1) * 100)


if __name__ == "__main__":
    oni = load_oni()
    price, fred_cutoff, level_corr, ret_corr = load_sugar_price()

    print(f"\n{'='*90}\nStep 1: which years clear the El Nino monsoon threshold "
          f"((JJA+JAS)/2 >= {ONI_THRESHOLD})?\n{'='*90}")
    candidates = [1990, 1991, 1992, 1997, 1998, 2002, 2004, 2009, 2015, 2023, 2026]
    flagged_years = []
    for y in candidates:
        row = oni.loc[y]
        jjas = row["jjas_mean"]
        pending = pd.isna(row["JAS"])
        val = row["JJA"] if pending else jjas
        clears = (row["JJA"] >= ONI_THRESHOLD) if pending else (jjas >= ONI_THRESHOLD)
        tag = " <- FLAGGED (JAS pending, using JJA alone: already clears)" if pending and clears else \
              (" <- FLAGGED" if clears else " -- does not clear")
        jas_str = "pending" if pending else f"{row['JAS']:+.1f}"
        jjas_str = "n/a" if pending else f"{jjas:+.2f}"
        print(f"  {y}: JJA={row['JJA']:+.1f} JAS={jas_str}  (JJA+JAS)/2={jjas_str}{tag}")
        if clears:
            flagged_years.append(y)
    print(f"\nFlagged years (consistent v6 ONI vintage): {flagged_years}")
    print("Original draft's list was [1991, 1997, 2002, 2004, 2009, 2015, 2023, 2026] -- "
          "2004 (0.45) does not clear under the current vintage and is dropped.")

    print(f"\n{'='*90}\nStep 2: real hit rates / lift, monsoon-month flag vs. forward "
          f"sugar price return (point estimates only, no p-values)\n{'='*90}")
    monsoon_months = [6, 7, 8, 9]
    flagged_dates = [pd.Timestamp(y, m, 1) for y in flagged_years for m in monsoon_months
                      if not (y == 2026 and m > 9)]
    all_dates = price.index[(price.index >= "1990-01-01")]

    for horizon in (1, 2, 3):
        rets = {d: forward_return(price, d, horizon) for d in all_dates}
        rets = pd.Series({d: r for d, r in rets.items() if r is not None})
        flagged_rets = rets.reindex([d for d in flagged_dates if d in rets.index]).dropna()
        for q in (0.55, 0.60, 0.65):
            thresh = rets.quantile(q)
            uncond = float((rets > thresh).mean())
            cpe = float((flagged_rets > thresh).mean()) if len(flagged_rets) else float("nan")
            lift = cpe / uncond if uncond > 0 else float("nan")
            print(f"  horizon={horizon}mo  q={q:.2f}  CPE={cpe:.3f}  unconditional={uncond:.3f}  "
                  f"lift={lift:.2f}x  n_flagged={len(flagged_rets)}")

    print(f"\n{'='*90}\nStep 3: leave-one-year-out real hit rates (2-month horizon, q=0.55)\n{'='*90}")
    horizon, q = 2, 0.55
    rets_all = pd.Series({d: forward_return(price, d, horizon) for d in all_dates})
    rets_all = rets_all.dropna()
    resolved_years = [y for y in flagged_years if y != 2026]  # 2026 handled separately below
    for test_year in resolved_years:
        train_years = [y for y in resolved_years if y != test_year]
        train_dates = [d for d in flagged_dates if d.year in train_years and d in rets_all.index]
        test_dates = [d for d in flagged_dates if d.year == test_year and d in rets_all.index]
        if not train_dates or not test_dates:
            continue
        thresh = rets_all.reindex([d for d in rets_all.index if d.year not in [test_year]]).quantile(q)
        test_rets = rets_all.reindex(test_dates)
        hit_rate = float((test_rets > thresh).mean())
        print(f"  {test_year}: held-out hit rate = {hit_rate:.2f}  (n={len(test_dates)} months)")

    print(f"\n{'='*90}\nStep 4: 2026 as the ninth event -- what's actually resolved\n{'='*90}")
    results_2026 = {}
    for m in monsoon_months:
        month = pd.Timestamp(2026, m, 1)
        results_2026[month.strftime("%Y-%m")] = {}
        for h in (1, 2, 3):
            r = forward_return(price, month, h)
            status = f"{r:+.1f}%" if r is not None else "not yet resolved"
            print(f"  {month.strftime('%b %Y')} + {h}mo forward: {status}")
            results_2026[month.strftime("%Y-%m")][f"{h}mo"] = r
        print()

    # ---- Save results JSON ----
    main_table = []
    for horizon in (1, 2, 3):
        rets = pd.Series({d: forward_return(price, d, horizon) for d in all_dates}).dropna()
        flagged_rets = rets.reindex([d for d in flagged_dates if d in rets.index]).dropna()
        for q in (0.55, 0.60, 0.65):
            thresh = rets.quantile(q)
            uncond = float((rets > thresh).mean())
            cpe = float((flagged_rets > thresh).mean()) if len(flagged_rets) else float("nan")
            main_table.append({"horizon_months": horizon, "quantile": q, "cpe": cpe,
                                "unconditional": uncond, "lift": cpe / uncond if uncond > 0 else None,
                                "n_flagged": int(len(flagged_rets))})

    loyo = []
    LOYO_Q = 0.55  # matches Step 3 above -- do not reuse the `q` loop variable from main_table
    for test_year in resolved_years:
        train_dates = [d for d in flagged_dates if d.year != test_year and d in rets_all.index]
        test_dates = [d for d in flagged_dates if d.year == test_year and d in rets_all.index]
        if not train_dates or not test_dates:
            continue
        thresh = rets_all.reindex([d for d in rets_all.index if d.year != test_year]).quantile(LOYO_Q)
        hit_rate = float((rets_all.reindex(test_dates) > thresh).mean())
        loyo.append({"year": test_year, "held_out_hit_rate_2mo_q55": hit_rate, "n_months": len(test_dates)})

    with open("elnino_sugar_results.json", "w") as f:
        json.dump({
            "flagged_years_current_vintage": flagged_years,
            "dropped_from_original": {"2004": {"jjas_mean": 0.45, "reason": "does not clear 0.5 under current ONI v6 vintage"}},
            "main_table": main_table,
            "leave_one_year_out": loyo,
            "results_2026": results_2026,
            "fred_cutoff": str(fred_cutoff.date()),
            "sbf_fred_correlation": {"level": level_corr, "return": ret_corr},
        }, f, indent=2, default=float)

    print("\nSaved: elnino_sugar_results.json")

    # ---- Figure 1: historical lift by horizon/quantile (point estimates only) ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [f"{r['horizon_months']}mo\nq{int(r['quantile']*100)}" for r in main_table]
    lifts = [r["lift"] for r in main_table]
    colors = ["#c96a3a" if r["horizon_months"] == 3 else "#4a7a9d" if r["horizon_months"] == 2 else "#9AA1AD" for r in main_table]
    bars = ax.bar(labels, lifts, color=colors)
    ax.axhline(1.0, color="#444", lw=1, ls="--")
    for b, v in zip(bars, lifts):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}x", ha="center", fontsize=9)
    ax.set_ylabel("Lift (conditional hit rate / unconditional hit rate)")
    ax.set_title(f"El Nino monsoon flag vs. forward global sugar price, {len(flagged_years)} historical years "
                 f"(1991-2023, real point estimates)\nNo permutation p-values -- real hit rates only")
    fig.tight_layout()
    fig.savefig("elnino_sugar_lift_plot.png", dpi=140)
    plt.close(fig)

    # ---- Figure 2: 2026 resolved outcomes ----
    # Pending months have no real value to plot -- a zero-height bar is
    # invisible regardless of color, which is misleading (looks like "no
    # data" rather than "unknown"). Draw a visible hatched placeholder of
    # a fixed nominal height instead, clearly distinct from a real value.
    PENDING_PLACEHOLDER_HEIGHT = 3.5
    fig, ax = plt.subplots(figsize=(10, 5.5))
    entries = [(mk, h, v) for mk, hz in results_2026.items() for h, v in hz.items()]
    rows = [f"{mk}\n+{h}" for mk, h, v in entries]
    x = np.arange(len(rows))

    resolved_x = [i for i, (mk, h, v) in enumerate(entries) if v is not None]
    resolved_v = [v for mk, h, v in entries if v is not None]
    pending_x = [i for i, (mk, h, v) in enumerate(entries) if v is None]

    ax.bar([x[i] for i in resolved_x], resolved_v,
           color=["#2E7D4F" if v > 0 else "#B0492F" for v in resolved_v])
    ax.bar([x[i] for i in pending_x], [PENDING_PLACEHOLDER_HEIGHT] * len(pending_x),
           color="none", edgecolor="#9AA1AD", hatch="////", lw=1.2)

    for i, (mk, h, v) in enumerate(entries):
        if v is not None:
            ax.text(x[i], v + 1 if v >= 0 else v - 3, f"{v:+.1f}%", ha="center", fontsize=8.5, fontweight="bold")
        else:
            ax.text(x[i], PENDING_PLACEHOLDER_HEIGHT + 1, "pending", ha="center", fontsize=8.5,
                    color="#666", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(rows)
    ax.axhline(0, color="#444", lw=1)
    ax.set_ylabel("Forward global sugar price return (%)")
    ax.set_title("2026: the ninth event, real outcomes as of today\n"
                 "Hatched grey = window not yet closed (needs Oct/Nov/Dec data) -- not a zero")
    fig.tight_layout()
    fig.savefig("elnino_sugar_2026_resolution_plot.png", dpi=140)
    plt.close(fig)
    print("Saved: elnino_sugar_lift_plot.png, elnino_sugar_2026_resolution_plot.png")

    # ---- Verification: harvest-season-lag design (Section 6 comparison) ----
    print(f"\n{'='*90}\nAppendix check: harvest-season-lag design (Oct-Mar following a flagged monsoon)\n{'='*90}")
    harvest_flagged = [pd.Timestamp(y, m, 1) for y in resolved_years for m in (10, 11, 12, 1, 2, 3)]
    # Oct-Dec belong to year y; Jan-Mar belong to y+1 -- both still "following year y's monsoon"
    harvest_flagged = []
    for y in resolved_years:
        for m in (10, 11, 12):
            harvest_flagged.append(pd.Timestamp(y, m, 1))
        for m in (1, 2, 3):
            harvest_flagged.append(pd.Timestamp(y + 1, m, 1))
    harvest_table = []
    for horizon in (3, 6, 12):
        rets = pd.Series({d: forward_return(price, d, horizon) for d in all_dates}).dropna()
        flagged_rets = rets.reindex([d for d in harvest_flagged if d in rets.index]).dropna()
        thresh = rets.quantile(0.60)
        uncond = float((rets > thresh).mean())
        cpe = float((flagged_rets > thresh).mean()) if len(flagged_rets) else float("nan")
        lift = cpe / uncond if uncond > 0 else float("nan")
        harvest_table.append({"horizon_months": horizon, "cpe": cpe, "unconditional": uncond,
                               "lift": lift, "n_flagged": int(len(flagged_rets))})
        print(f"  horizon={horizon}mo  q=0.60  CPE={cpe:.3f}  unconditional={uncond:.3f}  "
              f"lift={lift:.2f}x  n_flagged={len(flagged_rets)}")

    print("\nHarvest-lag leave-one-year-out (6-month horizon, q=0.60):")
    horizon = 6
    rets_h = pd.Series({d: forward_return(price, d, horizon) for d in all_dates}).dropna()
    harvest_loyo = []
    for test_year in resolved_years:
        test_dates = [pd.Timestamp(test_year, m, 1) for m in (10, 11, 12)] + \
                     [pd.Timestamp(test_year + 1, m, 1) for m in (1, 2, 3)]
        test_dates = [d for d in test_dates if d in rets_h.index]
        if not test_dates:
            continue
        thresh = rets_h.reindex([d for d in rets_h.index if d.year != test_year]).quantile(0.60)
        hit_rate = float((rets_h.reindex(test_dates) > thresh).mean())
        harvest_loyo.append({"year": test_year, "hit_rate": hit_rate, "n_months": len(test_dates)})
        print(f"  {test_year} harvest season: held-out hit rate = {hit_rate:.2f}  (n={len(test_dates)} months)")

    d = json.load(open("elnino_sugar_results.json"))
    d["harvest_lag_table"] = harvest_table
    d["harvest_lag_loyo"] = harvest_loyo
    json.dump(d, open("elnino_sugar_results.json", "w"), indent=2, default=float)
    print("\nUpdated elnino_sugar_results.json with harvest-lag verification")
