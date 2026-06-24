"""
scope_19_3_breakeven.py
========================
Scope 19.3: Transaction cost break-even analysis (Section 19.3 of the paper).

For each strategy, computes:
  - The number of one-way legs traded per year
  - The per-strategy per-leg break-even cost (bps) at which the Sharpe
    advantage over the no-tilt benchmark is exactly exhausted
  - Sharpe and return degradation as a function of one-way cost (0 to 50 bps)

Strategies covered:
  A. CPE five-sleeve hold-to-horizon (base sleeves, corrected weights)
     Results from paper Sections 16.1/17: 18.77% / Sharpe 1.224, 15 active days
  B. CPE cross-sectional top-quartile (Structure B)
     Results from paper Section 17.2: 24.58% / Sharpe 1.613, 30 active days
  C. Equal-weight (61-target episode-validated universe, no signal)
     Baseline: 14.45% / Sharpe 1.317, 0 active tilt days (always equal-weight)

Break-even definition:
  For strategy S vs benchmark B:
    excess_return = total_return_S - total_return_B
    TC_per_leg * n_legs * 100 (bps to decimal) = excess_return (decimal)
    => TC_break_even = excess_return / n_legs  (in bps)

'n_legs' counts one-way notional trades. For hold-to-horizon:
  - Each hold event = 2 legs (open + close)
  - 11 Equities holds * 2 = 22 one-way legs (15 days of non-neutral exposure,
    but 11 discrete hold events opened per paper Section 15.1)
For cross-sectional Structure B:
  - 30 firing days; on each day the top-quartile changes composition
  - Turnover is estimated at ~30% of the portfolio per firing day
    (conservative: assumes 15 of ~61 tickers rebalanced per day)
  - Total legs ~ 30 days * 2 sides * 0.30 turnover = 18 round-trip legs
    = 36 one-way legs (upper bound; actual is fewer since portfolio persists)
  - We report the range [18, 36] one-way legs for Structure B

Usage:
    python scope_19_3_breakeven.py
    python scope_19_3_breakeven.py --output breakeven_results.csv

No external data files required — all inputs are the paper's own reported
numbers, so this script runs without multiasset_prices.parquet.
"""

import argparse
import numpy as np
import pandas as pd

# ── PAPER-REPORTED RESULTS (from Sections 15-18) ──────────────────────────
# All Sharpe numbers use zero risk-free rate, matching the paper's convention.

RESULTS = {
    "no_tilt_benchmark_5sleeve": {
        "label": "No-tilt benchmark (5-sleeve, corrected weights)",
        "total_return_pct": 16.55,
        "sharpe": 1.089,
        "active_days": 0,
        "hold_events": 0,
        "note": "Section 16.1. Baseline for HTH comparison.",
    },
    "hth_5sleeve": {
        "label": "CPE 5-sleeve hold-to-horizon (corrected weights, base sleeves)",
        "total_return_pct": 18.77,
        "sharpe": 1.224,
        "active_days": 15,
        "hold_events": 11,          # paper Section 15.1: 11 distinct holds, all Equities
        "n_legs_low": 22,           # 11 holds * 2 (open + close)
        "n_legs_high": 22,          # deterministic — no estimation range needed
        "vs_benchmark": "no_tilt_benchmark_5sleeve",
        "note": "Section 16.1 / 15.1. Randomisation test: pct_exceeding 1.8%.",
    },
    "no_tilt_benchmark_xsect": {
        "label": "Equal-weight baseline (61-target episode-validated universe)",
        "total_return_pct": 14.45,
        "sharpe": 1.317,
        "active_days": 0,
        "hold_events": 0,
        "note": "Section 17.2. Baseline for cross-sectional comparison.",
    },
    "structure_b": {
        "label": "CPE cross-sectional top-quartile (Structure B)",
        "total_return_pct": 24.58,
        "sharpe": 1.613,
        "active_days": 30,
        "hold_events": 30,          # firing days, not discrete holds
        # Turnover estimation: top-quartile of 61 targets ≈ 15 tickers each day
        # Portfolio turns over roughly 30-60% of positions on each firing day
        # Conservative (low) estimate: 30% turnover / firing day
        # Upper (high) estimate: 60% turnover / firing day
        # n_legs = firing_days * 2_sides * turnover_fraction * n_tickers_in_portfolio
        # Simplified: each firing day incurs ~9 round-trips (30% of 15 active) = 18 one-way
        # Upper: ~18 round-trips (60%) = 36 one-way
        "n_legs_low": 18,           # 30 days * 30% turnover * 2 sides
        "n_legs_high": 54,          # 30 days * 60% turnover * 3 sides (entry+exit+new)
        "vs_benchmark": "no_tilt_benchmark_xsect",
        "note": "Section 17.2. Randomisation test: pct_exceeding 2.2%. "
                "Turnover per firing day is estimated (30-60% of ~15 active positions).",
    },
}


def breakeven_cost_bps(excess_return_pct: float, n_legs: float) -> float:
    """
    Return the one-way transaction cost in bps at which the strategy's
    total return excess over its benchmark is exactly zero.

    excess_return_pct: total return advantage in percentage points
    n_legs: number of one-way transaction legs over the evaluation year

    TC_break_even (bps) = (excess_return / n_legs) * 10000
    where excess_return is expressed as a decimal (excess_return_pct / 100).
    """
    if n_legs <= 0:
        return float("inf")
    excess_decimal = excess_return_pct / 100.0
    # TC costs are applied as: net_return = gross_return - n_legs * tc_per_leg
    # At break-even: net_excess = 0 → tc_per_leg = excess / n_legs
    return (excess_decimal / n_legs) * 10_000.0  # convert to bps


def sharpe_after_tc(strategy_key: str, tc_bps: float) -> dict:
    """
    Estimate Sharpe after transaction costs by subtracting total cost from
    returns, then recomputing Sharpe using the paper's reported annualised
    volatility (inferred from Sharpe = ann_ret / ann_vol → ann_vol = ann_ret/Sharpe).

    This is an approximation: TC is deducted from the annual return while
    holding volatility constant, which is reasonable for the small TC
    magnitudes considered here (TC cost << portfolio volatility).
    """
    s = RESULTS[strategy_key]
    gross_ret_pct = s["total_return_pct"]
    sharpe_gross = s["sharpe"]

    # Infer ann_vol from Sharpe = (total_return / sqrt(1yr)) / ann_vol
    # Since all strategies are 1-year (250 days), ann_ret ≈ total_return
    # Sharpe = ann_ret / ann_vol  →  ann_vol = ann_ret / Sharpe
    ann_ret_decimal = gross_ret_pct / 100.0
    ann_vol_decimal = ann_ret_decimal / sharpe_gross if sharpe_gross != 0 else 0.15

    tc_decimal = tc_bps / 10_000.0
    n_legs_mid = (s.get("n_legs_low", 0) + s.get("n_legs_high", 0)) / 2
    total_tc_decimal = tc_decimal * n_legs_mid

    net_ret_decimal = ann_ret_decimal - total_tc_decimal
    net_sharpe = net_ret_decimal / ann_vol_decimal if ann_vol_decimal > 0 else np.nan

    return {
        "strategy": s["label"],
        "tc_bps": tc_bps,
        "gross_ret_pct": gross_ret_pct,
        "net_ret_pct": round(net_ret_decimal * 100, 3),
        "ann_vol_pct": round(ann_vol_decimal * 100, 2),
        "gross_sharpe": sharpe_gross,
        "net_sharpe": round(net_sharpe, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="breakeven_results_19_3.csv")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  SCOPE 19.3 — TRANSACTION COST BREAK-EVEN ANALYSIS")
    print(f"  (paper Section 19.3; results from Sections 15–18)")
    print(f"{'='*72}")

    # ── SECTION 1: Break-even costs ────────────────────────────────────────
    print(f"\n  STRATEGY BREAK-EVEN COSTS")
    print(f"  {'─'*68}")
    print(f"  {'Strategy':<50} {'Excess ret':>10} {'Legs (low)':>10} {'Legs (hi)':>10} {'BE (low) bps':>13} {'BE (hi) bps':>12}")
    print(f"  {'─'*68}")

    be_rows = []
    for key, s in RESULTS.items():
        if "vs_benchmark" not in s:
            continue
        bench = RESULTS[s["vs_benchmark"]]
        excess_ret = s["total_return_pct"] - bench["total_return_pct"]
        n_low = s.get("n_legs_low", 0)
        n_high = s.get("n_legs_high", n_low)
        be_low = breakeven_cost_bps(excess_ret, n_high)   # more legs → lower BE per leg
        be_high = breakeven_cost_bps(excess_ret, n_low)   # fewer legs → higher BE per leg
        print(f"  {s['label']:<50} {excess_ret:>+10.2f}% {n_low:>10} {n_high:>10} {be_low:>12.1f} {be_high:>12.1f}")
        be_rows.append({
            "strategy": key,
            "label": s["label"],
            "gross_ret_pct": s["total_return_pct"],
            "benchmark_ret_pct": bench["total_return_pct"],
            "excess_ret_pct": round(excess_ret, 3),
            "n_legs_low": n_low,
            "n_legs_high": n_high,
            "breakeven_bps_conservative": round(be_low, 1),
            "breakeven_bps_generous": round(be_high, 1),
            "active_days": s["active_days"],
            "hold_events": s["hold_events"],
            "note": s["note"],
        })

    print(f"\n  Notes:")
    print(f"  - 'Legs (low/high)' = range of estimated one-way transaction legs over 2025.")
    print(f"  - 'BE (low)' = break-even cost when more legs are assumed (conservative).")
    print(f"  - 'BE (high)' = break-even cost when fewer legs are assumed (generous).")
    print(f"  - Typical institutional one-way cost: 1-5 bps (liquid ETFs/futures).")
    print(f"  - Retail brokerage one-way cost (no explicit commission): ~1-10 bps spread.")

    # ── SECTION 2: Sharpe degradation curve ───────────────────────────────
    print(f"\n\n  SHARPE DEGRADATION AS A FUNCTION OF ONE-WAY TRANSACTION COST")
    print(f"  (using mid-range of estimated legs; volatility held constant)")
    print(f"  {'─'*68}")

    tc_grid = [0, 1, 2, 5, 10, 15, 20, 30, 50]
    tradeable_strategies = ["hth_5sleeve", "structure_b"]

    header = f"  {'TC (bps)':>8}"
    for key in tradeable_strategies:
        lbl = RESULTS[key]["label"].replace("CPE ", "").replace(" (corrected weights, base sleeves)", "").replace(" (Structure B)", "")
        header += f"  {lbl[:28]:>28}"
    print(header)
    print(f"  {'─'*68}")

    degradation_rows = []
    for tc in tc_grid:
        row_str = f"  {tc:>8}"
        for key in tradeable_strategies:
            res = sharpe_after_tc(key, tc)
            net_sharpe = res["net_sharpe"]
            be_low = breakeven_cost_bps(
                RESULTS[key]["total_return_pct"] - RESULTS[RESULTS[key]["vs_benchmark"]]["total_return_pct"],
                RESULTS[key].get("n_legs_high", 1)
            )
            flag = " ← BE" if tc > be_low * 0.9 and tc < be_low * 1.1 else ""
            row_str += f"  {net_sharpe:>26.3f}{flag}"
            degradation_rows.append({
                "strategy": key,
                "tc_bps": tc,
                "gross_ret_pct": res["gross_ret_pct"],
                "net_ret_pct": res["net_ret_pct"],
                "net_sharpe": net_sharpe,
            })
        print(row_str)

    # ── SECTION 3: Context — typical market costs ─────────────────────────
    print(f"\n\n  CONTEXT: TYPICAL TRANSACTION COSTS FOR LIQUID ETFs")
    print(f"  {'─'*68}")
    cost_table = [
        ("SPY (S&P 500 ETF)",       "~0.3-0.5 bps",  "Extremely liquid; ~$40B ADV"),
        ("GC=F (Gold futures)",     "~0.5-1.0 bps",  "CME front-month, ~$30B ADV"),
        ("TLT (Bonds ETF)",         "~0.5-1.0 bps",  "~$3B ADV; wider spread than SPY"),
        ("BTC-USD (spot/perp)",     "~2-10 bps",     "FTX/Coinbase spread + borrow cost"),
        ("UUP (Dollar ETF)",        "~2-5 bps",      "Lower liquidity than equity ETFs"),
        ("LQD/XLY/XLI/XLP (ETFs)", "~0.5-2.0 bps",  "Liquid sector/credit ETFs"),
    ]
    print(f"  {'Instrument':<35} {'One-way cost':>14} {'Note'}")
    for name, cost, note in cost_table:
        print(f"  {name:<35} {cost:>14}  {note}")

    print(f"\n  For the 5-sleeve HTH strategy (22 one-way legs, all SPY/GC=F/TLT/BTC/UUP):")
    typical_cost = 3.0  # bps mid-range across the 5 sleeves
    total_drag = typical_cost * 22 / 10000 * 100  # pp
    print(f"    At {typical_cost:.0f} bps mid-range: total drag ≈ {total_drag:.2f}pp on 2.22pp excess return")
    print(f"    Remaining excess after TC: ≈ {2.22 - total_drag:.2f}pp")
    print(f"    Break-even: {breakeven_cost_bps(2.22, 22):.1f} bps (generous) to "
          f"{breakeven_cost_bps(2.22, 22):.1f} bps")

    print(f"\n  For cross-sectional Structure B (18-54 one-way legs across 61 targets):")
    xsect_excess = 24.58 - 14.45
    print(f"    Excess return over equal-weight: {xsect_excess:.2f}pp")
    for n in [18, 36, 54]:
        be = breakeven_cost_bps(xsect_excess, n)
        print(f"    At {n:>2} legs → break-even = {be:.1f} bps per one-way leg")

    # ── SAVE ──────────────────────────────────────────────────────────────
    be_df = pd.DataFrame(be_rows)
    deg_df = pd.DataFrame(degradation_rows)

    be_df.to_csv(args.output.replace(".csv", "_breakeven.csv"), index=False)
    deg_df.to_csv(args.output.replace(".csv", "_degradation_curve.csv"), index=False)

    print(f"\n\n  {'='*72}")
    print(f"  SUMMARY")
    print(f"  {'='*72}")
    print(f"\n  5-sleeve hold-to-horizon:")
    print(f"    Excess return:  {18.77-16.55:.2f}pp over no-tilt benchmark")
    print(f"    One-way legs:   22 (11 hold events * 2 sides)")
    print(f"    Break-even:     {breakeven_cost_bps(18.77-16.55, 22):.1f} bps per one-way leg")
    print(f"    Assessment:     COMFORTABLY above realistic ETF transaction costs (1-5 bps)")
    print(f"                    → signal survives realistic cost assumptions")
    print(f"\n  Cross-sectional Structure B:")
    print(f"    Excess return:  {24.58-14.45:.2f}pp over equal-weight baseline")
    print(f"    One-way legs:   18-54 (estimated; depends on daily turnover)")
    print(f"    Break-even:     {breakeven_cost_bps(10.13, 54):.1f}-{breakeven_cost_bps(10.13, 18):.1f} bps per one-way leg")
    print(f"    Assessment:     Also comfortably above realistic costs even at upper")
    print(f"                    bound estimate of 54 legs. However, cross-sectional")
    print(f"                    turnover estimation carries significant uncertainty —")
    print(f"                    actual trading costs require live tracking of positions.")
    print(f"\n  Saved: {args.output.replace('.csv','_breakeven.csv')}")
    print(f"  Saved: {args.output.replace('.csv','_degradation_curve.csv')}")
    print(f"  {'='*72}\n")


if __name__ == "__main__":
    main()
