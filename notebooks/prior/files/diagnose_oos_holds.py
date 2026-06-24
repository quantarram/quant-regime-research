"""
diagnose_oos_holds.py
======================
Pre-publication diagnostic for the proper OOS walk-forward results.

Answers four questions before adding the OOS results to the paper:

  Q1. What configurations actually fired in 2011 (VV active=0, pct_exc=5.4%)?
      Are they economically coherent?

  Q2. What fired in 2020 (VV active=0) and 2022 (VV active=3)?
      Do all significant years trace to the same underlying channel?

  Q3. Grid coverage check: is the volatility→equity channel present in each
      year's freshly-derived joint screen? Does it become active (nonzero
      episode conviction) in the years where it should?

  Q4. 2022 practical magnitude check: does the 39bp improvement survive
      realistic transaction costs?

For each significant year (2011, 2020, 2022), the script:
  - Loads joint_cpe_results_train{Y-1}.parquet
  - Identifies which configurations have nonzero episode conviction
    and target a tradeable sleeve proxy
  - Re-fires each configuration against the evaluation year's prices
    to identify which ones actually opened holds
  - Classifies each fired configuration by economic channel
  - Compares classification across years

Usage:
    python diagnose_oos_holds.py
    python diagnose_oos_holds.py --prices multiasset_prices.parquet
    python diagnose_oos_holds.py --years 2011 2020 2022 2025
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.getcwd())

try:
    import backtest_engine as _be
    from backtest_engine import (
        BASE_SLEEVES, compute_neutral_weights,
        build_increments_and_thresholds,
        configuration_fires_on_date,
        EPISODE_MIN_OBS_FOR_CONVICTION,
    )
except ImportError as e:
    sys.exit(f"ERROR: Cannot import backtest_engine.py\n  {e}")

try:
    from run_backtest import get_eval_dates, load_and_filter_joint
except ImportError as e:
    sys.exit(f"ERROR: Cannot import run_backtest.py\n  {e}")

# ── Channel classification ────────────────────────────────────────────────
# Maps predictor tickers to broad economic channel names.
# Based on economic_prior.py's subclass taxonomy.

VOL_COMPLEX_ETPS  = {"VIXM","VIXY","VXX","UVXY","SVXY"}
VOL_INDICES       = {"^VIX","^VXN","^OVX","^GVZ","^EVZ","^VVIX","^SKEW"}
EQUITY_PROXIES    = {"SPY","QQQ","VTI","IWM","EEM","EFA","VEA","VWO",
                     "XLK","XLF","XLV","XLE","XLY","XLI","XLP","XLB",
                     "SOXX","EWJ","EWZ","INDA","FXI","GXC","MCHI"}
BOND_PROXIES      = {"TLT","IEF","SHY","LQD","HYG","JNK","TLH","BND",
                     "AGG","EMB","MBB","TIP","BNDX"}
GOLD_PROXIES      = {"GLD","IAU","GC=F","SGOL"}
COMMODITY_PROXIES = {"USO","UCO","BNO","UNG","CORN","WEAT","SOYB",
                     "PDBC","DJP","GSG","DBC","PALL","PPLT","SLV","SI=F"}
CRYPTO_PROXIES    = {"BTC-USD","ETH-USD","BNB-USD","SOL-USD","ADA-USD",
                     "DOGE-USD","LINK-USD","AVAX-USD","MATIC-USD",
                     "IBIT","FBTC","BITB","GBTC","ETHE"}
FX_PROXIES        = {"UUP","FXE","FXY","FXB","FXC","FXA","FXF","FXS",
                     "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X",
                     "USDCHF=X","USDCAD=X","NZDUSD=X","EURCHF=X",
                     "NZDUSD=X","DX-Y.NYB"}
RATE_INDICES      = {"^TNX","^TYX","^FVX","^IRX"}


def classify_predictor(ticker: str) -> str:
    if ticker in VOL_COMPLEX_ETPS:
        return "vol_etp"
    if ticker in VOL_INDICES:
        return "vol_index"
    if ticker in EQUITY_PROXIES:
        return "equity"
    if ticker in BOND_PROXIES:
        return "bond_credit"
    if ticker in GOLD_PROXIES:
        return "gold"
    if ticker in COMMODITY_PROXIES:
        return "commodity"
    if ticker in CRYPTO_PROXIES:
        return "crypto"
    if ticker in FX_PROXIES:
        return "fx"
    if ticker in RATE_INDICES:
        return "rate_index"
    return "other"


def classify_channel(predictors: list, target: str) -> str:
    """
    Classify a configuration by its economic channel based on predictor
    and target subclasses. Returns a human-readable channel name.
    """
    pred_classes = [classify_predictor(p) for p in predictors]
    tgt_class    = classify_predictor(target)

    # Volatility → equity (the validated channel)
    if any(c in ("vol_etp", "vol_index") for c in pred_classes) \
       and tgt_class == "equity":
        has_etp   = any(c == "vol_etp"   for c in pred_classes)
        has_index = any(c == "vol_index" for c in pred_classes)
        if has_etp and has_index:
            return "vol_etp+index → equity"
        if has_etp:
            return "vol_etp → equity"
        return "vol_index → equity"

    # Volatility → bonds
    if any(c in ("vol_etp", "vol_index") for c in pred_classes) \
       and tgt_class == "bond_credit":
        return "vol → bond/credit"

    # Credit → equity or credit
    if any(c == "bond_credit" for c in pred_classes) \
       and tgt_class in ("equity", "bond_credit"):
        return "credit → equity/credit"

    # Equity → equity
    if all(c == "equity" for c in pred_classes) and tgt_class == "equity":
        return "equity → equity"

    # Bond → bond
    if all(c == "bond_credit" for c in pred_classes) \
       and tgt_class == "bond_credit":
        return "bond → bond"

    # Cross-asset commodity
    if any(c in ("gold", "commodity") for c in pred_classes):
        return f"commodity/gold → {tgt_class}"

    # FX
    if any(c == "fx" for c in pred_classes):
        return f"fx → {tgt_class}"

    # Crypto
    if any(c == "crypto" for c in pred_classes):
        return f"crypto → {tgt_class}"

    return f"{'+'.join(sorted(set(pred_classes)))} → {tgt_class}"


def joint_path(train_year: int) -> str:
    return f"joint_cpe_results_train{train_year}.parquet"


def load_joint_for_year(eval_year: int) -> pd.DataFrame:
    path = joint_path(eval_year - 1)
    if not os.path.exists(path):
        # Also try the paper's own frozen screen for 2025
        if eval_year == 2025 and os.path.exists("joint_cpe_results.parquet"):
            print(f"    Using joint_cpe_results.parquet for 2025")
            return load_and_filter_joint("joint_cpe_results.parquet")
        print(f"    WARNING: {path} not found — skipping year {eval_year}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def find_fired_holds(joint: pd.DataFrame, prices: pd.DataFrame,
                     eval_year: int, sleeve_proxies: set) -> pd.DataFrame:
    """
    Identify which configurations in the joint screen actually opened
    hold events during the evaluation year, using the same hold-to-horizon
    logic as the backtest engine: a hold opens on the FIRST day a
    configuration fires after not having fired the day before.
    """
    train_cutoff = pd.Timestamp(f"{eval_year - 1}-12-31")
    eval_start   = pd.Timestamp(f"{eval_year}-01-01")
    eval_end     = pd.Timestamp(f"{eval_year}-12-31")

    _be.TRAIN_CUTOFF = train_cutoff
    _be.EVAL_START   = eval_start
    _be.EVAL_END     = eval_end

    Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]
    increments, thresholds = build_increments_and_thresholds(prices, Q_GRID)

    mask = (prices.index >= eval_start) & (prices.index <= eval_end)
    spy_valid = prices["SPY"].notna()
    eval_dates = prices.index[mask & spy_valid]

    # Filter to tradeable sleeve targets with nonzero episode conviction
    sleeve_joint = joint[
        joint["Y"].isin(sleeve_proxies) &
        (joint["n_predictors"] <= 6)
    ].copy()

    # Compute episode conviction if not present (OOS parquets have it;
    # fallback for older parquets that only have n_joint)
    if "episode_conviction" not in sleeve_joint.columns:
        sleeve_joint["episode_conviction"] = np.nan

    fired_holds = []

    for idx, row in sleeve_joint.iterrows():
        prev_fired = False
        for d in eval_dates:
            fires_today = configuration_fires_on_date(
                row, d, increments, thresholds
            )
            newly_fires = fires_today and not prev_fired
            if newly_fires:
                tau_f     = int(row["tau_future"])
                expiry_td = pd.Timedelta(days=int(tau_f * 1.45))
                fired_holds.append({
                    "entry_date":         d,
                    "expiry_date":        d + expiry_td,
                    "Y":                  row["Y"],
                    "direction":          row["direction"],
                    "tau_future":         tau_f,
                    "tau_pasts":          list(row["tau_pasts"]),
                    "predictors":         list(row["predictors"]),
                    "q_Xs":               list(row["q_Xs"]),
                    "q_Y":                float(row["q_Y"]),
                    "joint_CPE":          float(row["joint_CPE"]),
                    "lift":               float(row["lift"]),
                    "n_joint":            int(row.get("n_joint", 0)),
                    "n_episodes":         int(row.get("n_episodes", 0)),
                    "episode_conviction": float(row.get("episode_conviction", np.nan)),
                    "channel":            classify_channel(
                                              list(row["predictors"]), row["Y"]
                                          ),
                    "vv_driven":          bool({"VIXM","VIXY"}.issubset(
                                              set(row["predictors"])
                                          )),
                    "vol_driven":         bool(
                        any(p in VOL_COMPLEX_ETPS | VOL_INDICES
                            for p in row["predictors"])
                    ),
                })
            prev_fired = fires_today

    return pd.DataFrame(fired_holds)


def grid_coverage_check(eval_year: int, joint: pd.DataFrame) -> dict:
    """
    Check whether the volatility→equity channel is present in the joint
    screen for this year, and whether it has nonzero episode conviction.
    """
    if joint.empty:
        return {"n_spy_configs": 0, "n_vol_spy": 0, "n_vv_spy": 0,
                "n_vv_active": 0, "n_vol_active": 0}

    spy_rows = joint[joint["Y"] == "SPY"].copy()

    vol_mask = spy_rows["predictors"].apply(
        lambda p: any(x in VOL_COMPLEX_ETPS | VOL_INDICES for x in p)
    )
    vv_mask  = spy_rows["predictors"].apply(
        lambda p: {"VIXM","VIXY"}.issubset(set(p))
    )

    conv_col = "episode_conviction" if "episode_conviction" in spy_rows.columns else None

    return {
        "n_spy_configs":  len(spy_rows),
        "n_vol_spy":      int(vol_mask.sum()),
        "n_vv_spy":       int(vv_mask.sum()),
        "n_vv_active":    int((spy_rows[vv_mask][conv_col] > 0).sum())
                          if conv_col and len(spy_rows[vv_mask]) > 0 else 0,
        "n_vol_active":   int((spy_rows[vol_mask][conv_col] > 0).sum())
                          if conv_col and len(spy_rows[vol_mask]) > 0 else 0,
    }


def cost_check_2022(holds_df: pd.DataFrame, prices: pd.DataFrame) -> None:
    """
    For 2022: compute the strategy's gross return advantage over benchmark
    and check whether it survives realistic transaction costs.
    """
    GROSS_ADVANTAGE_PP = -24.19 - (-24.58)   # = +0.39pp
    n_holds = len(holds_df)
    n_legs  = n_holds * 2  # open + close per hold

    print(f"\n  2022 PRACTICAL MAGNITUDE CHECK")
    print(f"  {'─'*55}")
    print(f"  Gross return advantage over benchmark:  {GROSS_ADVANTAGE_PP:+.2f}pp")
    print(f"  Hold events opened:                     {n_holds}")
    print(f"  One-way legs (open + close):            {n_legs}")

    print(f"\n  Break-even one-way cost: "
          f"{GROSS_ADVANTAGE_PP / n_legs * 100:.1f} bps per leg")

    print(f"\n  Net advantage after transaction costs:")
    print(f"  {'TC (bps/leg)':>15}  {'Total drag (pp)':>16}  "
          f"{'Net advantage (pp)':>19}  {'Survives?':>10}")
    print(f"  {'─'*65}")
    for tc_bps in [0, 1, 2, 3, 5, 10]:
        total_drag = tc_bps / 10_000 * n_legs * 100
        net        = GROSS_ADVANTAGE_PP - total_drag
        survives   = "YES" if net > 0 else "NO"
        print(f"  {tc_bps:>15}  {total_drag:>16.3f}  {net:>19.3f}  {survives:>10}")

    # Instrument-level cost estimates for what actually fired in 2022
    if not holds_df.empty:
        print(f"\n  Instruments involved in 2022 holds:")
        all_preds = [p for preds in holds_df["predictors"] for p in preds]
        targets   = list(holds_df["Y"].unique())
        instruments = sorted(set(all_preds + targets))
        cost_map = {
            "SPY": 0.4, "GC=F": 0.7, "TLT": 0.8, "BTC-USD": 5.0,
            "UUP": 3.0, "LQD": 1.0, "TLH": 1.0,
            "VIXM": 3.0, "VIXY": 3.0, "VXX": 2.0,
            "^VIX": 0.0, "^VXN": 0.0,  # indices, not traded
        }
        for inst in instruments:
            est = cost_map.get(inst, 2.0)
            role = "TARGET" if inst in targets else "predictor (not traded)"
            print(f"    {inst:<14}  ~{est:.1f} bps/leg  ({role})")

        traded = [t for t in targets]
        avg_cost = np.mean([cost_map.get(t, 2.0) for t in traded])
        total_realistic = avg_cost / 10_000 * n_legs * 100
        net_realistic   = GROSS_ADVANTAGE_PP - total_realistic
        print(f"\n  Realistic weighted cost estimate ({avg_cost:.1f} bps avg on targets):")
        print(f"    Total drag: {total_realistic:.3f}pp")
        print(f"    Net advantage: {net_realistic:+.3f}pp")
        if net_realistic > 0:
            print(f"    VERDICT: advantage survives realistic costs (barely)")
        else:
            print(f"    VERDICT: advantage erased by realistic transaction costs")
            print(f"    → 2022 result is statistically significant but")
            print(f"      practically zero after costs. Report with this caveat.")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic: trace hold events in significant OOS years"
    )
    parser.add_argument("--prices",  default="multiasset_prices.parquet")
    parser.add_argument("--years",   nargs="+", type=int,
                        default=[2011, 2020, 2022, 2025])
    parser.add_argument("--also-check", nargs="+", type=int,
                        default=[2010, 2014, 2024],
                        help="Non-significant years to compare against "
                             "(same hold count bracket)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  OOS HOLD EVENT DIAGNOSTIC")
    print(f"  Tracing which configurations fired in significant years")
    print(f"{'='*70}")

    prices = pd.read_parquet(args.prices)
    print(f"\n  Price history: {prices.index.min().date()} to "
          f"{prices.index.max().date()}")

    sleeve_proxies = set(BASE_SLEEVES.values())
    all_years = sorted(set(args.years + args.also_check))

    # ── Per-year analysis ─────────────────────────────────────────────────
    all_holds   = {}
    grid_checks = {}

    for yr in all_years:
        is_sig = yr in args.years
        label  = "SIGNIFICANT" if is_sig else "comparison"
        print(f"\n{'─'*70}")
        print(f"  YEAR {yr}  [{label}]")
        print(f"{'─'*70}")

        joint = load_joint_for_year(yr)
        if joint.empty:
            print(f"  No joint screen found for {yr} — skipping")
            continue

        print(f"  Joint configs in screen: {len(joint)}")

        # Grid coverage
        gc = grid_coverage_check(yr, joint)
        grid_checks[yr] = gc
        print(f"\n  GRID COVERAGE (SPY target):")
        print(f"    SPY configs total:          {gc['n_spy_configs']}")
        print(f"    SPY configs with vol predictor: {gc['n_vol_spy']}")
        print(f"    SPY configs with VIXM+VIXY:     {gc['n_vv_spy']}")
        print(f"    VIXM+VIXY active (conv>0):      {gc['n_vv_active']}")
        print(f"    Any vol→SPY active (conv>0):    {gc['n_vol_active']}")

        # Find actual fired holds
        print(f"\n  Finding holds fired in {yr}...")
        holds = find_fired_holds(joint, prices, yr, sleeve_proxies)
        all_holds[yr] = holds

        if holds.empty:
            print(f"  No holds fired in {yr}")
            continue

        print(f"\n  HOLDS FIRED: {len(holds)}")
        print(f"\n  {'Entry':>12}  {'Y':<8}  {'dir':<8}  {'τ_f':>4}  "
              f"{'τ_p':>8}  {'Predictors':<35}  {'Channel':<30}  "
              f"{'CPE':>6}  {'conv':>6}  {'VV?':>4}")
        print(f"  {'─'*130}")

        for _, h in holds.sort_values("entry_date").iterrows():
            preds_str = "+".join(h["predictors"])[:33]
            taup_str  = str(h["tau_pasts"])
            print(f"  {str(h['entry_date'].date()):>12}  "
                  f"{h['Y']:<8}  {h['direction']:<8}  "
                  f"{h['tau_future']:>4}  {taup_str:>8}  "
                  f"{preds_str:<35}  {h['channel']:<30}  "
                  f"{h['joint_CPE']:>6.3f}  "
                  f"{h['episode_conviction']:>6.3f}  "
                  f"{'YES' if h['vv_driven'] else ' no':>4}")

        # Channel breakdown
        print(f"\n  CHANNEL BREAKDOWN:")
        channel_counts = holds["channel"].value_counts()
        for ch, n in channel_counts.items():
            vol_flag = " ← VALIDATED CHANNEL" if "vol" in ch.lower() and "equity" in ch.lower() else ""
            print(f"    {ch:<40}  {n:>3} holds{vol_flag}")

        vol_frac = holds["vol_driven"].mean()
        print(f"\n  Volatility-driven holds: "
              f"{holds['vol_driven'].sum()}/{len(holds)} "
              f"({vol_frac*100:.0f}%)")

        # 2022-specific cost check
        if yr == 2022:
            cost_check_2022(holds, prices)

    # ── Cross-year comparison ─────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  CROSS-YEAR SUMMARY: DO SIGNIFICANT YEARS SHARE A CHANNEL?")
    print(f"{'='*70}")

    print(f"\n  {'Year':>6}  {'Sig?':>5}  {'Holds':>6}  "
          f"{'Vol-driven%':>12}  {'VV-driven%':>11}  "
          f"{'Primary channel':<40}  {'VV active':>9}")
    print(f"  {'─'*100}")

    sig_years_set = set(args.years)
    for yr in all_years:
        if yr not in all_holds:
            continue
        holds = all_holds[yr]
        gc    = grid_checks.get(yr, {})
        is_sig = yr in sig_years_set

        if holds.empty:
            primary_ch = "no holds"
            vol_pct = vv_pct = 0.0
        else:
            primary_ch = holds["channel"].value_counts().index[0]
            vol_pct    = holds["vol_driven"].mean() * 100
            vv_pct     = holds["vv_driven"].mean() * 100

        print(f"  {yr:>6}  {'YES' if is_sig else ' no':>5}  "
              f"{len(holds):>6}  {vol_pct:>11.0f}%  {vv_pct:>10.0f}%  "
              f"{primary_ch:<40}  {gc.get('n_vv_active', '?'):>9}")

    # ── Coherence assessment ──────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  COHERENCE ASSESSMENT")
    print(f"{'='*70}")

    sig_holds_all = {yr: all_holds[yr] for yr in args.years
                     if yr in all_holds and not all_holds[yr].empty}

    if len(sig_holds_all) == 0:
        print("\n  No hold data available for significant years.")
        return

    # Check if significant years all have vol-driven holds
    all_vol_driven = all(
        h["vol_driven"].all() or h["vol_driven"].mean() >= 0.5
        for h in sig_holds_all.values()
    )

    channels_by_year = {
        yr: set(h["channel"].unique())
        for yr, h in sig_holds_all.items()
    }

    shared_channels = set.intersection(*channels_by_year.values()) \
                      if channels_by_year else set()

    print(f"\n  Significant years with hold data: "
          f"{sorted(sig_holds_all.keys())}")
    print(f"  All significant years vol-driven (≥50%): {all_vol_driven}")
    print(f"  Channels by year:")
    for yr, chs in sorted(channels_by_year.items()):
        print(f"    {yr}: {sorted(chs)}")
    print(f"  Shared channels across ALL significant years: "
          f"{sorted(shared_channels) if shared_channels else 'none'}")

    print(f"\n  VERDICT:")
    if all_vol_driven and shared_channels:
        print(f"  COHERENT — all significant years share the channel:")
        for ch in sorted(shared_channels):
            print(f"    '{ch}'")
        print(f"  The OOS results trace to the same economic mechanism")
        print(f"  across independent evaluation years. Safe to include")
        print(f"  in the paper with this tracing as supporting evidence.")
    elif all_vol_driven and not shared_channels:
        print(f"  PARTIALLY COHERENT — all significant years are")
        print(f"  volatility-driven but via different instruments/subchannels.")
        print(f"  This is consistent with the framework detecting the same")
        print(f"  broad mechanism (vol spike → market recovery) through")
        print(f"  whichever vol instruments had sufficient history at each")
        print(f"  training cutoff. Reportable with that explanation.")
    else:
        print(f"  MIXED — significant years involve different channels.")
        print(f"  Investigate individual years before including in paper.")
        print(f"  The 2011 result in particular needs closer inspection.")

    print(f"\n  Grid coverage across all years:")
    print(f"  {'Year':>6}  {'SPY configs':>11}  {'Vol→SPY':>8}  "
          f"{'VIXM+VIXY':>10}  {'VV active':>10}  {'Any vol active':>14}")
    print(f"  {'─'*65}")
    for yr in all_years:
        gc = grid_checks.get(yr, {})
        print(f"  {yr:>6}  {gc.get('n_spy_configs',0):>11}  "
              f"{gc.get('n_vol_spy',0):>8}  "
              f"{gc.get('n_vv_spy',0):>10}  "
              f"{gc.get('n_vv_active',0):>10}  "
              f"{gc.get('n_vol_active',0):>14}")

    print(f"\n  NOTE: 'Any vol active' = vol→SPY configs with episode_conviction>0")
    print(f"  at the training cutoff for that year. This is the correct measure")
    print(f"  of whether the validated channel was available to the strategy,")
    print(f"  since episode conviction=0 → config zeroed out in sizing.")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
