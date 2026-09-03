#!/usr/bin/env python3
"""
============================================================
  IBKR PAPER TRADING LEDGER (simulated, no broker connection)
  Dr. Arun Ramanathan
============================================================
IBKR's own paper-trading API is unavailable on this account
(Lite tier ineligible for Pro without USD 20k net worth), so
this tracks a virtual portfolio locally instead, at the same
250,000 SGD notional size, using only BULLISH tilts (the
bearish side of the signal underperforms in backtest and is
held at neutral weight here rather than shorted/underweighted).

Run this AFTER build_portfolio_dashboard.py each day. It reads
today's suggested weights straight from portfolio_dashboard.html
(same source of truth as log_predictions.py), marks the ledger
to market, rebalances to today's bullish-only target weights,
and appends one row to ibkr_paper_ledger.csv.

A parallel "neutral" (no tilt, constant weights) track runs
alongside it as the benchmark, same as portfolio_predictions.csv
already does for resolved predictions.

A third "hold-to-horizon" (HTH) track runs alongside both of the
above. CPE's own claim is horizon-specific -- a signal firing at,
say, 63d is a statement about the next 63 days, not about tomorrow
-- so re-rebalancing to the daily signal (what the "tilt" track
above does) re-evaluates a horizon-specific claim every single day,
which doesn't actually respect what CPE is claiming to know. The
HTH track instead: when a sleeve fires bullish at one or more
horizons, opens a position sized at that day's suggested weight and
HOLDS those shares fixed (no rebalancing) until the shortest firing
horizon elapses, then reassesses. Sleeves with no active hold sit in
a shared, daily-rebalanced neutral pool, same fallback rule as the
tilt track (no shorting/underweighting). This can only start from
today forward -- the per-horizon signal detail it needs was never
persisted for past dates, only the aggregated daily tilt label the
other two tracks use, so there is no way to backfill its history.

Usage:
  python ibkr_paper_ledger.py
============================================================
"""

import os
import re
import json
import warnings
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_HTML = os.path.join(BASE_DIR, "portfolio_dashboard.html")
LEDGER_CSV     = os.path.join(BASE_DIR, "ibkr_paper_ledger.csv")

STARTING_CAPITAL_SGD = 250_000.0  # matches intended IBKR paper account size
# All instruments (SPY, GC=F, TLT, IBIT, UUP) trade in USD, but the account
# is SGD — NAV/returns are converted to SGD daily using USDSGD=X, so the
# ledger reflects real SGD currency risk, not just the USD asset returns.

TICKERS = {"equities": "SPY", "gold": "GC=F", "bonds": "TLT",
           "crypto": "IBIT", "fx": "UUP"}
CLS_MAP = {"equities": "Equities", "gold": "Gold", "bonds": "Bonds",
           "crypto": "Crypto", "fx": "FX"}

# Estimated ROUND-TRIP transaction cost per instrument, in bps of traded
# notional. These are NOT measured fills (no live broker connection exists
# here, see module docstring) -- they are conservative, published-spread-
# based estimates, used only to make the README's own flagged "transaction
# costs not modelled" limitation visible in this ledger as a cost drag,
# rather than silently assuming zero cost like the rest of the CPE backtest.
# SPY/TLT are amongst the most liquid ETFs traded (sub-bp typical spread);
# IBIT and UUP are thinner and carry wider estimated spreads.
COST_BPS = {"equities": 2.0, "gold": 3.0, "bonds": 4.0, "crypto": 6.0, "fx": 8.0}

TODAY = date.today()

LEDGER_COLS = [
    "date",
    "usdsgd_rate",
    "equities_tilt", "gold_tilt", "bonds_tilt", "crypto_tilt", "fx_tilt",
    "equities_weight", "gold_weight", "bonds_weight", "crypto_weight", "fx_weight",
    "equities_price", "gold_price", "bonds_price", "crypto_price", "fx_price",
    "equities_shares", "gold_shares", "bonds_shares", "crypto_shares", "fx_shares",
    "equities_shares_neutral", "gold_shares_neutral", "bonds_shares_neutral",
    "crypto_shares_neutral", "fx_shares_neutral",
    "equities_hth_weight", "gold_hth_weight", "bonds_hth_weight",
    "crypto_hth_weight", "fx_hth_weight",
    "equities_hth_shares", "gold_hth_shares", "bonds_hth_shares",
    "crypto_hth_shares", "fx_hth_shares",
    "equities_hth_hold_until", "gold_hth_hold_until", "bonds_hth_hold_until",
    "crypto_hth_hold_until", "fx_hth_hold_until",
    "nav_tilt_sgd", "nav_neutral_sgd", "nav_hth_sgd",
    "daily_return_tilt", "daily_return_neutral", "daily_return_hth",
    "cum_return_tilt", "cum_return_neutral", "cum_return_hth",
    "turnover_pct", "est_cost_drag_sgd", "cum_cost_drag_sgd",
    "hth_turnover_pct", "hth_est_cost_drag_sgd", "hth_cum_cost_drag_sgd",
    "notes",
]


def fetch_usdsgd_rate():
    hist = yf.download("USDSGD=X", period="5d", auto_adjust=True, progress=False)
    return float(hist["Close"].squeeze().dropna().iloc[-1])


def fetch_dashboard_state():
    """Read today's tilts, weights and prices from portfolio_dashboard.html."""
    if not os.path.exists(DASHBOARD_HTML):
        raise FileNotFoundError(
            "portfolio_dashboard.html not found. Run build_portfolio_dashboard.py first."
        )

    with open(DASHBOARD_HTML, "r", encoding="utf-8") as f:
        html = f.read()

    match = re.search(r'const D = (\{.*?\});\n', html, re.DOTALL)
    if not match:
        raise ValueError("Could not find 'const D = {...}' in portfolio_dashboard.html")
    D = json.loads(match.group(1))

    generated_date = D.get("gen", "")[:10]
    if generated_date != str(TODAY):
        raise RuntimeError(
            f"portfolio_dashboard.html was generated on {generated_date}, not today ({TODAY}).\n"
            f"Run build_portfolio_dashboard.py first, then re-run this script."
        )

    ts   = D["tilt_summaries"]
    snap = D["snapshot"]

    state = {}
    for cls, label in CLS_MAP.items():
        horizon_labels = {int(hz): lbl for hz, lbl in ts[label]["horizon_labels"].items()}
        state[cls] = {
            "tilt":           ts[label]["overall_label"],
            "neutral_w":      float(ts[label]["neutral_w"]) / 100.0,
            "suggested_w":    float(ts[label]["suggested_w"]) / 100.0,
            "price":          float(snap[label]["price"]),
            "horizon_labels": horizon_labels,
        }
    return state


def hth_bullish_horizons(state_cls):
    """Horizons (days) currently firing bullish (TILT UP/OVERWEIGHT) for one sleeve."""
    return sorted(
        hz for hz, lbl in state_cls["horizon_labels"].items()
        if lbl in ("TILT UP", "OVERWEIGHT")
    )


def bullish_only_weights(state):
    """
    Apply only the bullish side of the tilt. Bearish/neutral sleeves keep
    their exact neutral weight -- never diluted below it. Only truly
    bullish sleeves (TILT UP / OVERWEIGHT) are boosted above neutral,
    sharing whatever weight budget is left over once every other sleeve
    has taken its full neutral share, in proportion to each bullish
    sleeve's own suggested_w.

    Fixed 2026-09-03: the previous version set bullish sleeves to their
    already-globally-normalised suggested_w and every other sleeve to
    neutral_w, then renormalised the total back to 1 -- which silently
    diluted EVERY sleeve, including ones meant to be floored at neutral,
    any time a sleeve was tilted up (raw total > 1 whenever any bullish
    tilt existed, since nothing was ever allowed to go below neutral to
    fund it). Verified against the live daily record: FX was tilted up on
    most days since the ledger opened, which mechanically pulled crypto
    and gold below their true neutral share even while both sat at
    "NEUTRAL" tilt -- and those two were the best-performing sleeves of
    the period, so the bug was a real, measurable drag on the tilt
    account's NAV, not just a technicality.
    """
    floor_cls = [cls for cls, s in state.items() if s["tilt"] not in ("TILT UP", "OVERWEIGHT")]
    bull_cls  = [cls for cls, s in state.items() if s["tilt"] in ("TILT UP", "OVERWEIGHT")]

    floor_w = {cls: state[cls]["neutral_w"] for cls in floor_cls}
    budget_left = 1.0 - sum(floor_w.values())
    bull_raw_total = sum(state[cls]["suggested_w"] for cls in bull_cls)

    if not bull_cls or budget_left <= 0 or bull_raw_total <= 0:
        # No bullish sleeve to fund, or the floor group already claims the
        # whole budget (shouldn't happen since neutral_w's sum to 1) --
        # fall back to plain neutral weights, renormalised for safety.
        raw = {cls: state[cls]["neutral_w"] for cls in state}
        total = sum(raw.values())
        return {cls: w / total for cls, w in raw.items()}

    bull_w = {
        cls: budget_left * (state[cls]["suggested_w"] / bull_raw_total)
        for cls in bull_cls
    }
    return {**floor_w, **bull_w}


def neutral_weights(state):
    raw = {cls: s["neutral_w"] for cls, s in state.items()}
    total = sum(raw.values())
    return {cls: w / total for cls, w in raw.items()}


def main():
    print("=" * 60)
    print("  IBKR PAPER LEDGER (simulated)")
    print(f"  {TODAY}")
    print("=" * 60)

    state = fetch_dashboard_state()
    target_tilt    = bullish_only_weights(state)
    target_neutral = neutral_weights(state)
    usdsgd = fetch_usdsgd_rate()

    if not os.path.exists(LEDGER_CSV):
        pd.DataFrame(columns=LEDGER_COLS).to_csv(LEDGER_CSV, index=False)
        print(f"  Created: {LEDGER_CSV}")

    df = pd.read_csv(LEDGER_CSV)

    if str(TODAY) in df["date"].astype(str).values:
        print(f"  Already logged for {TODAY} — skipping.")
        return

    prices = {cls: state[cls]["price"] for cls in TICKERS}

    if df.empty:
        capital_usd = STARTING_CAPITAL_SGD / usdsgd
        nav_tilt_usd = nav_neutral_usd = capital_usd
        nav_neutral_sgd = STARTING_CAPITAL_SGD
        shares_tilt_prev = {cls: 0.0 for cls in TICKERS}
        cum_cost_drag_sgd_prev = 0.0
        daily_return_neutral = 0.0
        cum_return_neutral = 0.0
        notes = "Ledger opened."
    else:
        prev = df.iloc[-1]
        shares_tilt_prev    = {cls: float(prev[f"{cls}_shares"]) for cls in TICKERS}
        shares_neutral_prev = {cls: float(prev[f"{cls}_shares_neutral"]) for cls in TICKERS}
        nav_neutral_sgd_prev = float(prev["nav_neutral_sgd"])
        # Backward-compatible: the ledger's opening row predates the cost
        # columns, so fall back to 0 cumulative drag if not present yet.
        cum_cost_drag_sgd_prev = (
            float(prev["cum_cost_drag_sgd"]) if "cum_cost_drag_sgd" in df.columns else 0.0
        )

        nav_tilt_usd    = sum(shares_tilt_prev[cls] * prices[cls] for cls in TICKERS)
        nav_neutral_usd = sum(shares_neutral_prev[cls] * prices[cls] for cls in TICKERS)
        nav_neutral_sgd = nav_neutral_usd * usdsgd

        daily_return_neutral = nav_neutral_sgd / nav_neutral_sgd_prev - 1
        cum_return_neutral = nav_neutral_sgd / STARTING_CAPITAL_SGD - 1
        notes = ""

    # Estimated transaction cost of today's rebalance (bullish-tilt track
    # only -- see COST_BPS docstring above). Sized off PRE-cost NAV to
    # avoid a circular dependency (cost depends on the trade, the trade
    # depends on NAV), then the actual position is scaled down by the
    # resulting net-of-cost capital so the cost is a real drag on the
    # tracked account, not just a reported side figure.
    #
    # The ledger's very first row is the initial funding trade, not a
    # rebalance -- there is no prior position to have paid a "round trip"
    # against, so it is deliberately cost-free (matches a real account:
    # the cost model only accrues from the second rebalance onward).
    shares_target_gross = {cls: target_tilt[cls] * nav_tilt_usd / prices[cls] for cls in TICKERS}
    if df.empty:
        turnover_usd = est_cost_usd = 0.0
        nav_tilt_usd_net = nav_tilt_usd
        shares_tilt = dict(shares_target_gross)
    else:
        turnover_usd = sum(
            abs(shares_target_gross[cls] - shares_tilt_prev[cls]) * prices[cls] for cls in TICKERS
        )
        est_cost_usd = sum(
            abs(shares_target_gross[cls] - shares_tilt_prev[cls]) * prices[cls] * COST_BPS[cls] / 10_000
            for cls in TICKERS
        )
        nav_tilt_usd_net = nav_tilt_usd - est_cost_usd
        scale = nav_tilt_usd_net / nav_tilt_usd if nav_tilt_usd > 0 else 1.0
        shares_tilt = {cls: shares_target_gross[cls] * scale for cls in TICKERS}
    shares_neutral = {cls: target_neutral[cls] * nav_neutral_usd / prices[cls] for cls in TICKERS}

    nav_tilt_sgd = nav_tilt_usd_net * usdsgd
    turnover_pct = (turnover_usd / nav_tilt_usd * 100) if nav_tilt_usd > 0 else 0.0
    est_cost_drag_sgd = est_cost_usd * usdsgd
    cum_cost_drag_sgd = cum_cost_drag_sgd_prev + est_cost_drag_sgd

    if df.empty:
        daily_return_tilt = 0.0
        cum_return_tilt = nav_tilt_sgd / STARTING_CAPITAL_SGD - 1  # == 0.0
    else:
        nav_tilt_sgd_prev = float(df.iloc[-1]["nav_tilt_sgd"])
        daily_return_tilt = nav_tilt_sgd / nav_tilt_sgd_prev - 1
        cum_return_tilt    = nav_tilt_sgd / STARTING_CAPITAL_SGD - 1

    # ── Hold-to-horizon (HTH) track ──────────────────────────────────
    # Respects what CPE actually claims: a signal firing at horizon h is a
    # statement about the next h days, not about tomorrow. Sleeves with an
    # active hold keep their shares fixed (no rebalancing) until the hold
    # expires; sleeves with no active hold sit in a shared, daily-
    # rebalanced neutral pool. This track starts today -- see module
    # docstring for why it can't be backfilled.
    hth_prev_exists = (
        not df.empty and "equities_hth_shares" in df.columns
        and pd.notna(df.iloc[-1]["equities_hth_shares"])
    )

    if not hth_prev_exists:
        nav_hth_usd = STARTING_CAPITAL_SGD / usdsgd
        hold_until_prev = {cls: None for cls in TICKERS}
        shares_hth_prev = {cls: 0.0 for cls in TICKERS}
        cum_cost_drag_hth_prev = 0.0
    else:
        prev = df.iloc[-1]

        def _parse_hold_until(v):
            return None if pd.isna(v) or v == "" else date.fromisoformat(str(v))

        hold_until_prev = {cls: _parse_hold_until(prev[f"{cls}_hth_hold_until"]) for cls in TICKERS}
        shares_hth_prev = {cls: float(prev[f"{cls}_hth_shares"]) for cls in TICKERS}
        cum_cost_drag_hth_prev = (
            float(prev["hth_cum_cost_drag_sgd"]) if pd.notna(prev.get("hth_cum_cost_drag_sgd")) else 0.0
        )
        nav_hth_usd = sum(shares_hth_prev[cls] * prices[cls] for cls in TICKERS)

    held_cls = [cls for cls in TICKERS if hold_until_prev[cls] is not None and hold_until_prev[cls] > TODAY]
    free_cls = [cls for cls in TICKERS if cls not in held_cls]
    held_value_usd = sum(shares_hth_prev[cls] * prices[cls] for cls in held_cls)
    held_w_today = held_value_usd / nav_hth_usd if nav_hth_usd > 0 else 0.0

    new_hold_until = dict(hold_until_prev)
    entry_w = {}
    opening_cls = []
    for cls in free_cls:
        bull_hz = hth_bullish_horizons(state[cls])
        if bull_hz:
            chosen_hz = min(bull_hz)
            new_hold_until[cls] = TODAY + timedelta(days=chosen_hz)
            entry_w[cls] = state[cls]["suggested_w"]
            opening_cls.append(cls)
        else:
            new_hold_until[cls] = None
    neutral_pool_cls = [cls for cls in free_cls if cls not in opening_cls]

    budget_free = max(1.0 - held_w_today, 0.0)
    opening_w_total = sum(entry_w.values())
    if opening_w_total > budget_free and opening_w_total > 0:
        s = budget_free / opening_w_total
        entry_w = {cls: w * s for cls, w in entry_w.items()}
        opening_w_total = sum(entry_w.values())

    neutral_budget = max(budget_free - opening_w_total, 0.0)
    neutral_w_total = sum(state[cls]["neutral_w"] for cls in neutral_pool_cls)
    if neutral_pool_cls:
        neutral_w_alloc = (
            {cls: neutral_budget * (state[cls]["neutral_w"] / neutral_w_total) for cls in neutral_pool_cls}
            if neutral_w_total > 0 else
            {cls: neutral_budget / len(neutral_pool_cls) for cls in neutral_pool_cls}
        )
    else:
        neutral_w_alloc = {}

    target_hth_w = {}
    for cls in TICKERS:
        if cls in held_cls:
            target_hth_w[cls] = shares_hth_prev[cls] * prices[cls] / nav_hth_usd if nav_hth_usd > 0 else 0.0
        elif cls in opening_cls:
            target_hth_w[cls] = entry_w[cls]
        else:
            target_hth_w[cls] = neutral_w_alloc.get(cls, 0.0)

    # Held sleeves' shares never move (that's the definition of "hold").
    # Only free-bucket sleeves (opening a new hold, or sitting in the daily-
    # rebalanced neutral pool) actually trade today and pay estimated cost.
    shares_hth_free_gross = {cls: target_hth_w[cls] * nav_hth_usd / prices[cls] for cls in free_cls}
    if not hth_prev_exists or not free_cls:
        turnover_hth_usd = est_cost_hth_usd = 0.0
        shares_hth_free = shares_hth_free_gross
    else:
        turnover_hth_usd = sum(
            abs(shares_hth_free_gross[cls] - shares_hth_prev[cls]) * prices[cls] for cls in free_cls
        )
        est_cost_hth_usd = sum(
            abs(shares_hth_free_gross[cls] - shares_hth_prev[cls]) * prices[cls] * COST_BPS[cls] / 10_000
            for cls in free_cls
        )
        free_budget_gross_usd = sum(target_hth_w[cls] for cls in free_cls) * nav_hth_usd
        free_budget_net_usd = free_budget_gross_usd - est_cost_hth_usd
        scale_free = free_budget_net_usd / free_budget_gross_usd if free_budget_gross_usd > 0 else 1.0
        shares_hth_free = {cls: shares_hth_free_gross[cls] * scale_free for cls in free_cls}

    shares_hth = dict(shares_hth_prev)
    for cls in held_cls:
        shares_hth[cls] = shares_hth_prev[cls]
    for cls in free_cls:
        shares_hth[cls] = shares_hth_free[cls]

    nav_hth_usd_net = nav_hth_usd - est_cost_hth_usd
    nav_hth_sgd = nav_hth_usd_net * usdsgd
    hth_turnover_pct = (turnover_hth_usd / nav_hth_usd * 100) if nav_hth_usd > 0 else 0.0
    hth_est_cost_drag_sgd = est_cost_hth_usd * usdsgd
    hth_cum_cost_drag_sgd = cum_cost_drag_hth_prev + hth_est_cost_drag_sgd

    if not hth_prev_exists:
        daily_return_hth = 0.0
        cum_return_hth = 0.0
    else:
        nav_hth_sgd_prev_logged = float(df.iloc[-1]["nav_hth_sgd"])
        daily_return_hth = nav_hth_sgd / nav_hth_sgd_prev_logged - 1
        cum_return_hth = nav_hth_sgd / STARTING_CAPITAL_SGD - 1

    row = {"date": str(TODAY), "usdsgd_rate": round(usdsgd, 5)}
    for cls in TICKERS:
        row[f"{cls}_tilt"]            = state[cls]["tilt"]
        row[f"{cls}_weight"]          = round(target_tilt[cls] * 100, 2)
        row[f"{cls}_price"]           = round(prices[cls], 4)
        row[f"{cls}_shares"]          = round(shares_tilt[cls], 6)
        row[f"{cls}_shares_neutral"]  = round(shares_neutral[cls], 6)
        row[f"{cls}_hth_weight"]      = round(target_hth_w[cls] * 100, 2)
        row[f"{cls}_hth_shares"]      = round(shares_hth[cls], 6)
        row[f"{cls}_hth_hold_until"]  = str(new_hold_until[cls]) if new_hold_until[cls] else ""
    row["nav_tilt_sgd"]          = round(nav_tilt_sgd, 2)
    row["nav_neutral_sgd"]       = round(nav_neutral_sgd, 2)
    row["nav_hth_sgd"]           = round(nav_hth_sgd, 2)
    row["daily_return_tilt"]     = round(daily_return_tilt * 100, 4)
    row["daily_return_neutral"]  = round(daily_return_neutral * 100, 4)
    row["daily_return_hth"]      = round(daily_return_hth * 100, 4)
    row["cum_return_tilt"]       = round(cum_return_tilt * 100, 4)
    row["cum_return_neutral"]    = round(cum_return_neutral * 100, 4)
    row["cum_return_hth"]        = round(cum_return_hth * 100, 4)
    row["turnover_pct"]          = round(turnover_pct, 4)
    row["est_cost_drag_sgd"]     = round(est_cost_drag_sgd, 4)
    row["cum_cost_drag_sgd"]     = round(cum_cost_drag_sgd, 4)
    row["hth_turnover_pct"]      = round(hth_turnover_pct, 4)
    row["hth_est_cost_drag_sgd"] = round(hth_est_cost_drag_sgd, 4)
    row["hth_cum_cost_drag_sgd"] = round(hth_cum_cost_drag_sgd, 4)
    row["notes"]                 = notes

    df = pd.concat([df, pd.DataFrame([row], columns=LEDGER_COLS)], ignore_index=True)
    df.to_csv(LEDGER_CSV, index=False)

    print(f"  USDSGD        : {usdsgd:.4f}")
    print(f"  NAV (tilt)    : SGD {nav_tilt_sgd:,.2f}  ({cum_return_tilt*100:+.2f}% cum, "
          f"{daily_return_tilt*100:+.2f}% today)")
    print(f"  NAV (neutral) : SGD {nav_neutral_sgd:,.2f}  ({cum_return_neutral*100:+.2f}% cum, "
          f"{daily_return_neutral*100:+.2f}% today)")
    print(f"  Weights (bullish-only): " +
          "  ".join(f"{cls}={target_tilt[cls]*100:.1f}%" for cls in TICKERS))
    print(f"  Turnover today: {turnover_pct:.2f}% of NAV  |  "
          f"est. cost: SGD {est_cost_drag_sgd:,.2f} today, "
          f"SGD {cum_cost_drag_sgd:,.2f} cumulative (estimated, not measured fills)")
    print(f"  NAV (hold-to-horizon): SGD {nav_hth_sgd:,.2f}  ({cum_return_hth*100:+.2f}% cum, "
          f"{daily_return_hth*100:+.2f}% today)")
    held_str = ", ".join(f"{cls}→{new_hold_until[cls]}" for cls in held_cls + opening_cls) or "none"
    print(f"  HTH active holds: {held_str}")
    print(f"  Row appended → {LEDGER_CSV}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
