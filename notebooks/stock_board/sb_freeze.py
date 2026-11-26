"""
Freeze the stock board's rules and validation numbers (run by hand when re-validating, normally once a year in January;
the daily builder only READS stock_board_rule.json).

1. Build the labelled bet table for the ETF universe (barrier outcomes + point-in-time form percentiles).
2. Walk-forward 2010 onward: each year's rule is chosen from outcomes complete before 1 January of that year.
3. Pool the out-of-sample bets and compare with the same instruments' always-long bets over the same window.
4. A horizon is VALIDATED (allowed to publish picks) only if, pooled out of sample, hit-rate lift > 0, excess P&L > 0 and
   the block-clustered t-statistic of excess P&L >= 2.0. This gate was fixed before the final numbers were looked at.
5. The live rule for a validated horizon is chosen with every resolved outcome to date, the same way each walk-forward year was.
"""
import json
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

import sb_core as C
import sb_research as R

T_GATE = 2.0


def _clean(d):
    return {k: (None if (isinstance(v, float) and not np.isfinite(v)) else (float(v) if isinstance(v, (np.floating, np.integer)) else v))
            for k, v in d.items()}


def main(panel_path=None):
    P = C.load_panel(panel_path)
    asof = P.index[-1]
    meta = pd.read_parquet(C.NB / "multiasset_metadata.parquet").set_index("ticker")["full_name"].to_dict()
    T = R.build_table(P)
    rules = C.build_rules()
    M = R.rule_matrix(T, rules)
    oos, chosen = R.walk_forward(T, M, rules)
    out = {
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_through": str(asof.date()),
        "params": dict(k_target=C.K_TARGET, k_stop=C.K_STOP, cost_round_trip=C.COST_RT, vol_window=C.VOL_WIN,
                       pct_window=C.PCT_WIN, first_oos_year=R.FIRST_OOS_YEAR, max_fire_rate=R.MAX_FIRE_RATE,
                       min_blocks=R.MIN_BLOCKS, t_gate=T_GATE, n_rules_tested=len(rules)),
        "universe": {t: {"name": meta.get(t, t), "group": C.TICKER_GROUP[t]} for t in C.ALL_TICKERS},
        "horizons": {},
    }
    for h in C.HORIZONS:
        W = T[(T["h"] == h) & T["win"].notna() & (T["date"] >= f"{R.FIRST_OOS_YEAR}-01-01")]
        Rr = oos[oos["h"] == h].copy() if len(oos) else oos
        st = R.rule_stats(Rr, W, h)
        base = R.rule_stats(W, W, h)
        validated = bool(st["n"] > 0 and st["lift"] > 0 and st["exc_pnl"] > 0 and np.isfinite(st["t"]) and st["t"] >= T_GATE)
        bm = R.baseline_maps(W)
        Rr["bw"] = Rr["ticker"].map(bm[1]); Rr["bp"] = Rr["ticker"].map(bm[0]); Rr["exc"] = Rr["pnl"] - Rr["bp"]
        Rr["year"] = Rr["date"].dt.year
        yearly = (Rr.groupby("year").agg(n=("pnl", "size"), hit=("win", "mean"), base_hit=("bw", "mean"),
                                         pnl=("pnl", "mean"), exc=("exc", "mean")).reset_index().to_dict("records"))
        groups = (Rr.groupby("group").agg(n=("pnl", "size"), hit=("win", "mean"), base_hit=("bw", "mean"),
                                          pnl=("pnl", "mean"), exc=("exc", "mean")).reset_index().to_dict("records"))
        ch = chosen[chosen["h"] == h]
        hz = {"oos": _clean(st), "always_long": _clean(base), "validated": validated,
              "yearly": [_clean(r) for r in yearly], "by_group": [_clean(r) for r in groups],
              "rules_by_year": {int(r.year): r.rule for r in ch.itertuples()},
              "live_rule": None}
        if validated:
            cutoff = asof + pd.Timedelta(days=1)
            j, lst, n_ok = R.select_rule(T, M, rules, h, cutoff)
            if j is not None:
                conds, q = rules[j]
                hz["live_rule"] = {"conditions": list(conds), "q": q, "label": C.rule_label(rules[j]),
                                   "selected_on": _clean(lst)}
        out["horizons"][str(h)] = hz
    out["asof_bars"] = {t: str(P[t].dropna().index[-1].date()) for t in C.ALL_TICKERS if t in P.columns}
    (C.HERE / "stock_board_rule.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    o = main(sys.argv[1] if len(sys.argv) > 1 else None)
    for h, hz in o["horizons"].items():
        print(h, "validated" if hz["validated"] else "NOT validated", hz["oos"], hz["live_rule"])
