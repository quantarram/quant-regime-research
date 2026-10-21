"""Recompute the self_ref_score column of the live orig-group feature cache from the CURRENT cpe_results.parquet.

self_ref_score is the mean claimed CPE of the ticker's own (X == Y) table rows that fire on a date, with thresholds from the trailing
LOOKBACK window. The cache was seeded from research features built on an earlier table and then extended day by day, so after the table is
rebuilt (2026-10-14: FX alignment, horizons in trading days) old and new rows would mix two definitions. This rewrites the column for every
cached (ticker, date) so the whole history uses the current table. Other feature columns do not depend on the table.

  python refresh_self_ref.py            # rewrite live_features_orig_panel.parquet in place (atomic)
  python refresh_self_ref.py --dry-run  # report only
"""
import argparse, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import feature_lib as fl
import live_features as lf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    cache = pd.read_parquet(lf.ORIG_CACHE)
    pair = pd.read_parquet(os.path.join(REPO, "cpe_results.parquet"))
    prices = pd.read_parquet(os.path.join(REPO, "multiasset_prices_live_history.parquet"))
    old = cache["self_ref_score"].copy(); new = old.copy(); tot = 0
    for t in sorted(set(cache["ticker"]) - set(fl.CONTEXT_TICKERS)):
        rows = fl.get_self_ref_rows(pair, t)
        m = cache["ticker"] == t
        if len(rows) == 0:
            new[m] = 0.5; continue
        s = prices[t].dropna(); pos_of = pd.Series(np.arange(len(s)), index=s.index)
        taus = sorted(set(int(x) for x in rows["tau_past"].unique()))
        ret = fl.build_ret_series_by_tau(s, t in fl.RATE_LIKE, taus)
        idx = cache.index[m]; dates = cache.loc[idx, "date"]
        pos = pos_of.reindex(dates).values
        vals = np.array([fl.rolling_self_ref_score(rows, ret, int(p)) if np.isfinite(p) else np.nan for p in pos])
        new.loc[idx] = np.where(np.isfinite(vals), vals, old.loc[idx].values); tot += int(m.sum())
        print(f"  {t:9s} rows {len(rows):3d}  cached dates {int(m.sum()):6d}  changed {int((np.abs(new.loc[idx]-old.loc[idx])>1e-9).sum()):6d}", flush=True)
    d = (new - old).abs()
    print(f"changed {int((d > 1e-9).sum())} of {len(cache)} cached rows; max change {d.max():.3f}, mean change on changed rows {d[d > 1e-9].mean():.3f}")
    if not a.dry_run:
        cache["self_ref_score"] = new.values
        lf._atomic_to_parquet(cache, lf.ORIG_CACHE); print("rewrote", lf.ORIG_CACHE)


if __name__ == "__main__":
    main()
