#!/usr/bin/env python3
"""Replay ibkr_paper_ledger.py day by day from per-date portfolio dashboards built with DASH_ASOF (see asof_replay.py).

The ledger code itself is untouched: for each date the module's TODAY, DASHBOARD_HTML and LEDGER_CSV are pointed at that day's page,
that day's logged USDSGD rate (IBKR_USDSGD) and the output file, and main() is called. The hold-to-horizon track starts on `hth_start`
(its published opening date): HTH columns of earlier rows are blanked before that day is processed.

  python replay_ibkr_ledger.py --pages DIR --rates ibkr_paper_ledger_as_logged.csv --start 2026-07-11 --end 2026-10-29 --out OUT.csv
  (optionally --seed SEED.csv to continue from existing rows, then --start is the first date to replay)
Pages: DIR/p_YYYY-MM-DD/portfolio_dashboard.html
"""
import argparse, contextlib, io, os, re, sys
from datetime import date
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pages", required=True)
ap.add_argument("--rates", required=True, help="ledger CSV whose usdsgd_rate column supplies each date's rate")
ap.add_argument("--start", required=True)
ap.add_argument("--end", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed")
ap.add_argument("--hth-start", default="2026-09-03")
ap.add_argument("--note", default="")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ibkr_paper_ledger as ib

rates = pd.read_csv(args.rates).set_index("date")["usdsgd_rate"].to_dict()
if args.seed:
    pd.read_csv(args.seed).to_csv(args.out, index=False)
elif os.path.exists(args.out):
    os.remove(args.out)
ib.LEDGER_CSV = args.out
HTH_COLS = [c for c in ib.LEDGER_COLS if "_hth_" in c or c in ("nav_hth_sgd", "daily_return_hth", "cum_return_hth", "hth_turnover_pct", "hth_est_cost_drag_sgd", "hth_cum_cost_drag_sgd", "hth_cash_usd")]

for d in pd.date_range(args.start, args.end):
    ds = d.strftime("%Y-%m-%d")
    src = os.path.join(args.pages, f"p_{ds}", "portfolio_dashboard.html")
    html = open(src, encoding="utf-8").read()
    html = re.sub(r'("gen":\s*")\d{4}-\d{2}-\d{2}', r"\g<1>" + ds, html, count=1)   # the page was built later; the ledger checks it is dated today
    tmp = os.path.join(os.path.dirname(args.out), f"_page_{ds}.html")
    open(tmp, "w", encoding="utf-8").write(html)
    ib.DASHBOARD_HTML = tmp
    ib.TODAY = date.fromisoformat(ds)
    os.environ["IBKR_USDSGD"] = str(rates[ds])
    if ds == args.hth_start and os.path.exists(args.out):
        df = pd.read_csv(args.out)
        df[HTH_COLS] = float("nan")
        df.to_csv(args.out, index=False)
    with contextlib.redirect_stdout(io.StringIO()):
        ib.main()
    os.remove(tmp)

df = pd.read_csv(args.out)
if args.note:
    df.loc[df["date"] >= args.start, "notes"] = df.loc[df["date"] >= args.start, "notes"].fillna("").astype(str).map(lambda n: (n + " | " if n else "") + args.note)
df.to_csv(args.out, index=False)
print(f"replayed {args.start}..{args.end}: {len(df)} rows -> {args.out}")
