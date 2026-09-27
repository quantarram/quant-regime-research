"""
generate_daily_summary.py
==========================
Runs at the end of the daily GitHub Actions pipeline (after all dashboards,
prediction logs, the IBKR paper ledger, and the football checklist have been
rebuilt and committed). Reads today's actual output numbers, sends them to
the Claude API, and asks for the same kind of narrative daily summary that
used to be given interactively after each manual dashboard run -- headline
accuracy/win-rate stats, portfolio tilt performance, notable regime signals,
football qualifying picks -- so the owner can decide, even while away, when
a result is worth writing up as a LinkedIn/Substack post or a Zenodo
preprint.

Writes the result to daily_summaries/<date>.md and daily_summaries/latest.md
(both get committed by the workflow's existing commit step), and -- if
GH_TOKEN is set -- posts it as a comment on a standing "Daily Dashboard
Summary" tracking issue, which rides GitHub's own notification/email system
so it reaches the owner without any extra integration.

Never blocks the core pipeline: any failure here (missing key, API error,
malformed data) prints a warning and exits 0, since the dashboards and data
files it reads are already safely committed by the steps before it.
"""
import json
import os
import subprocess
import sys
from datetime import date

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TODAY = date.today().isoformat()
SUMMARY_DIR = os.path.join(BASE_DIR, "daily_summaries")

METALS_TICKERS = ["Silver", "Platinum"]  # matches log_predictions.py's METALS_TICKERS


def safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def resolved_stats(df, correct_col="prediction_correct"):
    if df.empty or "status" not in df.columns:
        return None
    n_pending = int((df["status"] == "PENDING").sum())
    res = df[df["status"] == "RESOLVED"]
    if len(res) == 0:
        return {"n_total": len(df), "n_pending": n_pending, "n_resolved": 0}
    acc = res[correct_col].astype(float).mean() * 100
    return {
        "n_total": len(df),
        "n_pending": n_pending,
        "n_resolved": len(res),
        "accuracy_pct": round(acc, 1),
    }


def today_new(df, col="date_predicted"):
    if df.empty or col not in df.columns:
        return []
    rows = df[df[col] == TODAY].copy()
    return json.loads(rows.to_json(orient="records"))


def build_payload():
    gold = safe_read_csv(os.path.join(BASE_DIR, "gold_predictions.csv"))
    port = safe_read_csv(os.path.join(BASE_DIR, "portfolio_predictions.csv"))
    metals = safe_read_csv(os.path.join(BASE_DIR, "metals_predictions.csv"))
    ledger = safe_read_csv(os.path.join(BASE_DIR, "ibkr_paper_ledger.csv"))
    qlog = safe_read_csv(os.path.join(BASE_DIR, "football_betting", "output", "qualifying_log.csv"))

    payload = {"date": TODAY}

    payload["gold_track_record"] = resolved_stats(gold)
    payload["metals_track_record_overall"] = resolved_stats(metals)

    if not metals.empty and "status" in metals.columns:
        res = metals[metals["status"] == "RESOLVED"]
        by_metal = {}
        for name in METALS_TICKERS:
            mres = res[res["metal"] == name] if "metal" in res.columns else res.iloc[0:0]
            if len(mres):
                by_metal[name] = {
                    "n_resolved": len(mres),
                    "accuracy_pct": round(mres["prediction_correct"].astype(float).mean() * 100, 1),
                }
        payload["metals_track_record_by_metal"] = by_metal

    if not port.empty and "status" in port.columns:
        res = port[port["status"] == "RESOLVED"]
        if len(res):
            win = res["tilt_beat_neutral"].astype(float).mean() * 100
            avg_edge = (res["tilt_pnl"].astype(float) - res["neutral_pnl"].astype(float)).mean()
            payload["portfolio_tilt_track_record"] = {
                "n_resolved": len(res),
                "tilt_win_rate_pct": round(win, 1),
                "avg_edge_vs_neutral_pct": round(float(avg_edge), 3),
            }

    combined_parts = []
    if not gold.empty and "status" in gold.columns:
        combined_parts.append(gold[gold["status"] == "RESOLVED"]["prediction_correct"])
    if not metals.empty and "status" in metals.columns:
        combined_parts.append(metals[metals["status"] == "RESOLVED"]["prediction_correct"])
    if combined_parts:
        combined = pd.concat(combined_parts).astype(float)
        if len(combined):
            payload["combined_gold_silver_platinum_track_record"] = {
                "n_resolved": len(combined),
                "accuracy_pct": round(combined.mean() * 100, 1),
            }

    payload["todays_new_gold_predictions"] = today_new(gold)
    payload["todays_new_metals_predictions"] = today_new(metals)
    payload["todays_new_portfolio_prediction"] = today_new(port)

    if not ledger.empty and "date" in ledger.columns:
        row = ledger[ledger["date"].astype(str) == TODAY]
        if len(row):
            r = row.iloc[-1].to_dict()
            payload["ibkr_paper_ledger_today"] = {
                "nav_tilt_sgd": r.get("nav_tilt_sgd"),
                "nav_neutral_sgd": r.get("nav_neutral_sgd"),
                "cum_return_tilt_pct": r.get("cum_return_tilt"),
                "cum_return_neutral_pct": r.get("cum_return_neutral"),
                "nav_hth_sgd": r.get("nav_hth_sgd"),
                "cum_return_hth_pct": r.get("cum_return_hth"),
                "turnover_pct": r.get("turnover_pct"),
            }

    if not qlog.empty and "run_date" in qlog.columns:
        today_picks = qlog[(qlog["run_date"].astype(str) == TODAY) & (qlog["qualifies"] == True)]  # noqa: E712
        cols = [c for c in ["fixture", "league", "pick", "odds", "start_time"] if c in today_picks.columns]
        payload["football_qualifying_picks_today"] = json.loads(today_picks[cols].to_json(orient="records")) if len(today_picks) else []
        if "status" in qlog.columns:
            resolved_picks = qlog[qlog["status"] == "RESOLVED"]
            if len(resolved_picks) and "won" in resolved_picks.columns:
                win_rate = resolved_picks["won"].astype(float).mean() * 100
                payload["football_track_record"] = {
                    "n_resolved": len(resolved_picks),
                    "win_rate_pct": round(win_rate, 1),
                }

    return payload


SYSTEM_PROMPT = """You write a short daily research-status note for a quant researcher (Arun) \
who runs the CPE (Conditional Probability Enhancement) regime-research framework. You are \
given one day's actual, already-computed output numbers from his automated dashboard pipeline \
(gold/portfolio/metals prediction tracking, an IBKR paper-trading ledger, and a football \
betting checklist that is a side application of the same framework). Your job is to write the \
same kind of grounded, numbers-first daily summary he used to get interactively after each \
manual run -- he uses it to decide whether today's result is worth writing up as a LinkedIn \
post, a Substack post, or a Zenodo preprint, so be precise and don't inflate anything.

Hard rules, no exceptions:
- Every number in your summary must come directly from the JSON payload. Never invent, round \
  suspiciously, or extrapolate a number that isn't there.
- Do not call a pattern a "new finding" unless the data plainly shows something outside normal \
  day-to-day noise -- routine confirmation of an already-stable track record is not new.
- Never frame climatology, a "no signal" result, or a null result as a failure -- a clean null \
  is a real, useful finding, not something to apologize for.
- Never mention code bugs, debugging, or fixes -- only report finished, correct results.
- "Beat buy-and-hold" or "the neutral portfolio" is not the same as alpha -- if you cite an edge, \
  say plainly what it's measured against (e.g. "tilt beat the neutral-weight portfolio by X pp"), \
  don't imply risk-adjusted outperformance you can't see from this payload.
- No randomization/permutation/bootstrap significance-testing framing -- this pipeline only ever \
  reports real out-of-sample track record, so just report the actual resolved accuracy/win-rate.
- If a field is missing or empty (e.g. no football picks qualified today, nothing resolved yet), \
  say so plainly in one line rather than skipping it silently.
- Keep it tight: a few short paragraphs or a compact bulleted structure, not a full report. Lead \
  with whatever is most decision-relevant today (a track record shift, a new qualifying pick, a \
  notable portfolio tilt), not a rote recitation of every field in order.
- End with one explicit line: whether today's numbers look like something worth a post right now, \
  or whether it's a routine day -- and say why in one sentence, without hedging ("it's routine \
  because X" not "you may want to consider whether X").
"""


def call_claude(payload):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[summary] ANTHROPIC_API_KEY not set -- skipping narrative generation.")
        return None

    body = {
        "model": "claude-sonnet-5",
        "max_tokens": 1200,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": f"Today's date: {TODAY}\n\nToday's pipeline output:\n{json.dumps(payload, indent=2, default=str)}",
            }
        ],
    }
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        parts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
        return "\n".join(parts).strip() or None
    except Exception as e:  # noqa: BLE001 -- never let a summary failure break the pipeline
        print(f"[summary] Claude API call failed: {e}")
        return None


def post_to_github_issue(body_md):
    """Posts the summary as a comment on a standing tracking issue, so it rides GitHub's
    own notification/email system. Best-effort: any failure here is only printed, never
    raised, since the summary is already safely committed as a file regardless."""
    if not os.environ.get("GH_TOKEN") and not os.environ.get("GITHUB_TOKEN"):
        print("[summary] No GH token in env -- skipping issue post.")
        return
    try:
        title = "Daily Dashboard Summary"
        find = subprocess.run(
            ["gh", "issue", "list", "--search", f'"{title}" in:title', "--state", "all",
             "--json", "number,title"],
            capture_output=True, text=True, cwd=BASE_DIR, timeout=30,
        )
        issue_number = None
        if find.returncode == 0 and find.stdout.strip():
            issues = json.loads(find.stdout)
            for issue in issues:
                if issue.get("title") == title:
                    issue_number = issue["number"]
                    break
        if issue_number is None:
            create = subprocess.run(
                ["gh", "issue", "create", "--title", title,
                 "--body", "Automated daily research summaries land here as comments, "
                            "one per day, so they show up in normal GitHub notifications."],
                capture_output=True, text=True, cwd=BASE_DIR, timeout=30,
            )
            if create.returncode != 0:
                print(f"[summary] Could not create tracking issue: {create.stderr}")
                return
            issue_number = create.stdout.strip().rsplit("/", 1)[-1]

        comment = subprocess.run(
            ["gh", "issue", "comment", str(issue_number), "--body", f"## {TODAY}\n\n{body_md}"],
            capture_output=True, text=True, cwd=BASE_DIR, timeout=30,
        )
        if comment.returncode != 0:
            print(f"[summary] Could not post comment: {comment.stderr}")
        else:
            print(f"[summary] Posted to issue #{issue_number}")
    except Exception as e:  # noqa: BLE001
        print(f"[summary] GitHub issue post failed: {e}")


def main():
    payload = build_payload()
    narrative = call_claude(payload)
    if narrative is None:
        print("[summary] No narrative generated today; leaving prior summaries untouched.")
        sys.exit(0)

    os.makedirs(SUMMARY_DIR, exist_ok=True)
    dated_path = os.path.join(SUMMARY_DIR, f"{TODAY}.md")
    latest_path = os.path.join(SUMMARY_DIR, "latest.md")
    header = f"# Daily dashboard summary -- {TODAY}\n\n"
    with open(dated_path, "w") as f:
        f.write(header + narrative + "\n")
    with open(latest_path, "w") as f:
        f.write(header + narrative + "\n")
    print(f"[summary] Wrote {dated_path}")

    post_to_github_issue(narrative)


if __name__ == "__main__":
    main()
