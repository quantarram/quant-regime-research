# Addendum to "From Descriptive Atlas to Tradeable Signal: An Out-of-Sample
# Test of the Multi-Asset Conditional Exceedance Framework as a Portfolio
# Tilt Strategy" (CPE_Portfolio_Tilt_OOS_2025)

**Status: working document, continuation of the original paper's investigation.**
**Companion code: see file manifest in Section 7.**
**Last updated: this session. Read this whole document before resuming work.**

---

## 0. How to use this document

This is written so that a fresh conversation — with no memory of this
session — can pick up exactly where it left off. It assumes familiarity
with the three prior papers (Single-Asset Validation, the Atlas, and the
original Portfolio Tilt paper) but explains everything done *since* the
original Portfolio Tilt paper from scratch. If you are a Claude instance
or a person reading this cold: read Sections 1-6 in order before touching
any code, then jump to Section 8 (Open Scopes) to see what's actually
left to do.

The single most important thing to understand before reading further:
**the original Portfolio Tilt paper's reported numbers (7.65% return,
Sharpe 0.560, etc.) could not be reproduced.** The code that generated
them was run once in an earlier session and not saved. This addendum
does not try to recover those numbers — it treats the original paper as
a qualitative guide (what mechanisms to test, what the right
methodology looks like) and reports a freshly-built, independently
verified pipeline's own results instead.

---

## 1. What this session set out to do

The original Portfolio Tilt paper's own conclusion (Section 16)
identified the central open question precisely: every statistically-
motivated correction tested in that paper (raw CPE, shrinkage,
sample-size filtering, episode-independence filtering) failed to
separate a spurious-but-recurring relationship (silver predicting
dogecoin, 5 episodes, no economic mechanism) from a genuine one
(volatility predicting equities, the one channel that survived every
test). The paper's own stated conclusion: *"the missing ingredient is
most likely not a better statistic, but a pre-specified economic prior
over which predictor-target relationships are admissible in the first
place."*

This session built that economic prior, found and fixed two real bugs
in the existing pipeline along the way, built a genuine episode-
independence-aware position-sizing mechanism (going beyond the original
paper's binary episode-count filter), and ran the resulting pipeline
against real, current data multiple times. None of the results below
constitute a demonstrated trading edge. What they do constitute is a
methodologically sound, internally-consistent pipeline that is ready
for the next round of testing.

---

## 2. The Economic Prior (`economic_prior.py`)

### 2.1 What it is

A pre-specified, data-blind whitelist of which (predictor, target)
pairs are even candidates for the CPE screen, built *before* looking at
which pairs happened to perform well in any backtest. Implemented as:

- A **sub-class taxonomy**, finer-grained than the 6 broad asset
  classes used in the Atlas (e.g. splits "commodities" into
  precious_metal / energy / agriculture / base_metal / broad_commodity;
  splits "crypto" into crypto_major / crypto_alt / crypto_btc_etf /
  crypto_eth_etf).
- An **admissible-channel whitelist**: ~91 (predictor_subclass,
  target_subclass) pairs, each with a one-line stated economic
  mechanism (e.g. `("vol_index_equity", "broad_equity_us")`: "vol spike
  -> equity reversal/drawdown (leverage & risk-off)").
- A **duplicate-instrument exclusion**: GLD/IAU/GC=F (same underlying
  gold exposure) cannot predict each other; same for ~15 other
  near-identical instrument groups.
- A **confidence-tier overlay** (see Section 3 below), refined via
  literature review, with a `min_confidence` parameter
  (`weak`/`caveat`/`standard`/`high`) so the screen can be run at
  different levels of strictness.

### 2.2 Verified impact

Confirmed directly against the real 161-instrument universe (not
estimated):
- Pairwise screen: 169,357 rows (original, unrestricted) -> 11,106 rows
  at `MIN_CONFIDENCE=weak` (6.6% of original) -> ~10,673 rows at
  `MIN_CONFIDENCE=standard`.
- Joint screen: 4,545 rows (original) -> ~1,084-1,094 rows (prior-gated,
  pre-episode-ranking).
- Specifically confirmed absent after the prior: SOL-USD<-[UDN,
  EURUSD=X] (no crypto mechanism), CORN<-[GLD, IAU] (gold predicting
  corn via two near-identical gold ETFs), and the silver->dogecoin
  pattern flagged in the original paper's Section 14.2.

### 2.3 Known limitation

The prior's content was built by Claude in a single session, from
general market-domain reasoning, not from the user's own independently
formed views. The user has not yet reviewed or edited the channel list.
**This is flagged as an open item, not resolved.** See Section 8.4.

---

## 3. Literature Review of Borderline Channels (`literature_review_tier2.md`)

19 channels originally tagged "debatable" (Tier 2) were checked against
actual published research (web search, not recall) for two separate
properties: (a) does literature support the relationship existing at
all, and (b) does that literature support it at the SPECIFIC horizon
and instrument resolution this screen tests (1-300 trading days,
single-ETF-to-single-ETF), as opposed to the macro/YoY horizons most
financial literature actually studies.

**Key findings:**
- `dollar_index -> em_fx` (the "dollar smile"): real historical pattern,
  but multiple sources confirm it broke down specifically in April 2025
  -- the same month the validated vol->equity channel fired correctly.
  Downgraded from `high` to `caveat`.
- `base_metal -> equity_country` ("Dr. Copper"): genuine literature
  support, but exclusively at 6-18 month/YoY horizons, not this
  screen's 1-300 trading-day resolution. Downgraded to `weak`.
- `credit_ig -> broad_equity_us`: same horizon-mismatch problem.
  Downgraded to `weak`.
- Two channels (`dollar_index -> yield_index`, `g10_fx -> yield_index`)
  were removed outright -- they were added as "reverse legs" of an
  already-admissible forward channel with no independent literature
  support for the reverse direction.
- Three channels were arguably under-rated originally and promoted:
  `broad_commodity -> em_fx`, `broad_equity_us -> broad_equity_intl`,
  `broad_equity_us -> broad_equity_em`.
- The validated `vol_index_equity -> broad_equity_us` channel was
  promoted to `high` confidence -- not just literature-supported but
  the one channel with actual traced out-of-sample accuracy (see
  Section 5 below and the original paper's own Section 10.1).

Quantified impact: even rejecting every "debatable" channel outright
touches only ~6.8% of the pairwise screen and ~4.8% of joint
configurations -- the result is not fragile to this review, but the
specific channels it touches are exactly the ones most likely to look
statistically clean while being economically thin.

---

## 4. Two Real Bugs Found and Fixed

These were found by *demanding reproducibility* -- specifically, by
trying to reproduce the original Portfolio Tilt paper's central traced
example (Section 10.1: VIXM+VIXY predicting SPY, 11/11 correct forward
returns in April 2025) and discovering it could not be reproduced from
the uploaded code as originally written.

### 4.1 Bug #1: `joint_cpe_engine.py`'s predictor exclusion

**The bug:** the original `joint_cpe_engine.py` used a single combined
`EXCLUDE_TICKERS` set, applied to BOTH the predicted (Y) side and the
predictor (X) side. This incorrectly stripped VIXM, VIXY, VXX, UVXY,
SVXY from the predictor pool entirely -- even though
`cpe_engine_parallel.py`'s own pairwise screen correctly retains them
as legitimate predictors (VIXM has CPE=0.977 for SPY at the exact
horizon the original paper traces). The joint engine could therefore
never construct the configuration the original paper's headline result
depends on.

**The fix:** split into `EXCLUDE_FROM_Y` (unchanged -- these are still
bad TARGETS, since their price action is mechanically derivative/decaying)
and `EXCLUDE_FROM_X` (matches `cpe_engine_parallel.py`'s own convention
-- managed currencies only).

**Verified:** after the fix, `Y=SPY, predictors=[VIXM,VIXY],
tau_future=63, joint_CPE=0.98, n_joint=147` appears in the joint screen,
closely matching the original paper's description.

### 4.2 Bug #2: `joint_cpe_engine.py`'s full-history thresholds

**The bug:** `joint_cpe_engine.py` computed quantile thresholds (and
therefore firing dates and joint_CPE itself) from the FULL price
history, with no `<=2024-12-31` cutoff -- unlike `cpe_engine_parallel.py`
and `backtest_engine.py`, both of which correctly restrict threshold
estimation to training-period data only. This was a genuine train/test
leak: every joint_CPE value this engine ever produced had already seen
2025+ data.

**Why it was found:** surfaced while reconciling episode counts between
`joint_cpe_engine.py`'s own diagnostic printout and `backtest_engine.py`'s
independently-computed conviction for the identical configuration -- the
two disagreed on which historical dates even counted as "firing,"
traced back to different threshold values from different date windows.

**The fix:** thresholds and the `common_idx` used for joint_CPE
computation are now both restricted to `<=2024-12-31`.

**Quantified impact:** ~2.89% of all price history rows are post-cutoff,
so the leak was real but modest in headline-number terms -- still
correctly worth fixing, and now fixed.

---

## 5. Episode-Independence-Aware Sizing (the methodology beyond the
## original paper's Section 14)

### 5.1 The problem this addresses

The original paper's `w(Pi) = CPE * lift * ln(n_joint) * h(Pi)` formula
uses `n_joint`, a count of OVERLAPPING DAILY conditioning observations.
A relationship that fired on 150 overlapping days inside one continuous
3-month episode gets the same credit as one that fired on 150 days
spread across 50 genuinely independent historical moments -- even
though only the second case represents real, repeated confirmation.

The original paper's own Section 14 introduced an episode-count idea,
but only as a binary eligibility filter (>=2 episodes passes, <2
fails), not as a continuous part of the sizing formula -- and that
filter was shown to underperform a cruder sample-size-based filter
specifically because it could not distinguish a spurious relationship
that happens to recur (silver->dogecoin, 5 episodes) from a genuine one.

### 5.2 What was actually built this session

`ln(n_joint)` was REPLACED (not supplemented) in `compute_quality_weights`
(`backtest_engine.py`) with an episode-conviction term:

1. Cluster a configuration's firing dates into genuinely separated
   episodes: a new episode begins when the gap since the previous
   firing date exceeds 1.5x the conditioning window (same convention
   as the original paper's Section 14, reused rather than reinvented).
2. Evaluate the target's outcome ONCE PER EPISODE (at the episode's
   last firing date, using the FORWARD tau_f-day return starting from
   that date -- see Section 6 below for a bug that initially got this
   backwards), not once per overlapping day.
3. `conviction = 0` if fewer than 3 independent episodes exist
   (a HARD FLOOR, not a discount -- mirrors the lesson from the
   MIN_TRAIN_OBS fix: discounting a thin signal still lets it fire,
   excluding it does not).
4. Above that floor: `conviction = log(n_episodes) * max(0, 2*hit_rate - 1)`
   -- a 50%-or-worse hit rate across any number of episodes earns ZERO
   credit (no better than chance), scaling up toward `log(n_episodes)`
   as the hit rate approaches 100%.

This also required changing the GREEDY JOINT SEARCH ITSELF
(`joint_cpe_engine.py`), not just the downstream sizing: the greedy
search was originally seeded and extended purely by raw CPE ranking,
which could not distinguish a candidate whose CPE=1.0 rests on one
continuous episode from one whose CPE=1.0 rests on several independently-
confirming episodes. Candidates are now ranked by `(episode_count, CPE)`
-- episode count first, raw CPE only as a tiebreaker.

### 5.3 The most important single finding from this work

When the validated VIXM+VIXY->SPY channel (the one configuration that
survived every test in the entire three-paper series) was checked at
EPISODE resolution rather than overlapping-day resolution, using a
FIRST DRAFT of the outcome-evaluation logic, it came back as **2 wins
out of 4 episodes -- a coin flip**. This was alarming and nearly led to
reporting "even your best signal is fake."

It turned out to be a bug, not a finding (see Section 6). The corrected
version shows this exact channel at **6 episodes, 100% hit rate,
conviction=1.792** -- the strongest episode-validated signal in the
entire screen. The discrepancy and its resolution are documented in
full because the INVESTIGATION PROCESS here is itself instructive: a
naive episode-level reimplementation can introduce exactly the kind of
silent measurement error that this whole exercise exists to catch, and
the only way to know which result to trust was to build two independent
implementations and force them to reconcile.

---

## 6. Bug #3 (the most subtle one): trailing vs. forward return

**The bug:** the first version of `_episode_conviction_for_row`
(`backtest_engine.py`) evaluated an episode's outcome by looking up
`increments[tau_f][y].get(anchor_date)` -- which is the target's
TRAILING tau_f-day return ENDING at the anchor date (i.e., "what
already happened by this date"), not its FORWARD tau_f-day return
STARTING at that date (i.e., "what happens next", the actual question
the CPE framework exists to answer). This is backwards in time.

**Why it evaded immediate detection:** the two quantities are often
numerically close for short horizons with low autocorrelation, so
simple sanity checks didn't catch it. It was found only by building a
SECOND, independent implementation (inside `joint_cpe_engine.py`'s own
`compute_episode_stats`, which correctly used the existing
`event_bull`/`event_bear` arrays already validated as part of the
engine's core joint_CPE computation) and discovering the two disagreed
(33% vs 100% hit rate) on the identical configuration.

**The fix:** `_episode_conviction_for_row` now uses
`increments[tau_f][y].shift(-tau_f)` (the forward-shifted series,
matching `future_inc` elsewhere in the pipeline) to look up each
episode's outcome.

**Verified:** post-fix, `_episode_conviction_for_row` and
`joint_cpe_engine.py`'s independently-computed `compute_episode_stats`
agree to 4 decimal places (1.7918 vs 1.792) on the VIXM+VIXY->SPY
configuration.

**Lesson for future work in this codebase:** any time two
independently-built implementations of the same quantity disagree, that
disagreement is more likely to indicate a real bug than to indicate
which one is "more conservative" or "more correct by default." Treat
agreement between independent implementations as a required check
before trusting either one's output.

---

## 7. Current File Manifest

All files are drop-in replacements for their same-named originals
unless noted. Run order: `cpe_engine_parallel.py` ->
`joint_cpe_engine.py` -> `cpe_signal_score.py` (optional, current-day
snapshot) -> `run_backtest.py`.

| File | What it does | Status |
|---|---|---|
| `economic_prior.py` | Sub-class taxonomy, admissible-channel whitelist, confidence tiers, duplicate-instrument exclusion | Verified working |
| `economic_prior_BYPASS.py` | Drop-in null-prior replacement for A/B comparison (admits everything except self-pairs) | Verified working |
| `cpe_engine_parallel.py` | Pairwise CPE screen. Prior-gated + hard `MIN_TRAIN_OBS` history-length floor | Verified working |
| `joint_cpe_engine.py` | Greedy joint CPE screen. Prior-gated, bugfixed (Section 4.1, 4.2), episode-aware ranking (Section 5.2) | Verified working |
| `backtest_engine.py` | Core simulation engine: neutral weights, static-tilt and hold-to-horizon mechanisms, episode-conviction sizing (Section 5.2, 6) | Verified working |
| `run_backtest.py` | Driver script. `--joint`, `--label`, `--skip-randomisation`, `--exclude-sleeve` flags | Verified working |
| `check_history_weights.py` | Standalone diagnostic: how much does the history-length floor actually constrain the screen | Verified working |
| `cpe_signal_score.py` | Current-day signal snapshot (unmodified from original upload) | Verified working, unmodified |
| `literature_review_tier2.md` | Citation-backed review of 19 borderline economic-prior channels | Reference document |
| `SPEC.md` | Prose-to-code spec extraction from the original Portfolio Tilt paper, documenting every ambiguity and the choice made | Reference document |

**Environment variables the pipeline reads:**
- `N_WORKERS`, `MIN_N`, `CPE_THRESH`, `MIN_LIFT`: standard CPE screen parameters
- `USE_PRIOR_BYPASS` (0/1): bypass the economic prior entirely (for A/B testing)
- `MIN_CONFIDENCE` (`weak`/`caveat`/`standard`/`high`): literature-review confidence floor
- `MIN_TRAIN_OBS` (default 500): hard minimum training-period observation count for a ticker to be eligible as a predictor

---

## 8. Results So Far (all on 2025 as the evaluation year -- see Section
## 8.5 on why this is now a limitation, not just a convention)

All runs below use the real 161-instrument universe,
`MIN_CONFIDENCE=standard`, `MIN_TRAIN_OBS=500`, 5 tradeable sleeves
(SPY, GC=F, TLT, BTC-USD, UUP).

| Run | Static tilt return/Sharpe | Hold-to-horizon return/Sharpe | Static tilt `pct_exceeding` |
|---|---|---|---|
| Unrestricted (no prior), pre-episode-fix | 19.54% / 1.222 | 37.39% / 2.681 | 34.3% |
| Prior-gated (`standard`), pre-episode-fix | 21.69% / 1.390 | 27.30% / 2.012 | 26.0% |
| + hard `MIN_TRAIN_OBS=500` filter | 17.09% / 1.035 | 18.88% / 1.157 | 52.0% (coin flip) |
| + episode-aware ranking, BEFORE bugs in Sec 4.2/6 fixed | 16.88% / 1.033 (degenerate: zero trading at all) | same | 100% (degenerate) |
| + all fixes (Section 4, 5, 6) applied, current state | **17.29% / 1.047** | **18.49% / 1.132** | **34.2%** |

No-tilt benchmark throughout: 16.88% / 1.033. SPY buy-and-hold: 18.01% / 0.955.

**Honest reading:** `pct_exceeding=34.2%` is NOT a statistically
significant result by the standard used throughout this entire paper
series (the bar has consistently been ~5-10%). The pipeline is now
methodologically sound and internally consistent, which it was not
before this session, but that is a precondition for a meaningful test,
not a positive result in itself.

**What did change qualitatively:** the strategy now trades on a
real, episode-validated signal (SPY<-[VIXM,VIXY], 6 episodes, 100%
historical hit rate) rather than on artifacts (short-history crypto
ETFs) or on nothing at all (the degenerate zero-trading states that
occurred mid-session while bugs were still present).

**A separate, important observation:** the richest episode-validated
signals in the FULL 136-target screen (`HYG`/`JNK` at `tau_past=5`
predicting `LQD`/`XLY`/`XLI`/`XLP`, with 45-88 episodes and 80-87% hit
rates) do NOT target any of the 5 tradeable sleeve proxies. The current
5-sleeve book structure cannot act on the strongest signals the screen
actually finds. This is the basis for Open Scope 2 below.

---

## 9. Open Scopes for Continuation (in the order discussed, not
## necessarily priority order -- decide priority in the next session)

### 9.1 Scope A: Randomisation-test Hold-to-horizon

The static-tilt mechanism's randomisation test has been run multiple
times (most recently: 34.2% exceeding). The hold-to-horizon mechanism's
result (18.49% / Sharpe 1.132, the better-looking number) has NOT yet
been put through the same randomisation test in this corrected
pipeline. This is the most direct, lowest-effort next step --
`run_backtest.py` already computes the static-tilt randomisation test;
the equivalent test for hold-to-horizon needs to be added (it currently
is not implemented at all -- check `run_backtest.py`'s `main()` for
where the static-tilt randomisation call happens and build an analogous
one for the hold-to-horizon tilt/hold pattern).

### 9.2 Scope B: Expand the tradeable universe

Section 8's closing observation: the strongest episode-validated
signals (HYG/JNK -> LQD/XLY/XLI/XLP) are inaccessible to the current
5-sleeve book. Two sub-options to decide between:
- Add specific new sleeves (e.g. LQD as a "Credit" sleeve) -- smaller,
  more controlled change, preserves the sleeve-based neutral-weight
  structure.
- Generalise to the cross-sectional, no-fixed-sleeve structure the
  original Portfolio Tilt paper tested in its own Section 11 (and which
  underperformed badly there, Sharpe 0.294, beaten by 89% of random
  reassignments) -- but that test pre-dates EVERY fix made in this
  session (economic prior, bugfixes, episode-aware ranking). It may be
  worth re-running now rather than assuming the original negative
  result still holds, since the original cross-sectional test's poor
  performance was explicitly diagnosed (Section 11.3 of the original
  paper) as caused by exactly the saturation/small-sample problem this
  session's fixes target directly.

### 9.3 Scope C: Refine the episode-counting methodology itself

Raised by the user as a third option alongside A and B. Specific,
known-imperfect aspects of the current episode methodology worth
revisiting:
- The 3-episode hard floor (`EPISODE_MIN_OBS_FOR_CONVICTION`) and the
  1.5x-gap clustering multiplier were both carried over from the
  original paper's Section 14 convention without independently
  re-deriving or stress-testing them in this session. Worth checking
  sensitivity: does conviction ranking change meaningfully at floor=2
  or floor=4, or at gap-multiplier=1.0 or 2.0?
- The outcome-evaluation anchor point (currently: each episode's LAST
  firing date) is one defensible choice but not the only one -- using
  the FIRST firing date, or the episode's midpoint, would change which
  forward-return window gets evaluated and could change hit rates
  materially for long episodes (e.g. the 82-day 2020 COVID episode in
  the VIXM/VIXY example).
- The agreement formula `max(0, 2*hit_rate - 1)` is a simple linear map
  with no statistical basis beyond "intuitively, 50% should mean zero
  credit and 100% should mean full credit" -- it has not been compared
  against alternative formulations (e.g. a proper one-sided binomial
  test against p=0.5, which would also naturally reward MORE episodes
  at a given hit rate, addressing a gap the current log(n_episodes)
  multiplicative term only partially captures).
- No interaction has been tested between episode-conviction and the
  EXISTING `h(Pi)` history-length down-weight -- both now operate on
  the same w(Pi) formula multiplicatively; whether this double-counts
  the "how much do we trust this" question (since long history and many
  episodes are correlated but not identical) has not been checked.

### 9.4 A fourth, not-yet-scoped item worth flagging

Every result in Section 8 is STILL on the same single 2025 evaluation
year used throughout this entire three-paper series. Whatever Scope is
tackled next, the deeper, repeatedly-deferred next step is a genuinely
fresh out-of-sample year (2026 as it accumulates, or a historical year
like 2018/2019 never touched in this series) with NO further tuning
between freezing the pipeline and evaluating it. This has been
acknowledged as necessary at multiple points across all three papers
and this session, and has not yet been done.

---

## 10. A Note on Process, for Whoever Continues This

This session's most valuable methodological lesson was not any single
result -- it was discovering, through direct, hands-on verification
rather than assumption, that:

1. A change that looks complete after one test (episode-aware ranking
   producing real conviction scores) can still be silently broken by an
   upstream issue (the threshold leakage in `joint_cpe_engine.py`) that
   only surfaces when you check a SECOND, independent computation of
   the same quantity.
2. The "uncomfortable" result (the validated channel showing 50% hit
   rate at episode resolution) was not the final truth -- it was a flag
   that something needed checking, and checking it found a real,
   fixable, fundamental bug (trailing vs. forward returns). The
   tempting move would have been to report the uncomfortable result as
   a finding ("even your best signal is fake") without first ruling out
   that it was a measurement error. Don't skip that step in future work
   either.
3. Every fix in this session was verified against REAL data with REAL,
   printed, checkable numbers -- not asserted. Whoever continues this
   should hold the same standard: before trusting any new result,
   reproduce it via at least two independent code paths if at all
   possible, the way Bug #3 was actually found.
