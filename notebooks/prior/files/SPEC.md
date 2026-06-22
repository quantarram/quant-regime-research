# Backtest Rebuild — Spec Extraction From the Portfolio Tilt Paper

This document extracts every concrete, checkable rule from the paper's
prose (Sections 2.3-2.4, 7, 10.3 primarily) and separates it into:
(A) explicit and unambiguous, (B) explicit but requiring interpretation,
(C) genuinely underspecified, requiring a documented choice.

The implementation in `backtest_engine.py` implements exactly this
document. If the engine's behaviour and this document ever disagree,
this document is wrong or the code is — they should be kept in sync.

---

## A. EXPLICIT AND UNAMBIGUOUS

1. **Train/test split** (Section 2.1): all thresholds and joint
   configurations frozen on data through 2024-12-31. Evaluation walks
   forward through every trading day of 2025 only (2026 data never
   enters this paper's evaluation, per Section 2.1's explicit statement
   "no 2026 data enters this paper at any point").

2. **Tradeable universe** (Section 2.4): five sleeves, single proxy each.
   - Equities: SPY
   - Gold: GC=F
   - Bonds: TLT
   - Crypto: BTC-USD
   - FX: UUP
   Volatility complex is explicitly NOT a tradeable sleeve (used as
   predictor only), per Section 2.4's explicit scoping note.

3. **Neutral (no-tilt) weights** (Section 2.4): derived from
   training-period (≤2024-12-31) Sharpe ratios, with a 5% floor per
   sleeve. The paper gives exact figures: Equities 30.87%, Gold 24.29%,
   Crypto 30.10%, FX 10.00% (fixed), Bonds 4.74% (floored from a
   negative raw Sharpe of -0.075).
   NOTE: the paper states FX is "fixed" at 10% rather than
   Sharpe-derived — this is stated as a design choice in the paper
   itself, not left to us to decide. We replicate it as stated.

4. **Quality weight formula** (Section 2.3, identical to Atlas Section
   9.1): `w(Π) = CPE(Π) × lift(Π) × ln(n_joint(Π)) × h(Π)`, where h(Π)
   is a history-length down-weight: linear ramp from 0.35 at 100
   training observations to 1.0 at 756 observations (~3 trading years),
   held at 1.0 above that. h(Π) for a joint set is the MINIMUM of the
   per-predictor weight across every predictor in the set (Section 2.3,
   explicit).

5. **Horizon weights for combining into a daily class score** (Section
   2.3): 21d: 20%, 63d: 30%, 126d: 30%, 252d: 20%. (300d horizon is
   present in the screen grid but explicitly excluded from this
   weighted combination — the paper's listed weights sum to 100% over
   exactly these four horizons, so 300d signals exist in the joint
   screen but do not contribute to the daily class score.)

6. **Tilt tiers** (Section 2.3): five symmetric tiers — ±15pp at
   |score| ≥ 0.30, ±8pp at |score| ≥ 0.05, 0pp otherwise. This is the
   ORIGINAL mechanism (Sections 2-9). Section 10 introduces a SECOND,
   different mechanism (hold-to-horizon, continuous conviction-scaled
   sizing) as an explicit alternative, not a replacement — both are
   separately specified and both should be implemented as distinct
   backtest runs, matching the paper's own Table 6 comparison.

7. **Position lag** (Section 2.4, explicit): "the weight applied to day
   t's realised return is the weight computed using information
   available through day t-1" — i.e. one-day lag between signal and
   position, to avoid lookahead.

8. **Weight clipping and renormalisation** (Section 2.4): each day's raw
   weight = neutral weight + tilt delta, clipped to [0, 50]% per sleeve,
   renormalised to sum to 100% across the five sleeves.

9. **Hold-to-horizon mechanism** (Section 10.3, explicit): when a joint
   configuration NEWLY fires (first day of a contiguous firing run, not
   every day it continues firing), the resulting tilt is held for the
   full τ_f trading days from that entry date, regardless of whether the
   predictor condition remains true. If a new signal fires for the same
   sleeve while a previous hold is active, the realised tilt is the
   LARGER-MAGNITUDE of the two (not summed), and the hold extends to the
   later expiry (explicit, Section 10.3).

10. **Hold-to-horizon conviction-scaled sizing** (Section 10.3,
    explicit): tilt magnitude is a continuous function of the firing
    configuration's quality weight w(Π), normalised against the 95th
    percentile of w(Π) computed ONLY across configurations whose target
    is one of the five tradeable sleeve proxies (explicit — "not the
    full 161-instrument universe"). No cap beyond the natural [0,100]%
    bound (explicit, stated as a deliberate choice to measure "raw
    exploitability ceiling").

11. **Randomisation test** (Section 3.4): shuffle each sleeve's daily
    tilt-delta sequence independently across the 250 evaluation days,
    PRESERVING each sleeve's marginal distribution of tilt values (how
    many OVERWEIGHT/TILT UP/NEUTRAL days), only randomising WHICH days
    they fall on. 3,000 reassignments. Compare actual Sharpe against
    this null distribution.

---

## B. EXPLICIT BUT REQUIRING INTERPRETATION

1. **"Fires" definition** (Section 2.3): "A configuration 'fires' on day
   t if every predictor in its set simultaneously clears its own
   threshold on day t." This requires computing, for every trading day
   in 2025, every predictor's trailing log-return over its own τ_p
   window and comparing against the FROZEN (≤2024-12-31) threshold for
   that (ticker, τ_p, q) combination. This is mechanically identical to
   what `cpe_signal_score.py` already does for a single "latest date" —
   the backtest just needs to do it for every day of 2025 instead of
   one day. Interpretation required: which exact threshold table to use
   — we use the same q_X/q_Y grid and quantile convention as the
   prior-gated joint screen already on disk, since that's the train
   data this rebuild has access to.

2. **Score normalisation denominator** (Section 2.3 / Atlas Section
   9.1): `S(class,t) = [Σw·fire(bullish) − Σw·fire(bearish)] /
   [Σw(bullish) + Σw(bearish)]`. The denominator sums are over ALL
   joint configurations for that class/horizon (fired or not), per the
   Atlas's original formula. We implement it exactly this way.

---

## C. GENUINELY UNDERSPECIFIED — DOCUMENTED CHOICES MADE HERE

1. **Exact set of joint configurations available.** The original 2025
   backtest used the OLD, unrestricted joint screen (4,545
   configurations, no economic prior). This rebuild has two natural
   options: (a) reproduce the original paper's numbers using the
   ORIGINAL unrestricted screen, to get an apples-to-apples check
   against the paper's stated 7.65%/0.560 figures, or (b) run only
   against the NEW prior-gated screen. *** DECISION: do both, as two
   separate, clearly labelled runs, in that order. Run 1 = exact spec
   replication against the unrestricted screen (the actual
   reproducibility check). Run 2 = identical engine, prior-gated screen
   substituted in (the actual question this whole exercise exists to
   answer). ***

2. **Tie-breaking when two NEW signals fire for the same sleeve on the
   same day** (static-tilt mechanism only — hold-to-horizon's tie
   handling is explicit, see A.9). The paper's static mechanism
   (Sections 2-9) computes one combined class score per day from ALL
   firing configurations weighted together — there is no literal
   "tie" to break in that mechanism, since it's a continuous weighted
   sum, not a discrete pick-one decision. No choice needed here; flagging
   only because it could be mistaken for an ambiguity.

3. **What happens to a hold-to-horizon position if its sleeve's neutral
   weight itself would push the clipped/renormalised total over 100%
   together with other sleeves' holds.** The paper does not explicitly
   address simultaneous hold-to-horizon positions across MULTIPLE
   sleeves (its own traced example, Section 10.1, only ever has one
   sleeve — Equities — active at a time in 2025). *** DECISION: apply
   the same clip-to-[0,50%]-then-renormalise-to-100% rule used
   throughout the rest of the paper (A.8) uniformly, including to
   hold-to-horizon weights, since no alternative rule is stated and this
   is the rule used everywhere else in the same paper. ***

4. **EWMA dynamic-sizing variant (Section 9)** is NOT implemented in
   this rebuild's first pass. It was a follow-up the paper itself
   describes as testing a different, narrower question (does dynamic
   base-weighting help, independent of the tilt signal) and is not
   needed to answer this exercise's core question (does the economic
   prior change the tilt strategy's performance). It can be added later
   if useful.

5. **Cross-sectional broad-universe extension (Sections 11-14)** is
   NOT implemented in this rebuild's first pass, for the same reason —
   it's a structurally different strategy (116 tickers, no sleeve
   structure) testing a separate question (does breadth help) that is
   downstream of, not a prerequisite for, the core 5-sleeve comparison
   this exercise needs.

---

## What "success" looks like for Run 1 (the reproducibility check)

Run 1 will NOT be expected to match the paper's 7.65% / Sharpe 0.560 to
the decimal — this is an independently-written implementation against
the same prose spec, not a recovered copy of the original code, and the
paper itself documents that even the ORIGINAL pipeline had real,
caught-after-the-fact implementation bugs (single-asset paper Section
12.1). The bar for "the spec rebuild is trustworthy" is:
  - Same SIGN and rough MAGNITUDE of return and Sharpe
  - Same qualitative finding: strategy ≈ no-tilt benchmark, both
    underperform SPY buy-and-hold
  - Same identified firing episode (April 2025, Equities sleeve, driven
    by volatility-complex predictors)
If Run 1 diverges sharply from these qualitative findings, that's a
genuine red flag worth stopping and investigating before trusting Run 2
at all.
