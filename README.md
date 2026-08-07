# Conditional Exceedance Probabilities for Event-Driven Trading

This repository contains the full code, notebooks, and experiments for an independent quantitative research program built around the **Conditional Probability of Exceedance (CPE)** framework — a live, production-deployed nonparametric signal system covering 161 instruments across equities, fixed income, gold, cryptocurrencies, commodities, volatility, and FX — plus two further research threads that test complementary angles on the same underlying question of what is predictable in financial markets, and over what horizon.

**Author:** Arun Ramanathan, PhD (Independent Researcher | Singapore)

---

## Overview

The work here spans three methodologically distinct threads, all documented as numbered papers below:

1. **The CPE framework** (Papers 1–5) — a nonparametric tail-co-movement signal system, described in full below, that powers the live dashboards.
2. **Climate-finance and event-study extensions** (Papers 6–10) — apply CPE's tail-exceedance logic to physical/weather predictors (temperature extremes, vapour pressure deficit, growing-season geography), plus one event study of hurricane landfalls against reinsurer equity using a different, CAR-based methodology.
3. **Multifractal predictability limits** (Paper 11) — adapts atmospheric turbulence cascade theory to ask how far ahead financial markets are structurally predictable at all, cross-validated against CPE's own signal density.
4. **Regime-conditioned price forecasting** (Paper 12) — a fully independent, ML-based (LightGBM quantile regression) forecasting system for 22 instruments, reusing Paper 11's multifractal features and two causally-validated market-regime signals. Its central finding: real, holdout-honest statistical skill for roughly half the panel does not translate into demonstrated tradeable alpha for *any* instrument, under five independently designed trading-strategy tests — a result treated as the paper's main contribution rather than suppressed, and now deployed as an honest, forecast-only (no buy/sell signal) live dashboard.
5. **Predictability limits as AI/ML regime detectors** (Paper 13) — repurposes Paper 11's empirical predictability limit as a practical, instrument-specific rule for how much history any AI/ML model should train on, and when its training data should be considered stale. Demonstrated across 12 instruments via direct predicted-vs-actual curve comparison (deliberately no significance testing), and framed explicitly as a general concept-drift/training-window rule applicable to any non-stationary AI/ML pipeline, not just financial markets.
6. **Does sophistication beat the limit? Testing AI/ML architecture, depth, and training-window size** (Paper 14) — asks the natural next question after Paper 13: can real architectural sophistication buy its way past the measured predictability limit anyway? Five architecturally distinct models — climatology, an unregularized tree, a reinforcement-learning forecaster, a conditional GAN, and a conditional VAE — given an identical fair training budget show no separation in skill; a follow-up giving three of them a genuine, hand-built hidden layer of nonlinearity at the same budget changes nothing; and sweeping training-window size from inside the limit out to 8x beyond it shows error climbing, not falling, as more data is added — the opposite of what a small-sample explanation would predict. Modeled explicitly on how Lorenz's atmospheric predictability limit was itself confirmed decades after being proposed, not by argument but by real, increasingly sophisticated weather models running into the same wall.
7. **Loss functions and calibrated uncertainty against the same limit** (Paper 15) — tests two further candidates for buying past Paper 14's ceiling. Alternative loss functions (quantile/pinball loss, escalating-order Lq loss) recover no genuine skill or tail behavior. A purpose-built generative downscaler is more informative: its point forecast is, by mathematical construction, provably identical to plain climatology's, but its calibrated predictive *distribution* — matched to each instrument's own predictability limit, corrected for decoder-variance miscalibration, and rescaled for real markets' sub-linear variance growth — beats all five Paper 14 architectures on the Continuous Ranked Probability Score when those architectures are held to the certainty they implicitly claim. Once each architecture instead gets its own honest, uncalibrated uncertainty estimate, the downscaler still beats every sophisticated architecture, but loses outright to climatology's own unmodeled empirical distribution on 7 of 12 instruments — evidence that honesty about uncertainty, not sophistication, is what's being rewarded, and that an unprocessed sample of real recent history is often harder to beat than a carefully calibrated synthetic approximation of the same thing.

### The CPE framework

Rather than forecasting prices or returns, CPE asks a fundamentally different question:

> When asset X is in the extreme upper tail of its historical distribution over the past N days, what is the empirical conditional probability that asset Y exceeds its own threshold over the next M days?

This is a nonparametric, empirical conditional frequency — computed directly from historical co-occurrences, with no distribution assumption, no fitted parameters, and no extrapolation beyond the data. **The tail co-movement structure itself is the signal.**

The framework sweeps 51 million candidate configurations across 161 instruments and retains only those where:
- Conditional probability exceeds 0.80
- Lift over the unconditional rate exceeds 1.5×
- At least 100 training-period observations support the estimate

This produces 169,357 pairwise signals. A greedy joint-conditioning procedure then identifies multi-predictor configurations — e.g. "when IBIT and BITB are both simultaneously in their upper tails, gold exceeds its 252-day median return with probability 0.94 vs 0.37 unconditionally" — a 2.54× lift confirmed genuinely calibrated out-of-sample in Paper 4.

**Key empirical results from the core CPE series (Papers 1–5):**
- **Calibration (Paper 4):** 97.0% realised hit rate against 93.0% stated CPE across 103,983 resolved instances over 528 trading days. The framework is slightly conservative — it understated its own edge by ~4 percentage points.
- **Gold dashboard:** BULLISH throughout all of 2025 (gold: $2,629 → $5,318/oz, +102%). Pivoted BEARISH in early February 2026. Directional discrimination: +7.23% average gold return on BULLISH days vs +2.46% on BEARISH days.
- **Portfolio tilt (Paper 3):** Hold-to-horizon Sharpe 1.224 (pct_exceeding 1.8% vs 1,000 randomisation repetitions). Cross-sectional extension: Sharpe 1.613 across 61 episode-validated targets. **See the Limitations section below** — a 2026-08 four-year re-test, benchmarked against published momentum strategies, found this result showed up in one of the four real years tested and did not repeat in the other three.
- **Portfolio tilt (Paper 4):** Cumulative +30.6% vs +17.4% neutral equal-weight over 1.4 years. Sharpe 1.43 vs 1.03 (60/40), 0.94 (time-series momentum), 0.71 (risk parity). Same caveat applies.

---

## Method Summary

This section describes the core CPE methodology (Papers 1–5). Paper 9 reuses this exact method with geography-corrected coordinates for its temperature predictors, Paper 10 uses a separate event-study/cumulative-abnormal-return methodology, and Paper 11 uses multifractal cascade and structure-function decomposition — see their respective papers and `notebooks/rein/` / `notebooks/predictability_paper/` for method details.

For a fixed horizon τ and two assets X and Y:

1. Compute τ-day price increments for all 161 instruments
2. Define exceedance events using empirical quantile thresholds (50th–99th percentile, both directions)
3. Estimate conditional probabilities: P(ΔY_future > r_qY | ΔX_past > r_qX)
4. Filter estimates using three quality gates: CPE ≥ 0.80, lift ≥ 1.5×, n_conditioning ≥ 100
5. Apply greedy joint-conditioning to build multi-predictor configurations
6. Score each asset daily using w = CPE × lift × ln(n_episodes) × max(0, 2 × hit_rate − 1)
7. Translate conviction scores into portfolio tilts or directional dashboard signals

All probability tables are computed using pre-2025 data and remain frozen during evaluation. No parameters are tuned on test data.

**Economic prior (Paper 3 refinement):** A pre-specified admissibility gate of ~91 economically justified predictor→target channel pairs reduces the pairwise screen from 169,357 rows to 11,106 (6.6%), eliminating spurious relationships (e.g. silver predicting dogecoin) while retaining validated channels (e.g. vol-complex→equity, crypto→gold).

**Episode-independence filter:** Firing dates are clustered into genuinely separated episodes (gap > 1.5× longest conditioning window). Minimum 3 independent episodes required for nonzero conviction. This prevents overlapping observations from inflating nominal signal counts.

---

## Reproducibility

**Core CPE framework (Papers 1–5):**
- **Data source:** Daily adjusted close prices via yfinance
- **Universe:** 161 instruments across 6 asset classes
- **Training period:** Full history through 2024-12-31
- **Evaluation period:** 2025–2026 (strictly out-of-sample)
- **Parameter sweep:** 8 horizons (1–300 days) × 8 quantile thresholds × 2 tail directions × 161² instrument pairs ≈ 51 million configurations
- **Surviving signals:** 169,357 pairwise | 11,106 prior-gated
- No parameters are tuned on test data. All configurations frozen at training cutoff.

**Climate-finance extensions (Papers 6–9):**
- **Weather data source:** Open-Meteo archive API (free, no API key) for daily city/crop-zone temperatures, growing degree days, and vapour pressure deficit
- **Financial data:** Same yfinance universe and CPE gating thresholds (CPE ≥ 0.80, lift ≥ 1.5×, n ≥ 100) as the core framework, with weather variables added as conditioning predictors
- **Paper 7 correction:** rebuilds Paper 6's temperature predictors using ERA5-consistent gridded crop-zone coordinates (rather than city centroids) — see Limitations below for what this correction found. Paper 9 is a standalone research-note summary of the whole programme that revisits this same reversal as its lead example, rather than an independent correction of its own.

**Hurricane/reinsurer event study (Paper 10):**
- **Event data:** Our World in Data, adapted from NOAA HURDAT (1990–2022), supplemented with NOAA/NHC official season totals (2023–2025) — the 14 costliest US hurricane landfalls since 1995
- **Equity data:** yfinance, RenaissanceRe (RNR) and Munich Re (MUV2.DE)
- **Method:** market-model event study — OLS alpha/beta estimated over a 250-trading-day window ending 30 days before each event (standard gap), used to compute cumulative abnormal returns (CAR) over the event window

**Multifractal predictability limits (Paper 11):**
- **Data:** raw, untransformed daily price (no log/return/normalization transform) via yfinance, 15-instrument sample (SPY, QQQ, IWM, XLK, XLF, XLE, AAPL, MSFT, JPM, XOM, GLD, BTC-USD, TLT, EURUSD=X, ^VIX)
- **Method:** Double Trace Moment (DTM) cascade estimation, structure-function scan, and correlated/decorrelated moment decomposition across τ = 1–300 trading days (q = 2, 4), cross-validated against the core CPE framework's own signal density at SPY's ~252-day horizon
- Full pipeline documented in `notebooks/predictability_paper/README.md`

**Regime-conditioned price forecasting (Paper 12):**
- **Data:** yfinance daily adjusted close, 22-instrument universe (equities, sector ETFs, gold, FX), plus credit (HYG/LQD) and VIX-term-structure (VIXM/VIXY) regime proxies
- **Method:** LightGBM quantile regression (5 quantile levels) on each instrument's own multifractal features (reused from Paper 11) interacted with two causally-validated regime signals, selected per instrument from four candidates (climatology, credit-regime, VIX-regime, combined) via a genuine chronological selection/holdout split (`HOLDOUT_START = 2022-01-01`) — a data-snooping bug in an earlier selection procedure was caught and fixed mid-project (one instrument's headline skill score was ~12× inflated before the fix). Gradient-boosted trees were chosen over an earlier model-family comparison (OLS, random forest, XGBoost, LightGBM, a feed-forward neural net) that found tree ensembles broadly dominant; LightGBM specifically for its native quantile ("pinball loss") objective, which produces the full 5-quantile forecast band directly.
- **Economic validation:** five independently designed trading-strategy tests (directional, price-target, portfolio, Kelly-sized, cross-sectional relative-value) and five independently designed post-processing/bias-correction designs, all benchmarked against each instrument's own buy-and-hold return (a benchmark-specification bug — testing against a generic market index instead — produced one spurious "significant alpha" result, caught and corrected)
- **Live-deployment bug, caught post-launch:** the five quantile levels are fit as independent models with no constraint that they stay ordered ("quantile crossing") — found via direct inspection of live dashboard output (one instrument's median forecast briefly fell below its own 10th-percentile forecast), invisible to every backtest metric since those only score the median. Fixed with monotone rearrangement (Chernozhukov, Fernández-Val & Galichon, 2010); verified 0/22 instruments affected after the fix.
- Full pipeline documented in `notebooks/predictor_v1/` and `notebooks/predictor_v1_paper_draft.md` (PDF also available, ready for Zenodo submission)

**Predictability limits as AI/ML regime detectors (Paper 13):**
- **Data:** reuses Paper 11's already-published predictability limits directly (no new estimation) for the 12 instruments where both a master-model decision and a predictability limit exist
- **Method:** a deliberately unregularized decision tree (`max_depth=None, min_samples_leaf=1, min_samples_split=2`; verified 0.000000 training error, one leaf per row) trained on a w = ⌊τ\*/2⌋-day window immediately preceding each test window ("fresh") versus an equally-sized window drawn from ≥2τ\* days earlier ("stale"), both evaluated on the identical test window, walked forward across each instrument's full history; tested on two independent feature variants (the full multifractal/regime-interaction feature set, and price-only lagged returns)
- Full pipeline documented in `notebooks/predictor_v1/` (scripts 59–64) and `notebooks/regime_detector_paper_draft.md` (PDF also available, published: zenodo.org/records/21482869)

**Testing AI/ML architecture, depth, and training-window size (Paper 14):**
- **Data:** yfinance daily adjusted close, reusing Paper 12's 22-instrument universe and Paper 11's already-published predictability-limit results directly (no new estimation performed)
- **Method:** three tests against the same measured limit — (1) five architecturally distinct models (climatology, an unregularized tree, RL policy-gradient, conditional GAN, conditional VAE) given an identical fair training budget, on the 3 instruments with the largest measured predictability limits; (2) a direct follow-up giving three of those models a genuine one-hidden-layer nonlinearity, hand-implemented with manually derived backpropagation and validated on synthetic data before use; (3) a training-window-size sweep from 0.5x to 8x the predictability limit across all 12 instruments, with fixed test segmentation throughout so only training-data amount and staleness vary
- Full pipeline documented in `notebooks/predictor_v1/` (scripts 65–70) and `notebooks/architecture_ceiling_paper_draft.md` (PDF also available, published: zenodo.org/records/21696948)

**Loss functions and calibrated uncertainty against the same limit (Paper 15):**
- **Data:** yfinance daily adjusted close, reusing Paper 12's 22-instrument universe and Paper 11's already-published predictability-limit results directly (no new estimation performed)
- **Method:** two further tests against the same measured limit — (1) quantile (pinball) loss at 5 quantile levels with monotone rearrangement, and escalating-order Lq loss (q = 2, 4, 6, 8), each on the largest-limit instruments; (2) a two-stage generative downscaler — climatology for the point forecast, plus a conditional VAE trained at each instrument's own predictability-limit scale (rather than the forecast horizon) to generate a calibrated distribution — with independent τ\*-scale blocks chained into full-horizon ensembles, decoder variance accounted for via the law of total variance, and ensemble dispersion rescaled against each instrument's real recent-window historical variance to reflect markets' sub-linear variance growth beyond the limit. Scored against all five Paper 14 architectures via the Continuous Ranked Probability Score (CRPS, an unbiased ensemble energy-score estimator), first holding those architectures to a single point forecast, then giving each its own natural uncertainty source (leaf-mate ensembles for the tree, the annealed policy variance for the RL forecaster, raw internal samples for the GAN/VAE) for a fair comparison
- Full pipeline documented in `notebooks/predictor_v1/` (scripts 73–80) and `notebooks/loss_uncertainty_ceiling_paper_draft.md` (PDF also available, published: zenodo.org/records/21802729)

---

## Limitations

This repository is intended for research purposes. Honest limitations reported across the paper series.

**A note on method for every 2026-08 entry below:** this project studies rare, extreme, non-stationary market behavior, which is properly the domain of extreme-value statistics, not mainstream large-*n*/Gaussian-adjacent inference. Findings here are reported as real point estimates (alpha, Sharpe, drawdown, episode counts) judged by whether they replicate across genuinely separate real out-of-sample evidence — a different real year, a different real instrument, more real independent episodes — never by a classical significance test (t-statistics, p-values, confidence-interval pass/fail gates, bootstrap or permutation nulls) layered on top. This is a deliberate, standing choice, not an oversight: rarity of extremes is this field's expected signature, not evidence of an untrustworthy small-sample estimate that a significance test needs to adjudicate.

**Core CPE framework (Papers 1–5):**
- **Overlapping weekly observations:** Portfolio return series use overlapping weekly observations, which inflate the apparent independent sample size of any classical significance test — one further reason this project reports real point estimates and out-of-sample replication rather than significance verdicts throughout (see the standing methodology note at the top of this section). **Update:** a follow-up re-test (below) checked whether the portfolio-tilt engine's single large 2025 result held up in other real years, rather than leaning on any corrected test statistic — it held up in that one year and did not repeat in the other three tested.
- **Bitcoin ETF concentration:** 103,983 calibration instances are dominated by IBIT, BITB, and FBTC — three near-identical instruments. Effective independent count is substantially smaller than nominal.
- **Bearish signal failure:** Bearish CPE signals for gold achieved only 29% realised hit rate against 83.7% stated CPE. Root cause: UVXY/VIXY structural regime change in 2025, where volatility spikes coincided with gold surges rather than gold weakness as in the training period.
- **Non-stationary predictor thresholds:** Leveraged VIX ETPs (UVXY, VIXY) lose value continuously through roll decay. Their 252-day quantile thresholds become structurally unachievable in live evaluation, causing high-CPE training configurations to never fire out-of-sample.
- **Single regime:** 1.4 years of evaluation covers one sustained gold bull market. Sustained bear markets, credit crises, and deflationary environments have not been tested.
- **Data sufficiency:** The validated vol→equity channel rests on 4 independent training-period episodes. Five observations (including the 2025 OOS result) cannot distinguish genuine predictive content from a well-supported coincidence. Approximately 3–5 additional independent episodes are needed.
- **Transaction costs:** Not modelled. Transaction cost break-even is ~10.1 bps per one-way leg for the 5-sleeve strategy — above realistic ETF costs but sensitive to AUM and operational overhead.
- **Multi-year re-test (2026-08): the originally reported portfolio-tilt Sharpe has not yet repeated out-of-sample.** Both portfolio-tilt mechanisms (`files/backtest_engine.py`, the current, independent reimplementation built from the Portfolio Tilt paper's own spec, since the original scoring code no longer exists) were re-run across four independent years (2022–2025, each trained honestly through the prior year-end), comparing real point estimates year by year rather than the original randomisation-test methodology: hold-to-horizon (Paper 3's mechanism) showed a large result in exactly 1 of the 4 years (2025, +23.87%/yr — concentrated almost entirely in gold, which opened 28 holds that year against 0 in 2022) and small, mixed-sign results the other 3 years (+0.16%, -1.53%, +1.73%); static tilt (Paper 4's mechanism) showed small, mixed-sign results close to zero in all 4 years (+0.13%, +0.93%, -0.86%, +0.56%). Benchmarked over the same years against two published, unfitted control strategies — Time-Series Momentum (Moskowitz, Ooi & Pedersen, 2012) and Cross-Sectional Momentum (Jegadeesh & Titman, 1993) — TSMOM showed the identical pattern (small, mixed-sign, near-zero results in every recent year: +0.02%, -2.37%, +1.68%, -1.78%, despite a large full-sample result of +2.57%/yr over 1994–2026 earned almost entirely during the 2008 crisis alone), while XSMOM showed large negative results in 2 of the 4 years (-15.88%, -15.42%). This does not invalidate the original Papers 3–4 results, which used a different methodology and evaluation window, but judged by real year-by-year replication, the portfolio-tilt strategy has not yet shown a repeatable, demonstrated edge. Full methodology and all four years' results: `files/cpe_hth_multi_year_test.py`, `files/cpe_static_tilt_multi_year_test.py`, `tsmom_benchmark.py`, `xsmom_benchmark.py`, `tsmom_crisis_alpha_check.py`, `master_alpha_comparison.py`.
- **Breadth expansion (2026-08): nominal breadth is not effective breadth.** The production portfolio-tilt engine trades only 5 sleeves, but the underlying, already economic-prior-gated joint screen (`joint_cpe_results.parquet` — confirmed directly from `joint_cpe_engine.py`'s own source to already apply the Paper 3 `is_admissible()` gate, not an unrestricted screen) has real, supported configurations for 132 distinct tradeable targets. Expanding to the top 40 of these by supporting-configuration count (a mechanical, pre-registered selection criterion — chosen by evidence volume, not by which targets happened to perform well) and re-running the same four-year test found small, mixed-sign results in every year (-0.19%, -0.00%, -0.22%, +0.84%), smaller in magnitude than the 5-sleeve version. The reason: at most 5 of the 40 targets ever placed a real tilt in any single year — the `EPISODE_MIN_OBS_FOR_CONVICTION=3` floor (a deliberate anti-luck gate, not an oversight) leaves most targets neutral most of the time, so more instruments in the universe did not translate into more independent bets. Lowering that floor to 2 (the only other defensible value — floor=1 is a shown-degenerate case: 2,987 one-episode configurations show a suspicious 95.2% "perfect" hit rate, an unmistakable small-sample/selection artifact) roughly doubled how many targets were actually active and moved every year's result positive (+0.21%, +0.03%, +1.35%, +1.03%), but magnitudes stayed small, and at a quantified reliability cost: two-episode configurations show their one out-of-training episode landing wrong entirely (0% hit rate) 7.3% of the time, more than 3× the three-episode rate of 2.1%. Read together: breadth is very likely the right lever in principle (Grinold's `IR ≈ IC·√breadth`; both changes moved every single year's result in the same direction) but the framework's own strict admissibility/episode-independence gates — the same rigor that makes any individual signal trustworthy — currently starve it of enough genuinely independent, reliable signal to realize that benefit at scale; closing this gap needs a much larger admissible universe or more years of accumulated episodes, not a looser threshold. Full methodology: `files/cpe_breadth_expansion_test.py`, `files/cpe_breadth_expansion_floor2_test.py`.
- **Crisis-type diversification, part 1 (2026-08): the original admissible-screen search found nothing, but that search itself was incomplete.** A systematic search of the *existing* joint screen for a mechanistically different crisis signal (credit-spread-widening / inflation-shock, the 2022 regime, where bonds sell off *with* stocks rather than rallying against them) found none among already-computed configurations. `economic_prior.py` already admits `("credit_hy", "broad_equity_us")` — no prior expansion needed, contrary to what was first stated here — the real gap was that the existing screen had never surfaced a bearish-direction configuration using it.
- **Crisis-type diversification, part 2 (2026-08): a pre-registered, targeted search for that missing configuration, complete with two of my own real errors caught and corrected along the way — documented in full because they're instructive, not despite that.** Hypothesis, fixed before touching any 2022+ data: high-yield credit (HYG/JNK) bearish *and* long-duration Treasuries (TLT/TLH) *also* bearish, jointly, predicts equity weakness. **Error 1:** the first version of the search silently discarded any configuration with fewer than 100 raw joint observations, a threshold borrowed from the original decades-spanning screen — structurally guaranteed to reject exactly the "rare, regime-specific" pattern being searched for. Removing that filter revealed the joint condition fires 36-52 times across all of available history, concentrated *entirely* in 2022 — a real, historically-corroborated fact (2022 was independently the worst joint stock/bond year in decades) that the first search never surfaced. **Error 2:** a quick manual check of this single-episode result used a sign-inverted forward-return calculation and was reported here as confirming the hypothesis (subsequent equity weakness). Direct, careful re-verification against raw price levels (not shift-based tricks) showed the opposite: the one real 2022 episode anchors within days of the actual market bottom (Sept-Nov 2022), and SPY/QQQ/IWM/DIA subsequently rose sharply at every horizon tested (median +8% to +19%) — confirmed robustly across ~75 of ~84 credit/rate/target/horizon combinations checked directly against price data. **The corrected, final finding:** this joint credit+rate extreme is real and historically singular to 2022, but it behaves as a *capitulation/bottom-marking* signal, not a crisis-continuation one — the mechanistic opposite of the original hypothesis. It remains, honestly, an n=1 finding (one real episode) and cannot be validated as reliable regardless of how directionally clean its single outcome is — the same standard this program already applies to any 1-2 episode configuration. Broader (less extreme, q=0.20) versions of the same joint condition do have enough real episodes (4-5) to be evaluable, and their actual episode-level hit rate is poor (0.0-0.25, below the 50% baseline) — so no version of this hypothesis, at any threshold tested, is both statistically evaluable and predictively reliable. Full methodology, including both bugs left visible in the script's own history rather than quietly fixed: `files/credit_stress_channel_search.py`.
- **Combining CPE's crisis detector with TSMOM's crisis-period return (2026-08): tested directly, does not help, and the reason is a genuine timing mismatch, not a tuning failure.** Hypothesis: scale up TSMOM's exposure specifically when CPE's vol-complex channel detects a crisis starting, rather than running TSMOM at constant weight. Tested at 1.5x and 2.0x scale-up (two round, non-optimized multipliers) on CPE's 127 active days (2014–2026 OOS): Sharpe *fell* slightly at every scale (0.277 → 0.264 → 0.249), and within 2020 — the only year the signal ever fired — TSMOM's own return went from -0.05%/yr unscaled to -0.13%/yr at 2x. Diagnosed directly: TSMOM's own large crisis-period return (established above) is earned over *sustained, multi-month* drawdowns (the 2008 episode alone, 543 days, carries almost the entire full-sample result), while CPE's detector fires at the *onset* of a fast, V-shaped shock (2020's crash resolved in weeks) — a 12-month-trailing trend signal is structurally too slow to have repositioned correctly by the time a sudden crisis is already underway, so scaling up exposure at onset just amplifies whatever TSMOM happens to be doing at that moment, which in 2020 was not yet right. A different combination logic (e.g. scaling on crisis *duration survived*, not onset) might align better, but that is a new, untested hypothesis. Full methodology: `files/tsmom_cpe_crisis_combo_test.py`.
- **A better-motivated TSMOM combination (2026-08): a small, real, replicated improvement, built from what the two failed attempts above actually diagnosed rather than a fresh guess.** Two design changes, each grounded in an already-validated finding rather than the mechanisms already shown not to work: (1) restrict TSMOM to only the instruments whose own measured predictability structure (Ramanathan 2026a) shows a genuine pocket near TSMOM's own 252-day lookback — SPY, IWM, AAPL, MSFT, GLD — rather than applying the identical lookback to every instrument regardless of its own structure (using tau* to select *which instruments to trade*, not to change *how the signal is built*, since the earlier tau*-as-lookback attempt failed for a specific, diagnosed reason: tau* measures magnitude/volatility persistence, not directional momentum persistence); (2) blend CPE's crisis detector into the book *additively*, as its own small, independent sleeve, rather than as a multiplier on TSMOM's exposure — avoiding the precise-timing requirement that broke the multiplicative combination above. Tested on 12 tau\*-measured instruments, 2006–2026 (the period all 12 genuinely have data — an earlier, incorrect version of this test used 1963 as the start date, which meant the 1963–2006 portion was silently running on 1–2 real instruments instead of a genuine multi-instrument portfolio; caught by inspecting the output directly and fixed before reporting): pocket-filtered TSMOM beat the unfiltered 12-instrument baseline in 3 of 5 real, non-overlapping historical blocks (full-sample Sharpe 0.32 vs. 0.29). Adding the CPE sleeve won outright in only 1 of 5 blocks — it is a drag on raw return most of the time, visibly so in the equity curve, exactly as expected of a sleeve that sits in cash over 96% of the time — but full-sample Sharpe still improved (0.36) and max drawdown shrank (-13.1% vs. -16.1% pocket-filtered alone), consistent with a genuine small diversifier rather than a return enhancer. Not a demonstrated edge on this evidence alone (one full-sample estimate, five blocks), but a small, real, mechanistically-coherent improvement that moved in the direction its own diagnosed logic predicted, unlike either of the two combinations tested before it. Full methodology: `combined_pocket_tsmom_cpe_strategy.py`.
- **Every strategy's real alpha, plotted on one scale (2026-08): no single approach shows a clean, broadly-replicated edge; most cluster near zero; both published controls sit clearly negative on this specific test.** Every individual real observation (a real year, a real historical block, a real instrument) for every strategy above is plotted as its own point, not collapsed into one number: `alpha_generated_plot.png`. Three things stand out directly from the plot. First, CPE hold-to-horizon's positive median is carried by one isolated point (+23.87%, 2025) sitting far from its other three observations clustered near zero — visible directly, not hidden by the median. Second, the per-instrument strategies (Paper 12 Kelly-sized and master model, RL sizing) show the widest real spread of any strategy tested — some instruments show real double-digit alpha in both directions — which is a different, more specific finding than "no signal": it points to real instrument-level inconsistency rather than a uniformly broken approach. Third, both published controls sit clearly on the negative side on this specific test: XSMOM's calendar-block observations cluster tightly negative with no outlier doing the work, and TSMOM's calendar-block alpha is also negative throughout — **but this specific chart uses calendar blocks as the replication unit for TSMOM, already shown above to be the wrong unit for a strategy whose real edge concentrates in real crisis episodes (9 of 13) rather than spreading evenly across arbitrary chronological time; TSMOM's negative showing here should be read as "no edge distributed evenly across calendar time," not "no edge at all."** Full methodology: `alpha_generated_plot.py`.
- **Which instruments show real edge, and added value vs. an existing quant-shop tool on those same instruments (2026-08).** Checking Paper 12's own two independently-designed per-instrument methods (Kelly-sized, master model) for *convergence* — the same instrument, positive alpha, in both, on the same 2022+ holdout — found six: JPM (Kelly +11.64%, master model +2.78%, RL sizing +0.13% — the only instrument positive across all three independently-built methods tested this session), GLD (+6.84%/+2.08%), XLE (+6.55%/+2.88%), XLB (+2.53%/+1.49%), XLU (+1.16%/+0.91%), MSFT (+0.48%/+0.12%). Standard, unfitted TSMOM (the same 252d/63d spec used throughout this session) was then applied to each of these exact six instruments, individually, over the identical 2022+ holdout, against the identical own-buy-and-hold benchmark: TSMOM was negative on every single one, often sharply (JPM -10.99%, GLD -8.32%, XLE -17.46%, XLB -9.64%, XLU -5.06%, MSFT -9.90%) — see `added_value_vs_tsmom_plot.png`. Read honestly, not as "beats TSMOM in general": TSMOM's own established edge (above) comes from a diversified, 29-instrument book and from rare, sustained crisis episodes, not from trading one instrument alone — stripped of diversification and applied single-name in a non-trending 2022–2025 stretch, standard trend-following struggles badly, which is what shows up here. The fair statement of added value is narrower and more specific: on the instruments where Paper 12's regime-conditioned, single-instrument methods show real convergent signal, they meaningfully outperform a diversification-dependent tool used outside the conditions its own edge depends on — not evidence of general superiority, evidence of being better suited to a different, specific problem. Full methodology: `added_value_vs_tsmom.py`.
- **Correction, same day: TSMOM was only tested on the 6 already-selected instruments above — running it against the other 16 changes the claim.** The prior entry's comparison had a real selection issue, flagged before it was run: it only checked TSMOM on instruments already chosen for showing the user's own convergent signal. Running the identical TSMOM spec against the remaining 16 of Paper 12's 22 instruments, same 2022+ holdout, same benchmark, found TSMOM negative on 15 of those 16 too (only IYR positive, +2.76%) — TSMOM is negative on 21 of 22 instruments in this period, essentially regardless of whether the user's own methods find signal there. **This means the "added value" finding above is not "our methods work specifically where TSMOM struggles" — TSMOM struggles almost everywhere single-name in this period, which is a separate fact about single-instrument trend-following in 2022–2025, not something caused by or specific to where the user's own edge lives.** There is a smaller, real, honest signal underneath that broader fact worth keeping: TSMOM's median alpha is meaningfully worse on the 6 own-signal instruments (-9.77%) than on the other 16 (-4.65%), more than double — so TSMOM's already-broad struggles are notably worse, in degree, on exactly the instruments where the user's own methods find real signal. The correctly-scoped claim is a differential in degree, not an exclusive or causal relationship. Full methodology: `tsmom_all22_vs_own_methods.py`.
- **Final synthesis of the entire 2026-08 investigation, stated plainly rather than left implicit across ~15 separate entries above.** What's now firmly established: five prediction-layer levers (Papers 12–15: architecture, depth, data volume, loss function, calibrated uncertainty) and a second wave of decision-layer levers tested here (RL sizing, tau*-informed conviction, breadth expansion, direct combination with a real published strategy) all fail to buy a demonstrated, broadly-repeatable, portfolio-level edge past the measured predictability limit — the single most triangulated finding in the whole program, from roughly a dozen independently-designed tests converging on the same wall. What's also been genuinely validated, not merely asserted: this program's founding premise that extreme, regime-specific, non-Gaussian structure is real and matters. CPE's vol-complex channel is a real, EVT-verified crisis detector; TSMOM's own established edge is itself proof of the thesis, earned almost entirely in one 543-day stretch of 2008, invisible to any smoothed analysis; JPM and five other instruments show real, convergent signal across independently-built methods, outperforming a diversification-dependent tool on those exact instruments; real, independently-measured predictability structure (SPY's own 252-day pocket) tracks real where momentum concentrates. One real, modest, honestly-validated construction came out of it: pocket-filtered TSMOM (instrument selection via real predictability structure) plus an additive CPE crisis sleeve, a small improvement (3/5 real blocks, Sharpe 0.29→0.36, reduced drawdown) built from what two failed combinations actually diagnosed. What is *not* yet shown: a demonstrated, portfolio-level, tradeable strategy that clearly and repeatably beats existing quant-shop tools — every positive finding here is real but narrow (specific instruments, specific episodes, one specific combination) and modest in magnitude. Closing that gap needs a genuinely different, validated crisis-type detector (searched for directly — see the credit-stress-channel entries above — and found something real but statistically unvalidatable from a single episode) and more real, live track record, which no amount of further backtesting substitutes for. The most durable outcome of this investigation may not be any single number: every finding above is now reported as a real point estimate plus genuine out-of-sample replication, matched to the right unit for each phenomenon, rather than a classical significance verdict — and negative results now get the same adversarial scrutiny as suspiciously good ones, a discipline that caught two real errors in the credit-stress-channel search alone.

**Climate-finance extensions (Papers 6–9):**
- **Geographic mismatch (found in Paper 6, corrected in Paper 7):** Paper 6's city-centroid temperature predictors did not align with the actual sugar-growing regions driving CANE futures — a signal with real statistical properties (lift 1.42–1.63×) but no plausible geographic transmission mechanism. Paper 7 rebuilt the predictor set using ERA5 gridded crop-zone coordinates; the sugar signal disappeared entirely (zero surviving configurations), while genuine wheat, corn, and natural-gas channels emerged instead. Reported as a negative result rather than suppressed.
- **Small independent-episode counts:** Several climate-predictor findings (e.g. the El Niño/monsoon → sugar analysis) rest on fewer than 10 historical episodes since 1990. Directionally consistent, but not enough to rule out coincidence to the same standard as the core CPE framework's larger-N signals.

**Hurricane/reinsurer event study (Paper 10):**
- **One of two hypotheses failed to replicate:** the RenaissanceRe (RNR) short-horizon loss reaction held up under two independent statistical methods, but a hypothesized medium-horizon repricing effect in Munich Re did not — it was numerically indistinguishable from a hit of identical strength in a non-cat-exposed control ticker, and is reported as a null result rather than reframed as a weaker positive.
- **Small event count:** 14 hurricane landfalls since 1995 limits statistical power relative to the core CPE framework's much larger signal-count studies.

**Multifractal predictability limits (Paper 11):**
- **Not a universal claim:** predictability regimes are instrument- and moment-order-dependent (persistent / single-crossing / oscillating), not a single decay law — see the paper's three-regime typology (Section 5.2) before generalizing any one instrument's result to others.
- **DTM regression fit is comparatively weak** (R² 0.44–0.46) due to price-trend contamination in the raw (deliberately untransformed) field, though the structure-function and correlated/decorrelated decomposition results this paper relies on most are much better fit (R² > 0.98).

**Regime-conditioned price forecasting (Paper 12):**
- **Zero instruments show demonstrated tradeable alpha.** This is the headline limitation, not a footnote: across five independently designed trading-strategy tests (directional, price-target, portfolio, Kelly-sized, cross-sectional relative-value long/short), none of the 22 instruments shows statistically significant risk-adjusted alpha against the properly specified benchmark (its own buy-and-hold return). Real, holdout-honest forecast-accuracy skill exists for roughly half the panel, but does not translate into economic value for any instrument tested — the live dashboard is deliberately built as a forecast-accuracy tracker with no buy/sell signal, for exactly this reason.
- **Post-processing/bias-correction only helps 2 of 22 instruments** (GLD, JPM), across five independently designed correction techniques — the other 20 are made worse by every correction attempted, evidence their forecast errors are irreducible noise rather than a correctable bias.
- **Overlapping-window t-statistics:** the alpha significance tests use analytic OLS standard errors appropriate to the point estimates tested, but do not yet correct for autocorrelation in overlapping long-horizon (63–252 day) return windows — an analytic effective-sample-size correction, not a resampling-based fix, is the natural next step.

**Predictability limits as AI/ML regime detectors (Paper 13):**
- **Coverage.** Only 12 of the 22 live-forecasting instruments have an already-published Paper 11 predictability limit; the paper's evidence is confined to those 12 deliberately, rather than estimating a limit for the remaining 10.
- **Short-horizon legibility.** IWM and QQQ's 21-day horizon produces a visibly noisier fresh-vs-stale comparison in both feature variants, making the paper's central visual claim harder — though not obviously wrong — to assess by eye for exactly these two instruments.
- **No independent ground truth for "regime change."** The paper demonstrates that fresh training tracks subsequent reality better than stale training; it does not independently establish that the moments where this gap widens correspond to any externally verifiable definition of a regime shift — a deliberate, stated choice of evidentiary standard (direct curve comparison, no significance testing), not an oversight.
- **The overfit-tree diagnostic is a probe, not a product.** It is deliberately built with zero regularization, specifically to be maximally sensitive to a training-window boundary; it is not proposed as a deployable forecasting model in its own right.

**Testing AI/ML architecture, depth, and training-window size (Paper 14):**
- **Scale of implementations.** All non-trivial architectures are numpy-native, sized to training windows of 11–504 rows, not production-scale deep networks — though depth itself is directly tested (a genuine hidden layer, hand-backpropagated) and found not to change the result, so the finding is not merely a fact about linear models. Large-scale, heavily-parameterized versions trained on data pooled across many instruments remain untested, and would themselves require training past the predictability limit for any single instrument — the very thing the window-size sweep shows degrades performance.
- **Gradual, not a sharp cliff.** Degradation as training-window size grows is progressive in every instrument, consistent with a genuine structural limit but meaning "beyond the limit" is a matter of degree, not a bright line a reader can point to on any single panel.
- **IWM and QQQ's flat/reversing error curves are explained, not just reported.** Their 21-day forecast horizon leaves them with only 6–17% fresh-trained skill over a naive no-change baseline, versus 57–76% for the other ten instruments — a dose-response relationship along horizon length, not an unexplained anomaly.
- **No metrics, by design, not by omission.** As in Paper 13, no significance test or skill score is reported for the architecture comparisons; the window-size degradation is quantified directly as mean absolute error (expressed as % of price), a descriptive summary rather than an inferential test.

**Loss functions and calibrated uncertainty against the same limit (Paper 15):**
- **The downscaler's point forecast is not new predictive skill.** It is, by mathematical construction, provably identical to plain climatology's — the entire contribution is a better-calibrated *distribution* around the same central prediction, and the paper is explicit that this is what CRPS is rewarding, not improved accuracy.
- **The generative downscaler loses to climatology's own empirical distribution on 7 of 12 instruments** once every architecture is given a fair, uncalibrated uncertainty estimate rather than being forced into a single point forecast — the headline result is nuanced, not a clean win, and is reported as such.
- **Escalating-order Lq loss is untested on flexible models at this paper's scale beyond the earlier instability already documented in Paper 14's own development history** — the linear-model result (no measurable change at any q) rules out loss-order as the mechanism, but does not itself demonstrate a flexible model's stability under high-order Lq loss.
- **Quantile and Lq loss results are confined to the largest-limit instruments**, not swept across the full 12/22-instrument panels used elsewhere in the series.

The outputs should be interpreted as evidence of statistical structure and a research prototype that has cleared a first significance threshold — not a deployable trading strategy.

---

## Live Dashboards

Updated daily via automated pipeline. All predictions are publicly timestamped and verifiable.

- [Gold Buy Signal Dashboard](https://quantarram.github.io/quant-regime-research/notebooks/gold_dashboard.html)
- [Multi-Asset Portfolio Tilt Dashboard](https://quantarram.github.io/quant-regime-research/notebooks/portfolio_dashboard.html)
- [Precious Metals Dashboard (Gold/Silver/Platinum)](https://quantarram.github.io/quant-regime-research/notebooks/precious_metals_dashboard.html)
- [CPE Atlas Explorer (169K signals)](https://quantarram.github.io/quant-regime-research/notebooks/cpe_dashboard.html)
- [Predictor Dashboard (22-instrument price forecasts, Paper 12)](https://quantarram.github.io/quant-regime-research/notebooks/predictor_dashboard.html) — forecast-accuracy tool only, deliberately no buy/sell signal (see Paper 12's limitations above)

---

## Visual Overview

- [Framework Infographic (PDF)](notebooks/infographic_combined.pdf) — a 3-panel explainer covering what conditional exceedance is, a worked gold example, and the multi-asset atlas. **Note:** this is a point-in-time snapshot (16 June 2026) — the embedded prices, CPE values, and signal counts are illustrative of the methodology, not current; see the Live Dashboards above for today's numbers.
- [CPE vs. Traditional Quant Tools](https://quantarram.github.io/quant-regime-research/notebooks/infographic_1_cpe_v2.html) — the framework's core pitch in one panel: traditional approaches fit a model, choose a distributional family, and hope it holds out-of-sample; CPE instead counts historical occurrences directly and states a verifiable probability, no distributional assumptions. Uses Paper 1's published statistics (161 instruments, 169k surviving signals, 4.85× peak lift), not live data.
- [Multi-Asset CPE Atlas](https://quantarram.github.io/quant-regime-research/notebooks/infographic_2_atlas_v2.html) — maps the strongest tail co-movement channels into gold (crypto ETFs, silver, gold volatility, USD weakness). Most figures are fixed Paper 1 statistics, but the "currently firing ✓" / "currently X% away" annotations reflect a mid-June 2026 snapshot, not today's signal status — check the CPE Atlas Explorer dashboard above for live firing status.
- [The Outliers Matter (Poster)](notebooks/final_poster_v12.png) — the fullest version of the "why CPE sees what standard tools miss" argument: correlation, ARIMA, GARCH, and ML are second-order, mean-seeking methods that minimize average error, while markets are disproportionately decided by rare extreme cases. Backs this with three same-data, two-views demonstrations — a simulated tail-shock example, the El Niño/sugar-price analysis (Paper 9), and the hurricane landfall/RenaissanceRe event study (Paper 10) — each shown once as an ordinary scatter/regression (which sees nothing) and once as CPE's conditional view (which finds a 2–2.6× effect). Companion piece to the ["Outliers Matter"](https://arunramanathans.substack.com/p/the-outliers-matter-and-most-of-your) Substack post.
- [Fresh-vs-Stale Training Schematic (Paper 13)](notebooks/predictor_v1/schematic_fresh_vs_stale_design.png) — a timeline diagram of Paper 13's training-window prescription: fit a model on the instrument's own predictability-limit window and apply it to the immediately following window (fresh) versus an equally-sized window from more than twice that limit in the past (stale), both tested against the same real outcome.
- [Architecture Bake-off, Before and After the Limit (Paper 14)](notebooks/predictor_v1/67_window_sweep_highlight_EURUSDX.png) — EURUSD=X, five architecturally distinct models all trained on the same fair budget: tracking the real exchange rate closely at half the predictability limit, a persistent overshoot appearing by twice the limit, and a gap that keeps widening — not just growing noisier — by eight times the limit.
- [Five Levers Against the Ceiling (Paper 15)](notebooks/predictor_v1/p15_graphical_abstract.png) — a graphical summary of every lever tested against the measured predictability limit across Papers 14 and 15 (architecture, depth, training-window size, loss function, calibrated uncertainty), and the one result that reshapes the headline: climatology's own honest, unmodeled uncertainty beats a carefully calibrated synthetic alternative on more than half the panel.

---

## Research Papers

- [Paper 15: The Ceiling Holds, and So Does Climatology: Loss Functions and Calibrated Uncertainty Against an Empirically Measured Predictability Limit](https://zenodo.org/records/21802729)
- [Paper 14: The Ceiling Holds: Testing AI/ML Architecture, Depth, and Training-Window Size Against an Empirically Measured Predictability Limit](https://zenodo.org/records/21696948)
- [Paper 13: Predictability Limits as Regime Detectors: A Practical Rule for How Much History an AI/ML Model Should Train On](https://zenodo.org/records/21482869)
- [Paper 12: A Master-Model Framework for Regime-Conditioned Price Forecasting: Real Statistical Skill, and Why It Mostly Isn't Alpha](https://zenodo.org/records/21454884)
- [Paper 11: Empirical Predictability Limits of Financial Markets via Correlated–Decorrelated Structure Function Decomposition: A Departure from Atmospheric Turbulence Theory](https://zenodo.org/records/21373459)
- [Paper 10: Do Major Hurricane Landfalls Move Reinsurer Equity?](https://zenodo.org/records/21231343)
- [Paper 9: When the Geography is Wrong, the Signal is Wrong](https://zenodo.org/records/21057110)
- [Paper 8: VPD and Moisture Stress...](https://zenodo.org/records/21021264)
- [Paper 7: Agricultural Crop-Zone Temperatures in the CPE Framework: Heat Stress Thresholds, Growing Degree Days, and the Reversal of the Paper 6 Sugar Signal](https://zenodo.org/records/20993837)
- [Paper 6: Beyond Tail Co-Movement: How Temperature Extremes Shift Financial Return Distributions](https://zenodo.org/records/20964819)
- [Paper 5: Corrected Inference for the CPE Portfolio Tilt Strategy: Newey-West HAC Standard Errors and Robustness Checks](https://zenodo.org/records/20908417)
- [Paper 4: Signal-Level Calibration and Dashboard Utility of the CPE Framework](https://zenodo.org/records/20830462)
- [Paper 3: From Descriptive Atlas to Tradeable Signal](https://zenodo.org/records/20815386)
- [Paper 2: A Conditional Exceedance Framework for Interpretable Trading Decisions](https://zenodo.org/records/20769150)
- [Paper 1: A Descriptive Atlas of Conditional Exceedance Structure Across a Multi-Asset Universe](https://zenodo.org/records/20606184)

---

## Substack Articles

- [I Built a Model That Admits What It Doesn't Know](https://arunramanathans.substack.com/p/i-built-a-model-that-admits-what) (Paper 15)
- [Can a Smarter Model Beat a Real Predictability Limit?](https://arunramanathans.substack.com/p/can-a-smarter-model-beat-a-real-predictability) (Paper 14)
- [How Much History Should Your Model Actually See?](https://arunramanathans.substack.com/p/how-much-history-should-your-model) (Paper 13)
- [What Weather Taught Me About Market Predictability](https://arunramanathans.substack.com/p/what-weather-taught-me-about-market)
- [33 Historical Episodes Were Actually 3](https://arunramanathans.substack.com/p/33-historical-episodes-were-actually)
- [Do Hurricanes Move Reinsurer Stocks? One Yes, One "Actually, No"](https://arunramanathans.substack.com/p/do-hurricanes-move-reinsurer-stocks)
- [The Outliers Matter (And Most of Your Tools Aren't Looking For Them)](https://arunramanathans.substack.com/p/the-outliers-matter-and-most-of-your)
- [Does the 2026 El Niño Move Sugar Prices? What Eight Historical Events Actually Show](https://arunramanathans.substack.com/p/does-the-2026-el-nino-move-sugar)
- [The Signal That Wasn't There](https://arunramanathans.substack.com/p/the-signal-that-wasnt-there)
- [Paper 8: When Heat Meets Drought — The Strongest Signals in the CPE Series](https://arunramanathans.substack.com/p/paper-8-when-heat-meets-drought-the)
- [From the City to the Field: Solving the Geographic Mismatch in Climate-Finance](https://arunramanathans.substack.com/p/from-the-city-to-the-field-solving) (Paper 7)
- [Paper 6: Temperature Extremes Don't Co-Move With Financial Tails — They Shift the Whole Distribution](https://arunramanathans.substack.com/p/paper-6-temperature-extremes-dont)
- [Building a Live Quant Signal System: What Worked, What Failed, and What the Data Says](https://arunramanathans.substack.com/p/building-a-live-quant-signal-system)
- [Paper 5: The Results Survive — HAC Correction, Block Bootstrap, and Four Robustness Checks](https://arunramanathans.substack.com/p/paper-5-the-results-survive-hac-correction)
- [Paper 4: The Conditional Probabilities Actually Hold](https://arunramanathans.substack.com/p/paper-4-the-conditional-probabilities)
- [Got One Statistically Significant Result. Then Spent Hours Trying to Break It.](https://arunramanathans.substack.com/p/from-descriptive-atlas-to-tradeable)
- [A Research Update: When a Strong Statistical Pattern Doesn't Become a Trading Edge](https://arunramanathans.substack.com/p/a-research-update-when-a-strong-statistical)
- [The Gold Dashboard Just Said BUY GRADUALLY. Here Is Exactly Why.](https://arunramanathans.substack.com/p/the-gold-dashboard-just-said-buy)
- [From One Asset to Five: Scaling the CPE Framework Into a Portfolio Tilt Dashboard](https://arunramanathans.substack.com/p/from-one-asset-to-five-scaling-the)
- [What If You Could Be Useful Without Predicting Anything?](https://arunramanathans.substack.com/p/what-if-you-could-be-useful-without)
- [I Built a Dashboard to Tell Me When to Buy Gold — Here's What It's Saying Right Now](https://arunramanathans.substack.com/p/i-built-a-dashboard-to-tell-me-when)
- [Mapping the Hidden Wiring of Global Financial Markets](https://arunramanathans.substack.com/p/mapping-the-hidden-wiring-of-global)
- [A Simple Conditional Exceedance Framework for Interpretable Trading Decisions](https://arunramanathans.substack.com/p/a-simple-conditional-exceedance-framework)
- [Conditional Exceedance Probabilities as a Basis for Systematic Trading](https://arunramanathans.substack.com/p/conditional-exceedance-probabilities)

---

## Citation

If you find this work useful, please cite the relevant paper(s):

**Paper 1 — Descriptive Atlas**
```
RAMANATHAN S, A. (2026). A Descriptive Atlas of Conditional Exceedance Structure
Across a Multi-Asset Universe. Zenodo.
https://doi.org/10.5281/zenodo.20606184
```

**Paper 2 — Single-Asset Trading Framework**
```
RAMANATHAN S, A. (2026). A Conditional Exceedance Framework for
Interpretable Trading Decisions. Zenodo.
https://doi.org/10.5281/zenodo.20769150
```

**Paper 3 — Portfolio Tilt Out-of-Sample Test**
```
RAMANATHAN S, A. (2026). From Descriptive Atlas to Tradeable Signal: An
Out-of-Sample Test of the Multi-Asset Conditional Exceedance Framework as
a Portfolio Tilt Strategy. Zenodo.
https://doi.org/10.5281/zenodo.20815386
```

**Paper 4 — Signal-Level Calibration and Dashboard Utility**
```
RAMANATHAN S, A. (2026). Signal-Level Calibration and Dashboard Utility of
the Conditional Probability Exceedance Framework: Pairwise Validation, Gold
Dashboard Evaluation, and Extended Portfolio Tilt Evidence Across 528 Trading
Days. Zenodo.
https://doi.org/10.5281/zenodo.20830462
```

**Paper 5 — Corrected Inference**
```
RAMANATHAN S, A. (2026). Corrected Inference for the CPE Portfolio Tilt
Strategy: Newey-West HAC Standard Errors and Robustness Checks. Zenodo.
https://doi.org/10.5281/zenodo.20908417
```

**Paper 6 — Beyond Tail Co-Movement**
```
RAMANATHAN S, A. (2026). Beyond Tail Co-Movement: How Temperature Extremes
Shift Financial Return Distributions. Zenodo.
https://doi.org/10.5281/zenodo.20964819
```

**Paper 7 — Agricultural Crop-Zone Temperatures in the CPE Framework**
```
RAMANATHAN S, A. (2026). Agricultural Crop-Zone Temperatures in the CPE
Framework: Heat Stress Thresholds, Growing Degree Days, and
the Reversal of the Paper 6 Sugar Signal. Zenodo.
https://doi.org/10.5281/zenodo.20993837
```

**Paper 8 — When Heat Meets Drought — The Strongest Signals in the CPE Series**
```
RAMANATHAN S, A. (2026). Vapour Pressure Deficit and Moisture Stress in the CPE Framework:
Joint Heat-Drought Conditions as the Strongest Climate-Finance Predictors. Zenodo.
https://doi.org/10.5281/zenodo.21021264
```

**Paper 9 — When the Geography is Wrong, the Signal is Wrong**
```
RAMANATHAN S, A. (2026). When the Geography is Wrong, the Signal is Wrong. Zenodo.
https://doi.org/10.5281/zenodo.21057110
```

**Paper 10 — Do Major Hurricane Landfalls Move Reinsurer Equity?**
```
RAMANATHAN S, A. (2026). Do Major Hurricane Landfalls Move Reinsurer Equity? Zenodo.
https://doi.org/10.5281/zenodo.21231343
```

**Paper 11 — Empirical Predictability Limits of Financial Markets**
```
RAMANATHAN S, A. (2026). Empirical Predictability Limits of Financial Markets via
Correlated-Decorrelated Structure Function Decomposition: A Departure from
Atmospheric Turbulence Theory. Zenodo.
https://doi.org/10.5281/zenodo.21373459
```

**Paper 12 — A Master-Model Framework for Regime-Conditioned Price Forecasting**
```
RAMANATHAN S, A. (2026). A Master-Model Framework for Regime-Conditioned Price
Forecasting: Real Statistical Skill, and Why It Mostly Isn't Alpha. Zenodo.
https://doi.org/10.5281/zenodo.21454884
```

**Paper 13 — Predictability Limits as Regime Detectors**
```
RAMANATHAN S, A. (2026). Predictability Limits as Regime Detectors: A Practical
Rule for How Much History an AI/ML Model Should Train On. Zenodo.
https://doi.org/10.5281/zenodo.21482869
```

**Paper 14 — The Ceiling Holds**
```
RAMANATHAN S, A. (2026). The Ceiling Holds: Testing AI/ML Architecture, Depth,
and Training-Window Size Against an Empirically Measured Predictability Limit.
Zenodo. https://doi.org/10.5281/zenodo.21696948
```

**Paper 15 — The Ceiling Holds, and So Does Climatology**
```
RAMANATHAN S, A. (2026). The Ceiling Holds, and So Does Climatology: Loss
Functions and Calibrated Uncertainty Against an Empirically Measured
Predictability Limit. Zenodo. https://doi.org/10.5281/zenodo.21802729
```

---

## LinkedIn

- [Author Profile](https://www.linkedin.com/in/arun-rs)
- [LinkedIn Posts on CPE Research](https://www.linkedin.com/in/arun-rs/recent-activity/all/)

---

## Repository Structure

```text
.
├── data/                          # Raw and processed price data
├── notebooks/
│   ├── cpe_engine_parallel.py       # Core CPE sweep engine (Papers 1-5)
│   ├── joint_cpe_engine.py          # Multi-predictor joint-conditioning
│   ├── build_gold_dashboard.py      # Gold buy-signal dashboard
│   ├── build_portfolio_dashboard.py # Multi-asset portfolio tilt dashboard
│   ├── build_metals_dashboard.py    # Precious metals dashboard
│   ├── build_predictor_dashboard.py # Paper 12 live price-forecast dashboard
│   ├── ibkr_paper_ledger.py         # IBKR paper-trading ledger
│   ├── temperature/                 # Papers 6-9 climate-finance pipelines
│   ├── rein/                        # Paper 10 hurricane/reinsurer event study
│   ├── predictability_paper/        # Paper 11 multifractal analysis pipeline
│   ├── predictor_v1/                # Paper 12 forecasting pipeline (features, model
│   │                                 # selection, trading strategies, post-processing,
│   │                                 # live-deployment modules) + Paper 13's regime-
│   │                                 # detector scripts (59-64_*.py) + Paper 14's
│   │                                 # architecture/depth/window-sweep scripts
│   │                                 # (65-70_*.py) + Paper 15's loss-function/
│   │                                 # generative-downscaler scripts (73-80_*.py)
│   │                                 # in the same dir
│   ├── predictor_v1_paper_draft.md  # Paper 12 preprint (published: zenodo.org/records/21454884)
│   ├── predictor_v1_paper_draft.pdf # Paper 12 PDF, as submitted to Zenodo
│   ├── regime_detector_paper_draft.md  # Paper 13 preprint (published: zenodo.org/records/21482869)
│   ├── regime_detector_paper_draft.pdf # Paper 13 PDF, as submitted to Zenodo
│   ├── architecture_ceiling_paper_draft.md  # Paper 14 preprint (published: zenodo.org/records/21696948)
│   ├── architecture_ceiling_paper_draft.pdf # Paper 14 PDF, as submitted to Zenodo
│   ├── loss_uncertainty_ceiling_paper_draft.md  # Paper 15 preprint (published: zenodo.org/records/21802729)
│   ├── loss_uncertainty_ceiling_paper_draft.pdf # Paper 15 PDF, as submitted to Zenodo
│   ├── dash_back/                   # Paper 5 dashboard backtest analysis
│   └── *.html                       # Live dashboard outputs (GitHub Pages)
├── src/                            # Core probability estimation and trading logic
├── requirements.txt
└── README.md
```
