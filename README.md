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
- **Portfolio tilt (Paper 3):** Hold-to-horizon Sharpe 1.224 (pct_exceeding 1.8% vs 1,000 randomisation repetitions). Cross-sectional extension: Sharpe 1.613 across 61 episode-validated targets.
- **Portfolio tilt (Paper 4):** Cumulative +30.6% vs +17.4% neutral equal-weight over 1.4 years. Sharpe 1.43 vs 1.03 (60/40), 0.94 (time-series momentum), 0.71 (risk parity).

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

---

## Limitations

This repository is intended for research purposes. Honest limitations reported across the paper series:

**Core CPE framework (Papers 1–5):**
- **Overlapping t-statistics:** Portfolio significance tests use overlapping weekly observations. Newey-West HAC correction with ~25 lags is required before formal journal submission and would reduce reported t-statistics.
- **Bitcoin ETF concentration:** 103,983 calibration instances are dominated by IBIT, BITB, and FBTC — three near-identical instruments. Effective independent count is substantially smaller than nominal.
- **Bearish signal failure:** Bearish CPE signals for gold achieved only 29% realised hit rate against 83.7% stated CPE. Root cause: UVXY/VIXY structural regime change in 2025, where volatility spikes coincided with gold surges rather than gold weakness as in the training period.
- **Non-stationary predictor thresholds:** Leveraged VIX ETPs (UVXY, VIXY) lose value continuously through roll decay. Their 252-day quantile thresholds become structurally unachievable in live evaluation, causing high-CPE training configurations to never fire out-of-sample.
- **Single regime:** 1.4 years of evaluation covers one sustained gold bull market. Sustained bear markets, credit crises, and deflationary environments have not been tested.
- **Data sufficiency:** The validated vol→equity channel rests on 4 independent training-period episodes. Five observations (including the 2025 OOS result) cannot distinguish genuine predictive content from a well-supported coincidence. Approximately 3–5 additional independent episodes are needed.
- **Transaction costs:** Not modelled. Transaction cost break-even is ~10.1 bps per one-way leg for the 5-sleeve strategy — above realistic ETF costs but sensitive to AUM and operational overhead.

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

---

## Research Papers

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
│   │                                 # detector scripts (59-64_*.py) in the same dir
│   ├── predictor_v1_paper_draft.md  # Paper 12 preprint (published: zenodo.org/records/21454884)
│   ├── predictor_v1_paper_draft.pdf # Paper 12 PDF, as submitted to Zenodo
│   ├── regime_detector_paper_draft.md  # Paper 13 preprint (published: zenodo.org/records/21482869)
│   ├── regime_detector_paper_draft.pdf # Paper 13 PDF, as submitted to Zenodo
│   ├── dash_back/                   # Paper 5 dashboard backtest analysis
│   └── *.html                       # Live dashboard outputs (GitHub Pages)
├── src/                            # Core probability estimation and trading logic
├── requirements.txt
└── README.md
```
