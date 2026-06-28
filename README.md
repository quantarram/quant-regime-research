# Conditional Exceedance Probabilities for Event-Driven Trading

This repository contains the full code, notebooks, and experiments for the **Conditional Probability of Exceedance (CPE)** framework — a live, production-deployed quantitative signal system covering 161 instruments across equities, fixed income, gold, cryptocurrencies, commodities, volatility, and FX.

**Author:** Arun Ramanathan, PhD (Independent Researcher | Singapore)

---

## Overview

Rather than forecasting prices or returns, CPE asks a fundamentally different question:

> When asset X is in the extreme upper tail of its historical distribution over the past N days, what is the empirical conditional probability that asset Y exceeds its own threshold over the next M days?

This is a nonparametric, empirical conditional frequency — computed directly from historical co-occurrences, with no distribution assumption, no fitted parameters, and no extrapolation beyond the data. **The tail co-movement structure itself is the signal.**

The framework sweeps 51 million candidate configurations across 161 instruments and retains only those where:
- Conditional probability exceeds 0.80
- Lift over the unconditional rate exceeds 1.5×
- At least 100 training-period observations support the estimate

This produces 169,357 pairwise signals. A greedy joint-conditioning procedure then identifies multi-predictor configurations — e.g. "when IBIT and BITB are both simultaneously in their upper tails, gold exceeds its 252-day median return with probability 0.94 vs 0.37 unconditionally" — a 2.54× lift confirmed genuinely calibrated out-of-sample in Paper 4.

**Key empirical results across the four-paper series:**
- **Calibration (Paper 4):** 97.0% realised hit rate against 93.0% stated CPE across 103,983 resolved instances over 528 trading days. The framework is slightly conservative — it understated its own edge by ~4 percentage points.
- **Gold dashboard:** BULLISH throughout all of 2025 (gold: $2,629 → $5,318/oz, +102%). Pivoted BEARISH in early February 2026. Directional discrimination: +7.23% average gold return on BULLISH days vs +2.46% on BEARISH days.
- **Portfolio tilt (Paper 3):** Hold-to-horizon Sharpe 1.224 (pct_exceeding 1.8% vs 1,000 randomisation repetitions). Cross-sectional extension: Sharpe 1.613 across 61 episode-validated targets.
- **Portfolio tilt (Paper 4):** Cumulative +30.6% vs +17.4% neutral equal-weight over 1.4 years. Sharpe 1.43 vs 1.03 (60/40), 0.94 (time-series momentum), 0.71 (risk parity).

---

## Method Summary

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

- **Data source:** Daily adjusted close prices via yfinance
- **Universe:** 161 instruments across 6 asset classes
- **Training period:** Full history through 2024-12-31
- **Evaluation period:** 2025–2026 (strictly out-of-sample)
- **Parameter sweep:** 8 horizons (1–300 days) × 8 quantile thresholds × 2 tail directions × 161² instrument pairs ≈ 51 million configurations
- **Surviving signals:** 169,357 pairwise | 11,106 prior-gated
- No parameters are tuned on test data. All configurations frozen at training cutoff.

---

## Limitations

This repository is intended for research purposes. Honest limitations reported across the paper series:

- **Overlapping t-statistics:** Portfolio significance tests use overlapping weekly observations. Newey-West HAC correction with ~25 lags is required before formal journal submission and would reduce reported t-statistics.
- **Bitcoin ETF concentration:** 103,983 calibration instances are dominated by IBIT, BITB, and FBTC — three near-identical instruments. Effective independent count is substantially smaller than nominal.
- **Bearish signal failure:** Bearish CPE signals for gold achieved only 29% realised hit rate against 83.7% stated CPE. Root cause: UVXY/VIXY structural regime change in 2025, where volatility spikes coincided with gold surges rather than gold weakness as in the training period.
- **Non-stationary predictor thresholds:** Leveraged VIX ETPs (UVXY, VIXY) lose value continuously through roll decay. Their 252-day quantile thresholds become structurally unachievable in live evaluation, causing high-CPE training configurations to never fire out-of-sample.
- **Single regime:** 1.4 years of evaluation covers one sustained gold bull market. Sustained bear markets, credit crises, and deflationary environments have not been tested.
- **Data sufficiency:** The validated vol→equity channel rests on 4 independent training-period episodes. Five observations (including the 2025 OOS result) cannot distinguish genuine predictive content from a well-supported coincidence. Approximately 3–5 additional independent episodes are needed.
- **Transaction costs:** Not modelled. Transaction cost break-even is ~10.1 bps per one-way leg for the 5-sleeve strategy — above realistic ETF costs but sensitive to AUM and operational overhead.

The outputs should be interpreted as evidence of statistical structure and a research prototype that has cleared a first significance threshold — not a deployable trading strategy.

---

## Live Dashboards

Updated daily via automated pipeline. All predictions are publicly timestamped and verifiable.

- [Gold Buy Signal Dashboard](https://quantarram.github.io/quant-regime-research/notebooks/gold_dashboard.html)
- [Multi-Asset Portfolio Tilt Dashboard](https://quantarram.github.io/quant-regime-research/notebooks/portfolio_dashboard.html)
- [CPE Atlas Explorer (169K signals)](https://quantarram.github.io/quant-regime-research/notebooks/cpe_dashboard.html)

---

## Research Papers

- [Paper 7: Agricultural Crop-Zone Temperatures in the CPE Framework: Heat Stress Thresholds, Growing Degree Days, and the Reversal of the Paper 6 Sugar Signal](https://zenodo.org/records/20993837) | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7012738)
- [Paper 6: Beyond Tail Co-Movement: How Temperature Extremes Shift Financial Return Distributions](https://zenodo.org/records/20964819) | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7008418)
- [Paper 5: Corrected Inference for the CPE Portfolio Tilt Strategy: Newey-West HAC Standard Errors and Robustness Checks](https://zenodo.org/records/20908417) | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7000559)
- [Paper 4: Signal-Level Calibration and Dashboard Utility of the CPE Framework](https://zenodo.org/records/20830462) | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6990498)
- [Paper 3: From Descriptive Atlas to Tradeable Signal](https://zenodo.org/records/20815386) | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6980979)
- [Paper 2: A Conditional Exceedance Framework for Interpretable Trading Decisions](https://zenodo.org/records/20769150) | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6969158)
- [Paper 1: A Descriptive Atlas of Conditional Exceedance Structure Across a Multi-Asset Universe](https://zenodo.org/records/20606184) | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6931838)

---

## Substack Articles

- [Paper 7: From the City to the Field: Solving the Geographic Mismatch in Climate-Finance](https://arunramanathans.substack.com/p/from-the-city-to-the-field-solving)
- [Paper 6: Temperature Extremes Don't Co-Move With Financial Tails — They Shift the Whole Distribution](https://arunramanathans.substack.com/p/paper-6-temperature-extremes-dont)
- [Building a Live Quant Signal System: What Worked, What Failed, and What the Data Says](https://arunramanathans.substack.com/p/building-a-live-quant-signal-system)
- [Paper 5: The Results Survive — HAC Correction, Block Bootstrap, and Four Robustness Checks](https://arunramanathans.substack.com/p/paper-5-the-results-survive-hac-correction)
- [Paper 4: The Conditional Probabilities Actually Hold](https://arunramanathans.substack.com/p/paper-4-the-conditional-probabilities)
- [Got One Statistically Significant Result. Then Spent Hours Trying to Break It.](https://arunramanathans.substack.com/p/from-descriptive-atlas-to-tradeable)
- [A Research Update: When a Strong Statistical Pattern Doesn't Become a Trading Edge](https://arunramanathans.substack.com/p/a-research-update-when-a-strong-statistical)
- [The Gold Dashboard Just Said BUY GRADUALLY. Here Is Exactly Why.](https://arunramanathans.substack.com/p/the-gold-dashboard-just-said-buy)
- [From One Asset to Five: Scaling the CPE Framework Into a Portfolio Tilt Dashboard](https://arunramanathans.substack.com/p/from-one-asset-to-five-scaling-the)
- [What If You Could Be Useful Without Predicting Anything?](https://arunramanathans.substack.com/p/what-if-you-could-be-useful-without)
- [I Built a Dashboard to Tell Me When to Buy Gold](https://arunramanathans.substack.com/p/i-built-a-dashboard-to-tell-me-when)
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
RAMANATHAN S, A. (2026). A Simple Conditional Exceedance Framework for
Interpretable Trading Decisions. Zenodo.
https://doi.org/10.5281/zenodo.18382687
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
---

## LinkedIn

- [Author Profile](https://www.linkedin.com/in/arun-rs)
- [LinkedIn Posts on CPE Research](https://www.linkedin.com/in/arun-rs/recent-activity/all/)

---

## Repository Structure

```text
.
├── data/                  # Raw and processed price data
├── notebooks/             # Exploratory and analysis notebooks
├── src/                   # Core probability estimation and trading logic
├── requirements.txt
└── README.md
```
