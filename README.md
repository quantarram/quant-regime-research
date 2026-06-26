# Conditional Exceedance Probabilities for Event-Driven Trading

This repository contains the full code and experiments accompanying the paper:

**“A Simple Conditional Exceedance Framework for Interpretable Trading Decisions”**  
Arun Ramanathan (Independent Researcher)

The project demonstrates how transparent conditional probability tables, estimated from historical data, can be translated directly into systematic, event-driven buy–hold–sell trading decisions.

---

## Overview

Rather than forecasting prices or returns, this framework asks a simpler question:

> Given that the market has already experienced a large move over a fixed time horizon, how likely is another large move over the same horizon?

Using daily SPY data, the repository implements:
- Conditional exceedance probability estimation
- Sample-size–based reliability filtering
- Fixed-horizon, event-driven trading rules
- Strict out-of-sample evaluation on 2025 data

The emphasis is on **interpretability, transparency, and statistical decision-making**, not model complexity.

---

## Method Summary

For a fixed horizon \( \tau \):

1. Compute \( \tau \)-day price increments
2. Define exceedance events using high-percentile thresholds
3. Estimate conditional probabilities of future exceedances given past exceedances
4. Filter probability estimates using data-support criteria
5. Translate qualifying probabilities into mechanical buy–hold–sell trades

All probability tables are computed using pre-2025 data and remain frozen during evaluation.

---

## Reproducibility

- Data source: Daily adjusted close prices of SPY
- Training period: Inception – 2024-12-31
- Evaluation period: 2025 (out-of-sample)
- No parameters are tuned on test data

All results in the paper can be reproduced using the code in this repository.

---

## Limitations

This repository is intended for research and educational purposes only.

Important limitations include:
- Transaction costs and market impact are not modeled
- Position sizing is simplified
- Results are shown for a single asset

The outputs should be interpreted as evidence of statistical structure, not as deployable trading strategies.

---

## Citation

If you find this work useful, please cite the accompanying paper:

RAMANATHAN S, A. (2026). A SIMPLE CONDITIONAL EXCEEDANCE FRAMEWORK FOR INTERPRETABLE TRADING DECISIONS. Zenodo. https://doi.org/10.5281/zenodo.18382687

---

## Research Papers
- Paper 4: Signal-Level Calibration and Dashboard Utility of the Conditional Probability Exceedance Framework: Pairwise Validation, Gold Dashboard Evaluation, and Extended Portfolio Tilt Evidence Across 528 Trading Days (https://zenodo.org/records/20830462) (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6990498)
- Paper 3: From Descriptive Atlas to Tradeable Signal: An Out-of-Sample Test of the Multi-Asset Conditional Exceedance Framework as a Portfolio Tilt Strategy (https://zenodo.org/records/20815386) (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6980979)
- Paper 2: A CONDITIONAL EXCEEDANCE FRAMEWORK FOR INTERPRETABLE TRADING DECISIONS (https://zenodo.org/records/20769150) (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6969158)
- Paper 1: A Descriptive Atlas of Conditional Exceedance Structure Across a Multi-Asset Universe (https://zenodo.org/records/20606184) (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6931838)

---

## Substack Articles
- [Paper 4: The Conditional Probabilities Actually Hold](https://arunramanathans.substack.com/p/paper-4-the-conditional-probabilities?r=6v2jrz)
- [Got One Statistically Significant Result. Then Spent Hours Trying to Break It.](https://arunramanathans.substack.com/p/from-descriptive-atlas-to-tradeable?r=6v2jrz)
- [A Research Update: When a Strong Statistical Pattern Doesn't Become a Trading Edge](https://arunramanathans.substack.com/p/a-research-update-when-a-strong-statistical?r=6v2jrz)
- [The Gold Dashboard Just Said BUY GRADUALLY. Here Is Exactly Why.](https://arunramanathans.substack.com/p/the-gold-dashboard-just-said-buy?r=6v2jrz)
- [From One Asset to Five: Scaling the CPE Framework Into a Portfolio Tilt Dashboard](https://arunramanathans.substack.com/p/from-one-asset-to-five-scaling-the?r=6v2jrz)
- [What If You Could Be Useful Without Predicting Anything?](https://arunramanathans.substack.com/p/what-if-you-could-be-useful-without?r=6v2jrz)
- [I Built a Dashboard to Tell Me When to Buy Gold — Here's What It's Saying Right Now](https://arunramanathans.substack.com/p/i-built-a-dashboard-to-tell-me-when?r=6v2jrz)
- [Mapping the Hidden Wiring of Global Financial Markets](https://arunramanathans.substack.com/p/mapping-the-hidden-wiring-of-global?r=6v2jrz)
- [A SIMPLE CONDITIONAL EXCEEDANCE FRAMEWORK FOR INTERPRETABLE TRADING DECISIONS](https://arunramanathans.substack.com/p/a-simple-conditional-exceedance-framework?r=6v2jrz)
- [Conditional Exceedance Probabilities as a Basis for Systematic Trading](https://arunramanathans.substack.com/p/conditional-exceedance-probabilities?r=6v2jrz)

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
├── src/                    # Core probability estimation and trading logic
├── requirements.txt
└── README.md
```
---
