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
- (https://arunramanathans.substack.com/p/paper-4-the-conditional-probabilities?r=6v2jrz)
- (https://arunramanathans.substack.com/p/from-descriptive-atlas-to-tradeable?r=6v2jrz)
- (https://arunramanathans.substack.com/p/a-research-update-when-a-strong-statistical?r=6v2jrz)
- (https://arunramanathans.substack.com/p/the-gold-dashboard-just-said-buy?r=6v2jrz)
- (https://arunramanathans.substack.com/p/from-one-asset-to-five-scaling-the?r=6v2jrz)
- (https://arunramanathans.substack.com/p/what-if-you-could-be-useful-without?r=6v2jrz)
- (https://arunramanathans.substack.com/p/i-built-a-dashboard-to-tell-me-when?r=6v2jrz)
- (https://arunramanathans.substack.com/p/mapping-the-hidden-wiring-of-global?r=6v2jrz)
- (https://arunramanathans.substack.com/p/a-simple-conditional-exceedance-framework?r=6v2jrz)
- (https://arunramanathans.substack.com/p/conditional-exceedance-probabilities?r=6v2jrz)

---

## LinkedIn posts
- https://www.linkedin.com/posts/arun-rs_github-quantarramquant-regime-research-activity-7475550682129813505-ypnr?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs
- https://www.linkedin.com/posts/arun-rs_github-quantarramquant-regime-research-activity-7475170857099677698-UIGd?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs
- https://www.linkedin.com/posts/arun-rs_a-pattern-worth-naming-in-quant-content-activity-7474077312653684736-21ZN?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs
- https://www.linkedin.com/posts/arun-rs_why-cpe-framework-looks-at-every-horizon-activity-7474052142320865282-WU-M?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs
- https://www.linkedin.com/posts/arun-rs_github-quantarramquant-regime-research-activity-7473976810528112640-eCQQ?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs
- https://www.linkedin.com/posts/arun-rs_the-signal-just-flipped-more-about-it-activity-7473313589815070720-QMWm?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs
- https://www.linkedin.com/posts/arun-rs_portfolio-tilt-dashboard-activity-7472951553549594624-ztSU?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs
- https://www.linkedin.com/posts/arun-rs_conditional-probability-of-exceedance-cpe-activity-7472626950503636992-oT3A?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs
- https://www.linkedin.com/posts/arun-rs_the-cpe-atlas-preprint-is-now-live-on-ssrn-activity-7471532889587728384-Eunm?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs
- https://www.linkedin.com/posts/arun-rs_quantfinance-research-empiricalfinance-activity-7471527162312011776-3uT0?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs
- https://www.linkedin.com/posts/arun-rs_i-built-a-dashboard-to-tell-me-when-to-buy-activity-7470784431092506624-Sjjr?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs
- https://www.linkedin.com/posts/arun-rs_mapping-the-hidden-wiring-of-global-financial-activity-7470054976476835840-hDj_?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs
- https://www.linkedin.com/posts/arun-rs_github-quantarramquant-regime-research-activity-7425120844235821056-vkoK?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs
- https://www.linkedin.com/posts/arun-rs_a-simple-conditional-exceedance-framework-activity-7421784315249889281-4T8C?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs
- https://www.linkedin.com/posts/arun-rs_over-the-past-few-months-i-worked-on-a-personal-activity-7417732007989940225-sS7h?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs
- https://www.linkedin.com/posts/arun-rs_quantresearch-systematictrading-finance-activity-7417562878351953922-4suQ?utm_source=share&utm_medium=member_desktop&rcm=ACoAACd2YkMB6RBzKCyadbbmOsyEfq_SKhAIJXs

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

---
