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

## Repository Structure

```text
.
├── data/                  # Raw and processed price data
├── notebooks/             # Exploratory and analysis notebooks
├── src/                    # Core probability estimation and trading logic
├── requirements.txt
└── README.md

## Reproducibility

Data source: Daily adjusted close prices of SPY
Training period: Inception – 2024-12-31
Evaluation period: 2025 (out-of-sample)
No parameters are tuned on test data

All results in the paper can be reproduced using the code in this repository.

## Limitations

This repository is intended for research and educational purposes only.
Important limitations include:
Transaction costs and market impact are not modeled
Position sizing is simplified
Results are shown for a single asset

The outputs should be interpreted as evidence of statistical structure, not as deployable trading strategies.

## Citation

If you find this work useful, please cite the accompanying paper:

Ramanathan, A. (2026).
A Simple Conditional Exceedance Framework for Interpretable Trading Decisions.
arXiv preprint.
