Most forecasting write-ups stop at "the model beat the baseline." I wanted to know if it actually made money — and the honest answer, after testing it five different ways, was no.

I just finished Paper 12 in my ongoing conditional-exceedance / multifractal research series: a real-time price forecasting system for 22 instruments (equities, sector ETFs, gold, FX), using LightGBM quantile regression conditioned on credit-spread and VIX-term-structure regime signals, benchmarked honestly against day-of-year climatology — which, for 12 of the 22 instruments, is still the best available forecast. That's not a failure. It's the finding.

The system has real, holdout-validated statistical skill for roughly half the panel. But run through five structurally different trading-strategy designs — a sign rule, price-target entries, portfolio construction, Kelly-criterion sizing, cross-sectional relative value — zero of 22 instruments show statistically significant, properly risk-adjusted alpha. Along the way I caught and fixed two of my own mistakes worth naming, not hiding: a data-snooping bug that had inflated one instrument's headline result ~12x, and a benchmark-misspecification bug that made gold look like it had real alpha when it was actually just decoupling from equities.

Where the forecasts ARE improvable, it's narrow and specific: gold and JPMorgan are the only two instruments, of 22, where five independent post-processing designs agree a real, correctable forecast bias exists.

It's live now: a public, real-time dashboard, forecast-only, no buy/sell signals — because I couldn't demonstrate that it should have any.

Dashboard: https://quantarram.github.io/quant-regime-research/notebooks/predictor_dashboard.html
Full preprint (Zenodo): [add link once uploaded]
Code: https://github.com/quantarram/quant-regime-research

Independent quantitative research. Not investment advice.
