# I Built a 22-Instrument Price Forecaster. It Has Real Skill — and Zero Alpha.

Most applied forecasting work reports a skill metric — an R², a hit rate, "beats the baseline" — and stops there, treating that as equivalent to "this is worth something." Paper 12 in this series is what happened when I refused to stop there: I built a real, live price-forecasting system, gave it every reasonable chance to prove it was tradeable, and then reported exactly what happened, including the two mistakes I made along the way and the bug I found four hours after publishing the dashboard.

## The system

Twenty-two instruments — broad-market and sector ETFs, three single-name equities, gold, one FX pair — each forecast with LightGBM quantile regression, conditioned on each instrument's own multifractal price dynamics (the same structure-function machinery from Paper 11) interacted with two forward-looking market regime signals: a credit-spread regime (HYG/LQD) and a VIX-term-structure regime (VIXM/VIXY), both independently causally validated before I let them anywhere near a model. Climatology — a frozen, day-of-year empirical forecast — competes as a genuine fourth candidate the whole way through, not a strawman baseline.

A "master model" picks, per instrument, whichever candidate has the lowest holdout-period price error on data it never touched during selection. The result: **climatology wins for 12 of 22 instruments.** Half the panel's best available forecast is "check what usually happens around this time of year." I used to instinctively read that as a null result. It isn't — it's a genuine finding about where forward-looking regime information does and doesn't add value, and burying it in favor of a more impressive-sounding "our model wins everywhere" headline would have been dishonest.

## The bug that inflated a result 12x

An earlier version of the selection procedure picked the best (horizon, model) combination by scanning the *entire* out-of-sample period, then reported that same period's performance for the winner. Classic data snooping. XLE's headline result — the single strongest number in an earlier pass of this work — dropped from +0.164 to +0.014 once I actually held out a genuine, never-touched test period. A 12x inflation, caught before it went into a paper, not after.

## The bug that gave gold fake alpha

The first pass of the alpha test benchmarked every instrument against SPY as a universal market proxy. Gold came back significant: t = 2.33, +16.3%/year. Except — gold is structurally almost uncorrelated with equities (β to SPY = 0.08), and over that window gold just happened to rally independently of stocks. I checked: *simply buying and holding gold, with zero model,* also showed "significant alpha vs. SPY" (+18.6%/year, t = 2.34). The test was measuring gold's decoupling from the equity market, not anything my model did. Re-benchmarked against gold's own buy-and-hold — the actually correct comparison for a single-instrument timing strategy — the alpha collapses to +2.1%/year, t = 0.61. Not significant.

## Five ways to try to make money, five null results

Once the benchmark was fixed, I ran five structurally different trading-strategy designs against the properly-specified benchmark: a directional sign rule, a price-target buy-low/sell-high strategy, portfolio construction across all 22 instruments, Kelly-criterion position sizing, and a market-neutral cross-sectional relative-value long/short. **Zero of 22 instruments show statistically significant, risk-adjusted alpha, under any of the five.** The Kelly-sized portfolio is the cleanest illustration of why: raw return goes up (+92.9% vs. the unlevered basket's +64.3%) purely from leverage, while Sharpe goes *down* (0.72 vs. 0.82) and drawdown gets *worse* (−33.6% vs. −21.3%). That's what added risk looks like when you mistake it for added skill.

## The narrow place where something real does exist

I then asked a different question: independent of tradeability, can the forecast's raw *accuracy* be improved after the fact? Five more structurally distinct post-processing designs later — a static recency-fit correction, a collinearity-prone stacked blend, per-candidate correction, a genuinely rolling/adaptive correction, a bi-weekly correction — **18 of 22 instruments were never improved by any of them.** But gold and JPMorgan converge across four and three of the five designs respectively, using techniques that share no common construction. That kind of agreement, from methods that don't agree by chance, is the strongest signal in the whole post-processing arc. Deployed live, the rolling correction takes gold from 20.2% to 12.0% MAPE and JPM from 13.4% to 9.6%, mature-phase, holdout-honest.

## It's live — and it doesn't tell you to buy anything

The whole thing runs as a public, real-time dashboard now: predicted price, quantile band, and an honest backtest MAPE per instrument. No buy/sell signal, no verdict score, nothing that could read as a trading recommendation — because Section 5 above found no instrument with demonstrated tradeable alpha, and building a dashboard that implied otherwise would contradict the paper's own central finding. A mechanical check greps every build for forbidden trading-verdict language before it ever gets published.

One more thing, caught after the dashboard went live: each of the five quantile levels is an independently-fit model, and nothing forces them to stay ordered relative to each other. For three long-horizon instruments, the median forecast briefly came out *below* the model's own 10th-percentile forecast — an internally impossible result invisible to every backtest number in the paper, since those only ever score the median. Fixed with a standard technique (monotone rearrangement — sort the five predictions, reassign them to the quantile levels in order) and written up as its own section rather than quietly patched.

## What the data says

Real, holdout-honest statistical skill exists for about half of a 22-instrument panel. Subjected to five independent trading-strategy tests, that skill translates into demonstrated economic value for effectively none of them. Subjected to five independent post-processing attempts, a real correctable bias exists for exactly two. That's the paper's actual finding: not "AI predicts stocks," but a precisely measured gap between statistical skill and economic value, reported the same way whether the news was good or bad.

Dashboard: https://quantarram.github.io/quant-regime-research/notebooks/predictor_dashboard.html
Full preprint (Zenodo): [add link once uploaded]
Code and data: https://github.com/quantarram/quant-regime-research

*Independent quantitative research. Not investment advice.*
