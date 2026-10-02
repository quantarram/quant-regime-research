# NVIDIA's Earnings Reports Don't Do What You Think They Do

There's a version of NVIDIA's story everyone already knows: it reports, it beats, the stock pops, repeat. I wanted to actually check this against the real record rather than the narrative, so I pulled every one of NVIDIA's quarterly earnings reports — 71 of them, going back to 2006 — and looked at what the stock did afterward, measured against what the market and its own semiconductor peers were doing at the same time.

The real answer: there's no reliable pop.

Looking at NVIDIA's return in the day, week, and month after each report, relative to the S&P 500, AMD, and the semiconductor sector (SOXX), the stock beats those benchmarks on only 44% to 55% of occasions depending on the window — close to a coin flip in every direction I checked. Even on the reaction day itself, NVIDIA underperforms the market more often than not. The "it always beats, it always pops" story isn't what twenty years of actual earnings reports show.

![NVIDIA's real excess return per earnings event, vs. market and sector peers, 2006-2026](nvda_earnings_study/nvda_earnings_excess_return_plot.png)

The more interesting thread was the one I expected to confirm easily and couldn't: that misses get punished hard. NVIDIA has only missed its own EPS estimate 4 times in 71 quarters, and the "miss = crash" pattern only shows up cleanly in 2 of those 4 (November 2018 and August 2022, both real, sharp, sustained selloffs). The other two misses — including a -17% EPS miss in November 2022 — were followed by flat-to-positive returns within a month. A four-event average was doing a lot of work there, carried by two extreme cases rather than a reliable pattern.

The part that actually surprised me: even through the 2023–2026 AI earnings supercycle — the stretch where NVIDIA's results have been genuinely extraordinary, each quarter outgrowing the last — its one-month return around its own earnings, *relative to its own sector peers* (AMD, SOXX), hasn't improved. If anything it's gotten a bit worse (-3.65% vs. AMD, -2.25% vs. SOXX in the AI era, compared with -1.20% and -0.29% before 2023). Only against the broad market does the AI-era number look better (+1.67% vs. +0.41% pre-2023), and even that split is close to even, not a clean run of wins.

![NVIDIA's 1-month earnings-reaction excess return vs. SPY, pre-2023 vs. the AI era, real points](nvda_earnings_study/nvda_earnings_regime_split_plot.png)

Put together, this says something specific: NVIDIA's enormous run over the past few years hasn't come from the market rewarding its earnings days more than its peers' earnings days. It's come from a sustained trend that plays out *between* earnings reports, not from a repeatable, tradeable earnings-reaction edge. The "AI premium," to whatever extent it's real, isn't concentrated in the four days a year everyone's watching the earnings print.

One more check, on a different question entirely: does the AI theme itself create more tail-level co-movement across the stocks everyone associates with it? I took a real, dated basket — the 52 US-listed holdings of Global X's AI-themed ETF (AIQ) exactly as they stood in December 2022, pulled from the fund's own archived holdings page rather than picked with hindsight — and compared genuine tail-event co-movement (the same conditional-exceedance test this whole research program runs) in the two equal-length windows immediately before and after ChatGPT's launch. The answer: no. Real tail co-movement actually *fell* across the basket post-launch, and NVIDIA, Microsoft, Meta, Amazon, and Apple each show zero surviving tail-level predictability from anything else in the basket in the post-ChatGPT window. Alphabet is the one exception, co-moving with Alibaba's own tail moves specifically — an odd, narrow pairing I'm reporting rather than explaining away.

None of this is an argument that AI stocks aren't a real story — they obviously are, at the level of the overall price trend. It's an argument that the specific, tradeable patterns people assume come with that story (reliable earnings pops, stronger cross-stock tail co-movement) don't show up when you actually go and check.

Code, all the raw data, and every figure: https://github.com/quantarram/quant-regime-research

*Independent quantitative research. Not investment advice.*
