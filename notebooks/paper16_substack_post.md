# Predictability-Informed Momentum: How I Made a 30-Year-Old Trading Strategy Better

My last three papers measured a real, hard ceiling on how far ahead a financial instrument can genuinely be predicted — and nothing, not bigger models, more depth, more training data, could buy past it. So this time I asked a different question: forget predicting better. Can you *decide* better with what you already know?

Here's what actually worked. Time-Series Momentum (TSMOM) is a real, 30-year-old published strategy — I implemented it exactly as written, no tuning, no fitting to my own data. Applied to a plain 12-instrument universe, it's a real but unremarkable performer. But restrict it to only the instruments it can actually predict well — using a real, measured predictability limit for each one, from an earlier paper in this series — and it gets meaningfully better: 3 of 5 real historical periods beaten instead of 2, full-sample Sharpe up from 0.29 to 0.32.

![Predictability-filtered TSMOM vs. the unfiltered baseline, and where else a real signal held up](paper16_graphical_abstract.png)

The idea is simple once you see it: TSMOM's signal is built on a 252-day lookback window, and that window only means something if the instrument's own price actually carries real, measurable structure at that horizon. Some do. Some don't. Applying the same strategy everywhere treats every stock the same; filtering first by what each one can actually support doesn't.

Getting the filter right required one real check along the way: TSMOM's edge is concentrated specifically in real market crises, not spread evenly across calendar time, so I made sure to validate the filtered version against real historical periods that actually include those crises, not arbitrary date ranges — otherwise a genuine improvement can get lost in the noise of how you happened to slice the data.

There's a smaller bonus finding too: two individual stocks, JPM and XLB, show a real edge from this program's own signals that survives every way I checked it, including recomputing the numbers the more careful way — four other candidates that looked good under the industry's usual method for computing alpha didn't hold up once I did.

The wall on prediction hasn't moved, and a sixth attempt at buying past it with smarter decision-making instead of better forecasting didn't move it either. But being more selective about *where* you apply a real, published strategy, based on what you can actually measure about each instrument, is a real, narrow, and genuinely useful edge.

Full preprint: https://zenodo.org/records/21842311
Code, all instruments, every figure: https://github.com/quantarram/quant-regime-research

*Independent quantitative research. Not investment advice.*
