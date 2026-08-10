Most backtested improvements don't survive being tested a different way. This one did — ten different times.

Time-Series Momentum (TSMOM) is one of the most established, widely-run trend-following strategies at systematic funds — a real, 30-year-old published strategy, implemented here exactly as written, no tuning, no fitting to my own data. I restricted it to only the instruments it can actually predict well, using nothing but a separate, already-published measurement of each instrument's own predictability. Same signal, same sizing rules, same rebalancing — just a smaller, pre-screened universe. It beat the standard version: full-sample Sharpe up from 0.29 to 0.32.

Then I stress-tested it properly. Instead of relying on the paper's original 5-block comparison, I re-ran it at 10 different block counts, slicing the same 20-year history anywhere from 3 pieces to 20. Predictability-Informed TSMOM's average annual return beat standard TSMOM's at every single one — 10 of 10 — and the size of the edge barely moved. Most edges narrow or flip sign under that kind of test. This one didn't.

![Predictability-Informed TSMOM infographic: mechanism schematic and the 10-block robustness check](paper16_pitsmom_infographic.png)

Interactive version: https://quantarram.github.io/quant-regime-research/notebooks/paper16_pitsmom_infographic.html

Full preprint: https://zenodo.org/records/21842311
Code: https://github.com/quantarram/quant-regime-research

Independent quantitative research. Not investment advice.
