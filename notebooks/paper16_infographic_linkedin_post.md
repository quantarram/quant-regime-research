Restricting Time-Series Momentum (TSMOM) — a real, 30-year-old, published trend-following strategy, implemented exactly as written — to only the instruments it can actually predict well beats the standard version. That's Paper 16's central result: full-sample Sharpe up from 0.29 to 0.32.

The obvious question with any backtested improvement is whether it survives being tested a different way. So I re-ran the comparison at 10 different block counts instead of the paper's original 5, slicing the same 20-year history anywhere from 3 pieces to 20. Predictability-Informed TSMOM's average annual return beat standard TSMOM's at every single one — the size of the edge barely moved, even though the paper's own "3 of 5" framing turns out to be sensitive to that specific choice of 5.

Put together a short infographic walking through how the filter actually works (same TSMOM engine, a smaller pre-screened universe) and showing that robustness check directly:

![Predictability-Informed TSMOM infographic: mechanism schematic and the 10-block robustness check](paper16_pitsmom_infographic.png)

Interactive version: https://quantarram.github.io/quant-regime-research/notebooks/paper16_pitsmom_infographic.html

Full preprint: https://zenodo.org/records/21842311
Code: https://github.com/quantarram/quant-regime-research

Independent quantitative research. Not investment advice.
