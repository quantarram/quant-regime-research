# Natural gas vs GFS degree-day forecasts: results (protocol in STUDY_PLAN.md)

Archive: GFS 00z, 34 metros, 2021-03-23 to 2026-10-10 (2,027 cycles; 2022-11-30 missing: the source file is corrupt).
Normals: ERA5 1996-2025 national population-weighted HDD/CDD. Formation 2021-03..2023-12; test 2024-01..2026-10, run once
after the definitions were fixed. Point estimates; n = signal days (trading days in season). NG=F front-month, entry at settle.

| signal | season | period | n | corr next-day | P(top-20% return given signal top 20%) | rule mean per active day | buy&hold same days |
|---|---|---|---|---|---|---|---|
| revision | winter | formation | 233 | +0.10 | 34% (47 days) | +85 bp | -52 bp |
| revision | winter | **test** | 249 | +0.05 | 16% (70 days) | -2 bp | -18 bp |
| revision | summer | formation | 255 | +0.09 | 16% | +10 bp | +22 bp |
| revision | summer | **test** | 255 | -0.03 | 12% | -10 bp | -3 bp |
| anomaly | winter | formation | 235 | +0.04 | 17% | +56 bp | -55 bp |
| anomaly | winter | **test** | 249 | +0.02 | 31% (51 days) | +15 bp | -18 bp |
| anomaly | summer | formation | 255 | -0.09 | 16% | -7 bp | +22 bp |
| anomaly | summer | **test** | 255 | -0.11 | 8% | -28 bp | -3 bp |

**Reading.** The winter revision signal that looked good in formation (+85 bp per active day, 34% tail hit rate) did not hold in 2024-2026
(-2 bp, 16%). Its formation result was concentrated: top five days = 58% of profit, one winter = most of the rest, and the rule was net short
in a falling market. No signal shows a consistent relationship with next-day or same-day returns (|corr| <= 0.12 everywhere), so there is
no evidence the market reacted to a single public model's degree-day change before settle either; the daily signal simply carries little.
The one test-period number above base rate (winter anomaly top tail, 31%) is not supported by the correlation or the rule's variability.

**Not tested / why this is not the end:** gas-weighted (not population-weighted) degree-days; 11-15 day outlooks; ECMWF alongside GFS
(consensus changes); intraday entry before settle; storage-surprise conditioning. Agricultural futures need their own studies.
