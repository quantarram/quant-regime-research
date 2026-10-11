# Climate predictors through the CPE engine (2026-10-11)

Engine: `notebooks/files/cpe_engine_parallel.py` logic unchanged (pairwise CPE, gates CPE >= 0.80, lift >= 1.5, n >= 100; economic-prior gate; tau grid 21-300), episode conviction from `files/backtest_engine.py`
(episodes: firing dates clustered with gap > 1.5 x tau_past x 1.45 calendar days; one outcome per episode at its last date; floor of 3 episodes). Climate prior registered in `economic_prior.py` / `CLIMATE_PRIOR.md` before any result.
Split: formation through 2012-12-31 (frozen thresholds), test 2013-01-01 to 2026-10-10.

| | rows passing gates | >= 3 formation episodes | with >= 100 test days | claimed CPE | test CPE | test base rate | mean test lift | share of rows with lift > 1 |
|---|---|---|---|---|---|---|---|---|
| Climate predictors (12 series, 6 indices x 2 forms) | 10,352 | 2,461 | 1,877 | 0.870 | 0.458 | 0.464 | 0.92 | 48% |
| Program's own predictors, identical split and prior (calibration) | 18,424 | 4,751 | 2,400 | 0.860 | 0.486 | 0.430 | 1.24 | 70% |

Climate by target class (test CPE vs base; share lift > 1): agriculture 0.355 vs 0.379 (38%); energy 0.541 vs 0.532 (48%); base metals 0.506 vs 0.453 (83%); equity-country 0.688 vs 0.632 (71%); em_fx 0.458 vs 0.548 (30%).
Rows with >= 2 test episodes (1,271): mean per-episode test hit rate 0.479 vs claimed 0.870.
Survivors (filter fixed beforehand: >= 4 formation episodes, >= 3 test episodes, test episode hit rate >= 2/3, test CPE >= test base + 0.10, >= 100 test days): 43 of 859 rows with >= 4 formation episodes;
41 bearish (Nino 3.4 / ONI in its lower tail): coffee 11, Korea equities 8, copper 6, soybeans 5, Brazil 4, China 4, INR 2, ZAR 2, gasoline 1; horizons 126-300 days.
Coffee: Nino 3.4 21-day average in its lowest quartile -> coffee 252/300-day forward return in its lower tail: formation CPE 0.85-0.94 (5-6 episodes), test 0.70-0.78 vs base 0.52-0.61 (3-6 episodes, hit 0.67-1.0).
Firing now (2026-10-10, strong El Nino): 2 survivor rows (ONI change -> ZAR, weak tier); no survivor for sugar; the survivors are La Nina-side conditions and are not firing.

## Observed extremes through the CPE engine (2026-10-11)

Predictors (registered in `economic_prior.py` / `CLIMATE_PRIOR.md` batch 2 before any result): corn-belt Palmer drought index and temperature anomaly (NCEI nClimDiv, state-weighted), gas-weighted heating and power-weighted cooling degree-days, Indian Ocean Dipole, accumulated cyclone energy (HURDAT2; research-only, data lag), each in `_chg` and `_mean` form. Same engine, gates, split and filter as above. Code: `cpe_extreme_data.py`, `cpe_extreme_run.py`, `cpe_extreme_eval.py`, `cpe_extreme_live.py`.

| | rows | claimed CPE | test CPE | test base | mean lift | share lift > 1 |
|---|---|---|---|---|---|---|
| Passing gates in formation | 1,228 (899 at q_X >= 0.90) | | | | | |
| >= 3 formation episodes, >= 100 test days | 343 | 0.869 | 0.325 | 0.384 | 0.86 | 37% |
| of which q_X >= 0.90 | 287 | 0.871 | 0.290 | 0.355 | | |
| Agriculture / energy | 296 / 47 | 0.871 / 0.857 | 0.300 / 0.482 | 0.363 / 0.510 | | 37% / 36% |

Episode hit rate: formation 0.74, test 0.41. By predictor (test CPE vs base): CBPDSI_chg 0.172 vs 0.485 (reversal), CBPDSI_mean 0.349 vs 0.470, CBTMP 0.30-0.32 vs 0.32, IOD_chg hit rate 0.86 -> 0.18, IOD_mean 0.306 vs 0.317; HDDGAS_chg 0.564 vs 0.489 and ACE_mean 0.546 vs 0.465 are the only predictors above base.
Calibration (program's own predictors, same split/prior): 169 survivors of 2,400 rows, lift 1.24. Extreme set: 3 survivors of 343.

Survivors (filter unchanged: >= 4 formation episodes, >= 3 test episodes, test hit >= 2/3, test CPE >= base + 0.10, >= 100 test days):
- HDDGAS_chg (tau_past 126, q 0.90) bullish -> NG=F, 63d: formation CPE 0.80 (4 episodes), test 0.70 vs 0.49 (6 episodes, hit 0.67). Not firing (October).
- HDDGAS_chg (tau_past 21, q 0.95) bullish -> RB=F, 63d: formation 0.80 (8 episodes), test 0.67 vs 0.39 (7 episodes, hit 0.71). Not firing.
- ACE_mean (tau_past 126, q 0.80) bearish -> RB=F, 126d: formation 0.86 (8 episodes), test 0.79 vs 0.65 (5 episodes, hit 0.80). Firing now, but ACE has no live feed (best-track released after season), so research-only.

Mechanism of the collapse: persistent states (drought, temperature, Dipole) deliver their formation evidence from a few multi-year episodes; 87% of the deepest-5% corn-belt drought days sit in formation (1963-64, 1976-77, 1988-89, 2012) and the test holds only 2022-24. The ocean-index set (Nino 3.4/ONI) is the one that had 43 survivors, with a mechanism through institutions and crop supply.
Live ledger: not extended. Three survivors from 1,228 rows, two of them 63-day (not slow) with 4-8 formation episodes, and one with no live data path, do not justify NCEI loaders in the daily workflow. HDDGAS->NG is kept as a watch item: revisit when the 2026-27 heating season gives a live onset.

## Batch 3: nine more slow climate states (2026-10-11)

Predictors (registered with channels in `economic_prior.py` / `CLIMATE_PRIOR.md` before the run): NINO12, NINO3, NINO4, MEI2, AAO, TNA, TSA, WHWP, PNA, each `_chg` and `_mean`. Dropped before any run: AMO (data ends 2023), QBO (2026-02), Indian monsoon rainfall (no current free series), EA/WP (no mechanism).
Code: `cpe_slow_data.py`, `cpe_slow_run.py`, `cpe_slow_eval.py`, `cpe_slow_live.py`, `cpe_slow_episodes.py`, `build_cpe_slow_table.py`. Same engine, gates, split (formation through 2012; OISST series start 1982) and survivor filter.

| | rows | claimed | test | base | lift | share > 1 |
|---|---|---|---|---|---|---|
| Pass gates in formation | 19,738 | | | | | |
| >= 3 formation episodes, >= 100 test days | 4,259 | 0.876 | 0.498 | 0.471 | 1.02 | 55% |

By class (test vs base): equity_country 0.634 vs 0.557 (72% above); energy 0.544 vs 0.502; base_metal 0.511 vs 0.462; agriculture 0.439 vs 0.426 (49%); em_fx 0.478 vs 0.487. By predictor: NINO12_mean 0.611 vs 0.530, TNA_mean 0.840 vs 0.673 (67 rows), MEI2 0.27-0.32 vs 0.32-0.39 (below), TNA_chg 0.290 vs 0.326, PNA 0.39-0.49 vs 0.48-0.53.
Survivors (>= 4 formation episodes, >= 3 test, hit >= 2/3, test >= base + 0.10, >= 100 test days): 181 of 1,932 rows (9.4%; batch 1 was 5.0%); watch tier 526. Survival rate by family: ENSO regions and MEI 151/1,511 (10.0%), TNA 16/93 (17.2%), AAO 3/33, TSA 3/40, PNA 6/164 (3.7%), WHWP 2/91 (2.2%). 142 of 151 ENSO-family survivors are lower-tail (bearish) conditions; horizons 252 d (102 rows), 300 d (59), 126 d (15), 63 d (5). 66 survivors at q_X >= 0.90.
Redundancy: the ENSO-family survivors reduce to 38 target-direction-horizon keys, 13 shared with batch 1; the non-ENSO families give 30 rows and 23 keys (2 shared). The full table has 53 distinct keys.
Episodes behind the main bundles: Nino 3 -> Korea/China equities test episodes 2013-04, 2013-10, 2021-06, 2022-08, 2023-02 (two ENSO phases); TNA -> wheat, soybean meal test episodes 2014-2019 (the commodity bust; base rates 0.71-0.74); Nino 1+2 -> rand has both tails (bearish 8 episodes 2014-2024 with 2 misses; bullish 5 episodes, all hits).
Frozen table: `cpe_climate_table_b3.parquet`, 181 rows, sha fac17e01ce, frozen 2026-10-11 (data cutoff 2026-10-10). Live now: 8 conditions / 9 rows, all warm-Pacific upper tails -> rand (and DBB) at 63-300 days.
Live ledger: dashboard repo `weatherdash/cpe_climate.py`, second book `b3`; ledger index parsers verified identical to the engine panel (all nine series and all 18 daily columns).

## Batch 4, the horizon ladder and the causal walk-forward (2026-10-11)

Batch 4 (registered first; `cpe_x4_*.py`): HRW/northern-plains/Delta/Southeast/Texas/Western PDSI, TNI, NPGO, GLBT, SNOW, ICE. 8,347 rows pass gates; 1,464 with >= 3 formation episodes; test 0.412 vs base 0.392 (lift 1.06, 60% above 1); survivors 38 of 567 (6.7%), 11 predictors x 13 targets, none from the HRW wheat belt (rows 0.28 vs 0.30/0.40). Frozen as `cpe_climate_table_b4.parquet` (sha 55a05316db); one condition fires (TXPDSI_mean tau 21 q 0.95 -> HO=F bearish 126 d).
Weather arm in the engine (`cpe_wx_*.py`): only 5 rows pass CPE >= 0.80 at 1-28 d windows / 1-21 d horizons; the level gate prejudges the horizon, so it is not used for the comparison.
Horizon ladder (`cpe_ladder.py`, 110 instruments, 6 weather + 31 climate series, formation-only admission: >= 100 days, >= 4 episodes, lift >= 1.5; test 2013-2026): 10% tails, all targets climate 1.137 (median 0.971, n 33,423) vs weather 0.933 (0.835, n 12,978); registered-channel targets climate 1.097 (0.976) vs weather 1.033 (0.947); no-channel targets 1.158 vs 0.875. 20% tails: all 1.052 vs 0.924; registered 1.056 (0.978) vs 1.100 (1.057). Climate median lift by forward horizon (registered, 10%): 1.10 (1d), 1.03 (21d), 0.93 (63d), 0.83 (126d), 0.68 (252d). Same-variable degree-day windows <= 14 d vs 63-252 d: 1.07 vs 0.97 (10%), 1.11 vs 1.11 (20%).
Causal walk-forward (`cpe_walkforward.py`; admission each 1 Jan 2016-2026 from resolved outcomes): climate admitted 534 events hit 0.558 vs base 0.551; not admitted 12,854: 0.445 vs 0.448; extreme tail (q_X >= 0.9) 188 events 0.606 vs 0.524 (+8.2), 168 distinct trades, excess return +2.95% (sd 16.4%; ratio 2.5 across entries, 0.9 across 11 entry years), 121 of 188 entries in 2020-2023; moderate tail 346: 0.532 vs 0.565 (-3.3). By batch (extreme): b1 -8.0 (11), b2 +9.9 (14), b3 +7.3 (82), b4 +11.0 (81). Standard tier -1.0 (38), weak tier +10.5 (150). Market control: extreme 0.694 vs 0.546 (+14.8, 111), all admitted 0.678 vs 0.601.
Decision rule (`cpe_wf_portfolio.py`, trading-day basis, 10 bp): climate all admitted -2.37%/yr (Sharpe -0.16; same targets long -3.52%), extreme tail -1.37% (Sharpe -0.08; long +0.70%), moderate -2.40%; market control +14.18% (0.76) vs +14.55%, extreme +4.71% vs +10.88%. Hit-rate edge does not become a return; bearish extreme entries lost 5.6% vs unconditional shorts -8.5% (mean raw return per trade -0.1%); 164 of 188 entries are q_Y = 0.5 direction bets.
Outputs: `ladder_10.parquet`, `ladder_20.parquet`, `walkforward_climate.parquet`, `walkforward_market.parquet`, `wf_export.json`; paper figures `paper17_new_figures.py`.

## Rarity gradient (2026-10-11, `cpe_rarity2.py`, `cpe_climate/rarity2.parquet`)
Question (the user's actual claim): does persistence rise as the conditioning event gets rarer, in any domain? One admission rule (formation only: >= 100 days, >= 4 episodes, lift >= 1.5), registered-channel pairs, test 2013-2026; arms weather / climate / market (each instrument's own log-price move, windows 1-252 d); tails 20/10/5/2%.
Both directions stored: lift on the admitted side, lift on the opposite side, and net directional lift = (p_adm - p_opp) / mean base (0 = no directional information; opposite side also above 1 = volatility co-movement, not direction).
Matched event tails, mean net directional lift (opposite-side lift): market 0.35 (1.31) at 20%, 0.66 (1.74) at 10%, 0.69 (1.73) at 5%; climate 0.16 (0.98), 0.03 (1.10), 0.07 (1.15) at 20/10/5%; weather 0.29 (0.83), 0.28 (0.85), 0.17 (1.00). Medians: market net 0.35 / 0.27 / 0.00; climate 0.06 / 0 / 0; weather 0.28 / 0.15 / 0.
Reading: in the market domain (where the CPE framework was built) most raw lift (1.5-2.1) is volatility co-movement, but a directional part remains and its mean rises from 20% to 10-5% tails (median does not); day-to-day market windows (1-7 d) are not noise there (median lift 1.34-1.65). Weather and climate show no rarity gradient. Weather net directional 0.28 at 20% tails is probably seasonal co-timing (anomaly tails and instrument tails share seasons), not tested here.

## Rarity-axis tests behind the rebuilt Paper 17 (2026-10-12)
Causal walk-forward by predictor tail inside the engine's admission (walkforward_*.parquet): climate admitted hit-minus-base by q_X bucket 0.5-0.75 / 0.8 / 0.9 / 0.95: -3.7 (n 282), -1.6 (64), +4.1 (92), +12.1 (96), none at 0.99; excess return -0.1/-0.8/+0.9/+5.0%. Market: +4.9 (259), +7.2 (145), +13.1 (101), +32.2 (10). Not-admitted entries: no gradient.
Gate-free tests (same lift-based rule for weather, climate, market; `cpe_wf2.py`, wf2s.parquet; `cpe_persist.py`, persist.parquet): hit minus same-month base: weather +0.1/+0.3/-0.3 (20/10/5% tails), climate -0.5/-1.3/-0.8/-0.1 (20/10/5/2%), market +3.5/-1.0/-1.2; vs whole-year base weather +2.4/+4.6/+3.4 (seasonality). Slopes of test edge on formation edge (matched events, season-matched bases): weather 0.121/0.083/0.041, climate 0.040/0.014/0.001/-0.003, market 0.084/0.107/0.026. Episodes behind a configuration (median): weather 27-35, market 20-41, climate 5-12.
Verdict: frequent weather events are noise (gate-free, season-matched); a rarity gradient exists inside the CPE gate + causal confirmation but not under a gate-free rule; market frequent-tail persistence is mostly volatility co-movement. Paper 17 rebuilt on this axis (30 pages).
