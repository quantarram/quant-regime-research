# Climate extension of the economic prior (registered 2026-10-11, before any climate CPE result)

Predictors: ONI, Nino 3.4 (OISST), SOI (enso_state); AO, NAO (polar_oscillation); PDO (pacific_decadal), each in two engine-native forms (`_chg`: change of the state over tau days;
`_mean`: cumulative sum, whose tau-day increment is the sum of the state, i.e. the average state over the past tau days), lagged for publication.
Channels (sub-class level; mechanism and tier in `economic_prior.py`): enso_state -> agriculture (standard), energy, base_metal, broad_commodity, equity_country, em_fx (weak);
polar_oscillation -> energy (standard), agriculture (weak); pacific_decadal -> agriculture (weak).
Run unchanged: pairwise CPE (CPE >= 0.80, lift >= 1.5, n >= 100), the episode-conviction audit (>= 3 separated episodes), chronological split (formation through 2012-12-31, test after) with frozen thresholds.
Disclosure: a simple ENSO-state excess-return scan had been seen before this file was written; channels cover every target of the sub-classes, not only the pairs that looked persistent.


## Batch 2: extreme-event predictors (registered 2026-10-11, before any result)
Predictors (each as `_chg` and `_mean`): CBPDSI (Corn Belt drought), CBTMP (Corn Belt temperature anomaly), HDDGAS (gas-consumption-weighted US heating degree-day anomaly), CDDPOW (power-gas-weighted US cooling degree-day anomaly),
IOD (Dipole Mode Index), ACE (Atlantic accumulated cyclone energy; research only, final data lag months).
Channels: us_drought -> agriculture, us_heat -> agriculture, us_heating_demand -> energy, us_cooling_demand -> energy, atlantic_hurricane -> energy (standard); iod_state -> agriculture, atlantic_hurricane -> agriculture (weak).
Run unchanged (engine gates, episode floor, formation through 2012, frozen-threshold test). Survivor filter fixed in advance: >= 4 formation episodes, >= 3 test episodes, test episode hit rate >= 2/3, test CPE >= base + 0.10, >= 100 test days;
a "watch tier" (>= 3 formation, >= 2 test episodes, same hit and lift criteria) is reported separately. Primary reading is the extreme subset (q_X >= 0.90) since the author's framework is about rare events.


## Batch 3: further slow climate states (registered 2026-10-11, before any result)
Predictors (each as `_chg` and `_mean`): NINO12, NINO3, NINO4 (CPC OISST regional anomalies), MEI2 (multivariate ENSO index v2), AAO (Southern Annular Mode), TNA, TSA (tropical North and South Atlantic SST),
WHWP (Atlantic warm pool), PNA (Pacific-North American pattern). Dropped before any run: AMO (file ends 2023), QBO (ends 2026-02), Indian monsoon rainfall (no free current series), EA and WP (no registered mechanism).
Channels: ENSO regional series and MEI2 use the batch-1 enso_state channels (agriculture standard; energy, base metals, broad commodity, country equity, EM FX weak); southern_annular -> agriculture (weak);
atlantic_sst -> energy (standard), agriculture (weak); pna_state -> energy (standard), agriculture (weak).
Run unchanged: engine gates, episode floor, formation through 2012-12-31, frozen-threshold test, survivor filter (>= 4 formation episodes, >= 3 test episodes, test hit >= 2/3, test CPE >= base + 0.10, >= 100 test days).
Publication lags: Nino regions, MEI2, AAO on the 15th of month m+1; TNA, TSA, WHWP, PNA on the 15th of month m+2. Formation for the OISST-based series starts in 1982.


## Batch 4: rare slow extreme climate states (registered 2026-10-11, before any result)
Predictors (each `_chg` and `_mean`): HRWPDSI, NPLPDSI (wheat-belt drought), DLTPDSI, SEPDSI, TXPDSI (southern drought), WSTPDSI (Western drought), TNI, NPGO (Pacific modes), GLBT (global temperature), SNOW, ICE (cryosphere; anomalies vs 1981-2010 normals).
Dropped before any run: PMM (file ends 2026-02), North Pacific index (ends 2024), IPO, Atlantic Nino, Indian Ocean basin SST (no current file), sea-ice and snow data before their start years. Channels: drought_wheat_belt -> agriculture (standard); drought_south -> agriculture (standard), energy (weak);
drought_west -> energy, agriculture (weak); pacific_modes, global_temperature, cryosphere -> agriculture, energy (weak). Run unchanged (engine gates, episode floor, formation through 2012, frozen-threshold test); survivor filter as in batch 3.
Primary reading: the extreme subset (q_X >= 0.90). Lags: NCEI the 12th of month m+1; TNI the 15th of month m+5 (file ends four months back); NPGO m+2; GLBT, SNOW, ICE m+1.


## Batch W: short-horizon weather predictors, the comparison arm (registered 2026-10-11, before any result)
Predictors (`_mean` form only: the engine's tau-day increment is the sum of the daily anomaly over the past tau days): WXHDDNAT, WXHDDCOLD (heating degree-day anomaly, all and cold-winter metros), WXCDDNAT, WXCDDSOUTH, WXCDDTX, WXCDDMW (cooling).
Daily population-weighted ERA5 metro temperatures, anomaly against formation-period (through 2012) day-of-year normals, one-day publication lag; history ends 2025-12-31.
Grids: tau_past 1, 3, 7, 14, 28 days; tau_future 1, 5, 10, 21 days. Channels deliberately broad (energy, energy single names, sector equity, broad commodity, agriculture): the arm must not be under-tested. Same engine, gates (CPE >= 0.80, lift >= 1.5, n >= 100),
episode floor, formation through 2012-12-31, frozen thresholds, survivor filter, and the same causal walk-forward as the climate batches.
