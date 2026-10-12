# Causal September forecast test, Tokyo-area JEPX (2026-10-12)

Question (A. Stace, comment on the Paper 17 post): standing on 20 August 2026, what would the model have said about September in the Tokyo area?
Design (fixed before results): for each year 2017-2026, forecast the September mean of all half-hourly Tokyo-area day-ahead prices on 20 August, using only data through 19 August and earlier years. JEPX data start in April 2016, so 2017 is the first year with a prior ratio.
M0 no change (1-19 August mean). M1 seasonal ratio (M0 x exp of the mean log September/August ratio of earlier years). M2 = M1 + irradiance channel (sensitivity of the daily Tokyo-minus-system premium to the daily ERA5 irradiance anomaly, estimated on days before 20 August of that year, times the expected September anomaly from a persistence coefficient fitted on 1996..Y-1). M3 oracle: as M2 with the realised September anomaly. M4 last September. M5 mean of earlier Septembers.
Irradiance units: anomaly = mean of daily z-scores (daily sd); September 2026 = -0.66 in these units = -2.6 in monthly-sd units.

Errors 2017-2026 (yen/kWh): MAE M0 2.30, M1 2.04, M2 2.04, M3 2.05, M4 5.75, M5 4.54; MAPE M1 15%. Excluding 2022: M0 1.96, M1 2.23, M2 2.23, M3 2.23.
Irradiance channel: persistence from 1-19 August to September is tiny (coefficients 0.03-0.17; corr of expected and realised September anomaly +0.03); estimated sensitivity about -0.08 yen per daily sd. Even with the realised September anomaly (M3), the forecast moves by at most about 0.1 yen.
20 August 2026: 1-19 August mean 20.06; M1 forecast 18.05; M2 18.05 (expected anomaly +0.07 sd from a +0.43 sd August); actual September mean 23.78 (M1 error -5.73, -24%); oracle M3 18.10. Premium over system: actual +4.66, forecast +1.94, oracle +1.99.
The 2026 error is within the historical spread of the baseline (M1 errors range from -30% in 2023 to +28% in 2019).
Limits: ten forecast years; naive baselines (no forward curve, fuel, outage or reserve-margin information that a professional forecaster would use); ERA5 grid cell, not the station; linear sensitivity cannot capture a nonlinear effect of extreme dullness, and only two record-dull Septembers exist (2016, 2026; 2016 is before the first forecast year).
