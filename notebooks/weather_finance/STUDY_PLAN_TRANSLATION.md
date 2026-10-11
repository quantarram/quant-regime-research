# From weather forecasts to fundamentals to positions: the natural-gas translation engine (written before any result is computed)

**Why (user, 2026-10-11).** The earlier decision tests turned a weather quantity (degree-days, demand anomaly) into a position with almost no structure in between. A
trading desk's value, plausibly, lies in the conversion: weather -> physical balance (demand, power burn, storage) -> a view on price relative to what is already priced
-> a sized position. This plan builds that chain for natural gas, the market where weather matters most, and scores every link.

## Links, each scored on its own before any trading
**L1 Power burn.** EIA-930 hourly generation by fuel for all balancing authorities (2018-07 to 2025-12): daily gas-fired generation from net load (demand - wind - solar),
nuclear and hydro availability (their own recent levels), day of week, season, and the gas price. Scored by out-of-sample (expanding window, test 2021-2025) error of daily
gas-fired generation, with and without the wind and solar information and against climatology.
**L2 Residential/commercial/industrial demand.** Heating degree-days weighted by actual gas consumption (EIA state residential + commercial consumption allocated to the
34 metros by population within state) instead of population; power-burn cooling weights from state gas-for-power consumption. Scored on the weekly storage balance.
**L3 Storage balance.** Weekly Lower-48 storage change from consumption-weighted degree-days and the weather-driven power burn, plus a trend; expanding-window refit; scored by
out-of-sample RMSE of the weekly change and of its departure from the five-year norm, against (a) the five-year norm, (b) the earlier population-weighted degree-day model
(Section 6.5), and (c) climatology-fed weather.
**L4 Decision.** Expected injection for the coming report from the new chain versus from the simple chain (the better a market consensus is approximated by simple
degree-days, the more the gap between the two predicts the surprise). Entry at Wednesday's close, held to Thursday's close and for five days; also the multi-week
projection revision. Costs 5 bp per side; baselines: flat, buy-and-hold on the same days, the simple-chain rule, the climatology-fed rule. Formation 2019-2022, test 2023-2025,
test run once. NG=F continuous-contract roll days excluded as a robustness run.
## Reported
Each link's accuracy gain (with n); the decision table; no significance language, no shuffling.
