# Weather and power prices, and the dollar value of forecast skill: study plan (written before any result is computed)

ERCOT's own price archive is behind bot protection and its API needs an account, so ERCOT is not tested here. EIA publishes ICE next-day peak power prices
for PJM West, Indiana Hub, Mass Hub (NEPOOL), Palo Verde, SP15, NP15 and Mid-C (2017-2025 files used).
**Price model.** For each hub and delivery weekday: log(volume-weighted price) on regional HDD and CDD of the delivery day (ERA5 daily mean
temperature, population-weighted over the hub's region), log of the front-month natural gas price, and month fixed effects. Reported: percent price
change per +10 degree-days, and out-of-sample (expanding window, refit each year, test years 2020-2025) R-squared of the model with weather versus the
same model without it.
**Dollar value of forecast skill.** Using the price sensitivity and the dashboard's measured temperature-forecast error by lead (ledger: mean
absolute error of the IFS and of climatology), the implied expected absolute error in the next-day peak price attributable to weather uncertainty, as a
percentage and in dollars per MWh at the hub's mean price, at each lead from 0 to 10 days, forecast versus climatology. This is the value to a
buyer or seller of power; it is not a trading result.
