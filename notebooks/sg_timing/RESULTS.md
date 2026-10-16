# Singapore buy-timing: first results (2026-10-16)

## Data
LTA COE bidding results from data.gov.sg (open licence, no key): 397 exercises, January 2010 to the first exercise of October 2026, five categories. Bids received is missing for 109 of 1,985 rows, mostly 2024-2026. Latest premiums: Category A S$130,001 (high S$133,009 on 1 September 2026), Category B S$130,100 (high S$150,001 in October 2023).
Quota is announced a quarter ahead and, since February 2023, equals the rolling average of deregistrations over the previous four quarters (it was two quarters from August 2022, with a one-time volatility adjustment in 2023): the market knows it well in advance.

## First CPE scan (`coe_cpe_scan.py`)
Four registered mechanisms (quota supply, demand ratio, value reversion, momentum), tau_p and tau_f of 2, 6, 12 exercises, tails of 20% and 10%, Category A and B: 288 conditions. Formation to 2017-12, test 2018 onward, frozen thresholds.
- 22 of 288 pass the (adapted) admission on formation: lift >= 1.5, >= 20 exercises, >= 5 episodes. Scored on the test window their mean lift is 1.09, 9% keep a lift >= 1.5 and half are above their base rate. No momentum row was admitted; value-reversion rows average 0.94.
- Reading: simple conditions on past premium, quota and bid pressure do not beat the base rate for the next one to six months. Base rates: the premium rose in 56% of exercises; over six exercises (about three months) it rose more than 10% in 29% of windows and fell more than 10% in 16%, an upward drift over a series that went from S$18,500 to S$130,000.
- Caveat: the quota formula and the Category A/B rules changed in 2022-2023, so formation thresholds from 2010-2017 describe a different regime.
