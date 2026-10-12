# CPE on irradiance extremes and JEPX prices (2026-10-12)

Script `jepx_cpe_scan.py`, table `jepx_cpe_scan.csv`, figure `jepx_cpe_figure.png`. The CPE definition and the program's gates, applied directly (not the full engine with its economic-prior registry).

Design, frozen before looking at JEPX. X: mean daily ERA5 irradiance z-score over the past p = 7, 14, 30, 60 days (z against the calendar-month mean and sd of 1996-2015, so the extreme thresholds come from before JEPX exists); both tails at 2%, 5%, 10%; Tokyo irradiance for Tokyo and the system price, Fukuoka for Kyushu. Y: next f = 7, 14, 30 days, (a) log change of the mean price against the previous f days, (b) mean premium over the system price; upper and lower 10% tails frozen on the formation window 2016-04..2020-12; test 2021-01..2026-09. Base rate matched to the same calendar months. 720 conditions.

Results
- Gate (CPE >= 0.80, lift >= 1.5, n >= 100 days): 0 of 720 pass on formation. 0 survivors.
- Looser, lift-only admission on formation (lift >= 1.5, n >= 100, >= 3 episodes): 73 conditions. In the test window their mean lift is 1.09 (median 1.09); 15% keep lift >= 1.5; 55% are above base.
- Average excess over the season-matched base in the test window (points): mechanism direction (dull -> price up, bright -> price down) -1.3 over 198 conditions; opposite direction -0.6. Dull states in the mechanism direction average -4.7.
- Only 7 conditions have lift >= 1.5 in both windows. Four run against the mechanism (dull -> Kyushu price down, bright -> Tokyo/system price up), and the others are lift 1.5-2 on 14-30 day windows with base rates of 4-15%.
- The one faint mechanism-consistent pocket is the Kyushu premium over the system price after bright spells (test excess +4.6 points over 30 conditions), but it is -1.7 on formation, the test base rate itself is about 0.5 because the Kyushu discount deepened after 2020, and it is not confirmed.

The 20 August 2026 state: over the past 30 and 60 days Tokyo irradiance stood at the 99th percentile of the 1996-2015 distribution, and Kyushu at the 100th (30-day state +0.88 sd against a bright-10% cut of +0.34). That is a bright extreme and in the mechanism direction implies a lower September price. The matching conditions (Tokyo, bright 10%, p = 30, f = 30) had formation lift 2.0 (CPE 0.25 against base 0.13) and test lift 0.93 (0.04 against 0.05). The outcome was the opposite: the dullest September since 1996 and a price of 23.8 against 20.1 in August.
Earlier note: the 80th percentile quoted in the first reply was for 1-19 August only.

Limits: ten years of prices; the windows overlap and conditions are highly correlated, so 720 rows are far fewer independent findings; no seasonal irradiance forecast; ERA5 grid cells.
