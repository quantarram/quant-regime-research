# Slow climate state (ENSO) and commodity and equity returns at seasonal horizons: study plan (written before any result is computed)

**Why.** Day-ahead weather rules found nothing (Paper 17). The one weather-to-price link that already holds in this program's own work is slow and mechanistic: El Nino -> Indian monsoon -> sugar
(Sugar paper). This study asks the same question of every commodity and weather-exposed equity in the universe, at 3-12 month horizons, with a publication lag and a chronological split.
**State.** NOAA CPC Oceanic Nino Index (monthly, 1950-). The ONI of the season centred on month m is public about ten days after the end of month m+1; the state used at the first trading day
of month t is the latest ONI whose availability date is on or before it. El Nino: ONI >= +0.5; La Nina: <= -0.5; neutral otherwise; strong: |ONI| >= 1.0.
**Targets.** Every instrument in the price universe with at least 1,500 daily observations before 2011. Forward log return from the close of the first trading day of each month over 63, 126, 189 and
252 trading days. Overlapping windows are valid; episodes (contiguous runs of a state) are counted so that overlap is not mistaken for independence.
**Statistic.** Excess forward return in a state = mean over state months minus mean over all months of the same period. Formation 1996-2010, test 2011-2025. A (state, instrument, horizon) is
*admitted* in formation if it spans at least 3 episodes, the mean excess exceeds 5 percentage points in absolute value (the same sign in at least two thirds of its episodes). The test reports the sign
agreement and mean test excess of the admitted set, and for the pre-specified mechanistic list: sugar (SB=F, CANE), coffee (KC=F), cocoa (CC=F), cotton (CT=F), orange juice (OJ=F, JO), wheat (ZW=F, WEAT),
soybeans (ZS=F, SOYB), corn (ZC=F, CORN), soybean meal and oil (ZM=F, ZL=F), natural gas (NG=F), heating oil (HO=F), copper (HG=F), cattle and hogs (LE=F, HE=F).
Point estimates with n; no significance language.
**Limits stated up front.** About nine El Nino and seven La Nina episodes since 1996; continuous front-month futures carry roll effects over multi-month horizons, so equities and ETFs are shown next to them.
