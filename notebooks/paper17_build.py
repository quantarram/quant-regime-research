"""Builds weather_impacts_paper_draft.md (Paper 17) from the verified body text of the earlier draft (renumbered) plus the new framing and climate sections.
Run (from notebooks/): python3 paper17_build.py   -- then pandoc/weasyprint as in paper17_build.sh"""
import re, os, sys
HERE=os.path.dirname(os.path.abspath(__file__))
OLD=open(os.environ.get('P17_OLD',os.path.join(HERE,'paper17_old_draft.md'))).read().split('\n')
def block(a,b): return '\n'.join(OLD[a-1:b-1]).strip('\n')       # 1-indexed [a, b)
def find(h):
    for i,l in enumerate(OLD,1):
        if l.startswith(h): return i
    raise KeyError(h)
L={k:find(k) for k in ('## 2. Data','## 6. Further','### 6.1 ','### 6.9','## 7. Physical','## 8. A registered','## 9. A record','## 10. Limits','## 11. Repro','## References')}
body=block(L['## 2. Data'],L['### 6.9'])                           # old Sections 2 to 6.8
sec7=block(L['## 7. Physical'],L['## 8. A registered']); sec8=block(L['## 8. A registered'],L['## 9. A record'])
lim_old=block(L['## 10. Limits'],L['## 11. Repro']); refs_old=block(L['## References'],len(OLD)+2)
p69=block(L['### 6.9'],L['## 7. Physical']).split('\n\n')
generic_enso=p69[1]; policy_par=[p for p in p69 if p.startswith('The mechanism, where there is one')][0]
sugar_sentence=re.search(r'The sugar paper.s own design was also run in the engine:.*?not on a generic ENSO signal\.',p69[2],re.S).group(0)
# ---- renumber the retained body: headings and cross references (descending to avoid collisions)
def renum(t):
    t=re.sub(r'^## (\d+)\. ',lambda m:f'## {int(m.group(1))+1}. ',t,flags=re.M)
    t=re.sub(r'^### 6\.(\d) ',lambda m:f'### 7.{m.group(1)} ',t,flags=re.M)
    t=re.sub(r'Section 6\.(\d)',lambda m:f'Section 7.{m.group(1)}',t)
    for old in (5,4,3):
        t=re.sub(rf'Section {old}\b(?!\.\d)',f'Section {old+1}',t)
    return t
body=renum(body)
# ---- front matter
TITLE='# Weather Is Secondary: Free Forecasts Give No Financial Edge, and the Rare, Extreme, Slow Climate Tail Is a Candidate for Live Testing'
ABSTRACT="""Free weather forecasts are public, cheap and often said to be worth money. This paper tests whether they are, and whether the slow climate states that lie beyond the forecast horizon are a better place to look. A free daily pipeline turns the ECMWF open forecast into electricity demand, wind and solar output, gas burn, storage and crop stress. Physically it beats climatology for about a week, and weather moves power prices strongly (ten extra cooling degree-days raise next-day peak prices about 60% at PJM West). Financially it does not: pre-specified gas rules, 12,824 weather-tail configurations over 97 instruments, about 630,000 walk-forward rules, machine-learning models and a gas-balance chain earned no more than market exposure.

Slow climate states are the natural alternative, and the evidence for them is narrower than the intuition. Four batches of 32 slow states through a conditional-exceedance engine claim a probability of 0.87 in formation and score a mean lift of 0.86 to 1.06 in 2013-2026. Under one admission rule applied to weather and climate predictors alike (110 instruments, 10% tails), the two are indistinguishable on targets with a registered economic channel (mean out-of-sample lift 1.03 and 1.10); the apparent climate lead sits in Treasuries and credit, where no mechanism exists. A causal walk-forward, admitting rows each January from resolved outcomes only, finds persistence in the rare extreme tail (60.6% against a 52.4% base over 188 entries; moderate tail 53.2% against 56.5%), clustered in 2019-2023, and a decision rule built on it returned -1.4% a year. Weather is secondary; the extreme, slow tail is a candidate that only forward scoring can settle, and two registered experiments, one a dashboard that converts frozen climate conditions into implied positions, began scoring on 12 October 2026."""
PLS="""Weather forecasts are free, and professional traders are often said to profit from them. This paper asks whether free forecasts can make money, and whether slow climate patterns such as El Niño are a better place to look. We built an automatic dashboard that turns the forecast into electricity demand, wind and solar output, gas burned in power plants, gas storage and crop stress, and checks itself against what happened. Physically it works for about a week, and a hot day can raise a power price by more than half.

Financially, short-horizon weather did not pay. A winter gas rule that looked profitable in 2021-2023 made nothing in 2024-2026. Searching 12,824 pairs of weather conditions and prices, and then about 630,000 trading rules, the ones that looked good early were no better than chance later. Even the weather that drives most of the weekly change in gas storage did not predict gas prices, because the market had already priced it.

Slow climate patterns were the obvious next place to look, and we tested them with the same care. Over four batches of 32 patterns, most apparent links faded after the first period. When weather and climate were put through one identical test, they were no different on the markets where a physical link exists. The one signal that persisted was in the rarest, most extreme conditions: when rows were chosen each January using only outcomes already known, they landed in the extreme tail about 61% of the time against 52% by chance. That edge was small, concentrated in a few years, and a trading rule built on it lost about 1.4% a year.

So the conclusion is bounded. Weather is secondary to money and flow data; the rare, extreme, slow tail is a candidate worth following, not a result. We built a dashboard that turns the frozen climate conditions into paper positions and records what they earn from 12 October 2026 onward, so the question will be answered by the future and not by this paper."""
GA_CAPTION="*Left, how many days ahead each forecast beats 11- to 30-year climatology (demand 8, wind 6, solar 9). Middle, the 614 weather-and-price pairs (of 12,824 tested) that looked predictive in 1996-2012: their probability of a big price move after a weather tail was 34.9% in formation and 15.5% in the 2013-2025 test, against a 15.9% base rate. Right, slow climate rows admitted causally each January from resolved outcomes: the share of entries landing in the extreme tail against its base rate (extreme +8.2 points, moderate -3.3). Built by `paper17_new_figures.py`.*"
front=f"""{TITLE}

By Arun Ramanathan

CPE research series, Paper 17

---

## Graphical Abstract

![](paper17_graphical_abstract.png)

{GA_CAPTION}

---

## Abstract

{ABSTRACT}

---

## Plain Language Summary

{PLS}

---
"""
INTRO="""## 1. Introduction

Weather forecasts are the cheapest and most public forecast input there is, and it is widely believed that they are worth money: that power, gas and crop prices follow the forecast, and that professional desks profit by reading it better than others. This paper tests that belief with free data, and then the more careful version of it, that the information, if any, lies not in the day-to-day forecast but in slow, rare climate states.

The first part builds a daily, automated, self-scoring pipeline from the ECMWF open forecast. It turns the forecast into regional electricity demand, wind and solar generation, gas burned in power plants, regional gas storage and crop stress, and records for each quantity how many days ahead the forecast still beats simply assuming the climatological average. That number is what a user needs, and it is rarely published. The answer is about eight days for demand, six for wind and nine for solar.

The second part asks whether any of it reaches prices. Natural gas is widely described as a weather market, and degree-day forecast revisions are the standard narrative for its day-to-day moves. A pre-specified test on five years of archived forecasts finds nothing a rule could trade, and a diagnostic shows the reason is not forecast quality. Further tests, on regional degree-days, on the weekly USDA crop-condition release, on 12,824 weather-tail configurations across 97 instruments, on a walk-forward search of about 630,000 decision rules, on machine-learning models and on a translation chain from weather to gas burn to storage, find the same thing: markets already price what public forecasts say.

The third part asks the question that result raises. If weather forecasts carry no unpriced information, where could any remain? Beyond about two weeks, forecast skill is generally attributed to slowly varying components of the climate system, and these components, El Nino above all, change the supply of crops and the demand for energy over months (Section 2). Four batches of 32 slow climate states were therefore run through the author's conditional-exceedance (CPE) engine; weather and climate predictors were placed under one admission rule and compared; the surviving conditions were tested causally, admitting rows each January from outcomes already resolved; and the conditions that survive were converted into a registered decision experiment with a live dashboard.

The scoring discipline follows this series' conventions. Every forecast is written to a ledger when it is issued and never revised; every score is reported next to a baseline with its sample size; an oracle (the same model fed observed weather) separates weather error from model error; formation and test periods are chronological and thresholds are frozen at formation; and no result is reported as significant or not. Where a ratio of an effect to its standard error is shown, it sizes the effect and does not test it. Section 2 sets out the horizons, Section 3 the data and models, Section 4 the verification of demand and renewables, Section 5 the crop layer, Section 6 the natural-gas study, Section 7 the further tests of the weather-to-price link, Section 8 the slow climate states, Section 9 the dashboard and the two registered experiments, Section 10 a record of the noise that was chased, Section 11 the limits and Section 12 code and data availability."""
SEC2="""## 2. Weather, sub-seasonal and climate: the horizons

Weather forecasting is limited by the atmosphere's own dynamics. Lorenz (1969) argued that errors in a flow with many scales of motion grow in a way that bounds deterministic forecasts, and recent estimates put the practical limit of skilful midlatitude forecasts at about 10 days, extendable by perhaps five more with a ten-fold reduction in initial-condition error and a nearly perfect model (Zhang et al., 2019). The verification in Section 4 finds the same order of magnitude: demand, wind and solar forecasts beat a climatological baseline for six to nine days.

Beyond that range, forecast skill is generally attributed to slowly varying components of the climate system and not to the initial state of the atmosphere. The WMO and WCRP sub-seasonal-to-seasonal (S2S) project covers lead times from two weeks to two months (Vitart et al., 2017), and seasonal prediction extends from there. This paper therefore uses three bands. *Weather* means lead times up to about two weeks, where information comes from the initial state of the atmosphere. *Sub-seasonal and seasonal* means two weeks to a year, where it comes from components such as tropical sea-surface temperature and soil moisture. *Climate states* are those components themselves: El Nino and the Southern Oscillation, the Indian Ocean Dipole, Atlantic and Pacific sea-surface temperature, drought indices. They are slow (their values persist for months), rare in their extremes (a strong El Nino is a handful of episodes in four decades) and reported monthly.

The predictors follow the bands. Weather predictors in this paper are daily degree-day anomalies and forecast-based quantities; climate predictors are monthly state series with their publication lags. Forward price horizons of 1 to 21 trading days belong to the weather and sub-seasonal bands and 63 to 300 trading days (three to fourteen months) to the seasonal and climate bands. The boundaries are conventions and not cuts in the physics, so every result below is reported by horizon and the reader can place their own."""
exec(open(os.path.join(HERE,'paper17_build_part2.py')).read())
