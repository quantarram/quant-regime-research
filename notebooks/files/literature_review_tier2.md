# Literature Review of Tier-2 "Debatable" Channels

## How to read this document

Each channel is rated on two separate axes, because they answer different
questions:

- **Literature support**: does published research document this
  relationship existing at all, anywhere, in any sample?
- **Horizon/instrument match**: does that literature actually support the
  SPECIFIC claim embedded in the channel — i.e. one ETF's daily price
  action over a 1-300 trading-day window predicting another ETF's price
  action over a similar window? Most financial literature studies
  macro-horizon relationships (months to years, often using
  year-over-year or quarterly aggregates) or relationships between an
  index and a raw economic variable (GDP, PMI), not ETF-to-ETF price
  prediction at trading-day granularity.

A channel can have strong literature support and still be a poor fit for
this screen if the horizon or instrument type doesn't match. That
mismatch turned out to be the single most common problem found in this
review — more common than channels simply being wrong.

**Recommendation key:**
- KEEP — literature support holds at a horizon/instrument resolution
  reasonably close to what this screen tests
- KEEP WITH CAVEAT — real relationship, but literature also documents
  specific failure regimes; flag this in the dashboard/strategy logic,
  don't silently trust the configuration
- DOWNGRADE — literature support exists but is for a different horizon,
  different instrument type, or is itself contested; treat any signal
  from this channel as lower-conviction than a Tier-1 signal
- CUT — literature support is weak, contested, or doesn't transfer to
  this instrument set at all

---

## 1. `credit_ig -> broad_equity_us` and `credit_ig -> credit_hy`

**Claim:** Investment-grade credit spread/ETF moves lead broad equity
weakness and lead high-yield spread moves.

**Literature found:** Real and substantial — Gilchrist & Zakrajsek's
excess bond premium (Fed working paper, widely cited), Fama & French
(1993) on credit-spread-related return predictability, Keim & Stambaugh
(1986). These are credible, well-established results.

**Problem:** Every one of these studies operates at macro/business-cycle
horizon (output growth, investment growth, multi-month-ahead equity
returns), using AGGREGATE credit spread series (e.g. BAA-AAA, or
constructed excess bond premium indices), not the price action of a
single ETF (LQD) over a 1-300 trading-day grid. The mechanism
(information flows from credit markets to equity markets because credit
investors are more sensitive to downside) is real, but whether it shows
up cleanly in LQD's own daily price relative to SPY's, at horizons as
short as 21 trading days, is a different and untested question.

**Recommendation: KEEP WITH CAVEAT** for the IG->HY relationship (more
mechanical: investment-grade contagion into high-yield during a genuine
credit event is documented and fast-moving). **DOWNGRADE** the
IG->equity legs specifically — the academic support is real but for a
longer horizon and a different instrument (constructed spread indices,
not LQD price returns) than what this screen actually tests.

---

## 2. `yield_index -> credit_ig` and `yield_index -> credit_hy`

**Claim:** Treasury yield moves predict IG/HY credit ETF pricing via
duration.

**Literature found:** This one is closer to mechanical than empirical —
investment-grade bond funds have meaningfully more duration exposure
than high-yield (HY's shorter effective duration and higher coupon
means more of its return is carry, less is rate-sensitivity). This is
standard fixed-income arithmetic, not really a "finding" that needs a
citation — it follows from how bond pricing works.

**Recommendation: KEEP** for `yield_index -> credit_ig` (genuinely close
to mechanical, well within standard duration mathematics). **KEEP** for
`yield_index -> credit_hy` too, but note the relationship is structurally
WEAKER for HY exactly because of the lower duration — which is already
correctly described in the channel's own one-line reason. No change
needed, this one holds up.

---

## 3. `dollar_index -> yield_index` and `g10_fx -> yield_index`

**Claim:** Dollar strength / G10 FX moves reflect or lead rate
differential repricing.

**Literature found:** Uncovered interest rate parity and the carry-trade
literature support a relationship between rate differentials and FX, but
the DIRECTION here is backwards from the textbook causal story. The
standard channel is rates -> FX (rate differentials drive carry flows,
which move currencies), which is already captured by the existing
`yield_index -> dollar_index` and `yield_index -> g10_fx` entries. This
reverse-direction pair (FX leading rates) requires FX markets to be
pricing in future rate decisions before bond markets do, which is a much
more specific and contested claim — it implies systematic informational
inefficiency in the more liquid, more closely-watched rates market.

**Recommendation: CUT.** This was added as a "reverse leg" for symmetry,
not because I found independent support for FX leading rates rather than
the other way around. The forward direction
(`yield_index -> dollar_index`, `yield_index -> g10_fx`) already exists
in Tier 1/3 and should be kept; the reverse legs should be removed unless
you have a specific reason (e.g. FX option market positioning data) to
believe FX genuinely leads here.

---

## 4. `dollar_index -> em_fx` (Tier 1, but tested here given the April 2025 finding)

**Claim:** USD strength -> EM currency weakness (dollar smile / safe
haven channel).

**Literature found:** The dollar smile framework (Stephen Jen, Morgan
Stanley, ~2002) is widely cited and the historical pattern (1997-98
Asian crisis, 1998 Russia, 2001 Argentina — dollar rallying during
EXTERNAL stress) is real and well documented.

**Critical finding:** Multiple sources, including Wellington Management's
own institutional research, document that this relationship **broke down
specifically in April 2025** — during the tariff-driven volatility
episode, the dollar sold off AT THE SAME TIME as equities plunged and
recession odds rose, which is the opposite of what the dollar smile
predicts. This is the *exact same month* your validated vol->equity
channel fired correctly (Portfolio Tilt paper, Section 10.1). So in the
identical window where one channel in your prior worked perfectly, an
adjacent and structurally similar "flight to dollar" channel failed.
Wellington's explanation: elevated foreign ownership of unhedged US
assets (~$26T, 88% of GDP) means foreign investors now sometimes need to
SELL dollars to de-risk during a US-centered shock, rather than buying
dollars as a haven — a structural change since the dollar smile theory
was formulated in 2002.

**Recommendation: DOWNGRADE from Tier 1 to KEEP WITH CAVEAT.** The
relationship has real historical support but is not currently stable —
there's a credible, recent, well-sourced argument that the mechanism has
partially reversed for US-centered (as opposed to externally-centered)
shocks. Any signal from this channel should be treated with materially
less confidence than the vol->equity channel it sits next to in Tier 1.

---

## 5. `base_metal -> equity_country` ("Dr. Copper")

**Claim:** Copper price moves predict commodity-exporting country
equities.

**Literature found:** "Dr. Copper" is real, extensively documented, and
correlates well with PMI and GDP growth historically (CME Group
research, multiple academic citations on China industrial growth and
base metal prices).

**Problem, found directly in the search:** Every credible source
describes this relationship operating at **6-18 month horizons using
year-over-year price changes**, correlated against macro variables (PMI,
GDP, recession dating) — not at the 21-300 trading-day, ETF-to-ETF
resolution this screen tests. One source (Recessionist Pro) is explicit:
"Copper signals work best for 6-18 month outlooks, not short-term
trading." Multiple sources from 2025-2026 also note the relationship is
weakening structurally as copper demand shifts toward the energy
transition (EV/grid buildout) rather than traditional industrial-cycle
demand, decoupling it somewhat from the construction/manufacturing cycle
it historically tracked.

**Recommendation: DOWNGRADE.** Real macro relationship, wrong horizon
for this screen, and contested even at its native horizon as of recent
literature. If kept, restrict to the longest horizons in your grid
(126-300 trading days) only — don't let it license short-horizon
(21-63 day) configurations.

---

## 6. `broad_commodity -> em_fx`

**Claim:** Commodity terms-of-trade shifts predict commodity-exporter
currency strength.

**Literature found:** This is a genuinely well-established relationship
in international macroeconomics (terms-of-trade models of exchange rate
determination go back decades) and is the most mechanically sound of the
debatable group — commodity-exporting countries' export revenues and
trade balances are directly affected by commodity price moves, and FX
markets price this in relatively efficiently and quickly for liquid EM
currencies (BRL, ZAR, MXN are all meaningfully commodity-linked).

**Recommendation: KEEP.** Of all 19 debatable channels, this is the one
I'd most confidently leave in Tier 1 rather than Tier 2. The mechanism
is direct, the instruments are well-matched (commodity baskets to
commodity-exporter FX), and the literature operates closer to the
relevant horizon than the copper/equity case above.

---

## 7. `crypto_major -> broad_equity_us`

**Claim:** Crypto drawdowns lead broader risk-asset de-leveraging.

**Literature found:** Mixed and genuinely contested. There is a body of
work (Bouri et al. and others, cited in your own Atlas paper) on crypto
as a "diversifier" with LOW correlation to traditional risk assets for
much of its history, which argues against a leading-indicator role.
Post-2024 spot-ETF-driven institutional adoption has plausibly increased
crypto-equity correlation and made cross-asset deleveraging spillovers
more likely (shared margin books, shared institutional holders), but
this is a recent structural shift, not an established long-run pattern.

**Recommendation: DOWNGRADE.** The mechanism is more plausible today
than it was five years ago, but "plausible given a recent structural
shift, with no long track record" is a different and weaker claim than
"established channel." Treat with real caution, especially since your
own Atlas paper documents the crypto ETF history is under 2 years —
exactly the short-history caveat already flagged elsewhere in your own
work.

---

## 8. `em_fx -> broad_equity_em`, `broad_equity_em -> em_fx`, `broad_equity_em -> credit_em`

**Claim:** Bidirectional EM currency/equity/credit stress sharing a
common sentiment driver.

**Literature found:** This is well supported — EM contagion literature
(Forbes & Rigobon 2002, cited in your own Atlas references; Bekaert,
Harvey & Ng 2005, also in your Atlas references) consistently finds EM
assets share a common risk-sentiment factor during stress episodes, and
the co-movement is genuinely close to contemporaneous rather than
strictly lead-lag in one direction — which is actually a problem for a
forecasting framework that needs a predictor->target direction.

**Recommendation: KEEP WITH CAVEAT.** The shared-driver story is solid,
but because the literature describes near-simultaneous co-movement
rather than a clean lead-lag relationship, a configuration that fires
because EM FX moved first and EM equities followed by some number of
trading days may just be picking up noise in which leg happened to cross
its threshold first on a given day, not a genuine timing edge. This is
conceptually the weakest type of "literature-supported" channel: real
correlation, uncertain direction.

---

## 9. `broad_equity_us -> broad_equity_intl`, `broad_equity_us -> broad_equity_em`

**Claim:** US market leadership feeds into international/EM equity
co-movement.

**Literature found:** Well established — US market leadership in global
equity co-movement is one of the most robustly documented findings in
international finance (this is essentially the global factor structure
literature, going back to King & Wadhwani 1990 and many successors).

**Recommendation: KEEP.** This is genuinely one of the better-supported
channels in the debatable list. If anything it arguably belongs in Tier
1, not Tier 2 — I'd guess I put it in "debatable" mainly because it
felt close to circular ("US predicts the world") rather than because the
evidence is actually weak.

---

## 10. `yield_index/credit_hy/sector_equity -> bank_single_name` (JPM) and `energy/sector_equity -> energy_single_name` (XOM)

**Claim:** Sector and macro drivers predict the relevant single-name
constituent.

**Literature found:** Not really a "finding to cite" — this is closer to
definitional. A bank's net interest margin mechanically depends on the
rate environment; an integrated energy major's earnings mechanically
depend on crude prices. The relevant question isn't "is there a
literature" but "is the single-name's PRICE move, at a 1-300 day
horizon, actually forecastable from the sector signal, or does the
single name often move first / independently due to idiosyncratic
factors (earnings, litigation, management changes)?"

**Recommendation: KEEP, with a structural note.** These channels are
fine in principle but inherently noisier than sector-to-sector channels
because single names carry idiosyncratic risk the sector ETF doesn't.
If you want a cleaner version of this same economic intuition, consider
replacing JPM/XOM with `XLF`/`XLE` as the target instead of the single
name — you already have `sector_equity` well covered elsewhere in the
prior, and it removes the idiosyncratic-risk problem entirely without
losing the underlying mechanism.

---

## Summary table

| Channel | Recommendation |
|---|---|
| credit_ig -> broad_equity_us | DOWNGRADE |
| credit_ig -> credit_hy | KEEP WITH CAVEAT |
| yield_index -> credit_ig | KEEP |
| yield_index -> credit_hy | KEEP |
| dollar_index -> yield_index | CUT |
| g10_fx -> yield_index | CUT |
| dollar_index -> em_fx | DOWNGRADE (from Tier 1) |
| base_metal -> equity_country | DOWNGRADE (long horizons only) |
| broad_commodity -> em_fx | KEEP (candidate for Tier 1) |
| crypto_major -> broad_equity_us | DOWNGRADE |
| em_fx -> broad_equity_em | KEEP WITH CAVEAT |
| broad_equity_em -> em_fx | KEEP WITH CAVEAT |
| broad_equity_em -> credit_em | KEEP WITH CAVEAT |
| broad_equity_us -> broad_equity_intl | KEEP (candidate for Tier 1) |
| broad_equity_us -> broad_equity_em | KEEP (candidate for Tier 1) |
| yield_index/credit_hy/sector_equity -> bank_single_name | KEEP, consider XLF instead of JPM |
| energy/sector_equity -> energy_single_name | KEEP, consider XLE instead of XOM |

**Net effect if you accept all recommendations as-is:** 2 channels cut
outright (`dollar_index -> yield_index`, `g10_fx -> yield_index`), 4
downgraded to lower-conviction status, 3 promoted toward Tier 1, the
remainder kept as-is or kept with an explicit caveat flag. This is a
real, substantive revision — not a rubber stamp — but it's also not a
wholesale rejection of the original list. Most of Tier 2 holds up; the
specific failures found were concentrated in places where a textbook
macro relationship was being asked to operate at a much shorter horizon
and a narrower instrument set than the literature actually tested.
