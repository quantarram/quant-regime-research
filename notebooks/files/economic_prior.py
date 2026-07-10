"""
economic_prior.py
==================
A pre-specified, data-blind map of which predictor -> target relationships
are economically admissible for the CPE screen, plus the sub-class
taxonomy it is built on.

WHY THIS EXISTS
----------------
Across all three CPE papers, every purely statistical correction tried
(shrinkage, sample-size filters, episode-independence filters) failed to
separate genuine relationships from coincidental ones, because all of
them are computed from price history alone. The clearest example in the
existing joint screen: CORN's strongest bearish joint configuration is
conditioned on [GLD, IAU] -- two gold proxies that are near-identical
instruments, with no economic reason to predict the price of corn.
Likewise SOL-USD conditioned on [UDN, EURUSD=X], and (per the Portfolio
Tilt paper) silver predicting dogecoin across five separate historical
episodes. No amount of additional historical support fixes this, because
the problem isn't insufficient data -- it's the absence of any
constraint on which pairs are allowed to be tested in the first place.

This module is that constraint. It is built and frozen BEFORE looking at
which pairs happened to perform well in any out-of-sample year, using
only domain knowledge about how these instruments relate (shared risk
factors, macro transmission channels, hedging relationships). It does
not replace CPE / lift / sample-size as the within-channel ranking
statistic -- it only restricts the candidate (X, Y) universe the greedy
search and pairwise screen are allowed to consider.

USAGE
-----
    from economic_prior import is_admissible, get_subclass, explain_pair

    if is_admissible(predictor="^VIX", target="SPY"):
        ...

Both cpe_engine_parallel.py and joint_cpe_engine.py import
`is_admissible` and apply it as a pre-filter on (Y, X) pairs before any
CPE computation runs, so disallowed pairs never enter the loop.

DESIGN NOTES
------------
- The admissibility map is directional: X -> Y does not imply Y -> X.
  E.g. volatility spikes plausibly forecast equity weakness; the reverse
  (equity weakness "forecasting" a vol spike) is close to definitional
  and is excluded to avoid trivially circular configurations.
- Within-subclass pairs are admissible only where they share a genuine
  common driver, not merely a shared broad asset_class label. Two gold
  proxies (GLD, IAU) are functionally the same instrument and are
  excluded as predictor/target pairs of each other (self-referential,
  not informative).
- This is intentionally permissive enough to keep the volatility -> equity
  and intra-crypto channels that the Atlas and single-asset paper both
  found to be real and (in the one traced case) accurate -- it narrows
  the search space, it does not collapse it to one channel.
- Anything not explicitly listed is INADMISSIBLE by default. This is a
  whitelist, not a blacklist, by design: the failure mode we are
  correcting for is "too permissive", so the safe default is exclusion.
"""

from __future__ import annotations
from typing import Dict, Set, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────
# 1. SUB-CLASS TAXONOMY
#    Finer-grained than the 6 broad asset_class labels in
#    multiasset_metadata.parquet, because "commodities" alone conflates
#    gold with corn, and "equities" conflates a single growth stock with
#    a utilities sector ETF. The prior is written at this resolution.
# ─────────────────────────────────────────────────────────────────────────

SUBCLASS: Dict[str, str] = {
    # ── Commodities ──────────────────────────────────────────────────
    "GLD": "precious_metal", "IAU": "precious_metal", "GC=F": "precious_metal",
    "SLV": "precious_metal", "SI=F": "precious_metal",
    "PPLT": "precious_metal", "PALL": "precious_metal",

    "USO": "energy", "BNO": "energy", "CL=F": "energy", "BZ=F": "energy",
    "UNG": "energy", "UGA": "energy", "NG=F": "energy",

    "CORN": "agriculture", "WEAT": "agriculture", "SOYB": "agriculture",
    "CANE": "agriculture", "NIB": "agriculture", "JO": "agriculture",
    "ZC=F": "agriculture", "ZW=F": "agriculture", "ZS=F": "agriculture",

    "CPER": "base_metal", "DBB": "base_metal", "HG=F": "base_metal",

    "DJP": "broad_commodity", "PDBC": "broad_commodity", "DBC": "broad_commodity",
    "GSG": "broad_commodity",

    # ── Crypto ───────────────────────────────────────────────────────
    "BTC-USD": "crypto_major", "ETH-USD": "crypto_major",
    "IBIT": "crypto_btc_etf", "FBTC": "crypto_btc_etf",
    "GBTC": "crypto_btc_etf", "BITB": "crypto_btc_etf",
    "ETHE": "crypto_eth_etf",
    "SOL-USD": "crypto_alt", "BNB-USD": "crypto_alt", "XRP-USD": "crypto_alt",
    "ADA-USD": "crypto_alt", "AVAX-USD": "crypto_alt", "DOGE-USD": "crypto_alt",
    "DOT-USD": "crypto_alt", "LINK-USD": "crypto_alt", "MATIC-USD": "crypto_alt",
    "LTC-USD": "crypto_alt", "BCH-USD": "crypto_alt", "UNI-USD": "crypto_alt",
    "ATOM-USD": "crypto_alt",

    # ── Equities ─────────────────────────────────────────────────────
    "SPY": "broad_equity_us", "QQQ": "broad_equity_us", "IWM": "broad_equity_us",
    "DIA": "broad_equity_us", "VTI": "broad_equity_us",
    "VT": "broad_equity_global", "EFA": "broad_equity_intl", "VEA": "broad_equity_intl",
    "EEM": "broad_equity_em", "VWO": "broad_equity_em",
    "EWJ": "equity_country", "EWZ": "equity_country", "FXI": "equity_country",
    "INDA": "equity_country", "EWY": "equity_country",

    "XLK": "sector_equity", "XLF": "sector_equity", "XLE": "sector_equity",
    "XLV": "sector_equity", "XLI": "sector_equity", "XLP": "sector_equity",
    "XLY": "sector_equity", "XLU": "sector_equity", "XLRE": "sector_equity",
    "XLB": "sector_equity", "XLC": "sector_equity", "XBI": "sector_equity",
    "SOXX": "sector_equity", "ITB": "sector_equity", "ICLN": "sector_equity",

    "VTV": "factor_equity", "VUG": "factor_equity", "MTUM": "factor_equity",
    "USMV": "factor_equity", "QUAL": "factor_equity", "SIZE": "factor_equity",

    "SSO": "leveraged_equity", "SDS": "leveraged_equity", "TQQQ": "leveraged_equity",
    "ARKK": "thematic_equity",

    "AAPL": "megacap_tech", "MSFT": "megacap_tech", "NVDA": "megacap_tech",
    "AMZN": "megacap_tech", "GOOGL": "megacap_tech", "META": "megacap_tech",
    "TSLA": "megacap_tech",
    "JPM": "bank_single_name", "XOM": "energy_single_name",
    "BRK-B": "single_name_other",

    # ── FX ───────────────────────────────────────────────────────────
    "EURUSD=X": "g10_fx", "GBPUSD=X": "g10_fx", "JPYUSD=X": "g10_fx",
    "CHFUSD=X": "g10_fx", "CADUSD=X": "g10_fx", "AUDUSD=X": "g10_fx",
    "NZDUSD=X": "g10_fx",
    "SGDUSD=X": "em_fx", "CNYUSD=X": "em_fx", "INRUSD=X": "em_fx",
    "BRLUSD=X": "em_fx", "MXNUSD=X": "em_fx", "ZARUSD=X": "em_fx",
    "KRWUSD=X": "em_fx", "THBUSD=X": "em_fx",
    "UUP": "dollar_index", "UDN": "dollar_index",
    "EURJPY=X": "fx_cross", "EURGBP=X": "fx_cross", "GBPJPY=X": "fx_cross",
    "AUDJPY=X": "fx_cross", "EURCHF=X": "fx_cross",

    # ── Rates / credit ───────────────────────────────────────────────
    "SHY": "treasury_short", "IEI": "treasury_short",
    "IEF": "treasury_mid", "TLH": "treasury_long", "TLT": "treasury_long",
    "ZROZ": "treasury_long", "EDV": "treasury_long",
    "TIP": "tips", "SCHP": "tips",
    "LQD": "credit_ig", "VCSH": "credit_ig", "VCIT": "credit_ig", "VCLT": "credit_ig",
    "HYG": "credit_hy", "JNK": "credit_hy",
    "EMB": "credit_em",
    "AGG": "broad_bond", "BND": "broad_bond", "MBB": "broad_bond",
    "MUB": "muni",
    "^TNX": "yield_index", "^TYX": "yield_index", "^FVX": "yield_index",
    "^IRX": "yield_index",
    "TMF": "leveraged_rates", "TBT": "leveraged_rates", "TBF": "leveraged_rates",

    # ── Volatility ───────────────────────────────────────────────────
    "^VIX": "vol_index_equity", "^VXN": "vol_index_equity",
    "^VVIX": "vol_index_equity", "^SKEW": "vol_index_equity",
    "^OVX": "vol_index_other", "^GVZ": "vol_index_other", "^EVZ": "vol_index_other",
    "UVXY": "vol_etp", "SVXY": "vol_etp", "VXX": "vol_etp",
    "VIXY": "vol_etp", "VIXM": "vol_etp",
}


def get_subclass(ticker: str) -> str:
    """Return the sub-class label for a ticker, or 'unknown' if unmapped."""
    return SUBCLASS.get(ticker, "unknown")


# ─────────────────────────────────────────────────────────────────────────
# 2. ADMISSIBLE SUB-CLASS -> SUB-CLASS CHANNELS
#    Each entry is (predictor_subclass, target_subclass) -> short reason.
#    This is the actual economic prior. Add to it only with a one-line
#    mechanism, never just because a pair "showed up" in a screen.
# ─────────────────────────────────────────────────────────────────────────

ADMISSIBLE_CHANNELS: Dict[Tuple[str, str], str] = {

    # ── Volatility -> Equities (risk-off propagation) ──────────────────
    # The one channel both prior papers found to be real and, in the
    # Portfolio Tilt paper's traced April 2025 episode, accurate.
    ("vol_index_equity", "broad_equity_us"):  "vol spike -> equity reversal/drawdown (leverage & risk-off)",
    ("vol_index_equity", "broad_equity_intl"):"vol spike -> equity reversal/drawdown",
    ("vol_index_equity", "broad_equity_em"):  "vol spike -> EM equity drawdown (risk-off, capital flight)",
    ("vol_index_equity", "sector_equity"):    "vol spike -> sector-level risk-off",
    ("vol_index_equity", "equity_country"):   "vol spike -> country equity risk-off",
    ("vol_index_equity", "factor_equity"):    "vol spike -> factor rotation (e.g. into low-vol/quality)",
    ("vol_index_equity", "credit_hy"):        "vol spike -> high-yield credit spread widening",
    ("vol_index_equity", "treasury_long"):    "vol spike -> flight-to-quality bond buying",
    ("vol_etp", "broad_equity_us"):           "same mechanism as vol_index_equity, via decaying ETPs (caution: roll-decay non-stationarity, see single-asset paper Sec 3.3)",
    ("vol_index_other", "energy"):            "implied-vol spike in a commodity complex -> price stress in that complex",
    ("vol_index_other", "precious_metal"):    "implied-vol spike (e.g. ^GVZ) -> precious metal price stress",

    # ── Equities -> Volatility is EXCLUDED (definitional/circular):
    #    a falling equity market mechanically raises VIX; this is not an
    #    independent forecasting relationship.

    # ── Credit -> Equities (credit leads risk sentiment) ───────────────
    ("credit_hy", "broad_equity_us"):   "HY spread widening -> equity weakness (credit leads risk appetite)",
    ("credit_hy", "sector_equity"):     "HY spread widening -> cyclical sector weakness",
    ("credit_em", "broad_equity_em"):   "EM credit stress -> EM equity stress (shared sovereign risk)",

    # ── Rates -> FX / Equities (carry & rate-differential channel) ─────
    ("yield_index", "dollar_index"):    "rate differential -> USD strength (rate-driven capital flows)",
    ("yield_index", "g10_fx"):          "rate differential -> bilateral FX moves",
    ("treasury_long", "factor_equity"): "long-duration rate moves -> duration-sensitive equity factors",
    ("treasury_long", "sector_equity"): "rate moves -> rate-sensitive sectors (utilities, real estate, financials)",

    # ── Dollar -> EM FX / EM equities / commodities (USD as global driver) ─
    ("dollar_index", "em_fx"):          "USD strength -> EM currency weakness (classic dollar-smile channel)",
    ("dollar_index", "broad_equity_em"):"USD strength -> EM equity headwind (dollar-denominated debt, capital flows)",
    ("dollar_index", "precious_metal"): "USD strength -> gold/silver priced inversely to dollar",
    ("dollar_index", "broad_commodity"):"USD strength -> broad commodity complex (dollar-denominated pricing)",
    ("dollar_index", "credit_em"):      "USD strength -> EM dollar-debt credit stress",

    # ── Precious metals -> precious metals (shared real-rate / safe-haven driver) ─
    # NOTE: explicitly excludes GLD<->IAU<->GC=F (functionally identical
    # instruments; see module docstring). Implemented via same-ticker-
    # group exclusion below, not by omitting this channel entirely,
    # since e.g. SLV led/lagged by GC=F is a genuine distinct-instrument
    # relationship (gold/silver ratio dynamics).
    ("precious_metal", "precious_metal"): "shared real-rate and safe-haven demand driver (gold/silver ratio dynamics)",

    # ── Energy -> Energy (shared supply/demand and crack-spread dynamics) ─
    ("energy", "energy"): "shared crude/products supply-demand and crack-spread linkages",

    # ── Base metals <-> broad commodity / equities (growth proxy) ──────
    ("base_metal", "broad_commodity"):  "industrial-demand growth signal shared across commodity complex",
    ("base_metal", "equity_country"):   "Dr. Copper as a global-growth proxy -> commodity-exporting equities",
    ("broad_commodity", "em_fx"):       "commodity terms-of-trade -> commodity-exporter currency strength",

    # ── Crypto -> Crypto (single dominant risk factor; well documented) ─
    ("crypto_major", "crypto_alt"):     "BTC/ETH as dominant risk factor for altcoins (high realized beta)",
    ("crypto_major", "crypto_major"):   "BTC/ETH co-movement (shared crypto risk factor)",
    ("crypto_alt", "crypto_alt"):       "shared altcoin risk factor / sector rotation within crypto",
    ("crypto_btc_etf", "crypto_major"): "ETF flows -> spot price (flow-through mechanism, but see short-history caveat below)",
    ("crypto_major", "crypto_btc_etf"): "spot price -> ETF NAV tracking (near-mechanical, retained for completeness)",

    # ── Crypto -> risk assets broadly (when crypto itself is the stress signal) ─
    ("crypto_major", "broad_equity_us"): "crypto drawdown as a leading indicator of broader risk-asset de-leveraging",

    # ── Yield curve / rates as TARGETS, not just predictors ─────────────
    # The original draft only let yield_index predict other things; rates
    # markets are themselves forecastable from credit stress and equity
    # vol, which the literature (flight-to-quality) supports directly.
    ("vol_index_equity", "tips"):          "vol spike -> flight-to-quality real-rate demand",
    ("vol_index_equity", "treasury_short"):"vol spike -> short-end flight-to-quality / rate-cut expectations",
    ("vol_index_equity", "treasury_mid"):  "vol spike -> flight-to-quality bond buying, belly of curve",
    ("vol_index_equity", "broad_bond"):    "vol spike -> aggregate bond demand (flight-to-quality)",
    ("vol_index_equity", "credit_ig"):     "vol spike -> investment-grade spread widening (milder than HY)",
    ("vol_index_equity", "credit_em"):     "vol spike -> EM credit spread widening (global risk-off)",
    ("vol_index_equity", "muni"):          "vol spike -> muni demand via broad flight-to-quality (weaker, rate-driven)",
    ("vol_index_equity", "leveraged_rates"):"vol spike -> rate-direction bets via leveraged rate products",

    ("credit_hy", "treasury_long"):    "HY spread widening -> duration bid (classic risk-off rotation)",
    ("credit_hy", "credit_ig"):        "HY spread contagion into IG credit (shared credit-cycle driver)",
    ("credit_ig", "broad_equity_us"):  "IG spread widening -> equity weakness (credit leads risk, milder than HY)",
    ("credit_ig", "credit_hy"):        "IG spread move as an early signal for HY (credit-quality cascade)",

    # ── Yield curve internal structure (tenor-to-tenor) ─────────────────
    ("yield_index", "treasury_short"): "rate-level/curve moves -> short-end ETF pricing",
    ("yield_index", "treasury_mid"):   "rate-level/curve moves -> belly ETF pricing",
    ("yield_index", "treasury_long"):  "rate-level/curve moves -> long-end ETF pricing (duration channel)",
    ("yield_index", "tips"):           "nominal yield moves -> real yield / breakeven-linked TIPS pricing",
    ("yield_index", "credit_ig"):      "rate moves -> IG credit ETF pricing (duration component of credit)",
    ("yield_index", "credit_hy"):      "rate moves -> HY credit ETF pricing (smaller duration component)",
    ("yield_index", "broad_bond"):     "rate moves -> aggregate bond index pricing",
    ("yield_index", "muni"):           "rate moves -> muni bond pricing (correlated duration exposure)",
    ("yield_index", "leveraged_rates"):"rate moves -> direct mechanical driver of leveraged rate products",
    ("treasury_long", "tips"):         "nominal long-duration moves -> real-duration TIPS co-movement",
    ("treasury_long", "leveraged_rates"):"long-duration ETF moves -> mechanically linked leveraged rate products",

    # ── Dollar / FX feedback into rates ──────────────────────────────────
    ("dollar_index", "yield_index"):   "USD strength via rate-differential expectations (carry channel, reverse leg)",
    ("g10_fx", "yield_index"):         "G10 FX moves reflecting/leading rate-differential repricing",

    # ── EM equity/FX feedback (two-way shared risk sentiment) ───────────
    ("em_fx", "broad_equity_em"):      "EM currency stress -> EM equity stress (shared sovereign/capital-flow risk)",
    ("broad_equity_em", "credit_em"):  "EM equity stress -> EM sovereign/corporate credit stress",
    ("broad_equity_em", "em_fx"):      "EM equity stress -> EM currency stress (reverse leg of shared sentiment)",

    # ── Base metals as targets (not just predictors) ────────────────────
    ("broad_commodity", "base_metal"): "broad commodity-complex demand signal -> industrial metals",
    ("dollar_index", "base_metal"):    "USD strength -> base metals priced inversely to dollar (same as precious metals)",
    ("vol_index_other", "base_metal"): "commodity-complex stress -> industrial metals",

    # ── Vol-of-vol and cross-vol-complex structure ───────────────────────
    ("vol_index_equity", "vol_index_other"): "broad equity-vol regime shift -> other implied-vol complexes (shared risk regime)",
    ("vol_etp", "vol_index_other"):          "vol ETP moves reflecting broader cross-asset vol regime",

    # ── Single megacap names as targets of broad market/sector moves ────
    ("broad_equity_us", "megacap_tech"):  "broad index moves -> mega-cap constituent moves (reverse of index-weighting channel)",
    ("sector_equity", "megacap_tech"):    "sector moves -> sector-constituent mega-cap moves",
    ("vol_index_equity", "megacap_tech"): "vol spike -> high-beta mega-cap weakness",

    # ── Leveraged/thematic equity as targets of their underlying driver ──
    ("broad_equity_us", "leveraged_equity"): "underlying index moves -> mechanically leveraged product moves",
    ("sector_equity", "thematic_equity"):    "related sector moves -> thematic ETF moves (e.g. clean energy/tech overlap)",
    ("vol_index_equity", "thematic_equity"): "vol spike -> high-beta thematic ETF weakness",

    # ── International developed equities as targets ──────────────────────
    ("vol_index_equity", "broad_equity_intl"): "vol spike -> developed-market equity risk-off (already listed once; kept here for clarity of intl coverage)",
    ("broad_equity_us", "broad_equity_intl"):  "US market leadership -> developed international equity co-movement",
    ("broad_equity_us", "broad_equity_em"):    "US market leadership -> EM equity co-movement (global risk appetite)",

    # ── FX crosses as targets of their component-currency drivers ───────
    ("g10_fx", "fx_cross"):    "component bilateral USD-pair moves -> cross-rate moves (arithmetic/economic linkage)",
    ("dollar_index", "fx_cross"): "USD regime -> cross-rate moves via shared USD leg dynamics",

    # ── Global broad equity (VT) and total-market (covering zero-coverage gap) ─
    ("broad_equity_us", "broad_equity_global"): "US dominates global-market-cap weighting -> total-world index moves",
    ("vol_index_equity", "broad_equity_global"): "vol spike -> global equity risk-off",

    # ── Crypto ETH-denominated trust as target ───────────────────────────
    ("crypto_major", "crypto_eth_etf"): "ETH spot price -> ETH trust NAV tracking (near-mechanical, retained for completeness)",

    # ── Specific single-name mechanisms (only where a clean driver exists;
    #    BRK-B deliberately excluded as a diversified conglomerate with no
    #    single dominant macro driver) ────────────────────────────────────
    ("yield_index", "bank_single_name"):  "rate moves -> bank net-interest-margin-sensitive name (JPM)",
    ("credit_hy", "bank_single_name"):    "credit stress -> financials-sector single name (JPM)",
    ("sector_equity", "bank_single_name"):"financials-sector ETF moves -> constituent bank (JPM)",
    ("energy", "energy_single_name"):     "energy-complex moves -> integrated energy major (XOM)",
    ("sector_equity", "energy_single_name"): "energy-sector ETF moves -> constituent major (XOM)",


    ("sector_equity", "broad_equity_us"):  "sector leadership/rotation feeding into broad index moves",
    ("factor_equity", "broad_equity_us"):  "factor rotation feeding into broad index moves",
    ("megacap_tech", "broad_equity_us"):   "mega-cap concentration -> index-level moves (index weighting mechanism)",
    ("megacap_tech", "sector_equity"):     "mega-cap concentration -> sector-level moves",

    # ── Country/regional equities <-> broad EM / global ────────────────
    ("equity_country", "broad_equity_em"): "single-country EM weakness/strength feeding into broad EM index",
    ("broad_equity_em", "em_fx"):          "EM risk sentiment shared across EM equity and EM FX",

    # ── Agriculture -> Agriculture (shared weather/planting-cycle driver) ─
    ("agriculture", "agriculture"): "shared weather, planting-cycle, and biofuel-demand drivers across crops",
}

# Reasons stored for documentation / audit; the actual gate is the key set.
ADMISSIBLE_KEYS: Set[Tuple[str, str]] = set(ADMISSIBLE_CHANNELS.keys())


# ─────────────────────────────────────────────────────────────────────────
# 2a. CONFIDENCE OVERRIDES — from literature review
#    See literature_review_tier2.md for the full per-channel writeup this
#    table implements. Every channel in ADMISSIBLE_CHANNELS defaults to
#    "standard" confidence; entries here override that default based on
#    a check against published research, specifically distinguishing
#    "literature exists" from "literature exists AT THE HORIZON AND
#    INSTRUMENT RESOLUTION THIS SCREEN ACTUALLY TESTS" -- the latter
#    turned out to be the rarer and more important property.
#
#    Tiers, in order of trust:
#      "high"      -- literature support at a matching horizon/instrument
#                      resolution, or the validated channel itself
#      "standard"  -- the default; no specific override below
#      "caveat"    -- real relationship, but literature also documents a
#                      specific, named failure mode or regime-dependence;
#                      a firing signal here should be sized down or
#                      flagged, not silently trusted at face value
#      "weak"      -- literature support exists but is for a different
#                      horizon (commonly: macro/YoY horizons of 6+
#                      months vs this screen's 1-300 trading days) or a
#                      different instrument type (constructed index vs
#                      single ETF); treat as the least trustworthy
#                      admissible tier, not as equivalent to "standard"
#    A channel can also be entirely removed by literature review (see
#    REMOVED_BY_LITERATURE_REVIEW below) rather than merely downgraded.
# ─────────────────────────────────────────────────────────────────────────

CONFIDENCE_OVERRIDES: Dict[Tuple[str, str], str] = {
    # The one channel with actual out-of-sample validation, not just
    # literature support: Portfolio Tilt paper Section 10.1 traced this
    # exact channel's April 2025 firing to 11/11 correct forward SPY
    # returns. This is a stronger evidentiary basis than any purely
    # literature-derived channel in this module.
    ("vol_index_equity", "broad_equity_us"): "high",

    # Promoted -- literature support found to be at least as strong as
    # the Tier-1 defaults already in the module
    ("broad_commodity", "em_fx"):              "high",
    ("broad_equity_us", "broad_equity_intl"):  "high",
    ("broad_equity_us", "broad_equity_em"):    "high",

    # Downgraded -- real literature, wrong horizon or wrong instrument
    # type for what this screen actually tests
    ("credit_ig", "broad_equity_us"):  "weak",
    ("base_metal", "equity_country"):  "weak",   # Dr. Copper: literature is 6-18mo/YoY, this screen is 1-300 trading days
    ("dollar_index", "em_fx"):         "caveat", # dollar smile broke down specifically in Apr 2025; see review doc
    ("crypto_major", "broad_equity_us"): "caveat", # plausible only post-2024 ETF-driven correlation shift, no long track record

    # Kept, but flagged as correlation-without-clear-direction rather
    # than a clean lead-lag relationship
    ("em_fx", "broad_equity_em"):       "caveat",
    ("broad_equity_em", "em_fx"):       "caveat",
    ("broad_equity_em", "credit_em"):   "caveat",
    ("credit_ig", "credit_hy"):         "caveat",
}

# Removed outright by literature review: added originally as a "reverse
# leg" of an already-admissible forward channel (rates -> FX), but no
# independent literature support was found for FX leading rates rather
# than the reverse, which is the much better-supported direction and is
# already captured elsewhere in this module.
REMOVED_BY_LITERATURE_REVIEW: Set[Tuple[str, str]] = {
    ("dollar_index", "yield_index"),
    ("g10_fx", "yield_index"),
}
for _k in REMOVED_BY_LITERATURE_REVIEW:
    ADMISSIBLE_CHANNELS.pop(_k, None)
    ADMISSIBLE_KEYS.discard(_k)


def get_confidence(predictor_subclass: str, target_subclass: str) -> str:
    """Return the confidence tier for a subclass-pair channel:
    'high', 'standard' (default), 'caveat', or 'weak'."""
    return CONFIDENCE_OVERRIDES.get((predictor_subclass, target_subclass), "standard")


# ─────────────────────────────────────────────────────────────────────────
# 3. SAME-INSTRUMENT-GROUP EXCLUSION
#    Even within an admissible channel, near-duplicate instruments
#    tracking the same underlying (GLD/IAU/GC=F; CL=F/USO; multiple
#    Bitcoin ETFs) are not informative predictor/target pairs of each
#    other -- conditioning GLD on IAU is closer to a data-quality
#    artifact than a forecasting relationship.
# ─────────────────────────────────────────────────────────────────────────

DUPLICATE_GROUPS = [
    {"GLD", "IAU", "GC=F"},
    {"SLV", "SI=F"},
    {"USO", "CL=F"},
    {"BNO", "BZ=F"},
    {"UNG", "NG=F"},
    {"CORN", "ZC=F"},
    {"WEAT", "ZW=F"},
    {"SOYB", "ZS=F"},
    {"CPER", "HG=F"},
    {"DJP", "PDBC", "DBC", "GSG"},  # all broad-commodity baskets, heavily overlapping
    {"IBIT", "FBTC", "GBTC", "BITB"},  # all spot/quasi-spot BTC wrappers
    {"SHY", "IEI"}, {"IEF", "TLH"}, {"TLT", "ZROZ", "EDV"},
    {"TIP", "SCHP"},
    {"LQD", "VCIT"}, {"HYG", "JNK"},
    {"AGG", "BND"},
    {"UUP", "UDN"},  # literally inverse of each other
    {"SPY", "VTI", "VT"},
    {"^TNX", "^TYX", "^FVX", "^IRX"},  # same yield curve, different tenors -- correlated by construction
]

_DUP_LOOKUP: Dict[str, frozenset] = {}
for grp in DUPLICATE_GROUPS:
    fs = frozenset(grp)
    for t in grp:
        _DUP_LOOKUP[t] = fs


def _same_duplicate_group(x: str, y: str) -> bool:
    gx = _DUP_LOOKUP.get(x)
    return gx is not None and y in gx


# ─────────────────────────────────────────────────────────────────────────
# 4. PUBLIC API
# ─────────────────────────────────────────────────────────────────────────

_CONFIDENCE_RANK = {"weak": 0, "caveat": 1, "standard": 2, "high": 3}


def is_admissible(predictor: str, target: str, min_confidence: str = "weak") -> bool:
    """
    Return True iff (predictor -> target) is in the pre-specified
    economic prior AND clears the given confidence floor. This is the
    single gate both engines must call before computing or retaining any
    CPE for a (X, Y) pair.

    min_confidence: one of "weak" (default -- accept anything admissible,
    matching pre-literature-review behaviour), "caveat", "standard", or
    "high". Raising this excludes channels the literature review found
    to have a horizon/instrument mismatch or a documented failure mode
    (see CONFIDENCE_OVERRIDES and literature_review_tier2.md).
    """
    if predictor == target:
        return False
    if _same_duplicate_group(predictor, target):
        return False

    x_sub = get_subclass(predictor)
    y_sub = get_subclass(target)
    if x_sub == "unknown" or y_sub == "unknown":
        return False

    if (x_sub, y_sub) not in ADMISSIBLE_KEYS:
        return False

    channel_conf = get_confidence(x_sub, y_sub)
    return _CONFIDENCE_RANK[channel_conf] >= _CONFIDENCE_RANK[min_confidence]


def explain_pair(predictor: str, target: str) -> str:
    """Human-readable explanation of why a pair is or isn't admissible,
    including its literature-review confidence tier."""
    if predictor == target:
        return "REJECTED: predictor and target are the same instrument."
    if _same_duplicate_group(predictor, target):
        return (f"REJECTED: {predictor} and {target} are near-duplicate "
                f"instruments of the same underlying ({_DUP_LOOKUP[predictor]}).")
    x_sub, y_sub = get_subclass(predictor), get_subclass(target)
    if x_sub == "unknown" or y_sub == "unknown":
        return f"REJECTED: unmapped sub-class ({predictor}={x_sub}, {target}={y_sub})."
    reason = ADMISSIBLE_CHANNELS.get((x_sub, y_sub))
    if reason:
        conf = get_confidence(x_sub, y_sub)
        return f"ADMISSIBLE [{x_sub} -> {y_sub}] (confidence={conf}): {reason}"
    return f"REJECTED: no admissible channel for [{x_sub} -> {y_sub}]."


def admissible_predictors_for(target: str, universe: list, min_confidence: str = "weak") -> list:
    """Given a target ticker and a universe of candidate tickers, return
    the subset that are admissible predictors of that target at or above
    the given confidence floor."""
    return [x for x in universe if is_admissible(x, target, min_confidence)]


def coverage_report(universe: list) -> "pd.DataFrame":
    """
    Diagnostic: for every ticker in `universe`, how many admissible
    predictors does it have, and from which sub-classes. Run this once
    after editing ADMISSIBLE_CHANNELS to sanity-check coverage before
    re-running the full screen.
    """
    import pandas as pd
    rows = []
    for y in universe:
        preds = admissible_predictors_for(y, universe)
        pred_subclasses = sorted({get_subclass(x) for x in preds})
        rows.append({
            "target": y,
            "target_subclass": get_subclass(y),
            "n_admissible_predictors": len(preds),
            "predictor_subclasses": ", ".join(pred_subclasses),
        })
    return pd.DataFrame(rows).sort_values("n_admissible_predictors", ascending=False)


if __name__ == "__main__":
    # Quick self-test against the exact problem cases identified in the
    # Portfolio Tilt paper and visible in joint_cpe_results.parquet.
    test_cases = [
        ("GLD", "IAU"),          # duplicate instrument -> reject
        ("GLD", "CORN"),         # gold -> corn, no mechanism -> reject
        ("UDN", "SOL-USD"),      # dollar index -> altcoin, no mechanism -> reject
        ("SLV", "DOGE-USD"),     # silver -> dogecoin, the paper's own flagged case -> reject
        ("^VIX", "SPY"),         # vol spike -> equity, the validated channel -> accept
        ("VIXM", "SPY"),         # vol ETP -> equity, same channel via ETP -> accept
        ("BTC-USD", "ETH-USD"),  # major crypto co-movement -> accept
        ("BTC-USD", "DOGE-USD"), # major crypto -> alt, dominant risk factor -> accept
        ("HYG", "SPY"),          # credit -> equity -> accept
        ("UUP", "ZARUSD=X"),     # dollar -> EM FX -> accept
        ("SPY", "^VIX"),         # reverse of validated channel -> reject (circular)
    ]
    print(f"{'predictor':<12} {'target':<10} {'result'}")
    print("-" * 80)
    for x, y in test_cases:
        print(f"{x:<12} {y:<10} {explain_pair(x, y)}")
