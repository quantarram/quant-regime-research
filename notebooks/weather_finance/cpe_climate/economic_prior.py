"""Shadow of notebooks/files/economic_prior.py with the climate-state predictors registered. Everything in the original is unchanged; the extension below is
written BEFORE any climate CPE result exists, as whole sub-class channels with a one-line mechanism and a confidence tier (the module's own rule: no pair is added because it
'showed up' in a screen). Disclosure: the author of this extension had already seen a simple ENSO-state excess-return scan (coffee and cotton looked persistent); the channels
below are the standard teleconnection mechanisms for EVERY agricultural, energy and metal target, not only those, and all tiers are run and reported."""
import importlib.util, os
_ORIG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'files', 'economic_prior.py'))
_spec = importlib.util.spec_from_file_location('economic_prior_orig', _ORIG)
_orig = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_orig)

CLIMATE_SUBCLASS = {}
for _idx, _sub in (('ONI', 'enso_state'), ('NINO34', 'enso_state'), ('SOI', 'enso_state'), ('AO', 'polar_oscillation'), ('NAO', 'polar_oscillation'), ('PDO', 'pacific_decadal')):
    for _form in ('chg', 'mean'):
        CLIMATE_SUBCLASS[f'{_idx}_{_form}'] = _sub
# agricultural and energy futures added to the universe for this run take the existing sub-class labels
EXTRA_TARGETS = {t: 'agriculture' for t in ('KC=F', 'CC=F', 'CT=F', 'SB=F', 'OJ=F', 'LE=F', 'HE=F', 'ZL=F', 'ZM=F', 'KE=F', 'ZO=F')}
EXTRA_TARGETS.update({t: 'energy' for t in ('HO=F', 'RB=F')})
CLIMATE_CHANNELS = {
    ('enso_state', 'agriculture'): ('standard', 'ENSO shifts monsoon, South-East Asian, Australian, South American and West African rainfall and temperature, changing crop supply (sugar, rice, wheat, soy, coffee, cocoa, palm oil); the author\'s Sugar paper validates the monsoon pathway prospectively'),
    ('enso_state', 'energy'): ('weak', 'ENSO modulates North American winter temperature and Atlantic hurricane activity (heating demand, Gulf supply); small and regime-dependent'),
    ('enso_state', 'base_metal'): ('weak', 'ENSO-linked rainfall and flood disruption at Chilean, Peruvian and Australian mines'),
    ('enso_state', 'broad_commodity'): ('weak', 'aggregate of the agriculture and energy exposures'),
    ('enso_state', 'equity_country'): ('weak', 'monsoon and crop channel into India and Brazil equity through income and inflation'),
    ('enso_state', 'em_fx'): ('weak', 'crop and inflation shocks move commodity-importer and exporter currencies (INR, BRL, ZAR, MXN)'),
    ('polar_oscillation', 'energy'): ('standard', 'AO / NAO phase sets winter cold-air outbreaks over North America and Europe, hence heating demand for gas and heating oil'),
    ('polar_oscillation', 'agriculture'): ('weak', 'winter and spring temperature and precipitation anomalies over Europe, the Black Sea and North America affect winter-crop conditions'),
    ('pacific_decadal', 'agriculture'): ('weak', 'PDO phase alters drought frequency in the Americas, Australia and Asia on multi-year scales'),
}
_orig.SUBCLASS.update(CLIMATE_SUBCLASS); _orig.SUBCLASS.update(EXTRA_TARGETS)
for (_k, (_conf, _why)) in CLIMATE_CHANNELS.items():
    _orig.ADMISSIBLE_CHANNELS[_k] = _why; _orig.ADMISSIBLE_KEYS.add(_k)
    if _conf != 'standard': _orig.CONFIDENCE_OVERRIDES[_k] = _conf
# ---- BATCH 2: extreme-event predictors (registered 2026-10-11, before any result) ----
EXTREME_SUBCLASS = {}
for _n, _sub in (('CBPDSI', 'us_drought'), ('CBTMP', 'us_heat'), ('HDDGAS', 'us_heating_demand'), ('CDDPOW', 'us_cooling_demand'), ('IOD', 'iod_state'), ('ACE', 'atlantic_hurricane')):
    for _form in ('chg', 'mean'):
        EXTREME_SUBCLASS[f'{_n}_{_form}'] = _sub
EXTREME_CHANNELS = {
    ('us_drought', 'agriculture'): ('standard', 'Midwest drought cuts corn, soybean and wheat yields and lifts feed-grain, oilseed and livestock prices (1934, 1988, 2012)'),
    ('us_heat', 'agriculture'): ('standard', 'summer heat during pollination and grain fill cuts corn and soybean yields'),
    ('us_heating_demand', 'energy'): ('standard', 'extreme cold or mild winters change heating demand and the storage draw for natural gas and heating oil'),
    ('us_cooling_demand', 'energy'): ('standard', 'extreme summer heat raises power-sector gas burn and cooling load'),
    ('iod_state', 'agriculture'): ('weak', 'the Indian Ocean Dipole controls Indian monsoon and Australian rainfall (wheat, sugar, rice, cotton)'),
    ('atlantic_hurricane', 'energy'): ('standard', 'Gulf of Mexico storms shut in oil and gas production, refining and pipelines'),
    ('atlantic_hurricane', 'agriculture'): ('weak', 'Gulf and Florida storms damage orange juice, cotton and sugar crops'),
}
_orig.SUBCLASS.update(EXTREME_SUBCLASS)
for (_k, (_conf, _why)) in EXTREME_CHANNELS.items():
    _orig.ADMISSIBLE_CHANNELS[_k] = _why; _orig.ADMISSIBLE_KEYS.add(_k)
    if _conf != 'standard': _orig.CONFIDENCE_OVERRIDES[_k] = _conf
# ---- BATCH 3: further slow climate states (registered 2026-10-11, before any result) ----
SLOW_SUBCLASS = {}
for _n, _sub in (('NINO12', 'enso_state'), ('NINO3', 'enso_state'), ('NINO4', 'enso_state'), ('MEI2', 'enso_state'), ('AAO', 'southern_annular'),
                 ('TNA', 'atlantic_sst'), ('TSA', 'atlantic_sst'), ('WHWP', 'atlantic_sst'), ('PNA', 'pna_state')):
    for _form in ('chg', 'mean'):
        SLOW_SUBCLASS[f'{_n}_{_form}'] = _sub
SLOW_CHANNELS = {   # enso_state channels (all targets) already registered in batch 1 and apply to the Nino 1+2 / 3 / 4 and MEI v2 series
    ('southern_annular', 'agriculture'): ('weak', 'the Southern Annular Mode shifts winter and spring rainfall over southern Australia, Argentina, southern Brazil and South Africa (wheat, corn, soybeans, sugar)'),
    ('atlantic_sst', 'energy'): ('standard', 'warm tropical Atlantic and Western Hemisphere warm pool water raise Atlantic hurricane activity and Gulf of Mexico disruption'),
    ('atlantic_sst', 'agriculture'): ('weak', 'tropical Atlantic SST sets Sahel, West African and north-east Brazil rainfall and Midwest summer moisture (cocoa, coffee, sugar, corn)'),
    ('pna_state', 'energy'): ('standard', 'the Pacific-North American pattern sets US winter temperature, hence heating demand for natural gas and heating oil'),
    ('pna_state', 'agriculture'): ('weak', 'PNA phase alters western US winter precipitation and snowpack (winter wheat, cattle feed)'),
}
_orig.SUBCLASS.update(SLOW_SUBCLASS)
for (_k, (_conf, _why)) in SLOW_CHANNELS.items():
    _orig.ADMISSIBLE_CHANNELS[_k] = _why; _orig.ADMISSIBLE_KEYS.add(_k)
    if _conf != 'standard': _orig.CONFIDENCE_OVERRIDES[_k] = _conf
# ---- BATCH 4: rare slow extreme climate states (registered 2026-10-11, before any result) ----
X4_SUBCLASS = {}
for _n, _sub in (('HRWPDSI', 'drought_wheat_belt'), ('NPLPDSI', 'drought_wheat_belt'), ('DLTPDSI', 'drought_south'), ('SEPDSI', 'drought_south'), ('TXPDSI', 'drought_south'),
                 ('WSTPDSI', 'drought_west'), ('TNI', 'pacific_modes'), ('NPGO', 'pacific_modes'), ('GLBT', 'global_temperature'), ('SNOW', 'cryosphere'), ('ICE', 'cryosphere')):
    for _form in ('chg', 'mean'):
        X4_SUBCLASS[f'{_n}_{_form}'] = _sub
X4_CHANNELS = {
    ('drought_wheat_belt', 'agriculture'): ('standard', 'drought in the hard-red-winter (KS, OK, NE, CO, TX) and northern-plains spring-wheat belts cuts wheat yields and lifts wheat and feed-grain prices (1934, 1956, 2011-12)'),
    ('drought_south', 'agriculture'): ('standard', 'Delta, Southeast and Texas drought cuts cotton, rice, soybean, citrus and peanut yields and forces cattle herd liquidation'),
    ('drought_south', 'energy'): ('weak', 'Texas and Gulf drought and heat raise power burn and cooling demand'),
    ('drought_west', 'energy'): ('weak', 'Western drought cuts hydropower and snowpack, raising gas-fired generation'),
    ('drought_west', 'agriculture'): ('weak', 'Western drought and water limits cut California and Pacific Northwest specialty-crop, hay and wheat output'),
    ('pacific_modes', 'agriculture'): ('weak', 'the Trans-Nino index and North Pacific Gyre Oscillation alter North American, Australian and Asian rainfall regimes on seasonal-to-decadal scales'),
    ('pacific_modes', 'energy'): ('weak', 'North Pacific states shift winter temperature and storm tracks over North America'),
    ('global_temperature', 'agriculture'): ('weak', 'record global warmth coincides with heat, drought and flood extremes in major producers'),
    ('global_temperature', 'energy'): ('weak', 'record global warmth shifts heating and cooling demand and hurricane intensity'),
    ('cryosphere', 'energy'): ('weak', 'low Arctic sea ice and Eurasian snow cover are linked to winter cold outbreaks and polar-vortex disruption (contested)'),
    ('cryosphere', 'agriculture'): ('weak', 'snow cover sets spring soil moisture and planting conditions in northern-hemisphere grain belts'),
}
_orig.SUBCLASS.update(X4_SUBCLASS)
for (_k, (_conf, _why)) in X4_CHANNELS.items():
    _orig.ADMISSIBLE_CHANNELS[_k] = _why; _orig.ADMISSIBLE_KEYS.add(_k)
    if _conf != 'standard': _orig.CONFIDENCE_OVERRIDES[_k] = _conf
# ---- BATCH W: short-horizon WEATHER predictors, the comparison arm (registered 2026-10-11, before any result) ----
WX_SUBCLASS = {}
for _n, _sub in (('WXHDDNAT', 'weather_heating'), ('WXHDDCOLD', 'weather_heating'), ('WXCDDNAT', 'weather_cooling'), ('WXCDDSOUTH', 'weather_cooling'), ('WXCDDTX', 'weather_cooling'), ('WXCDDMW', 'weather_cooling')):
    WX_SUBCLASS[f'{_n}_mean'] = _sub
WX_CHANNELS = {   # deliberately broad: every class with a plausible weather-demand or weather-supply link is admitted, so the arm is not under-tested
    ('weather_heating', 'energy'): ('standard', 'cold spells raise heating demand and storage draws for natural gas and heating oil'),
    ('weather_heating', 'energy_single_name'): ('weak', 'energy producers and utilities earn on gas and power prices that follow heating demand'),
    ('weather_heating', 'sector_equity'): ('weak', 'utility and energy sector equities follow weather-driven gas and power demand'),
    ('weather_heating', 'broad_commodity'): ('weak', 'aggregate of the energy exposure'),
    ('weather_heating', 'agriculture'): ('weak', 'winter cold affects wheat survival and livestock feed demand'),
    ('weather_cooling', 'energy'): ('standard', 'heat raises cooling load, power-sector gas burn and power prices'),
    ('weather_cooling', 'energy_single_name'): ('weak', 'energy producers and utilities earn on power prices that follow cooling demand'),
    ('weather_cooling', 'sector_equity'): ('weak', 'utility and energy sector equities follow weather-driven power demand'),
    ('weather_cooling', 'broad_commodity'): ('weak', 'aggregate of the energy and crop exposures'),
    ('weather_cooling', 'agriculture'): ('standard', 'summer heat during pollination and grain fill cuts corn and soybean yields'),
}
_orig.SUBCLASS.update(WX_SUBCLASS)
for (_k, (_conf, _why)) in WX_CHANNELS.items():
    _orig.ADMISSIBLE_CHANNELS[_k] = _why; _orig.ADMISSIBLE_KEYS.add(_k)
    if _conf != 'standard': _orig.CONFIDENCE_OVERRIDES[_k] = _conf
is_admissible = _orig.is_admissible
admissible_predictors_for = _orig.admissible_predictors_for
get_subclass = _orig.get_subclass
explain_pair = _orig.explain_pair
get_confidence = _orig.get_confidence
