"""
Paper 8 — VPD and Moisture Stress Pipeline
===========================================
Extends Paper 7 crop-zone data with humidity and precipitation
to compute Vapour Pressure Deficit (VPD) and moisture stress predictors.

Key new predictors:
1. VPD_mean_q80/q90     — daily VPD exceedance (80th/90th pct)
2. VPD_stress_crop      — VPD above crop-specific threshold during season
3. VPD_30d_q80/q90      — 30-day rolling mean VPD exceedance
4. combined_heatvpd     — heat stress AND VPD stress both active (joint flag)
5. dry_heat             — heat stress active AND precip < 1mm/day

VPD formula (Tetens):
  SVP(T) = 0.6108 * exp(17.27 * T / (T + 237.3))   [kPa]
  VPD    = (1 - RH/100) * SVP(T)                    [kPa]

Crop VPD stress thresholds (Lobell et al. 2014, Hatfield & Prueger 2015):
  Wheat:     VPD > 2.0 kPa
  Corn:      VPD > 2.5 kPa
  Soybeans:  VPD > 2.0 kPa
  Sugar cane:VPD > 1.5 kPa
  Energy:    VPD > 2.0 kPa (human comfort / cooling demand)

Open-Meteo variables (NEW vs Paper 7):
  relative_humidity_2m_mean   — for VPD computation
  precipitation_sum            — for dry heat flag
  et0_fao_evapotranspiration  — evaporative demand (bonus predictor)

Run on your LOCAL machine in notebooks/temperature/
pip install openmeteo-requests requests-cache retry-requests pandas numpy
"""

import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
import os, time

# ── SETUP ─────────────────────────────────────────────────────────────
cache_session = requests_cache.CachedSession('.cache_vpd', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
om = openmeteo_requests.Client(session=retry_session)

START_DATE = "2000-01-01"
END_DATE   = "2026-06-01"
SLEEP_OK   = 5
SLEEP_FAIL = 65

# ── CROP ZONE DEFINITIONS (same bounding boxes as Paper 7) ────────────
# Format: name → {lat_range, lon_range, grid_step, season_months,
#                  vpd_stress_kpa, heatstress_tmax_C, crops}
CROP_ZONES = {
    'EU_Wheat_Belt': {
        'lat_range': (45,55), 'lon_range': (5,30), 'grid_step': 3,
        'season_months': list(range(3,9)),
        'vpd_stress_kpa': 2.0, 'heatstress_tmax_C': 30,
        'crops': ['WEAT','ZW=F'],
    },
    'Ukraine_Russia_Wheat': {
        'lat_range': (45,55), 'lon_range': (25,60), 'grid_step': 5,
        'season_months': list(range(3,9)),
        'vpd_stress_kpa': 2.0, 'heatstress_tmax_C': 30,
        'crops': ['WEAT','ZW=F','GC=F'],
    },
    'US_Great_Plains_Wheat': {
        'lat_range': (35,50), 'lon_range': (-105,-92), 'grid_step': 3,
        'season_months': list(range(4,8)),
        'vpd_stress_kpa': 2.0, 'heatstress_tmax_C': 32,
        'crops': ['WEAT','ZW=F'],
    },
    'US_Corn_Belt': {
        'lat_range': (38,47), 'lon_range': (-97,-82), 'grid_step': 3,
        'season_months': list(range(5,10)),
        'vpd_stress_kpa': 2.5, 'heatstress_tmax_C': 32,
        'crops': ['CORN','ZC=F','SOYB'],
    },
    'Brazil_Corn': {
        'lat_range': (-25,-10), 'lon_range': (-55,-45), 'grid_step': 4,
        'season_months': [10,11,12,1,2,3],
        'vpd_stress_kpa': 2.5, 'heatstress_tmax_C': 35,
        'crops': ['CORN','ZC=F'],
    },
    'US_Soybean_Belt': {
        'lat_range': (38,47), 'lon_range': (-97,-82), 'grid_step': 3,
        'season_months': list(range(6,11)),
        'vpd_stress_kpa': 2.0, 'heatstress_tmax_C': 32,
        'crops': ['SOYB','ZS=F'],
    },
    'Brazil_Soy': {
        'lat_range': (-25,-5), 'lon_range': (-60,-45), 'grid_step': 4,
        'season_months': [10,11,12,1,2,3],
        'vpd_stress_kpa': 2.0, 'heatstress_tmax_C': 35,
        'crops': ['SOYB','ZS=F'],
    },
    'Brazil_Sugar_Sao_Paulo': {
        'lat_range': (-25,-20), 'lon_range': (-52,-44), 'grid_step': 2,
        'season_months': list(range(4,12)),
        'vpd_stress_kpa': 1.5, 'heatstress_tmax_C': 38,
        'crops': ['CANE'],
    },
    'India_Sugar': {
        'lat_range': (15,30), 'lon_range': (73,85), 'grid_step': 3,
        'season_months': [10,11,12,1,2,3],
        'vpd_stress_kpa': 1.5, 'heatstress_tmax_C': 38,
        'crops': ['CANE'],
    },
    'Thailand_Sugar': {
        'lat_range': (13,18), 'lon_range': (98,103), 'grid_step': 2,
        'season_months': [11,12,1,2,3,4],
        'vpd_stress_kpa': 1.5, 'heatstress_tmax_C': 38,
        'crops': ['CANE'],
    },
    'EU_Urban_Energy': {
        'points': [(48.86,2.35),(51.51,-0.13),(50.11,8.68),(52.52,13.40),(48.21,16.37)],
        'season_months': list(range(1,13)),
        'vpd_stress_kpa': 2.0, 'heatstress_tmax_C': 30,
        'crops': ['NG=F','UNG','XLU','XLE'],
    },
    'US_Urban_Energy': {
        'points': [(40.71,-74.01),(41.88,-87.63),(29.76,-95.37),(34.05,-118.24),(33.45,-84.39)],
        'season_months': list(range(1,13)),
        'vpd_stress_kpa': 2.0, 'heatstress_tmax_C': 32,
        'crops': ['NG=F','UNG','XLU'],
    },
}

# ── VPD COMPUTATION ───────────────────────────────────────────────────
def compute_vpd(tmax_c, rh_mean_pct):
    """
    Vapour Pressure Deficit using Tetens formula.
    Uses tmax (°C) and mean relative humidity (%).
    Returns VPD in kPa.
    """
    svp = 0.6108 * np.exp(17.27 * tmax_c / (tmax_c + 237.3))
    vpd = (1 - rh_mean_pct / 100.0) * svp
    return vpd.clip(lower=0)

# ── GRID POINT GENERATOR ──────────────────────────────────────────────
def get_grid_points(lat_range, lon_range, step):
    lats = np.arange(lat_range[0], lat_range[1]+1, step)
    lons = np.arange(lon_range[0], lon_range[1]+1, step)
    return [(float(la), float(lo)) for la in lats for lo in lons]

# ── FETCH ONE POINT (temperature + humidity + precip) ─────────────────
def fetch_point(lat, lon):
    """Fetch tmax, tavg, RH, precip, ET0 for one point."""
    url    = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "et0_fao_evapotranspiration",
        ],
        "timezone": "UTC",
    }
    for attempt in range(1, 6):
        try:
            responses = om.weather_api(url, params=params)
            r     = responses[0]
            daily = r.Daily()
            df = pd.DataFrame({
                'tmax':   daily.Variables(0).ValuesAsNumpy(),
                'tavg':   daily.Variables(1).ValuesAsNumpy(),
                'rh':     daily.Variables(2).ValuesAsNumpy(),
                'precip': daily.Variables(3).ValuesAsNumpy(),
                'et0':    daily.Variables(4).ValuesAsNumpy(),
            }, index=pd.date_range(
                start=pd.to_datetime(daily.Time(),    unit="s", utc=True),
                end=  pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left",
            ))
            df.index = df.index.tz_localize(None)
            df.index.name = 'date'
            time.sleep(SLEEP_OK)
            return df
        except Exception as e:
            err = str(e)
            if 'rate' in err.lower() or 'limit' in err.lower():
                print(f"    Rate limit (attempt {attempt}/5) — waiting {SLEEP_FAIL}s...")
                time.sleep(SLEEP_FAIL)
            else:
                print(f"    Error at ({lat},{lon}): {err}")
                return None
    return None

# ── FETCH ZONE ────────────────────────────────────────────────────────
def fetch_zone(zone_name, zone_config):
    print(f"\n  {zone_name}: {zone_config.get('crops',[])}")

    if 'points' in zone_config:
        points = zone_config['points']
    else:
        points = get_grid_points(
            zone_config['lat_range'], zone_config['lon_range'],
            zone_config['grid_step'])
    print(f"  Fetching {len(points)} grid points...")

    cols = {k: [] for k in ['tmax','tavg','rh','precip','et0']}
    n_ok = 0
    for i, (lat, lon) in enumerate(points):
        df = fetch_point(lat, lon)
        if df is not None:
            for k in cols:
                cols[k].append(df[k])
            n_ok += 1
            print(f"    ({lat:.1f},{lon:.1f}) ✓  [{i+1}/{len(points)}]")
        else:
            print(f"    ({lat:.1f},{lon:.1f}) ✗  [{i+1}/{len(points)}]")

    if n_ok == 0:
        print(f"  {zone_name}: NO DATA")
        return None

    result = pd.DataFrame({
        'tmax':   pd.concat(cols['tmax'],  axis=1).mean(axis=1),
        'tavg':   pd.concat(cols['tavg'],  axis=1).mean(axis=1),
        'rh':     pd.concat(cols['rh'],    axis=1).mean(axis=1),
        'precip': pd.concat(cols['precip'],axis=1).mean(axis=1),
        'et0':    pd.concat(cols['et0'],   axis=1).mean(axis=1),
    })
    result.index.name = 'date'
    print(f"  {zone_name}: {len(result)} days from {n_ok}/{len(points)} points")
    return result

# ── COMPUTE VPD EXCEEDANCES ────────────────────────────────────────────
def compute_vpd_exceedances(zone_data, zone_name, zone_config,
                             train_end='2024-12-31'):
    results = {}
    df      = zone_data.dropna(subset=['tmax','rh'])
    train   = df[df.index <= train_end]
    season  = zone_config['season_months']
    season_mask = df.index.month.isin(season)
    vpd_thr = zone_config['vpd_stress_kpa']
    hs_thr  = zone_config['heatstress_tmax_C']

    # ── 1. Compute daily VPD ─────────────────────────────────────────
    df['vpd'] = compute_vpd(df['tmax'], df['rh'])
    train['vpd'] = compute_vpd(train['tmax'], train['rh'])

    print(f"\n    {zone_name} VPD stats:")
    print(f"      mean={df['vpd'].mean():.2f} kPa, "
          f"q80={train['vpd'].quantile(0.80):.2f}, "
          f"q90={train['vpd'].quantile(0.90):.2f}, "
          f"stress threshold={vpd_thr:.1f} kPa")

    # ── 2. VPD percentile exceedance (all-year) ───────────────────────
    for q in [0.80, 0.90]:
        thr = train['vpd'].quantile(q)
        key = f"{zone_name}_VPD_q{int(q*100)}"
        results[key] = (df['vpd'] > thr).astype(float)
        print(f"      {key}: {results[key].sum():.0f} days, thr={thr:.2f} kPa")

    # ── 3. VPD seasonal exceedance ───────────────────────────────────
    train_season = train[train.index.month.isin(season)]
    if len(train_season) > 100:
        for q in [0.80, 0.90]:
            thr = train_season['vpd'].quantile(q)
            key = f"{zone_name}_VPD_seasonal_q{int(q*100)}"
            flag = ((df['vpd'] > thr) & season_mask).astype(float)
            results[key] = flag
            print(f"      {key}: {flag.sum():.0f} seasonal days")

    # ── 4. VPD crop-specific stress threshold ────────────────────────
    key = f"{zone_name}_VPD_stress_{str(vpd_thr).replace('.','p')}kPa"
    flag = ((df['vpd'] > vpd_thr) & season_mask).astype(float)
    results[key] = flag
    print(f"      {key}: {flag.sum():.0f} stress days (>{vpd_thr} kPa)")

    # ── 5. 30-day rolling VPD exceedance ─────────────────────────────
    vpd_30d = df['vpd'].rolling(30, min_periods=20).mean()
    train_30d = vpd_30d[vpd_30d.index <= train_end].dropna()
    if len(train_30d) > 100:
        for q in [0.80, 0.90]:
            thr = train_30d.quantile(q)
            key = f"{zone_name}_VPD_30d_q{int(q*100)}"
            results[key] = (vpd_30d > thr).astype(float)
            print(f"      {key}: {results[key].sum():.0f} high-VPD days")

    # ── 6. Combined heat + VPD stress (joint flag) ───────────────────
    heat_flag = ((df['tmax'] > hs_thr) & season_mask)
    vpd_flag  = ((df['vpd']  > vpd_thr) & season_mask)
    key = f"{zone_name}_combined_heat_vpd"
    results[key] = (heat_flag & vpd_flag).astype(float)
    print(f"      {key}: {results[key].sum():.0f} combined stress days")

    # ── 7. Dry heat flag (heat stress + low precip) ───────────────────
    dry_flag = (df['precip'] < 1.0)  # less than 1mm/day
    key = f"{zone_name}_dry_heat"
    results[key] = (heat_flag & dry_flag).astype(float)
    print(f"      {key}: {results[key].sum():.0f} dry heat days")

    # ── 8. ET0 exceedance (evaporative demand) ───────────────────────
    if 'et0' in df.columns and df['et0'].notna().sum() > 100:
        train_et0 = df.loc[df.index <= train_end, 'et0'].dropna()
        if len(train_et0) > 100:
            for q in [0.80, 0.90]:
                thr = train_et0.quantile(q)
                key = f"{zone_name}_ET0_q{int(q*100)}"
                results[key] = (df['et0'] > thr).astype(float)
                print(f"      {key}: {results[key].sum():.0f} high-ET0 days")

    return pd.DataFrame(results)

# ── ALIGN TO FINANCIAL CALENDAR ───────────────────────────────────────
def align_to_financial(exceedance_df, prices_path='multiasset_prices.parquet'):
    try:
        prices  = pd.read_parquet(prices_path)
        aligned = exceedance_df.reindex(prices.index, method='ffill')
        aligned.to_parquet('data/paper8_vpd_exceedances_aligned.parquet')
        print(f"\nAligned to financial calendar: {aligned.shape}")
        return aligned
    except FileNotFoundError:
        exceedance_df.to_parquet('data/paper8_vpd_exceedances_raw.parquet')
        print(f"\nSaved raw (no prices file): {exceedance_df.shape}")
        return exceedance_df

# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("PAPER 8 — VPD AND MOISTURE STRESS PIPELINE")
    print("=" * 70)
    print(f"Zones: {len(CROP_ZONES)}")
    print(f"New variables: RH, precipitation, ET0 → VPD, dry heat, combined stress")
    print(f"Date range: {START_DATE} to {END_DATE}")

    os.makedirs('data', exist_ok=True)

    all_exceedances = []

    for zone_name, zone_config in CROP_ZONES.items():
        print(f"\n{'='*50}")
        print(f"Zone: {zone_name}")

        cache_path = f'data/vpd_{zone_name}.parquet'
        if os.path.exists(cache_path):
            print(f"  Loading from cache: {cache_path}")
            zone_df = pd.read_parquet(cache_path)
            # check it has the new columns
            if 'rh' not in zone_df.columns:
                print(f"  Cache missing RH — re-fetching...")
                zone_df = fetch_zone(zone_name, zone_config)
                if zone_df is not None:
                    zone_df.to_parquet(cache_path)
        else:
            zone_df = fetch_zone(zone_name, zone_config)
            if zone_df is not None:
                zone_df.to_parquet(cache_path)

        if zone_df is None:
            print(f"  SKIP: {zone_name}")
            continue

        print(f"  Computing VPD exceedances...")
        exc_df = compute_vpd_exceedances(zone_df, zone_name, zone_config)
        all_exceedances.append(exc_df)

    if all_exceedances:
        combined = pd.concat(all_exceedances, axis=1)
        combined.to_parquet('data/paper8_vpd_exceedances.parquet')
        print(f"\n{'='*70}")
        print(f"DONE — Total VPD predictors: {combined.shape[1]}")
        print(f"Predictor names:")
        for col in combined.columns:
            print(f"  {col}")
        print(f"\nAligning to financial calendar...")
        align_to_financial(combined)
        print(f"\nNext: run paper8_cpe_analysis.py")
    else:
        print("\nNo zone data loaded.")
