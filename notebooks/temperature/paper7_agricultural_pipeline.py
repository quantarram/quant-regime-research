"""
Paper 7 — Agricultural Crop-Zone Temperature Pipeline
======================================================
Replaces city-temperature proxies with proper crop-zone ERA5
gridded temperature averages for agricultural instruments.

Key improvement over Paper 6:
- Wheat: European Plain, Black Sea belt, US Great Plains
- Corn: US Corn Belt, Brazil
- Soybeans: US Soybean Belt, Brazil
- Sugar: Brazil São Paulo, India Maharashtra, Thailand
- Energy: keeps city temperatures (correct for energy demand)

Also adds:
- Growing Degree Days (GDD) — more relevant than raw temperature
- Seasonal conditioning — only fires during crop growing season
- Heat stress flags — above critical thresholds (e.g. 32°C for corn)

Data source: Open-Meteo archive API (ERA5-consistent, free, no auth)
Grid averaging: spatial mean over bounding box

Run on your LOCAL machine.
pip install openmeteo-requests requests-cache retry-requests pandas numpy
"""

import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
import os
import time

# ── SETUP ─────────────────────────────────────────────────────────────
cache_session = requests_cache.CachedSession('.cache_agri', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
om = openmeteo_requests.Client(session=retry_session)

START_DATE = "2000-01-01"
END_DATE   = "2026-06-01"
SLEEP_OK   = 5    # seconds between successful fetches
SLEEP_FAIL = 65   # seconds after rate limit

# ── CROP ZONE DEFINITIONS ─────────────────────────────────────────────
# For each zone: sample a grid of points within the bounding box
# and average their temperatures
# Grid resolution: ~2 degree spacing (coarse but representative)

CROP_ZONES = {
    # ── WHEAT ─────────────────────────────────────────────────────────
    'EU_Wheat_Belt': {
        'desc': 'European Plain wheat belt (France/Germany/Poland)',
        'lat_range': (45, 55), 'lon_range': (5, 30),
        'grid_step': 3,   # degrees between sample points
        'crops': ['WEAT', 'ZW=F'],
        'season_months': list(range(3, 9)),   # Mar-Aug
        'stress_tmax_C': 30,                  # above = heat stress
        'type': 'agricultural',
    },
    'Ukraine_Russia_Wheat': {
        'desc': 'Black Sea wheat belt (Ukraine/Russia)',
        'lat_range': (45, 55), 'lon_range': (25, 60),
        'grid_step': 5,
        'crops': ['WEAT', 'ZW=F'],
        'season_months': list(range(3, 9)),
        'stress_tmax_C': 30,
        'type': 'agricultural',
    },
    'US_Great_Plains_Wheat': {
        'desc': 'US Great Plains wheat (Kansas/Oklahoma/Montana)',
        'lat_range': (35, 50), 'lon_range': (-105, -92),
        'grid_step': 3,
        'crops': ['WEAT', 'ZW=F'],
        'season_months': list(range(4, 8)),   # Apr-Jul
        'stress_tmax_C': 32,
        'type': 'agricultural',
    },

    # ── CORN ──────────────────────────────────────────────────────────
    'US_Corn_Belt': {
        'desc': 'US Corn Belt (Iowa/Illinois/Indiana/Ohio/Nebraska)',
        'lat_range': (38, 47), 'lon_range': (-97, -82),
        'grid_step': 3,
        'crops': ['CORN', 'ZC=F'],
        'season_months': list(range(5, 10)),  # May-Sep
        'stress_tmax_C': 32,                  # pollination stress threshold
        'type': 'agricultural',
    },
    'Brazil_Corn': {
        'desc': 'Brazil corn belt (Mato Grosso/Paraná)',
        'lat_range': (-25, -10), 'lon_range': (-55, -45),
        'grid_step': 4,
        'crops': ['CORN', 'ZC=F'],
        'season_months': [10, 11, 12, 1, 2, 3],  # Oct-Mar (S. hemisphere)
        'stress_tmax_C': 35,
        'type': 'agricultural',
    },

    # ── SOYBEANS ──────────────────────────────────────────────────────
    'US_Soybean_Belt': {
        'desc': 'US Soybean Belt (Iowa/Illinois/Indiana)',
        'lat_range': (38, 47), 'lon_range': (-97, -82),
        'grid_step': 3,
        'crops': ['SOYB', 'ZS=F'],
        'season_months': list(range(6, 11)),  # Jun-Oct
        'stress_tmax_C': 32,
        'type': 'agricultural',
    },
    'Brazil_Soy': {
        'desc': 'Brazil soy belt (Mato Grosso/Paraná/Goiás)',
        'lat_range': (-25, -5), 'lon_range': (-60, -45),
        'grid_step': 4,
        'crops': ['SOYB', 'ZS=F'],
        'season_months': [10, 11, 12, 1, 2, 3],
        'stress_tmax_C': 35,
        'type': 'agricultural',
    },

    # ── SUGAR ─────────────────────────────────────────────────────────
    'Brazil_Sugar_Sao_Paulo': {
        'desc': 'São Paulo state — 40% of global sugar cane production',
        'lat_range': (-25, -20), 'lon_range': (-52, -44),
        'grid_step': 2,
        'crops': ['CANE'],
        'season_months': list(range(4, 12)),  # Apr-Nov harvest
        'stress_tmax_C': 38,
        'type': 'agricultural',
    },
    'India_Sugar': {
        'desc': 'Maharashtra/Uttar Pradesh sugar belt',
        'lat_range': (15, 30), 'lon_range': (73, 85),
        'grid_step': 3,
        'crops': ['CANE'],
        'season_months': [10, 11, 12, 1, 2, 3],  # Oct-Mar harvest
        'stress_tmax_C': 38,
        'type': 'agricultural',
    },
    'Thailand_Sugar': {
        'desc': 'Central Thailand sugar region',
        'lat_range': (13, 18), 'lon_range': (98, 103),
        'grid_step': 2,
        'crops': ['CANE'],
        'season_months': [11, 12, 1, 2, 3, 4],
        'stress_tmax_C': 38,
        'type': 'agricultural',
    },

    # ── ENERGY (cities — correct for demand signals) ───────────────────
    'EU_Urban_Energy': {
        'desc': 'Major European urban centres (energy demand)',
        'points': [  # use specific city points, not grid
            (48.86, 2.35),   # Paris
            (51.51, -0.13),  # London
            (50.11,  8.68),  # Frankfurt
            (52.52, 13.40),  # Berlin
            (48.21, 16.37),  # Vienna
        ],
        'crops': ['NG=F', 'UNG', 'XLU', 'XLE', 'ICLN'],
        'season_months': list(range(1, 13)),  # all year (heating + cooling)
        'stress_tmax_C': 30,                  # cooling demand trigger
        'type': 'energy',
    },
    'US_Urban_Energy': {
        'desc': 'US population centres (energy demand)',
        'points': [
            (40.71, -74.01),  # New York
            (41.88, -87.63),  # Chicago
            (29.76, -95.37),  # Houston
            (34.05,-118.24),  # Los Angeles
            (33.45, -84.39),  # Atlanta
        ],
        'crops': ['NG=F', 'UNG', 'XLU', 'XLE'],
        'season_months': list(range(1, 13)),
        'stress_tmax_C': 32,
        'type': 'energy',
    },
}

# ── GRID POINT GENERATOR ──────────────────────────────────────────────
def get_grid_points(lat_range, lon_range, step):
    """Generate a regular grid of sample points within a bounding box."""
    lats = np.arange(lat_range[0], lat_range[1]+1, step)
    lons = np.arange(lon_range[0], lon_range[1]+1, step)
    points = [(float(lat), float(lon)) for lat in lats for lon in lons]
    return points

# ── FETCH ONE POINT ───────────────────────────────────────────────────
def fetch_point_temp(lat, lon, label=""):
    """Fetch daily tmax and tavg for one lat/lon point."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "daily": ["temperature_2m_max", "temperature_2m_mean"],
        "timezone": "UTC",
    }
    for attempt in range(1, 6):
        try:
            responses = om.weather_api(url, params=params)
            response  = responses[0]
            daily     = response.Daily()
            df = pd.DataFrame({
                'tmax': daily.Variables(0).ValuesAsNumpy(),
                'tavg': daily.Variables(1).ValuesAsNumpy(),
            }, index=pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
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
    """Fetch all grid points for a zone and return spatial average."""
    print(f"\n  {zone_name}: {zone_config['desc']}")

    # Get points
    if 'points' in zone_config:
        points = zone_config['points']
    else:
        points = get_grid_points(
            zone_config['lat_range'],
            zone_config['lon_range'],
            zone_config['grid_step']
        )
    print(f"  Fetching {len(points)} grid points...")

    tmax_series = []
    tavg_series = []

    for i, (lat, lon) in enumerate(points):
        df = fetch_point_temp(lat, lon)
        if df is not None:
            tmax_series.append(df['tmax'])
            tavg_series.append(df['tavg'])
            print(f"    ({lat:.1f},{lon:.1f}) ✓  [{i+1}/{len(points)}]")
        else:
            print(f"    ({lat:.1f},{lon:.1f}) ✗  [{i+1}/{len(points)}]")

    if not tmax_series:
        print(f"  {zone_name}: NO DATA")
        return None

    # Spatial average
    zone_tmax = pd.concat(tmax_series, axis=1).mean(axis=1)
    zone_tavg = pd.concat(tavg_series, axis=1).mean(axis=1)

    result = pd.DataFrame({'tmax': zone_tmax, 'tavg': zone_tavg})
    result.index.name = 'date'

    print(f"  {zone_name}: {len(result)} days from {len(tmax_series)}/{len(points)} points")
    return result

# ── COMPUTE AGRICULTURAL EXCEEDANCES ──────────────────────────────────
def compute_agri_exceedances(zone_data, zone_name, zone_config,
                              train_end='2024-12-31'):
    """
    Compute three types of exceedance flags:

    1. Standard: tmax > historical 90th pct (same as Paper 6)
    2. Seasonal: same but only during growing season
    3. Heat stress: tmax > crop-specific stress threshold during season
    4. Growing Degree Days: cumulative GDD above baseline during season
    """
    results = {}
    df = zone_data.dropna()
    train = df[df.index <= train_end]

    # 1. Standard tmax exceedance (Paper 6 equivalent)
    for q in [0.80, 0.90, 0.95]:
        thr = train['tmax'].quantile(q)
        key = f"{zone_name}_tmax_q{int(q*100)}"
        results[key] = (df['tmax'] > thr).astype(float)
        n = results[key].sum()
        print(f"    {key}: {n:.0f} exceedance days, threshold={thr:.1f}°C")

    # 2. Seasonal exceedance — only fires during growing season
    season = zone_config.get('season_months', list(range(1,13)))
    season_mask = df.index.month.isin(season)

    for q in [0.80, 0.90]:
        train_season = train[train.index.month.isin(season)]
        if len(train_season) < 100:
            continue
        thr = train_season['tmax'].quantile(q)
        key = f"{zone_name}_seasonal_q{int(q*100)}"
        flag = ((df['tmax'] > thr) & season_mask).astype(float)
        results[key] = flag
        print(f"    {key}: {flag.sum():.0f} seasonal exceedance days")

    # 3. Heat stress flag — above crop-specific threshold during season
    stress_thr = zone_config.get('stress_tmax_C', 32)
    key = f"{zone_name}_heatstress_{stress_thr}C"
    flag = ((df['tmax'] > stress_thr) & season_mask).astype(float)
    results[key] = flag
    print(f"    {key}: {flag.sum():.0f} heat stress days (>{stress_thr}°C)")

    # 4. Growing Degree Days (GDD) exceedance
    # GDD = max(0, tavg - base_temp), accumulated over rolling 30-day window
    base_temp = 10  # standard base for most crops
    daily_gdd = (df['tavg'] - base_temp).clip(lower=0)
    gdd_30d = daily_gdd.rolling(30, min_periods=20).sum()
    train_gdd = gdd_30d[gdd_30d.index <= train_end].dropna()
    if len(train_gdd) > 100:
        for q in [0.80, 0.90]:
            thr = train_gdd.quantile(q)
            key = f"{zone_name}_GDD30_q{int(q*100)}"
            results[key] = (gdd_30d > thr).astype(float)
            print(f"    {key}: {results[key].sum():.0f} high-GDD days")

    return pd.DataFrame(results)

# ── ALIGN TO FINANCIAL CALENDAR ───────────────────────────────────────
def align_to_financial(exceedance_df, prices_path='multiasset_prices.parquet'):
    try:
        prices = pd.read_parquet(prices_path)
        aligned = exceedance_df.reindex(prices.index, method='ffill')
        aligned.to_parquet('data/paper7_agri_exceedances_aligned.parquet')
        print(f"\nAligned to financial calendar: {aligned.shape}")
        print(f"Columns: {aligned.columns.tolist()}")
        return aligned
    except FileNotFoundError:
        print(f"\nCould not find {prices_path}")
        exceedance_df.to_parquet('data/paper7_agri_exceedances_raw.parquet')
        return exceedance_df

# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("PAPER 7 — AGRICULTURAL CROP-ZONE TEMPERATURE PIPELINE")
    print("=" * 70)
    print(f"Zones: {len(CROP_ZONES)}")
    print(f"Date range: {START_DATE} to {END_DATE}")

    os.makedirs('data', exist_ok=True)

    all_zone_data   = {}
    all_exceedances = []

    for zone_name, zone_config in CROP_ZONES.items():
        print(f"\n{'='*50}")
        print(f"Zone: {zone_name}")

        # Check if already cached
        cache_path = f'data/zone_{zone_name}.parquet'
        if os.path.exists(cache_path):
            print(f"  Loading from cache: {cache_path}")
            zone_df = pd.read_parquet(cache_path)
        else:
            zone_df = fetch_zone(zone_name, zone_config)
            if zone_df is not None:
                zone_df.to_parquet(cache_path)
                print(f"  Saved to cache: {cache_path}")

        if zone_df is None:
            print(f"  SKIP: {zone_name} — no data")
            continue

        all_zone_data[zone_name] = zone_df

        # Compute exceedances
        print(f"\n  Computing exceedances for {zone_name}...")
        exc_df = compute_agri_exceedances(zone_df, zone_name, zone_config)
        all_exceedances.append(exc_df)

    if all_exceedances:
        # Combine all exceedance flags
        combined = pd.concat(all_exceedances, axis=1)
        combined.to_parquet('data/paper7_agri_exceedances.parquet')
        print(f"\nTotal exceedance predictors: {combined.shape[1]}")
        print(f"Predictor names:")
        for col in combined.columns:
            print(f"  {col}")

        # Align to financial calendar
        print("\nAligning to financial calendar...")
        aligned = align_to_financial(combined)

        print("\n" + "="*70)
        print("DONE — Ready for CPE analysis")
        print("="*70)
        print(f"Next: run paper7_cpe_analysis.py")
        print(f"Input: data/paper7_agri_exceedances_aligned.parquet")
        print(f"Predictors: {combined.shape[1]} agricultural+energy exceedance flags")
        print(f"\nKey improvement over Paper 6:")
        print(f"  - Wheat: EU Plain, Black Sea, US Great Plains")
        print(f"  - Corn: US Corn Belt, Brazil")
        print(f"  - Soy: US Soybean Belt, Brazil")
        print(f"  - Sugar: Brazil São Paulo, India, Thailand")
        print(f"  - Energy: City temperatures (same as Paper 6 — correct)")
        print(f"  - New: seasonal flags, heat stress flags, GDD flags")
    else:
        print("\nNo zone data loaded. Check network connection.")
