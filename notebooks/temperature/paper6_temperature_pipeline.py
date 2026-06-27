"""
Paper 6 — Temperature Data Pipeline v3
Fetches daily temperature data for key cities using Open-Meteo (free, no API key).
Runs in a single pass — retries failed cities automatically with backoff.

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
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
om = openmeteo_requests.Client(session=retry_session)

START_DATE = "2000-01-01"
END_DATE   = "2026-06-01"

SLEEP_BETWEEN_CITIES  = 5   # seconds between each city fetch
SLEEP_ON_FAIL         = 65  # seconds to wait after a rate-limit failure (just over 1 minute)
MAX_RETRIES_PER_CITY  = 5   # how many times to retry a failed city before giving up

# ── CITY DEFINITIONS ──────────────────────────────────────────────────
CITIES = {
    'Europe': [
        ('Paris',       48.8566,   2.3522),
        ('London',      51.5074,  -0.1278),
        ('Frankfurt',   50.1109,   8.6821),
        ('Madrid',      40.4168,  -3.7038),
        ('Rome',        41.9028,  12.4964),
        ('Berlin',      52.5200,  13.4050),
    ],
    'North_America': [
        ('New_York',    40.7128, -74.0060),
        ('Chicago',     41.8781, -87.6298),
        ('Houston',     29.7604, -95.3698),
        ('Los_Angeles', 34.0522,-118.2437),
    ],
    'Asia': [
        ('Tokyo',       35.6762, 139.6503),
        ('Mumbai',      19.0760,  72.8777),
        ('Singapore',    1.3521, 103.8198),
        ('Beijing',     39.9042, 116.4074),
        ('Shanghai',    31.2304, 121.4737),
    ],
}

# ── FETCH WITH RETRY ──────────────────────────────────────────────────
def fetch_city_temp(city_name, lat, lon):
    """
    Fetch daily temperature from Open-Meteo.
    Retries up to MAX_RETRIES_PER_CITY times on rate-limit errors,
    waiting SLEEP_ON_FAIL seconds between attempts.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "daily":      ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"],
        "timezone":   "UTC",
    }

    for attempt in range(1, MAX_RETRIES_PER_CITY + 1):
        try:
            responses = om.weather_api(url, params=params)
            response  = responses[0]
            daily     = response.Daily()

            df = pd.DataFrame({
                'tmax': daily.Variables(0).ValuesAsNumpy(),
                'tmin': daily.Variables(1).ValuesAsNumpy(),
                'tavg': daily.Variables(2).ValuesAsNumpy(),
            }, index=pd.date_range(
                start=pd.to_datetime(daily.Time(),    unit="s", utc=True),
                end=  pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left",
            ))
            df.index = df.index.tz_localize(None)
            df.index.name = 'date'

            print(f"  ✓ {city_name}: {len(df)} days "
                  f"({df.index[0].date()} → {df.index[-1].date()})")
            time.sleep(SLEEP_BETWEEN_CITIES)
            return df

        except Exception as e:
            err = str(e)
            if 'rate' in err.lower() or 'limit' in err.lower():
                print(f"  ⚠ {city_name}: rate limit (attempt {attempt}/{MAX_RETRIES_PER_CITY}) "
                      f"— waiting {SLEEP_ON_FAIL}s...")
                time.sleep(SLEEP_ON_FAIL)
            else:
                print(f"  ✗ {city_name}: unexpected error — {err}")
                return None  # don't retry non-rate-limit errors

    print(f"  ✗ {city_name}: gave up after {MAX_RETRIES_PER_CITY} attempts")
    return None


# ── BUILD DATASET ─────────────────────────────────────────────────────
def build_temperature_dataset():
    os.makedirs('data', exist_ok=True)
    all_city_data  = {}
    regional_indices = {}

    for region, cities in CITIES.items():
        print(f"\n── {region} ──────────────────────────────")
        region_tavg, region_tmax = [], []

        for city_name, lat, lon in cities:
            df = fetch_city_temp(city_name, lat, lon)
            if df is not None:
                all_city_data[city_name] = df
                region_tavg.append(df['tavg'])
                region_tmax.append(df['tmax'])

        if region_tavg:
            regional_indices[f'{region}_tavg'] = pd.concat(region_tavg, axis=1).mean(axis=1)
            regional_indices[f'{region}_tmax'] = pd.concat(region_tmax, axis=1).mean(axis=1)
            print(f"  → {region} index: {len(region_tavg)}/{len(cities)} cities")
        else:
            print(f"  ✗ {region}: no cities loaded — index skipped")

    # Save city parquets
    for city, df in all_city_data.items():
        df.to_parquet(f'data/temp_{city.lower()}.parquet')
    print(f"\nSaved {len(all_city_data)}/15 city files → data/")

    # Save regional indices
    regional_df = pd.DataFrame(regional_indices)
    regional_df.to_parquet('data/temperature_regional_indices.parquet')
    print(f"Saved regional indices: {regional_df.shape}")
    print(regional_df.tail(3).to_string())

    return regional_df, all_city_data


# ── COMPUTE EXCEEDANCES ───────────────────────────────────────────────
def compute_temperature_exceedances(regional_df, quantiles=[0.80, 0.90, 0.95]):
    results = {}
    for col in regional_df.columns:
        series   = regional_df[col].dropna()
        training = series[series.index < '2025-01-01']
        for q in quantiles:
            threshold  = training.quantile(q)
            exceedance = (series > threshold).astype(int)
            key        = f"{col}_q{int(q*100)}_exceed"
            results[key] = exceedance
            n   = exceedance.sum()
            pct = n / len(exceedance) * 100
            print(f"  {key}: {n} days ({pct:.1f}%), threshold={threshold:.1f}°C")

    exceedance_df = pd.DataFrame(results)
    exceedance_df.to_parquet('data/temperature_exceedances.parquet')
    print(f"\nSaved exceedance flags: {exceedance_df.shape}")
    return exceedance_df


# ── ALIGN TO FINANCIAL CALENDAR ───────────────────────────────────────
def format_for_cpe(exceedance_df, prices_path='multiasset_prices.parquet'):
    try:
        prices      = pd.read_parquet(prices_path)
        temp_aligned = exceedance_df.reindex(prices.index, method='ffill')
        temp_aligned.to_parquet('data/temperature_exceedances_aligned.parquet')
        print(f"Loaded prices: {prices.shape}")
        print(f"Aligned temperature data: {temp_aligned.shape}")
        print(f"Saved: data/temperature_exceedances_aligned.parquet")
        return temp_aligned
    except FileNotFoundError:
        print(f"⚠ Could not find {prices_path} — skipping alignment step.")
        print("  Copy multiasset_prices.parquet here and re-run.")
        return exceedance_df


# ── SUMMARY ───────────────────────────────────────────────────────────
def print_summary(regional_df, exceedance_df):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Regional indices : {list(regional_df.columns)}")
    print(f"Date range       : {regional_df.index[0].date()} → {regional_df.index[-1].date()}")
    print(f"Total days       : {len(regional_df)}")
    print(f"Exceedance cols  : {len(exceedance_df.columns)}")
    for c in exceedance_df.columns:
        print(f"  {c}")
    print("\nReady to feed into CPE pipeline as new predictors.")


# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("PAPER 6 — TEMPERATURE DATA PIPELINE  v3")
    print("=" * 60)

    print("\nStep 1: Fetching city temperature data...")
    regional_df, city_data = build_temperature_dataset()

    print("\nStep 2: Computing exceedance flags...")
    exceedance_df = compute_temperature_exceedances(regional_df)

    print("\nStep 3: Aligning to financial calendar...")
    aligned = format_for_cpe(exceedance_df)

    print_summary(regional_df, exceedance_df)
