"""
Paper 10 -- True CPE Exceedance Test on Market-Adjusted (CAR) Returns
============================================================================
CORRECTED DESIGN: the threshold defining "exceedance" must come from an
INDEPENDENT null distribution, not from the same 14 events being tested --
otherwise lift is circular and trivially ~1.0 by construction.

Method:
  1. For each ticker x horizon, build a genuine null distribution by
     estimating the SAME market-model abnormal return for 3,000 RANDOM
     windows drawn from that ticker's full trading history (not the 14
     real events).
  2. Define the exceedance threshold from that null distribution's
     quantile (25th/10th pctile for losses, 75th/90th for gains) --
     this is the correct "unconditional" reference point.
  3. Compute CPE = fraction of the 14 REAL event CARs (already computed
     by paper10_car_comparison_fixed.py) that cross that threshold.
  4. Lift = CPE / q. This is now a real, non-circular test.

Usage: needs BOTH files in the same folder --
  - paper10_workdir/reinsurer_equity_prices.parquet (full price history)
  - paper10_car_detail.csv (the 14 real event CARs, already computed)
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

CAT_EXPOSED = ['RNR', 'EG', 'ACGL', 'SREN.SW', 'MUV2.DE']
CONTROL = ['PGR', 'TRV']
HORIZONS_DAYS = {'1 week': 5, '2 weeks': 10, '1 month': 21, '3 months': 63}
ESTIMATION_WINDOW = 250
GAP_BEFORE_EVENT = 30
N_NULL_DRAWS = 3000
RNG = np.random.default_rng(17)


def estimate_market_model(stock_ret, market_ret, pos, window=ESTIMATION_WINDOW, gap=GAP_BEFORE_EVENT):
    end = pos - gap
    start = end - window
    if start < 0:
        return None, None
    y = stock_ret[start:end]
    x = market_ret[start:end]
    mask = ~(np.isnan(y) | np.isnan(x))
    if mask.sum() < 100:
        return None, None
    slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])
    return intercept, slope


def build_null_cars(stock_ret, market_ret, days, n_draws=N_NULL_DRAWS):
    """Genuine null distribution: real market-model AR/CAR computed at random positions."""
    n = len(stock_ret)
    valid_positions = np.arange(ESTIMATION_WINDOW + GAP_BEFORE_EVENT, n - days)
    if len(valid_positions) < 50:
        return np.array([])
    draws = RNG.choice(valid_positions, size=min(n_draws, len(valid_positions)), replace=False)
    null_cars = []
    for pos in draws:
        alpha, beta = estimate_market_model(stock_ret, market_ret, pos)
        if alpha is None:
            continue
        actual = stock_ret[pos:pos + days]
        expected = alpha + beta * market_ret[pos:pos + days]
        ar = actual - expected
        if np.isnan(ar).any():
            continue
        null_cars.append(np.nansum(ar))
    return np.array(null_cars)


def run():
    prices = pd.read_parquet('paper10_workdir/reinsurer_equity_prices.parquet')
    detail = pd.read_csv('paper10_workdir/paper10_car_detail.csv')
    spy_price = prices['SPY'].dropna()
    spy_ret_full = spy_price.pct_change()

    results = []
    for ticker in CAT_EXPOSED + CONTROL:
        if ticker not in prices.columns:
            continue
        group = 'cat-exposed' if ticker in CAT_EXPOSED else 'control'
        stock_price = prices[ticker].dropna()
        stock_ret_series = stock_price.pct_change()
        common_idx = stock_ret_series.index.intersection(spy_ret_full.index)
        stock_ret = stock_ret_series.reindex(common_idx).values
        market_ret = spy_ret_full.reindex(common_idx).values

        for horizon_label, days in HORIZONS_DAYS.items():
            real_cars = detail[(detail['ticker'] == ticker) & (detail['horizon'] == horizon_label)]['car'].dropna().values
            if len(real_cars) < 10:
                continue

            print(f"Building null distribution: {ticker}, {horizon_label} ({N_NULL_DRAWS} draws)...")
            null_cars = build_null_cars(stock_ret, market_ret, days)
            if len(null_cars) < 100:
                continue

            for q, direction in [(0.10, 'loss'), (0.25, 'loss'), (0.75, 'gain'), (0.90, 'gain')]:
                if direction == 'loss':
                    thr = np.quantile(null_cars, q)
                    cpe = (real_cars <= thr).mean()
                    lift = cpe / q
                else:
                    thr = np.quantile(null_cars, q)
                    cpe = (real_cars >= thr).mean()
                    lift = cpe / (1 - q)

                results.append({
                    'ticker': ticker, 'group': group, 'horizon': horizon_label,
                    'direction': direction, 'q': q, 'threshold_pct': thr * 100,
                    'cpe': cpe, 'lift': lift, 'n_real_events': len(real_cars),
                    'n_null_draws': len(null_cars),
                })

    out = pd.DataFrame(results)
    out.to_csv('paper10_workdir/paper10_cpe_on_car_results.csv', index=False)
    pd.set_option('display.width', 160)
    pd.set_option('display.float_format', lambda x: f'{x:.3f}')

    print("\n=== RNR, short horizons, loss side ===")
    print(out[(out['ticker']=='RNR') & (out['horizon'].isin(['1 week','2 weeks'])) & (out['direction']=='loss')].to_string(index=False))

    print("\n=== Munich Re, medium horizons, gain side ===")
    print(out[(out['ticker']=='MUV2.DE') & (out['horizon'].isin(['1 month','3 months'])) & (out['direction']=='gain')].to_string(index=False))

    print("\n=== Swiss Re, medium horizons, gain side ===")
    print(out[(out['ticker']=='SREN.SW') & (out['horizon'].isin(['1 month','3 months'])) & (out['direction']=='gain')].to_string(index=False))

    print("\n=== Control group (should show weak/no lift) ===")
    print(out[out['group']=='control'].sort_values('lift', ascending=False).head(10).to_string(index=False))

    print("\nSaved -> paper10_cpe_on_car_results.csv")
    return out


if __name__ == "__main__":
    run()
