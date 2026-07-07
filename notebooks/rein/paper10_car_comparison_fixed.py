"""
Paper 10 -- Comparison Framework: Market-Model Cumulative Abnormal Returns (CAR)
====================================================================================
This is the STANDARD academic event-study method for this exact question --
the same approach used in Howerton & Bacon (2017, cited via Schuh 2023) for
their Katrina abnormal-returns finding, already cited in this paper.

Method:
  1. For each ticker and each event, estimate a market model (alpha, beta)
     via OLS regression of daily returns on SPY over a 250-trading-day
     estimation window ending 30 days BEFORE the event (standard practice,
     avoids contaminating the beta estimate with the event itself).
  2. Predict expected daily returns over the event window using that beta.
  3. Abnormal return (AR) = actual return - expected (market-model-predicted) return.
  4. Cumulative Abnormal Return (CAR) = sum of ARs over the horizon.
  5. Standard t-test across events: is mean CAR significantly different from zero?

This directly tests whether the raw-return CPE/permutation approach found
something a standard, market-adjusted method would also find -- and
specifically whether beta-adjustment automatically resolves the Ike/2008
Lehman-collapse confound that had to be manually diagnosed and excluded
in the CPE version.
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

EVENTS = {
    'Katrina_2005':  '2005-08-29', 'Rita_2005': '2005-09-24', 'Wilma_2005': '2005-10-24',
    'Ike_2008': '2008-09-13', 'Sandy_2012': '2012-10-29',
    'Harvey_2017': '2017-08-25', 'Irma_2017': '2017-09-10', 'Maria_2017': '2017-09-20',
    'Michael_2018': '2018-10-10', 'Laura_2020': '2020-08-27', 'Ida_2021': '2021-08-29',
    'Ian_2022': '2022-09-28', 'Helene_2024': '2024-09-26', 'Milton_2024': '2024-10-09',
}
CAT_EXPOSED = ['RNR', 'EG', 'ACGL', 'SREN.SW', 'MUV2.DE']
CONTROL = ['PGR', 'TRV']
HORIZONS_DAYS = {'1 week': 5, '2 weeks': 10, '1 month': 21, '3 months': 63}
ESTIMATION_WINDOW = 250
GAP_BEFORE_EVENT = 30


def estimate_market_model(stock_ret, market_ret, event_pos, window=ESTIMATION_WINDOW, gap=GAP_BEFORE_EVENT):
    """OLS alpha/beta from [event_pos - gap - window, event_pos - gap)."""
    end = event_pos - gap
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


def run():
    prices = pd.read_parquet('paper10_workdir/reinsurer_equity_prices.parquet')
    spy_price = prices['SPY'].dropna()
    spy_ret = spy_price.pct_change()

    results = []
    detail_rows = []

    for ticker in CAT_EXPOSED + CONTROL:
        if ticker not in prices.columns:
            continue
        group = 'cat-exposed' if ticker in CAT_EXPOSED else 'control'
        stock_price = prices[ticker].dropna()
        stock_ret_series = stock_price.pct_change()

        # Align stock and market returns on a common index
        common_idx = stock_ret_series.index.intersection(spy_ret.index)
        stock_ret = stock_ret_series.reindex(common_idx).values
        market_ret = spy_ret.reindex(common_idx).values
        idx = common_idx

        for horizon_label, days in HORIZONS_DAYS.items():
            cars = []
            for event_name, event_date in EVENTS.items():
                pos = idx.searchsorted(pd.Timestamp(event_date))
                if pos <= 0 or pos + days >= len(idx):
                    continue
                alpha, beta = estimate_market_model(stock_ret, market_ret, pos)
                if alpha is None:
                    continue
                actual = stock_ret[pos:pos + days]
                expected = alpha + beta * market_ret[pos:pos + days]
                ar = actual - expected
                if np.isnan(ar).any():
                    continue
                car = np.nansum(ar)
                cars.append(car)
                detail_rows.append({'ticker': ticker, 'group': group, 'event': event_name,
                                     'horizon': horizon_label, 'car': car, 'beta': beta})

            if len(cars) < 8:
                continue
            cars = np.array(cars)
            mean_car = cars.mean()
            t_stat, p_value = stats.ttest_1samp(cars, 0)
            results.append({
                'ticker': ticker, 'group': group, 'horizon': horizon_label,
                'n_events': len(cars), 'mean_car_pct': mean_car * 100,
                't_stat': t_stat, 'p_value': p_value,
            })

    out = pd.DataFrame(results)
    out.to_csv('paper10_workdir/paper10_car_results.csv', index=False)
    pd.DataFrame(detail_rows).to_csv('paper10_workdir/paper10_car_detail.csv', index=False)

    pd.set_option('display.width', 160)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(out.sort_values('p_value').to_string(index=False))

    # Multiple-testing correction, same standard applied to the CPE version
    n_tests = len(out)
    out_sorted = out.sort_values('p_value').reset_index(drop=True)
    out_sorted['rank'] = np.arange(1, len(out_sorted) + 1)
    out_sorted['bh_thresh_10'] = out_sorted['rank'] / n_tests * 0.10
    out_sorted['survives_fdr10'] = out_sorted['p_value'] <= out_sorted['bh_thresh_10']
    print(f"\nTotal tests: {n_tests}")
    print(f"Survives BH FDR q=0.10: {out_sorted['survives_fdr10'].sum()}")
    print(out_sorted[out_sorted['survives_fdr10']][['ticker', 'group', 'horizon', 'mean_car_pct', 'p_value']].to_string(index=False))

    return out


if __name__ == "__main__":
    run()
