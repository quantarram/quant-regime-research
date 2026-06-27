"""
Paper 6 — Diagnostic Script
Checks what's actually happening with one pair before the full run.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

TRAIN_END  = '2024-12-31'
EVAL_START = '2025-01-01'

# Load data
prices  = pd.read_parquet('multiasset_prices.parquet')
returns = pd.read_parquet('multiasset_returns.parquet')
temp    = pd.read_parquet('data/temperature_exceedances_aligned.parquet')

print("=== DATA SHAPES ===")
print(f"Prices:  {prices.shape}  {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"Returns: {returns.shape}  {returns.index[0].date()} → {returns.index[-1].date()}")
print(f"Temp:    {temp.shape}    {temp.index[0].date()} → {temp.index[-1].date()}")

# Check index types
print(f"\nPrices index type:  {type(prices.index)}")
print(f"Returns index type: {type(returns.index)}")
print(f"Temp index type:    {type(temp.index)}")

# Check temp values
print("\n=== TEMPERATURE EXCEEDANCE VALUES ===")
for col in temp.columns:
    n1 = (temp[col] == 1).sum()
    n0 = (temp[col] == 0).sum()
    nna = temp[col].isna().sum()
    print(f"  {col}: n=1:{n1}, n=0:{n0}, NaN:{nna}")

# Pick one pair: Europe_tmax_q90_exceed → XLU
temp_col = 'Europe_tmax_q90_exceed'
ticker   = 'XLU'
horizon  = 63
q_target = 0.80

print(f"\n=== DIAGNOSTIC: {temp_col} → {ticker}, horizon={horizon}d, q={q_target} ===")

temp_series   = temp[temp_col]
return_series = returns[ticker]

print(f"\nReturn series: {len(return_series)} obs, NaN: {return_series.isna().sum()}")
print(f"Temp series:   {len(temp_series)} obs, NaN: {temp_series.isna().sum()}")
print(f"Temp series values: {temp_series.value_counts().to_dict()}")

# Check alignment
common = temp_series.index.intersection(return_series.index)
print(f"\nCommon dates: {len(common)}")

temp_c = temp_series.loc[common]
ret_c  = return_series.loc[common]

# Forward returns
trading_horizon = max(1, int(round(horizon * 252 / 365)))
print(f"Trading horizon: {trading_horizon} days")

fwd_ret = ret_c.shift(-trading_horizon)
print(f"Forward returns: {fwd_ret.notna().sum()} non-NaN values")
print(f"Forward returns sample:\n{fwd_ret.dropna().describe()}")

# Training period
train_temp = temp_c[temp_c.index <= TRAIN_END]
train_fwd  = fwd_ret[fwd_ret.index <= TRAIN_END].dropna()

print(f"\nTraining temp flag=1: {(train_temp==1).sum()}")
print(f"Training fwd returns: {len(train_fwd)}")

if len(train_fwd) > 0:
    threshold  = train_fwd.quantile(q_target)
    uncond_prob = (train_fwd > threshold).mean()
    print(f"Threshold ({q_target}): {threshold:.4f}")
    print(f"Uncond prob: {uncond_prob:.4f}")

    # CPE computation
    cond_mask   = (train_temp == 1)
    print(f"Conditioning events (train): {cond_mask.sum()}")

    # Check if conditioning dates have forward returns
    cond_dates  = train_temp[cond_mask].index
    cond_fwd    = fwd_ret.loc[cond_dates].dropna()
    print(f"Conditioning events with resolved fwd returns: {len(cond_fwd)}")

    if len(cond_fwd) > 0:
        cpe = (cond_fwd > threshold).mean()
        lift = cpe / uncond_prob if uncond_prob > 0 else 0
        print(f"\nCPE (train): {cpe:.4f}")
        print(f"Lift:        {lift:.4f}x")
        print(f"Uncond:      {uncond_prob:.4f}")
        print(f"\nDistribution of forward returns on hot days:")
        print(cond_fwd.describe())
        print(f"\nDistribution of all training forward returns:")
        print(train_fwd.describe())
    else:
        print("NO conditioning events have resolved forward returns!")
        print("This is the bug — let's check why:")
        print(f"  Cond dates range: {cond_dates[0].date()} → {cond_dates[-1].date()}")
        print(f"  fwd_ret dates range: {fwd_ret.dropna().index[0].date()} → {fwd_ret.dropna().index[-1].date()}")
        print(f"  fwd_ret at cond dates (first 5):")
        print(fwd_ret.loc[cond_dates[:5]])

# Also check what the returns look like
print(f"\n=== RETURNS CHECK FOR {ticker} ===")
print(returns[ticker].describe())
print(f"\nFirst 5 values:\n{returns[ticker].head()}")
print(f"\nAre returns in % or decimal?")
print(f"  Mean: {returns[ticker].mean():.6f}")
print(f"  Std:  {returns[ticker].std():.6f}")
print(f"  Max:  {returns[ticker].max():.6f}")
print(f"  Min:  {returns[ticker].min():.6f}")
