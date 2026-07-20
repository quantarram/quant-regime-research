"""
Wave-2 predictor panel: classical realized skewness/kurtosis, per instrument.

Rolling 60d realized skew/kurtosis of daily log returns -- a simpler, more
standard higher-moment family from the options/vol literature, distinct from
the multifractal-cascade q=4 structure functions already in the baseline
panel. Useful as an independent cross-check on whether the multifractal
framing specifically adds value over classical higher moments.

Run: python 08e_skew_kurtosis_features.py
Output: features_skew_kurtosis.parquet
"""
import pandas as pd
import numpy as np
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

INSTRUMENTS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "AAPL", "MSFT", "JPM", "XOM",
               "GLD", "BTC-USD", "TLT", "EURUSD=X", "^VIX"]
WINDOW = 60

prices = pd.read_parquet(os.path.join(REPO_DIR, "multiasset_prices.parquet"))

frames = []
for t in INSTRUMENTS:
    # dropna BEFORE shift/rolling -- see 08a_vrp_features.py's comment: the shared calendar
    # started including weekend rows (NaN for this ticker) once crypto joined it (~2014-09-17),
    # which silently breaks positional .shift(1)/.rolling() on the raw column from that date on.
    s = prices[t].dropna()
    ret = np.log(s / s.shift(1))
    df = pd.DataFrame({
        "date": s.index,
        "ticker": t,
        "realized_skew_60d": ret.rolling(WINDOW, min_periods=30).skew().values,
        "realized_kurt_60d": ret.rolling(WINDOW, min_periods=30).kurt().values,
    })
    frames.append(df)

feat = pd.concat(frames, ignore_index=True)
out_path = os.path.join(OUT_DIR, "features_skew_kurtosis.parquet")
feat.to_parquet(out_path)
print(f"Saved {len(feat)} rows x {len(feat.columns)} cols to {out_path}")
print(feat.dropna().groupby("ticker")[["realized_skew_60d", "realized_kurt_60d"]].mean())
