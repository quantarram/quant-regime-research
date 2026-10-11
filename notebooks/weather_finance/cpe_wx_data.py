"""Short-horizon WEATHER predictors for the CPE engine (the comparison arm of the weather-versus-climate test).
Daily degree-day anomalies (population-weighted metro temperatures, ERA5; anomaly vs formation-period (<= 2012) day-of-year normals), available the next calendar day (one-day lag), as the engine's rate-type cumulative-sum form:
the engine's tau-day increment of `<name>_mean` is the SUM of the daily anomaly over the past tau days, i.e. the weather of the past tau days.
  WXHDDNAT / WXHDDCOLD  heating degree-day anomaly, all metros / cold-winter metros;  WXCDDNAT / WXCDDSOUTH / WXCDDTX / WXCDDMW  cooling degree-day anomaly (all, southern, Texas, Midwest metros)."""
import os
import numpy as np, pandas as pd
import weather_scan as ws
HERE=os.path.dirname(os.path.abspath(__file__))
def daily_anomaly(T,cities,kind):
    w=np.array([ws.POP[c] for c in cities],float); w/=w.sum(); X=T[cities].values
    dd=(np.maximum(65-X,0) if kind=='hdd' else np.maximum(X-65,0))*w; s=pd.Series(dd.sum(1),index=T.index)
    form=s[s.index<=ws.FORM_END]; n=form.groupby(ws.doy366(form.index)).mean().reindex(range(1,367)).interpolate(limit_direction='both')
    sm=pd.concat([n.iloc[-7:],n,n.iloc[:7]]).rolling(15,center=True).mean().iloc[7:-7]; sm.index=range(1,367)
    return s-pd.Series(sm.loc[ws.doy366(s.index)].values,index=s.index)
def build(price_index):
    T=ws.temps(); defs={'WXHDDNAT':(ws.NAMES,'hdd'),'WXHDDCOLD':(ws.COLD,'hdd'),'WXCDDNAT':(ws.NAMES,'cdd'),'WXCDDSOUTH':(ws.SOUTH,'cdd'),'WXCDDTX':(ws.TX,'cdd'),'WXCDDMW':(ws.MW,'cdd')}
    cols={}
    for k,(c,kind) in defs.items():
        a=daily_anomaly(T,c,kind).shift(1)                      # weather through the previous calendar day
        d=a.reindex(a.index.union(price_index)).fillna(0.0).reindex(price_index)   # days without a metro record count as zero anomaly; the cumulative sum is continuous
        d[d.index<a.dropna().index.min()]=np.nan
        d[d.index>T.index.max()+pd.Timedelta(days=1)]=np.nan    # temperature history ends 2025-12-31
        cols[k+'_mean']=d.cumsum()
    return pd.DataFrame(cols), T
if __name__=='__main__':
    px=pd.read_parquet(os.path.join(HERE,'..','multiasset_prices_live_history.parquet'))
    X,T=build(px.index); print('temps',T.index.min().date(),T.index.max().date(),T.shape); print(X.dropna(how='all').index.min().date(),X.shape)
    os.makedirs(os.path.join(HERE,'data','wx'),exist_ok=True); X.to_parquet(os.path.join(HERE,'data','wx','wx_daily.parquet'))
    print(X.tail(2).round(1).T)
