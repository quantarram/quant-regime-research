"""Batch 3 slow climate states for the CPE engine, same two engine-native forms as cpe_climate_data.py (`_chg` level, `_mean` cumulative sum), lagged for publication.
Lags: Nino 1+2 / 3 / 4 (CPC OISST file) and MEI v2, AAO: public on the 15th of month m+1. TNA, TSA, WHWP, PNA (NOAA PSL files, updated irregularly, last value was two months old when fetched): 15th of month m+2.
Dropped before any run for data lag or availability: AMO (file ends 2023), QBO (ends 2026-02), Indian monsoon rainfall (no free current series)."""
import os, re
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,'data','climate3')
def avail(vals, months=1):
    s=pd.Series(vals).dropna()
    idx=[pd.Timestamp(y,m,1)+pd.DateOffset(months=months)+pd.Timedelta(days=14) for (y,m) in s.index]
    return pd.Series(s.values,index=pd.DatetimeIndex(idx)).sort_index()
def ym_table(path,missing=99):
    out={}
    for line in open(path):
        p=line.split()
        if len(p)>=13 and re.fullmatch(r'\d{4}',p[0]):
            for m,v in enumerate(p[1:13],1):
                try: x=float(v)
                except ValueError: continue
                if abs(x)<missing: out[(int(p[0]),m)]=x
    return out
def sst_region(col):
    d=pd.read_csv(os.path.join(D,'sstoi.txt'),sep=r'\s+'); d.columns=['YR','MON','N12','N12A','N3','N3A','N4','N4A','N34','N34A']
    return avail({(int(r.YR),int(r.MON)):getattr(r,col) for r in d.itertuples()},1)
def aao():
    out={}
    for l in open(os.path.join(D,'aao.txt')):
        p=l.split()
        if len(p)==3: out[(int(p[0]),int(p[1]))]=float(p[2])
    return avail(out,1)
def series():
    return {'NINO12':sst_region('N12A'),'NINO3':sst_region('N3A'),'NINO4':sst_region('N4A'),
            'MEI2':avail(ym_table(os.path.join(D,'meiv2.txt')),1),'AAO':aao(),
            'TNA':avail(ym_table(os.path.join(D,'tna.txt')),2),'TSA':avail(ym_table(os.path.join(D,'tsa.txt')),2),
            'WHWP':avail(ym_table(os.path.join(D,'whwp.txt')),2),'PNA':avail(ym_table(os.path.join(D,'pna.txt')),2)}
def build(price_index):
    cols={}
    for k,s in series().items():
        daily=s.reindex(s.index.union(price_index)).ffill().reindex(price_index); daily[daily.index<s.index.min()]=np.nan
        cols[f'{k}_chg']=daily; cols[f'{k}_mean']=daily.cumsum()
    return pd.DataFrame(cols)
if __name__=='__main__':
    for k,s in series().items(): print(k,s.index.min().date(),s.index.max().date(),len(s),'last',round(float(s.iloc[-1]),2))
    px=pd.read_parquet(os.path.join(HERE,'..','multiasset_prices_live_history.parquet'))
    C=build(px.index); os.makedirs(os.path.join(HERE,'data','slow'),exist_ok=True); C.to_parquet(os.path.join(HERE,'data','slow','slow_daily.parquet')); print(C.shape)
