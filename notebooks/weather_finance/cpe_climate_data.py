"""Builds the climate predictor panel for the CPE engine: monthly NOAA indices -> daily series with a publication lag, in two engine-native forms.
  <name>_chg   the index level (rate-type series: the engine's tau-day increment is the CHANGE in the state over the past tau days)
  <name>_mean  the cumulative sum of the daily level (rate-type: the tau-day increment is the SUM of the level over the past tau days, i.e. the average state)
Availability: ONI of the season centred on month m is public ~10 days into month m+2; every other monthly index is taken as public on the 15th of month m+1.
"""
import os, re
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,'data','climate')
def monthly_to_avail(vals, lag='next15'):
    s=pd.Series(vals).dropna()
    idx=[pd.Timestamp(y,m,1)+pd.DateOffset(months=1)+pd.Timedelta(days=14) for (y,m) in s.index]
    return pd.Series(s.values,index=pd.DatetimeIndex(idx)).sort_index()
def oni():
    SEAS={'DJF':1,'JFM':2,'FMA':3,'MAM':4,'AMJ':5,'MJJ':6,'JJA':7,'JAS':8,'ASO':9,'SON':10,'OND':11,'NDJ':12}
    o=pd.read_csv(os.path.join(D,'oni.txt'),sep=r'\s+')
    idx=[pd.Timestamp(int(y),SEAS[s],1)+pd.DateOffset(months=2)+pd.Timedelta(days=9) for y,s in zip(o.YR,o.SEAS)]
    return pd.Series(o.ANOM.values,index=pd.DatetimeIndex(idx)).sort_index()
def nino34():
    d=pd.read_csv(os.path.join(D,'sstoi.txt'),sep=r'\s+'); d.columns=['YR','MON','N12','N12A','N3','N3A','N4','N4A','N34','N34A']
    return monthly_to_avail({(int(r.YR),int(r.MON)):r.N34A for r in d.itertuples()})
def ao():
    d=pd.read_csv(os.path.join(D,'ao.txt'),sep=r'\s+',header=None); return monthly_to_avail({(int(r[0]),int(r[1])):r[2] for r in d.itertuples(index=False)})
def table_year_month(path,missing,skip=1):
    out={}
    for line in open(path):
        parts=line.split()
        if len(parts)>=13 and re.fullmatch(r'\d{4}',parts[0]):
            for m,v in enumerate(parts[1:13],1):
                try: x=float(v)
                except ValueError: continue
                if abs(x)<missing: out[(int(parts[0]),m)]=x
    return out
def soi():
    txt=open(os.path.join(D,'soi.txt')).read().splitlines(); start=[i for i,l in enumerate(txt) if 'STANDARDIZED' in l][0]
    out={}
    for l in txt[start:]:
        if re.match(r'^\d{4}',l):
            y=int(l[:4]); body=l[4:]
            for m in range(12):
                tok=body[m*6:(m+1)*6].strip()
                try: x=float(tok)
                except ValueError: continue
                if abs(x)<99: out[(y,m+1)]=x
    return monthly_to_avail(out)
def build(price_index):
    series={'ONI':oni(),'NINO34':nino34(),'AO':ao(),'NAO':monthly_to_avail(table_year_month(os.path.join(D,'nao.txt'),99)),
            'PDO':monthly_to_avail(table_year_month(os.path.join(D,'pdo.txt'),9.0)),'SOI':soi()}
    cols={}
    for k,s in series.items():
        daily=s.reindex(s.index.union(price_index)).ffill().reindex(price_index)
        first=s.index.min(); daily[daily.index<first]=np.nan
        cols[f'{k}_chg']=daily
        cols[f'{k}_mean']=daily.cumsum()
    return pd.DataFrame(cols), series
if __name__=='__main__':
    px=pd.read_parquet(os.path.join(HERE,'..','multiasset_prices_live_history.parquet'))
    C,series=build(px.index)
    for k,s in series.items(): print(k, s.index.min().date(), s.index.max().date(), len(s), 'last', round(float(s.iloc[-1]),2))
    print(C.tail(2).round(2).T.head(12)); C.to_parquet(os.path.join(HERE,'data','climate','climate_daily.parquet'))
