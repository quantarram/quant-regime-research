"""Batch 4: rare slow extreme climate states for the CPE engine, daily with publication lags, two engine-native forms (`_chg` level, `_mean` cumulative sum).
  HRWPDSI  hard-red-winter wheat belt drought: mean state PDSI of KS, OK, NE, CO, TX      (NCEI nClimDiv; negative = drought; public on the 12th of month m+1)
  NPLPDSI  northern plains spring-wheat drought: ND, SD, MN, MT
  DLTPDSI  Mississippi Delta drought: AR, LA, MS, TN
  SEPDSI   Southeast drought: GA, FL, AL, SC
  TXPDSI   Texas drought: TX
  WSTPDSI  Western drought: CA, OR, WA, ID, NV, UT, AZ
  TNI      Trans-Nino index (HadISST, NOAA PSL; the file ends ~4 months back: public on the 15th of month m+5)
  NPGO     North Pacific Gyre Oscillation (NOAA PSL; 15th of month m+2)
  GLBT     global land-ocean temperature anomaly (NASA GISTEMP v4; 15th of month m+1)
  SNOW     Northern Hemisphere snow-cover extent anomaly vs 1981-2010 monthly normals (Rutgers; 15th of month m+1)
  ICE      Arctic sea-ice extent anomaly vs 1981-2010 monthly normals (NSIDC G02135 v4; 15th of month m+1)"""
import os, re
import numpy as np, pandas as pd
import cpe_extreme_data as ex
HERE=os.path.dirname(os.path.abspath(__file__)); D4=os.path.join(HERE,'data','climate4')
REGIONS={'HRWPDSI':['KS','OK','NE','CO','TX'],'NPLPDSI':['ND','SD','MN','MT'],'DLTPDSI':['AR','LA','MS','TN'],'SEPDSI':['GA','FL','AL','SC'],'TXPDSI':['TX'],'WSTPDSI':['CA','OR','WA','ID','NV','UT','AZ']}
def anomaly_1981_2010(s):
    base=s[(s.index.year>=1981)&(s.index.year<=2010)]; clim=base.groupby(base.index.month).mean()
    return s-s.index.month.map(clim).values
def avail(s,months,day):
    return pd.Series(s.values,index=pd.DatetimeIndex([d+pd.DateOffset(months=months)+pd.Timedelta(days=day) for d in s.index])).sort_index()
def ym_table(path,missing=99):
    out={}
    for line in open(path):
        p=line.split()
        if len(p)>=13 and re.fullmatch(r'\d{4}',p[0]):
            for m,v in enumerate(p[1:13],1):
                try: x=float(v)
                except ValueError: continue
                if abs(x)<missing: out[pd.Timestamp(int(p[0]),m,1)]=x
    return pd.Series(out).sort_index()
def gistemp():
    out={}
    for l in open(os.path.join(D4,'gistemp.csv')):
        p=l.strip().split(',')
        if p and re.fullmatch(r'\d{4}',p[0]):
            for m,v in enumerate(p[1:13],1):
                try: out[pd.Timestamp(int(p[0]),m,1)]=float(v)
                except ValueError: pass
    return pd.Series(out).sort_index()
def snow():
    out={}
    for l in open(os.path.join(D4,'snow.txt')):
        p=l.split()
        if len(p)==3: out[pd.Timestamp(int(p[0]),int(p[1]),1)]=float(p[2])/1e6
    return anomaly_1981_2010(pd.Series(out).sort_index())
def seaice():
    out={}
    for m in range(1,13):
        for l in open(os.path.join(D4,f'seaice_{m:02d}.csv')):
            p=[x.strip() for x in l.split(',')]
            if len(p)>=6 and re.fullmatch(r'\d{4}',p[0]):
                try:
                    v=float(p[4])
                    if v>0: out[pd.Timestamp(int(p[0]),int(p[1]),1)]=v   # -9999 marks missing months
                except ValueError: pass
    return anomaly_1981_2010(pd.Series(out).sort_index())
def monthly():
    pdsi=ex.read_ncei('climdiv-pdsist.txt'); out={}; lags={}
    for k,st in REGIONS.items():
        out[k]=pdsi[st].mean(axis=1).dropna(); lags[k]=(1,11)
    out['TNI']=ym_table(os.path.join(D4,'tni.txt')); lags['TNI']=(5,14)
    out['NPGO']=ym_table(os.path.join(D4,'npgo.txt')); lags['NPGO']=(2,14)
    out['GLBT']=gistemp(); lags['GLBT']=(1,14)
    out['SNOW']=snow(); lags['SNOW']=(1,14)
    out['ICE']=seaice(); lags['ICE']=(1,14)
    return out,lags
def build(price_index):
    # Climate states are monthly; they are laid on Monday-Friday rows (business days) so that a horizon of tau is tau business days, and the
    # cumulative-sum form sums tau weekday readings. Weekend rows of the 7-day price panel stay NaN (same unit as the targets' own trading days).
    price_index = pd.bdate_range(price_index.min(), price_index.max())
    ms,lags=monthly(); cols={}
    for k,s in ms.items():
        a=avail(s,*lags[k]); d=a.reindex(a.index.union(price_index)).ffill().reindex(price_index); d[d.index<a.index.min()]=np.nan
        cols[k+'_chg']=d; cols[k+'_mean']=d.cumsum()
    return pd.DataFrame(cols), ms
if __name__=='__main__':
    px=pd.read_parquet(os.path.join(HERE,'..','multiasset_prices_live_history.parquet'))
    X,ms=build(px.index)
    for k,s in ms.items(): print(f'{k:8s} {s.index.min().date()} {s.index.max().date()} n={len(s)} last {s.iloc[-1]:.2f}  min {s.min():.2f} ({s.idxmin().strftime("%Y-%m")})  max {s.max():.2f} ({s.idxmax().strftime("%Y-%m")})')
    os.makedirs(os.path.join(HERE,'data','x4'),exist_ok=True); X.to_parquet(os.path.join(HERE,'data','x4','x4_daily.parquet')); print(X.shape)
