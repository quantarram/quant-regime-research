import json, os, sys, zipfile
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
sys.path.insert(0,os.path.abspath(os.path.join(HERE,'..','..','..','climate-energy-dashboard')))
import edge_search as es
from weatherdash.calendar_utils import doy366
z=zipfile.ZipFile(os.path.join(HERE,'data','eia','NG.zip'))
with z.open('NG.txt') as fh:
    for line in fh:
        d=json.loads(line)
        if d.get('series_id')=='NG.NW2_EPG0_SWO_R33_BCF.W': s=pd.Series({pd.Timestamp(a):float(b) for a,b in d['data']}).sort_index()
chg=s.diff().dropna()
cities=['Dallas','Houston','Austin','San Antonio','Nashville']; w=es.weights(cities); T=es.metro_temps(); Xm=T[cities].values
h=pd.Series((np.maximum(65-Xm,0)*w).sum(1),index=T.index); c=pd.Series((np.maximum(Xm-65,0)*w).sum(1),index=T.index)
def wk(x,idx): return pd.Series({F:(x.loc[F-pd.Timedelta(days=6):F].sum() if len(x.loc[F-pd.Timedelta(days=6):F])==7 else np.nan) for F in idx})
D=pd.DataFrame({'chg':chg,'H':wk(h,chg.index),'C':wk(c,chg.index)})
b=pd.read_parquet(os.path.join(HERE,'data','eia930_daily.parquet')); b['date']=pd.to_datetime(b['date'])
def burn(bas):
    x=b[b.ba.isin(bas)].groupby('date').gas.sum()/1000; return wk(x,chg.index)
D['b_ercot']=burn(['ERCO']); D['b_sc']=burn(['ERCO','SWPP','SOCO','TVA','EPE','AECI']); D['b_ercot_rest']=D.b_sc-D.b_ercot
D=D.dropna(); D['doy']=doy366(D.index); D['year']=D.index.year
def design(df,extra):
    a=2*np.pi*df.doy.values/366; cols=[np.ones(len(df)),np.sin(a),np.cos(a),np.sin(2*a),np.cos(2*a),df.H.values/100,df.C.values/100,(df.H.values/100)**2,(df.C.values/100)**2]
    for k in extra: cols.append(df[k].values/1000)
    return np.column_stack(cols)
print('weeks',len(D),D.index.min().date(),D.index.max().date())
for name,extra in (('weather only',[]),('+ ERCOT gas burn (observed)',['b_ercot']),('+ ERCOT + SPP/Southern/TVA gas burn (observed)',['b_sc'])):
    out=[]
    for Y in range(2022,2026):
        tr=D[D.year<Y]; te=D[D.year==Y]
        bb=np.linalg.lstsq(design(tr,extra),tr.chg.values,rcond=None)[0]; out.append(pd.Series(te.chg.values-design(te,extra)@bb,index=te.index))
    e=pd.concat(out); print(f"{name:50s} South Central OOS RMSE {np.sqrt((e**2).mean()):5.1f} Bcf (weeks {len(e)})")
