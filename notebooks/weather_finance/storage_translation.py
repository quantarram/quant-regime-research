import json, os, sys, zipfile
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import edge_search as es
from weatherdash.calendar_utils import doy366
W=json.load(open(os.path.join(HERE,'data','gas_weights.json')))
def load_storage():
    z=zipfile.ZipFile(os.path.join(HERE,'data','eia','NG.zip'))
    with z.open('NG.txt') as fh:
        for line in fh:
            d=json.loads(line)
            if d.get('series_id')=='NG.NW2_EPG0_SWO_R48_BCF.W': return pd.Series({pd.Timestamp(a):float(b) for a,b in d['data']}).sort_index()
S=load_storage(); chg=S.diff().dropna()
def prior_norm(series):
    out={}
    for d in series.index:
        wk=d.isocalendar()[1]
        prev=[series[x] for x in series.index if x.year<d.year and x.year>=d.year-5 and x.isocalendar()[1]==wk]
        out[d]=np.mean(prev) if len(prev)>=3 else np.nan
    return pd.Series(out)
norm=prior_norm(chg)
T=es.metro_temps()
def dd(wdict,q):
    w=np.array([wdict[m] for m in es.NAMES]); X=T[es.NAMES].values
    v=(np.maximum(65-X,0) if q=='hdd' else np.maximum(X-65,0))*w
    return pd.Series(v.sum(1),index=T.index)
pw=es.weights(es.NAMES); pop={m:pw[i] for i,m in enumerate(es.NAMES)}
series={'H_pop':dd(pop,'hdd'),'C_pop':dd(pop,'cdd'),'H_gas':dd(W['heat'],'hdd'),'C_pow':dd(W['power'],'cdd'),'C_heat_w':dd(W['heat'],'cdd')}
def weekly(x): 
    return pd.Series({F:(x.loc[F-pd.Timedelta(days=6):F].sum() if len(x.loc[F-pd.Timedelta(days=6):F])==7 else np.nan) for F in chg.index})
X=pd.DataFrame({'chg':chg,'norm':norm}); 
for k,v in series.items(): X[k]=weekly(v)
X=X.dropna(); X['doy']=doy366(X.index); X['year']=X.index.year
def harm(df):
    a=2*np.pi*df.doy.values/366; return [np.ones(len(df)),np.sin(a),np.cos(a),np.sin(2*a),np.cos(2*a)]
def design(df,cols):
    c=harm(df)
    for k in cols: c+= [df[k].values/100,(df[k].values/100)**2]
    return np.column_stack(c)
MODELS={'M1 population HDD/CDD':['H_pop','C_pop'],'M2 gas-weighted HDD + power-weighted CDD':['H_gas','C_pow']}
def oos(cols,years):
    err=[]; 
    for Y in years:
        tr=X[X.year<Y]; te=X[X.year==Y]
        if len(tr)<150: continue
        b=np.linalg.lstsq(design(tr,cols),tr.chg.values,rcond=None)[0]
        err.append(pd.Series(te.chg.values-design(te,cols)@b,index=te.index))
    return pd.concat(err)
yrs=range(2016,2026)
base=(X.chg-X.norm)[X.year>=2016]
print('weeks',len(base),'| RMSE (Bcf) of weekly change: five-year norm only %.1f'%np.sqrt((base**2).mean()))
for name,cols in MODELS.items():
    e=oos(cols,yrs); print(f"{name:45s} OOS RMSE {np.sqrt((e**2).mean()):.1f}  (2021-25 only {np.sqrt((e[e.index.year>=2021]**2).mean()):.1f})  n={len(e)}")
X.to_csv(os.path.join(HERE,'data','storage_translation_weekly.csv'))
