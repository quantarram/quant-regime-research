import json, os, sys, zipfile
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
sys.path.insert(0,os.path.abspath(os.path.join(HERE,'..','..','..','climate-energy-dashboard')))
import edge_search as es
from weatherdash.calendar_utils import doy366
z=zipfile.ZipFile(os.path.join(HERE,'data','eia','NG.zip')); ser={}
with z.open('NG.txt') as fh:
    for line in fh:
        d=json.loads(line)
        if d.get('f')=='W' and 'Working Underground Storage' in d.get('name','') and d['series_id'].startswith('NG.NW2_EPG0_SWO_R'):
            ser[d['series_id']]=(d['name'],pd.Series({pd.Timestamp(a):float(b) for a,b in d['data']}).sort_index())
for k,(n,s) in ser.items(): print(k,n[:70],len(s))
REG={'R31':['New York','Philadelphia','Washington','Boston','Baltimore','Pittsburgh','Atlanta','Charlotte','Miami','Orlando','Tampa'],
     'R32':['Chicago','Detroit','Minneapolis','St. Louis','Cincinnati','Columbus','Indianapolis','Cleveland','Kansas City'],
     'R33':['Dallas','Houston','Austin','San Antonio','Nashville'],
     'R34':['Denver','Phoenix','Las Vegas'],
     'R35':['Los Angeles','San Francisco','San Diego','Sacramento','Seattle','Portland']}
LAB={'R31':'East','R32':'Midwest','R33':'South Central','R34':'Mountain','R35':'Pacific'}
T=es.metro_temps()
def wk(x,idx): return pd.Series({F:(x.loc[F-pd.Timedelta(days=6):F].sum() if len(x.loc[F-pd.Timedelta(days=6):F])==7 else np.nan) for F in idx})
preds={}; actual={}; norms={}
for r,cities in REG.items():
    s=ser[f'NG.NW2_EPG0_SWO_{r}_BCF.W'][1]; chg=s.diff().dropna(); w=es.weights(cities); Xm=T[cities].values
    h=pd.Series((np.maximum(65-Xm,0)*w).sum(1),index=T.index); c=pd.Series((np.maximum(Xm-65,0)*w).sum(1),index=T.index)
    D=pd.DataFrame({'chg':chg,'H':wk(h,chg.index),'C':wk(c,chg.index)}).dropna(); D['doy']=doy366(D.index); D['year']=D.index.year
    # five-year norm
    nm={}
    for d in D.index:
        wkn=d.isocalendar()[1]; prev=[D.chg[x] for x in D.index if x.year<d.year and x.year>=d.year-5 and x.isocalendar()[1]==wkn]; nm[d]=np.mean(prev) if len(prev)>=3 else np.nan
    D['norm']=pd.Series(nm)
    def design(df):
        a=2*np.pi*df.doy.values/366; return np.column_stack([np.ones(len(df)),np.sin(a),np.cos(a),np.sin(2*a),np.cos(2*a),df.H.values/100,df.C.values/100,(df.H.values/100)**2,(df.C.values/100)**2])
    out=[]
    for Y in range(2016,2026):
        tr=D[(D.year<Y)&D.norm.notna()]; te=D[D.year==Y]
        if len(tr)<150: continue
        b=np.linalg.lstsq(design(tr),tr.chg.values,rcond=None)[0]; out.append(pd.Series(design(te)@b,index=te.index))
    p=pd.concat(out); preds[r]=p; actual[r]=D.chg.reindex(p.index); norms[r]=D.norm.reindex(p.index)
    e=actual[r]-p; e0=actual[r]-norms[r]
    print(f"{LAB[r]:14s} RMSE model {np.sqrt((e**2).mean()):5.1f}  norm only {np.sqrt((e0**2).mean()):5.1f}  (weeks {len(p)}; mean |chg| {actual[r].abs().mean():.1f})")
P=pd.DataFrame(preds).dropna(); A=pd.DataFrame(actual).reindex(P.index)
nat_err=(A.sum(1)-P.sum(1)); print('sum of regional models vs national change: RMSE %.1f Bcf (national model M1: ~22.6 over 2016-25)'%np.sqrt((nat_err**2).mean()))
print('2021-2025 only: %.1f'%np.sqrt((nat_err[nat_err.index.year>=2021]**2).mean()))
