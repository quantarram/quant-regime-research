"""Kitchen-sink machine-learning pass: per instrument, walk-forward (expanding window, refit yearly) ridge and gradient-boosted models on all
weather signals at once, predicting the next-5-day return; position = sign of the prediction when its magnitude is in the top half of the
training predictions; equal-weight across instruments; versus the same model fed calendar features only (climatology twin).

  python edge_ml.py L|E|F
"""
import pickle, sys, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
HERE=os.path.dirname(os.path.abspath(__file__))
U=sys.argv[1]
sigs=pickle.load(open(os.path.join(HERE,'data','edge_signals.pkl'),'rb'))
px=pd.read_parquet(os.path.join(HERE,'data','universe_all.parquet')); days=px['SPY'].dropna().index; px=px.reindex(days)
cfg={'L':('1998-01-01','2025-12-31',range(2008,2026),('obs',),True),'E':('2016-01-01','2025-12-31',range(2019,2026),('obs',),False),
     'F':('2021-03-23','2026-12-31',[2024,2025,2026],('fc',),False)}[U]
start,end,test_years,kinds,_=cfg
px=px.loc[start:end]
inst=[c for c in px.columns if px[c].first_valid_index() is not None and (U!='L' or px[c].first_valid_index()<=pd.Timestamp('2002-12-31'))]
if U=='L': inst=[c for c in inst if not c.endswith('=F')]   # continuous futures carry roll artifacts (see edge_diag_noF.py)
px=px[inst]; dates=px.index
names=[n for n in sigs if n.split('|')[0] in kinds]
X=pd.DataFrame({n:sigs[n].reindex(dates) for n in names}).fillna(0.0).clip(-4,4)
# calendar-only features: month dummies
C=pd.get_dummies(dates.month,prefix='m').astype(float); C.index=dates
H=5; COST=0.0005
r1=np.log(px/px.shift(1)); y=np.log(px.shift(-H)/px)
def walk(F,model_kind):
    pnl={}
    for tk in inst:
        s=y[tk]; f=F.copy(); ok=s.notna()&f.notna().all(axis=1)&px[tk].notna()
        pos_all=pd.Series(0.0,index=dates)
        for Y in test_years:
            tr=(dates.year<Y)&ok.values; te=(dates.year==Y)
            if tr.sum()<500: continue
            # purge the last H days of training (their targets overlap the test year)
            idx=np.where(tr)[0][:-H]
            Xt,yt=f.values[idx],s.values[idx]
            if model_kind=='ridge': m=Ridge(alpha=1000.0)
            else: m=HistGradientBoostingRegressor(max_depth=3,max_iter=80,learning_rate=0.05,min_samples_leaf=100)
            m.fit(Xt,yt); ptr=m.predict(Xt); thr=np.quantile(np.abs(ptr),0.5)
            p=m.predict(f.values[te])
            pos_all[te]=np.where(np.abs(p)>=thr,np.sign(p),0.0)
        ph=pos_all.rolling(H,min_periods=1).mean().shift(1).fillna(0.0)      # overlapping 5-day holds, earned from the next day
        cost=COST*ph.diff().abs().fillna(0.0)
        pnl[tk]=(ph*r1[tk].fillna(0.0)-cost)
    P=pd.DataFrame(pnl); m=P.index.year.isin(list(test_years))
    return P[m].mean(axis=1)
out={}
spy=r1['SPY'][np.isin(dates.year,list(test_years))]
for label,F in (('weather',X),('calendar',C)):
    for mk in ('ridge','gbm'):
        p=walk(F,mk); s=spy.reindex(p.index).fillna(0.0)
        var=s.var(); beta=p.cov(s)/var
        d=dict(ann_mean=float(p.mean()*252),ann_vol=float(p.std()*np.sqrt(252)),sharpe=float(p.mean()/p.std()*np.sqrt(252)),beta=float(beta),
               alpha=float((p.mean()-beta*s.mean())*252),yearly={int(k):float(v) for k,v in p.groupby(p.index.year).sum().items()})
        out[f'{label}_{mk}']=d
        print(U,label,mk,{k:(round(v,3) if not isinstance(v,dict) else {a:round(b*100,1) for a,b in v.items()}) for k,v in d.items()},flush=True)
json.dump(out,open(os.path.join(HERE,f'edge_ml_{U}.json'),'w'))
