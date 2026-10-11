"""Survivors (filter fixed before looking: >=4 formation episodes, >=3 test episodes, test episode hit rate >= 2/3, test CPE >= test base + 0.10, >=100 test days) and which fire now."""
import sys, os
import numpy as np, pandas as pd
sys.argv=['x','formation']
import cpe_climate_run as R
CUT=R.CUTOFF
px,C=R.load_panel(); inc_f,fut_f,thr_f,pred,climate=R.build(px,C,CUT); inc_a,fut_a,thr_a,_,_=R.build(px,C,None)
E=pd.read_parquet(os.path.join(R.OUTD,'formation_eval.parquet'))
S=E[(E.n_ep_f>=4)&(E.n_ep_t>=3)&(E.hit_t>=2/3)&(E.cpe_test>=E.base_test+0.10)&(E.n_test>=100)].copy()
print('survivors',len(S),'of',len(E),'(admitted rows with >=4 formation episodes:',int((E.n_ep_f>=4).sum()),')')
print(S.groupby('ysub').size().to_dict()); print(S.groupby('X').size().to_dict()); print(S.groupby('Y').size().sort_values(ascending=False).head(12).to_dict())
last=px.index.max(); print('last price date',last.date())
def fires_now(r):
    s=inc_a[int(r.tau_past)][r.X]; v=s.iloc[-1]
    if pd.isna(v): return np.nan
    thr=thr_f[(int(r.tau_past),float(r.q_X))][r.X] if r.direction=='bullish' else thr_f[(int(r.tau_past),round(1-float(r.q_X),10))][r.X]
    return bool(v>thr) if r.direction=='bullish' else bool(v<thr)
S['fires_now']=S.apply(fires_now,axis=1)
print('\nsurvivors firing now (stale NAO/PDO excluded):',int((S.fires_now==True).sum()))
live=S[(S.fires_now==True)&~S.X.str.startswith(('NAO','PDO'))]
g=live.groupby(['X','Y','direction']).agg(rows=('CPE','size'),claimed=('CPE','mean'),n_ep_f=('n_ep_f','max'),test_cpe=('cpe_test','mean'),test_base=('base_test','mean'),n_ep_t=('n_ep_t','max'),hit_t=('hit_t','mean'),horizons=('tau_future',lambda s:sorted(set(int(x) for x in s)))).reset_index()
pd.set_option('display.width',250); pd.set_option('display.max_columns',20)
print(g.sort_values('rows',ascending=False).head(25).round(3).to_string(index=False))
S.to_parquet(os.path.join(R.OUTD,'survivors.parquet')); live.to_parquet(os.path.join(R.OUTD,'survivors_live.parquet'))
