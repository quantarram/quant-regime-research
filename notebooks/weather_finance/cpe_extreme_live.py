"""Survivors for the extreme-event predictors, same pre-registered filter as cpe_climate_live.py."""
import sys, os
import numpy as np, pandas as pd
sys.argv=['x','formation']
import cpe_extreme_run as ER
R=ER.R
CUT=R.CUTOFF
px,C=R.load_panel(); inc_a,fut_a,thr_a,_,_=R.build(px,C,None); inc_f,fut_f,thr_f,_,_=R.build(px,C,CUT)
E=pd.read_parquet(os.path.join(R.OUTD,'extreme_formation_eval.parquet'))
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
S=E[(E.n_ep_f>=4)&(E.n_ep_t>=3)&(E.hit_t>=2/3)&(E.cpe_test>=E.base_test+0.10)&(E.n_test>=100)].copy()
print('admitted',len(E),'| survivors',len(S))
print('by X:',S.groupby('X').size().to_dict()); print('by ysub:',S.groupby('ysub').size().to_dict())
print('by q_X:',S.groupby('q_X').size().to_dict(),' | by tau_future:',S.groupby('tau_future').size().to_dict())
last=px.index.max(); print('last price date',last.date())
def fires_now(r):
    s=inc_a[int(r.tau_past)][r.X]; v=s.iloc[-1]
    if pd.isna(v): return np.nan
    thr=thr_f[(int(r.tau_past),float(r.q_X))][r.X] if r.direction=='bullish' else thr_f[(int(r.tau_past),round(1-float(r.q_X),10))][r.X]
    return bool(v>thr) if r.direction=='bullish' else bool(v<thr)
if len(S):
    S['fires_now']=S.apply(fires_now,axis=1)
    g=S.groupby(['X','Y','direction']).agg(rows=('CPE','size'),claimed=('CPE','mean'),n_ep_f=('n_ep_f','max'),test_cpe=('cpe_test','mean'),test_base=('base_test','mean'),n_ep_t=('n_ep_t','max'),hit_t=('hit_t','mean'),fires=('fires_now','max'),horizons=('tau_future',lambda s:sorted(set(int(x) for x in s)))).reset_index()
    print(g.sort_values('rows',ascending=False).head(30).round(3).to_string(index=False))
    S.to_parquet(os.path.join(R.OUTD,'extreme_survivors.parquet'))
# diagnosis: test-period lift by q_X and by formation episode count
E['lift_t']=E.cpe_test/E.base_test
print('\nlift by q_X'); print(E.groupby('q_X').agg(n=('X','size'),claimed=('CPE','mean'),test=('cpe_test','mean'),base=('base_test','mean')).round(3))
print('\nlift by n_ep_f bucket'); print(E.groupby(pd.cut(E.n_ep_f,[2,3,5,8,100])).agg(n=('X','size'),claimed=('CPE','mean'),test=('cpe_test','mean'),base=('base_test','mean'),hit_t=('hit_t','mean')).round(3))
print('\nbase rate formation vs test (1-q_Y proxy) by ysub'); print(E.groupby('ysub').agg(base_test=('base_test','mean')).round(3))
