"""Survivors for the batch-4 slow climate states, same pre-registered filter as cpe_climate_live.py, plus the registered watch tier."""
import sys, os
import numpy as np, pandas as pd
sys.argv=['x','formation']
import cpe_x4_run as ER
R=ER.R; CUT=R.CUTOFF
px,C=R.load_panel(); inc_a,fut_a,thr_a,_,_=R.build(px,C,None); inc_f,fut_f,thr_f,_,_=R.build(px,C,CUT)
E=pd.read_parquet(os.path.join(R.OUTD,'x4_formation_eval.parquet'))
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
S=E[(E.n_ep_f>=4)&(E.n_ep_t>=3)&(E.hit_t>=2/3)&(E.cpe_test>=E.base_test+0.10)&(E.n_test>=100)].copy()
W=E[(E.n_ep_f>=3)&(E.n_ep_t>=2)&(E.hit_t>=2/3)&(E.cpe_test>=E.base_test+0.10)&(E.n_test>=100)].copy()
print('admitted',len(E),'| with >=4 formation episodes',int((E.n_ep_f>=4).sum()),'| survivors',len(S),'| watch tier',len(W))
for nm,D in (('survivors',S),):
    print(nm,'by X:',D.groupby('X').size().to_dict()); print(nm,'by ysub:',D.groupby('ysub').size().to_dict()); print(nm,'by Y:',D.groupby('Y').size().sort_values(ascending=False).head(15).to_dict())
    print(nm,'by tau_future:',D.groupby('tau_future').size().to_dict(),'| q_X>=0.9:',int((D.q_X>=0.9).sum()),'| tier:',D.groupby('tier').size().to_dict())
def fires_now(r):
    s=inc_a[int(r.tau_past)][r.X]; v=s.iloc[-1]
    if pd.isna(v): return np.nan
    thr=thr_f[(int(r.tau_past),float(r.q_X))][r.X] if r.direction=='bullish' else thr_f[(int(r.tau_past),round(1-float(r.q_X),10))][r.X]
    return bool(v>thr) if r.direction=='bullish' else bool(v<thr)
if len(S):
    S['fires_now']=S.apply(fires_now,axis=1)
    g=S.groupby(['X','Y','direction']).agg(rows=('CPE','size'),claimed=('CPE','mean'),n_ep_f=('n_ep_f','max'),test_cpe=('cpe_test','mean'),test_base=('base_test','mean'),n_ep_t=('n_ep_t','max'),hit_t=('hit_t','mean'),fires=('fires_now','max'),horizons=('tau_future',lambda s:sorted(set(int(x) for x in s)))).reset_index()
    print(g.sort_values('rows',ascending=False).head(40).round(3).to_string(index=False))
    S.to_parquet(os.path.join(R.OUTD,'x4_survivors.parquet'))
W.to_parquet(os.path.join(R.OUTD,'x4_watch.parquet'))
# standard-tier survivors only, and extreme-tail subset
print('\nstandard-tier survivors:',int((S.tier=='standard').sum()),'| standard-tier admitted rows:',int(((E.tier=="standard")&(E.n_ep_f>=4)).sum()))
print('survival rate among rows with >=4 formation episodes: %.2f%%'%(100*len(S)/max(1,int((E.n_ep_f>=4).sum()))))
