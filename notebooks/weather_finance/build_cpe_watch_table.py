"""Freezes the watch-tier book (batch 5): distinct, non-ENSO conditions, into cpe_climate/cpe_climate_table_b5.parquet and seeds its price store.
Rule (written 2026-10-12 before freezing): watch-tier rows (>= 3 formation episodes, >= 2 test episodes, test hit >= 2/3, test CPE >= base + 0.10, >= 100 test days) from the four batches that are not already frozen;
exclude the ENSO family (already frozen in 224 rows), series with no live feed (ACE), full-record-climatology series (CBTMP) and series needing outside weights (HDDGAS, CDDPOW);
keep ONE row per (predictor family, target, direction): the highest formation conviction conv_f."""
import hashlib, os, sys
import numpy as np, pandas as pd
sys.argv=['x','formation']
import cpe_climate_run as R
CD=R.OUTD
W=pd.read_parquet(os.path.join(CD,'watch_candidates.parquet'))
fam=lambda x: x.split('_')[0]
W['fam']=W.X.map(fam)
excl={'ACE','CBTMP','HDDGAS','CDDPOW','ICE','IOD'}   # ICE, IOD: no live feed (FREEZE_RULES.md)
W=W[(W.fam!='ENSO')&~W.fam.isin(excl)]
W=W[~W.X.map(lambda x: x.split('_')[0] in ('ONI','NINO34','NINO12','NINO3','NINO4','MEI2','SOI'))]
print('candidates after exclusions:',len(W),W.groupby('fam').size().to_dict())
W=W.sort_values('conv_f',ascending=False).drop_duplicates(['fam','Y','direction']).copy()
print('after one-per-(family,target,direction):',len(W)); print(W.groupby('fam').size().to_dict())
px=pd.read_parquet(os.path.join(R.NB,'multiasset_prices_live_history.parquet')); ex=pd.read_parquet(os.path.join(R.HERE,'data','universe_all.parquet'))
for t in R.EXTRA:
    if t in ex.columns and t not in px.columns: px[t]=ex[t].reindex(px.index)
parts=[pd.read_parquet(os.path.join(R.HERE,'data',f)).reindex(px.index) for f in ('climate/climate_daily.parquet','extremes/extremes_daily.parquet','slow/slow_daily.parquet','x4/x4_daily.parquet')]
C=pd.concat(parts,axis=1); C=C.loc[:,~C.columns.duplicated()]
inc_f,fut_f,thr_f,_,_=R.build(px,C,R.CUTOFF)
keep=['Y','X','tau_past','tau_future','q_X','q_Y','direction','CPE','n_condition','n_ep_f','hit_f','conv_f','cpe_test','base_test','n_test','n_ep_t','hit_t','tier','ysub']
T=W[keep].copy(); T['uncond_prob']=1.0-T.q_Y; T['lift']=T.CPE/T.uncond_prob
def xthr(r): return thr_f[(int(r.tau_past),float(r.q_X))][r.X] if r.direction=='bullish' else thr_f[(int(r.tau_past),round(1-float(r.q_X),10))][r.X]
def ythr(r): return thr_f[(int(r.tau_future),float(r.q_Y))][r.Y] if r.direction=='bullish' else thr_f[(int(r.tau_future),round(1-float(r.q_Y),10))][r.Y]
T['x_threshold']=T.apply(xthr,axis=1); T['y_threshold']=T.apply(ythr,axis=1)
T['threshold_frozen_through']='2012-12-31'; T['table_frozen']='2026-10-14'
T=T.sort_values(['Y','X','tau_past','tau_future','q_X','q_Y']).reset_index(drop=True)
path=os.path.join(CD,'cpe_climate_table_b5.parquet'); T.to_parquet(path,index=False)
print(len(T),'rows; sha',hashlib.sha1(open(path,'rb').read()).hexdigest()[:10]); print('distinct conditions',T.groupby(['X','tau_past','q_X','direction']).ngroups,'| targets',T.Y.nunique(),'| X series',sorted(T.X.map(lambda x:x.rsplit('_',1)[0]).unique()))
Y=sorted(T.Y.unique()); P=pd.DataFrame(index=px.index)
for y in Y: P[y]=px[y]
P.to_parquet(os.path.join(CD,'prices_b5.parquet')); print('prices',P.shape,'Y',Y)
print(T[['X','Y','direction','tau_past','tau_future','q_X','n_ep_f','n_ep_t','CPE','cpe_test','base_test','tier']].round(3).to_string(index=False))
