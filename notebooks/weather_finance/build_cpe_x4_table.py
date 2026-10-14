"""Freezes the batch-4 survivors into cpe_climate/cpe_climate_table_b4.parquet (with the formation-period X and Y thresholds the claims were computed on) and seeds their price store."""
import hashlib, os, sys
import numpy as np, pandas as pd
sys.argv=['x','formation']
import cpe_x4_run as ER
R=ER.R
px,C=R.load_panel(); inc_f,fut_f,thr_f,pred,climate=R.build(px,C,R.CUTOFF)
S=pd.read_parquet(os.path.join(R.OUTD,'x4_survivors.parquet')).reset_index(drop=True)
keep=['Y','X','tau_past','tau_future','q_X','q_Y','direction','CPE','n_condition','n_ep_f','hit_f','conv_f','cpe_test','base_test','n_test','n_ep_t','hit_t','tier','ysub']
T=S[keep].copy(); T['uncond_prob']=1.0-T.q_Y; T['lift']=T.CPE/T.uncond_prob
def xthr(r): return thr_f[(int(r.tau_past),float(r.q_X))][r.X] if r.direction=='bullish' else thr_f[(int(r.tau_past),round(1-float(r.q_X),10))][r.X]
def ythr(r): return thr_f[(int(r.tau_future),float(r.q_Y))][r.Y] if r.direction=='bullish' else thr_f[(int(r.tau_future),round(1-float(r.q_Y),10))][r.Y]
T['x_threshold']=T.apply(xthr,axis=1); T['y_threshold']=T.apply(ythr,axis=1)
T['threshold_frozen_through']='2012-12-31'; T['table_frozen']='2026-10-14'
T=T.sort_values(['Y','X','tau_past','tau_future','q_X','q_Y']).reset_index(drop=True)
path=os.path.join(R.OUTD,'cpe_climate_table_b4.parquet'); T.to_parquet(path,index=False)
print(len(T),'rows; sha',hashlib.sha1(open(path,'rb').read()).hexdigest()[:10]); print(T.groupby(['X','direction']).size().to_dict())
Y=sorted(T.Y.unique()); print('Y needed',Y); missing=[y for y in Y if y not in px.columns]; print('missing from panel:',missing)
P=pd.DataFrame(index=px.index)
for y in Y: P[y]=px[y]
P.to_parquet(os.path.join(R.OUTD,'prices_b4.parquet')); print(P.shape,P.index.max().date(),P.tail(3).notna().sum().sum(),'non-null in last 3 rows')
