"""Freezes the 43 climate survivors into notebooks/cpe_climate_table.parquet (with the formation-period X and Y thresholds the claims were computed on) and caches the NOAA index files."""
import hashlib, os, shutil, sys
import numpy as np, pandas as pd
sys.argv=['x','formation']
import cpe_climate_run as R
NB=R.NB
px,C=R.load_panel(); inc_f,fut_f,thr_f,pred,climate=R.build(px,C,R.CUTOFF)
S=pd.read_parquet(os.path.join(R.OUTD,'survivors.parquet')).reset_index(drop=True)
keep=['Y','X','tau_past','tau_future','q_X','q_Y','direction','CPE','n_condition','n_ep_f','hit_f','conv_f','cpe_test','base_test','n_test','n_ep_t','hit_t','tier','ysub']
T=S[keep].copy()
T['uncond_prob']=1.0-T.q_Y; T['lift']=T.CPE/T.uncond_prob
def xthr(r): return thr_f[(int(r.tau_past),float(r.q_X))][r.X] if r.direction=='bullish' else thr_f[(int(r.tau_past),round(1-float(r.q_X),10))][r.X]
def ythr(r): return thr_f[(int(r.tau_future),float(r.q_Y))][r.Y] if r.direction=='bullish' else thr_f[(int(r.tau_future),round(1-float(r.q_Y),10))][r.Y]
T['x_threshold']=T.apply(xthr,axis=1); T['y_threshold']=T.apply(ythr,axis=1)
T['threshold_frozen_through']='2012-12-31'; T['table_frozen']='2026-10-10'
T=T.sort_values(['Y','X','tau_past','tau_future','q_X','q_Y']).reset_index(drop=True)
T.to_parquet(os.path.join(NB,'cpe_climate_table.parquet'),index=False)
sha=hashlib.sha1(open(os.path.join(NB,'cpe_climate_table.parquet'),'rb').read()).hexdigest()[:10]
d=os.path.join(NB,'cpe_climate_data'); os.makedirs(d,exist_ok=True)
for f in ('oni.txt','sstoi.txt','ao.txt','nao.txt','pdo.txt','soi.txt'): shutil.copy(os.path.join(R.HERE,'data','climate',f),os.path.join(d,f))
print(len(T),'rows; sha',sha); print(T.groupby(['X','direction']).size().to_dict()); print(T[['Y','X','tau_past','tau_future','q_X','q_Y','x_threshold','y_threshold']].head(6).round(3).to_string(index=False))
print('X needed:',sorted(T.X.unique()),'Y needed:',sorted(T.Y.unique()))
