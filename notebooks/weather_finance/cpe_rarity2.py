"""Rarity gradient: does persistence rise as the conditioning event gets rarer? One admission rule (formation only: >= 100 conditioned days, >= 4 episodes, conditional probability >= 1.5 x base), scored 2013-2026 as out-of-sample lift.
Arms: weather (daily degree-day anomaly windows 1-252 d), climate (31 monthly states, windows 21-252 d), market (each instrument's own log-price move over 1-252 d, the program's own predictor class).
Predictor tail TAIL_X in {20%, 10%, 5%, 2%}; event tail either matched (= TAIL_X) or fixed at a frequent 20%. Registered-channel pairs only (is_admissible), plus all pairs for reference."""
import os, sys, time
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); NB=os.path.abspath(os.path.join(HERE,'..'))
sys.path.insert(0,os.path.join(NB,'files')); sys.path.insert(0,os.path.join(HERE,'cpe_climate'))
import backtest_engine as be, economic_prior as ep
CUT=pd.Timestamp('2012-12-31'); GAP=be.EPISODE_GAP_MULTIPLIER
px=pd.read_parquet(os.path.join(NB,'multiasset_prices_live_history.parquet')); ex=pd.read_parquet(os.path.join(HERE,'data','universe_all.parquet'))
for t in ['KC=F','CC=F','CT=F','SB=F','OJ=F','LE=F','HE=F','ZL=F','ZM=F','KE=F','HO=F','RB=F','ZO=F']:
    if t in ex.columns and t not in px.columns: px[t]=ex[t].reindex(px.index)
idx=px.index; T=len(idx); cutpos=int(idx.searchsorted(CUT+pd.Timedelta(days=1))); pos=np.arange(T)
inst=[c for c in px.columns if px.loc[:CUT,c].notna().sum()>=1000 and not c.startswith('^')]
rd=lambda f: pd.read_parquet(os.path.join(HERE,'data',f)).reindex(idx)
WX=rd('wx/wx_daily.parquet'); CL=pd.concat([rd('climate/climate_daily.parquet'),rd('extremes/extremes_daily.parquet'),rd('slow/slow_daily.parquet'),rd('x4/x4_daily.parquet')],axis=1)
CL=CL[[c for c in CL.columns if c.endswith('_mean') and not c.startswith('ACE')]]
MK=np.log(px[inst])
ARMS={'weather':(WX,[1,3,7,14,28,63,126,252]),'climate':(CL,[21,63,126,252]),'market':(MK,[1,3,7,14,28,63,126,252])}; TF=[1,5,10,21,63,126,252]
tf_arr=[]; names=[]; fwdq={}
FW={}
for tk in inst:
    p=px[tk]
    for tf in TF: FW[(tk,tf)]=np.log(p.shift(-tf)/p).values
def events(tail):
    E=[];V=[];nm=[]
    for tk in inst:
        for tf in TF:
            fwd=FW[(tk,tf)]; ok=~np.isnan(fwd); form=ok&(pos<=cutpos-1-tf); lo,hi=np.nanquantile(fwd[form],tail),np.nanquantile(fwd[form],1-tail)
            for side,e in (('up',fwd>=hi),('dn',fwd<=lo)): nm.append((tk,tf,side)); E.append((e&ok).astype(np.float32)); V.append(ok.astype(np.float32))
    return np.array(E),np.array(V),nm
def episodes(mpos,tp):
    if len(mpos)==0: return 0
    d=idx[mpos]; return int(1+np.sum(np.diff(d.values).astype('timedelta64[D]').astype(int)>tp*1.45*GAP))
formmask={tf:(pos<=cutpos-1-tf).astype(np.float32) for tf in TF}; testmask=(pos>=cutpos).astype(np.float32)
# registered-channel matrix (X name x Y)
regcache={}
def reg(x,y):
    k=(x,y)
    if k not in regcache: regcache[k]=ep.is_admissible(x,y)
    return regcache[k]
out=[]; t0=time.time()
for ev_mode in ('matched','fixed20'):
  for tail in (0.2,0.1,0.05,0.02):
    et=tail if ev_mode=='matched' else 0.2
    E,V,nm=events(et); tf_of=np.array([k[1] for k in nm]); y_of=[k[0] for k in nm]
    for arm,(P,tps) in ARMS.items():
        for tp in tps:
            S=P-P.shift(tp)
            for x in S.columns:
                s=S[x].values; f=S[x].iloc[:cutpos].dropna()
                if len(f)<500: continue
                hi,lo=np.quantile(f.values,1-tail),np.quantile(f.values,tail)
                for side,fire in (('hi',s>=hi),('lo',s<=lo)):
                    fire=np.where(np.isnan(s),False,fire); fpf=np.where(fire&(pos<cutpos))[0]
                    if len(fpf)<100: continue
                    ep_f=episodes(fpf,tp)
                    if ep_f<4: continue
                    fv=fire.astype(np.float32)
                    for tf in TF:
                        j=np.where(tf_of==tf)[0]; fm=fv*formmask[tf]; n1=V[j]@fm; n11=E[j]@fm; bf=(E[j]@formmask[tf])/np.maximum(V[j]@formmask[tf],1)
                        ft=fv*testmask; n1t=V[j]@ft; n11t=E[j]@ft; bt=(E[j]@testmask)/np.maximum(V[j]@testmask,1)
                        pf=n11/np.maximum(n1,1); pt=n11t/np.maximum(n1t,1)
                        for q in range(0,len(j),2):          # j alternates up, dn for each instrument
                            y=y_of[j[q]]
                            if arm=='market' and x==y: continue
                            if n1[q]<100 or n1t[q]<100: continue
                            au=pf[q]>=1.5*bf[q] and n11[q]>0 and bt[q]>0; ad=pf[q+1]>=1.5*bf[q+1] and n11[q+1]>0 and bt[q+1]>0
                            if not (au or ad): continue
                            if not reg(x,y): continue
                            out.append((ev_mode,tail,arm,x,tp,side,y,tf,int(n1[q]),ep_f,int(n1t[q]),pf[q],bf[q],pt[q],bt[q],pf[q+1],bf[q+1],pt[q+1],bt[q+1],au,ad))
    print(ev_mode,tail,len(out),f'{time.time()-t0:.0f}s',flush=True)
D=pd.DataFrame(out,columns=['ev_mode','tail','arm','X','tau_p','xside','Y','tau_f','n_form','ep_form','n_test','pf_up','bf_up','pt_up','bt_up','pf_dn','bf_dn','pt_dn','bt_dn','adm_up','adm_dn'])
D.to_parquet(os.path.join(HERE,'cpe_climate','rarity2.parquet')); print('configs',len(D))
