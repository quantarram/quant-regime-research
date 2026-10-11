"""Horizon ladder: identical admission and scoring for short-horizon WEATHER predictors and slow CLIMATE predictors across past-window (tau_p) and forward (tau_f) horizons.
Predictor tail (formation quantile, frozen) -> instrument forward-return tail event (formation quantile, frozen). Admission uses formation only (outcomes purged at the cutoff):
  >= 100 conditioned days, >= 4 episodes (the engine's gap rule), P(event | condition) >= 1.5 x base.   Scoring: the same configuration on 2013-2026 (>= 100 conditioned days): P(event | condition) against the test base rate.
No test information enters admission. Both directions of the predictor tail and both directions of the event tail are run. TAIL = 0.2 (20% tails) or 0.1 (10% tails, the extreme reading).
  python cpe_ladder.py 0.2 ; python cpe_ladder.py 0.1"""
import os, sys, time
import numpy as np, pandas as pd
sys.argv_saved=list(sys.argv); TAIL=float(sys.argv[1]) if len(sys.argv)>1 else 0.2
HERE=os.path.dirname(os.path.abspath(__file__)); NB=os.path.abspath(os.path.join(HERE,'..'))
sys.path.insert(0,os.path.join(NB,'files'))
import backtest_engine as be
CUT=pd.Timestamp('2012-12-31'); GAP_MULT=be.EPISODE_GAP_MULTIPLIER
px=pd.read_parquet(os.path.join(NB,'multiasset_prices_live_history.parquet')); ex=pd.read_parquet(os.path.join(HERE,'data','universe_all.parquet'))
for t in ['KC=F','CC=F','CT=F','SB=F','OJ=F','LE=F','HE=F','ZL=F','ZM=F','KE=F','HO=F','RB=F','ZO=F']:
    if t in ex.columns and t not in px.columns: px[t]=ex[t].reindex(px.index)
idx=px.index; T=len(idx); cutpos=int(idx.searchsorted(CUT+pd.Timedelta(days=1)))
inst=[c for c in px.columns if px.loc[:CUT,c].notna().sum()>=1000 and not c.startswith('^')]
rd=lambda f: pd.read_parquet(os.path.join(HERE,'data',f)).reindex(idx)
WX=rd('wx/wx_daily.parquet'); CL=pd.concat([rd('climate/climate_daily.parquet'),rd('extremes/extremes_daily.parquet'),rd('slow/slow_daily.parquet'),rd('x4/x4_daily.parquet')],axis=1)
CL=CL[[c for c in CL.columns if c.endswith('_mean') and not c.startswith('ACE')]]
ARMS={'weather':(WX,[1,3,7,14,28,63,126,252]),'climate':(CL,[21,63,126,252])}; TF=[1,5,10,21,63,126,252]
# ---- event matrices (instrument x horizon x event direction)
t0=time.time(); ev_names=[]; E=[]; V=[]
for tk in inst:
    p=px[tk]
    for tf in TF:
        fwd=np.log(p.shift(-tf)/p).values; ok=~np.isnan(fwd); form=ok&(np.arange(T)<=cutpos-1-tf)
        lo,hi=np.nanquantile(fwd[form],TAIL),np.nanquantile(fwd[form],1-TAIL)
        for side,e in (('up',fwd>=hi),('dn',fwd<=lo)):
            ev_names.append((tk,tf,side)); E.append((e&ok).astype(np.float32)); V.append(ok.astype(np.float32))
E=np.array(E); V=np.array(V); print('events',E.shape,f'{time.time()-t0:.0f}s',flush=True)
tf_of=np.array([k[1] for k in ev_names]); pos=np.arange(T)
formmask={tf:(pos<=cutpos-1-tf).astype(np.float32) for tf in TF}; testmask=(pos>=cutpos).astype(np.float32)
def episodes(mask_pos,tp):
    if len(mask_pos)==0: return 0
    d=idx[mask_pos]; gap=tp*1.45*GAP_MULT; return int(1+np.sum(np.diff(d.values).astype('timedelta64[D]').astype(int)>gap))
out=[]
for arm,(P,tps) in ARMS.items():
    for tp in tps:
        S=(P-P.shift(tp)); form_rows=S.iloc[:cutpos]
        for x in S.columns:
            s=S[x].values; f=form_rows[x].dropna()
            if len(f)<500: continue
            hi,lo=np.quantile(f.values,1-TAIL),np.quantile(f.values,TAIL)
            for side,fire in (('hi',s>=hi),('lo',s<=lo)):
                fire=np.where(np.isnan(s),False,fire)
                fpos_form=np.where(fire&(pos<cutpos))[0]; fpos_test=np.where(fire&(pos>=cutpos))[0]
                if len(fpos_form)<100: continue
                ep_f=episodes(fpos_form,tp); ep_t=episodes(fpos_test,tp)
                if ep_f<4: continue
                fv=fire.astype(np.float32)
                for tf in TF:
                    j=np.where(tf_of==tf)[0]
                    fm=fv*formmask[tf]; n1=V[j]@fm; n11=E[j]@fm
                    bf=(E[j]@formmask[tf])/np.maximum(V[j]@formmask[tf],1)
                    ft=fv*testmask; n1t=V[j]@ft; n11t=E[j]@ft; bt=(E[j]@testmask)/np.maximum(V[j]@testmask,1)
                    for k,jj in enumerate(j):
                        if n1[k]>=100:
                            out.append((arm,x,tp,side,ev_names[jj][0],tf,ev_names[jj][2],int(n1[k]),n11[k]/n1[k],bf[k],ep_f,int(n1t[k]),(n11t[k]/n1t[k]) if n1t[k]>0 else np.nan,bt[k],ep_t))
    print(arm,'done',len(out),f'{time.time()-t0:.0f}s',flush=True)
D=pd.DataFrame(out,columns=['arm','X','tau_p','xside','Y','tau_f','evside','n_form','p_form','base_form','ep_form','n_test','p_test','base_test','ep_test'])
D.to_parquet(os.path.join(HERE,'cpe_climate',f'ladder_{int(TAIL*100)}.parquet')); print('configs',len(D))
A=D[(D.p_form>=1.5*D.base_form)&(D.p_form>0)&(D.n_test>=100)].copy()
A['lift_test']=A.p_test/A.base_test; A['diff_test']=A.p_test-A.base_test; A['lift_form']=A.p_form/A.base_form
print(f'\nTAIL {TAIL}: admitted configs with >=100 test days: {len(A)} of {len(D)}')
def table(g):
    return A.groupby(g).agg(n=('X','size'),form_lift=('lift_form','mean'),test_lift=('lift_test','mean'),test_diff=('diff_test','mean'),share_up=('diff_test',lambda s:(s>0).mean()),share_rep=('lift_test',lambda s:(s>=1.5).mean())).round(3)
pd.set_option('display.width',250)
print('\nby arm:'); print(table('arm')); print('\nby arm x forward horizon:'); print(table(['arm','tau_f']))
print('\nby arm x past window:'); print(table(['arm','tau_p']))
