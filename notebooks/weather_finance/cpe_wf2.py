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
# ---- uniform causal walk-forward by predictor-tail rarity, same admission for weather, climate and market
YEARS=list(range(2016,2026)); ys_pos={y:int(idx.searchsorted(pd.Timestamp(f'{y}-01-01'))) for y in YEARS+[2026]}
regcache={}
def reg(x,y):
    k=(x,y)
    if k not in regcache: regcache[k]=ep.is_admissible(x,y)
    return regcache[k]
EVTAIL=0.2
E,V,nm=events(EVTAIL); tf_of=np.array([k[1] for k in nm]); y_of=[k[0] for k in nm]; side_of=[k[2] for k in nm]   # up, dn alternate
OUTCOME={}   # (y,tf,side) -> (o array with NaN, forward-return array)
for jj,(y,tf,sd) in enumerate(nm):
    fa=FW[(y,tf)]; OUTCOME[(y,tf,sd)]=np.where(~np.isnan(fa),E[jj],np.nan)
month=idx.month.values; year=idx.year.values
rows=[]; t0=time.time()
for tail in (0.2,0.1,0.05,0.02):
  for arm,(P,tps) in ARMS.items():
    for tp in tps:
        S=P-P.shift(tp)
        for x in S.columns:
            s_=S[x].values; f=S[x].iloc[:cutpos].dropna()
            if len(f)<500: continue
            hi,lo=np.quantile(f.values,1-tail),np.quantile(f.values,tail)
            for xs,fire,esd in (('hi',s_>=hi,'up'),('lo',s_<=lo,'dn')):
                fire=np.where(np.isnan(s_),False,fire); fpos=np.where(fire)[0]
                fpf=fpos[fpos<cutpos]
                if len(fpf)<100: continue
                ep_f=episodes(fpf,tp)
                if ep_f<4: continue
                eps=[]
                if len(fpos):
                    d=idx[fpos]; gap=tp*1.45*GAP; cut=np.where(np.diff(d.values).astype('timedelta64[D]').astype(int)>gap)[0]+1; eps=np.split(fpos,cut)
                fv=fire.astype(np.float32)
                for tf in TF:
                    j=np.where(tf_of==tf)[0]; fm=fv*formmask[tf]; n1=V[j]@fm; n11=E[j]@fm; bf=(E[j]@formmask[tf])/np.maximum(V[j]@formmask[tf],1)
                    for q in range(0,len(j),2):
                        k=q+(0 if esd=='up' else 1); jj=j[k]
                        y=y_of[jj]
                        if arm=='market' and x==y: continue
                        if n1[k]<100 or n11[k]/n1[k]<1.5*bf[k] or n11[k]==0 or not reg(x,y): continue
                        o=OUTCOME[(y,tf,esd)]; fa=FW[(y,tf)]; okv=~np.isnan(o)
                        sg=1.0 if esd=='up' else -1.0
                        entries=np.array([int(e[okv[e]][0]) for e in eps if okv[e].any()],dtype=int)
                        if len(entries)==0: continue
                        for Y in YEARS:
                            Sx=ys_pos[Y]; lim=Sx-tf
                            e_t=entries[(entries>=cutpos)&(entries<lim)]
                            if len(e_t)<3: continue
                            fd=np.where(fire&okv&(pos>=cutpos)&(pos<lim))[0]
                            if len(fd)<100: continue
                            allv=okv&(pos>=cutpos)&(pos<lim); base=float(np.mean(o[allv])) if allv.any() else np.nan
                            cpe=float(np.mean(o[fd]))
                            if not (base>0 and cpe>=1.5*base): continue
                            e_y=entries[(entries>=Sx)&(entries<ys_pos[Y+1])]
                            if len(e_y)==0: continue
                            yr=okv&(pos>=Sx)&(pos<ys_pos[Y+1]); by=float(np.mean(o[yr])); dr=float(np.mean(sg*fa[yr][~np.isnan(fa[yr])]))
                            pre=okv&(pos<lim)
                            for e in e_y:
                                mk=okv&(year==Y)&(month==month[e]); bm=float(np.mean(o[mk])) if mk.sum()>10 else np.nan
                                dm=float(np.mean(sg*fa[mk])) if mk.sum()>10 else np.nan
                                rows.append((tail,arm,x,tp,xs,y,tf,Y,idx[e],o[e],sg*fa[e],by,dr,bm,dm))
    print(tail,arm,len(rows),f'{time.time()-t0:.0f}s',flush=True)
W=pd.DataFrame(rows,columns=['tail','arm','X','tau_p','xside','Y','tau_f','year','entry','hit','ret','base_year','drift_year','base_month','drift_month'])
W.to_parquet(os.path.join(HERE,'cpe_climate','wf2s.parquet')); print('entries',len(W))
