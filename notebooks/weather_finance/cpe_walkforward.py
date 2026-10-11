"""Causal walk-forward test of the survivor filter.
At the start of each year Y (2016-2026) a candidate row (>= 4 formation episodes, engine gates passed) is ADMITTED if, using only test-period (2013+) observations whose forward outcome had already resolved before 1 January Y,
it meets the same criteria as the survivor filter: >= 3 episodes with hit rate >= 2/3, daily CPE >= base + 0.10 on >= 100 firing days. The cohort scored is every episode ENTRY (first firing day of an episode) made during year Y.
No choice uses information after 1 January Y. Outputs events with the admission flag; summary printed.
  WF_SET=climate   rows of the four climate batches (batch 1, extreme, slow, x4)
  WF_SET=market    the program's own market predictors under the identical split and prior (positive control)"""
import sys, os
import numpy as np, pandas as pd
sys.argv=['x','formation']
import cpe_climate_run as R
import backtest_engine as be
SET=os.environ.get('WF_SET','climate'); CUT=R.CUTOFF
px=pd.read_parquet(os.path.join(R.NB,'multiasset_prices_live_history.parquet')); ex=pd.read_parquet(os.path.join(R.HERE,'data','universe_all.parquet'))
for t in R.EXTRA:
    if t in ex.columns and t not in px.columns: px[t]=ex[t].reindex(px.index)
if SET=='climate':
    parts=[pd.read_parquet(os.path.join(R.HERE,'data','climate','climate_daily.parquet')),pd.read_parquet(os.path.join(R.HERE,'data','extremes','extremes_daily.parquet')),
           pd.read_parquet(os.path.join(R.HERE,'data','slow','slow_daily.parquet')),pd.read_parquet(os.path.join(R.HERE,'data','x4','x4_daily.parquet'))]
    C=pd.concat([p.reindex(px.index) for p in parts],axis=1)
    files=[('b1','formation_eval'),('b2','extreme_formation_eval'),('b3','slow_formation_eval'),('b4','x4_formation_eval')]
else:
    C=pd.DataFrame(index=px.index); files=[('mkt','standard_formation_eval')]
inc_f,fut_f,thr_f,_,_=R.build(px,C,CUT); inc_a,fut_a,_,_,_=R.build(px,C,None)
KEY=['X','Y','tau_past','tau_future','q_X','q_Y','direction']
rows=pd.concat([pd.read_parquet(os.path.join(R.OUTD,f+'.parquet')).assign(batch=b) for b,f in files],ignore_index=True)
rows=rows[rows.n_ep_f>=4].drop_duplicates(KEY).reset_index(drop=True)
idx=px.index; n=len(idx); pos_of=lambda ts: int(idx.searchsorted(pd.Timestamp(ts)))
print(SET,'candidate rows',len(rows),rows.groupby('batch').size().to_dict(),flush=True)
YEARS=list(range(2016,2027)); cut_pos=pos_of(CUT+pd.Timedelta(days=1)); ys_pos={y:pos_of(f'{y}-01-01') for y in YEARS+[2027]}
ev=[]; adm=[]
for (x,tp,q,d),grp in rows.groupby(['X','tau_past','q_X','direction']):
    s=inc_a[int(tp)][x]; thr=thr_f[(int(tp),float(q))][x] if d=='bullish' else thr_f[(int(tp),round(1-float(q),10))][x]
    fire=((s>thr) if d=='bullish' else (s<thr)).fillna(False).values
    fdates=idx[fire]; eps=be._cluster_into_episodes(fdates,int(tp)) if len(fdates) else []
    ep_pos=[np.array([pos_of(t) for t in e]) for e in eps]
    sgn=1.0 if d=='bullish' else -1.0
    for r in grp.itertuples():
        y,tf,qy=r.Y,int(r.tau_future),float(r.q_Y)
        ty=thr_f[(tf,qy)][y] if d=='bullish' else thr_f[(tf,round(1-qy,10))][y]
        fa=fut_a[tf][y].values; ok=~np.isnan(fa)
        o=np.where(ok,((fa>ty) if d=='bullish' else (fa<ty)).astype(float),np.nan)
        p=np.arange(n)
        entries=[]
        for e in ep_pos:
            v=e[ok[e]]
            if len(v): entries.append(int(v[0]))
        entries=np.array(entries,dtype=int)
        for Y in YEARS:
            S=ys_pos[Y]; lim=S-tf            # an outcome is known before S only if origin < S - tf
            e_t=entries[(entries>=cut_pos)&(entries<lim)]
            n_ep_t=len(e_t); hit_t=float(np.mean(o[e_t])) if n_ep_t else np.nan
            fd=np.where(fire&ok&(p>=cut_pos)&(p<lim))[0]
            n_d=len(fd); cpe=float(np.mean(o[fd])) if n_d else np.nan
            allv=ok&(p>=cut_pos)&(p<lim); base=float(np.mean(o[allv])) if allv.any() else np.nan
            admitted=bool(n_ep_t>=3 and hit_t>=2/3 and n_d>=100 and cpe>=base+0.10)
            e_y=entries[(entries>=S)&(entries<ys_pos[Y+1])]
            if len(e_y)==0: continue
            yr=ok&(p>=S)&(p<ys_pos[Y+1]); b_y=float(np.mean(o[yr])) if yr.any() else np.nan; drift=float(np.mean(sgn*fa[yr])) if yr.any() else np.nan
            for e in e_y:
                ev.append((r.batch,x,y,d,tp,tf,q,qy,r.tier,Y,idx[e],o[e],sgn*fa[e],b_y,drift,admitted,n_ep_t,hit_t,cpe,base))
E=pd.DataFrame(ev,columns=['batch','X','Y','direction','tau_past','tau_future','q_X','q_Y','tier','year','entry','hit','ret','base_year','drift_year','admitted','n_ep_t','hit_t_prior','cpe_prior','base_prior'])
E.to_parquet(os.path.join(R.OUTD,f'walkforward_{SET}.parquet'))
print('events',len(E),'admitted events',int(E.admitted.sum()))
def summ(D,label):
    if D.empty: print(f'{label:34s} none'); return
    print(f"{label:34s} events {len(D):6d}  hit {D.hit.mean():.3f}  base {D.base_year.mean():.3f}  excess hit {D.hit.mean()-D.base_year.mean():+.3f}  mean ret {D.ret.mean()*100:+.2f}%  drift {D.drift_year.mean()*100:+.2f}%  excess {(D.ret-D.drift_year).mean()*100:+.2f}%")
A=E[E.admitted]; N=E[~E.admitted]
summ(A,'ADMITTED (causal)'); summ(N,'not admitted (>=4 formation ep)')
print('\nby year, admitted:'); [summ(A[A.year==y],f'  {y}') for y in YEARS]
print('\nadmitted by batch:'); [summ(A[A.batch==b],f'  {b}') for b in sorted(A.batch.unique())]
print('admitted by tier:'); [summ(A[A.tier==t],f'  {t}') for t in sorted(A.tier.unique())]
print('admitted q_X>=0.9:'); summ(A[A.q_X>=0.9],'  extreme tail'); summ(A[A.q_X<0.9],'  moderate tail')
print('admitted by horizon:'); [summ(A[A.tau_future==h],f'  {h}') for h in sorted(A.tau_future.unique())]
# distinct-trade view: collapse duplicates of the same target, direction, entry month and horizon
dd=A.assign(m=A.entry.dt.to_period('M')).drop_duplicates(['Y','direction','tau_future','m']); summ(dd,'ADMITTED distinct (Y,dir,h,month)')
