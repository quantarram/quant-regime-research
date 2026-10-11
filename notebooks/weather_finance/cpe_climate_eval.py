"""Episode-conviction audit (backtest_engine.py logic) and chronological out-of-sample evaluation of the climate CPE rows, with thresholds frozen at the formation cutoff."""
import sys, os
import numpy as np, pandas as pd
MODE=os.environ.get('EVAL_MODE','formation')
sys.argv=['x','formation']
import cpe_climate_run as R
import backtest_engine as be
CUT=R.CUTOFF
px,C=R.load_panel()
inc_f,fut_f,thr_f,pred,climate=R.build(px,C,CUT)
inc_a,fut_a,thr_a,_,_=R.build(px,C,None)
rows=pd.read_parquet(os.path.join(R.OUTD,f'{MODE}_rows.parquet'))
import economic_prior as ep
rows['tier']=[ep.get_confidence(ep.get_subclass(x),ep.get_subclass(y)) for x,y in zip(rows.X,rows.Y)]
rows['ysub']=[ep.get_subclass(y) for y in rows.Y]
end=px.index.max()
def cond_mask(inc,x,tp,q,direction,thr):
    s=inc[tp][x]
    return (s>thr[(tp,q)][x]) if direction=='bullish' else (s<thr[(tp,round(1-q,10))][x])
out=[]
for (x,tp,q,d),grp in rows.groupby(['X','tau_past','q_X','direction']):
    fire_f=cond_mask(inc_f,x,tp,q,d,thr_f); fd_f=fire_f[fire_f.fillna(False)].index
    fire_a=cond_mask(inc_a,x,tp,q,d,thr_f); fd_t=fire_a[fire_a.fillna(False)&(fire_a.index>CUT)].index
    ep_f=be._cluster_into_episodes(fd_f,tp) if len(fd_f) else []; ep_t=be._cluster_into_episodes(fd_t,tp) if len(fd_t) else []
    anc_f=[e[-1] for e in ep_f]; anc_t=[e[-1] for e in ep_t]
    for r in grp.itertuples():
        y,tf,qy=r.Y,int(r.tau_future),float(r.q_Y)
        ty=thr_f[(tf,qy)][y] if d=='bullish' else thr_f[(tf,round(1-qy,10))][y]
        def ev(series): return (series>ty) if d=='bullish' else (series<ty)
        ff=fut_f[tf][y]; fa=fut_a[tf][y]
        o_f=[ev(ff.get(a,np.nan)) for a in anc_f if not pd.isna(ff.get(a,np.nan))]
        n_ep=len(o_f); hit=float(np.mean(o_f)) if n_ep else np.nan
        conv=float(np.log(n_ep)*max(0,2*hit-1)) if n_ep>=3 else 0.0
        # test: daily CPE with frozen thresholds, base rate, episodes
        v=fa.loc[fd_t].dropna(); n_t=len(v); cpe_t=float(ev(v).mean()) if n_t else np.nan
        tdates=fa.index[fa.index>CUT]; vb=fa.loc[tdates].dropna(); base_t=float(ev(vb).mean()) if len(vb) else np.nan
        o_t=[ev(fa.get(a,np.nan)) for a in anc_t if not pd.isna(fa.get(a,np.nan))]
        out.append((r.Index,n_ep,hit,conv,n_t,cpe_t,base_t,len(o_t),float(np.mean(o_t)) if o_t else np.nan))
E=pd.DataFrame(out,columns=['idx','n_ep_f','hit_f','conv_f','n_test','cpe_test','base_test','n_ep_t','hit_t']).set_index('idx')
R2=rows.join(E); R2.to_parquet(os.path.join(R.OUTD,f'{MODE}_eval.parquet'))
print('rows',len(R2),'| episodes>=3 in formation:',int((R2.n_ep_f>=3).sum()))
print(R2.groupby('tier').agg(rows=('CPE','size'),ep3=('n_ep_f',lambda s:int((s>=3).sum()))))
adm=R2[R2.n_ep_f>=3].copy(); adm['lift_test']=adm.cpe_test/adm.base_test
t=adm[adm.n_test>=100]
print('\nADMITTED (>=3 formation episodes):',len(adm),' with >=100 test days:',len(t))
print('claimed (formation) CPE mean %.3f | test CPE mean %.3f | test base rate mean %.3f | mean test lift %.2f | share with test lift>1: %.2f | share test CPE >= 0.60: %.2f'%(t.CPE.mean(),t.cpe_test.mean(),t.base_test.mean(),t.lift_test.mean(),(t.lift_test>1).mean(),(t.cpe_test>=0.6).mean()))
for g,name in ((t.groupby('ysub'),'target class'),(t.groupby('tier'),'tier'),(t.groupby('X'),'predictor'),(t.groupby('tau_future'),'horizon')):
    print(f'\nby {name}:'); print(name and g.agg(n=('CPE','size'),claimed=('CPE','mean'),test=('cpe_test','mean'),base=('base_test','mean'),lift_gt1=('lift_test',lambda s:(s>1).mean())).round(3).to_string())
