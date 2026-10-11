"""Decision-rule backtest on the CAUSAL walk-forward events (cpe_walkforward.py): the exact position rule the live layer uses.
Rule: an entry event (first firing day of an episode of an admitted row) opens a vote on (target, direction) from the next row until its horizon (rows of the price calendar) expires.
One vote per (target, direction); opposite votes cancel; equal weight 1/N across the N targets with a nonzero vote; rebalanced daily; cost 10 bp per unit of turnover (one way).
Benchmarks: the same targets held LONG at the same equal weights over the same windows (same-instrument buy and hold), SPY. Year range 2016-2026."""
import os, sys, json
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); NB=os.path.abspath(os.path.join(HERE,'..'))
SET=os.environ.get('WF_SET','climate'); COST=0.0010
px=pd.read_parquet(os.path.join(NB,'multiasset_prices_live_history.parquet')); ex=pd.read_parquet(os.path.join(HERE,'data','universe_all.parquet'))
for t in ['KC=F','CC=F','CT=F','SB=F','OJ=F','LE=F','HE=F','ZL=F','ZM=F','KE=F','HO=F','RB=F','ZO=F']:
    if t in ex.columns and t not in px.columns: px[t]=ex[t].reindex(px.index)
idx=px.index; n=len(idx); pos_of=lambda ts:int(idx.searchsorted(pd.Timestamp(ts)))
E=pd.read_parquet(os.path.join(HERE,'cpe_climate',f'walkforward_{SET}.parquet'))
rets={}
def ret_series(y):
    if y not in rets:
        s=px[y].dropna(); r=np.log(s/s.shift(1)).dropna(); rets[y]=r.reindex(idx).fillna(0.0).values
    return rets[y]
spy=ret_series('SPY')
def run(Ev,label,verbose=True):
    if len(Ev)==0: return None
    start=pos_of(f"{Ev.entry.min().year}-01-01")
    votes={}   # (Y) -> array of +1/-1 counts
    for r in Ev.itertuples():
        e=pos_of(r.entry); a=votes.setdefault(r.Y,np.zeros((2,n),dtype=np.int16)); k=0 if r.direction=='bullish' else 1
        a[k,e+1:min(n,e+1+int(r.tau_future))]+=1
    ys=sorted(votes); W=np.zeros((len(ys),n)); L=np.zeros((len(ys),n))
    for i,y in enumerate(ys):
        a=votes[y]; v=np.where((a[0]>0)&(a[1]==0),1.0,np.where((a[1]>0)&(a[0]==0),-1.0,0.0)); W[i]=v; L[i]=(a.sum(0)>0).astype(float)
    N=np.maximum((W!=0).sum(0),1); w=W/N; wl=L/np.maximum(L.sum(0),1)
    # weights apply to the return of the day AFTER the vote becomes active (votes were opened from the next row already)
    R=np.array([ret_series(y) for y in ys])
    gross=(w*R).sum(0); cost=COST*np.abs(np.diff(w,axis=1,prepend=0)).sum(0); strat=gross-cost; bh=(wl*R).sum(0)
    act=(w!=0).any(0)
    sl=slice(start,n); s=pd.Series(strat[sl],index=idx[sl]); b=pd.Series(bh[sl],index=idx[sl]); sp=pd.Series(spy[sl],index=idx[sl]); a_=pd.Series(act[sl],index=idx[sl])
    mt=float((W!=0).sum(0)[sl][a_.values].mean()) if a_.any() else 0.0; to=float(np.abs(np.diff(w,axis=1,prepend=0)).sum(0)[sl].sum()/max(1,a_.sum())*252)
    # trading-day basis: weekend rows (calendar since 2014 includes weekends) are rolled into the next weekday
    bd=pd.Series([d if d.weekday()<5 else d+pd.offsets.BDay(1) for d in s.index],index=s.index)
    s,b,sp=s.groupby(bd).sum(),b.groupby(bd).sum(),sp.groupby(bd).sum(); a_=a_.groupby(bd).max()
    # stats on active days only (cash otherwise) and over the whole window
    def st(p,benchmark):
        sd=p.std(); beta=float(p.cov(benchmark)/benchmark.var()) if benchmark.var()>0 else np.nan
        return dict(ann=float(p.mean()*252),vol=float(sd*np.sqrt(252)),sharpe=float(p.mean()/sd*np.sqrt(252)) if sd>0 else np.nan,beta=beta,alpha=float((p.mean()-beta*benchmark.mean())*252))
    full=st(s,sp); fb=st(b,sp); sa=s[a_]; ba=b[a_]; sa_st=st(sa,sp[a_]); ba_st=st(ba,sp[a_])
    res=dict(label=label,events=len(Ev),days=len(s),active_days=int(a_.sum()),strategy_full=full,hold_same_full=fb,strategy_active=sa_st,hold_same_active=ba_st,
             mean_targets=mt,turnover=to)
    if verbose:
        print(f"{label:34s} ev {len(Ev):4d} act {a_.sum():5d}d | strat ann {full['ann']*100:+6.2f}% sharpe {full['sharpe']:+.2f} alpha {full['alpha']*100:+6.2f}% | hold-same ann {fb['ann']*100:+6.2f}% sharpe {fb['sharpe']:+.2f} | ACTIVE days: strat {sa_st['ann']*100:+6.1f}% vs hold-same {ba_st['ann']*100:+6.1f}% (diff {(sa_st['ann']-ba_st['ann'])*100:+.1f} pts) vs SPY-beta alpha {sa_st['alpha']*100:+.1f}%")
    return res,s,b,sp,a_
if __name__=='__main__':
    A=E[E.admitted]
    out={}
    for lab,D in (('all admitted',A),('extreme tail q_X>=0.9',A[A.q_X>=0.9]),('moderate q_X<0.9',A[A.q_X<0.9])):
        r=run(D,lab); 
        if r: out[lab]=r[0]
    if SET=='climate':
        for b in sorted(A.batch.unique()):
            run(A[(A.batch==b)&(A.q_X>=0.9)],f'  extreme, {b}')
        r=run(A[A.q_X>=0.9],'extreme, no costs (diagnostic)') if False else None
    res,s,b,sp,a_=run(A[A.q_X>=0.9] if SET=='climate' else A,'primary',verbose=False)
    yr=pd.DataFrame({'strat':s,'hold':b,'spy':sp,'act':a_}); y=yr.groupby(yr.index.year).agg(strat=('strat','sum'),hold=('hold','sum'),spy=('spy','sum'),active_days=('act','sum'))
    print('\nby year (log-return sums; active days):'); print((y[['strat','hold','spy']]*100).round(1).join(y.active_days).to_string())
    json.dump(out,open(os.path.join(HERE,'cpe_climate',f'wf_portfolio_{SET}.json'),'w'),indent=1)
    pd.DataFrame({'strat':s,'hold':b,'spy':sp,'active':a_}).to_parquet(os.path.join(HERE,'cpe_climate',f'wf_portfolio_{SET}_daily.parquet'))
