"""Exports the causal walk-forward and decision-rule backtest (climate and the market-predictor control) to a compact JSON for the dashboard and the paper."""
import os, sys, json
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,'cpe_climate')
def ev_stats(D):
    if len(D)==0: return None
    exc=D.ret-D.drift_year; yr=exc.groupby(D.year).mean()
    dd=D.assign(m=D.entry.dt.to_period('M')).drop_duplicates(['Y','direction','tau_future','m'])
    return dict(events=int(len(D)),distinct_trades=int(len(dd)),hit=round(float(D.hit.mean()),4),base=round(float(D.base_year.mean()),4),hit_diff=round(float(D.hit.mean()-D.base_year.mean()),4),
                mean_return=round(float(D.ret.mean()),4),mean_excess=round(float(exc.mean()),4),excess_sd=round(float(exc.std()),4),t_events=round(float(exc.mean()/(exc.std()/np.sqrt(len(D)))),2) if len(D)>2 else None,
                t_year_clustered=round(float(yr.mean()/(yr.std()/np.sqrt(len(yr)))),2) if len(yr)>2 else None,years=int(len(yr)))
out={'rule':'one vote per (target, direction) from the next row for the row horizon; opposite votes cancel; equal weight over active targets; 10 bp per unit of turnover; admission re-decided on 1 January each year from resolved outcomes only',
     'window':'entries 2016-2026 (admission needs 3 resolved test episodes, so the first admissions are in 2016)'}
for name in ('climate','market'):
    E=pd.read_parquet(os.path.join(OUT,f'walkforward_{name}.parquet')); A=E[E.admitted]; N=E[~E.admitted]
    d={'candidate_events':int(len(E)),'admitted':ev_stats(A),'not_admitted':ev_stats(N),'extreme':ev_stats(A[A.q_X>=0.9]),'moderate':ev_stats(A[A.q_X<0.9])}
    if name=='climate':
        d['by_batch_extreme']={b:ev_stats(A[(A.batch==b)&(A.q_X>=0.9)]) for b in sorted(A.batch.unique())}
        d['extreme_standard_tier']=ev_stats(A[(A.q_X>=0.9)&(A.tier=='standard')]); d['extreme_weak_tier']=ev_stats(A[(A.q_X>=0.9)&(A.tier=='weak')])
        d['extreme_events_by_year']={int(k):int(v) for k,v in A[A.q_X>=0.9].groupby('year').size().items()}
    out[name]=d
# portfolio
for name in ('climate','market'):
    j=json.load(open(os.path.join(OUT,f'wf_portfolio_{name}.json')))
    out[name]['portfolio']=j
    D=pd.read_parquet(os.path.join(OUT,f'wf_portfolio_{name}_daily.parquet'))
    w=D[['strat','hold','spy']].resample('W').sum().cumsum()
    out[name]['curves']={'dates':[d.strftime('%Y-%m-%d') for d in w.index],'strategy':w.strat.round(4).tolist(),'hold_same':w.hold.round(4).tolist(),'spy':w.spy.round(4).tolist(),
                         'basis':'primary arm: climate = extreme tail (q_X >= 0.9) admitted rows; market = all admitted rows'}
json.dump(out,open(os.path.join(OUT,'wf_export.json'),'w'),indent=1); print('wrote wf_export.json',os.path.getsize(os.path.join(OUT,'wf_export.json'))//1024,'KB')
c=out['climate']; print({k:c[k] for k in ('admitted','extreme','moderate')})
