"""Climate predictors through the CPE engine (cpe_engine_parallel.py), unchanged logic: pairwise CPE with the engine's gates, then the episode-conviction audit of backtest_engine.py.

  python cpe_climate_run.py formation   # engine on data through CUTOFF (2012-12-31); writes cpe_climate/formation_rows.parquet
  python cpe_climate_run.py full        # engine on all data (how the program's own table is built); writes cpe_climate/full_rows.parquet
"""
import glob, os, sys, time
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); NB=os.path.abspath(os.path.join(HERE,'..'))
sys.path.insert(0,NB); sys.path.insert(0,os.path.join(NB,'files')); sys.path.insert(0,HERE)
OUTD0=os.path.join(HERE,'cpe_climate'); sys.path.insert(0,OUTD0)   # shadow economic_prior with the climate channels registered
OUTD=os.path.join(HERE,'cpe_climate'); os.makedirs(OUTD,exist_ok=True); os.chdir(OUTD)
import cpe_engine_parallel as eng
import cpe_climate_data as cd
from multiprocessing import Pool
CUTOFF=pd.Timestamp('2012-12-31')
TAU_PAST=[21,63,126,252,300]; TAU_FUTURE=[21,63,126,252,300]
EXTRA=['KC=F','CC=F','CT=F','SB=F','OJ=F','LE=F','HE=F','ZL=F','ZM=F','KE=F','HO=F','RB=F','ZO=F']
def load_panel():
    px=pd.read_parquet(os.path.join(NB,'multiasset_prices_live_history.parquet'))
    ex=pd.read_parquet(os.path.join(HERE,'data','universe_all.parquet'))
    for t in EXTRA:
        if t in ex.columns and t not in px.columns: px[t]=ex[t].reindex(px.index)
    C=pd.read_parquet(os.path.join(HERE,'data','climate','climate_daily.parquet')).reindex(px.index)
    return px,C
def build(px,C,cutoff=None):
    if cutoff is not None: px,C=px[px.index<=cutoff],C[C.index<=cutoff]
    P=pd.concat([px,C],axis=1)
    climate=list(C.columns)
    price_tickers=[t for t in px.columns if t not in eng.RATE_INDEX_TICKERS]
    rate_tickers=[t for t in px.columns if t in eng.RATE_INDEX_TICKERS]+climate
    predicted=[t for t in price_tickers if t not in eng.EXCLUDE_FROM_Y]
    taus=sorted(set(TAU_PAST+TAU_FUTURE)); inc={}
    for tau in taus:
        d={}
        for t in price_tickers: d[t]=np.log(P[t]/P[t].shift(tau))
        for t in rate_tickers: d[t]=P[t]-P[t].shift(tau)
        inc[tau]=pd.DataFrame(d)
    fut={tf:inc[tf][predicted].shift(-tf) for tf in TAU_FUTURE}
    qs=sorted(set(eng.Q_GRID+[round(1-q,10) for q in eng.Q_GRID]))
    thr={(tau,q):inc[tau].quantile(q,numeric_only=True).to_dict() for tau in taus for q in qs}
    return inc,fut,thr,predicted,climate
def run_engine(inc,fut,thr,predicted,climate,nw=8):
    for f in glob.glob('cpe_tmp_chunks/*.parquet'): os.remove(f)
    cfg=dict(cpe_thresh=eng.CPE_THRESH,min_n=eng.MIN_SAMPLE,min_lift=eng.MIN_LIFT,tau_past=TAU_PAST,tau_future=TAU_FUTURE,q_grid=eng.Q_GRID,min_confidence=os.environ.get('MIN_CONFIDENCE','weak'))
    tasks=[(y,i) for i,y in enumerate(predicted)]; t0=time.time(); kept=0
    with Pool(nw,initializer=eng._init_worker,initargs=(inc,fut,thr,climate,cfg)) as pool:
        for i,n in enumerate(pool.imap_unordered(eng._compute_cpe_for_y,tasks)):
            kept+=n; print(f'  [{i+1}/{len(tasks)}] kept {kept:,} {time.time()-t0:.0f}s',end='\r',flush=True)
    files=sorted(glob.glob('cpe_tmp_chunks/*.parquet'))
    df=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True) if files else pd.DataFrame()
    for f in files: os.remove(f)
    return df
if __name__=='__main__':
    mode=sys.argv[1]
    px,C=load_panel()
    cutoff=CUTOFF if mode in ('formation','standard_formation') else None
    inc,fut,thr,predicted,climate=build(px,C,cutoff)
    if mode=='standard_formation':   # the program's own predictors under the identical split, prior and gates (calibration baseline)
        s_=px[px.index<=pd.Timestamp('2024-12-31')]
        climate=[t for t in px.columns if t not in eng.EXCLUDE_FROM_X and s_[t].notna().sum()>=500]
    print(mode,'Y',len(predicted),'X',len(climate),'dates',inc[21].index.min().date(),inc[21].index.max().date())
    df=run_engine(inc,fut,thr,predicted,climate)
    print('\nrows kept',len(df)); df.to_parquet(f'{mode}_rows.parquet')
    if len(df): print(df.groupby(['X']).size().sort_values(ascending=False).head(12)); print(df.groupby('tau_future').size())
