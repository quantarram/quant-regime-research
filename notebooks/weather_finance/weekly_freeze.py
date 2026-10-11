"""Weekly freeze of NEW DISTINCT watch-tier conditions into the Climate Signals dashboard (run from notebooks/weather_finance/ with the research venv).

Fixed rule (written 2026-10-12, applies unchanged every week):
  1. Refresh inputs: the latest committed index files of the dashboard repo, the research price panel (git pull the finance repo first), the four daily climate panels.
  2. Re-evaluate the four batches' formation rows (formation through 2012, thresholds frozen) with test data through the freeze date, so the test episode counts and hit rates are those known on that date.
  3. Candidates = watch-tier rows (>= 3 formation episodes, >= 2 test episodes, test hit >= 2/3, test CPE >= test base + 0.10, >= 100 test days) not already in any frozen table.
  4. Exclude: the ENSO family (frozen in 224 overlapping rows); series without a live feed or with look-ahead or external-weight problems (ACE, CBTMP, HDDGAS, CDDPOW, ICE, IOD).
  5. Distinct: skip any row whose (predictor family, target, direction) already appears in a frozen table; among the rest keep one per (family, target, direction), the highest formation conviction.
  6. If no row remains, freeze nothing. Otherwise write the next table (b6, b7, ...) with its price store, register it in data/cpe_climate/books.json (condition ledger only, not part of the registered decision rule), run the ledger and check parity and firing.
Usage:
  python weekly_freeze.py                       # full weekly run
  python weekly_freeze.py --skip-eval           # reuse the existing eval files (debugging)
  python weekly_freeze.py --as-if-unfrozen b5 --no-cross-distinct --skip-eval   # reproducibility test: pretend table b5 does not exist, rebuild it, compare, write nothing
"""
import argparse, glob, hashlib, json, os, shutil, subprocess, sys, time
from datetime import date
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
DASH_DEFAULT='/Users/arrams/Documents/personal_work/quant/wid-exposure'
DASH_PY='/Users/arrams/Documents/personal_work/quant/climate-energy-dashboard/.venv/bin/python'
KEY=['X','Y','tau_past','tau_future','q_X','q_Y','direction']
ENSO={'ONI','NINO34','NINO12','NINO3','NINO4','MEI2','SOI'}; EXCL={'ACE','CBTMP','HDDGAS','CDDPOW','ICE','IOD'}
EVALS=[('b1','formation','cpe_climate_eval.py'),('b2','extreme_formation','cpe_extreme_eval.py'),('b3','slow_formation','cpe_slow_eval.py'),('b4','x4_formation','cpe_x4_eval.py')]
INDEX_MAP={'climate':['oni.txt','sstoi.txt','ao.txt','nao.txt','pdo.txt','soi.txt'],'climate3':['meiv2.txt','aao.txt','tna.txt','tsa.txt','whwp.txt','pna.txt','sstoi.txt'],
           'climate4':['tni.txt','npgo.txt','gistemp.csv','snow.txt'],'extremes':['climdiv-pdsist.txt']}
def run(cmd,env=None,cwd=HERE):
    r=subprocess.run(cmd,cwd=cwd,env={**os.environ,**(env or {})},capture_output=True,text=True)
    if r.returncode: print(r.stdout[-1500:],r.stderr[-1500:]); raise SystemExit(f'failed: {" ".join(cmd)}')
    return r.stdout
def fam(x): return x.split('_')[0]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dash',default=DASH_DEFAULT); ap.add_argument('--date',default=str(date.today())); ap.add_argument('--skip-eval',action='store_true'); ap.add_argument('--as-if-unfrozen'); ap.add_argument('--no-cross-distinct',action='store_true',help='test only: skip rule 5 across tables (table b5 was built before that rule existed)')
    a=ap.parse_args(); DASH=a.dash; CD=os.path.join(DASH,'data','cpe_climate'); t0=time.time()
    # ---- frozen tables of the dashboard
    tabs={os.path.basename(p):pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(CD,'cpe_climate_table*.parquet')))}
    if a.as_if_unfrozen: tabs={k:v for k,v in tabs.items() if not k.endswith(f'_{a.as_if_unfrozen}.parquet')}
    frozen=pd.concat(tabs.values()); fkeys=set(map(tuple,frozen[KEY].values.tolist())); ffd=set(zip(frozen.X.map(fam),frozen.Y,frozen.direction))
    print(f'frozen tables: {list(tabs)} | {len(frozen)} rows | {len(ffd)} (family,target,direction) combinations')
    if not a.skip_eval:
        for d,fs in INDEX_MAP.items():
            for f in fs:
                src=os.path.join(CD,'indices',f)
                if os.path.exists(src): shutil.copy2(src,os.path.join(HERE,'data',d,f))
        for scr in ('cpe_climate_data.py','cpe_slow_data.py','cpe_x4_data.py','cpe_extreme_data.py'): run([sys.executable,scr]); print('rebuilt panel',scr,f'{time.time()-t0:.0f}s',flush=True)
        for b,mode,scr in EVALS:
            run([sys.executable,scr],env={'EVAL_MODE':mode}); print('evaluated',b,f'{time.time()-t0:.0f}s',flush=True)
    E=pd.concat([pd.read_parquet(os.path.join(HERE,'cpe_climate',f'{m}_eval.parquet')).assign(batch=b) for b,m,_ in EVALS],ignore_index=True).drop_duplicates(KEY)
    W=E[(E.n_ep_f>=3)&(E.n_ep_t>=2)&(E.hit_t>=2/3)&(E.cpe_test>=E.base_test+0.10)&(E.n_test>=100)].copy()
    W['fam']=W.X.map(fam); print('watch-tier rows with data through the latest bar:',len(W))
    W=W[~W.set_index(KEY).index.isin(list(fkeys))]; print('not already frozen:',len(W))
    W=W[~W.fam.isin(ENSO|EXCL)]; print('after exclusions:',len(W))
    if not a.no_cross_distinct: W=W[[ (f,y,d) not in ffd for f,y,d in zip(W.fam,W.Y,W.direction)]]
    print('distinct from every frozen (family,target,direction):',len(W))
    W=W.sort_values('conv_f',ascending=False).drop_duplicates(['fam','Y','direction'])
    if a.as_if_unfrozen:
        ref=tabs and pd.read_parquet(os.path.join(CD,f'cpe_climate_table_{a.as_if_unfrozen}.parquet')); same=set(map(tuple,ref[KEY].values.tolist()))==set(map(tuple,W[KEY].values.tolist()))
        print(f'REPRODUCIBILITY: rebuilt {len(W)} rows vs frozen {len(ref)}; identical keys: {same}'); 
        if not same: print('only in rebuilt:',len(set(map(tuple,W[KEY].values.tolist()))-set(map(tuple,ref[KEY].values.tolist()))),'| only in frozen:',len(set(map(tuple,ref[KEY].values.tolist()))-set(map(tuple,W[KEY].values.tolist()))))
        return
    if W.empty: print('NO NEW DISTINCT CONDITIONS: nothing frozen this week.'); return
    # ---- build the table with formation thresholds
    sys.argv=['x','formation']; sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(HERE,'cpe_climate'))
    import cpe_climate_run as R
    px=pd.read_parquet(os.path.join(R.NB,'multiasset_prices_live_history.parquet')); ex=pd.read_parquet(os.path.join(R.HERE,'data','universe_all.parquet'))
    for t in R.EXTRA:
        if t in ex.columns and t not in px.columns: px[t]=ex[t].reindex(px.index)
    parts=[pd.read_parquet(os.path.join(R.HERE,'data',f)).reindex(px.index) for f in ('climate/climate_daily.parquet','extremes/extremes_daily.parquet','slow/slow_daily.parquet','x4/x4_daily.parquet')]
    C=pd.concat(parts,axis=1); C=C.loc[:,~C.columns.duplicated()]; inc_f,fut_f,thr_f,_,_=R.build(px,C,R.CUTOFF)
    keep=['Y','X','tau_past','tau_future','q_X','q_Y','direction','CPE','n_condition','n_ep_f','hit_f','conv_f','cpe_test','base_test','n_test','n_ep_t','hit_t','tier','ysub']
    T=W[keep].copy(); T['uncond_prob']=1.0-T.q_Y; T['lift']=T.CPE/T.uncond_prob
    xthr=lambda r: thr_f[(int(r.tau_past),float(r.q_X))][r.X] if r.direction=='bullish' else thr_f[(int(r.tau_past),round(1-float(r.q_X),10))][r.X]
    ythr=lambda r: thr_f[(int(r.tau_future),float(r.q_Y))][r.Y] if r.direction=='bullish' else thr_f[(int(r.tau_future),round(1-float(r.q_Y),10))][r.Y]
    T['x_threshold']=T.apply(xthr,axis=1); T['y_threshold']=T.apply(ythr,axis=1); T['threshold_frozen_through']='2012-12-31'; T['table_frozen']=a.date
    T=T.sort_values(['Y','X','tau_past','tau_future','q_X','q_Y']).reset_index(drop=True)
    man_path=os.path.join(CD,'books.json'); man=json.load(open(man_path)) if os.path.exists(man_path) else []
    ids=[int(os.path.basename(p).split('_b')[-1].split('.')[0]) for p in glob.glob(os.path.join(CD,'cpe_climate_table_b*.parquet'))]; n=max(ids+[5])+1; bid=f'b{n}'
    tp=os.path.join(CD,f'cpe_climate_table_{bid}.parquet'); T.to_parquet(tp,index=False); sha=hashlib.sha1(open(tp,'rb').read()).hexdigest()[:10]
    P=pd.DataFrame(index=px.index)
    for y in sorted(T.Y.unique()): P[y]=px[y]
    P.to_parquet(os.path.join(CD,f'prices_{bid}.parquet'))
    man.append({'id':bid,'label':f'Batch {n}: weekly watch tier, new distinct conditions ({a.date})','table':f'cpe_climate_table_{bid}.parquet','events':f'cpe_climate_events_{bid}.csv','prices':f'prices_{bid}.parquet','frozen':a.date,'decisions':False,
        'note':'Weekly watch-tier freeze: only conditions distinct from every earlier table (ENSO family excluded). Weaker evidence; a condition ledger only, not part of the registered decision rule.'})
    json.dump(man,open(man_path,'w'),indent=1)
    print(f'FROZE {bid}: {len(T)} rows, {T.groupby(["X","tau_past","q_X","direction"]).ngroups} conditions, families {sorted(set(T.X.map(fam)))}, sha {sha}')
    # ---- ledger: parity of every series, then run (logs the ongoing events)
    sys.path.insert(0,DASH); from weatherdash import cpe_climate as c
    bad=[]
    for col in sorted(T.X.unique()):
        x=c.climate_series(col,P.index); y=C[col]; m=x.notna()&y.notna()
        if not ((x.isna()==y.isna()).all() and np.allclose(x[m].values,y[m].values)): bad.append(col)
    print('series parity with the engine panel:','identical' if not bad else f'MISMATCH {bad}')
    if bad: raise SystemExit('parity failure: table written but not valid; fix before committing')
    out=run([DASH_PY,'-m','weatherdash.cpe_climate'],cwd=DASH); print([l for l in out.split('\n') if bid.upper() in l or 'FIRING' in l][:12])
    print(f'done in {time.time()-t0:.0f}s. Next: pull/push the dashboard repo (git add -A; commit; push; workflow dispatch).')
if __name__=='__main__': main()
