"""The Sugar paper's design inside the CPE engine's definitions: El Nino-state predictors, firing restricted to the monsoon season (June-September), target = global sugar price (FRED benchmark,
monthly, known at month end; SB=F where available is shown separately). Pre-specified direction: bullish (X in its upper tail -> sugar's forward move in its upper tail).
Same gates (CPE >= 0.80, lift >= 1.5 against 1-q_Y, n >= 100), episodes (>= 3 floor), formation through 2012-12-31 and test afterwards with frozen thresholds."""
import os, sys
import numpy as np, pandas as pd
sys.argv=['x','formation']
import cpe_climate_run as R
import backtest_engine as be
HERE=R.HERE
CUT=R.CUTOFF
px,C=R.load_panel()
f=pd.read_csv(os.path.join(HERE,'policy','fred_PSUGAISAUSDM.csv'),parse_dates=['observation_date']).set_index('observation_date').iloc[:,0].dropna()
# monthly average for month m becomes known on the last trading day of month m
td=px.index; known={}
for d,v in f.items():
    me=d+pd.offsets.MonthEnd(0); i=td.searchsorted(me,side='right')-1
    if 0<=i<len(td): known[td[i]]=v
px['SUGAR_FRED']=pd.Series(known).reindex(td).ffill()
SEAS=[6,7,8,9]; XS=['ONI_chg','ONI_mean','NINO34_chg','NINO34_mean','SOI_chg','SOI_mean']
TP=[21,63,126]; TF=[21,63,126]; QG=eng_q=R.eng.Q_GRID
def run(cutoff,label):
    inc,fut,thr,pred,climate=R.build(px,C,cutoff)
    out=[]
    for tp in TP:
        for x in XS:
            s=inc[tp][x]
            for q in QG:
                for d in ('bullish','bearish'):
                    tx=thr[(tp,q)][x] if d=='bullish' else thr[(tp,round(1-q,10))][x]
                    fire=(s>tx) if d=='bullish' else (s<tx)
                    fire=fire&s.index.month.isin(SEAS)
                    fd=fire[fire.fillna(False)].index
                    if len(fd)<R.eng.MIN_SAMPLE: continue
                    eps=be._cluster_into_episodes(fd,tp)
                    for tf in TF:
                        for qy in QG:
                            ty=thr[(tf,qy)]['SUGAR_FRED'] if d=='bullish' else thr[(tf,round(1-qy,10))]['SUGAR_FRED']
                            fy=fut[tf]['SUGAR_FRED']; v=fy.loc[fd].dropna()
                            if len(v)<R.eng.MIN_SAMPLE: continue
                            ev=(v>ty) if d=='bullish' else (v<ty); cpe=float(ev.mean()); unc=1-qy
                            allv=fy.dropna(); base_all=float(((allv>ty) if d=='bullish' else (allv<ty)).mean())
                            seas=allv[allv.index.month.isin(SEAS)]; base_seas=float(((seas>ty) if d=='bullish' else (seas<ty)).mean())
                            o=[bool((fy.get(e[-1],np.nan)>ty) if d=='bullish' else (fy.get(e[-1],np.nan)<ty)) for e in eps if not pd.isna(fy.get(e[-1],np.nan))]
                            out.append(dict(X=x,tau_past=tp,tau_future=tf,q_X=q,q_Y=qy,direction=d,CPE=cpe,lift=cpe/unc,lift_season=cpe/base_seas if base_seas>0 else np.nan,n_cond=len(v),n_ep=len(o),hit=float(np.mean(o)) if o else np.nan,eps=[str(e[-1].year) for e in eps],x_thr=float(tx),y_thr=float(ty)))
    return pd.DataFrame(out)
F=run(CUT,'formation')
G=F[(F.CPE>=0.80)&(F.lift>=1.5)]
print('formation rows tested',len(F),'| pass gates',len(G),'| bullish',int((G.direction=='bullish').sum()),'bearish',int((G.direction=='bearish').sum()))
G3=G[G.n_ep>=3]; print('>=3 formation episodes:',len(G3), 'bullish',int((G3.direction=='bullish').sum()))
print(G3.groupby(['X','direction']).size().to_dict())
G.to_parquet(os.path.join(R.OUTD,'sugar_season_formation.parquet'))
F.to_parquet(os.path.join(R.OUTD,'sugar_season_formation_all.parquet'))
inc_a,fut_a,thr_a,_,_=R.build(px,C,None)
def oos(G3,label):
    res=[]
    for r in G3.itertuples():
        s=inc_a[r.tau_past][r.X]; fire=((s>r.x_thr) if r.direction=='bullish' else (s<r.x_thr))&s.index.month.isin(SEAS)&(s.index>CUT)
        fd=fire[fire.fillna(False)].index; fy=fut_a[r.tau_future]['SUGAR_FRED']; v=fy.loc[fd].dropna()
        evf=lambda z:(z>r.y_thr) if r.direction=='bullish' else (z<r.y_thr)
        t=fy.dropna(); t=t[t.index>CUT]; ts=t[t.index.month.isin(SEAS)]
        eps=be._cluster_into_episodes(fd,r.tau_past) if len(fd) else []
        o=[bool(evf(fy.get(e[-1],np.nan))) for e in eps if not pd.isna(fy.get(e[-1],np.nan))]
        res.append((len(v),float(evf(v).mean()) if len(v) else np.nan,float(evf(ts).mean()),len(o),float(np.mean(o)) if o else np.nan))
    T=pd.concat([G3.reset_index(drop=True),pd.DataFrame(res,columns=['n_test','cpe_test','base_season','n_ep_t','hit_t'])],axis=1)
    b=T[(T.direction=='bullish')&(T.n_test>=30)]
    print(f'\n[{label}] bullish rows, >=3 formation episodes, >=30 test days: {len(b)} | claimed {b.CPE.mean():.3f} | test CPE {b.cpe_test.mean():.3f} | test season base {b.base_season.mean():.3f} | share above base {(b.cpe_test>b.base_season).mean():.2f} | rows with >=1 test episode {int((b.n_ep_t>=1).sum())}, mean episode hit {b.hit_t.mean():.2f}')
    return T
oos(F[(F.CPE>=0.80)&(F.lift>=1.5)&(F.n_ep>=3)],'strict gate 0.80/1.5')
for lo,li in ((0.70,1.4),(0.65,1.3)):
    G=F[(F.CPE>=lo)&(F.lift>=li)&(F.n_ep>=3)]
    print(f'relaxed gate CPE>={lo}, lift>={li}: rows {len(G)}, bullish {int((G.direction=="bullish").sum())}')
    T=oos(G,f'relaxed {lo}/{li}')
    if lo==0.65:
        b=T[(T.direction=='bullish')&(T.n_test>=30)]
        print(b.groupby(['X','tau_future']).agg(n=('CPE','size'),claimed=('CPE','mean'),test=('cpe_test','mean'),base=('base_season','mean')).round(3).to_string())
