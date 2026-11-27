"""
Monthly-games version of the football form edge + home PPG for ETFs (research record; no board uses it).

Redefinition of each football term from its football meaning:
  game      = one calendar month, non-overlapping, ETF vs its opponent = equal-weight mean of the other ETFs in its asset class
  result    = relative return that month: win (3) if > +band, draw (1) within +-band, loss (0) if < -band  (band 0.5%)
  PPG       = mean points of the last N=12 games;  home PPG = same over the last N games in months SPY rose
  form edge = PPG minus the mean PPG of the other ETFs in its class
  pick      = both at/above the 90th percentile of all earlier ETF-months (frozen each 1 January), then hold the ETF next month
Evaluated 2010+ against the same ETF's own always-hold record: absolute (return > 0) and relative (beats peers).
Result: 457 picks over 168 months, hit 58.0% vs 60.3% always-hold, excess return -0.13%/month; across 18 settings of (N, band,
percentile) no cell reaches t = 2 for either outcome. Run: python stock_board/sb_monthly_games.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pandas as pd, numpy as np
import sb_core as C, sb_research as R, sb_form as F
pd.set_option('display.width',250,'display.max_columns',30,'display.max_rows',200)
P=C.load_panel()
last_bar=P.index[-1]
mc=pd.DataFrame({t:P[t].dropna().resample('ME').last() for t in C.ALL_TICKERS if t in P.columns})
# drop the current, unfinished month
if last_bar < (last_bar + pd.offsets.MonthEnd(0)): mc=mc.iloc[:-1]
ret=mc.pct_change()
spy=ret['SPY']
cls=pd.Series({t:F.ASSET_CLASS[t] for t in mc.columns})
def build(N=12,band=0.005,q=0.9,first=2010):
    diff=pd.DataFrame(index=ret.index,columns=ret.columns,dtype=float)
    for c in cls.unique():
        cols=cls[cls==c].index
        for t in cols:
            others=ret[cols.drop(t)]
            peer=others.mean(axis=1).where(others.notna().sum(axis=1)>=3)
            diff[t]=ret[t]-peer
    pts=pd.DataFrame(np.where(diff.isna(),np.nan,np.where(diff>band,3.0,np.where(diff>=-band,1.0,0.0))),index=diff.index,columns=diff.columns)
    ppg=pts.rolling(N,min_periods=N).mean()
    homepts=pts.where(pd.DataFrame({c:(spy>0) for c in pts.columns}))
    hppg=homepts.apply(lambda s: s.dropna().rolling(N).mean().reindex(s.index).ffill())
    peer_ppg=pd.DataFrame({t:ppg[cls[cls==cls[t]].index.drop(t)].mean(axis=1) for t in ret.columns})
    fe=ppg-peer_ppg
    nxt=ret.shift(-1); nd=diff.shift(-1)
    L=[]
    for t in ret.columns:
        d=pd.DataFrame({'date':ret.index,'ticker':t,'form_edge':fe[t].values,'home_ppg':hppg[t].values,'ppg':ppg[t].values,
                        'ret':nxt[t].values,'rel':nd[t].values,'pos':np.arange(len(ret))})
        L.append(d)
    T=pd.concat(L,ignore_index=True)
    T=T[T.ret.notna()].reset_index(drop=True)
    T['group']=T.ticker.map(C.TICKER_GROUP)
    # absolute-bet columns
    T['pnl']=T.ret-0.0005; T['win']=(T.ret>0).astype(float)
    T['pnl_rel']=T.rel; T['win_rel']=(T.rel>band).astype(float)
    m,thr=F.pick_mask(T.assign(),first,2026,q) if False else (None,None)
    return T,thr
def pick(T,q,first=2010):
    m=np.zeros(len(T),bool); thr={}
    for Y in range(first,2027):
        past=T[T.date<pd.Timestamp(f'{Y}-01-01')]
        fe=past.form_edge.dropna(); hp=past.home_ppg.dropna()
        if len(fe)<1500 or len(hp)<1500: thr[Y]=None; continue
        a,b=fe.quantile(q),hp.quantile(q); thr[Y]=(round(a,2),round(b,2))
        s=(T.date.dt.year==Y).values
        m[s]=(T.form_edge.values[s]>=a)&(T.home_ppg.values[s]>=b)
    return m,thr
def evaluate(T,m,first=2010):
    W=T[(T.date>=f'{first}-01-01')]
    out={}
    Rr=T[m]
    for lab,(pn,wn) in {'absolute (return>0)':('pnl','win'),'relative (beats peers)':('pnl_rel','win_rel')}.items():
        W2=W.rename(columns={pn:'_p',wn:'_w'}).assign(pnl=lambda d:d._p,win=lambda d:d._w)
        R2=Rr.rename(columns={pn:'_p',wn:'_w'}).assign(pnl=lambda d:d._p,win=lambda d:d._w)
        st=R.rule_stats(R2,W2,1)
        out[lab]=st
    return out
T,_=build(12,0.005,0.9)
m,thr=pick(T,0.9)
print('thr',{k:v for k,v in list(thr.items())[::4]})
print('picks',m.sum(),'of',(T.date>='2010-01-01').sum(),'fire',round(m.sum()/(T.date>='2010-01-01').sum(),4))
for lab,st in evaluate(T,m).items():
    print(lab,{k:round(float(v),4) for k,v in st.items() if k in('n','n_blk','hit','base_hit','lift','pnl','base_pnl','exc_pnl','t')})
Tp=T[m]; print(Tp.group.value_counts().to_dict()); print(Tp.date.dt.year.value_counts().sort_index().to_dict())
print('--- grid: N games, draw band, quantile')
rows=[]
for N in (6,12,24):
    for band in (0.0025,0.005,0.01):
        Tg,_=build(N,band,0.9)
        for q in (0.8,0.9):
            mg,_=pick(Tg,q)
            ev=evaluate(Tg,mg)
            a=ev['absolute (return>0)']; r=ev['relative (beats peers)']
            rows.append(dict(N=N,band=band,q=q,picks=int(mg.sum()),abs_hit=a['hit'],abs_base=a['base_hit'],abs_lift=a['lift'],abs_exc=a['exc_pnl'],abs_t=a['t'],rel_hit=r['hit'],rel_base=r['base_hit'],rel_exc=r['exc_pnl'],rel_t=r['t']))
G=pd.DataFrame(rows).round(4); print(G.to_string())
print('cells with abs_t>=2:',(G.abs_t>=2).sum(),'rel_t>=2:',(G.rel_t>=2).sum(),'of',len(G))
