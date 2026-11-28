"""Rare football-style picks: at most CAP picks per match. Definition chosen on 2008-14 only (hit rate above the stock's own rate)."""
import sys, itertools, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import pandas as pd, numpy as np
import sb_core as C, sb_match_search as S
src=open('research_stocks/search_stocks.py').read()
exec(src.split("print('stocks'")[0].replace('import sys, time, itertools; sys.path.insert(0,\'.\')',''))
exec('def blocks'+src.split('def blocks')[1].split('_orig=')[0])
_orig=S.opponent_diff; _cache={}
def cached(ret,mode,cls_=None,spy_ret=None):
    k=(id(ret),mode)
    if k not in _cache: _cache[k]=_orig(ret,mode,cls_,spy_ret)
    return _cache[k]
S.opponent_diff=cached
CAP=2
def capped(T,m):
    s=T[m].sort_values(['pos','form_edge','home_ppg','ticker'],ascending=[True,False,False,True])
    keep=s.groupby('pos').head(CAP).index
    out=np.zeros(len(T),bool); out[keep.values]=True; return out
rows=[]; keepB={}
for H in (5,10,21,63):
    B_=blocks(H); keepB[H]=B_; ends,ret,reg,sb=B_
    for years,opp,band,venue in itertools.product((0.5,1,2),('class','spy','zero'),(0.5,1.0),('trend','vol','prev','none')):
        T,bd=S.build_cell(ret,reg,H,years,opp,band,venue,cls=cls,spy_ret=sb)
        yr=T.date.dt.year
        for q in (0.8,0.9,0.95):
            m=capped(T,S.picks(T,q,2008,2026,min_rows=5000))
            kind='abs' if opp=='zero' else 'rel'
            d=S.stats_for(T,m,2008,2014,kind,bd,H); h=S.stats_for(T,m,2015,2026,kind,bd,H)
            rows.append(dict(H=H,yrs=years,opp=opp,band=band,venue=venue,q=q,
              d_n=d['n'],d_years=yr[m&(yr<=2014)&(yr>=2008)].nunique(),d_hit=d['hit'],d_base=d['base'],d_lift=d['lift'],
              h_n=h['n'],h_hit=h['hit'],h_base=h['base'],h_lift=h['lift']))
    print('H',H,'done',len(rows),flush=True)
R=pd.DataFrame(rows); R.to_csv('research_stocks/rare_search.csv',index=False)
# choose on discovery only: must fire in every discovery year, highest hit rate above the stocks' own
ok=R[R.d_years==7].sort_values(['d_lift','d_n'],ascending=[False,False])
best=ok.iloc[0]; print(best.to_string()); print('usable definitions',len(ok))
H=int(best.H); ends,ret,reg,sb=keepB[H]
_cache.clear()
T,bd=S.build_cell(ret,reg,H,best.yrs,best.opp,best.band,best.venue,cls=cls,spy_ret=sb)
m=capped(T,S.picks(T,best.q,2008,2026,min_rows=5000))
# money: buy at the close one day after the signal, sell one day after the match ends, 5 bps round trip
ip=spy.index.get_indexer(ends)
def px(off):
    j=ip+off; v=np.full((len(ends),SPc.shape[1]),np.nan); good=j<len(spy.index); v[good]=SPc.values[j[good]]; return pd.DataFrame(v,index=ends,columns=SPc.columns)
e=px(1); x=e.shift(-1); money=(x/e-1)
avg=money.mean(axis=1)
col={t:i for i,t in enumerate(ret.columns)}
P_=T[m].copy()
P_['stock_ret']=[money.at[d,t] for d,t in zip(P_.date,P_.ticker)]
P_['avg_stock_ret']=P_.date.map(avg)
P_['sector']=P_.ticker.map(cls)
if best.opp=='zero':
    P_['margin']=P_.ret; P_['result']=np.where(P_.ret>0,'W','L')
else:
    P_['margin']=P_.rel; P_['result']=np.where(P_.rel>bd,'W',np.where(P_.rel>=-bd,'D','L'))
P_['end']=P_.pos.map(lambda i: ends[i+1])
P_=P_[['date','end','ticker','sector','form_edge','home_ppg','margin','stock_ret','avg_stock_ret','result']]
P_.to_csv('research_stocks/rare_picks.csv',index=False)
for name,a,b in (('chosen on 2008-14',2008,2014),('after, 2015 on',2015,2026)):
    s=P_[(P_.date.dt.year>=a)&(P_.date.dt.year<=b)]; mm=s.dropna(subset=['stock_ret'])
    print(name,len(s),s.result.value_counts().to_dict(),'| per $1000 pick: picks net',round((1000*(mm.stock_ret-0.0005)).sum()),'same money in avg stock',round((1000*(mm.avg_stock_ret-0.0005)).sum()),'on',len(mm))
    mb=T[(T.date.dt.year>=a)&(T.date.dt.year<=b)]
s=P_[P_.date.dt.year>=2015]
print(s.groupby([s.date.dt.year,'result']).size().unstack(fill_value=0).to_string())
print(s.tail(16).round(3).to_string(index=False))
print('picks per match max',P_.groupby('date').size().max())
