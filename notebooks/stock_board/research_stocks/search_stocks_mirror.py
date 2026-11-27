import sys, time, itertools; sys.path.insert(0,'.')
import pandas as pd, numpy as np
import sb_core as C, sb_match_search as S
P=C.load_panel()
SP=pd.read_parquet('/private/tmp/stocks/stock_prices.parquet')
con=pd.read_csv('/private/tmp/stocks/constituents.csv'); con['Symbol']=con.Symbol.str.replace('.','-',regex=False)
sector=con.set_index('Symbol')['GICS Sector']; added=pd.to_datetime(con.set_index('Symbol')['Date added'],errors='coerce')
spy=P['SPY'].dropna().loc['2002-01-01':]
vix=P['^VIX'].dropna().reindex(spy.index).ffill(); vix_med=vix.rolling(252).median(); spy_ma=spy.rolling(200).mean()
SPc=SP.reindex(spy.index).ffill(limit=5)
# keep names with >= 3 years of data
SPc=SPc.loc[:,SPc.notna().sum()>=750]
# survivor-bias mitigation: a stock only plays from the day it joined the S&P 500 (names that joined before 2002 play from the start)
for t in SPc.columns:
    a=added.get(t)
    if pd.notna(a) and a>pd.Timestamp('2002-01-01'): SPc.loc[SPc.index<a,t]=np.nan
cls=sector.reindex(SPc.columns).dropna(); SPc=SPc[cls.index]
print('stocks',SPc.shape[1],'sectors',cls.nunique(),flush=True)
def blocks(H):
    ends=spy.index[H-1::H]
    cb=SPc.loc[ends]; ret=cb.pct_change(fill_method=None)
    reg=pd.DataFrame(index=ends)
    reg['trend']=(spy.loc[ends]>spy_ma.loc[ends]).where(spy_ma.loc[ends].notna())
    reg['vol']=(vix.loc[ends]>vix_med.loc[ends]).where(vix_med.loc[ends].notna())
    sb=spy.loc[ends].pct_change(); reg['prev']=(sb>0).where(sb.notna()); reg['none']=True
    return ends,ret,reg,sb
_orig=S.opponent_diff; _cache={}
def cached(ret,mode,cls_=None,spy_ret=None):
    k=(id(ret),mode)
    if k not in _cache: _cache[k]=_orig(ret,mode,cls_,spy_ret)
    return _cache[k]
S.opponent_diff=cached
rows=[]; t0=time.time()
for H in (5,10,21,63):
    ends,ret,reg,sb=blocks(H)
    for years,opp,band,venue in itertools.product((0.5,1,2),('class','spy','zero'),(0.5,1.0),('trend','vol','prev','none')):
        T,bd=S.build_cell(ret,reg,H,years,opp,band,venue,cls=cls,spy_ret=sb)
        for q in (0.8,0.9,0.95):
            m=S.picks(T,q,2008,2026,min_rows=5000,side='lo')
            kind='abs' if opp=='zero' else 'rel'
            d=S.stats_for(T,m,2008,2014,kind,bd,H); h_=S.stats_for(T,m,2015,2026,kind,bd,H)
            da=S.stats_for(T,m,2008,2014,'abs',bd,H); ha=S.stats_for(T,m,2015,2026,'abs',bd,H)
            rows.append(dict(H=H,yrs=years,opp=opp,band=band,venue=venue,q=q,kind=kind,
                d_n=d['n'],d_hit=d['hit'],d_base=d['base'],d_lift=d['lift'],d_exc=d['exc'],d_t=d['t'],
                h_n=h_['n'],h_hit=h_['hit'],h_base=h_['base'],h_lift=h_['lift'],h_exc=h_['exc'],h_t=h_['t'],
                da_t=da['t'],ha_n=ha['n'],ha_lift=ha['lift'],ha_exc=ha['exc'],ha_t=ha['t']))
    print('H',H,'done',round(time.time()-t0),'s',len(rows),flush=True)
    pd.DataFrame(rows).to_csv('/private/tmp/stocks/search_stocks_mirror.csv',index=False)
print('finished')
