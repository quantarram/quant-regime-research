import sys; sys.path.insert(0,'.')
exec(open('/private/tmp/stocks/search_stocks.py').read().split("_orig=S.opponent_diff")[0])
import pandas as pd, numpy as np
G=pd.read_csv('/private/tmp/stocks/search_stocks_mirror.csv'); ok=G[(G.d_n>=150)&(G.h_n>=150)&(G.d_t>=2)&(G.d_lift>0)]
tops=pd.concat([ok.sort_values('d_t',ascending=False).head(5).assign(sel='top5 discovery'),
                ok[ok.q==0.95].sort_values('d_t',ascending=False).head(5).assign(sel='top5 q=.95 (fewest picks)')])
out=[]
for _,c in tops.iterrows():
    H=int(c.H); ends,ret,reg,sb=blocks(H)
    T,bd=S.build_cell(ret,reg,H,c.yrs,c.opp,c.band,c.venue,cls=cls,spy_ret=sb)
    m=S.picks(T,c.q,2008,2026,min_rows=5000,side='lo')
    T['pick']=m
    for lab,(a,b) in {'discovery 08-14':(2008,2014),'holdout 15-26':(2015,2026)}.items():
        W=T[(T.date.dt.year>=a)&(T.date.dt.year<=b)]
        uni=W.groupby('date').ret.mean()
        pk=W[W.pick].groupby('date').ret.mean()
        n_pk=W[W.pick].groupby('date').size()
        df=pd.DataFrame({'pick':pk,'uni':uni}).dropna()
        per_year=252/H
        ex=(df.pick-df.uni)
        net=ex-0.002   # 20 bps round trip per match, every match
        t=ex.mean()/(ex.std(ddof=1)/np.sqrt(len(ex)))
        out.append(dict(sel=c.sel,H=H,yrs=c.yrs,opp=c.opp,venue=c.venue,q=c.q,period=lab,matchdays=len(df),picks_per_matchday=round(n_pk.mean(),1),
            pick_ann=round((1+df.pick.mean())**per_year-1,3),uni_ann=round((1+df.uni.mean())**per_year-1,3),excess_ann=round(ex.mean()*per_year,4),net_excess_ann=round(net.mean()*per_year,4),t=round(t,2),
            pick_vol=round(df.pick.std()*np.sqrt(per_year),3),uni_vol=round(df.uni.std()*np.sqrt(per_year),3)))
pd.set_option('display.width',260,'display.max_columns',30)
print(pd.DataFrame(out).to_string())
