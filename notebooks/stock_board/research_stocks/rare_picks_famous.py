import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import pandas as pd, numpy as np
import sb_core as C, sb_match_search as S
src=open('research_stocks/search_stocks.py').read()
exec(src.split("print('stocks'")[0].replace("import sys, time, itertools; sys.path.insert(0,'.')",''))
exec('def blocks'+src.split('def blocks')[1].split('_orig=')[0])
H=21; ends,ret,reg,sb=blocks(H); diff=S.opponent_diff(ret,'class',cls,sb); bd=0.005*np.sqrt(H/21)
P=pd.read_csv('research_stocks/rare_picks.csv',parse_dates=['date','end'])
print('--- last picks'); print(P.sort_values('date').tail(14)[['date','end','ticker','sector','form_edge','home_ppg','margin','stock_ret','result']].round(3).to_string(index=False))
d=pd.Timestamp('2026-10-15'); b=list(ends).index(d); V=reg['prev'].shift(1)
for t in P[P.date==d].ticker:
    res=lambda k:'W' if diff[t].iloc[k]>bd else ('D' if diff[t].iloc[k]>=-bd else 'L')
    print('===',t,'upcoming venue SPY prev up =',reg['prev'].iloc[b])
    ks=list(range(b-5,b+1)); hk=[k for k in range(b+1) if V.iloc[k]==reg['prev'].iloc[b]][-6:]
    for name,K in (('last6',ks),('home6',hk)):
        for k in K: print(name,ends[k-1].date(),'->',ends[k].date(),'stock',round(100*ret[t].iloc[k],1),'peers',round(100*(ret[t].iloc[k]-diff[t].iloc[k]),1),res(k))
    peers=cls[cls==cls[t]].index.drop(t)
    pts=pd.DataFrame(np.where(diff.isna(),np.nan,np.where(diff>bd,3.,np.where(diff>=-bd,1.,0.))),index=ret.index,columns=ret.columns)
    print('peer PPG',round(pts[peers].iloc[b-5:b+1].mean(skipna=False).mean(),2),'n peers',len(peers))
nx=diff.shift(-1).iloc[:-1]; nx=nx[nx.index.year>=2015]
base=(nx>bd).where(nx.notna()).astype(float).mean()
q=P[P.date.dt.year>=2015]
for name,z in (('all',q),('five',q[q.ticker.isin(['NVDA','WDC','DVN','SKYD','LRCX'])]),('rest',q[~q.ticker.isin(['NVDA','WDC','DVN','SKYD','LRCX'])])):
    print('##',name,len(z),z.result.value_counts().to_dict(),'win',round(100*(z.result=='W').mean(),1),'usual',round(100*z.ticker.map(base).mean(),1))
A=P.sort_values(['date','ticker']).reset_index(drop=True)
# record of the same stock's picks that had FINISHED by the signal day
prior=[]
for r in A.itertuples():
    h=A[(A.ticker==r.ticker)&(A.end<=r.date)]
    prior.append((len(h),(h.result=='W').sum(),(h.result=='L').sum()))
A['pn'],A['pw'],A['pl']=zip(*prior)
early=A[A.end<=pd.Timestamp('2014-12-31')].groupby('ticker').result.agg(n='size',w=lambda s:(s=='W').sum(),l=lambda s:(s=='L').sum())
club=early[(early.n>=2)&(early.w>early.l)].index
q=A[(A.date.dt.year>=2015)&A.stock_ret.notna()]
def show(name,z):
    print(f"@@ {name} | picks {len(z)} | {(z.result=='W').sum()}-{(z.result=='D').sum()}-{(z.result=='L').sum()} | win {100*(z.result=='W').mean():.1f} usual {100*z.ticker.map(base).mean():.1f} | picks ${1000*(z.stock_ret-.0005).sum():,.0f} avg ${1000*(z.avg_stock_ret-.0005).sum():,.0f} | stocks {z.ticker.nunique()}")
show('all',q)
show('first time picked',q[q.pn==0])
for k in (1,2,3,5):
    show(f'winning record, {k}+ earlier picks',q[(q.pn>=k)&(q.pw>q.pl)])
show('had earlier picks, record not winning',q[(q.pn>=1)&(q.pw<=q.pl)])
show('club fixed on 2008-14 record',q[q.ticker.isin(club)]); print('@@ club size',len(club),sorted(club))
k=q[(q.pn>=2)&(q.pw>q.pl)]; print('@@ famous(2+) tickers', k.ticker.value_counts().head(15).to_dict())
for t in ('NVDA','WDC','DVN','SKYD','LRCX'):
    z=A[A.ticker==t]; print('@@',t,[(str(r.date.date())[:7],r.result,f'{r.pw}-{r.pl}') for r in z.itertuples()])
five=['NVDA','WDC','DVN','SKYD','LRCX']
for k in (1,2,3):
    z=q[(q.pn>=k)&(q.pw>q.pl)]
    show(f'## {k}+ without NVDA',z[z.ticker!='NVDA']); show(f'## {k}+ without the five',z[~z.ticker.isin(five)])
z=q[(q.pn>=2)&(q.pw>q.pl)]
g=z.groupby(z.date.dt.year); 
for y,v in g: show(f'## year {y}',v)
