"""If we KNEW a month's irradiance on the 20th of the month before, would it make money in JEPX? (perfect-foresight upper bound; ERA5 stands in for the forecast; only irradiance and JEPX are used)

For every target month M from 2018-05 to 2026-09 (so that at least two earlier years of the same calendar month exist):
  reference R  = mean price over days 1-19 of month M-1 (what is known on the 20th; the 'forward' we would trade against)
  outcome  O   = mean price over month M;  change = O - R  (yen/kWh)
  expected     = mean of (O - R) for the same calendar month in EARLIER years only (the seasonal drift, no look-ahead);  surprise = change - expected
  signal       = irradiance anomaly of month M: mean daily sum of the month against the 1996-2015 mean of that calendar month, in sd of the 1996-2015 monthly means
Position: long 1 kWh if the signal is dull (<= the 10% cut of 1996-2015), short if bright (>= the 90% cut), flat otherwise; P&L = position * surprise.
Prices: all 48 slots ('day') and midday slots 21-30 ('midday'); areas Tokyo, Kyushu (Fukuoka irradiance), system (Tokyo irradiance).
"""
import glob, io, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,'data')
fr=[]
for f in sorted(glob.glob(os.path.join(D,'spot_*.csv'))):
    df=pd.read_csv(io.StringIO(open(f,encoding='utf-8').read())); df.columns=['date','slot','sell','buy','vol','system','hokkaido','tohoku','tokyo','chubu','hokuriku','kansai','chugoku','shikoku','kyushu','a','b','c','d']; fr.append(df[['date','slot','system','tokyo','kyushu']])
J=pd.concat(fr); J['date']=pd.to_datetime(J['date'],format='%Y/%m/%d'); J=J.drop_duplicates(['date','slot'])
idx=pd.date_range('2016-04-01','2026-09-30')
def daily(sl):
    g=J[(J.slot>=sl[0])&(J.slot<=sl[1])]; d=g.groupby('date').agg(system=('system','mean'),tokyo=('tokyo','mean'),kyushu=('kyushu','mean'),n=('slot','count')); return d[d.n==sl[1]-sl[0]+1].drop(columns='n').reindex(idx)
PRICE={'day':daily((1,48)),'midday':daily((21,30))}
def era(fn):
    d=json.load(open(os.path.join(D,fn)))['daily']; return pd.Series(d['shortwave_radiation_sum'],index=pd.to_datetime(d['time'])).astype(float)
SITE={'tokyo':era('era5_tokyo.json'),'kyushu':era('era5_fukuoka.json')}; SITE['system']=SITE['tokyo']
rows=[]
for pk,P in PRICE.items():
    for area in ['tokyo','kyushu','system']:
        S=SITE[area]; m=S.resample('MS').mean(); b=m[(m.index.year>=1996)&(m.index.year<=2015)]; c=b.groupby(b.index.month).agg(['mean','std'])
        z=(m-m.index.month.map(c['mean']).values)/m.index.month.map(c['std']).values
        zb=z[(z.index.year>=1996)&(z.index.year<=2015)]; lo,hi=zb.quantile(0.10),zb.quantile(0.90)
        p=P[area]
        for M in pd.date_range('2017-05-01','2026-09-01',freq='MS'):
            prev=M-pd.DateOffset(months=1); r=p[prev:prev+pd.Timedelta(days=18)]; o=p[M:M+pd.offsets.MonthEnd(0)]
            if r.notna().sum()<15 or o.notna().sum()<25: continue
            rows.append(dict(price=pk,area=area,month=M,R=r.mean(),O=o.mean(),chg=o.mean()-r.mean(),z=z[M],lo=lo,hi=hi))
R=pd.DataFrame(rows)
out=[]
for (pk,area),g in R.groupby(['price','area']):
    g=g.sort_values('month').copy(); exp=[]
    for _,r in g.iterrows():
        pr=g[(g.month.dt.month==r.month.month)&(g.month<r.month)]; exp.append(pr.chg.mean() if len(pr)>=2 else np.nan)
    g['exp']=exp; g['surprise']=g.chg-g.exp; g=g.dropna(subset=['surprise']); g['pos']=np.where(g.z<=g.lo,1,np.where(g.z>=g.hi,-1,0)); g['pnl']=g.pos*g.surprise; out.append(g)
A=pd.concat(out); A.to_csv(os.path.join(HERE,'oracle_money_months.csv'),index=False)
def summ(g):
    t=g[g.pos!=0]; n=len(t)
    return pd.Series(dict(months=len(g),corr_z_surprise=g.z.corr(g.surprise),slope_yen_per_sd=np.polyfit(g.z,g.surprise,1)[0],n_dull=int((g.pos==1).sum()),n_bright=int((g.pos==-1).sum()),
        surprise_dull=g.surprise[g.pos==1].mean(),surprise_bright=g.surprise[g.pos==-1].mean(),surprise_normal=g.surprise[g.pos==0].mean(),
        hit_rate=(t.pnl>0).mean() if n else np.nan,mean_pnl=t.pnl.mean() if n else np.nan,t_stat=(t.pnl.mean()/(t.pnl.std()/np.sqrt(n))) if n>2 else np.nan,sd_surprise=g.surprise.std()))
S=A.groupby(['price','area']).apply(summ).round(2); print(S.to_string())
print('\nWorst-case view: the extreme months (|z| beyond the 10%/90% cut), Tokyo and Kyushu midday:')
print(A[(A.price=='midday')&(A.area.isin(['tokyo','kyushu']))&(A.pos!=0)][['area','month','z','R','O','exp','surprise','pos','pnl']].round(2).assign(month=lambda d:d.month.dt.strftime('%Y-%m')).to_string(index=False))
print('\nSeptember 2026:'); print(A[(A.month=='2026-09-01')][['price','area','z','R','O','exp','surprise','pos','pnl']].round(2).to_string(index=False))
fig,ax=plt.subplots(1,3,figsize=(15,4.8))
for a,(pk,area) in zip(ax,[('midday','kyushu'),('midday','tokyo'),('day','tokyo')]):
    g=A[(A.price==pk)&(A.area==area)]; c=np.where(g.pos==1,'tab:red',np.where(g.pos==-1,'tab:blue','lightgray')); a.scatter(g.z,g.surprise,c=c,s=22); a.axhline(0,color='k',lw=.6); a.axvline(0,color='k',lw=.6)
    k=np.polyfit(g.z,g.surprise,1); xs=np.linspace(g.z.min(),g.z.max(),10); a.plot(xs,k[0]*xs+k[1],color='gray',lw=1)
    for _,r in g[g.pos!=0].iterrows():
        if abs(r.surprise)>np.nanpercentile(abs(g.surprise),90): a.annotate(r.month.strftime('%y-%m'),(r.z,r.surprise),fontsize=7)
    a.set_title(f'{area} {pk}: corr {g.z.corr(g.surprise):.2f}, slope {k[0]:.2f} yen/kWh per sd',fontsize=9); a.set_xlabel('irradiance anomaly of the month (sd; red = dull extreme, blue = bright extreme)'); a.set_ylabel('price change surprise vs the 1-19 of the prior month (yen/kWh)')
plt.tight_layout(); plt.savefig(os.path.join(HERE,'oracle_money_figure.png'),dpi=140); print('wrote oracle_money_months.csv, oracle_money_figure.png')
# robust view: drop 2021-02 (its reference days, Jan 2021, are the winter price crisis: 50-65 yen, a fuel/supply spike, not weather)
rows=[]
for (pk,ar),g in A.groupby(['price','area']):
    g2=g[g.month!='2021-02-01']; t=g2[g2.pos!=0]
    rows.append(dict(price=pk,area=ar,spearman=round(g2.z.corr(g2.surprise,method='spearman'),2),pearson=round(g2.z.corr(g2.surprise),2),slope=round(np.polyfit(g2.z,g2.surprise,1)[0],2),n_trades=len(t),hit=round((t.pnl>0).mean(),2),mean_pnl=round(t.pnl.mean(),2),median_pnl=round(t.pnl.median(),2),t=round(t.pnl.mean()/(t.pnl.std()/np.sqrt(len(t))),2),dull_n=int((t.pos==1).sum()),dull_hit=round((t[t.pos==1].pnl>0).mean(),2),bright_n=int((t.pos==-1).sum()),bright_hit=round((t[t.pos==-1].pnl>0).mean(),2),sd_surprise=round(g2.surprise.std(),2)))
print('\nRobust (excluding 2021-02):'); print(pd.DataFrame(rows).to_string(index=False))
