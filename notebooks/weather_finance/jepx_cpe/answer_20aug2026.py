"""Antony Stace's question, answered with the perfect-foresight CPE: standing on 20 August 2026, what would the model have said about the Tokyo-area JEPX outcome for September 2026?

Logic (same as cpe_oracle_forward.py, with a lead of 11 days because September starts 12 days after the 20th):
  decision day t; baseline B_t = mean price over the 30 days up to t (known at t); window = days t+12 .. t+41 (for t = 20 Aug 2026: 1-30 September);
  X = mean daily ERA5 irradiance anomaly over the window (ORACLE: perfect foresight of the weather, a ceiling for any real forecast);
  Y = ln(window mean price / B_t) (prices floored at 1 yen); also the plain yen change.
  X tails (dull) at q = 20%, 10%, 5%, 2% of the 1996-2015 distribution of 30-day mean anomalies; Y top decile frozen on decision days 2016-05..2020-12.
  AS OF 20 AUGUST 2026: only windows that had ended by 19 August 2026 (decision day <= 2026-07-09) are used to estimate CPE, lift (season matched on the calendar month of the decision day), episodes (decision days more than 30 days apart) and the mean surprise (yen change minus the seasonal drift of that calendar month).
Then the realised September 2026 outcome is shown next to it."""
import glob, io, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,'data')
LEAD=11; F=30; T0=pd.Timestamp('2026-08-20'); CUT=T0-pd.Timedelta(days=1)+pd.Timedelta(days=0)
FORM_END=pd.Timestamp('2020-12-31'); QS=[0.20,0.10,0.05,0.02]
fr=[]
for f in sorted(glob.glob(os.path.join(D,'spot_*.csv'))):
    d=pd.read_csv(io.StringIO(open(f,encoding='utf-8').read())); d=d.iloc[:,[0,1,5,8,14]]; d.columns=['date','slot','system','tokyo','kyushu']; fr.append(d)
J=pd.concat(fr); J['date']=pd.to_datetime(J['date'],format='%Y/%m/%d'); J=J.drop_duplicates(['date','slot'])
idx=pd.date_range('2016-04-01','2026-09-30')
def daily(lo,hi):
    g=J[(J.slot>=lo)&(J.slot<=hi)].groupby('date').agg(system=('system','mean'),tokyo=('tokyo','mean'),kyushu=('kyushu','mean'),n=('slot','count')); return g[g.n==hi-lo+1].drop(columns='n').reindex(idx)
PRICE={'all-day':daily(1,48),'midday':daily(21,30)}
def era(fn):
    j=json.load(open(os.path.join(D,fn)))['daily']; return pd.Series(j['shortwave_radiation_sum'],index=pd.to_datetime(j['time'])).astype(float)
ERA={'tokyo':era('era5_tokyo.json'),'kyushu':era('era5_fukuoka.json')}; ERA['system']=ERA['tokyo']
def zdaily(S):
    b=S[(S.index.year>=1996)&(S.index.year<=2015)]; c=b.groupby(b.index.month).agg(['mean','std']); return (S-S.index.month.map(c['mean']).values)/S.index.month.map(c['std']).values
def episodes(dates,gap):
    if len(dates)==0: return 0
    d=np.array(sorted(dates),dtype='datetime64[D]'); return 1+int((np.diff(d).astype(int)>gap).sum())
res=[]; detail={}
for pk in ['all-day','midday']:
    for area in ['tokyo','system','kyushu']:
        p=PRICE[pk][area].clip(lower=1.0).where(PRICE[pk][area].notna()); praw=PRICE[pk][area]
        zf=zdaily(ERA[area]); z=zf.reindex(idx)
        B=p.rolling(30,min_periods=24).mean(); Braw=praw.rolling(30,min_periods=24).mean()
        O=p.rolling(F,min_periods=int(0.8*F)).mean().shift(-(LEAD+F)); Oraw=praw.rolling(F,min_periods=int(0.8*F)).mean().shift(-(LEAD+F))
        X=z.rolling(F,min_periods=F).mean().shift(-(LEAD+F))
        zb=zf.rolling(F,min_periods=F).mean(); zb=zb[(zb.index.year>=1996)&(zb.index.year<=2015)].dropna()
        df=pd.DataFrame({'X':X,'Y':np.log(O/B),'dY':Oraw-Braw}); now=df.loc[T0]; df=df.dropna(); df['end']=df.index+pd.Timedelta(days=LEAD+F)
        hist=df[df.end<=T0-pd.Timedelta(days=1)]          # windows already finished on the morning of 20 August 2026
        y90=hist.Y[hist.index<=FORM_END].quantile(.9); ex=hist.Y>=y90; drift=hist.dY.groupby(hist.index.month).mean()
        xnow=X.loc[T0] if not np.isnan(X.loc[T0]) else np.nan
        for q in QS:
            th=zb.quantile(q); m=(hist.X<=th).values; n=int(m.sum()); base_m=ex.groupby(hist.index.month).mean()
            rec=dict(price=pk,area=area,q=q,xth=round(th,2),n=n,episodes=episodes(hist.index[m],F))
            if n:
                cpe=float(ex[m].mean()); base=float(base_m.reindex(hist.index[m].month).mean()); sur=hist.dY[m]-drift.reindex(hist.index[m].month).values
                rec.update(cpe=cpe,base=base,lift=cpe/base if base>0 else np.nan,mean_surprise_yen=float(sur.mean()),median_surprise_yen=float(sur.median()),share_up=float((sur>0).mean()))
            res.append(rec)
        detail[(pk,area)]=dict(xnow=xnow,pct=(zb<xnow).mean()*100 if not np.isnan(xnow) else np.nan,Bnow=Braw.loc[T0],Onow=Oraw.shift(-(LEAD+F)) .loc[T0] if False else df['dY'].get(T0,np.nan), th=zb.quantile([.02,.05,.10,.20]).round(2).tolist(),drift_aug=float(drift.get(8,np.nan)),y90=y90)
        # realised September 2026 values
        sep=praw['2026-09-01':'2026-09-30'].mean(); detail[(pk,area)]['sep_mean']=sep; detail[(pk,area)]['dy_actual']=sep-Braw.loc[T0]
R=pd.DataFrame(res); R.to_csv(os.path.join(HERE,'answer_20aug2026.csv'),index=False)
pd.set_option('display.width',200)
print('STATE ON 20 AUGUST 2026 (perfect foresight of September 2026)')
for k,v in detail.items():
    if k[1]!='tokyo' and k!=('midday','kyushu'): continue
    print(k,'| baseline (30 days to 20 Aug) %.2f yen/kWh | Sep 2026 window mean anomaly %.2f daily-sd, rarer than %.1f%% of 30-day windows in 1996-2015 (tail cuts 2/5/10/20%%: %s) | Sep mean %.2f, change %+.2f yen | drift for August start dates %+.2f | top-decile cut of the log move %.3f'%(v['Bnow'],v['xnow'],100-v['pct'],v['th'],v['sep_mean'],v['dy_actual'],v['drift_aug'],v['y90']))
for pk in ['all-day','midday']:
    for area in ['tokyo','system','kyushu']:
        s=R[(R.price==pk)&(R.area==area)]; print('\n',pk,area,'- windows finished by 19 Aug 2026, dull tails (CPE = P(price over the window in its top decile); mean surprise in yen/kWh vs seasonal drift)'); print(s[['q','xth','n','episodes','cpe','base','lift','mean_surprise_yen','median_surprise_yen','share_up']].round(2).to_string(index=False))
# the ten Septembers as a plain cross-check: decision on 20 Aug, window Sept 1-30
print('\nTEN SEPTEMBERS (decision 20 Aug; Tokyo; mean price Sep minus mean price of 22 Jul-20 Aug): anomaly of September (daily-sd mean), change in yen/kWh')
for pk in ['all-day','midday']:
    praw=PRICE[pk]['tokyo']; z=zdaily(ERA['tokyo']); rows=[]
    for y in range(2016,2027):
        t=pd.Timestamp(f'{y}-08-20'); b=praw[t-pd.Timedelta(days=29):t].mean(); o=praw[f'{y}-09-01':f'{y}-09-30'].mean(); rows.append((y,round(z[f'{y}-09-01':f'{y}-09-30'].mean(),2),round(o-b,2)))
    print(pk,rows)
fig,ax=plt.subplots(1,2,figsize=(12.5,4.6))
for a,pk in zip(ax,['all-day','midday']):
    s=R[(R.price==pk)&(R.area=='tokyo')].sort_values('q',ascending=False); a.bar(np.arange(len(s))-0.18,s.cpe,0.36,label='CPE: P(price in its top decile)',color='tab:red'); a.bar(np.arange(len(s))+0.18,s.base,0.36,label='season-matched base rate',color='lightgray')
    for i,r in enumerate(s.itertuples()): a.text(i,max(r.cpe,r.base)*1.03,f'{r.n} days, {r.episodes} episodes',ha='center',fontsize=7)
    a.set_xticks(range(len(s))); a.set_xticklabels([f'dullest {int(q*100)}%' for q in s.q],fontsize=8); a.set_title(f'Tokyo {pk}: as of 20 Aug 2026, if September is known to be dull',fontsize=9); a.set_ylabel('probability'); a.set_ylim(0,max(s.base.max(),s.cpe.max())*1.45); a.legend(fontsize=7,loc='upper left')
plt.tight_layout(); plt.savefig(os.path.join(HERE,'answer_20aug2026_figure.png'),dpi=140); print('wrote answer_20aug2026.csv and figure')

# ---- which episodes drive the Tokyo all-day dull tail (q = 5%)? decision days grouped into episodes (gap > 30 days)
p=PRICE['all-day']['tokyo'].clip(lower=1.0).where(PRICE['all-day']['tokyo'].notna()); praw=PRICE['all-day']['tokyo']; zf=zdaily(ERA['tokyo']); z=zf.reindex(idx)
B=p.rolling(30,min_periods=24).mean(); Braw=praw.rolling(30,min_periods=24).mean(); Oraw=praw.rolling(F,min_periods=int(0.8*F)).mean().shift(-(LEAD+F)); X=z.rolling(F,min_periods=F).mean().shift(-(LEAD+F))
zb=zf.rolling(F,min_periods=F).mean(); zb=zb[(zb.index.year>=1996)&(zb.index.year<=2015)].dropna(); th=zb.quantile(0.05)
df=pd.DataFrame({'X':X,'dY':Oraw-Braw,'B':Braw}).dropna(); df['end']=df.index+pd.Timedelta(days=LEAD+F); hist=df[df.end<=T0-pd.Timedelta(days=1)]; drift=hist.dY.groupby(hist.index.month).mean()
hist=hist.assign(sur=hist.dY-drift.reindex(hist.index.month).values); d=hist[hist.X<=th]
grp=(d.index.to_series().diff().dt.days>30).cumsum(); print('\nTokyo all-day dull tail q=5%: episodes (first/last decision day, window of the first day, mean anomaly, baseline level, mean surprise yen/kWh)')
for k,gg in d.groupby(grp.values):
    print(gg.index[0].date(),gg.index[-1].date(),'window starts',(gg.index[0]+pd.Timedelta(days=LEAD+1)).date(),'n=%d'%len(gg),'X %.2f'%gg.X.mean(),'baseline %.1f'%gg.B.mean(),'surprise %+.2f'%gg.sur.mean())
