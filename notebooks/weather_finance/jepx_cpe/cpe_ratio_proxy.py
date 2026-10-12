"""Remove everything that is not solar from the price before the CPE: use the midday-to-night price ratio of the same days.
Night (slots 1-8, 00:00-04:00 JST) has no solar, so fuel prices, system-wide moves, demand level and slow drift move midday and night together; the log ratio ln(midday / night) keeps the solar-hours discount.
r_d = ln(midday mean / night mean) of day d, both floored at 1 yen/kWh.
Part 1  Is the response era-dependent after this correction? Daily r against the day's irradiance anomaly (dull < -1.5 sd, normal |z| < 0.5, bright > 1.5), each day relative to the mean r of the other days within +/-15 days; 2016-2020 against 2021-2026.
Part 2  AS OF 20 AUGUST 2026 (11-day lead, 30-day window, perfect foresight of September) on Y = mean r over the window minus mean r over the 30 days to the decision day; same CPE (top decile of Y frozen on decision days to 2020; dull tails 20/10/5/2%), history = windows finished by 19 August 2026.
Part 3  Perfect-foresight CPE, 7-day and 14-day windows (lead 0), formation and test, on Y; and the yen-free mean surprise in log-ratio points."""
import glob, io, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,'data')
fr=[]
for f in sorted(glob.glob(os.path.join(D,'spot_*.csv'))):
    d=pd.read_csv(io.StringIO(open(f,encoding='utf-8').read())); d=d.iloc[:,[0,1,5,8,14]]; d.columns=['date','slot','system','tokyo','kyushu']; fr.append(d)
J=pd.concat(fr); J['date']=pd.to_datetime(J['date'],format='%Y/%m/%d'); J=J.drop_duplicates(['date','slot'])
idx=pd.date_range('2016-04-01','2026-09-30')
def daily(lo,hi):
    g=J[(J.slot>=lo)&(J.slot<=hi)].groupby('date').agg(system=('system','mean'),tokyo=('tokyo','mean'),kyushu=('kyushu','mean'),n=('slot','count')); return g[g.n==hi-lo+1].drop(columns='n').reindex(idx)
MID,NIGHT=daily(21,30),daily(1,8)
def era(fn):
    j=json.load(open(os.path.join(D,fn)))['daily']; return pd.Series(j['shortwave_radiation_sum'],index=pd.to_datetime(j['time'])).astype(float)
ERA={'tokyo':era('era5_tokyo.json'),'kyushu':era('era5_fukuoka.json')}; ERA['system']=ERA['tokyo']
def zdaily(S):
    b=S[(S.index.year>=1996)&(S.index.year<=2015)]; c=b.groupby(b.index.month).agg(['mean','std']); return (S-S.index.month.map(c['mean']).values)/S.index.month.map(c['std']).values
def episodes(dates,gap):
    if len(dates)==0: return 0
    d=np.array(sorted(dates),dtype='datetime64[D]'); return 1+int((np.diff(d).astype(int)>gap).sum())
R={a:np.log(MID[a].clip(lower=1.0).where(MID[a].notna())/NIGHT[a].clip(lower=1.0).where(NIGHT[a].notna())) for a in ['tokyo','kyushu','system']}
print('PART 1: daily log(midday/night) response to irradiance, by era (mean r minus the mean of the other days within +/-15 days; log points)')
for area in ['tokyo','kyushu']:
    r=R[area]; s=r.rolling(31,center=True,min_periods=1).sum()-r.fillna(0); c=r.rolling(31,center=True,min_periods=1).count()-r.notna().astype(int); y=(r-(s/c).where(c>=8)); z=zdaily(ERA[area]).reindex(idx)
    df=pd.DataFrame({'z':z,'y':y}).dropna()
    for lab,a,b in (('2016-2020','2016-04-01','2020-12-31'),('2021-2026','2021-01-01','2026-09-30')):
        d=df[a:b]; print(area,lab,'dull<-1.5: %+.3f (n=%d) | normal: %+.3f | bright>1.5: %+.3f (n=%d)'%(d.y[d.z<-1.5].mean(),(d.z<-1.5).sum(),d.y[d.z.abs()<0.5].mean(),d.y[d.z>1.5].mean(),(d.z>1.5).sum()))
    for lab,a,b in (('2016-2020','2016-04-01','2020-12-31'),('2021-2026','2021-01-01','2026-09-30')):
        d=df[a:b]; print('    ',area,lab,'slope of response on anomaly: %.3f log-points per sd'%np.polyfit(d.z,d.y,1)[0])
LEAD,F=11,30; T0=pd.Timestamp('2026-08-20'); FORM_END=pd.Timestamp('2020-12-31'); QS=[0.20,0.10,0.05,0.02]
print('\nPART 2: as of 20 August 2026, Tokyo, Y = change in the midday/night log ratio from the 30 days to 20 Aug to the window 1-30 Sept')
rr=R['tokyo']; zf=zdaily(ERA['tokyo']); z=zf.reindex(idx); Bm=rr.rolling(30,min_periods=24).mean(); Om=rr.rolling(F,min_periods=int(.8*F)).mean().shift(-(LEAD+F)); X=z.rolling(F,min_periods=F).mean().shift(-(LEAD+F))
zb=zf.rolling(F,min_periods=F).mean(); zb=zb[(zb.index.year>=1996)&(zb.index.year<=2015)].dropna()
df=pd.DataFrame({'X':X,'Y':Om-Bm}); now=df.loc[T0]; df=df.dropna(); df['end']=df.index+pd.Timedelta(days=LEAD+F); hist=df[df.end<=T0-pd.Timedelta(days=1)]
y90=hist.Y[hist.index<=FORM_END].quantile(.9); ex=hist.Y>=y90; drift=hist.Y.groupby(hist.index.month).mean(); bm=ex.groupby(hist.index.month).mean()
print('September 2026: Y = %+.3f log points (top-decile cut %.3f); seasonal drift for August decision days %+.3f; surprise %+.3f'%(now.Y if not np.isnan(now.Y) else float('nan'),y90,drift[8],(now.Y-drift[8]) if not np.isnan(now.Y) else np.nan))
rows=[]
for q in QS:
    th=zb.quantile(q); m=(hist.X<=th).values; n=int(m.sum()); cpe=float(ex[m].mean()); base=float(bm.reindex(hist.index[m].month).mean()); sur=hist.Y[m]-drift.reindex(hist.index[m].month).values
    rows.append(dict(q=q,n=n,episodes=episodes(hist.index[m],F),cpe=cpe,base=base,lift=cpe/base if base>0 else np.nan,mean_surprise=sur.mean(),share_up=(sur>0).mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
d5=hist[hist.X<=zb.quantile(0.05)]; grp=(d5.index.to_series().diff().dt.days>30).cumsum(); print('episodes at q=5%: first decision day, n, mean anomaly, mean surprise (log points)')
for k,g_ in d5.groupby(grp.values): print('  ',g_.index[0].date(),len(g_),'%.2f'%g_.X.mean(),'%+.3f'%(g_.Y-drift.reindex(g_.index.month).values).mean())
print('\nPART 3: perfect-foresight CPE (lead 0) on the ratio proxy, mechanism direction; lift on formation (2016-05..2020-12) and test (2021-01..2026-08)')
for area in ['tokyo','kyushu']:
    rr=R[area]; zf=zdaily(ERA[area]); z=zf.reindex(idx); Bm=rr.rolling(30,min_periods=24).mean()
    for f in (7,14):
        Om=rr.rolling(f,min_periods=int(.8*f)).mean().shift(-f); X=z.rolling(f,min_periods=f).mean().shift(-f); zb=zf.rolling(f,min_periods=f).mean(); zb=zb[(zb.index.year>=1996)&(zb.index.year<=2015)].dropna()
        df=pd.DataFrame({'X':X,'Y':Om-Bm}).dropna(); fm=(df.index>='2016-05-01')&(df.index<='2020-12-31'); tm=(df.index>='2021-01-01')&(df.index<='2026-08-31')
        y90,y10=df.Y[fm].quantile(.9),df.Y[fm].quantile(.1)
        for xt in ('dull','bright'):
            out=[]
            for q in (0.10,0.05):
                th=zb.quantile(q) if xt=='dull' else zb.quantile(1-q); cond=(df.X<=th) if xt=='dull' else (df.X>=th); ex=(df.Y>=y90) if xt=='dull' else (df.Y<=y10); cell=[]
                for w,win in (('form',fm),('test',tm)):
                    m=win&cond.values; n=int(m.sum())
                    if n<10: cell.append(f'{w}: n={n}'); continue
                    cpe=float(ex[m].mean()); bmn=ex[win].groupby(df.index[win].month).mean(); b=float(bmn.reindex(df.index[m].month).mean()); cell.append(f'{w}: CPE {cpe:.2f} base {b:.2f} lift {cpe/b:.2f} (n={n}, ep={episodes(df.index[m],f)})')
                out.append(f'q={q:.2f} '+' | '.join(cell))
            print(area,f'{f}d',xt,'::',' || '.join(out))
