"""THE question, in the CPE framework: if we knew today that the next f days will be an extreme in irradiance (perfect foresight; ERA5 stands in), is the JEPX price over those days an extreme move from the level we can see today, and would trading it make money?

For every start date t (decision at the close of day t):
  baseline B_t = mean midday price over the 30 days before t (known at t)                       [causal]
  outcome  Y   = ln( mean midday price over days t+1..t+f / B_t ),  prices floored at 1 yen/kWh before the log
  signal   X   = mean daily ERA5 irradiance anomaly over days t+1..t+f (z against the calendar-month climate of 1996-2015)   [ORACLE: not knowable at t]
  X tails: dull / bright at q = 20%, 10%, 5%, 2% of the 1996-2015 distribution of the f-day mean anomaly (frozen before JEPX).
  Y tails: top / bottom decile of Y over FORMATION start dates 2016-05..2020-12 (frozen). Test start dates 2021-01..2026-08 (windows end by 2026-09-30).
  Mechanism (registered): dull -> Y up, bright -> Y down.
CPE = P(Y beyond its tail | X in its tail); lift = CPE / base rate matched on the calendar month of t. f = 7, 14, 30 days. Windows overlap (every start date is used); episodes = start dates more than f days apart.
Money: long when X is dull, short when X is bright; P&L = Y minus the formation-window mean of Y for that calendar month of t (the usual seasonal drift), in log points; no costs.
Areas: Tokyo, Kyushu (Fukuoka irradiance), system (Tokyo irradiance); price: midday slots 21-30 (10:00-15:00 JST) and all slots."""
import glob, io, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,'data')
FORM=('2016-05-01','2020-12-31'); TEST=('2021-01-01','2026-08-31'); QS=[0.20,0.10,0.05,0.02]; FS=[7,14,30]
fr=[]
for f in sorted(glob.glob(os.path.join(D,'spot_*.csv'))):
    d=pd.read_csv(io.StringIO(open(f,encoding='utf-8').read())); d=d.iloc[:,[0,1,5,8,14]]; d.columns=['date','slot','system','tokyo','kyushu']; fr.append(d)
J=pd.concat(fr); J['date']=pd.to_datetime(J['date'],format='%Y/%m/%d'); J=J.drop_duplicates(['date','slot'])
idx=pd.date_range('2016-04-01','2026-09-30')
def daily(lo,hi):
    g=J[(J.slot>=lo)&(J.slot<=hi)].groupby('date').agg(system=('system','mean'),tokyo=('tokyo','mean'),kyushu=('kyushu','mean'),n=('slot','count')); return g[g.n==hi-lo+1].drop(columns='n').reindex(idx)
PRICE={'midday':daily(21,30),'all-day':daily(1,48)}
def era(fn):
    j=json.load(open(os.path.join(D,fn)))['daily']; return pd.Series(j['shortwave_radiation_sum'],index=pd.to_datetime(j['time'])).astype(float)
ERA={'tokyo':era('era5_tokyo.json'),'kyushu':era('era5_fukuoka.json')}; ERA['system']=ERA['tokyo']
def zdaily(S):
    b=S[(S.index.year>=1996)&(S.index.year<=2015)]; c=b.groupby(b.index.month).agg(['mean','std']); return (S-S.index.month.map(c['mean']).values)/S.index.month.map(c['std']).values
def episodes(dates,gap):
    if len(dates)==0: return 0
    d=np.array(sorted(dates),dtype='datetime64[D]'); return 1+int((np.diff(d).astype(int)>gap).sum())
rows=[]; pnl_rows=[]
for pk,P in PRICE.items():
    for area in ['tokyo','kyushu','system']:
        p=P[area].clip(lower=1.0).where(P[area].notna()); z=zdaily(ERA[area]); zfull=z.copy(); z=z.reindex(idx)
        B=p.rolling(30,min_periods=24).mean().shift(1)   # mean of the 30 days before t... shifted so it ends at t-1; decision at close of t uses t as well, so use window ending t
        B=p.rolling(30,min_periods=24).mean()             # days t-29..t, known at the close of t
        for f in FS:
            O=p.rolling(f,min_periods=int(0.8*f)).mean().shift(-f)           # mean of t+1..t+f
            Y=np.log(O/B)
            Xz=z.rolling(f,min_periods=f).mean().shift(-f)                   # oracle f-day irradiance anomaly over t+1..t+f
            zb=zfull.rolling(f,min_periods=f).mean(); zb=zb[(zb.index.year>=1996)&(zb.index.year<=2015)].dropna()
            df=pd.DataFrame({'X':Xz,'Y':Y}).dropna(); fm=(df.index>=FORM[0])&(df.index<=FORM[1]); tm=(df.index>=TEST[0])&(df.index<=TEST[1])
            y90,y10=df.Y[fm].quantile(.9),df.Y[fm].quantile(.1); drift=df.Y[fm].groupby(df.index[fm].month).mean()
            for xtail in ['dull','bright']:
                for q in QS:
                    th=zb.quantile(q) if xtail=='dull' else zb.quantile(1-q); cond=(df.X<=th) if xtail=='dull' else (df.X>=th); ydir='up' if xtail=='dull' else 'down'
                    ex=(df.Y>=y90) if ydir=='up' else (df.Y<=y10); rec=dict(price=pk,area=area,f=f,xtail=xtail,q=q,ydir=ydir)
                    for w,win in (('form',fm),('test',tm)):
                        m=win&cond.values; n=int(m.sum()); rec['n_'+w]=n
                        if n==0: rec.update({'cpe_'+w:np.nan,'base_'+w:np.nan,'lift_'+w:np.nan,'ep_'+w:0}); continue
                        cpe=float(ex[m].mean()); bm=ex[win].groupby(df.index[win].month).mean(); base=float(bm.reindex(df.index[m].month).mean())
                        rec.update({'cpe_'+w:cpe,'base_'+w:base,'lift_'+w:cpe/base if base>0 else np.nan,'ep_'+w:episodes(df.index[m],f)})
                        if w=='test':
                            sur=df.Y[m]-drift.reindex(df.index[m].month).values; pos=1 if xtail=='dull' else -1; pn=pos*sur
                            rec.update(pnl_mean=float(pn.mean()),pnl_hit=float((pn>0).mean()),pnl_sd=float(pn.std()),n_eff=n/f)
                    rows.append(rec)
R=pd.DataFrame(rows); R.to_csv(os.path.join(HERE,'cpe_oracle_forward.csv'),index=False)
pd.set_option('display.width',220)
print('PERFECT-FORESIGHT CPE (mechanism direction). cpe/base/lift on formation and test; episodes = start dates >f days apart')
for pk in ['midday','all-day']:
    for f in FS:
        s=R[(R.price==pk)&(R.f==f)&(R.q.isin([0.10,0.02]))]
        print('\n',pk,'f =',f,'days'); print(s[['area','xtail','q','n_form','ep_form','cpe_form','base_form','lift_form','n_test','ep_test','cpe_test','base_test','lift_test','pnl_mean','pnl_hit','n_eff']].round(2).to_string(index=False))
# summary: lift by rarity, test, averaged over areas, per f (midday)
S=R[(R.price=='midday')]; print('\nMean TEST lift by rarity (midday, all areas):'); print(S.groupby(['f','xtail','q']).lift_test.mean().round(2).unstack('q').to_string())
print('\nMean FORMATION lift by rarity (midday, all areas):'); print(S.groupby(['f','xtail','q']).lift_form.mean().round(2).unstack('q').to_string())
print('\nTEST P&L, log points per trade (mechanism direction), midday; dull = long, bright = short; mean [hit rate] and effective N (n/f):')
for f in FS:
    s=S[(S.f==f)&(S.q==0.10)]; print('f=%d q=10%%:'%f,'; '.join(f"{r.area} {r.xtail}: {r.pnl_mean*100:+.1f}% [{r.pnl_hit:.2f}] neff {r.n_eff:.0f}" for r in s.itertuples()))
# figure: test lift by rarity for each f, midday, dull on left, bright on right
fig,ax=plt.subplots(1,2,figsize=(12.5,4.6))
for a,xt in zip(ax,['dull','bright']):
    for f,ls in zip([7,14],['-','--']):
        for area,col in zip(['tokyo','kyushu'],['tab:red','tab:green']):
            s=S[(S.area==area)&(S.f==f)&(S.xtail==xt)].sort_values('q',ascending=False); a.plot(range(len(s)),s.lift_test,color=col,ls=ls,marker='o',ms=3,label=f'{area} {f}d')
    a.set_xticks(range(len(QS))); a.set_xticklabels([f'{int(q*100)}%' for q in sorted(QS,reverse=True)]); a.axhline(1,color='k',lw=.6); a.set_xlabel('rarity of the f-day irradiance anomaly (rarer to the right)'); a.set_ylabel('test lift over base'); a.set_title(('dull window -> price over the window rises' if xt=='dull' else 'bright window -> price over the window falls'),fontsize=9); a.legend(fontsize=8,ncol=2)
fig.suptitle('PERFECT-FORESIGHT CPE, midday price: if the next f days are known to be extreme, how likely is an extreme price move from today\'s level?\nTest 2021-2026; 30-day windows and the system price are in the csv',fontsize=9)
plt.tight_layout(rect=(0,0,1,0.94)); plt.savefig(os.path.join(HERE,'cpe_oracle_forward_figure.png'),dpi=140); print('wrote csv and figure')
