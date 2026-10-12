"""CPE form of the dose-response: does the RARITY of a dull (or bright) day change the probability of an extreme midday JEPX price on that day?

CPE(q_X) = P(Y beyond its tail | X in its q_X tail), lift = CPE / season-matched base rate. Contemporaneous (tau_f = 0): the day-ahead price of day D is set with the forecast for D, and ERA5 stands in for what the sky did.
Frozen before looking at JEPX:
  X  = daily ERA5 irradiance anomaly (calendar-month z against 1996-2015; Tokyo irradiance for Tokyo and system, Fukuoka for Kyushu).
       Tails: dull (z <= its q quantile) and bright (z >= its 1-q quantile) with q in 20%, 10%, 5%, 2%, thresholds from the 1996-2015 distribution of daily z.
  Y  = response r of the day's midday (10:00-15:00 JST) mean price against the mean of the other non-holiday weekdays within +/-15 days, in percent (baseline floored: days with baseline below 1 yen/kWh are dropped).
       Exceedance: r in the top 10% (for dull X) or bottom 10% (for bright X) of r in the FORMATION window; thresholds frozen there.
Mechanism (registered): dull -> midday price up; bright -> midday price down. The other directions are reported as a control.
Formation 2016-04..2020-12; test 2021-01..2026-09. Base rate: frequency of the same exceedance on the same calendar months within the same window.
Episodes: extreme days separated by more than 3 calendar days start a new episode.
Admission on formation (adapted from the program's gates to a daily same-day effect): lift >= 1.5, >= 20 days, >= 10 episodes. Scored on the test window."""
import glob, io, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, jpholiday
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,'data')
FORM=('2016-04-01','2020-12-31'); TEST=('2021-01-01','2026-09-30'); QS=[0.20,0.10,0.05,0.02]
fr=[]
for f in sorted(glob.glob(os.path.join(D,'spot_*.csv'))):
    d=pd.read_csv(io.StringIO(open(f,encoding='utf-8').read())); d=d.iloc[:,[0,1,5,8,14]]; d.columns=['date','slot','system','tokyo','kyushu']; fr.append(d)
J=pd.concat(fr); J['date']=pd.to_datetime(J['date'],format='%Y/%m/%d'); J=J.drop_duplicates(['date','slot']); J=J[(J.slot>=21)&(J.slot<=30)]
g=J.groupby('date').agg(system=('system','mean'),tokyo=('tokyo','mean'),kyushu=('kyushu','mean'),n=('slot','count')); g=g[g.n==10].drop(columns='n').reindex(pd.date_range('2016-04-01','2026-09-30'))
work=pd.Series([not(x.weekday()>=5 or jpholiday.is_holiday(x.date())) for x in g.index],index=g.index)
def era(fn):
    j=json.load(open(os.path.join(D,fn)))['daily']; return pd.Series(j['shortwave_radiation_sum'],index=pd.to_datetime(j['time'])).astype(float)
ERA={'tokyo':era('era5_tokyo.json'),'kyushu':era('era5_fukuoka.json')}; ERA['system']=ERA['tokyo']
def daily_z(S):
    b=S[(S.index.year>=1996)&(S.index.year<=2015)]; c=b.groupby(b.index.month).agg(['mean','std']); return (S-S.index.month.map(c['mean']).values)/S.index.month.map(c['std']).values, (S-S.index.month.map(c['mean']).values)/S.index.month.map(c['std']).values
def episodes(dates):
    if len(dates)==0: return 0
    d=np.array(sorted(dates),dtype='datetime64[D]'); return 1+int((np.diff(d).astype(int)>3).sum())
rows=[]; ndrop={}
for area in ['tokyo','kyushu','system']:
    p=g[area].where(work); zfull=daily_z(ERA[area])[0]; z=zfull.reindex(g.index); zb=zfull[(zfull.index.year>=1996)&(zfull.index.year<=2015)]
    s=p.rolling(31,center=True,min_periods=1).sum()-p.fillna(0); c=p.rolling(31,center=True,min_periods=1).count()-p.notna().astype(int); base=(s/c).where(c>=8)
    ok=base>=1.0; ndrop[area]=int((p.notna()&base.notna()&~ok).sum()); r=((p/base-1)*100).where(ok)
    df=pd.DataFrame({'z':z,'r':r}).dropna(); fm=(df.index>=FORM[0])&(df.index<=FORM[1])
    r90,r10=df.r[fm].quantile(0.90),df.r[fm].quantile(0.10)
    for xtail in ['dull','bright']:
        for q in QS:
            th=zb.quantile(q) if xtail=='dull' else zb.quantile(1-q); cond=(df.z<=th) if xtail=='dull' else (df.z>=th)
            for ydir in ['up','down']:
                ex=(df.r>=r90) if ydir=='up' else (df.r<=r10); rec=dict(area=area,xtail=xtail,q=q,ydir=ydir,mechanism=(xtail=='dull')==(ydir=='up'),xth=round(th,2),rth=round(r90 if ydir=='up' else r10,1))
                for w,(a,b) in (('form',FORM),('test',TEST)):
                    win=(df.index>=a)&(df.index<=b); m=win&cond.values; n=int(m.sum()); rec['n_'+w]=n
                    if n==0: rec.update({'cpe_'+w:np.nan,'base_'+w:np.nan,'lift_'+w:np.nan,'ep_'+w:0}); continue
                    cpe=float(ex[m].mean()); bm=ex[win].groupby(df.index[win].month).mean(); base_=float(bm.reindex(df.index[m].month).mean())
                    rec.update({'cpe_'+w:cpe,'base_'+w:base_,'lift_'+w:cpe/base_ if base_>0 else np.nan,'ep_'+w:episodes(df.index[m])})
                rows.append(rec)
R=pd.DataFrame(rows); R['admit']=(R.lift_form>=1.5)&(R.n_form>=20)&(R.ep_form>=10)
R.to_csv(os.path.join(HERE,'cpe_daily_dose.csv'),index=False)
print('days dropped for baseline < 1 yen/kWh:',ndrop)
pd.set_option('display.width',200)
M=R[R.mechanism]
print('\nMECHANISM DIRECTION (dull -> price up, bright -> price down): CPE and lift by rarity q_X')
print(M[['area','xtail','q','n_form','ep_form','cpe_form','base_form','lift_form','n_test','ep_test','cpe_test','base_test','lift_test','admit']].round(2).to_string(index=False))
print('\nCONTROL (opposite direction) test lift, mean by q_X:'); print(R[~R.mechanism].groupby(['xtail','q']).lift_test.mean().round(2).unstack(0))
print('\nAdmitted on formation:',int(R.admit.sum()),'of',len(R),'| of the mechanism rows:',int(M.admit.sum()),'of',len(M))
A=M[M.admit]; print('Admitted mechanism rows, test: mean lift %.2f, share with lift >= 1.5: %.2f, share above base: %.2f'%(A.lift_test.mean(),(A.lift_test>=1.5).mean(),(A.cpe_test>A.base_test).mean()))
# figure: test lift by rarity, mechanism direction
fig,ax=plt.subplots(1,2,figsize=(12.5,4.6))
for a,xt,ttl in zip(ax,['dull','bright'],['dull day -> midday price in its top decile','bright day -> midday price in its bottom decile']):
    for area,col in zip(['tokyo','kyushu','system'],['tab:red','tab:green','tab:gray']):
        s=M[(M.area==area)&(M.xtail==xt)].sort_values('q',ascending=False); a.plot(range(len(s)),s.lift_test,marker='o',color=col,label=area+' test'); a.plot(range(len(s)),s.lift_form,marker='o',ls=':',color=col,alpha=.6,label=area+' formation')
        a.set_xticks(range(len(s))); a.set_xticklabels([f'{int(q*100)}%' for q in s.q])
    a.axhline(1,color='k',lw=.6); a.set_xlabel('rarity of the day (X tail, rarer to the right)'); a.set_ylabel('lift over season-matched base rate'); a.set_title(ttl,fontsize=9); a.legend(fontsize=7,ncol=2)
fig.suptitle('Same-day CPE of irradiance extremes and the midday JEPX price (dotted: formation 2016-2020; solid: test 2021-2026)',fontsize=9.5); plt.tight_layout(rect=(0,0,1,0.95)); plt.savefig(os.path.join(HERE,'cpe_daily_dose_figure.png'),dpi=140); print('wrote cpe_daily_dose.csv, cpe_daily_dose_figure.png')
