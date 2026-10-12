"""CPE on irradiance extremes and JEPX prices (answers: "look at all irradiance extremes and relate them to JEPX prices using the CPE framework").

CPE(X tail -> Y tail) = P(Y's forward f-day move beyond its q_Y threshold | X's past p-day state beyond its q_X tail), lift = CPE / season-matched base rate.
Frozen before looking at JEPX: X = mean daily ERA5 irradiance z-score over the past p days, z against the calendar-month mean and sd of 1996-2015 (before JEPX exists);
  X thresholds = quantiles of that 1996-2015 distribution of the p-day state (both tails, q_X in 2%, 5%, 10%); p in 7, 14, 30, 60.
Y (Tokyo, Kyushu, system): (a) price move = ln(mean price over the next f days / mean price over the previous f days); (b) premium = mean of (area - system) over the next f days; f in 7, 14, 30.
  Y thresholds (upper and lower 10%) are frozen from the formation window 2016-04..2020-12; the test window is 2021-01..2026-09.
Base rate is season matched: the exceedance frequency of Y inside the same window on the same calendar months as the conditioning days.
Episodes: conditioning days separated by more than f days start a new episode.
Gates as in the program: CPE >= 0.80, lift >= 1.5, n >= 100 days (formation); survivor test-side: >= 3 test episodes, test CPE >= base + 0.10.
"""
import glob, io, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,'data')
FORM=('2016-04-01','2020-12-31'); TEST=('2021-01-01','2026-09-30')
fr=[]
for f in sorted(glob.glob(os.path.join(D,'spot_*.csv'))):
    df=pd.read_csv(io.StringIO(open(f,encoding='utf-8').read())); df.columns=['date','slot','sell','buy','vol','system','hokkaido','tohoku','tokyo','chubu','hokuriku','kansai','chugoku','shikoku','kyushu','a','b','c','d']; fr.append(df[['date','slot','system','tokyo','kyushu']])
J=pd.concat(fr); J['date']=pd.to_datetime(J['date'],format='%Y/%m/%d'); J=J.drop_duplicates(['date','slot'])
day=J.groupby('date').agg(system=('system','mean'),tokyo=('tokyo','mean'),kyushu=('kyushu','mean'),n=('slot','count')); day=day[day.n==48].drop(columns='n')
idx=pd.date_range('2016-04-01','2026-09-30'); day=day.reindex(idx)
def era(fn):
    d=json.load(open(os.path.join(D,fn)))['daily']; return pd.Series(d['shortwave_radiation_sum'],index=pd.to_datetime(d['time'])).astype(float)
SITE={'tokyo':era('era5_tokyo.json'),'kyushu':era('era5_fukuoka.json')}
SITE['system']=SITE['tokyo']
PS=[7,14,30,60]; FS=[7,14,30]; QX=[0.02,0.05,0.10]
def zser(S):
    b=S[(S.index.year>=1996)&(S.index.year<=2015)]; c=b.groupby(b.index.month).agg(['mean','std']); return (S-S.index.month.map(c['mean']).values)/S.index.month.map(c['std']).values
def run_lengths_episodes(dates,gap):
    if len(dates)==0: return 0
    d=np.array(sorted(dates),dtype='datetime64[D]'); return 1+int((np.diff(d).astype(int)>gap).sum())
rows=[]
for area in ['tokyo','kyushu','system']:
    z=zser(SITE[area]); zz=z.reindex(pd.date_range('1996-01-01','2026-09-30'))
    for p in PS:
        xs=zz.rolling(p,min_periods=p).mean(); pre=xs[:'2015-12-31'].dropna(); xs=xs.reindex(idx)
        for tail in ['dull','bright']:
            for q in QX:
                th=pre.quantile(q) if tail=='dull' else pre.quantile(1-q); cond=(xs<=th) if tail=='dull' else (xs>=th)
                for f in FS:
                    for ytype in (['move','prem'] if area!='system' else ['move']):
                        px=day[area]
                        if ytype=='move': y=np.log(px.rolling(f,min_periods=f).mean().shift(-f)/px.rolling(f,min_periods=f).mean())
                        else: y=(day[area]-day['system']).rolling(f,min_periods=f).mean().shift(-f)
                        fm=(y.index>=FORM[0])&(y.index<=FORM[1])
                        for ydir in ['up','down']:
                            yth=y[fm].quantile(0.90 if ydir=='up' else 0.10); ex=(y>=yth) if ydir=='up' else (y<=yth)
                            rec=dict(area=area,ytype=ytype,f=f,p=p,xtail=tail,qx=q,ydir=ydir,xth=round(th,3),yth=round(yth,4))
                            for w,(a,b) in (('form',FORM),('test',TEST)):
                                m=(y.index>=a)&(y.index<=b)&y.notna().values&cond.fillna(False).values
                                allm=(y.index>=a)&(y.index<=b)&y.notna().values
                                n=int(m.sum()); rec['n_'+w]=n
                                if n==0: rec.update({'cpe_'+w:np.nan,'base_'+w:np.nan,'lift_'+w:np.nan,'ep_'+w:0}); continue
                                cpe=float(ex[m].mean()); mon=y.index[m].month; bm=ex[allm].groupby(y.index[allm].month).mean(); base=float(bm.reindex(mon).mean())
                                rec.update({'cpe_'+w:cpe,'base_'+w:base,'lift_'+w:cpe/base if base>0 else np.nan,'ep_'+w:run_lengths_episodes(y.index[m],f)})
                            rows.append(rec)
R=pd.DataFrame(rows)
R['gate_form']=(R.cpe_form>=0.80)&(R.lift_form>=1.5)&(R.n_form>=100)
R['lift_ok_form']=(R.lift_form>=1.5)&(R.n_form>=100)&(R.ep_form>=3)
R['survivor']=R.gate_form&(R.ep_test>=3)&(R.cpe_test>=R.base_test+0.10)
R.to_csv(os.path.join(HERE,'jepx_cpe_scan.csv'),index=False)
print('conditions scanned',len(R),'| distinct (area,ytype,f,p,tail,qx,dir)')
print('pass the program gate on formation (CPE>=0.80, lift>=1.5, n>=100 days):',int(R.gate_form.sum()))
print('survivors (gate + >=3 test episodes + test CPE >= base+0.10):',int(R.survivor.sum()))
A=R[R.lift_ok_form]; print('\nlift-only admission on formation (lift>=1.5, n>=100, >=3 episodes):',len(A),'of',len(R))
if len(A):
    A=A.assign(dl=A.lift_test-A.lift_form)
    print('test lift of admitted rows: mean %.2f, median %.2f, share with test lift>=1.5: %.2f, share with test CPE > base: %.2f'%(A.lift_test.mean(),A.lift_test.median(),(A.lift_test>=1.5).mean(),(A.cpe_test>A.base_test).mean()))
    print('mean test excess (CPE - base, points) by q_X:'); print((A.assign(ex=(A.cpe_test-A.base_test)*100).groupby('qx').ex.agg(['mean','count']).round(1)))
    print('by area:'); print((A.assign(ex=(A.cpe_test-A.base_test)*100).groupby('area').ex.agg(['mean','count']).round(1)))
    print('by X tail:'); print((A.assign(ex=(A.cpe_test-A.base_test)*100).groupby('xtail').ex.agg(['mean','count']).round(1)))
# unrestricted: dull-irradiance -> price up, bright -> price down (the physical mechanism), all conditions with n_form>=100
M=R[(R.n_form>=100)&(R.n_test>=30)].copy(); M['ex_test']=(M.cpe_test-M.base_test)*100
mech=M[((M.xtail=='dull')&(M.ydir=='up'))|((M.xtail=='bright')&(M.ydir=='down'))]; anti=M[((M.xtail=='dull')&(M.ydir=='down'))|((M.xtail=='bright')&(M.ydir=='up'))]
print('\nTEST-window excess over season-matched base (points), all conditions with >=100 formation days and >=30 test days:')
print('  mechanism-consistent (dull->price up, bright->price down): mean %.1f over %d conditions'%(mech.ex_test.mean(),len(mech)))
print('  opposite (dull->down, bright->up): mean %.1f over %d conditions'%(anti.ex_test.mean(),len(anti)))
print(mech.groupby(['area','ytype']).ex_test.agg(['mean','count']).round(1))
# where do the largest formation+test agreements sit
top=R[(R.n_form>=100)&(R.n_test>=30)&(R.lift_form>=1.5)&(R.lift_test>=1.5)].sort_values('lift_test',ascending=False)
print('\nconditions with lift>=1.5 in BOTH windows (n>=100 form, >=30 test):',len(top))
print(top.head(15)[['area','ytype','f','p','xtail','qx','ydir','n_form','n_test','cpe_form','base_form','cpe_test','base_test','ep_form','ep_test']].round(2).to_string(index=False))
# Antony's date: where irradiance stood on 20 Aug 2026
for area in ['tokyo','kyushu']:
    z=zser(SITE[area]); zz=z.reindex(pd.date_range('1996-01-01','2026-09-30'))
    for p in PS:
        xs=zz.rolling(p,min_periods=p).mean(); pre=xs[:'2015-12-31'].dropna(); v=xs.loc['2026-08-19']
        print('%s p=%d on 19-Aug-2026: state %.2f sd, percentile of 1996-2015 distribution %.0f; dull 10%% cut %.2f, bright 10%% cut %.2f'%(area,p,v,(pre<v).mean()*100,pre.quantile(.10),pre.quantile(.90)))
# figure
fig,ax=plt.subplots(1,2,figsize=(13,5.2))
S=R[(R.n_form>=100)&(R.n_test>=30)]
col=np.where(((S.xtail=='dull')&(S.ydir=='up'))|((S.xtail=='bright')&(S.ydir=='down')),'tab:blue','tab:orange')
ax[0].scatter(S.lift_form,S.lift_test,s=10+S.qx*300,c=col,alpha=.45); ax[0].axhline(1,color='k',lw=.6); ax[0].axvline(1,color='k',lw=.6); ax[0].axvline(1.5,color='gray',ls=':'); ax[0].plot([0,6],[0,6],color='gray',lw=.5)
ax[0].set_xlim(0,5); ax[0].set_ylim(0,5); ax[0].set_xlabel('lift on formation window 2016-2020'); ax[0].set_ylabel('lift on test window 2021-2026')
ax[0].set_title('Every irradiance-extreme condition vs JEPX (blue: mechanism direction; orange: opposite)',fontsize=9)
g=S.assign(ex=(S.cpe_test-S.base_test)*100,mech=np.where(col=='tab:blue','mechanism','opposite')).groupby(['xtail','mech']).ex.mean().unstack()
g.plot.bar(ax=ax[1],color=['tab:blue','tab:orange'],rot=0); ax[1].axhline(0,color='k',lw=.6); ax[1].set_ylabel('test CPE minus season-matched base (points)'); ax[1].set_xlabel('irradiance extreme'); ax[1].set_title('Average excess in the test window (2021-2026)',fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE,'jepx_cpe_figure.png'),dpi=140)
print('wrote jepx_cpe_scan.csv, jepx_cpe_figure.png')
