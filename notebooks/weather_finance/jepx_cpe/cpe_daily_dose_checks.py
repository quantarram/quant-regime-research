"""Checks on cpe_daily_dose.py: (1) Tokyo by season; (2) Kyushu with the day's temperature held near normal (dull days are also cold or wet days, so demand could be doing part of the work)."""
import os, sys, json, glob, runpy, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
ns=runpy.run_path(os.path.join(HERE,'cpe_daily_dose.py'))   # reuses g, work, ERA, FORM, TEST, daily_z (prints its own output; ignore)
g,work,ERA,FORM,TEST,daily_z=ns['g'],ns['work'],ns['ERA'],ns['FORM'],ns['TEST'],ns['daily_z']
print('\n=========== CHECKS ===========')
def frame(area):
    p=g[area].where(work); z=daily_z(ERA[area])[0].reindex(g.index); s=p.rolling(31,center=True,min_periods=1).sum()-p.fillna(0); c=p.rolling(31,center=True,min_periods=1).count()-p.notna().astype(int); base=(s/c).where(c>=8)
    r=((p/base-1)*100).where(base>=1.0); return pd.DataFrame({'z':z,'r':r}).dropna()
def lift(df,cond,ex,win):
    m=win&cond.values; n=int(m.sum())
    if n<10: return (n,np.nan,np.nan,np.nan)
    cpe=float(ex[m].mean()); bm=ex[win].groupby(df.index[win].month).mean(); b=float(bm.reindex(df.index[m].month).mean()); return (n,round(cpe,2),round(b,2),round(cpe/b,2) if b>0 else np.nan)
for area in ['tokyo','kyushu']:
    df=frame(area); fm=(df.index>=FORM[0])&(df.index<=FORM[1]); r90=df.r[fm].quantile(.9); ex=df.r>=r90
    zfull=daily_z(ERA[area])[0]; zb=zfull[(zfull.index.year>=1996)&(zfull.index.year<=2015)]; test=(df.index>=TEST[0])&(df.index<=TEST[1])
    for q in (0.10,0.05):
        th=zb.quantile(q); cond=df.z<=th
        for lab,mo in (('Jun-Sep',[6,7,8,9]),('Oct-May',[10,11,12,1,2,3,4,5])):
            win=test&np.isin(df.index.month,mo); print(area,'dull q=%.2f test %s: n, CPE, base, lift ='%(q,lab),lift(df,cond,ex,win))
# temperature control for Kyushu (Fukuoka hourly 2m temperature, daily mean; anomaly vs 2016-2026 calendar-month mean and sd)
T=[]
for f in sorted(glob.glob(os.path.join(HERE,'..','kyushu','data','era5','fukuoka_*.json'))):
    j=json.load(open(f))['hourly']; T.append(pd.Series(j['temperature_2m'],index=pd.to_datetime(j['time'])))
T=pd.concat(T); T=T[~T.index.duplicated()].resample('D').mean(); c=T.groupby(T.index.month).agg(['mean','std']); tz=(T-T.index.month.map(c['mean']).values)/T.index.month.map(c['std']).values
df=frame('kyushu'); df['tz']=tz.reindex(df.index); df=df.dropna(); fm=(df.index>=FORM[0])&(df.index<=FORM[1]); r90=df.r[fm].quantile(.9); ex=df.r>=r90; test=(df.index>=TEST[0])&(df.index<=TEST[1])
zfull=daily_z(ERA['kyushu'])[0]; zb=zfull[(zfull.index.year>=1996)&(zfull.index.year<=2015)]
print('\nKyushu with temperature near normal (|temp z| < 0.75) vs all days:')
for q in (0.20,0.10,0.05):
    th=zb.quantile(q); cond=df.z<=th
    print('q=%.2f  all days (test):'%q,lift(df,cond,ex,test),'| temperature near normal (test):',lift(df,cond&(df.tz.abs()<0.75),ex,test))
