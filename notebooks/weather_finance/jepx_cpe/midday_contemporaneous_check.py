"""Why the midday CPE scan shows the opposite sign: the 'move' target compares the next f days with the previous f days, and with p = f the previous days ARE the conditioning (dull or bright) days, so a real same-week weather effect reverses next week.
Correlations of the 7-day midday (slots 21-30) log price with the 7-day irradiance anomaly (z against 1996-2015)."""
import glob,io,json,os,warnings; warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data'); fr=[]
for f in sorted(glob.glob(os.path.join(D,'spot_*.csv'))):
    df=pd.read_csv(io.StringIO(open(f,encoding='utf-8').read())); df.columns=['date','slot','sell','buy','vol','system','hokkaido','tohoku','tokyo','chubu','hokuriku','kansai','chugoku','shikoku','kyushu','a','b','c','d']; fr.append(df[['date','slot','system','tokyo','kyushu']])
J=pd.concat(fr); J['date']=pd.to_datetime(J['date'],format='%Y/%m/%d'); J=J.drop_duplicates(['date','slot']); J=J[(J.slot>=21)&(J.slot<=30)]
day=J.groupby('date').agg(system=('system','mean'),tokyo=('tokyo','mean'),kyushu=('kyushu','mean'),n=('slot','count')); day=day[day.n==10].drop(columns='n'); day=day.reindex(pd.date_range('2016-04-01','2026-09-30'))
def era(fn):
    d=json.load(open(os.path.join(D,fn)))['daily']; return pd.Series(d['shortwave_radiation_sum'],index=pd.to_datetime(d['time'])).astype(float)
for area,fn in (('tokyo','era5_tokyo.json'),('kyushu','era5_fukuoka.json')):
    S=era(fn); b=S[(S.index.year>=1996)&(S.index.year<=2015)]; c=b.groupby(b.index.month).agg(['mean','std']); z=((S-S.index.month.map(c['mean']).values)/S.index.month.map(c['std']).values).reindex(day.index)
    x=z.rolling(7).mean(); lp=np.log(day[area].clip(lower=0.01)); cur=lp.rolling(7).mean(); prev=cur.shift(7); nxt=cur.shift(-7)
    d=pd.DataFrame({'x':x,'a':cur-prev,'b':nxt-cur,'c':nxt-prev}).dropna()
    ym=lp.groupby([lp.index.year,lp.index.month]).transform('mean'); dev=(lp-ym).rolling(7).mean()
    print(area,'| same-week price vs irradiance %.2f | this week vs last week %.2f | next week vs this week %.2f | next week vs previous week %.2f'%(x.corr(dev),d.x.corr(d.a),d.x.corr(d.b),d.x.corr(d.c)))
