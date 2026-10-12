"""How much of next month's irradiance anomaly is knowable on the 20th of the previous month from irradiance alone (persistence of ERA5 anomalies)? The oracle test assumes it is all of it.
x = mean daily anomaly of days 1-19 of month M-1 (z against 1996-2015 same-calendar-month daily mean/sd); y = monthly anomaly of M in monthly sd (as in oracle_money.py).
Fit on 1996-2015 pairs only; applied to 2017-2026."""
import json, os, numpy as np, pandas as pd
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
def era(fn):
    j=json.load(open(os.path.join(D,fn)))['daily']; return pd.Series(j['shortwave_radiation_sum'],index=pd.to_datetime(j['time'])).astype(float)
for area,fn in (('tokyo','era5_tokyo.json'),('kyushu','era5_fukuoka.json')):
    S=era(fn); m=S.resample('MS').mean(); b=m[(m.index.year>=1996)&(m.index.year<=2015)]; c=b.groupby(b.index.month).agg(['mean','std']); zm=(m-m.index.month.map(c['mean']).values)/m.index.month.map(c['std']).values
    bd=S[(S.index.year>=1996)&(S.index.year<=2015)]; cd=bd.groupby(bd.index.month).agg(['mean','std']); zd=(S-S.index.month.map(cd['mean']).values)/S.index.month.map(cd['std']).values
    rows=[]
    for M in pd.date_range('1996-02-01','2026-09-01',freq='MS'):
        prev=M-pd.DateOffset(months=1); x=zd[prev:prev+pd.Timedelta(days=18)].mean(); rows.append((M,x,zm[M]))
    P=pd.DataFrame(rows,columns=['M','x','y']).set_index('M'); fit=P[P.index.year<=2015]; k=np.polyfit(fit.x,fit.y,1); te=P[(P.index.year>=2017)]; te['hat']=k[0]*te.x+k[1]
    lo,hi=zm[(zm.index.year>=1996)&(zm.index.year<=2015)].quantile([.10,.90])
    print(area,'fit on 1996-2015: slope %.2f, R2 %.3f | 2017-2026: corr(forecast, actual) %.2f, R2 %.3f'%(k[0],np.corrcoef(fit.x,fit.y)[0,1]**2,te.hat.corr(te.y),te.hat.corr(te.y)**2))
    print('   months where the forecast reaches a dull/bright extreme (cuts %.2f/%.2f): %d of %d, while actual extremes occurred in %d'%(lo,hi,int(((te.hat<=lo)|(te.hat>=hi)).sum()),len(te),int(((te.y<=lo)|(te.y>=hi)).sum())))
    s=P.loc['2026-09-01']; print('   Sep 2026: forecast on 20 Aug = %.2f sd (actual %.2f sd); 1-19 Aug anomaly x = %.2f'%(k[0]*s.x+k[1],s.y,s.x))
