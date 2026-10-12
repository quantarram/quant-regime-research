"""Causal September forecast test for the Tokyo-area JEPX day-ahead price (the question asked by A. Stace on the Paper 17 post):
"Standing back on 20 August, what would your model have said would be the JEPX outcome for September in the Tokyo area?"

Fixed before results:
  For every year Y = 2017..2026 (JEPX data start in April 2016) the forecast is made on 20 August using only data through 19 August of Y (JEPX prices, observed irradiance) and data of earlier years.
  Target: September mean of all half-hourly Tokyo-area prices (yen/kWh); second target: the Tokyo-minus-system premium.
  M0  no change: the Tokyo mean price of 1-19 August of Y.
  M1  seasonal ratio: M0 times exp(mean of ln(September mean / 1-19 August mean) over earlier years).
  M2  M1 + irradiance channel: sensitivity of the daily Tokyo-minus-system premium to the daily irradiance anomaly (estimated on days before 20 August of Y)
      times the expected September anomaly given 1-19 August (persistence coefficient estimated on 1996..Y-1).
  M3  oracle: as M2 but with the REALISED September anomaly (the ceiling for any irradiance forecast, using the same sensitivity).
  M4  last September; M5  mean of earlier Septembers.
Irradiance: ERA5 daily shortwave sum at Tokyo, anomaly against the calendar-month mean and sd of 1996..Y-1.
"""
import glob, io, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, jpholiday
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,'data')
fr=[]
for f in sorted(glob.glob(os.path.join(D,'spot_*.csv'))):
    df=pd.read_csv(io.StringIO(open(f,encoding='utf-8').read())); df.columns=['date','slot','sell','buy','vol','system','hokkaido','tohoku','tokyo','chubu','hokuriku','kansai','chugoku','shikoku','kyushu','a','b','c','d']; fr.append(df[['date','slot','system','tokyo']])
J=pd.concat(fr); J['date']=pd.to_datetime(J['date'],format='%Y/%m/%d'); J=J.drop_duplicates(['date','slot'])
day=J.groupby('date').agg(tokyo=('tokyo','mean'),system=('system','mean'),n=('slot','count')); day=day[day.n==48].drop(columns='n'); day['prem']=day.tokyo-day.system
d=json.load(open(os.path.join(D,'era5_tokyo.json')))['daily']; S=pd.Series(d['shortwave_radiation_sum'],index=pd.to_datetime(d['time'])).astype(float); S=S[S.index<='2026-09-30']
def zscore(Y):
    base=S[(S.index.year>=1996)&(S.index.year<=Y-1)]; c=base.groupby(base.index.month).agg(['mean','std']); return (S-S.index.month.map(c['mean']).values)/S.index.month.map(c['std']).values
def ols(y,x,extra=None):
    Z=[np.ones(len(x)),x]+([] if extra is None else [extra[:,i] for i in range(extra.shape[1])]); Z=np.column_stack(Z); return np.linalg.lstsq(Z,y,rcond=None)[0]
rows=[]
for Y in range(2017,2027):
    z=zscore(Y); cut=pd.Timestamp(f'{Y}-08-19')
    # price information set
    pr=day[day.index<=cut]; aug=lambda y: day[f'{y}-08-01':f'{y}-08-19']; sep=lambda y: day[f'{y}-09-01':f'{y}-09-30']
    m0=aug(Y).tokyo.mean(); prior=range(2016,Y); lr=[np.log(sep(y).tokyo.mean()/aug(y).tokyo.mean()) for y in prior]; m1=m0*np.exp(np.mean(lr))
    # irradiance persistence (1996..Y-1) and expected September anomaly
    pa=[(z[f'{y}-08-01':f'{y}-08-19'].mean(),z[f'{y}-09-01':f'{y}-09-30'].mean()) for y in range(1996,Y)]; pa=np.array(pa); bp=ols(pa[:,1],pa[:,0])[1]; za=z[f'{Y}-08-01':f'{Y}-08-19'].mean(); ez=bp*za; zs=z[f'{Y}-09-01':f'{Y}-09-30'].mean()
    # sensitivity of the daily Tokyo-minus-system premium to the daily anomaly, days before 20 Aug of Y
    t=pr.join(z.rename('z'),how='inner'); hol=np.array([int(x.weekday()>=5 or jpholiday.is_holiday(x.date())) for x in t.index]); mon=pd.get_dummies(t.index.month.astype(str),drop_first=True).values.astype(float)
    b=ols(t.prem.values,t.z.values,np.column_stack([hol,mon])); s=b[1]
    m2=m1+s*ez; m3=m1+s*zs; actual=sep(Y).tokyo.mean(); m4=sep(Y-1).tokyo.mean(); m5=np.mean([sep(y).tokyo.mean() for y in prior])
    # premium target
    p0=aug(Y).prem.mean(); pd_=[sep(y).prem.mean()-aug(y).prem.mean() for y in prior]; p1=p0+np.mean(pd_); pa_=sep(Y).prem.mean()
    rows.append(dict(year=Y,actual=actual,M0=m0,M1=m1,M2=m2,M3=m3,M4=m4,M5=m5,sens=s,persist=bp,z_aug=za,z_exp=ez,z_sep=zs,prem_actual=pa_,P0=p0,P1=p1,P2=p1+s*ez,P3=p1+s*zs))
R=pd.DataFrame(rows).set_index('year')
pd.set_option('display.width',230); print('Tokyo-area September mean price, yen/kWh (forecast made on 20 August from data through 19 August):'); print(R[['actual','M0','M1','M2','M3','M4','M5']].round(2).to_string())
print('\nIrradiance channel: anomaly in 1-19 Aug, expected September anomaly (persistence), realised September anomaly, estimated sensitivity (yen/kWh per sd of daily anomaly)'); print(R[['z_aug','z_exp','z_sep','sens','persist']].round(2).to_string())
def met(c,t='actual'):
    e=R[c]-R[t]; return pd.Series(dict(MAE=e.abs().mean(),RMSE=np.sqrt((e**2).mean()),MAPE_pct=100*(e.abs()/R[t]).mean(),bias=e.mean()))
print('\nErrors over 2017-2026 (yen/kWh):'); print(pd.DataFrame({m:met(m) for m in ('M0','M1','M2','M3','M4','M5')}).T.round(2).to_string())
print('\nExcluding the 2022 energy-crisis year:'); R2=R.drop(index=2022); print(pd.DataFrame({m:pd.Series(dict(MAE=(R2[m]-R2.actual).abs().mean(),RMSE=np.sqrt(((R2[m]-R2.actual)**2).mean()))) for m in ('M0','M1','M2','M3','M4','M5')}).T.round(2).to_string())
print('\nPremium over system (Tokyo - system, yen/kWh), September mean:'); print(R[['prem_actual','P0','P1','P2','P3']].round(2).to_string())
print('\nSkill of the irradiance channel: corr(expected September anomaly, realised) over 2017-2026 = %+.2f ; corr(realised anomaly, actual Tokyo premium) = %+.2f'%(np.corrcoef(R.z_exp,R.z_sep)[0,1],np.corrcoef(R.z_sep,R.prem_actual)[0,1]))
y=2026; r=R.loc[y]; lr=[np.log(R.loc[k,'actual']/R.loc[k,'M0']) for k in R.index if k<y]; print(f'\n=== 20 August 2026 ===\n  Aug 1-19 mean Tokyo price: {r.M0:.2f}; M1 seasonal-ratio forecast for September: {r.M1:.2f}; M2 with irradiance channel: {r.M2:.2f} (expected anomaly {r.z_exp:+.2f} sd from a 1-19 Aug anomaly of {r.z_aug:+.2f} sd; sensitivity {r.sens:+.2f} yen per sd)\n  Actual September mean: {r.actual:.2f}  -> M1 error {r.M1-r.actual:+.2f} ({100*(r.M1/r.actual-1):+.0f}%), M2 error {r.M2-r.actual:+.2f}; oracle with the true record-low irradiance (M3): {r.M3:.2f}\n  September 2026 irradiance anomaly: {r.z_sep:+.2f} sd; Tokyo premium over system: actual {r.prem_actual:+.2f}, M1-style forecast {r.P1:+.2f}, oracle-irradiance {r.P3:+.2f}')
R.to_csv(os.path.join(HERE,'sept_forecast_results.csv'))
fig,ax=plt.subplots(1,2,figsize=(13,4.6)); GREEN,BLUE,AMBER,GREY,RED="#2e7d4f","#2f5f8f","#c98a1b","#8a94a3","#c0392b"
ax[0].plot(R.index,R.actual,'o-',color='k',lw=2,label='Actual September mean'); ax[0].plot(R.index,R.M1,'s--',color=BLUE,label='M1 seasonal ratio (20 Aug)'); ax[0].plot(R.index,R.M2,'^--',color=GREEN,label='M2 + irradiance channel'); ax[0].plot(R.index,R.M3,'v:',color=AMBER,label='M3 oracle irradiance')
ax[0].set_ylabel('Tokyo-area JEPX price (yen/kWh)'); ax[0].set_title('Forecast on 20 August for the September mean',fontsize=11); ax[0].legend(fontsize=8.5,frameon=False)
ax[1].scatter(R.z_sep,R.prem_actual,color=BLUE,s=40)
for yr,rr in R.iterrows(): ax[1].annotate(str(yr),(rr.z_sep,rr.prem_actual),fontsize=8,xytext=(4,3),textcoords='offset points')
ax[1].axhline(0,color='k',lw=0.7); ax[1].set_xlabel('September irradiance anomaly at Tokyo (mean of daily z; -0.66 = -2.6 monthly sd)'); ax[1].set_ylabel('September Tokyo minus system price (yen/kWh)'); ax[1].set_title('Realised, not forecastable on 20 August',fontsize=11)
for a in ax:
    for s_ in ('top','right'): a.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE,'sept_forecast_figure.png'),dpi=160,bbox_inches='tight'); print('\nwrote sept_forecast_figure.png, sept_forecast_results.csv')
