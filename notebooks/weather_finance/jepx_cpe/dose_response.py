"""Does the MAGNITUDE of dullness matter? Dose-response of the midday JEPX price to irradiance, on every day.
Daily irradiance anomaly z: ERA5 daily sum against the calendar-month mean and sd of 1996-2015 (Tokyo irradiance for Tokyo and system; Fukuoka for Kyushu).
Price response: midday (slots 21-30, 10:00-15:00 JST) mean price of the day minus the mean midday price of the other non-holiday weekdays within +/-15 days (absorbs fuel, season and level shifts; weekends and holidays excluded).
Also the 7-day version: mean z over 7 days against the 7-day mean price response, to capture sustained dullness.
Bins of z; mean response with its naive standard error, number of days, and number of distinct ISO weeks (days within a spell are not independent)."""
import glob, io, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, jpholiday
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,'data')
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
    b=S[(S.index.year>=1996)&(S.index.year<=2015)]; c=b.groupby(b.index.month).agg(['mean','std']); return ((S-S.index.month.map(c['mean']).values)/S.index.month.map(c['std']).values)
BINS=[-9,-2,-1.5,-1,-0.5,0.5,1,1.5,9]; LAB=['<-2','-2..-1.5','-1.5..-1','-1..-0.5','-0.5..0.5','0.5..1','1..1.5','>1.5']
res={}
for area in ['tokyo','kyushu','system']:
    p=g[area].where(work); z=daily_z(ERA[area]).reindex(g.index)
    # baseline: mean of other working days within +/-15 days
    s=p.rolling(31,center=True,min_periods=1).sum()-p.fillna(0); c=p.rolling(31,center=True,min_periods=1).count()-p.notna().astype(int)
    base=(s/c).where(c>=8); y=(p-base)
    df=pd.DataFrame({'z':z,'y':y}).dropna(); df['bin']=pd.cut(df.z,BINS,labels=LAB); df['week']=df.index.to_period('W').astype(str); df['late']=df.index>='2021-01-01'
    # 7-day version
    p7=(y.rolling(7,min_periods=4).mean()); z7=z.rolling(7,min_periods=7).mean(); d7=pd.DataFrame({'z':z7,'y':p7}).dropna()
    res[area]=(df,d7)
    print('\n====',area,'| working days',len(df))
    t=df.groupby('bin').y.agg(['count','mean','std']); t['se']=t['std']/np.sqrt(t['count']); t['weeks']=df.groupby('bin').week.nunique(); print(t.round(2).to_string())
    for lab,m in (('2016-2020',~df.late),('2021-2026',df.late)):
        tt=df[m].groupby('bin').y.mean().round(2); print(lab,tt.to_dict())
    mild=df[(df.z>-1)&(df.z<1)]; k=np.polyfit(mild.z,mild.y,1); ext=df[df.z<-1.5]
    print('linear slope in the mild range |z|<1: %.2f yen/kWh per sd; predicted at the mean of z<-1.5 days (z=%.2f): %+.2f; observed %+.2f (n=%d, weeks=%d)'%(k[0],ext.z.mean(),k[0]*ext.z.mean()+k[1],ext.y.mean(),len(ext),ext.week.nunique()))
    q=d7.copy(); q['bin']=pd.cut(q.z,[-9,-1.5,-1,-0.5,0.5,1,1.5,9],labels=['<-1.5','-1.5..-1','-1..-0.5','-0.5..0.5','0.5..1','1..1.5','>1.5']); print('7-day windows:',q.groupby('bin').y.agg(['count','mean']).round(2).T.to_dict())
fig,ax=plt.subplots(1,2,figsize=(13,4.8))
for a,area in zip(ax,['tokyo','kyushu']):
    df=res[area][0]; t=df.groupby('bin').y.agg(['count','mean','std']); t['se']=t['std']/np.sqrt(t['count'])
    cols=['tab:red']*2+['salmon']*2+['lightgray']+['lightsteelblue']*1+['tab:blue']*2
    a.bar(range(len(t)),t['mean'],yerr=t.se,color=cols,capsize=3); a.axhline(0,color='k',lw=.6); a.set_xticks(range(len(t))); a.set_xticklabels(t.index,fontsize=8)
    for i,(m,n) in enumerate(zip(t['mean'],t['count'])): a.text(i,m+(0.25 if m>=0 else -0.5),f'n={n}',ha='center',fontsize=7)
    a.set_title(f'{area}: midday price vs surrounding weekdays, by daily irradiance anomaly',fontsize=9); a.set_xlabel('daily irradiance anomaly (sd; left = dull, right = bright)'); a.set_ylabel('price response (yen/kWh)')
plt.tight_layout(); plt.savefig(os.path.join(HERE,'dose_response_figure.png'),dpi=140); print('wrote dose_response_figure.png')
# --- season split and medians (dull days are also cold or wet days, so demand can contribute; spikes make means noisy)
SEAS={'Dec-Feb':[12,1,2],'Mar-May':[3,4,5],'Jun-Sep':[6,7,8,9],'Oct-Nov':[10,11]}
for area in ['tokyo','kyushu']:
    df=res[area][0]; df=df.assign(season=[next(k for k,v in SEAS.items() if m in v) for m in df.index.month])
    print('\n',area,'by season: [mean, median, n] of price response for dull bins vs normal vs bright')
    for sname in SEAS:
        d=df[df.season==sname]; out=[]
        for lab,msk in (('dull<-1.5',d.z<-1.5),('mild dull -1.5..-0.5',(d.z>=-1.5)&(d.z<-0.5)),('normal',(d.z>=-0.5)&(d.z<=0.5)),('bright>0.5',d.z>0.5)):
            s=d[msk].y; out.append(f'{lab}: [{s.mean():+.1f}, {s.median():+.1f}, n={len(s)}]')
        print(' ',sname,' | '.join(out))
