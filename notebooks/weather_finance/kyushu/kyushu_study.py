"""Kyushu midday price discount and irradiance: JEPX day-ahead spot (half-hourly, FY2016-FY2026) against ERA5 irradiance at five Kyushu points.

Plan fixed before looking at results:
  Window: JEPX slots 21-30 (10:00-15:00 JST), the solar-heavy middle of the day.  Prices: Kyushu area, system, Chugoku (neighbour, to flag market splitting).
  Irradiance: ERA5 shortwave radiation, mean of five points (Fukuoka, Kumamoto, Oita, Miyazaki, Kagoshima), 10:00-15:00; anomaly z against the calendar-month mean and sd of the FORMATION years (2016-04 to 2022-03); clearness index kt = GHI / extraterrestrial.
  A. Oracle relation (explanatory): daily midday spread (Kyushu - system) on irradiance anomaly, controls: calendar month x year fixed effects, day of week, Japanese holiday, temperature, temperature squared.
  B. Rarity test (chronological): formation 2016-04..2022-03, test 2022-04..2026-09, thresholds frozen on formation. Events: near-zero Kyushu midday price (<= 1 yen/kWh) and deep discount (Kyushu - system <= -3 yen).
     Conditioning: bright-day tails (top 20%, 10%, 5%, 2% of the formation anomaly distribution). Scored as P(event | tail) against the test base rate and against a base matched to month and day type.
  C. Knowability: persistence of the anomaly from earlier days (lag 1, 3, 7, 14, 28) and months.
"""
import glob, io, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, jpholiday
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,'data'); FORM_END=pd.Timestamp('2022-03-31'); OUT=HERE
LAT,LON=32.5,130.8
# ---------------- ERA5
frames={}
for f in sorted(glob.glob(os.path.join(D,'era5','*.json'))):
    nm=os.path.basename(f).split('_')[0]; h=json.load(open(f))['hourly']
    df=pd.DataFrame({'ghi':h['shortwave_radiation'],'t':h['temperature_2m']},index=pd.to_datetime(h['time'])); frames.setdefault(nm,[]).append(df)
pts=[pd.concat(v).sort_index() for v in frames.values()]; H=sum(p for p in pts)/len(pts)           # equal-weight mean of the points
doy=H.index.dayofyear.values; hour=H.index.hour.values+H.index.minute.values/60
# extraterrestrial horizontal irradiance at the middle of each hour (the ERA5 value is the mean of the preceding hour)
dec=np.radians(23.45)*np.sin(2*np.pi*(284+doy)/365); E0=1+0.033*np.cos(2*np.pi*doy/365); st=(hour-0.5)+(LON-135)/15; om=np.radians(15*(st-12))
cz=np.sin(np.radians(LAT))*np.sin(dec)+np.cos(np.radians(LAT))*np.cos(dec)*np.cos(om); H['et']=np.maximum(1367*E0*cz,0)
mid=H[(H.index.hour>=11)&(H.index.hour<=15)]       # hour labels 11..15 = 10:00-15:00
W=pd.DataFrame({'ghi':mid.groupby(mid.index.date).ghi.mean(),'et':mid.groupby(mid.index.date).et.mean()}); W.index=pd.to_datetime(W.index); W['kt']=W.ghi/W.et
W['temp']=H.groupby(H.index.date).t.mean().rename(lambda d: pd.Timestamp(d)).reindex(W.index)
# ---------------- JEPX
fr=[]
for f in sorted(glob.glob(os.path.join(D,'spot_*.csv'))):
    df=pd.read_csv(io.StringIO(open(f,encoding='utf-8').read())); df.columns=['date','slot','sell','buy','vol','system','hokkaido','tohoku','tokyo','chubu','hokuriku','kansai','chugoku','shikoku','kyushu','a','b','c','d']; fr.append(df[['date','slot','system','chugoku','kyushu']])
J=pd.concat(fr); J['date']=pd.to_datetime(J['date'],format='%Y/%m/%d'); J=J.drop_duplicates(['date','slot']); m=J[(J.slot>=21)&(J.slot<=30)]
P=pd.DataFrame({'sys':m.groupby('date').system.mean(),'chu':m.groupby('date').chugoku.mean(),'kyu':m.groupby('date').kyushu.mean(),'n':m.groupby('date').slot.count()}); P=P[P.n==10].drop(columns='n')
P['spread']=P.kyu-P.sys; P['split']=P.kyu-P.chu
X=P.join(W,how='inner').dropna(); X=X[X.index<='2026-09-30'].copy()
X['hol']=[int(d.weekday()>=5 or jpholiday.is_holiday(d.date())) for d in X.index]; X['dow']=X.index.dayofweek; X['month']=X.index.month; X['year']=X.index.year
X['fy']=np.where(X.index.month>=4,X.index.year,X.index.year-1)
print(f'days {len(X)} | {X.index.min().date()} to {X.index.max().date()} | formation days {(X.index<=FORM_END).sum()} | test days {(X.index>FORM_END).sum()}')
# anomaly z against FORMATION calendar-month climatology
fm=X[X.index<=FORM_END]; clim=fm.groupby('month').agg(gm=('ghi','mean'),gs=('ghi','std'),km=('kt','mean'),ks=('kt','std'))
for c,a,b in (('z','gm','gs'),('zk','km','ks')): X[c]=((X.ghi if c=='z' else X.kt)-X.month.map(clim[a]))/X.month.map(clim[b])
X['near0']=(X.kyu<=1.0).astype(int); X['deep']=(X.spread<=-3.0).astype(int)
X.to_parquet(os.path.join(D,'kyushu_daily.parquet'))
# ---------------- A. oracle relation
def ols(y,xcols,ctrl,fe,data=None):
    """Slope(s) on xcols with controls ctrl (numeric) and fixed effects fe; partial R2 = 1 - SSR(with x) / SSR(controls only)."""
    d=(X if data is None else data)[[y]+xcols+ctrl+fe].dropna(); C=[np.ones(len(d))]+[d[c].values.astype(float) for c in ctrl]
    for f in fe: C.append(pd.get_dummies(d[f].astype(str),drop_first=True).values.astype(float))
    C=np.column_stack(C); F=np.column_stack([d[xcols].values.astype(float),C]); yv=d[y].values
    bF=np.linalg.lstsq(F,yv,rcond=None)[0]; bC=np.linalg.lstsq(C,yv,rcond=None)[0]
    return bF[:len(xcols)],1-((yv-F@bF)**2).sum()/((yv-C@bC)**2).sum(),len(d)
X['ym']=X.year.astype(str)+'-'+X.month.astype(str); X['temp2']=X.temp**2
print('\nA. midday spread (Kyushu - system, yen/kWh) on irradiance anomaly z (per sd):')
for lab,sub in (('all days',X),('weekdays',X[X.hol==0]),('weekends/holidays',X[X.hol==1])):
    b,pr,n=ols('spread',['z'],[],[],sub); b2,pr2,n2=ols('spread',['z'],['temp','temp2'],['ym','dow'],sub)
    print(f'  {lab:18s} n {n:5d} | raw slope {b[0]:+.2f} | with month-year, weekday, temperature controls: slope {b2[0]:+.2f}, partial R2 of irradiance {pr2:.3f}')
print('  by season (controls on):')
seas={'spring Mar-May':[3,4,5],'summer Jun-Aug':[6,7,8],'autumn Sep-Nov':[9,10,11],'winter Dec-Feb':[12,1,2]}
SS={}
for k,ms in seas.items():
    b2,pr2,n2=ols('spread',['z'],['temp','temp2','hol'],['ym','dow'],X[X.month.isin(ms)]); SS[k]=(b2[0],pr2,n2); print(f'    {k:16s} n {n2:5d} slope {b2[0]:+.2f} yen/kWh per sd, partial R2 {pr2:.3f}')
# out-of-sample: fit on formation, test on test (within-year-month demeaned spread removes capacity and fuel regime)
Xa=X.copy(); Xa['sp_adj']=Xa.spread-Xa.groupby('ym').spread.transform('mean'); Xa['z_adj']=Xa.z-Xa.groupby('ym').z.transform('mean')
fo,te=Xa[Xa.index<=FORM_END],Xa[Xa.index>FORM_END]; Z=lambda d: np.column_stack([d.z_adj,d.hol-d.groupby('ym').hol.transform('mean')])
bf=np.linalg.lstsq(np.column_stack([Z(fo),np.ones(len(fo))]),fo.sp_adj,rcond=None)[0]; pred=np.column_stack([Z(te),np.ones(len(te))])@bf
r2=1-((te.sp_adj-pred)**2).sum()/((te.sp_adj-te.sp_adj.mean())**2).sum(); print(f'\n  out-of-sample: within-month day-to-day spread variance explained by irradiance + day type, fitted 2016-2021, tested 2022-2026: R2 {r2:.3f} (irradiance slope {bf[0]:+.2f}, holiday slope {bf[1]:+.2f})')
# ---------------- B. rarity test
fo,te=X[X.index<=FORM_END],X[X.index>FORM_END]; rows=[]
for ev,lab in (('near0','price <= 1 yen'),('deep','spread <= -3 yen')):
    for tail in (0.20,0.10,0.05,0.02):
        thr=fo.z.quantile(1-tail)
        def stat(df):
            sel=df[df.z>=thr]; 
            if len(sel)<10: return len(sel),np.nan,np.nan,np.nan
            p=sel[ev].mean(); base=df[ev].mean()
            st=df.groupby(['month','hol'])[ev].mean(); w=sel.groupby(['month','hol']).size()/len(sel); bm=float((w*st.reindex(w.index).fillna(0)).sum()); return len(sel),p,base,bm
        nf,pf,bf_,bmf=stat(fo); nt,pt,bt,bmt=stat(te)
        rows.append((lab,tail,nf,pf,bf_,bmf,nt,pt,bt,bmt))
B=pd.DataFrame(rows,columns=['event','tail','n_form','p_form','base_form','matched_form','n_test','p_test','base_test','matched_test']); B['lift_test']=B.p_test/B.base_test; B['lift_matched_test']=B.p_test/B.matched_test
pd.set_option('display.width',220); print('\nB. P(event | bright-day tail), thresholds frozen on 2016-2021:'); print(B.round(3).to_string(index=False))
print('  test-period base rates: near-zero price',round(te.near0.mean(),3),'| deep discount',round(te.deep.mean(),3),' | formation:',round(fo.near0.mean(),3),round(fo.deep.mean(),3))
# ---------------- C. knowability
print('\nC. persistence of the daily irradiance anomaly z (correlation of the day with the mean of the 7 days ending k days earlier):')
zs=X.z.asfreq('D'); pers={}
for k in (1,3,7,14,28):
    prev=zs.shift(k).rolling(7,min_periods=5).mean(); d=pd.concat([zs,prev],axis=1).dropna(); pers[k]=np.corrcoef(d.iloc[:,0],d.iloc[:,1])[0,1]; print(f'  lag {k:2d} d: corr {pers[k]:+.2f}')
mz=X.z.resample('MS').mean(); mp=pd.concat([mz,mz.shift(1)],axis=1).dropna(); print(f'  monthly mean anomaly vs previous month: corr {np.corrcoef(mp.iloc[:,0],mp.iloc[:,1])[0,1]:+.2f}')
# ---------------- figures
fig,ax=plt.subplots(1,3,figsize=(15,4.6)); GREEN,BLUE,AMBER,GREY,RED="#2e7d4f","#2f5f8f","#c98a1b","#8a94a3","#c0392b"
for lab,sub,col in (('weekdays',X[X.hol==0],BLUE),('weekends and holidays',X[X.hol==1],AMBER)):
    q=pd.qcut(sub.z,12,duplicates='drop'); g=sub.groupby(q,observed=True).agg(z=('z','mean'),s=('spread','mean')); ax[0].plot(g.z,g.s,'o-',color=col,lw=1.8,label=lab)
ax[0].axhline(0,color='k',lw=0.7); ax[0].set_xlabel('Irradiance anomaly, Kyushu midday (sd)'); ax[0].set_ylabel('Kyushu minus system midday price (yen/kWh)'); ax[0].set_title('Binned means, 2016-2026',fontsize=11); ax[0].legend(fontsize=9,frameon=False)
b0=B[B.event=='price <= 1 yen']; x=np.arange(len(b0)); w=0.38
ax[1].bar(x-w/2,b0.p_test,w,color=GREEN,label='P(event | bright tail)'); ax[1].bar(x+w/2,b0.matched_test,w,color=GREY,label='base matched to month and day type')
for i,(a,b_,n) in enumerate(zip(b0.p_test,b0.matched_test,b0.n_test)): ax[1].text(i-w/2,a+0.01,f'n={n}',ha='center',fontsize=8.5)
ax[1].set_xticks(x); ax[1].set_xticklabels([f'top {int(t*100)}%' for t in b0['tail']]); ax[1].set_ylabel('Share of days with Kyushu midday price <= 1 yen'); ax[1].set_title('Rarity test, 2022-2026 (thresholds frozen on 2016-2021)',fontsize=11); ax[1].set_ylim(0,0.62); ax[1].legend(fontsize=8.5,frameon=False,loc='upper center',ncol=1)
ax[2].plot(list(pers),list(pers.values()),'o-',color=BLUE,lw=2); ax[2].axhline(0,color='k',lw=0.7); ax[2].set_xlabel('Days before (mean of the 7 days ending k days earlier)'); ax[2].set_ylabel('Correlation with the day\'s anomaly'); ax[2].set_title('How much of the day was knowable from persistence',fontsize=11)
for a in ax:
    for s in ('top','right'): a.spines[s].set_visible(False)
fig.suptitle('Kyushu: irradiance and the midday price discount',fontsize=13); fig.tight_layout(); fig.savefig(os.path.join(OUT,'kyushu_figure.png'),dpi=160,bbox_inches='tight')
B.to_csv(os.path.join(OUT,'kyushu_rarity_test.csv'),index=False); print('\nwrote kyushu_figure.png, kyushu_rarity_test.csv')
