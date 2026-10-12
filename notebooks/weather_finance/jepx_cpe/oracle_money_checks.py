"""Independent re-computation and robustness checks of oracle_money.py (different code path: Period-based monthly grouping, no shared helpers)."""
import io, glob, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,'data')
raw=[]
for f in sorted(glob.glob(os.path.join(D,'spot_*.csv'))):
    d=pd.read_csv(io.StringIO(open(f,encoding='utf-8').read())); d=d.iloc[:,[0,1,5,8,14]]; d.columns=['date','slot','system','tokyo','kyushu']; raw.append(d)
raw=pd.concat(raw); raw['date']=pd.to_datetime(raw['date'],format='%Y/%m/%d'); raw=raw.drop_duplicates(['date','slot'])
print('JEPX rows',len(raw),'dates',raw.date.min().date(),raw.date.max().date(),'| duplicates dropped:',len(raw)-len(raw.drop_duplicates(['date','slot'])))
perday=raw.groupby('date').slot.count(); print('days with <48 slots:',int((perday<48).sum()),'of',len(perday))
def era(fn):
    j=json.load(open(os.path.join(D,fn)))['daily']; return pd.Series(j['shortwave_radiation_sum'],index=pd.to_datetime(j['time'])).astype(float)
ERA={'tokyo':era('era5_tokyo.json'),'kyushu':era('era5_fukuoka.json')}; ERA['system']=ERA['tokyo']
print('ERA5 NaN days:',{k:int(v.isna().sum()) for k,v in ERA.items()})
def sig(area):
    S=ERA[area].dropna(); m=S.groupby(S.index.to_period('M')).mean(); y=m.index.year; mo=m.index.month
    b=m[(y>=1996)&(y<=2015)]; mu=b.groupby(b.index.month).mean(); sd=b.groupby(b.index.month).std()
    z=(m-mu.reindex(mo).values)/sd.reindex(mo).values; zb=z[(y>=1996)&(y<=2015)]; return z,zb.quantile(.10),zb.quantile(.90)
def build(price,area,ref='first19',cut=(.10,.90)):
    g=raw[(raw.slot>=(21 if price=='midday' else 1))&(raw.slot<=(30 if price=='midday' else 48))]
    nslot=10 if price=='midday' else 48; c=g.groupby('date').slot.count(); keep=c[c==nslot].index; p=g[g.date.isin(keep)].groupby('date')[area].mean()
    z,lo,hi=sig(area); lo,hi=(z[(z.index.year>=1996)&(z.index.year<=2015)].quantile(cut[0]),z[(z.index.year>=1996)&(z.index.year<=2015)].quantile(cut[1]))
    rows=[]
    for M in pd.period_range('2017-05','2026-09',freq='M'):
        prev=M-1; pm=p[(p.index.to_period('M')==prev)]; om=p[(p.index.to_period('M')==M)]
        r=pm[pm.index.day<=19] if ref=='first19' else pm
        if len(r)<(15 if ref=='first19' else 25) or len(om)<25: continue
        rows.append(dict(M=M,R=r.mean(),O=om.mean(),z=z[M]))
    d=pd.DataFrame(rows); d['chg']=d.O-d.R; d['cm']=d.M.dt.month
    d['exp']=[d[(d.cm==r.cm)&(d.M<r.M)].chg.mean() if ((d.cm==r.cm)&(d.M<r.M)).sum()>=2 else np.nan for r in d.itertuples()]
    d['sur']=d.chg-d.exp; d=d.dropna(subset=['sur']); d['pos']=np.where(d.z<=lo,1,np.where(d.z>=hi,-1,0)); d['pnl']=d.pos*d.sur; return d,lo,hi
def stat(d,excl=True):
    d=d[d.M!=pd.Period('2021-02')] if excl else d; t=d[d.pos!=0]
    return dict(months=len(d),first=str(d.M.min()),pearson=round(d.z.corr(d.sur),2),spearman=round(d.z.corr(d.sur,method='spearman'),2),slope=round(np.polyfit(d.z,d.sur,1)[0],2),trades=len(t),dull=int((t.pos==1).sum()),bright=int((t.pos==-1).sum()),hit=round((t.pnl>0).mean(),2),mean=round(t.pnl.mean(),2),t=round(t.pnl.mean()/(t.pnl.std()/np.sqrt(len(t))),2) if len(t)>2 else np.nan)
print('\n--- A. independent recomputation (should match oracle_money.py: Tokyo midday 30 trades hit .50 mean -0.39; Kyushu midday 23 trades hit .61 mean +1.11)')
for pk in ['midday','day']:
    for ar in ['tokyo','kyushu','system']:
        d,lo,hi=build(pk,ar); print(pk,ar,'cuts %.2f %.2f'%(lo,hi),stat(d))
d,_,_=build('midday','tokyo'); print('\nSep 2026 Tokyo midday:',d[d.M==pd.Period('2026-09')][['R','O','exp','sur','z','pos','pnl']].round(2).to_dict('records'))
print('\n--- B. alternative reference: whole previous month (not only days 1-19)')
for ar in ['tokyo','kyushu']:
    d,_,_=build('midday',ar,ref='full'); print(ar,stat(d))
print('\n--- C. alternative cuts (15%/85% and 20%/80%)')
for cut in [(.15,.85),(.20,.80)]:
    for ar in ['tokyo','kyushu']:
        d,_,_=build('midday',ar,cut=cut); print(cut,ar,stat(d))
print('\n--- D. sub-periods (chronological halves of the sample)')
for ar in ['tokyo','kyushu']:
    d,_,_=build('midday',ar); d=d[d.M!=pd.Period('2021-02')]
    for lab,m in (('2019-05..2022-12',d.M<=pd.Period('2022-12')),('2023-01..2026-09',d.M>=pd.Period('2023-01'))):
        s=d[m]; t=s[s.pos!=0]; print(ar,lab,'months',len(s),'slope %.2f'%np.polyfit(s.z,s.sur,1)[0],'spearman %.2f'%s.z.corr(s.sur,method='spearman'),'trades',len(t),'hit %.2f'%(t.pnl>0).mean() if len(t) else '', 'mean %.2f'%t.pnl.mean() if len(t) else '')
print('\n--- E. relative (percent) version, so that price level does not matter: surprise = ln(O/R) minus same-month earlier-year mean (floor 0.5 yen)')
for ar in ['tokyo','kyushu','system']:
    d,lo,hi=build('midday',ar); d=d.copy(); d['lc']=np.log(d.O.clip(lower=.5)/d.R.clip(lower=.5)); d['lexp']=[d[(d.cm==r.cm)&(d.M<r.M)]['lc'].mean() if ((d.cm==r.cm)&(d.M<r.M)).sum()>=2 else np.nan for r in d.itertuples()]; d['ls']=d['lc']-d['lexp']; d=d.dropna(subset=['ls']); d=d[d.M!=pd.Period('2021-02')]; t=d[d.pos!=0]; pnl=t.pos*t.ls
    print(ar,'spearman %.2f'%d.z.corr(d.ls,method='spearman'),'slope %.3f log-pts/sd'%np.polyfit(d.z,d.ls,1)[0],'trades',len(t),'hit %.2f'%(pnl>0).mean(),'mean %.3f'%pnl.mean(),'t %.2f'%(pnl.mean()/(pnl.std()/np.sqrt(len(t)))))
print('\n--- F. contemporaneous check, no reference price at all: month-mean midday price deviation from its own calendar-month average across years (level, yen/kWh) vs irradiance anomaly')
for ar in ['tokyo','kyushu']:
    g=raw[(raw.slot>=21)&(raw.slot<=30)]; c=g.groupby('date').slot.count(); p=g[g.date.isin(c[c==10].index)].groupby('date')[ar].mean(); pm=p.groupby(p.index.to_period('M')).mean(); z,_,_=sig(ar)
    df=pd.DataFrame({'p':pm,'z':z}).dropna(); df=df[df.index>=pd.Period('2016-05')]; df=df[df.index.month.isin(range(1,13))]
    df['dev']=df.p-df.groupby(df.index.month).p.transform('mean'); df['yr']=df.index.year; df['dev2']=df.dev-df.groupby('yr').dev.transform('mean')
    print(ar,'corr(z, month price deviation) %.2f'%df.z.corr(df.dev),'| after removing each year\'s average (fuel/crisis years) %.2f'%df.z.corr(df.dev2),'| slope after year-removal %.2f yen/kWh per sd'%np.polyfit(df.z,df.dev2,1)[0],'n',len(df))
