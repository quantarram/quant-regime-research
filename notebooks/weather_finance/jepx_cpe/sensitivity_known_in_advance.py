"""If the price sensitivity to sunlight is known in advance, can the price move be predicted in advance?
Two ingredients: (1) the sensitivity s (log-points of the midday/night price ratio per sd of daily irradiance anomaly), estimated ONLY from data up to 20 days before the decision date, on the previous 3 years (recent) or on 2016-2020 (fixed early era);
(2) the irradiance of the target month: known exactly (oracle ERA5, a ceiling) or not known (replaced by its climatological value, 0).
Target: for every month M from 2019-05 to 2026-09, decision on the 20th of the previous month t:
  realised  dR = mean daily ln(midday/night) over month M  -  mean over the 30 days to t ; surprise = dR minus the mean dR of the same calendar month in EARLIER years
  predicted dR_hat = s * ( zM - zB ), zM = mean daily anomaly of M (oracle) or 0 (no foresight), zB = mean daily anomaly over the 30 days to t;  surprise_hat = dR_hat minus the same-month earlier-year mean of dR_hat
Scores: correlation between surprise_hat and surprise over the months, and the mean realised surprise on months where the predicted surprise is in its top and bottom quarter."""
import glob, io, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,'data')
fr=[]
for f in sorted(glob.glob(os.path.join(D,'spot_*.csv'))):
    d=pd.read_csv(io.StringIO(open(f,encoding='utf-8').read())); d=d.iloc[:,[0,1,5,8,14]]; d.columns=['date','slot','system','tokyo','kyushu']; fr.append(d)
J=pd.concat(fr); J['date']=pd.to_datetime(J['date'],format='%Y/%m/%d'); J=J.drop_duplicates(['date','slot'])
idx=pd.date_range('2016-04-01','2026-09-30')
def daily(lo,hi):
    g=J[(J.slot>=lo)&(J.slot<=hi)].groupby('date').agg(tokyo=('tokyo','mean'),kyushu=('kyushu','mean'),n=('slot','count')); return g[g.n==hi-lo+1].drop(columns='n').reindex(idx)
MID,NIGHT=daily(21,30),daily(1,8)
def era(fn):
    j=json.load(open(os.path.join(D,fn)))['daily']; return pd.Series(j['shortwave_radiation_sum'],index=pd.to_datetime(j['time'])).astype(float)
ERA={'tokyo':era('era5_tokyo.json'),'kyushu':era('era5_fukuoka.json')}
def zdaily(S):
    b=S[(S.index.year>=1996)&(S.index.year<=2015)]; c=b.groupby(b.index.month).agg(['mean','std']); return (S-S.index.month.map(c['mean']).values)/S.index.month.map(c['std']).values
out={}
for area in ['tokyo','kyushu']:
    r=np.log(MID[area].clip(lower=1.0).where(MID[area].notna())/NIGHT[area].clip(lower=1.0).where(NIGHT[area].notna())); z=zdaily(ERA[area]).reindex(idx)
    ym=r.groupby([r.index.year,r.index.month]).transform('mean'); dev=(r-ym)      # within-month deviation: removes level, fuel, slow drift
    zdev=z-z.groupby([z.index.year,z.index.month]).transform('mean')
    def slope(a,b):
        d=pd.DataFrame({'x':zdev[a:b],'y':dev[a:b]}).dropna(); return np.polyfit(d.x,d.y,1)[0] if len(d)>200 else np.nan
    s_early=slope('2016-04-01','2020-12-31')
    rows=[]
    for M in pd.date_range('2019-05-01','2026-09-01',freq='MS'):
        t=(M-pd.DateOffset(months=1))+pd.Timedelta(days=19); cut=t-pd.Timedelta(days=20)
        s_recent=slope(cut-pd.DateOffset(years=3),cut)
        B=r[t-pd.Timedelta(days=29):t].mean(); Om=r[M:M+pd.offsets.MonthEnd(0)]; zB=z[t-pd.Timedelta(days=29):t].mean(); zM=z[M:M+pd.offsets.MonthEnd(0)].mean()
        if Om.notna().sum()<25 or np.isnan(B): continue
        rows.append(dict(M=M,dR=Om.mean()-B,zB=zB,zM=zM,s_recent=s_recent,s_early=s_early))
    df=pd.DataFrame(rows); df['cm']=df.M.dt.month
    for name,sc,zm in (('recent sensitivity, irradiance KNOWN','s_recent','zM'),('early-era sensitivity, irradiance KNOWN','s_early','zM'),('recent sensitivity, irradiance NOT known (climatology)','s_recent',None)):
        zz=df.zM if zm else 0.0; df['pred']=df[sc]*(zz-df.zB)
        sur=[];sh=[]
        for i,rw in df.iterrows():
            pr=df[(df.cm==rw.cm)&(df.M<rw.M)]; sur.append(rw.dR-pr.dR.mean() if len(pr)>=2 else np.nan); sh.append(rw.pred-pr.pred.mean() if len(pr)>=2 else np.nan)
        d=df.assign(sur=sur,sur_hat=sh).dropna(subset=['sur','sur_hat']); hi=d.sur_hat>=d.sur_hat.quantile(.75); lo=d.sur_hat<=d.sur_hat.quantile(.25)
        out[(area,name)]=(len(d),d.sur_hat.corr(d.sur),d.sur_hat.corr(d.sur,method='spearman'),d.sur[hi].mean(),d.sur[lo].mean(),d.s_recent.mean() if sc=='s_recent' else d.s_early.mean())
print('Predicting the monthly change in the midday/night log ratio (surprise vs same-month earlier years), 2019-05..2026-09')
for (area,name),v in out.items(): print(f'{area:7s} {name:55s} months {v[0]}  corr {v[1]:+.2f}  rank-corr {v[2]:+.2f}  realised surprise: top-quarter prediction {v[3]:+.3f}, bottom-quarter {v[4]:+.3f}  (mean slope used {v[5]:+.3f})')
for area in ['tokyo','kyushu']:
    r=np.log(MID[area].clip(lower=1.0).where(MID[area].notna())/NIGHT[area].clip(lower=1.0).where(NIGHT[area].notna())); z=zdaily(ERA[area]).reindex(idx); ym=r.groupby([r.index.year,r.index.month]).transform('mean'); dev=(r-ym); zdev=z-z.groupby([z.index.year,z.index.month]).transform('mean')
    for lab,a,b in (('2016-2018','2016-04-01','2018-12-31'),('2019-2021','2019-01-01','2021-12-31'),('2022-2024','2022-01-01','2024-12-31'),('2024-2026','2024-01-01','2026-09-30')):
        d=pd.DataFrame({'x':zdev[a:b],'y':dev[a:b]}).dropna(); print(area,lab,'within-month sensitivity %.3f log-points per sd'%np.polyfit(d.x,d.y,1)[0])

# ---- is the skill without foresight just ordinary mean reversion of the ratio? compare with a naive reverter and test whether irradiance adds to it
print('\nCHECK: irradiance-based rebound (known at the decision date) against a naive mean-reversion predictor')
for area in ['tokyo','kyushu']:
    r=np.log(MID[area].clip(lower=1.0).where(MID[area].notna())/NIGHT[area].clip(lower=1.0).where(NIGHT[area].notna())); z=zdaily(ERA[area]).reindex(idx); ym=r.groupby([r.index.year,r.index.month]).transform('mean'); dev=(r-ym); zdev=z-z.groupby([z.index.year,z.index.month]).transform('mean')
    rows=[]
    for M in pd.date_range('2019-05-01','2026-09-01',freq='MS'):
        t=(M-pd.DateOffset(months=1))+pd.Timedelta(days=19); cut=t-pd.Timedelta(days=20); a,b=cut-pd.DateOffset(years=3),cut
        d=pd.DataFrame({'x':zdev[a:b],'y':dev[a:b]}).dropna(); s=np.polyfit(d.x,d.y,1)[0]
        B=r[t-pd.Timedelta(days=29):t].mean(); Om=r[M:M+pd.offsets.MonthEnd(0)]; zB=z[t-pd.Timedelta(days=29):t].mean(); longrun=r[t-pd.Timedelta(days=364):t].mean()
        if Om.notna().sum()<25 or np.isnan(B): continue
        rows.append(dict(M=M,cm=M.month,dR=Om.mean()-B,irr=s*(0-zB),naive=-(B-longrun)))
    df=pd.DataFrame(rows)
    for c in ['dR','irr','naive']: df[c+'_s']=[rw[c]-df[(df.cm==rw.cm)&(df.M<rw.M)][c].mean() if ((df.cm==rw.cm)&(df.M<rw.M)).sum()>=2 else np.nan for _,rw in df.iterrows()]
    d=df.dropna(subset=['dR_s','irr_s','naive_s']); X=np.column_stack([np.ones(len(d)),d.naive_s,d.irr_s]); beta=np.linalg.lstsq(X,d.dR_s,rcond=None)[0]; res_n=d.dR_s-np.polyval(np.polyfit(d.naive_s,d.dR_s,1),d.naive_s)
    # partial correlation of irradiance rebound with the surprise after removing what the naive reverter explains
    ri=d.irr_s-np.polyval(np.polyfit(d.naive_s,d.irr_s,1),d.naive_s)
    print(area,'months',len(d),'| corr(naive reverter, surprise) %+.2f | corr(irradiance rebound, surprise) %+.2f | corr between the two predictors %+.2f | partial corr of irradiance rebound after the naive reverter %+.2f'%(d.naive_s.corr(d.dR_s),d.irr_s.corr(d.dR_s),d.naive_s.corr(d.irr_s),ri.corr(res_n)))
