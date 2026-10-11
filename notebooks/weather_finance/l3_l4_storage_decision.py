import json, os, sys
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
sys.path.insert(0,os.path.abspath(os.path.join(HERE,'..','..','..','climate-energy-dashboard')))
from weatherdash.calendar_utils import doy366
X=pd.read_csv(os.path.join(HERE,'data','storage_translation_weekly.csv'),index_col=0,parse_dates=True)   # weekly chg, norm, degree-day sums (week ending Friday)
nat=pd.read_csv(os.path.join(HERE,'data','national_power_daily.csv'),index_col=0,parse_dates=True)
l1=pd.read_pickle(os.path.join(HERE,'data','l1_oos.pkl'))
def wk(daily,F): 
    x=daily.loc[F-pd.Timedelta(days=6):F]; return x.sum() if len(x)==7 else np.nan
X['burn_obs']=[wk(nat.gas,F) for F in X.index]
# predicted burn from L1 (out-of-sample daily predictions, 2021-2025): weather+wind/solar model, and weather-only
for key,col in (('+ wind and solar generation','burn_ws'),('weather only (degree-days, gas price)','burn_w')):
    p=l1[key]['p']; X[col]=[wk(p,F) for F in X.index]
X=X.dropna(subset=['burn_obs','burn_ws','burn_w','H_pop','C_pop','H_gas','C_pow']).copy()
def design(df,extra):
    a=2*np.pi*df.doy.values/366; c=[np.ones(len(df)),np.sin(a),np.cos(a),np.sin(2*a),np.cos(2*a)]
    c+=[df.H_pop.values/100,df.C_pop.values/100,(df.H_pop.values/100)**2,(df.C_pop.values/100)**2]
    for k in extra: c.append(df[k].values/1000)
    return np.column_stack(c)
def oos(extra,years):
    out=[]
    for Y in years:
        tr=X[X.year<Y]; te=X[X.year==Y]
        b=np.linalg.lstsq(design(tr,extra),tr.chg.values,rcond=None)[0]
        out.append(pd.Series(design(te,extra)@b,index=te.index))
    return pd.concat(out)
# note: burn_ws / burn_w exist only for 2021-2025 (out-of-sample L1), so weekly storage models are fit on 2021+ for the burn variants
X['year']=X.index.year
years=range(2023,2026)
tr_all=X[X.year>=2021]
res={}
for name,extra in (('M1 degree-days only',[]),('M3 + OBSERVED power burn (upper bound)',['burn_obs']),('M4 + predicted burn (weather + wind/solar)',['burn_ws']),('M5 + predicted burn (weather only)',['burn_w'])):
    out=[]
    for Y in years:
        tr=X[(X.year<Y)&(X.year>=2021)]; te=X[X.year==Y]
        b=np.linalg.lstsq(design(tr,extra),tr.chg.values,rcond=None)[0]
        out.append(pd.Series(design(te,extra)@b,index=te.index))
    p=pd.concat(out); res[name]=p
    e=X.chg.reindex(p.index)-p
    print(f"{name:45s} OOS RMSE {np.sqrt((e**2).mean()):5.1f} Bcf  (weeks {len(p)}) | MAE {e.abs().mean():5.1f}")
e0=(X.chg-X.norm).reindex(res['M1 degree-days only'].index); print(f"{'five-year norm only':45s} OOS RMSE {np.sqrt((e0**2).mean()):5.1f} Bcf")
json.dump({},open(os.path.join(HERE,'l3_done.json'),'w'))
# ---- L4: Thursday decision. gap = expected injection (better chain) minus expected injection (simple chain). Bigger injection than consensus -> bearish.
px=pd.read_parquet(os.path.join(HERE,'data','universe_all.parquet'))['NG=F'].dropna(); days=px.index
rows=[]
for F in res['M1 degree-days only'].index:
    thu=F+pd.Timedelta(days=6); i=days.searchsorted(thu)
    if i>=len(days) or days[i]!=thu or i+5>=len(days): continue
    rows.append({'F':F,'R_thu':np.log(px.iloc[i]/px.iloc[i-1]),'R_5':np.log(px.iloc[i+5]/px.iloc[i-1]),'chg':X.chg[F],'norm':X.norm[F]})
R=pd.DataFrame(rows).set_index('F')
for name,p in res.items(): R[name]=p.reindex(R.index)
R['gap_M4']=R['M4 + predicted burn (weather + wind/solar)']-R['M1 degree-days only']
R['gap_M5']=R['M5 + predicted burn (weather only)']-R['M1 degree-days only']
R['gap_M3']=R['M3 + OBSERVED power burn (upper bound)']-R['M1 degree-days only']
R['surprise_vs_M1']=R.chg-R['M1 degree-days only']
print('weeks',len(R))
for g in ('gap_M3','gap_M4','gap_M5'):
    print(g,'Spearman with Thursday return %.3f | with 5-day %.3f | corr of gap with actual surprise vs simple chain %.3f'%(R[g].corr(R.R_thu,method='spearman'),R[g].corr(R.R_5,method='spearman'),R[g].corr(R.surprise_vs_M1)))
print('actual surprise vs simple chain: Spearman with Thursday return %.3f'%R.surprise_vs_M1.corr(R.R_thu,method='spearman'))
# rule: short gas when better chain expects bigger injection than simple chain (gap > +0.5 sd), long when smaller
for g in ('gap_M4','gap_M5'):
    sd=R[g].std(); pos=np.where(R[g]<-0.5*sd,1.0,np.where(R[g]>0.5*sd,-1.0,0.0)); act=pos!=0
    for col in ('R_thu','R_5'):
        pnl=pos*R[col]-0.001*np.abs(pos)
        print(f"rule {g} hold {col}: n {int(act.sum())}  net bp/trade {pnl[act].mean()*1e4:6.1f}  buy&hold same days {R[col][act].mean()*1e4:6.1f}")
R.to_csv(os.path.join(HERE,'data','l4_weekly.csv'))
