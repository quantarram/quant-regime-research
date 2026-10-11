import json, os, sys
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import edge_search as es
from weatherdash.calendar_utils import doy366
d=pd.read_parquet(os.path.join(HERE,'data','eia930_daily.parquet'))
d['date']=pd.to_datetime(d['date']); d=d[(d.date>='2019-01-01')&(d.date<='2025-12-31')]
nd=d.date.nunique()
cov=d.groupby('ba').apply(lambda g:pd.Series({'days':g.date.nunique(),'gas_full':(g.gas_hours>=23).mean(),'dem_full':((g.hours>=23)&(g.hours<=25)).mean(),'gas_mean':g.gas.mean()/1000}))
good=cov[(cov.days>=0.9*nd)&(cov.gas_full>=0.9)&(cov.dem_full>=0.9)&(cov.gas_mean>=5)].index.tolist()
tot_gas=d.groupby('ba').gas.sum()/1000
print('BAs used',len(good),'covering %.0f%% of all reported gas generation'%(100*tot_gas.reindex(good).sum()/tot_gas.sum()))
g=d[d.ba.isin(good)].copy()
for c in ('gas','coal','nuclear','hydro','wind','solar','oil','demand'): g[c]=g[c].fillna(0)
# a day counts only if every selected BA reported
cnt=g.groupby('date').ba.nunique(); okdays=cnt[cnt==len(good)].index
nat=g[g.date.isin(okdays)].groupby('date')[['gas','coal','nuclear','hydro','wind','solar','oil','demand']].sum()/1000   # GWh
nat=nat[(nat.gas>0)&(nat.demand>0)]
allba=d.copy()
for c in ('wind','solar','nuclear','hydro'): allba[c]=allba[c].fillna(0)
tot=allba.groupby('date')[['wind','solar','nuclear','hydro']].sum()/1000
nat['wind']=tot.wind.reindex(nat.index); nat['solar']=tot.solar.reindex(nat.index); nat['nuclear']=tot.nuclear.reindex(nat.index); nat['hydro']=tot.hydro.reindex(nat.index)
nat=nat.dropna()
print('days',len(nat),'mean daily GWh: demand %.0f gas %.0f coal %.0f nuclear %.0f hydro %.0f wind %.0f solar %.0f'%tuple(nat[c].mean() for c in ['demand','gas','coal','nuclear','hydro','wind','solar']))
W=json.load(open(os.path.join(HERE,'data','gas_weights.json')))
T=es.metro_temps()
def dd(wd,q):
    w=np.array([wd[m] for m in es.NAMES]); X=T[es.NAMES].values
    return pd.Series(((np.maximum(65-X,0) if q=='hdd' else np.maximum(X-65,0))*w).sum(1),index=T.index)
pw=es.weights(es.NAMES); pop={m:pw[i] for i,m in enumerate(es.NAMES)}
nat['H']=dd(W['heat'],'hdd').reindex(nat.index); nat['C']=dd(W['power'],'cdd').reindex(nat.index)
nat['Hp']=dd(pop,'hdd').reindex(nat.index); nat['Cp']=dd(pop,'cdd').reindex(nat.index)
gas=pd.read_parquet(os.path.join(HERE,'data','universe_all.parquet'))['NG=F'].dropna()
nat['lg']=np.log(gas.reindex(nat.index,method='ffill'))
nat['nl']=nat.demand-nat.wind-nat.solar
nat['nuc30']=nat.nuclear.rolling(30,min_periods=10).mean().shift(1); nat['hyd30']=nat.hydro.rolling(30,min_periods=10).mean().shift(1)
nat['dow']=nat.index.dayofweek; nat['doy']=doy366(nat.index); nat['year']=nat.index.year; nat['t']=(nat.index-pd.Timestamp('2019-01-01')).days/365.25
nat=nat.dropna()
def design(df,cols):
    a=2*np.pi*df.doy.values/366; c=[np.ones(len(df)),np.sin(a),np.cos(a),np.sin(2*a),np.cos(2*a),df.t.values]
    c+=[(df.dow.values==k).astype(float) for k in range(6)]
    for k in cols: c+=[df[k].values/100 if k in('H','C','Hp','Cp') else df[k].values/1000]
    for k in cols:
        if k in('H','C'): c.append((df[k].values/100)**2)
    return np.column_stack(c)
MODELS={'weather only (degree-days, gas price)':['H','C','lg'],
        '+ wind and solar generation':['H','C','lg','wind','solar'],
        '+ nuclear and hydro availability':['H','C','lg','wind','solar','nuc30','hyd30'],
        'net load direct (demand - wind - solar)':['nl','H','C','lg','nuc30','hyd30']}
def oos(cols):
    out=[]
    for Y in range(2021,2026):
        tr=nat[nat.year<Y]; te=nat[nat.year==Y]
        b=np.linalg.lstsq(design(tr,cols),tr.gas.values,rcond=None)[0]
        out.append(pd.DataFrame({'y':te.gas.values,'p':design(te,cols)@b},index=te.index))
    return pd.concat(out)
# climatology: day-of-year mean of prior years, level-matched by nothing
clim=[]
for Y in range(2021,2026):
    tr=nat[nat.year<Y]; te=nat[nat.year==Y]; m=tr.groupby('doy').gas.mean().reindex(range(1,367)).interpolate(limit_direction='both').rolling(15,center=True,min_periods=1).mean()
    clim.append(pd.DataFrame({'y':te.gas.values,'p':m.loc[te.doy.values].values},index=te.index))
clim=pd.concat(clim)
rm=lambda r:np.sqrt(((r.y-r.p)**2).mean()); ma=lambda r:np.abs((r.y-r.p)/r.y).mean()*100
print('mean daily gas-fired generation %.0f GWh'%nat.gas.mean())
print(f"{'climatology (day of year)':45s} RMSE {rm(clim):6.1f} GWh  MAPE {ma(clim):5.1f}%  weekly-sum RMSE {np.sqrt(((clim.y-clim.p).resample('W-FRI').sum()**2).mean()):7.0f}")
res={}
for nme,cols in MODELS.items():
    r=oos(cols); res[nme]=r
    print(f"{nme:45s} RMSE {rm(r):6.1f} GWh  MAPE {ma(r):5.1f}%  weekly-sum RMSE {np.sqrt(((r.y-r.p).resample('W-FRI').sum()**2).mean()):7.0f}")
nat.to_csv(os.path.join(HERE,'data','national_power_daily.csv'))
pd.concat({k:v for k,v in res.items()},axis=1).to_pickle(os.path.join(HERE,'data','l1_oos.pkl'))
