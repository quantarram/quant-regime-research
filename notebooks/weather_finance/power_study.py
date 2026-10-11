import os, sys, json
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from xlsx_min import read_xlsx
import edge_search as es
HUBS={'PJM WH Real Time Peak':('PJM West',['Philadelphia','Washington','Chicago','Pittsburgh','Columbus','Baltimore','Cleveland']),
      'Nepool MH DA LMP Peak':('Mass Hub',['Boston','New York']),
      'Palo Verde Peak':('Palo Verde',['Phoenix','Las Vegas']),
      'SP15 EZ Gen DA LMP Peak':('SP15',['Los Angeles','San Diego']),
      'NP15 EZ Gen DA LMP Peak':('NP15',['San Francisco','Sacramento']),
      'Mid C Peak':('Mid-C',['Seattle','Portland']),
      'Indiana Hub RT Peak':('Indiana Hub',['Indianapolis','Cincinnati','Columbus','Chicago'])}
def num(v):
    try: return float(str(v).replace(',','').strip() or 0)
    except ValueError: return 0.0
rows=[]
for y in range(2017,2026):
    d=read_xlsx(os.path.join(HERE,'data','eia',f'ice_{y}.xlsx')); sh=list(d.values())[0]
    for r in sh[1:]:
        if r and r[0] in HUBS and r[2] is not None and r[6] is not None:
            rows.append({'hub':r[0],'date':pd.Timestamp('1899-12-30')+pd.Timedelta(days=num(r[2])),'price':num(r[6]),'vol':num(r[8])})
P=pd.DataFrame(rows); P=P[(P.price>0)]
P=P.groupby(['hub','date']).apply(lambda g:np.average(g.price,weights=g.vol if g.vol.sum()>0 else None)).rename('price').reset_index()
T=es.metro_temps(); gas=pd.read_parquet(os.path.join(HERE,'data','universe_all.parquet'))['NG=F'].dropna()
out={}
fc=pd.read_csv(os.path.join(HERE,'..','..','..','climate-energy-dashboard','data','ledger_temp_demand.csv'))
fc=fc[fc.status=='RESOLVED']
for hub,(label,cities) in HUBS.items():
    p=P[P.hub==hub].set_index('date').price.sort_index()
    w=es.weights(cities); X=T[cities].values
    hdd=pd.Series((np.maximum(65-X,0)*w).sum(1),index=T.index); cdd=pd.Series((np.maximum(X-65,0)*w).sum(1),index=T.index)
    df=pd.DataFrame({'lp':np.log(p),'hdd':hdd,'cdd':cdd,'lg':np.log(gas.reindex(p.index,method='ffill'))}).dropna()
    df=df[df.index.year<=2025]; df['m']=df.index.month; df['y']=df.index.year
    M=pd.get_dummies(df.m,prefix='m',drop_first=True).astype(float)
    def fit(cols,tr):
        A=np.column_stack([np.ones(len(tr)),tr[cols].values,M.loc[tr.index].values]); b=np.linalg.lstsq(A,tr.lp.values,rcond=None)[0]; return b
    def pred(cols,te,b): return np.column_stack([np.ones(len(te)),te[cols].values,M.loc[te.index].values])@b
    # full-sample sensitivity
    b=fit(['hdd','cdd','lg'],df)
    sens_h,sens_c=b[1]*10*100,b[2]*10*100   # percent per +10 degree-days
    # out-of-sample R2 with and without weather, test years 2020-2025
    res={}
    for name,cols in (('with_weather',['hdd','cdd','lg']),('no_weather',['lg'])):
        pr=[];ac=[]
        for Y in range(2020,2026):
            tr=df[df.y<Y]; te=df[df.y==Y]
            if len(tr)<200 or len(te)<20: continue
            bb=fit(cols,tr); pr.append(pred(cols,te,bb)); ac.append(te.lp.values)
        pr=np.concatenate(pr); ac=np.concatenate(ac)
        res[name]=float(1-((ac-pr)**2).sum()/((ac-ac.mean())**2).sum()); res[name+'_mae_pct']=float(np.abs(ac-pr).mean()*100)
    mean_price=float(np.exp(df.lp).mean())
    out[label]={'n':int(len(df)),'pct_per_10hdd':float(sens_h),'pct_per_10cdd':float(sens_c),'gas_elasticity':float(b[3]),'oos':res,'mean_price':mean_price}
    print(f"{label:12s} n={len(df):4d} mean ${mean_price:5.1f} | +10 HDD {sens_h:+5.1f}%  +10 CDD {sens_c:+5.1f}%  gas elast {b[3]:+.2f} | OOS R2 with weather {res['with_weather']:.2f} vs without {res['no_weather']:.2f}  MAE% {res['with_weather_mae_pct']:.1f} vs {res['no_weather_mae_pct']:.1f}")
json.dump(out,open(os.path.join(HERE,'power_price_results.json'),'w'),indent=1)
