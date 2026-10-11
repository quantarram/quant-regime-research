import os, sys, glob
import numpy as np, pandas as pd
sys.argv=[sys.argv[0]]+sys.argv[1:]
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import edge_search as es
FORM_END=pd.Timestamp('2025-05-31'); MIN_ACT=int(os.environ.get('MIN_ACT','30'))
h=pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE,'data','ecmwf_hist_s3.csv'))])
h['issue_date']=pd.to_datetime(h.issue_date)
for c in ('gen_fc_mwh_wind','gen_clim_mwh_wind','gen_fc_mwh_solar','gen_clim_mwh_solar'): h[c]=h[c].astype(float)
h['nl_fc']=h.demand_fc_mwh-h.gen_fc_mwh_wind.fillna(0)-h.gen_fc_mwh_solar.fillna(0)
h['nl_clim']=h.demand_clim_mwh-h.gen_clim_mwh_wind.fillna(0)-h.gen_clim_mwh_solar.fillna(0)
h['nl']=(h.nl_fc-h.nl_clim)/h.demand_clim_mwh*100; h['dem']=(h.demand_fc_mwh-h.demand_clim_mwh)/h.demand_clim_mwh*100
h['wind']=-(h.gen_fc_mwh_wind-h.gen_clim_mwh_wind)/h.demand_clim_mwh*100; h['sol']=-(h.gen_fc_mwh_solar-h.gen_clim_mwh_solar)/h.demand_clim_mwh*100
sigs={}
wins={'13':(1,3),'47':(4,7),'17':(1,7)}
def zs(x): 
    sd=x.expanding(min_periods=int(os.environ.get('MINP','40'))).std().shift(1); return x/sd
natw=h.groupby('region').demand_clim_mwh.mean(); 
for q in ('nl','dem','wind','sol'):
    for wn,(a,b) in wins.items():
        sel=h[(h.lead_days>=a)&(h.lead_days<=b)]
        pv=sel.groupby(['issue_date','region'])[q].mean().unstack()
        for r in pv.columns: sigs[f'ec|{r}|{q}|{wn}']=zs(pv[r])
        w=natw.reindex(pv.columns); sigs[f'ec|NAT|{q}|{wn}']=zs((pv*w).sum(1)/w.sum())
names=list(sigs)
px=pd.read_parquet(os.path.join(HERE,'data','universe_all.parquet')); days=px['SPY'].dropna().index; px=px.reindex(days).loc['2024-06-01':'2026-10-09']
dates=px.index; R=np.log(px/px.shift(1)).fillna(0.0).values; avail=(px.notna()&px.shift(1).notna()).astype(float).values; spy=R[:,list(px.columns).index('SPY')]
period=np.where(dates<=FORM_END,0,np.where(dates<=pd.Timestamp('2026-05-31'),1,2))
mask=period<2
sg={n:sigs[n].reindex(dates[mask]) for n in names}
names=[n for n in names if (sg[n].abs()>0.5).sum()>15]
S,ys=es.evaluate({n:sg[n] for n in names},names,R[mask],spy[mask],avail[mask],period[mask])
print('signals',len(names),'instruments',px.shape[1],'days',int(mask.sum()))
P=S[...,0]; T=S[...,1]
na,sa,ssa,sall,sps,nd,ssp,sssp,sr=P
with np.errstate(all='ignore'):
    mean=sa/na; mp,ms=sall/nd,ssp/nd; beta=(sps/nd-mp*ms)/(sssp/nd-ms**2); alpha=(mp-beta*ms)*252; bh=sr/na
ok=(na>=MIN_ACT)&(mean>0)&(alpha>0); lo=es.RULES.index('long_only'); ok[:,lo]&=(mean[:,lo]>bh[:,lo])
tn,ts,tss,tall,tps,tnd,tsp,tssp,tsr=T
with np.errstate(all='ignore'):
    tmean=ts/tn; tmp,tms=tall/tnd,tsp/tnd; tbeta=(tps/tnd-tmp*tms)/(tssp/tnd-tms**2); talpha=(tmp-tbeta*tms)*252; tbh=tsr/tn
valid=tn>=20
def summ(m,label):
    m=m&valid
    print(f"{label:28s} n={int(m.sum()):6d}  test bp/active mean {np.nanmean(tmean[m])*1e4:6.1f} median {np.nanmedian(tmean[m])*1e4:6.1f} share>0 {np.nanmean(tmean[m]>0):.2f}  beta-adj alpha ann {np.nanmean(talpha[m]):7.3f} share>0 {np.nanmean(talpha[m]>0):.2f}  form bp {np.nanmean(mean[m])*1e4:5.1f}")
summ(np.ones_like(ok),'ALL configs'); summ(ok,'ADMITTED on formation')
fam=np.array([n.split('|')[2] for n in names]); reg=np.array([n.split('|')[1] for n in names])
for q in ('nl','dem','wind','sol'): summ(ok&(fam==q)[:,None,None,None],f'  admitted, signal {q}')
for r in sorted(set(reg)): summ(ok&(reg==r)[:,None,None,None],f'  admitted, region {r}')
for i,rl in enumerate(es.RULES): 
    m=np.zeros_like(ok); m[:,i]=ok[:,i]; summ(m,f'  admitted, rule {rl}')
inst=list(px.columns)
idx=np.argwhere(ok&valid&(tmean>0)&(talpha>0)&(tn>=60))
rows=[]
for (i,ri,hi,j) in idx: rows.append((names[i],es.RULES[ri],es.HOLDS[hi],inst[j],mean[i,ri,hi,j]*1e4,tmean[i,ri,hi,j]*1e4,tbh[i,ri,hi,j]*1e4,int(tn[i,ri,hi,j]),talpha[i,ri,hi,j],tbeta[i,ri,hi,j]))
df=pd.DataFrame(rows,columns=['signal','rule','hold','inst','form_bp','test_bp','test_bh_bp','test_n','test_alpha','test_beta']).sort_values('test_alpha',ascending=False)
print('admitted and confirmed in test (n_active>=60):',len(df)); print(df.head(12).round(3).to_string(index=False))
df.to_csv(os.path.join(HERE,'ecmwf_decision_confirmed.csv'),index=False)
