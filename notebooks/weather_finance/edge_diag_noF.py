import sys, pickle, json, collections
import numpy as np, pandas as pd
sys.argv=['x']
import edge_search as es
U='L'
sigs_all=pickle.load(open(es.SIGCACHE,'rb')); px=pd.read_parquet(es.PRICES); days=px['SPY'].dropna().index; px=px.reindex(days)
inst=[c for c in px.columns if px[c].first_valid_index() is not None and px[c].first_valid_index()<=pd.Timestamp('2002-12-31') and not c.endswith('=F') or c=='SPY']
px=px.loc['1998-01-01':'2025-12-31',inst]; dates=px.index
R=np.log(px/px.shift(1)).fillna(0.0).values; avail=(px.notna()&px.shift(1).notna()).astype(float).values; spy=R[:,inst.index('SPY')]
names=[n for n in sigs_all if n.split('|')[0] in ('obs','cal')]
sigs={n:sigs_all[n].reindex(dates) for n in names}; names=[n for n in names if (sigs[n].abs()>0.5).sum()>30]
fam=np.array([n.split('|')[0] for n in names])
def run(cost_mult):
    es.COST=0.0005*cost_mult
    S,ys=es.evaluate(sigs,names,R,spy,avail,dates.year.values)
    res={}
    for label,mask in (('weather',fam!='cal'),('calendar',fam=='cal')):
        ch_all=collections.Counter(); sel={}
        allp=[];spys=[];dts=[]
        for Y in range(2008,2026):
            ok,score=es.admit(S,ys,Y,3,None); ok=ok&mask[:,None,None,None]
            sc=np.where(ok,score,-np.inf); flat=np.argsort(sc.ravel())[::-1][:50]
            idx=np.array(np.unravel_index(flat,sc.shape)).T; chosen=[tuple(int(v) for v in r) for r in idx]
            p=es.meta_pnl(sigs,names,chosen,R,spy,avail,dates,Y,inst); allp.append(p); mY=dates.year==Y; spys.append(spy[mY]); dts.append(dates[mY])
            for (i,ri,hi,j) in chosen: ch_all[(names[i],es.RULES[ri],es.HOLDS[hi],inst[j])]+=1
            sel[Y]=chosen
        p=np.concatenate(allp); s=np.concatenate(spys); d=pd.DatetimeIndex(np.concatenate([np.asarray(x) for x in dts]))
        res[label]=(es.summarise(p,s,d),ch_all,sel,p,d)
    return res
print('instruments',len(inst)); print(inst)
for cm in (1,):
    r=run(cm)
    for k in ('weather','calendar'):
        s=r[k][0]; print(f"cost x{cm} {k:9s} ann {s['ann_mean']*100:5.2f}% vol {s['ann_vol']*100:4.2f}% sharpe {s['sharpe']:.2f} alpha {s['alpha_md_ann']*100:5.2f}% beta {s['beta']:.2f}")
    if cm==1:
        cal=r['calendar']; ch=cal[1]
        print('calendar: most frequently chosen (signal,rule,hold,inst):'); [print('  ',k,v) for k,v in ch.most_common(14)]
        print('by instrument:',collections.Counter(k[3] for k in ch.elements()).most_common(10))
        print('by signal:',collections.Counter(k[0] for k in ch.elements()).most_common(10))
        print('by rule:',collections.Counter(k[1] for k in ch.elements()).most_common(), 'by hold:',collections.Counter(k[2] for k in ch.elements()).most_common())
        p,d=cal[3],cal[4]; ser=pd.Series(p,index=d)
        for a,b in (('2008','2012'),('2013','2017'),('2018','2022'),('2023','2025')):
            x=ser[a:b]; print(f"  calendar {a}-{b}: ann {x.mean()*252*100:5.2f}% sharpe {x.mean()/x.std()*np.sqrt(252):.2f}")
        # weather top picks
        wch=r['weather'][1]; print('weather: most chosen:'); [print('  ',k,v) for k,v in wch.most_common(8)]
        ser=pd.Series(r['weather'][3],index=r['weather'][4])
        for a,b in (('2008','2012'),('2013','2017'),('2018','2022'),('2023','2025')):
            x=ser[a:b]; print(f"  weather  {a}-{b}: ann {x.mean()*252*100:5.2f}% sharpe {x.mean()/x.std()*np.sqrt(252):.2f}")
