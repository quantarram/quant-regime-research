import os, sys
import numpy as np, pandas as pd
sys.argv=['x']
exec(open('climate_study.py').read().split("H=[63,126,189,252]")[0])   # reuse state construction (S, px, days)
def episode_table(tk,h):
    p=px[tk].dropna(); fwd=[]
    for t in S.t:
        i=p.index.searchsorted(t)
        fwd.append(np.nan if i>=len(p) or p.index[i]>t+pd.Timedelta(days=5) or i+h>=len(p) or p.index[i]<S.index[0] else np.log(p.iloc[i+h]/p.iloc[i]))
    S['f']=fwd; base=S.f.mean()
    out=[]
    for ep,g in S.groupby('ep'):
        st=g.state.iloc[0]
        if st=='neutral' or g.f.notna().sum()==0: continue
        out.append({'state':st,'start':g.index[0].strftime('%Y-%m'),'end':g.index[-1].strftime('%Y-%m'),'months':len(g),'peak_oni':g.oni.abs().max()*np.sign(g.oni.iloc[g.oni.abs().argmax()]),'fwd':g.f.mean()*100,'excess':(g.f.mean()-base)*100})
    return pd.DataFrame(out),base
for tk in ('KC=F','SB=F','CT=F'):
    for h in (126,252):
        E,b=episode_table(tk,h)
        print(f"\n=== {tk}  forward {h} trading days from the first trading day of each month; unconditional mean {b*100:.1f}%")
        print(E.round(1).to_string(index=False))
# a transparent rule: long coffee while the published ONI state is El Nino, short while La Nina, monthly rebalanced, return over the NEXT month
for tk in ('KC=F','SB=F','CT=F'):
    p=px[tk].dropna(); r=[]
    for k in range(len(S)-1):
        t0,t1=S.t.iloc[k],S.t.iloc[k+1]
        i0,i1=p.index.searchsorted(t0),p.index.searchsorted(t1)
        if i1>=len(p) or i0>=len(p) or p.index[i0]>t0+pd.Timedelta(days=5) or p.index[i0]<S.index[0]: r.append(np.nan); continue
        r.append(np.log(p.iloc[i1]/p.iloc[i0]))
    S['m']=r; pos=S.state.map({'el':1,'la':-1,'neutral':0}); pnl=(pos*S.m-0.0005*pos.diff().abs().fillna(0))
    for lab,(a,b_) in (('form 1996-2010',FORM),('test 2011-2025',TEST)):
        x=pnl[(S.index>=a)&(S.index<=b_)].dropna(); act=(pos[(S.index>=a)&(S.index<=b_)]!=0)
        print(f"{tk} monthly El Nino long / La Nina short  {lab}: ann return {x.mean()*12*100:5.1f}%  vol {x.std()*np.sqrt(12)*100:5.1f}%  Sharpe {x.mean()/x.std()*np.sqrt(12):5.2f}  active months {int(act.sum())}  El Nino-only ann {(pnl[(S.index>=a)&(S.index<=b_)&(pos==1)].mean()*12*100):5.1f}%  La Nina-only ann {(pnl[(S.index>=a)&(S.index<=b_)&(pos==-1)].mean()*12*100):5.1f}%")
