import json, os, sys
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
SEAS={'DJF':1,'JFM':2,'FMA':3,'MAM':4,'AMJ':5,'MJJ':6,'JJA':7,'JAS':8,'ASO':9,'SON':10,'OND':11,'NDJ':12}
o=pd.read_csv(os.path.join(HERE,'data','climate','oni.txt'),sep=r'\s+')
o['center']=o.SEAS.map(SEAS)
o['year']=o.YR+np.where(o.SEAS=='NDJ',0,0)   # ONI season labelled by the year of its centre month (NDJ centred on December of YR)
o['avail']=[pd.Timestamp(int(y),int(m),1)+pd.DateOffset(months=2)+pd.Timedelta(days=9) for y,m in zip(o.year,o.center)]
o=o.sort_values('avail')
px=pd.read_parquet(os.path.join(HERE,'data','universe_all.parquet')); days=px['SPY'].dropna().index; px=px.reindex(days)
months=pd.date_range('1996-01-01','2025-12-01',freq='MS')
first=[days[days.searchsorted(m)] for m in months]
state=[]; oni=[]
for t in first:
    r=o[o.avail<=t].iloc[-1]; oni.append(r.ANOM); state.append('el' if r.ANOM>=0.5 else ('la' if r.ANOM<=-0.5 else 'neutral'))
S=pd.DataFrame({'t':first,'oni':oni,'state':state},index=months)
S['strong']=np.where(S.oni.abs()>=1.0,S.state,'none')
S['ep']=(S.state!=S.state.shift()).cumsum()
H=[63,126,189,252]
FORM=(pd.Timestamp('1996-01-01'),pd.Timestamp('2010-12-31')); TEST=(pd.Timestamp('2011-01-01'),pd.Timestamp('2025-12-31'))
inst=[c for c in px.columns if px[c].notna().sum()>=1500 and px[c].first_valid_index()<=pd.Timestamp('2010-12-31') or c in ('CANE','WEAT','CORN','SOYB','JO','UNG')]
inst=[c for c in inst if c!='SPY']
rows=[]
for tk in inst:
    p=px[tk].dropna()
    for h in H:
        fwd=[]
        for t in S.t:
            i=p.index.searchsorted(t)
            if i>=len(p) or p.index[i]>t+pd.Timedelta(days=5) or i+h>=len(p) or p.index[i]<S.index[0]: fwd.append(np.nan)
            else: fwd.append(np.log(p.iloc[i+h]/p.iloc[i]))
        S[f'f']=fwd
        for per,(a,b) in (('form',FORM),('test',TEST)):
            d=S[(S.index>=a)&(S.index<=b)].dropna(subset=['f'])
            if len(d)<60: continue
            base=d.f.mean()
            for st in ('el','la'):
                x=d[d.state==st]
                if len(x)<6: continue
                eps=x.groupby('ep').f.mean()-base   # episode-average excess
                rows.append({'inst':tk,'h':h,'state':st,'period':per,'n_months':len(x),'n_eps':int(len(eps)),'excess':float(x.f.mean()-base),'ep_excess_mean':float(eps.mean()),'ep_pos':float((eps>0).mean())})
R=pd.DataFrame(rows); R.to_csv(os.path.join(HERE,'climate_results.csv'),index=False)
F=R[R.period=='form'].set_index(['inst','h','state']); T=R[R.period=='test'].set_index(['inst','h','state'])
J=F.join(T,lsuffix='_f',rsuffix='_t',how='inner')
print('configs (state x instrument x horizon) with both periods:',len(J))
adm=J[(J.n_eps_f>=3)&(J.excess_f.abs()>=0.05)&(((J.excess_f>0)&(J.ep_pos_f>=2/3))|((J.excess_f<0)&(J.ep_pos_f<=1/3)))]
print('admitted in formation:',len(adm))
sgn=np.sign(adm.excess_f)
print('test: same sign share %.2f | mean test excess in formation direction %.3f (formation mean %.3f) | share of admitted whose test episodes support (>=half)  %.2f'%((np.sign(adm.excess_t)==sgn).mean(),(adm.excess_t*sgn).mean(),(adm.excess_f*sgn).mean(),((adm.ep_pos_t*(sgn>0)+(1-adm.ep_pos_t)*(sgn<0))>=0.5).mean()))
print('correlation of formation and test excess across ALL configs: %.3f'%J.excess_f.corr(J.excess_t))
print(adm.groupby(level='state').size().to_dict())
MECH=['SB=F','CANE','KC=F','CC=F','CT=F','OJ=F','JO','ZW=F','WEAT','ZS=F','SOYB','ZC=F','CORN','ZM=F','ZL=F','NG=F','UNG','HO=F','HG=F','LE=F','HE=F']
m=J.reset_index(); m=m[m.inst.isin(MECH)]
pd.set_option('display.width',250)
for st in ('el','la'):
    t=m[m.state==st].pivot_table(index='inst',columns='h',values=['excess_f','excess_t'])
    print('\n==',st.upper(),'excess forward return (pp): formation | test; horizons 63 126 189 252 trading days')
    for tk in MECH:
        if tk in t.index:
            print(f"{tk:7s} form "+' '.join(f"{t.loc[tk,('excess_f',h)]*100:6.1f}" if ('excess_f',h) in t.columns and not np.isnan(t.loc[tk,('excess_f',h)]) else '    nan' for h in H)+'  | test '+' '.join(f"{t.loc[tk,('excess_t',h)]*100:6.1f}" if ('excess_t',h) in t.columns and not np.isnan(t.loc[tk,('excess_t',h)]) else '    nan' for h in H))
print('\nepisodes:',S.groupby('state').ep.nunique().to_dict(), 'months by state',S.state.value_counts().to_dict())
