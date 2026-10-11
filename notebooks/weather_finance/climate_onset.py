import os, sys
import numpy as np, pandas as pd
exec(open('climate_study.py').read().split("H=[63,126,189,252]")[0])
FORM=(pd.Timestamp('1996-01-01'),pd.Timestamp('2010-12-31')); TEST=(pd.Timestamp('2011-01-01'),pd.Timestamp('2025-12-31'))
# episode onsets: first month of each El Nino / La Nina run (published state)
onsets=S[(S.state!='neutral')&(S.state!=S.state.shift())]
print('onsets:',onsets.state.value_counts().to_dict())
MECH=['KC=F','CT=F','SB=F','CC=F','OJ=F','ZW=F','ZS=F','ZC=F','ZM=F','ZL=F','NG=F','HO=F','HG=F','LE=F','HE=F','ZO=F','KE=F','CL=F','GC=F']
rows=[]
for tk in [c for c in px.columns if px[c].notna().sum()>=1500 and px[c].first_valid_index()<=pd.Timestamp('2010-12-31') and c!='SPY']:
    p=px[tk].dropna()
    for h in (126,252):
        # unconditional mean forward return over all months (same instrument/horizon) per period
        allf=[]
        for t in S.t:
            i=p.index.searchsorted(t); allf.append(np.nan if i>=len(p) or p.index[i]>t+pd.Timedelta(days=5) or i+h>=len(p) or p.index[i]<S.index[0] else np.log(p.iloc[i+h]/p.iloc[i]))
        S['f']=allf
        for per,(a,b) in (('form',FORM),('test',TEST)):
            base=S.f[(S.index>=a)&(S.index<=b)].mean()
            for st in ('el','la'):
                o=onsets[(onsets.state==st)&(onsets.index>=a)&(onsets.index<=b)]
                v=S.f.reindex(o.index).dropna()
                if len(v)<2: continue
                rows.append({'inst':tk,'h':h,'state':st,'period':per,'n':len(v),'excess':float((v-base).mean()),'dir_hit':float(((v-base)>0).mean())})
R=pd.DataFrame(rows); R.to_csv('climate_onset_results.csv',index=False)
J=R[R.period=='form'].set_index(['inst','h','state']).join(R[R.period=='test'].set_index(['inst','h','state']),lsuffix='_f',rsuffix='_t',how='inner')
print('configs',len(J),'| corr formation vs test excess %.3f'%J.excess_f.corr(J.excess_t))
adm=J[(J.n_f>=3)&(J.excess_f.abs()>=0.08)]
sg=np.sign(adm.excess_f); print('admitted (|excess|>=8pp, >=3 onsets in formation):',len(adm),'| test same sign %.2f | mean test excess in formation direction %.3f (formation %.3f)'%((np.sign(adm.excess_t)==sg).mean(),(adm.excess_t*sg).mean(),(adm.excess_f*sg).mean()))
for h in (126,252):
    a=adm.xs(h,level='h'); sg=np.sign(a.excess_f); print(f' h={h}: n {len(a)} same sign {((np.sign(a.excess_t)==sg).mean()):.2f} test mean in direction {((a.excess_t*sg).mean()*100):.1f}pp')
pd.set_option('display.width',200)
for st in ('el','la'):
    print(f'\n== {st.upper()} onset: mean excess forward return (pp) and share of onsets above the unconditional mean; formation | test; h=126 252')
    for tk in MECH:
        out=[]
        for per in ('form','test'):
            for h in (126,252):
                r=R[(R.inst==tk)&(R.state==st)&(R.period==per)&(R.h==h)]
                out.append(f"{r.excess.iloc[0]*100:6.1f} ({r.dir_hit.iloc[0]*100:3.0f}%,{int(r.n.iloc[0])})" if len(r) else '      -        ')
        print(f"{tk:6s} form {out[0]} {out[1]} | test {out[2]} {out[3]}")
