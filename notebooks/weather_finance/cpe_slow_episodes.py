"""Where do the batch-3 survivors' test hits come from? Lists test episodes (anchor date, outcome) for the main survivor bundles."""
import sys, os
import numpy as np, pandas as pd
sys.argv=['x','formation']
import cpe_slow_run as ER
R=ER.R; CUT=R.CUTOFF
import backtest_engine as be
px,C=R.load_panel(); inc_a,fut_a,thr_a,_,_=R.build(px,C,None); inc_f,fut_f,thr_f,_,_=R.build(px,C,CUT)
S=pd.read_parquet(os.path.join(R.OUTD,'slow_survivors.parquet'))
def episodes(r):
    x,tp,q,d=r.X,int(r.tau_past),float(r.q_X),r.direction
    s=inc_a[tp][x]; thr=thr_f[(tp,q)][x] if d=='bullish' else thr_f[(tp,round(1-q,10))][x]
    fire=(s>thr) if d=='bullish' else (s<thr)
    fd=fire[fire.fillna(False)&(fire.index>CUT)].index
    anc=[e[-1] for e in be._cluster_into_episodes(fd,tp)] if len(fd) else []
    tf,qy,y=int(r.tau_future),float(r.q_Y),r.Y
    ty=thr_f[(tf,qy)][y] if d=='bullish' else thr_f[(tf,round(1-qy,10))][y]
    fa=fut_a[tf][y]; out=[]
    for a in anc:
        v=fa.get(a,np.nan)
        if pd.isna(v): continue
        out.append((a.strftime('%Y-%m'),bool(v>ty if d=='bullish' else v<ty)))
    return out
for key in [('NINO3_mean','EWY','bearish'),('NINO3_mean','FXI','bearish'),('NINO12_mean','ZARUSD=X','bearish'),('NINO12_mean','CT=F','bearish'),('TNA_mean','KE=F','bearish'),('TNA_mean','ZM=F','bearish'),('PNA_mean',None,'bullish'),('NINO12_mean','ZARUSD=X','bullish')]:
    D=S[(S.X==key[0])&(S.direction==key[2])&((S.Y==key[1]) if key[1] else True)]
    if D.empty: continue
    r=D.sort_values('n_ep_t',ascending=False).iloc[0]
    print(f"{key[0]} -> {r.Y} {key[2]} tau_p {int(r.tau_past)} q {r.q_X} tau_f {int(r.tau_future)} | formation episodes {int(r.n_ep_f)} hit {r.hit_f:.2f} | test CPE {r.cpe_test:.2f} base {r.base_test:.2f}")
    print('   test episodes (anchor month, outcome):',[(a,'hit' if h else 'miss') for a,h in episodes(r)])
