"""Money check for cpe_oracle_forward.py in yen/kWh (log points overstate moves from near-zero prices).
P&L of one trade started at t: pos * ( (mean price t+1..t+f) - (mean price t-29..t) - drift ), pos = +1 if the oracle window is dull (<= q tail), -1 if bright (>= 1-q tail).
drift = mean of (mean price t+1..t+f) - (mean price t-29..t) over EARLIER start dates s of the same calendar month whose outcome window had ended before t (causal, expanding; at least 30 observations)."""
import os, runpy, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()): ns=runpy.run_path(os.path.join(HERE,'cpe_oracle_forward.py'))
PRICE,ERA,zdaily,idx,TEST,FS,QS=ns['PRICE'],ns['ERA'],ns['zdaily'],ns['idx'],ns['TEST'],ns['FS'],ns['QS']
print('YEN/kWh P&L, midday, perfect foresight, test start dates 2021-01..2026-08 (overlapping windows; n_eff = n/f)')
out=[]
for area in ['tokyo','kyushu','system']:
    p=PRICE['midday'][area]; zf=zdaily(ERA[area]); z=zf.reindex(idx)
    B=p.rolling(30,min_periods=24).mean()
    for f in FS:
        O=p.rolling(f,min_periods=int(0.8*f)).mean().shift(-f); dY=(O-B)
        X=z.rolling(f,min_periods=f).mean().shift(-f); zb=zf.rolling(f,min_periods=f).mean(); zb=zb[(zb.index.year>=1996)&(zb.index.year<=2015)].dropna()
        df=pd.DataFrame({'X':X,'dY':dY}).dropna(); df['m']=df.index.month
        # causal drift: for each start date t use start dates s with s+f < t and the same month
        drift=pd.Series(index=df.index,dtype=float)
        for m,gm in df.groupby('m'):
            ix=gm.index; vals=gm.dY.values
            for i,t in enumerate(ix):
                k=np.where(ix< t-pd.Timedelta(days=f+1))[0]; drift[t]=vals[k].mean() if len(k)>=30 else np.nan
        df['drift']=drift; df=df.dropna(); df=df[(df.index>=TEST[0])&(df.index<=TEST[1])]
        for q in (0.20,0.10,0.05):
            for xt in ('dull','bright'):
                th=zb.quantile(q) if xt=='dull' else zb.quantile(1-q); m=(df.X<=th) if xt=='dull' else (df.X>=th); s=df[m]; pos=1 if xt=='dull' else -1; pn=pos*(s.dY-s.drift)
                if len(s)==0: continue
                out.append(dict(area=area,f=f,q=q,xtail=xt,n=len(s),n_eff=round(len(s)/f,1),mean_yen=pn.mean(),median_yen=pn.median(),hit=(pn>0).mean(),sd=pn.std(),price_level=s.dY.abs().mean()))
O_=pd.DataFrame(out); pd.set_option('display.width',200)
print(O_[O_.q==0.10].round(2).to_string(index=False)); print('\nq=5%'); print(O_[O_.q==0.05].round(2).to_string(index=False)); O_.to_csv(os.path.join(HERE,'cpe_oracle_forward_yen.csv'),index=False)

# ---- stability by period (Kyushu and Tokyo, f = 7 and 14, q = 10%): earlier years, then the two halves of the test window
print('\nSTABILITY BY PERIOD (yen/kWh per trade; mean [hit]; n windows), q = 10%')
PER=[('2018-01..2020-12','2018-01-01','2020-12-31'),('2021-01..2023-06','2021-01-01','2023-06-30'),('2023-07..2026-08','2023-07-01','2026-08-31')]
for area in ['kyushu','tokyo']:
    p=PRICE['midday'][area]; zf=zdaily(ERA[area]); z=zf.reindex(idx); B=p.rolling(30,min_periods=24).mean()
    for f in (7,14):
        O=p.rolling(f,min_periods=int(0.8*f)).mean().shift(-f); dY=(O-B); X=z.rolling(f,min_periods=f).mean().shift(-f); zb=zf.rolling(f,min_periods=f).mean(); zb=zb[(zb.index.year>=1996)&(zb.index.year<=2015)].dropna()
        df=pd.DataFrame({'X':X,'dY':dY}).dropna(); df['m']=df.index.month; drift=pd.Series(index=df.index,dtype=float)
        for m,gm in df.groupby('m'):
            ix=gm.index; vals=gm.dY.values
            for i,t in enumerate(ix):
                k=np.where(ix< t-pd.Timedelta(days=f+1))[0]; drift[t]=vals[k].mean() if len(k)>=20 else np.nan
        df['drift']=drift; df=df.dropna()
        for xt in ('dull','bright'):
            th=zb.quantile(0.10) if xt=='dull' else zb.quantile(0.90); pos=1 if xt=='dull' else -1; res=[]
            for lab,a,b in PER:
                s=df[(df.index>=a)&(df.index<=b)]; s=s[(s.X<=th) if xt=='dull' else (s.X>=th)]; pn=pos*(s.dY-s.drift); res.append(f'{lab}: {pn.mean():+.2f} [{(pn>0).mean():.2f}] n={len(s)}' if len(s) else f'{lab}: n=0')
            print(area,f'f={f}',xt,' | '.join(res))
