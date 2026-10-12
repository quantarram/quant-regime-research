"""Does a dull (bright) day change the midday price on the days AFTER it? Same-day CPE (cpe_daily_dose.py) with the price outcome k working days later (k = 0 is the same day). Same frozen thresholds, season-matched base, test 2021-2026.
Also: how often the later day is itself dull (irradiance persistence), which is what would carry the effect forward."""
import os, runpy, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
ns=runpy.run_path(os.path.join(HERE,'cpe_daily_dose.py')); g,work,ERA,FORM,TEST,daily_z=ns['g'],ns['work'],ns['ERA'],ns['FORM'],ns['TEST'],ns['daily_z']
print('\n=========== LAGS ===========')
for area in ['tokyo','kyushu']:
    p=g[area].where(work); z=daily_z(ERA[area])[0].reindex(g.index); s=p.rolling(31,center=True,min_periods=1).sum()-p.fillna(0); c=p.rolling(31,center=True,min_periods=1).count()-p.notna().astype(int); base=(s/c).where(c>=8)
    r=((p/base-1)*100).where(base>=1.0); fm=(r.index>=FORM[0])&(r.index<=FORM[1]); r90=r[fm].quantile(.9)
    zfull=daily_z(ERA[area])[0]; zb=zfull[(zfull.index.year>=1996)&(zfull.index.year<=2015)]
    for q in (0.10,0.05):
        th=zb.quantile(q); out=[]
        for k in (0,1,2,3,5,7,14):
            yk=r.shift(-k); df=pd.DataFrame({'z':z,'y':yk}).dropna(); test=(df.index>=TEST[0])&(df.index<=TEST[1]); cond=(df.z<=th)&test; ex=df.y>=r90
            n=int(cond.sum()); cpe=ex[cond].mean(); bm=ex[test].groupby(df.index[test].month).mean(); b=bm.reindex(df.index[cond].month).mean()
            zk=z.shift(-k).reindex(df.index); pers=(zk[cond]<=th).mean(); out.append(f'k={k}: lift {cpe/b:.2f} (CPE {cpe:.2f}, base {b:.2f}, later day also dull {pers:.2f})')
        print(area,'dull q=%.2f test:'%q); [print('   ',o) for o in out]
