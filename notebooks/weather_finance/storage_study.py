import json, os, sys, zipfile
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import edge_search as es
from weatherdash.calendar_utils import doy366

def load_storage():
    z=zipfile.ZipFile(os.path.join(HERE,'data','eia','NG.zip'))
    with z.open('NG.txt') as fh:
        for line in fh:
            d=json.loads(line)
            if d.get('series_id')=='NG.NW2_EPG0_SWO_R48_BCF.W':
                s=pd.Series({pd.Timestamp(a):float(b) for a,b in d['data']}).sort_index(); return s
S=load_storage(); S.index=pd.DatetimeIndex(S.index)
chg=S.diff().dropna()
# five-year norms by ISO week
def prior_norm(series):
    out={}
    for d in series.index:
        wk=d.isocalendar()[1]
        prev=[series[x] for x in series.index if x.year<d.year and x.year>=d.year-5 and x.isocalendar()[1]==wk]
        out[d]=np.mean(prev) if len(prev)>=3 else np.nan
    return pd.Series(out)
norm=prior_norm(chg); lvl_norm=prior_norm(S)
D=(chg-norm); L=(S-lvl_norm)/lvl_norm
# weekly degree-days Sat..Fri
T=es.metro_temps(); dd=es.region_dd(T,es.REGIONS['NAT'])
def weekly(q):
    x=dd[q]; out={}
    for F in chg.index:
        w=x.loc[F-pd.Timedelta(days=6):F]; out[F]=w.sum() if len(w)==7 else np.nan
    return pd.Series(out)
H,C=weekly('hdd'),weekly('cdd')
X=pd.DataFrame({'chg':chg,'norm':norm,'D':D,'L':L.reindex(chg.index),'H':H,'C':C}).dropna(subset=['H','C'])
X['doy']=doy366(X.index)
def design(df):
    h,c=df.H.values/100,df.C.values/100; a=2*np.pi*df.doy.values/366
    return np.column_stack([np.ones(len(df)),h,c,h**2,c**2,np.sin(a),np.cos(a),np.sin(2*a),np.cos(2*a)])
X['E']=np.nan
for Y in sorted(set(X.index.year)):
    tr=X[(X.index.year<Y)&X.norm.notna()]
    if len(tr)<150: continue
    b=np.linalg.lstsq(design(tr),tr.chg.values,rcond=None)[0]
    te=X.index.year==Y; X.loc[te,'E']=design(X[te])@b
X['u']=X.chg-X.E; X['W']=X.E-X.norm
px=pd.read_parquet(os.path.join(HERE,'data','universe_all.parquet'))['NG=F'].dropna()
days=px.index
rows=[]
for F in X.index:
    thu=F+pd.Timedelta(days=6)
    i=days.searchsorted(thu)
    if i>=len(days) or days[i]!=thu or i<1 or i+5>=len(days): continue
    rows.append({'F':F,'thu':thu,'R_thu':np.log(px.iloc[i]/px.iloc[i-1]),'R_5':np.log(px.iloc[i+5]/px.iloc[i-1]),'roll':bool((days[i:].to_period('M')[0]!=days[min(i+4,len(days)-1):].to_period('M')[0]) or False),
                 'Rf5':np.log(px.iloc[i+5]/px.iloc[i]),'Rf21':np.log(px.iloc[min(i+21,len(days)-1)]/px.iloc[i])})
R=pd.DataFrame(rows).set_index('F'); X=X.join(R,how='inner').dropna(subset=['E','D','R_thu'])
# month-end roll window flag for the Wednesday-Thursday day
lastdays=set()
for p,g in pd.Series(days,index=days).groupby(days.to_period('M')): lastdays.update(g.index[-4:])
X['rollday']=[ (t in lastdays) for t in X.thu]
X['year']=X.index.year
form=X[X.year<=2017]; test=X[(X.year>=2018)&(X.year<=2025)]
print('weeks formation',len(form),'test',len(test))
def sp(a,b,df): return df[a].corr(df[b],method='spearman')
for nm,df in (('formation',form),('test',test),('test no roll days',test[~test.rollday])):
    print(f"{nm:18s} n={len(df):3d}  Spearman(R_thu): D {sp('D','R_thu',df):+.2f}   u {sp('u','R_thu',df):+.2f}   W {sp('W','R_thu',df):+.2f}   | forward 5d vs W {sp('W','Rf5',df):+.2f}  21d vs L {sp('L','Rf21',df):+.2f}")
    print(f"{'':18s} weather explains share of weekly change variance: {1-df.u.var()/(df.chg-df.norm).var():.2f}")
# rule: long if W below formation 20th pct, short if above 80th, entered Wed close
lo,hi=form.W.quantile(0.2),form.W.quantile(0.8)
def rule(df,col):
    pos=np.where(df.W<=lo,1.0,np.where(df.W>=hi,-1.0,0.0)); pnl=pos*df[col]-0.001*np.abs(pos)
    act=pos!=0; return float(pnl[act].mean()*1e4) if act.any() else np.nan, int(act.sum()), float(df[col][act].mean()*1e4) if act.any() else np.nan
for nm,df in (('formation',form),('test',test),('test no roll',test[~test.rollday])):
    print(nm,'rule hold Thu (bp/trade, n, buy&hold bp same days):',[round(v,1) for v in rule(df,'R_thu')],'| hold 5d:',[round(v,1) for v in rule(df,'R_5')])
# state conditioning
for nm,df in (('formation',form),('test',test)):
    fq=form.L.quantile([0.2,0.8]).values
    for lab,m in (('deficit(top quintile of L<)',df.L<=fq[0]),('surplus',df.L>=fq[1])):
        print(nm,lab,'n',int(m.sum()),'mean fwd 5d bp %.0f 21d bp %.0f | all-weeks 5d %.0f 21d %.0f'%(df.Rf5[m].mean()*1e4,df.Rf21[m].mean()*1e4,df.Rf5.mean()*1e4,df.Rf21.mean()*1e4))
X.to_csv(os.path.join(HERE,'data','storage_weekly.csv'))
