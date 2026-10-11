import numpy as np, pandas as pd, os, csv
HERE=os.path.dirname(os.path.abspath(__file__))
ev=pd.read_csv(os.path.join(HERE,'policy_events.csv'))
px=pd.read_parquet(os.path.join(HERE,'..','data','universe_all.parquet'))
ma=pd.read_parquet(os.path.join(HERE,'..','..','multiasset_prices_live_history.parquet'))
daily={'SB=F':px['SB=F'].dropna(),'ZW=F':ma['ZW=F'].dropna()}
fred={k:pd.read_csv(os.path.join(HERE,f'fred_{k}.csv'),parse_dates=['observation_date']).set_index('observation_date').iloc[:,0].dropna() for k in ('PRICENPQUSDM','PPOILUSDM','PSUGAISAUSDM','PWHEAMTUSDM')}
MONTHLY={'ZR=F':'PRICENPQUSDM','ZL=F':'PPOILUSDM'}
def win_ret(p,i0,i1): return float(np.log(p.iloc[i1]/p.iloc[i0]))
def pct_rank(p,length,val,end_offset=0):
    r=np.log(p.shift(-length)/p).dropna()
    return float((r<val).mean()*100)
rows=[]
for _,e in ev.iterrows():
    ins=e.instruments
    if ins in daily:
        p=daily[ins]; t=pd.Timestamp(e.official_date[:10])
        i=p.index.searchsorted(t)
        if i<130 or i+125>=len(p): continue
        W={'pre 60d to -1':(i-61,i-1,60),'announcement -1..+1':(i-1,i+1,2),'post +2..+20':(i+1,i+20,19),'post +21..+120':(i+20,i+120,100)}
        out={'event':e.id,'instr':ins,'date':str(t.date()),'class':e['class'],'weather':e.weather_link}
        for k,(a,b,L) in W.items():
            v=win_ret(p,a,b); out[k]=round(v*100,1); out[k+' pct']=round(pct_rank(p,L,v),0)
        rows.append(out)
    elif ins in MONTHLY:
        s=fred[MONTHLY[ins]]; t=pd.Timestamp(e.official_date[:10]).to_period('M').to_timestamp()
        if t not in s.index: continue
        i=s.index.get_loc(t)
        if i<4 or i+6>=len(s): continue
        W={'pre 3m':(i-3,i-0,3),'event month':(i-1,i,1),'post +1..+3m':(i,i+3,3),'post +4..+6m':(i+3,i+6,3)}
        out={'event':e.id,'instr':ins+' (monthly Thai 5% / palm)','date':str(t.date()),'class':e['class'],'weather':e.weather_link}
        for k,(a,b,L) in W.items():
            v=win_ret(s,a,b); out[k]=round(v*100,1); r=np.log(s.shift(-L)/s).dropna(); out[k+' pct']=round(float((r<v).mean()*100),0)
        rows.append(out)
R=pd.DataFrame(rows); pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
d=R[R.instr.isin(['SB=F','ZW=F'])]; m=R[~R.instr.isin(['SB=F','ZW=F'])]
print('DAILY futures (log return %, and percentile among all same-length windows 2000-2026):'); print(d.drop(columns=['weather']).to_string(index=False))
print('\nMONTHLY (rice Thai 5% / palm oil):'); print(m.drop(columns=['weather']).to_string(index=False))
R.to_csv(os.path.join(HERE,'event_study_results.csv'),index=False)
