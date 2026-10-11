"""EIA-930 six-month balance files -> daily demand and generation by fuel for every balancing authority. Streams one file at a time; keeps only daily aggregates.
  python eia930_ingest.py     -> data/eia930_daily.parquet (resumable per file)"""
import io, os, sys, time
import pandas as pd, requests
HERE=os.path.dirname(os.path.abspath(__file__))
BASE="https://www.eia.gov/electricity/gridmonitor/sixMonthFiles"
FUELS={"coal":["Coal"],"gas":["Natural Gas"],"nuclear":["Nuclear"],"oil":["All Petroleum Products"],"hydro":["Hydropower Excluding Pumped Storage","Hydropower and Pumped Storage"],
       "solar":["Solar without Integrated Battery Storage","Solar"],"wind":["Wind without Integrated Battery Storage","Wind"]}  # older files use the short names
OUT=os.path.join(HERE,'data','eia930_parts2'); os.makedirs(OUT,exist_ok=True)
files=[f"{y}_{h}" for y in range(2018,2026) for h in ("Jan_Jun","Jul_Dec")][1:]  # 2018_Jul_Dec .. 2025_Jul_Dec
def col(df,base,fuel=None):
    name=f"Net Generation (MW) from {fuel}" if fuel else base
    adj=name+" (Adjusted)" if fuel else None
    s=pd.to_numeric(df[name],errors='coerce') if name in df else pd.Series(float('nan'),index=df.index)
    if adj and adj in df: s=pd.to_numeric(df[adj],errors='coerce').where(lambda x:x.notna(),s)
    return s
for f in files:
    path=os.path.join(OUT,f+'.parquet')
    if os.path.exists(path): continue
    for a in range(5):
        r=requests.get(f"{BASE}/EIA930_BALANCE_{f}.csv",timeout=300)
        if r.status_code==200: break
        time.sleep(3)
    df=pd.read_csv(io.BytesIO(r.content),thousands=',',low_memory=False)
    d=pd.DataFrame({'ba':df['Balancing Authority'],'date':pd.to_datetime(df['Data Date'],format='%m/%d/%Y')})
    dem=pd.to_numeric(df['Demand (MW) (Adjusted)'],errors='coerce') if 'Demand (MW) (Adjusted)' in df else pd.Series(float('nan'),index=df.index)
    d['demand']=dem.where(dem.notna(),pd.to_numeric(df['Demand (MW)'],errors='coerce'))
    for k,vs in FUELS.items():
        d[k]=pd.concat([col(df,None,v) for v in vs],axis=1).bfill(axis=1).iloc[:,0]
    g=d.groupby(['ba','date'])
    agg=g[['demand']+list(FUELS)].sum(min_count=1)
    agg['hours']=g['demand'].count()
    agg['gas_hours']=g['gas'].count()
    agg.reset_index().to_parquet(path)
    print(f,len(agg),flush=True)
parts=[pd.read_parquet(os.path.join(OUT,f+'.parquet')) for f in files]
pd.concat(parts).to_parquet(os.path.join(HERE,'data','eia930_daily.parquet'))
print('done')
