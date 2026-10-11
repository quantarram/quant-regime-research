"""Batch-3 slow climate states through the CPE engine (same machinery as cpe_climate_run.py); writes cpe_climate/slow_formation_rows.parquet."""
import os, sys
import pandas as pd
sys.argv=['x','formation']
import cpe_climate_run as R
def load_panel():
    px=pd.read_parquet(os.path.join(R.NB,'multiasset_prices_live_history.parquet'))
    ex=pd.read_parquet(os.path.join(R.HERE,'data','universe_all.parquet'))
    for t in R.EXTRA:
        if t in ex.columns and t not in px.columns: px[t]=ex[t].reindex(px.index)
    C=pd.read_parquet(os.path.join(R.HERE,'data','slow','slow_daily.parquet')).reindex(px.index)
    return px,C
R.load_panel=load_panel
if __name__=='__main__':
    px,C=load_panel(); inc,fut,thr,predicted,climate=R.build(px,C,R.CUTOFF)
    print('Y',len(predicted),'X',len(climate))
    df=R.run_engine(inc,fut,thr,predicted,climate); print('\nrows kept',len(df)); df.to_parquet(os.path.join(R.OUTD,'slow_formation_rows.parquet'))
    if len(df): print(df.groupby('X').size().sort_values(ascending=False).to_string()); print(df.groupby('tau_future').size().to_dict()); print('q_X>=0.9 rows:',int((df.q_X>=0.9).sum()))
