"""Extreme-event predictor panel for the CPE engine (rare, slow, mechanism-backed states), daily, with publication lags:
  CBPDSI   Corn Belt drought: mean Palmer Drought Severity Index of IL, IN, IA, KS, MN, MO, NE, ND, OH, SD (NCEI nClimDiv; negative = drought)
  CBTMP    Corn Belt temperature anomaly (same states, deg F, vs each calendar month's 1895-2025 mean)
  HDDGAS   US heating degree-day anomaly, weighted by state residential+commercial gas consumption (vs 1991-2020 NCEI normals)
  CDDPOW   US cooling degree-day anomaly, weighted by state gas-for-power consumption
  IOD      Dipole Mode Index (HadISST, NOAA PSL)
  ACE      Atlantic accumulated cyclone energy per day (HURDAT2 best track; research only: final data lag months)
Forms: <name>_chg (the level; engine increment = change over tau) and <name>_mean (cumulative sum; increment = sum over tau days). NCEI monthly values are taken as public on the 12th of month m+1,
the IOD on the 15th."""
import json, os, re, zipfile
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,'data','extremes')
STATES={'IL':'011','IN':'012','IA':'013','KS':'014','MN':'021','MO':'023','NE':'025','ND':'032','OH':'033','SD':'039'}
CODE_TO_ST={'001':'AL','002':'AZ','003':'AR','004':'CA','005':'CO','006':'CT','007':'DE','008':'FL','009':'GA','010':'ID','011':'IL','012':'IN','013':'IA','014':'KS','015':'KY','016':'LA','017':'ME','018':'MD','019':'MA','020':'MI','021':'MN','022':'MS','023':'MO','024':'MT','025':'NE','026':'NV','027':'NH','028':'NJ','029':'NM','030':'NY','031':'NC','032':'ND','033':'OH','034':'OK','035':'OR','036':'PA','037':'RI','038':'SC','039':'SD','040':'TN','041':'TX','042':'UT','043':'VT','044':'VA','045':'WA','046':'WV','047':'WI','048':'WY'}
def read_ncei(fn):
    """-> DataFrame indexed by month-start Timestamp, columns state abbreviation."""
    rows={}
    for line in open(os.path.join(D,fn)):
        if len(line)<20: continue
        code=line[:3]
        if code not in CODE_TO_ST: continue
        year=int(line[6:10]); vals=line[10:].split()
        if len(vals)<12: continue
        for m,v in enumerate(vals[:12],1):
            x=float(v)
            if x<=-99: continue
            rows.setdefault(CODE_TO_ST[code],{})[pd.Timestamp(year,m,1)]=x
    return pd.DataFrame(rows).sort_index()
def state_weights():
    z=zipfile.ZipFile(os.path.join(HERE,'data','eia','NG.zip')); ser={}
    with z.open('NG.txt') as fh:
        for line in fh:
            d=json.loads(line)
            if d.get('f')=='M' and d.get('series_id','').startswith(('NG.N3010','NG.N3020','NG.N3045')) and d['series_id'].endswith('2.M'):
                ser[d['series_id']]=pd.Series({pd.Timestamp(a+'01'):float(b) for a,b in d['data'] if b not in (None,'')})
    def tot(prefix,months):
        out={}
        for st in set(CODE_TO_ST.values()):
            v=0.0
            for p in prefix:
                s=ser.get(f'NG.{p}{st}2.M')
                if s is not None:
                    s=s[(s.index.year>=2016)&(s.index.year<=2025)&np.isin(s.index.month,months)]; v+=float(s.mean()) if len(s) else 0.0
            out[st]=v
        t=sum(out.values()); return {k:v/t for k,v in out.items()}
    return tot(['N3010','N3020'],[12,1,2]), tot(['N3045'],[6,7,8])
def anomalies(df):
    """Anomaly of a monthly state panel vs each state's mean for that calendar month over the full record."""
    return df-df.groupby(df.index.month).transform('mean')
def monthly_series():
    pdsi=read_ncei('climdiv-pdsist.txt'); tmp=read_ncei('climdiv-tmpcst.txt'); hdd=read_ncei('climdiv-hddcst.txt'); cdd=read_ncei('climdiv-cddcst.txt')
    cb=list(STATES)
    out={'CBPDSI':pdsi[cb].mean(axis=1),'CBTMP':anomalies(tmp[cb]).mean(axis=1)}
    wh,wp=state_weights()
    def weighted(df,w):
        a=anomalies(df)   # anomaly vs each state's own calendar-month mean over the full record
        cols=[c for c in a.columns if c in w]; ww=pd.Series({c:w[c] for c in cols}); ww/=ww.sum()
        return (a[cols]*ww).sum(axis=1,min_count=1)
    out['HDDGAS']=weighted(hdd,wh); out['CDDPOW']=weighted(cdd,wp)
    out={k:v.dropna() for k,v in out.items()}
    return out
def iod_series():
    rows={}
    for line in open(os.path.join(D,'dmi.txt')):
        p=line.split()
        if len(p)==13 and re.fullmatch(r'\d{4}',p[0]):
            for m,v in enumerate(p[1:],1):
                x=float(v)
                if abs(x)<90: rows[pd.Timestamp(int(p[0]),m,1)]=x
    return pd.Series(rows).sort_index()
def ace_daily():
    """Daily Atlantic ACE: sum over 6-hourly best-track fixes at 00/06/12/18Z of (vmax kt)^2 / 10^4 for fixes with vmax >= 34 kt."""
    ace={}
    for line in open(os.path.join(D,'hurdat2.txt')):
        p=[x.strip() for x in line.split(',')]
        if len(p)>=7 and re.fullmatch(r'\d{8}',p[0]) and p[1] in ('0000','0600','1200','1800'):
            try: v=float(p[6])
            except ValueError: continue
            if v>=34: ace[pd.Timestamp(p[0])]=ace.get(pd.Timestamp(p[0]),0.0)+v*v/1e4
    return pd.Series(ace).sort_index()
def build(price_index):
    # Climate states are monthly; they are laid on Monday-Friday rows (business days) so that a horizon of tau is tau business days, and the
    # cumulative-sum form sums tau weekday readings. Weekend rows of the 7-day price panel stay NaN (same unit as the targets' own trading days).
    price_index = pd.bdate_range(price_index.min(), price_index.max())
    ms=monthly_series(); cols={}
    for k,s in ms.items():
        avail=pd.DatetimeIndex([d+pd.DateOffset(months=1)+pd.Timedelta(days=11) for d in s.index])
        sa=pd.Series(s.values,index=avail).sort_index()
        d=sa.reindex(sa.index.union(price_index)).ffill().reindex(price_index); d[d.index<sa.index.min()]=np.nan
        cols[k+'_chg']=d; cols[k+'_mean']=d.cumsum()
    iod=iod_series(); ia=pd.Series(iod.values,index=pd.DatetimeIndex([d+pd.DateOffset(months=1)+pd.Timedelta(days=14) for d in iod.index])).sort_index()
    d=ia.reindex(ia.index.union(price_index)).ffill().reindex(price_index); d[d.index<ia.index.min()]=np.nan
    cols['IOD_chg']=d; cols['IOD_mean']=d.cumsum()
    a=ace_daily().reindex(price_index.union(ace_daily().index)).fillna(0.0).reindex(price_index)
    # ACE daily values outside the price index (weekends) are folded into the next price bar by cumulative sum
    full=ace_daily(); cs=full.reindex(full.index.union(price_index)).fillna(0.0).cumsum().reindex(price_index); cs[price_index<full.index.min()]=0.0
    cols['ACE_mean']=cs
    return pd.DataFrame(cols), ms
if __name__=='__main__':
    px=pd.read_parquet(os.path.join(HERE,'..','multiasset_prices_live_history.parquet'))
    X,ms=build(px.index)
    for k,s in ms.items(): print(k,s.index.min().date(),s.index.max().date(),'last',round(float(s.iloc[-1]),2),'min',round(float(s.min()),2),'max',round(float(s.max()),2))
    print(X.tail(1).T.round(2)); X.to_parquet(os.path.join(D,'extremes_daily.parquet'))
    # sanity: known extreme drought years in the Corn Belt PDSI
    s=ms['CBPDSI']; print('CBPDSI lowest months:',[(d.strftime('%Y-%m'),round(v,1)) for d,v in s.nsmallest(6).items()])
