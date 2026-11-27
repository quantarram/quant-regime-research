import pandas as pd, yfinance as yf, time, warnings; warnings.filterwarnings('ignore')
c=pd.read_csv('constituents.csv'); syms=[s.replace('.','-') for s in c.Symbol]
out=[]
for i in range(0,len(syms),100):
    ch=syms[i:i+100]
    d=yf.download(ch,start='2000-01-01',auto_adjust=True,progress=False,threads=True,group_by='column')
    px=d['Close'] if 'Close' in d else d
    out.append(px); print(i,px.shape,flush=True); time.sleep(2)
P=pd.concat(out,axis=1); P=P.loc[:,~P.columns.duplicated()]
P.to_parquet('stock_prices.parquet'); print(P.shape,P.index[0],P.index[-1], P.notna().sum().describe())
