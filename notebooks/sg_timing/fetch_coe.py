"""Fetch the LTA COE bidding results from data.gov.sg (open licence, no key): quota, bids, premium by category for every bidding exercise (Jan 2010 onward)."""
import io, json, os, time
import requests, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,'data','coe_results.csv')
ID='d_69b3380ad7e51aff3a7dcc84eba52b8a'
for attempt in range(5):
    r=requests.get(f'https://api-open.data.gov.sg/v1/public/api/datasets/{ID}/poll-download',timeout=60).json()
    if r['data'].get('status')=='DOWNLOAD_SUCCESS': break
    time.sleep(3)
url=r['data']['url']; raw=requests.get(url,timeout=120).content; open(OUT,'wb').write(raw)
meta=requests.get(f'https://api-production.data.gov.sg/v2/public/api/datasets/{ID}/metadata',timeout=60).json()['data']
df=pd.read_csv(io.BytesIO(raw)); print(df.shape); print(df.head(6).to_string()); print(df.tail(10).to_string()); print('last updated',meta.get('lastUpdatedAt'),'coverage',meta.get('coverageStart'),meta.get('coverageEnd'))
