"""First look at the COE series: premium, quota and the bids/quota ratio for Category A and B; one row per bidding exercise."""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
d=pd.read_csv(os.path.join(HERE,'data','coe_results.csv'))
for c in ['quota','bids_success','bids_received','premium']: d[c]=pd.to_numeric(d[c],errors='coerce')
d['t']=pd.to_datetime(d.month+'-01')+pd.to_timedelta((d.bidding_no-1)*14,unit='D'); d['ratio']=d.bids_received/d.quota
print('rows',len(d),'nan',d[['quota','bids_received','premium']].isna().sum().to_dict()); print(d.groupby('vehicle_class').size().to_dict())
fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
for cat,col in (('Category A','tab:blue'),('Category B','tab:red')):
    s=d[d.vehicle_class==cat].set_index('t'); ax[0].plot(s.index,s.premium/1000,color=col,label=cat); ax[1].plot(s.index,s.quota,color=col); ax[2].plot(s.index,s.ratio,color=col)
ax[0].set_ylabel('premium (S$ thousand)'); ax[0].legend(); ax[1].set_ylabel('quota per round'); ax[2].set_ylabel('bids received / quota'); ax[2].axhline(1,color='k',lw=.6)
ax[0].set_title('COE bidding results by exercise, 2010-2026'); plt.tight_layout(); plt.savefig(os.path.join(HERE,'coe_eda.png'),dpi=130)
A=d[d.vehicle_class=='Category A'].set_index('t').sort_index(); B=d[d.vehicle_class=='Category B'].set_index('t').sort_index()
for nm,s in (('A',A),('B',B)):
    p=s.premium; ch=np.log(p).diff(); print(nm,'exercises',len(p),'| premium min %.0f (%s) max %.0f (%s) now %.0f'%(p.min(),p.idxmin().date(),p.max(),p.idxmax().date(),p.iloc[-1]),'| 1-round log change sd %.3f, 6-round (~3 months) sd %.3f'%(ch.std(),np.log(p).diff(6).std()))
    print('   share of rounds where premium rose:',round((ch>0).mean(),2),'| 6-round-ahead change > +10%:',round((np.log(p).diff(6).shift(-6)>np.log(1.1)).mean(),2),'< -10%:',round((np.log(p).diff(6).shift(-6)<np.log(0.9)).mean(),2))
print('\nlast 10 rounds, Category A:'); print(A.tail(10)[['quota','bids_received','ratio','premium']].round(2).to_string())
