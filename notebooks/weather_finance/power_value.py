"""Dollar value of weather-forecast skill in power prices: price sensitivity (power_price_results.json) x the dashboard's measured temperature error by lead."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); DASH=os.path.abspath(os.path.join(HERE,'..','..','..','climate-energy-dashboard'))
sys.path.insert(0,DASH); os.chdir(DASH)
from weatherdash import ledger
s=ledger.stats(); bl=pd.DataFrame(s['by_lead'])
res=json.load(open(os.path.join(HERE,'power_price_results.json')))
F=1.8  # deg C error -> deg F
out={}
for hub,v in res.items():
    cdd,hdd=v['pct_per_10cdd']/10,v['pct_per_10hdd']/10   # percent price per degree-day
    rows=[]
    for _,r in bl.iterrows():
        ef,ec=r.t_mae_fc*F,r.t_mae_clim*F       # mean absolute temperature error, deg F (a degree-day error in the same direction)
        rows.append({'lead':int(r.lead),'err_fc_pct':ef*cdd,'err_clim_pct':ec*cdd,'err_fc_usd':ef*cdd/100*v['mean_price'],'err_clim_usd':ec*cdd/100*v['mean_price'],
                     'saving_usd':(ec-ef)*cdd/100*v['mean_price']})
    out[hub]=rows
json.dump(out,open(os.path.join(HERE,'power_value_results.json'),'w'),indent=1)
pj=pd.DataFrame(out['PJM West']).set_index('lead')
print('PJM West (summer sensitivity %.1f%% per cooling degree-day, mean price $%.1f/MWh):'%(res['PJM West']['pct_per_10cdd']/10,res['PJM West']['mean_price']))
print(pj.loc[[0,1,3,5,7,9],['err_fc_pct','err_clim_pct','err_fc_usd','err_clim_usd','saving_usd']].round(1).to_string())
for hub in out:
    d=pd.DataFrame(out[hub]).set_index('lead'); print(f"{hub:12s} lead1 weather-driven price error fc {d.loc[1,'err_fc_pct']:.0f}% vs clim {d.loc[1,'err_clim_pct']:.0f}% | saving ${d.loc[1,'saving_usd']:.1f}/MWh (lead 1), ${d.loc[5,'saving_usd']:.1f} (lead 5), ${d.loc[8,'saving_usd']:.1f} (lead 8)")
fig,ax=plt.subplots(1,2,figsize=(12,4.2),gridspec_kw={'width_ratios':[1,1.15]})
hubs=list(res); x=np.arange(len(hubs))
ax[0].bar(x-0.2,[res[h]['pct_per_10cdd'] for h in hubs],0.4,color='#c0392b',label='+10 cooling degree-days')
ax[0].bar(x+0.2,[res[h]['pct_per_10hdd'] for h in hubs],0.4,color='#2f5f8f',label='+10 heating degree-days')
ax[0].set_xticks(x); ax[0].set_xticklabels(hubs,rotation=30,ha='right',fontsize=8.5); ax[0].set_ylabel('Change in next-day peak power price (%)'); ax[0].legend(frameon=False,fontsize=8.5)
ax[0].set_title('What a hotter or colder day is worth',fontsize=11)
for sp in ('top','right'): ax[0].spines[sp].set_visible(False)
ax[1].plot(pj.index,pj.err_clim_usd,color='#8a94a3',lw=2,label='Using climatology')
ax[1].plot(pj.index,pj.err_fc_usd,color='#2e7d4f',lw=2.4,label='Using the IFS forecast')
ax[1].fill_between(pj.index,pj.err_fc_usd,pj.err_clim_usd,where=pj.err_clim_usd>pj.err_fc_usd,color='#2e7d4f',alpha=0.15)
ax[1].set_xlabel('Forecast lead (days)'); ax[1].set_ylabel('Weather-driven expected price error ($/MWh)')
ax[1].set_title('PJM West, summer: what forecast skill is worth per MWh',fontsize=11); ax[1].legend(frameon=False,fontsize=9)
for sp in ('top','right'): ax[1].spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE,'..','paper17_figs','fig_power.png'),dpi=170)
