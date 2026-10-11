"""Consumption weights for the 34 metros from EIA state gas consumption: heating weights from residential + commercial (Dec-Feb mean 2016-2025),
power-burn weights from gas delivered to electric power (Jun-Aug mean 2016-2025); a state's consumption is split among its metros by population."""
import json, os, zipfile
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
from gfs_archive import METROS
STATE={'New York':'NY','Los Angeles':'CA','Chicago':'IL','Dallas':'TX','Houston':'TX','Washington':'VA','Philadelphia':'PA','Miami':'FL','Atlanta':'GA','Boston':'MA',
 'Phoenix':'AZ','San Francisco':'CA','Detroit':'MI','Seattle':'WA','Minneapolis':'MN','San Diego':'CA','Tampa':'FL','Denver':'CO','Baltimore':'MD','St. Louis':'MO',
 'Orlando':'FL','Charlotte':'NC','San Antonio':'TX','Portland':'OR','Sacramento':'CA','Pittsburgh':'PA','Austin':'TX','Las Vegas':'NV','Cincinnati':'OH','Kansas City':'MO',
 'Columbus':'OH','Indianapolis':'IN','Cleveland':'OH','Nashville':'TN'}
z=zipfile.ZipFile(os.path.join(HERE,'data','eia','NG.zip')); ser={}
with z.open('NG.txt') as fh:
    for line in fh:
        d=json.loads(line)
        if d.get('f')=='M' and d.get('series_id','').startswith(('NG.N3010','NG.N3020','NG.N3045')) and d['series_id'].endswith('2.M'):
            ser[d['series_id']]=pd.Series({pd.Timestamp(a+'01'):float(b) for a,b in d['data'] if b not in (None,'')})
def state_total(prefix,months):
    out={}
    for st in set(STATE.values()):
        v=0.0
        for p in prefix:
            s=ser.get(f'NG.{p}{st}2.M')
            if s is not None:
                s=s[(s.index.year>=2016)&(s.index.year<=2025)&np.isin(s.index.month,months)]; v+=float(s.mean()) if len(s) else 0.0
        out[st]=v
    return out
heat=state_total(['N3010','N3020'],[12,1,2]); power=state_total(['N3045'],[6,7,8])
pop={m[0]:m[4] for m in METROS}
def metro_w(stt):
    w={}
    for m in pop:
        st=STATE[m]; share=pop[m]/sum(pop[x] for x in pop if STATE[x]==st); w[m]=stt[st]*share
    tot=sum(w.values()); return {k:v/tot for k,v in w.items()}
W={'heat':metro_w(heat),'power':metro_w(power)}
json.dump(W,open(os.path.join(HERE,'data','gas_weights.json'),'w'),indent=1)
top=lambda d:sorted(d.items(),key=lambda x:-x[1])[:8]
print('heat weights top',[(k,round(v,3)) for k,v in top(W['heat'])]); print('power weights top',[(k,round(v,3)) for k,v in top(W['power'])])
print('vs population top',[(k,round(pop[k]/sum(pop.values()),3)) for k in sorted(pop,key=lambda x:-pop[x])[:6]])
