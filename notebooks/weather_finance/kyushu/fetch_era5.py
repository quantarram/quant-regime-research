"""Hourly ERA5 (Open-Meteo archive) shortwave radiation and 2 m temperature at five Kyushu points, 2016-03-25 to 2026-10-10, cached per point and year in data/era5/."""
import os, sys, time, json
import requests
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,'data','era5'); os.makedirs(OUT,exist_ok=True)
POINTS={'fukuoka':(33.59,130.40),'kumamoto':(32.80,130.71),'oita':(33.24,131.61),'miyazaki':(31.91,131.42),'kagoshima':(31.60,130.56)}
YEARS=list(range(2016,2027)); fails=[]
for name,(la,lo) in POINTS.items():
    for y in YEARS:
        f=os.path.join(OUT,f'{name}_{y}.json')
        if os.path.exists(f): continue
        s,e=(f'{y}-01-01' if y>2016 else '2016-03-25'),(f'{y}-12-31' if y<2026 else '2026-10-10')
        for attempt in range(4):
            r=requests.get('https://archive-api.open-meteo.com/v1/archive',params={'latitude':la,'longitude':lo,'start_date':s,'end_date':e,'hourly':'shortwave_radiation,temperature_2m','timezone':'Asia/Tokyo'},timeout=180)
            if r.status_code==200: open(f,'w').write(r.text); break
            if 'limit' in r.text.lower(): print('rate limit:',r.text[:100]); fails.append((name,y)); break
            time.sleep(5*(attempt+1))
        else: fails.append((name,y))
        time.sleep(0.6)
    print(name,'done',flush=True)
print('failures:',fails)
