"""What the exchange futures settle on, from public JEPX data: Tokyo base (all 30-minute prices of the month) and peak (08:00-20:00 JST, weekdays excluding Japanese holidays), and the peak-minus-base spread for each September; change from the 30 days to 20 August.
Entry prices (the futures prices on 20 August) are not in this data set."""
import io, glob, warnings, os
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, jpholiday
HERE=os.path.dirname(os.path.abspath(__file__)); fr=[]
for f in sorted(glob.glob(os.path.join(HERE,'data','spot_*.csv'))):
    d=pd.read_csv(io.StringIO(open(f,encoding='utf-8').read())); d=d.iloc[:,[0,1,8]]; d.columns=['date','slot','tokyo']; fr.append(d)
J=pd.concat(fr); J['date']=pd.to_datetime(J['date'],format='%Y/%m/%d'); J=J.drop_duplicates(['date','slot'])
base=J.groupby('date').tokyo.mean(); pk=J[(J.slot>=17)&(J.slot<=40)].groupby('date').tokyo.mean(); pk=pk[[not(x.weekday()>=5 or jpholiday.is_holiday(x.date())) for x in pk.index]]
for y in range(2016,2027):
    t=pd.Timestamp(f'{y}-08-20'); tb=base[t-pd.Timedelta(days=29):t].mean(); tp=pk[t-pd.Timedelta(days=29):t].mean(); sb=base[f'{y}-09-01':f'{y}-09-30'].mean(); sp=pk[f'{y}-09-01':f'{y}-09-30'].mean()
    print(y,'Sept base %.2f peak %.2f spread %+.2f | 30d to 20 Aug spread %+.2f | change %+.2f'%(sb,sp,sp-sb,tp-tb,(sp-sb)-(tp-tb)))
