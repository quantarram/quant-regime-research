"""First CPE scan of the Singapore COE premium (Category A and B), one row per bidding exercise (about every two weeks), January 2010 to October 2026.

CPE = P(Y in its tail | X in its tail); lift = CPE / base rate of the same Y event in the same window.
Registered before looking at any scan result (mechanisms, from how the auction works):
  quota   : quota rising over the past tau_p exercises -> more supply -> premium LOWER over the next tau_f exercises; quota cut -> premium HIGHER.
  demand  : bids received / quota (mean over the past tau_p exercises) high -> premium HIGHER; low -> LOWER.
  value   : premium above its own trailing 26-exercise (about one year) average -> reversion, premium LOWER; below -> HIGHER.
  momentum: premium change over the past tau_p exercises high -> continuation, premium HIGHER; low -> LOWER.
tau_p in 2, 6, 12 exercises; tau_f in 2, 6, 12 exercises (about 1, 3, 6 months). X tails q = 20% and 10% (each side), thresholds frozen on the formation window.
Y event = forward log change of the premium in its top 20% (for 'higher') or bottom 20% (for 'lower'), thresholds frozen on the formation window.
Formation: exercises up to 2017-12; test: 2018-01 onward (forward windows must end by the last exercise). Episodes: X-tail exercises more than tau_f exercises apart.
Admission on formation (adapted from the program's gates to about 200 formation exercises): lift >= 1.5, >= 20 exercises, >= 5 episodes. Scored on the test window.
Bids received is missing for 109 of the 1,985 rows, mostly 2024-2026, so the demand predictor loses those rows.
The current state of each predictor (latest exercise) is printed at the end."""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
d=pd.read_csv(os.path.join(HERE,'data','coe_results.csv'))
for c in ['quota','bids_success','bids_received','premium']: d[c]=pd.to_numeric(d[c],errors='coerce')
d['t']=pd.to_datetime(d['month']+'-01')+pd.to_timedelta((d['bidding_no']-1)*14,unit='D')
FORM_END=pd.Timestamp('2017-12-31'); TP=[2,6,12]; TF=[2,6,12]; QS=[0.20,0.10]
REG={'quota':('down_X_means_up_Y',),'demand':(),'value':(),'momentum':()}   # directions are encoded below
rows=[]; state={}
for cat in ['Category A','Category B']:
    s=d[d['vehicle_class']==cat].sort_values('t').reset_index(drop=True); s['ratio']=s['bids_received']/s['quota']; lp=np.log(s['premium']); lq=np.log(s['quota'])
    for tp in TP:
        X={'quota':lq-lq.shift(tp),'demand':s['ratio'].rolling(tp,min_periods=max(1,tp//2)).mean(),'value':lp-lp.rolling(26,min_periods=20).mean(),'momentum':lp-lp.shift(tp)}
        # sign: +1 means a HIGH X implies premium HIGHER; -1 means a high X implies premium LOWER
        SIGN={'quota':-1,'demand':+1,'value':-1,'momentum':+1}
        if tp==TP[0]: state[cat]={}
        for tf in TF:
            Y=lp.shift(-tf)-lp
            df=pd.DataFrame({'t':s['t'],'Y':Y,**{k:v for k,v in X.items()}}); fm=(df['t']<=FORM_END)
            y80,y20=df['Y'][fm].quantile(0.80),df['Y'][fm].quantile(0.20)
            for fam,xs in X.items():
                dd=df[['t','Y',fam]].dropna(); f=(dd['t']<=FORM_END); te=~f
                for q in QS:
                    hi=dd[fam][f].quantile(1-q); lo=dd[fam][f].quantile(q)
                    for side,cond in (('high',dd[fam]>=hi),('low',dd[fam]<=lo)):
                        implies_up=(SIGN[fam]>0)==(side=='high')
                        ex=(dd['Y']>=y80) if implies_up else (dd['Y']<=y20)
                        rec=dict(cat=cat,fam=fam,tp=tp,tf=tf,q=q,xside=side,y_implied='higher' if implies_up else 'lower')
                        for w,win in (('form',f),('test',te)):
                            m=(win&cond).values; n=int(m.sum()); rec['n_'+w]=n
                            if n==0: rec.update({'cpe_'+w:np.nan,'base_'+w:np.nan,'lift_'+w:np.nan,'ep_'+w:0}); continue
                            idxs=np.where(m)[0]; ep=1+int((np.diff(idxs)>tf).sum()); base=float(ex[win].mean()); cpe=float(ex[m].mean())
                            rec.update({'cpe_'+w:cpe,'base_'+w:base,'lift_'+w:cpe/base if base>0 else np.nan,'ep_'+w:ep})
                        rows.append(rec)
        for fam,xs in X.items():
            v=xs.iloc[-1]; xv=xs[s['t']<=FORM_END]; state[cat][(fam,tp)]=(float(v) if pd.notna(v) else np.nan, float((xv<v).mean()*100) if pd.notna(v) else np.nan)
R=pd.DataFrame(rows); R['admit']=(R['lift_form']>=1.5)&(R['n_form']>=20)&(R['ep_form']>=5); R.to_csv(os.path.join(HERE,'coe_cpe_scan.csv'),index=False)
print('conditions',len(R),'| admitted on formation',int(R['admit'].sum()))
A=R[R['admit']]
print('ADMITTED rows scored on the test window: mean lift %.2f, share with lift >= 1.5: %.2f, share above base: %.2f, mean test CPE %.2f vs base %.2f'%(A['lift_test'].mean(),(A['lift_test']>=1.5).mean(),(A['cpe_test']>A['base_test']).mean(),A['cpe_test'].mean(),A['base_test'].mean()))
print('\nBy predictor family (admitted rows): count, mean test lift, share above base'); print(A.groupby('fam').agg(n=('lift_test','size'),mean_lift=('lift_test','mean'),above=('cpe_test',lambda x: float((x>A.loc[x.index,'base_test']).mean()))).round(2).to_string())
print('\nBy horizon tf (admitted):'); print(A.groupby('tf').agg(n=('lift_test','size'),mean_lift=('lift_test','mean')).round(2).to_string())
print('\nBy X rarity q (admitted):'); print(A.groupby('q').agg(n=('lift_test','size'),mean_lift=('lift_test','mean')).round(2).to_string())
print('\nTop admitted rows by test lift (n_test >= 20):')
print(A[A['n_test']>=20].sort_values('lift_test',ascending=False).head(14)[['cat','fam','tp','tf','q','xside','y_implied','n_form','ep_form','cpe_form','base_form','lift_form','n_test','ep_test','cpe_test','base_test','lift_test']].round(2).to_string(index=False))
print('\nCURRENT STATE (latest exercise): value, percentile of the formation distribution')
for cat in state:
    print(cat,{f'{k[0]}{k[1]}':(round(v[0],3),round(v[1])) for k,v in state[cat].items() if k[1] in (6,)})
