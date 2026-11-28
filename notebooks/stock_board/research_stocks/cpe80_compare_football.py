import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
F='/Users/arrams/Documents/personal_work/quant/quant-regime-research/notebooks/football_betting/'
sys.path.insert(0,F)
from data_download import CORE_LEAGUES
m=pd.read_parquet(F+'data/matches.parquet'); m=m[m.league.isin(CORE_LEAGUES)].sort_values('date')
anyp,homep,rows={}, {}, []
for r in m.itertuples():
    h,a=r.home_team,r.away_team; x=anyp.get(h,[]); y=anyp.get(a,[]); z=homep.get(h,[])
    if len(x)>=30 and len(y)>=30 and len(z)>=30: rows.append((r.date,np.mean(x[-30:])-np.mean(y[-30:]),np.mean(z[-30:]),float(r.ftr=='H')))
    hp,ap=(3,0) if r.ftr=='H' else ((0,3) if r.ftr=='A' else (1,1))
    anyp.setdefault(h,[]).append(hp); anyp.setdefault(a,[]).append(ap); homep.setdefault(h,[]).append(hp)
T=pd.DataFrame(rows,columns=['date','fe','hp','win']); split=pd.Timestamp('2022-02-01'); d=T[T.date<split]; h=T[T.date>=split]
Q=[0.50,0.60,0.70,0.75,0.80,0.90,0.95,0.98,0.99]; fb=[]
for q in Q:
    a,b=d.fe.quantile(q),d.hp.quantile(q); s=h[(h.fe>a)&(h.hp>b)]; sd=d[(d.fe>a)&(d.hp>b)]; fb.append((q,len(sd),sd.win.mean(),len(s),s.win.mean()))
fb=pd.DataFrame(fb,columns=['q','dn','dp','hn','hp']); print('football usual later',round(h.win.mean(),3),'from',h.date.min().date()); print(fb.round(3).to_string(index=False))
R=pd.read_csv('cpe80_search.csv'); R=R[(R.q_edge==R.q_home)]
fig,ax=plt.subplots(figsize=(10.5,5.8)); fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb'); x=np.arange(len(Q))
def pooled(sel):
    g=sel.dropna(subset=['h_p']).assign(w=lambda t:t.h_p*t.h_n).groupby('q_edge')[['w','h_n']].sum(); return (g.w/g.h_n).reindex(Q)
def typical(win_def): return R[R.win_def==win_def].groupby('q_edge').h_p.median().reindex(Q)
def best(win_def): return R[(R.win_def==win_def)&(R.h_n>=20)].groupby('q_edge').h_p.max().reindex(Q)
ax.axhline(.8,color='#52514e',lw=1.2,ls='--'); ax.text(0,0.807,'80% line',fontsize=9,color='#52514e')
series=[('Football: home win (30-match form)',fb.set_index('q').hp.reindex(Q),'#2a78d6'),('Stocks: finishes the match up (usual ~59%)',typical('zero'),'#eb6834'),('Stocks: beats a sector rival (usual ~47%)',typical('pair'),'#1baf7a'),('Stocks: beats sector average (usual ~45%)',typical('class'),'#4a3aa7')]
for lab,s,c in series:
    ax.plot(x,s.values,color=c,lw=2,marker='o',ms=7,label=lab); ax.annotate(f'{100*s.values[-1]:.0f}%',(x[-1],s.values[-1]),xytext=(9,{'#1baf7a':6,'#4a3aa7':-6}.get(c,0)),textcoords='offset points',va='center',fontsize=9,color='#0b0b0b')
ax.set_xticks(x,[f'top {round((1-q)*100)}%' for q in Q]); ax.set_xlabel('How strict the cut-off is on both form edge and home PPG',fontsize=10)
ax.set_ylim(0.35,1.02); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0,decimals=0))
ax.set_title('Win probability in the later check period as the filter gets stricter',loc='left',fontsize=12)
ax.grid(axis='y',color='#e4e3df',lw=.8); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ('top','right','left')]
ax.legend(frameon=False,loc='upper left',fontsize=9,ncol=2,title='Stock lines: the middle value across all form windows and home definitions',title_fontsize=8.5,alignment='left')
plt.tight_layout(); plt.savefig(sys.argv[1],dpi=150)
for lab,s,c in series: print(lab,[None if pd.isna(v) else round(v,3) for v in s.values])
A=pd.read_csv('cpe80_search.csv'); z=A[A.h_n>=20].sort_values('h_p',ascending=False).head(8)
print('best later-period P anywhere (hindsight)'); print(z[['H','win_def','venue','window','q_edge','q_home','d_n','d_p','h_n','h_p','h_usual']].round(3).to_string(index=False))
z=A[(A.h_n>=20)&(A.h_p>=.8)]; print('conditions with later P>=80 anywhere:',len(z))
