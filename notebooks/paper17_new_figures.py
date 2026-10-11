"""
paper17_new_figures.py
Figures for the climate sections of Paper 17: the horizon ladder (weather against climate under one admission rule), the causal walk-forward, and the three-panel graphical abstract.
Every number is read from weather_finance/cpe_climate/ladder_*.parquet, wf_export.json and paper17_figs/numbers.json.

Run (from notebooks/):  ../.venv/bin/python paper17_new_figures.py
"""
import json, os, sys
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,'paper17_figs'); CD=os.path.join(HERE,'weather_finance','cpe_climate')
sys.path.insert(0,CD); import economic_prior as ep
GREEN,GREY,RED,BLUE,AMBER="#2e7d4f","#8a94a3","#c0392b","#2f5f8f","#c98a1b"
CLS={'climate':{'agriculture','energy','base_metal','broad_commodity','equity_country','em_fx'},'weather':{'energy','energy_single_name','sector_equity','broad_commodity','agriculture'}}
def ladder(tail):
    D=pd.read_parquet(os.path.join(CD,f'ladder_{tail}.parquet')); A=D[(D.p_form>=1.5*D.base_form)&(D.p_form>0)&(D.n_test>=100)].copy()
    A['lift']=A.p_test/A.base_test; A['ysub']=[ep.get_subclass(y) for y in A.Y]; A['mech']=[r.ysub in CLS[r.arm] for r in A.itertuples()]; return A
def spines(ax):
    for s in ('top','right'): ax.spines[s].set_visible(False)
def fig_ladder():
    A=ladder(10); fig,ax=plt.subplots(1,2,figsize=(13,4.8),gridspec_kw={'width_ratios':[1.25,1]})
    S=A[A.mech]; tfs=[1,5,10,21,63,126,252]; x=np.arange(len(tfs)); w=0.38
    med=lambda arm:[S[(S.arm==arm)&(S.tau_f==t)].lift.median() for t in tfs]
    ax[0].bar(x-w/2,med('weather'),w,color=BLUE,label='Weather predictors (windows 1-252 days)'); ax[0].bar(x+w/2,med('climate'),w,color=GREEN,label='Slow climate states')
    ax[0].axhline(1.0,color='k',lw=1,ls='--'); ax[0].set_xticks(x); ax[0].set_xticklabels([f'{t}' for t in tfs]); ax[0].set_xlabel('Forward horizon of the price move (trading days)')
    ax[0].set_ylabel('Median out-of-sample lift (1.0 = chance)'); ax[0].set_ylim(0.5,1.42); ax[0].legend(fontsize=8.5,frameon=False,loc='upper center',ncol=1,bbox_to_anchor=(0.5,1.0))
    ax[0].set_title('Targets with a registered economic channel, 10% tails',fontsize=11); spines(ax[0])
    cats=[('All targets',A),('Registered\nchannel',A[A.mech]),('No channel\n(control)',A[~A.mech])]; x2=np.arange(3)
    ax[1].bar(x2-w/2,[c[1][c[1].arm=='weather'].lift.mean() for c in cats],w,color=BLUE,label='Weather'); ax[1].bar(x2+w/2,[c[1][c[1].arm=='climate'].lift.mean() for c in cats],w,color=GREEN,label='Climate')
    for i,c in enumerate(cats):
        for j,arm in enumerate(('weather','climate')): ax[1].text(i+(j-0.5)*w*2*0.5+ (-0.0 if j==0 else 0.0),c[1][c[1].arm==arm].lift.mean()+0.02,f"{c[1][c[1].arm==arm].lift.mean():.2f}",ha='center',fontsize=9)
    ax[1].axhline(1.0,color='k',lw=1,ls='--'); ax[1].set_xticks(x2); ax[1].set_xticklabels([c[0] for c in cats]); ax[1].set_ylim(0.6,1.3); ax[1].set_ylabel('Mean out-of-sample lift'); ax[1].legend(fontsize=9,frameon=False,loc='upper left')
    ax[1].set_title('The climate lead sits where no mechanism exists',fontsize=11); spines(ax[1])
    fig.suptitle('Weather and climate predictors under one admission rule (formation through 2012, test 2013-2026)',fontsize=12.5); fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_ladder.png'),dpi=170,bbox_inches='tight'); plt.close(fig)
def fig_walkforward():
    W=json.load(open(os.path.join(CD,'wf_export.json'))); fig,ax=plt.subplots(1,2,figsize=(13,4.8),gridspec_kw={'width_ratios':[1.1,1]})
    c,m=W['climate'],W['market']
    groups=[('Climate\nextreme tail',c['extreme']),('Climate\nmoderate tail',c['moderate']),('Climate\nnot admitted',c['not_admitted']),('Market control\nextreme tail',m['extreme']),('Market control\nmoderate tail',m['moderate'])]
    x=np.arange(len(groups)); w=0.38
    ax[0].bar(x-w/2,[g[1]['hit'] for g in groups],w,color=GREEN,label='Hit rate'); ax[0].bar(x+w/2,[g[1]['base'] for g in groups],w,color=GREY,label='Base rate (same year)')
    for i,g in enumerate(groups): ax[0].text(i,max(g[1]['hit'],g[1]['base'])+0.015,f"{g[1]['hit_diff']*100:+.1f} pts\nn={g[1]['events']:,}",ha='center',fontsize=8.5)
    ax[0].set_xticks(x); ax[0].set_xticklabels([g[0] for g in groups],fontsize=8.5); ax[0].set_ylim(0,0.85); ax[0].set_ylabel('Share of entries whose forward move lands in the tail'); ax[0].legend(fontsize=9,frameon=False,loc='upper right')
    ax[0].set_title('Rows admitted each 1 January from resolved outcomes only',fontsize=11); spines(ax[0])
    cv=c['curves']; d=pd.to_datetime(cv['dates'])
    ax[1].plot(d,np.array(cv['strategy'])*100,color=AMBER,lw=1.8,label='Rule on climate extreme-tail rows'); ax[1].plot(d,np.array(cv['hold_same'])*100,color=BLUE,lw=1.3,label='Same targets, held long'); ax[1].plot(d,np.array(cv['spy'])*100,color=GREY,lw=1.1,label='SPY')
    ax[1].axhline(0,color='k',lw=0.6); ax[1].set_ylabel('Cumulative log return (%)'); ax[1].legend(fontsize=8.5,frameon=False,loc='upper left'); ax[1].set_title('The same rule as a paper portfolio, 10 bp costs',fontsize=11); spines(ax[1])
    fig.suptitle('A causal walk-forward: persistence in the extreme tail, no portfolio return',fontsize=12.5); fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_walkforward.png'),dpi=170,bbox_inches='tight'); plt.close(fig)
def graphical_abstract():
    num=json.load(open(os.path.join(OUT,'numbers.json'))); fig,ax=plt.subplots(1,3,figsize=(15.5,5.2),gridspec_kw={'width_ratios':[1,1,1.15]})
    days=[num['demand']['crossover_lead'],num['renewables']['crossover']['wind'],num['renewables']['crossover']['solar']]
    ax[0].bar(range(3),days,color=[GREEN,BLUE,AMBER],width=0.55)
    for i,dv in enumerate(days): ax[0].text(i,dv+0.15,f'{dv} days',ha='center',fontsize=13,fontweight='bold')
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(['Electricity\ndemand','Wind\ngeneration','Solar\ngeneration'],fontsize=10.5); ax[0].set_ylim(0,11); ax[0].set_ylabel('Days ahead the forecast still beats climatology',fontsize=10)
    ax[0].set_title('Free forecasts work for about\na week physically',fontsize=11.5); spines(ax[0])
    sc=pd.read_parquet(os.path.join(HERE,'weather_finance','scan_results.parquet')); adm=sc[(sc.eps_form>=5)&(sc.p_form>=0.30)]; vals=[adm.p_form.mean(),adm.p_test.mean(),adm.base_test.mean()]
    ax[1].bar(range(3),vals,color=[GREEN,RED,GREY],width=0.55)
    for i,v in enumerate(vals): ax[1].text(i,v+0.008,f'{v:.1%}',ha='center',fontsize=12.5,fontweight='bold')
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(['Formation\n1996-2012','Test\n2013-2025','Test base\n(chance)'],fontsize=10.5); ax[1].set_ylim(0,0.42); ax[1].set_ylabel('P(big price move | weather tail)',fontsize=10)
    ax[1].set_title(f'{len(adm)} weather-price pairs that looked\npredictive (of {len(sc):,}) fall to chance',fontsize=11.5); spines(ax[1])
    W=json.load(open(os.path.join(CD,'wf_export.json')))['climate']; g=[('Extreme\ntail',W['extreme']),('Moderate\ntail',W['moderate'])]; x=np.arange(2); w=0.36
    ax[2].bar(x-w/2,[a[1]['hit'] for a in g],w,color=GREEN,label='Hit rate'); ax[2].bar(x+w/2,[a[1]['base'] for a in g],w,color=GREY,label='Base rate')
    for i,a in enumerate(g): ax[2].text(i,max(a[1]['hit'],a[1]['base'])+0.015,f"{a[1]['hit_diff']*100:+.1f} pts",ha='center',fontsize=12,fontweight='bold')
    ax[2].set_xticks(x); ax[2].set_xticklabels([a[0] for a in g],fontsize=10.5); ax[2].set_ylim(0,0.85); ax[2].legend(fontsize=9,frameon=False,loc='upper right'); ax[2].set_ylabel('Share landing in the tail',fontsize=10)
    ax[2].set_title('Slow climate rows, admitted causally:\npersistence only in the rare extreme tail',fontsize=11.5); spines(ax[2])
    fig.suptitle('Weather is secondary: no edge in free forecasts; the rare, extreme, slow tail is the only candidate',fontsize=13.5,y=1.03); fig.tight_layout(); fig.savefig(os.path.join(HERE,'paper17_graphical_abstract.png'),dpi=170,bbox_inches='tight'); plt.close(fig)

def gated_gradient():
    out={}
    for name in ('climate','market'):
        E=pd.read_parquet(os.path.join(CD,f'walkforward_{name}.parquet')); E['b']=pd.cut(E.q_X,[0.49,0.75,0.85,0.92,0.97,1.0],labels=['0.5-0.75','0.8','0.9','0.95','0.99'])
        A=E[E.admitted]; rows=[]
        for b in ['0.5-0.75','0.8','0.9','0.95','0.99']:
            D=A[A.b==b]; N=E[(~E.admitted)&(E.b==b)]
            rows.append((b,len(D),100*(D.hit.mean()-D.base_year.mean()) if len(D) else np.nan,100*(N.hit.mean()-N.base_year.mean()) if len(N) else np.nan,int((E.b==b).sum())))
        out[name]=rows
    return out
def fig_gradient():
    G=gated_gradient(); fig,ax=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
    for a,(name,col,ttl) in zip(ax,(('climate',GREEN,'Slow climate rows'),('market',BLUE,"The program's own market predictors"))):
        r=G[name]; x=np.arange(len(r)); w=0.38
        a.bar(x-w/2,[v[2] if not np.isnan(v[2]) else 0 for v in r],w,color=col,label='Rows admitted causally'); a.bar(x+w/2,[v[3] if not np.isnan(v[3]) else 0 for v in r],w,color=GREY,label='Rows not admitted')
        for i,v in enumerate(r):
            lab=f"n={v[1]}" if v[1] else 'none admitted'
            a.text(i-w/2,(v[2] if not np.isnan(v[2]) else 0)+(1.0 if (v[2] if not np.isnan(v[2]) else 0)>=0 else -1.0),lab,ha='center',va='bottom' if (v[2] if not np.isnan(v[2]) else 0)>=0 else 'top',fontsize=8.5)
        a.axhline(0,color='k',lw=0.8); a.set_xticks(x); a.set_xticklabels(['0.50-0.75','0.80','0.90','0.95','0.99']); a.set_xlabel('Predictor tail (q_X): more extreme to the right'); a.set_title(ttl,fontsize=11.5); a.set_ylim(-9,36); spines(a)
    ax[0].set_ylabel('Hit rate minus base rate (points)'); ax[0].legend(fontsize=9,frameon=False,loc='upper left')
    fig.suptitle('Inside the CPE framework the gain rises with the rarity of the conditioning event (causal walk-forward, 2016-2026)',fontsize=12.5); fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_gradient.png'),dpi=170,bbox_inches='tight'); plt.close(fig)
def fig_free():
    W=pd.read_parquet(os.path.join(CD,'wf2s.parquet')).dropna(subset=['base_month']); W['dm']=100*(W.hit-W.base_month)
    P=pd.read_parquet(os.path.join(CD,'persist.parquet'))
    P['ef']=(P.pf_up-P.bmf_up)-(P.pf_dn-P.bmf_dn); P['et']=(P.pt_up-P.bmt_up)-(P.pt_dn-P.bmt_dn); P=P[P.ev_mode=='matched']
    cols={'weather':BLUE,'climate':GREEN,'market':AMBER}; tails=[0.2,0.1,0.05,0.02]
    fig,ax=plt.subplots(1,2,figsize=(13,4.8))
    for arm in ('weather','climate','market'):
        v=[W[(W.arm==arm)&(W['tail']==t)].dm.mean() if len(W[(W.arm==arm)&(W['tail']==t)])>=30 else np.nan for t in tails]; ax[0].plot(range(4),v,'o-',color=cols[arm],lw=2,label=arm.capitalize())
        sl=[]
        for t in tails:
            g=P[(P.arm==arm)&(P['tail']==t)]; sl.append(np.cov(g.ef,g.et)[0,1]/g.ef.var() if len(g)>30 else np.nan)
        ax[1].plot(range(4),sl,'o-',color=cols[arm],lw=2,label=arm.capitalize())
    for a in ax: a.set_xticks(range(4)); a.set_xticklabels(['20%','10%','5%','2%']); a.set_xlabel('Predictor tail: rarer to the right'); a.axhline(0,color='k',lw=0.8); spines(a)
    ax[0].set_ylabel('Hit rate minus same-month base rate (points)'); ax[0].set_title('One lift-based admission rule for all three arms\n(causal walk-forward, season-matched)',fontsize=11); ax[0].legend(fontsize=9,frameon=False)
    ax[1].set_ylabel('Slope of test edge on formation edge'); ax[1].set_title('Persistence of each configuration\'s directional edge\n(no admission filter, season-matched bases)',fontsize=11)
    fig.suptitle('Without the framework\'s own admission there is no rarity gradient',fontsize=12.5); fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_free.png'),dpi=170,bbox_inches='tight'); plt.close(fig)
def graphical_abstract2():
    num=json.load(open(os.path.join(OUT,'numbers.json'))); G=gated_gradient(); W=pd.read_parquet(os.path.join(CD,'wf2s.parquet')).dropna(subset=['base_month']); W['dm']=100*(W.hit-W.base_month)
    fig,ax=plt.subplots(1,3,figsize=(15.5,5.2),gridspec_kw={'width_ratios':[1,1.15,1.15]})
    sc=pd.read_parquet(os.path.join(HERE,'weather_finance','scan_results.parquet')); adm=sc[(sc.eps_form>=5)&(sc.p_form>=0.30)]; vals=[adm.p_form.mean(),adm.p_test.mean(),adm.base_test.mean()]
    ax[0].bar(range(3),vals,color=[GREEN,RED,GREY],width=0.55)
    for i,v in enumerate(vals): ax[0].text(i,v+0.008,f'{v:.1%}',ha='center',fontsize=12.5,fontweight='bold')
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(['Formation\n1996-2012','Test\n2013-2025','Test base\n(chance)'],fontsize=10.5); ax[0].set_ylim(0,0.42); ax[0].set_ylabel('P(big price move | weather tail)',fontsize=10)
    ax[0].set_title(f'Frequent events: {len(adm)} weather-price pairs\nthat looked predictive fall to chance',fontsize=11.5); spines(ax[0])
    x=np.arange(5); w=0.38
    for k,(name,col) in enumerate((('climate',GREEN),('market',BLUE))):
        r=G[name]; ax[1].plot(x,[v[2] for v in r],'o-',color=col,lw=2,label={'climate':'Slow climate rows','market':'Market predictors'}[name])
    ax[1].axhline(0,color='k',lw=0.8); ax[1].set_xticks(x); ax[1].set_xticklabels(['0.5-0.75','0.8','0.9','0.95','0.99'],fontsize=10); ax[1].set_xlabel('Predictor tail (q_X), rarer to the right'); ax[1].set_ylabel('Hit minus base (points)',fontsize=10)
    ax[1].legend(fontsize=9,frameon=False,loc='upper left'); ax[1].set_title("Inside the CPE framework's admission:\nthe gain rises with rarity",fontsize=11.5); spines(ax[1])
    for arm,col in (('weather',BLUE),('climate',GREEN),('market',AMBER)):
        v=[W[(W.arm==arm)&(W['tail']==t)].dm.mean() if len(W[(W.arm==arm)&(W['tail']==t)])>=30 else np.nan for t in (0.2,0.1,0.05,0.02)]; ax[2].plot(range(4),v,'o-',color=col,lw=2,label=arm.capitalize())
    ax[2].axhline(0,color='k',lw=0.8); ax[2].set_xticks(range(4)); ax[2].set_xticklabels(['20%','10%','5%','2%'],fontsize=10.5); ax[2].set_xlabel('Predictor tail, rarer to the right'); ax[2].set_ylabel('Hit minus same-month base (points)',fontsize=10)
    ax[2].legend(fontsize=9,frameon=False,loc='upper right'); ax[2].set_title('One gate-free admission rule:\nno gradient, weather flat at zero',fontsize=11.5); spines(ax[2])
    fig.suptitle('Frequent weather events are noise; the rare-event gain lives in the CPE framework\'s own admission and small samples',fontsize=13,y=1.03); fig.tight_layout(); fig.savefig(os.path.join(HERE,'paper17_graphical_abstract.png'),dpi=170,bbox_inches='tight'); plt.close(fig)
if __name__=='__main__':
    fig_ladder(); fig_walkforward(); fig_gradient(); fig_free(); graphical_abstract2(); print('wrote fig_ladder.png, fig_walkforward.png, fig_gradient.png, fig_free.png, paper17_graphical_abstract.png')
