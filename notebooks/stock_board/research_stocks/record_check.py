"""Check rules: year by year, against an ordinary eligible stock bought in the very same weeks."""
F=open('research_stocks/record_search_mkt.py').read(); exec(F.split('rows=[]; t0=time.time()')[0])
M,y,ret,sret,k,disc,hold,base_ok,usual=run(252)
RULES={'A  win rate 3y top 10% + market win record 2y top half':(('win rate 3y',0.90),('strong market: market win record 2y',0.5)),
       'B  A + same-market record 2y top 20%':(('same-market record 2y',0.80),('win rate 3y',0.90),('strong market: market win record 2y',0.5)),
       'C  best on early years only':(('edge over sector 2y',0.90),('weak market: market win record 5y',0.35),('edge over sector 3y',0.80))}
for name,rule in RULES.items():
    mask=base_ok.copy(); score=0; cuts=[]
    for m,q in rule:
        A=M[m]; v=A[disc][base_ok[disc]&np.isfinite(A[disc])]; c=np.quantile(v,q); mask&=(A>=c); cuts.append((m,round(float(c),3)))
        score=score+np.nan_to_num(pd.DataFrame(np.where(base_ok,A,np.nan)).rank(axis=1,pct=True).values)
    print('\n#',name,cuts)
    p=picks(mask,score,np.where(disc|hold)[0],k)
    R=pd.DataFrame([dict(date=W[w].date(),stock=T[j],win=y[w,j],ret=ret[w,j],spy=sret[w],same_week=y[w][base_ok[w]].mean()) for w,j in p]); R['yr']=pd.to_datetime(R.date).dt.year
    R.to_csv(f'research_stocks/record_rule_{name[0]}.csv',index=False)
    for lab,z in (('2008-16',R[R.yr<2017]),('2017 on',R[R.yr>=2017])):
        print(lab,'picks',len(z),'wins',int(z.win.sum()),'rate',round(z.win.mean(),3),'| ordinary stock in the SAME weeks',round(z.same_week.mean(),3),'| mean return',round(z.ret.mean(),3),'SPY',round(z.spy.mean(),3),'| stocks',z.stock.nunique())
    print(R.groupby('yr').agg(picks=('win','size'),wins=('win','sum'),rate=('win','mean'),ordinary=('same_week','mean')).round(2).to_string())
    # signals whose result is not finished yet
    last=np.where(mask.any(1))[0]; print('last week with any qualifying stock',W[last[-1]].date() if len(last) else None,'| data to',spy.index[-1].date())
    recent=[]; busy={}
    for w in range(np.where(hold)[0][-1]-0,nW):
        if W[w]<spy.index[-1]-pd.Timedelta(days=90): continue
        c=np.where(mask[w])[0]
        if len(c): recent.append((str(W[w].date()),len(c),[T[j] for j in c[np.argsort(-score[w][c])][:5]]))
    print('qualifying in the last 90 days:',recent[-6:])

# is rule A / B switched on now? (signals whose result is not finished yet: no finished-result filter here)
live=elig
for name,rule in RULES.items():
    mask=live.copy(); on=np.ones(nW,bool)
    for m,q in rule:
        A=M[m]; v=A[disc][base_ok[disc]&np.isfinite(A[disc])]; c=np.quantile(v,q); mask&=(A>=c)
        if 'market:' in m: on&=(A[:,0]>=c)
    rec=np.where(W>=W[-1]-pd.Timedelta(days=370))[0]
    print('NOW',name[0],'| last signal week',W[-1].date(),'| market switch on now',bool(on[-1]),'| weeks with switch on in last year',int(on[rec].sum()),'of',len(rec),
          '| stocks qualifying now',int(mask[-1].sum()),[T[j] for j in np.where(mask[-1])[0]][:12],'| last week anything qualified',(str(W[np.where(mask.any(1))[0][-1]].date()) if mask.any() else None))
