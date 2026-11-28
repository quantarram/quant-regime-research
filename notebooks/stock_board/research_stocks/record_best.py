"""Year-by-year record of the best 12-month rule found on 2008-16 (win = ends higher)."""
F=open('research_stocks/record_search_up.py').read(); exec(F.split('rows=[]; t0=time.time()')[0])
M,y,ret,sret,k,disc,hold,base_ok,usual=run(252)
rule=(('same-market record 2y',0.80),('win rate 3y',0.90),('win rate 1y',0.80))
mask=base_ok.copy(); score=0
for m,q in rule:
    A=M[m]; v=A[disc][base_ok[disc]&np.isfinite(A[disc])]; mask&=(A>=np.quantile(v,q))
    print(m,'cut-off',round(float(np.quantile(v,q)),3))
    score=score+np.nan_to_num(pd.DataFrame(np.where(base_ok,A,np.nan)).rank(axis=1,pct=True).values)
weeks=np.where(disc|hold)[0]; p=picks(mask,score,weeks,k)
R=pd.DataFrame([dict(date=W[w].date(),stock=T[j],sector=sec[j],ret=ret[w,j],spy=sret[w],win=y[w,j]) for w,j in p]); R['yr']=pd.to_datetime(R.date).dt.year
R.to_csv('research_stocks/record_best_picks.csv',index=False)
g=R.groupby('yr'); Y=pd.DataFrame({'picks':g.size(),'wins':g.win.sum().astype(int),'rate':g.win.mean().round(2),'mean_ret':g.ret.mean().round(3),'spy_ret':g.spy.mean().round(3)})
u={yr:y[(W.year==yr)&(disc|hold)][base_ok[(W.year==yr)&(disc|hold)]].mean() for yr in Y.index}; Y['usual']=[round(u[i],2) for i in Y.index]
print(Y.to_string()); print('last entry with a finished result',R.date.max())
# which stocks
h=R[R.yr>=2017]; print(h.stock.value_counts().head(10).to_dict())
