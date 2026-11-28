"""Famous = the stock's own FINISHED earlier picks won more than 80% of the time (draws are not wins), known on the signal day."""
F=open('research_stocks/rare_picks_famous.py').read()
exec(F.split('def show(')[0])
exec('def show('+F.split('def show(')[1].split("show('all',q)")[0])
A['famous']=(A.pn>=1)&(A.pw>0.8*A.pn)
q=A[(A.date.dt.year>=2015)&A.stock_ret.notna()]
show('all rare picks',q); show('famous at the time',q[q.famous]); show('not famous (has earlier picks)',q[(q.pn>=1)&~q.famous]); show('first pick, no record',q[q.pn==0])
five=['NVDA','WDC','DVN','SKYD','LRCX']; show('famous without the five',q[q.famous&~q.ticker.isin(five)])
f=q[q.famous]
for y,v in f.groupby(f.date.dt.year): show(f'year {y}',v)
print('@@ earlier picks behind each famous pick', f.pn.value_counts().sort_index().to_dict())
# famous list as it stands now: every finished pick to date
g=A[A.result!='open'].groupby('ticker')
T_=pd.DataFrame({'picks':g.size(),'W':g.result.apply(lambda s:(s=='W').sum()),'D':g.result.apply(lambda s:(s=='D').sum()),'L':g.result.apply(lambda s:(s=='L').sum()),'last':g.date.max().dt.strftime('%Y-%m')})
now=T_[T_.W>0.8*T_.picks].sort_values(['picks','last'],ascending=[False,False])
now.to_csv('research_stocks/famous80_now.csv')
print('@@ famous now',len(now)); print('@@ by picks',now.picks.value_counts().sort_index().to_dict())
for t,r in now[now.picks>=2].iterrows(): print('@@T',t,r.picks,f'{r.W}-{r.D}-{r.L}',r['last'],cls[t],con.set_index('Symbol').loc[t,'Security'],sep='|')
print('@@ one-pick names',', '.join(now[now.picks==1].sort_values('last',ascending=False).index))
for name,z in (('behind: 1 earlier pick',f[f.pn==1]),('behind: 2 earlier picks',f[f.pn==2]),('behind: 3+ earlier picks',f[f.pn>=3]),('behind: 5+ earlier picks',f[f.pn>=5])):
    show(name,z)
z=f[f.pn>=3]; print('@@Z3', z.ticker.value_counts().to_dict()); show('3+ without the five',z[~z.ticker.isin(five)])
