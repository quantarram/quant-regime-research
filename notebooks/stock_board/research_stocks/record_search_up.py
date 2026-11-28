"""Win = bought at the next close and held H days, the stock ends higher than it was bought (after 0.05% cost); loss otherwise.
Metrics are built only from each stock's own FINISHED wins and losses. At most one pick a week, one open pick per stock.
Rules chosen on signals whose results finished by 2016; checked on signals from 2017."""
import sys, warnings, itertools, time; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import pandas as pd, numpy as np
import sb_core as C
P=C.load_panel(); SP=pd.read_parquet('/private/tmp/stocks/stock_prices.parquet')
con=pd.read_csv('/private/tmp/stocks/constituents.csv'); con['Symbol']=con.Symbol.str.replace('.','-',regex=False)
sector=con.set_index('Symbol')['GICS Sector']; added=pd.to_datetime(con.set_index('Symbol')['Date added'],errors='coerce')
spy=P['SPY'].dropna().loc['2000-01-01':]
X=SP.reindex(spy.index).ffill(limit=5); X=X[[t for t in X.columns if t in sector.index and X[t].notna().sum()>=750]]
T=list(X.columns); sec=sector.reindex(T).values
iw=np.arange(4,len(spy),5); W=spy.index[iw]; nW,nS=len(W),len(T)
elig=np.ones((nW,nS),bool)
for j,t in enumerate(T):
    a=added.get(t)
    if pd.notna(a): elig[:,j]=W>=a
elig&=np.isfinite(X.values[iw])
up=(spy>spy.rolling(200).mean()).values[iw]
Qs=(0.80,0.90,0.95,0.98,0.99); WIN={'1y':52,'2y':104,'3y':156,'5y':260}
SPLIT=pd.Timestamp('2017-01-01')
def roll(a,N,m=None): return pd.DataFrame(a).rolling(N,min_periods=m or N).mean().values
def shift(a,k): out=np.full_like(a,np.nan); out[k:]=a[:-k]; return out
def run(H):
    ent=iw+1; ex=ent+H; ok=ex<len(spy)
    px=X.values; s=spy.values
    exc=np.full((nW,nS),np.nan); ret=np.full((nW,nS),np.nan); sret=np.full(nW,np.nan)
    ret[ok]=px[ex[ok]]/px[ent[ok]]-1; sret[ok]=s[ex[ok]]/s[ent[ok]]-1; exc=ret-0.0005
    y=np.where(np.isfinite(exc),(exc>0).astype(float),np.nan)
    k=int(np.ceil((H+1)/5))                      # weeks until a match result is known
    yk=shift(y,k); ek=shift(exc,k)
    M={}
    for n,N in WIN.items():
        f=roll(yk,N); M[f'win rate {n}']=f
        sm=np.full_like(f,np.nan)
        for c in pd.unique(sec):
            g=np.where(sec==c)[0]; x=f[:,g]; tot=np.nansum(x,1,keepdims=True); cnt=np.isfinite(x).sum(1,keepdims=True)
            sm[:,g]=(tot-np.nan_to_num(x))/(cnt-np.isfinite(x))
        M[f'edge over sector {n}']=f-sm
        M[f'average margin {n}']=roll(ek,N)
        h=np.full_like(f,np.nan)                # record in matches that started in the same market state as today
        for v in (True,False):
            ks=np.where(up==v)[0]; m=roll(y[ks],N); b=np.where(up==v)[0]; pos=np.searchsorted(ks,b-k,side='right')-1; g=pos>=0
            h[b[g]]=m[pos[g]]
        M[f'same-market record {n}']=h
    M['career win rate']=pd.DataFrame(yk).expanding(min_periods=104).mean().values
    for n,thr in (('after 60% form',0.6),('after 75% form',0.75)):
        f0=M['win rate 1y']; yy=np.where(f0>=thr,y,np.nan)        # how it did in matches that started while in strong form
        M[f'record {n}']=roll(shift(yy,k),260,20)
    disc=(W>=pd.Timestamp('2008-01-01'))&ok; disc&=np.array([ok[i] and spy.index[ex[i]]<SPLIT for i in range(nW)])
    hold=(W>=SPLIT)&ok
    base_ok=elig&np.isfinite(y)
    usual=(y[disc][base_ok[disc]].mean(), y[hold][base_ok[hold]].mean())
    return M,y,ret,sret,k,disc,hold,base_ok,usual
def picks(mask,score,weeks,k):
    out=[]; busy=np.zeros(nS,int)-1
    sc=np.where(mask,score,-np.inf)
    for w in weeks[mask[weeks].any(1)]:
        row=np.where(busy>w,-np.inf,sc[w]); j=int(np.argmax(row))
        if row[j]>-np.inf: out.append((w,j)); busy[j]=w+k
    return out
rows=[]; t0=time.time()
for H,lab in ((63,'3 months'),(126,'6 months'),(252,'12 months')):
    M,y,ret,sret,k,disc,hold,base_ok,usual=run(H)
    dw=np.where(disc)[0]; hw=np.where(hold)[0]; ny=(len(dw)/52.18,len(hw)/52.18)
    names=list(M)
    for side in ('win','loss'):
        out=y if side=='win' else 1-y
        R={}; conds={}
        for m in names:
            A=M[m]; v=A[disc][base_ok[disc]&np.isfinite(A[disc])]
            if len(v)<1000: continue
            Z=A if side=='win' else -A; zv=v if side=='win' else -v
            r=pd.DataFrame(np.where(base_ok,Z,np.nan)).rank(axis=1,pct=True).values; R[m]=np.nan_to_num(r)
            for q in Qs: conds[(m,q)]=(Z>=np.quantile(zv,q))&base_ok
        def evaluate(combo):
            mask=conds[combo[0]].copy()
            for c in combo[1:]: mask&=conds[c]
            sc=sum(R[c[0]] for c in combo)
            res={}
            for nm,wk,yrs in (('d',dw,ny[0]),('h',hw,ny[1])):
                p=picks(mask,sc,wk,k)
                if not p: return None
                a=np.array(p); v=out[a[:,0],a[:,1]]
                res[nm]=(len(p),v.mean(),len(p)/yrs,1000*np.sum(ret[a[:,0],a[:,1]]-0.0005),1000*np.sum(sret[a[:,0]]-0.0005))
            return res
        def add(combo,res):
            rows.append(dict(H=lab,side=side,rule=' + '.join(f'{m} top {round((1-q)*100)}%' for m,q in combo),n_metrics=len(combo),
                d_picks=res['d'][0],d_p=res['d'][1],d_per_year=res['d'][2],d_money=res['d'][3],d_spy=res['d'][4],
                h_picks=res['h'][0],h_p=res['h'][1],h_per_year=res['h'][2],h_money=res['h'][3],h_spy=res['h'][4],
                d_usual=usual[0] if side=='win' else 1-usual[0],h_usual=usual[1] if side=='win' else 1-usual[1]))
        pairs=[]
        for c1,c2 in itertools.combinations(conds,2):
            if c1[0]==c2[0]: continue
            mask=conds[c1]&conds[c2]
            if mask[dw].any(1).sum()<20*ny[0]*0.5: continue      # cannot reach about 20 picks a year
            res=evaluate((c1,c2))
            if res is None or res['d'][2]<20: continue
            add((c1,c2),res); pairs.append((res['d'][1],(c1,c2)))
        # third metric on the best two-metric rules
        pairs.sort(key=lambda t:-t[0]); seen=set()
        for _,combo in pairs[:150]:
            for c3 in conds:
                if c3[0] in (combo[0][0],combo[1][0]): continue
                key=frozenset(combo+(c3,))
                if key in seen: continue
                seen.add(key); mask=conds[combo[0]]&conds[combo[1]]&conds[c3]
                if mask[dw].any(1).sum()<20*ny[0]*0.5: continue
                res=evaluate(combo+(c3,))
                if res is None or res['d'][2]<20: continue
                add(combo+(c3,),res)
        pd.DataFrame(rows).to_csv('research_stocks/record_search_up.csv',index=False)
        print(lab,side,'usual',[round(u,3) for u in usual],'rules',len(rows),round(time.time()-t0),'s',flush=True)
D=pd.DataFrame(rows)
for (lab,side),g in D.groupby(['H','side'],sort=False):
    p=g[g.d_p>=0.8]
    print(f'\n### {lab} / {side}: rules {len(g)} | 80%+ in 2008-16: {len(p)} | of those 80%+ from 2017: {(p.h_p>=0.8).sum()} | their middle 2017+ rate {p.h_p.median() if len(p) else np.nan:.3f} | usual 2017+ {g.h_usual.iloc[0]:.3f}')
    b=g.sort_values(['d_p','d_picks'],ascending=False).head(3)
    print(b[['rule','d_picks','d_p','h_picks','h_p','h_per_year','h_money','h_spy']].round(3).to_string(index=False))
print('finished')
