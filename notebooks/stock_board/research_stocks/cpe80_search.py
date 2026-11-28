"""The football recipe on stocks: qualify only where P(win | form edge > a, home PPG > b) >= 80% (football_joint_cpe.py gates).
Discovery 2008-14 sets the cut-offs and finds the conditions; 2015 on is the check. Windows and percentile grid follow football."""
import sys, warnings, itertools, time; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
import pandas as pd, numpy as np
import sb_core as C, sb_match_search as S
src=open('research_stocks/search_stocks.py').read()
exec(src.split("print('stocks'")[0].replace("import sys, time, itertools; sys.path.insert(0,'.')",''))
exec('def blocks'+src.split('def blocks')[1].split('_orig=')[0])
Q=np.array([0.50,0.60,0.70,0.75,0.80,0.90,0.95,0.98,0.99]); K=len(Q)
WINDOWS=(3,5,8,10,15,20,25,30,38); VENUES=('trend','vol','prev','none')
def hist(fe,hp,win,ta,tb):
    ok=np.isfinite(fe)&np.isfinite(hp)&np.isfinite(win)
    i=np.searchsorted(ta,fe[ok],side='left'); j=np.searchsorted(tb,hp[ok],side='left')
    k=i*(K+1)+j
    n=np.bincount(k,minlength=(K+1)**2).reshape(K+1,K+1); w=np.bincount(k,weights=win[ok],minlength=(K+1)**2).reshape(K+1,K+1)
    return n,w,ok.sum(),win[ok].sum()
def upper(a): return a[::-1,::-1].cumsum(0).cumsum(1)[::-1,::-1]
rows=[]; t0=time.time()
for H in (5,10,21,63):
    ends,ret,reg,sb=blocks(H); band=0.005*np.sqrt(H/21); B=len(ends); R=ret.values
    yrs=ends.year.values; D=np.where((yrs>=2008)&(yrs<=2014))[0]; Ho=np.where(yrs>=2015)[0]
    sec=cls.reindex(ret.columns).values; groups=[np.where(sec==c)[0] for c in pd.unique(sec)]
    diffs={o:S.opponent_diff(ret,o,cls,sb) for o in ('class','spy','zero')}
    for opp in ('class','spy','zero','pair'):
        if opp=='pair' and H<21: continue
        diff=diffs['class' if opp=='pair' else opp].values
        pts=np.where(np.isnan(diff),np.nan,np.where(diff>band,3.,np.where(diff>=-band,1.,0.)))
        nxt=np.vstack([diff[1:],np.full((1,diff.shape[1]),np.nan)]); nr=np.vstack([R[1:],np.full((1,R.shape[1]),np.nan)])
        if opp=='zero': win=np.where(np.isnan(nr),np.nan,(nr>0).astype(float))
        else: win=np.where(np.isnan(nxt),np.nan,(nxt>band).astype(float))
        pdf=pd.DataFrame(pts)
        for N in WINDOWS:
            ppg=pdf.rolling(N,min_periods=N).mean().values
            peer=np.full_like(ppg,np.nan)
            for g in groups:
                x=ppg[:,g]; s=np.nansum(x,1,keepdims=True); c=np.isfinite(x).sum(1,keepdims=True)
                peer[:,g]=(s-np.nan_to_num(x))/(c-np.isfinite(x))
            for venue in VENUES:
                V=reg[venue].shift(1).values.astype(float); U=reg[venue].values.astype(float)
                if venue=='none': hp=ppg
                else:
                    hp=np.full_like(pts,np.nan)
                    for val in (0.,1.):
                        ks=np.where(V==val)[0]
                        if len(ks)<N: continue
                        m=pd.DataFrame(pts[ks]).rolling(N,min_periods=N).mean().values
                        bs=np.where(U==val)[0]; pos=np.searchsorted(ks,bs,side='right')-1
                        good=pos>=0; hp[bs[good]]=m[pos[good]]
                if opp!='pair':
                    fe=ppg-peer
                    a=fe[D][np.isfinite(fe[D])&np.isfinite(win[D])]; b=hp[D][np.isfinite(hp[D])&np.isfinite(win[D])]
                    if len(a)<1000 or len(b)<1000: continue
                    ta=np.quantile(a,Q); tb=np.quantile(b,Q); out=[]
                    for idx in (D,Ho):
                        n,w,tn,tw=hist(fe[idx].ravel(),hp[idx].ravel(),win[idx].ravel(),ta,tb); out.append((upper(n),upper(w),tw/tn))
                else:
                    # a fixture is stock i (home) against a specific sector rival j: form edge = PPG_i - PPG_j, win = i beats j
                    def pairs(idx):
                        for g in groups:
                            n_=len(g); off=~np.eye(n_,dtype=bool)
                            p=ppg[np.ix_(idx,g)]; h=hp[np.ix_(idx,g)]; r=nr[np.ix_(idx,g)]
                            fe=(p[:,:,None]-p[:,None,:])[:,off]
                            d=(r[:,:,None]-r[:,None,:])[:,off]
                            hh=np.repeat(h[:,:,None],n_,2)[:,off]
                            yield fe.ravel(),hh.ravel(),np.where(np.isnan(d),np.nan,(d>band).astype(float)).ravel()
                    samp=[(f[::7],h[::7],w[::7]) for f,h,w in pairs(D)]
                    a=np.concatenate([f[np.isfinite(f)&np.isfinite(w)] for f,h,w in samp]); b=np.concatenate([h[np.isfinite(h)&np.isfinite(w)] for f,h,w in samp])
                    if len(a)<1000 or len(b)<1000: continue
                    ta=np.quantile(a,Q); tb=np.quantile(b,Q); out=[]
                    for idx in (D,Ho):
                        N_=0;W_=0;tn=0;tw=0
                        for f,h,w in pairs(idx):
                            n,ww,x,y=hist(f,h,w,ta,tb); N_=N_+n; W_=W_+ww; tn+=x; tw+=y
                        out.append((upper(N_),upper(W_),tw/tn))
                (dn,dw,db),(hn,hw,hb)=out
                for k,l in itertools.product(range(K),range(K)):
                    n1=dn[k+1,l+1]; n2=hn[k+1,l+1]
                    if n1<1: continue
                    rows.append((H,opp,venue,N,Q[k],Q[l],ta[k],tb[l],int(n1),dw[k+1,l+1]/n1,db,int(n2),hw[k+1,l+1]/n2 if n2 else np.nan,hb))
        print('H',H,opp,'done',round(time.time()-t0),'s',len(rows),flush=True)
        pd.DataFrame(rows,columns=['H','win_def','venue','window','q_edge','q_home','edge_cut','home_cut','d_n','d_p','d_usual','h_n','h_p','h_usual']).to_csv('research_stocks/cpe80_search.csv',index=False)
print('finished')
