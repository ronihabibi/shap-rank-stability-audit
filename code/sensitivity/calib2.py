import pickle,numpy as np,pandas as pd
res=pickle.load(open('results/sensitivity/res.pkl','rb'))
def odds_p(lr,pr):
    pr=min(max(pr,1e-6),1-1e-6); o=lr*pr/(1-pr); return o/(1+o)
out={}
for var in ['boot','seed']:
  for rule in ['modal','srsi']:
    d=res[var].copy(); d['flag']=(d.modal>=0.8) if rule=='modal' else (d.srsi>=0.85)
    d['sid']=d.ds+'|'+d.n.astype(str)+'|'+d.draw.astype(str)
    recs=[]
    for ds in d.ds.unique():
        tr=d[d.ds!=ds]; te=d[d.ds==ds]
        y=tr.repro==1; rec=(tr.flag&y).sum()/y.sum(); fpr=(tr.flag&~y).sum()/(~y).sum(); lr=rec/fpr
        prior_pool=y.mean()
        for sid,g in te.groupby('sid'):
            fl=g[g.flag]
            if len(fl)==0: continue
            f=g.flag.mean(); prior_hat=min(max((f-fpr)/(rec-fpr),0.0),1.0)
            for _,r in fl.iterrows():
                yy=float(r.repro==1)
                recs.append(dict(ds=ds,y=yy,face=1.0,pool=odds_p(lr,prior_pool),rg=odds_p(lr,max(prior_hat,0.005)) if prior_hat<1 else 1.0,oracle=odds_p(lr,te.repro.mean()) if te.repro.mean()>0 else 0.0,
                                 prior_hat=prior_hat))
    R=pd.DataFrame(recs)
    # macro over datasets to avoid large datasets dominating
    def brier(c): return ((R[c]-R.y)**2).mean()
    def brier_ds(c): return R.groupby('ds').apply(lambda g:((g[c]-g.y)**2).mean()).mean()
    out[(var,rule)]=R
    print(var,rule,'flagged domains',len(R),'true',int(R.y.sum()),'| Brier face %.3f pool %.3f RG %.3f oracle %.3f'%tuple(brier(c) for c in ['face','pool','rg','oracle']),
          '| macro-by-dataset face %.3f pool %.3f RG %.3f oracle %.3f'%tuple(brier_ds(c) for c in ['face','pool','rg','oracle']),
          '| mean pred pool %.3f RG %.3f obs %.3f'%(R.pool.mean(),R.rg.mean(),R.y.mean()))
pickle.dump(out,open('results/sensitivity/calib2.pkl','wb'))
