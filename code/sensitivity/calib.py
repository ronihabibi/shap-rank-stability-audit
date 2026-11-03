import pickle,numpy as np,pandas as pd
res=pickle.load(open('/tmp/en/res.pkl','rb'))
def lr_plus(d,flag):
    y=d.repro==1; tp=(flag&y).sum(); fp=(flag&~y).sum(); P=y.sum(); N=(~y).sum()
    rec=tp/P; fpr=fp/N
    return (rec/fpr if fpr>0 else np.nan), rec, fpr
def run(var,rule):
    d=res[var].copy()
    d['flag']=(d.modal>=0.8) if rule=='modal' else (d.srsi>=0.85)
    rows=[]
    for ds in d.ds.unique():
        tr=d[d.ds!=ds]; te=d[d.ds==ds]
        lr,rec,fpr=lr_plus(tr,tr.flag)
        prior_pool=tr.repro.mean(); prior_true=te.repro.mean()
        f=te.flag; nf=int(f.sum()); tp=int((f&(te.repro==1)).sum())
        def pred(pr):
            if not (0<pr<1): return 0.0 if pr<=0 else 1.0
            o=lr*pr/(1-pr); return o/(1+o)
        rows.append(dict(ds=ds,n=len(te),prior=prior_true,nflag=nf,true=tp,obs=(tp/nf if nf else np.nan),pred_pool=pred(prior_pool),pred_oracle=pred(prior_true),lr=lr))
    return pd.DataFrame(rows)
if __name__=='__main__':
  for var in ['boot','seed']:
    for rule in ['modal','srsi']:
        R=run(var,rule); S=R[R.nflag>0]
        w=S.nflag/S.nflag.sum()
        print(var,rule,'ds with flags',len(S),'flags',int(S.nflag.sum()),'true',int(S.true.sum()),
              '| expected true (pooled prior)',round((S.nflag*S.pred_pool).sum(),1),'(oracle prior)',round((S.nflag*S.pred_oracle).sum(),1),
              '| wMAE pooled',round((w*(S.obs-S.pred_pool).abs()).sum(),3),'oracle',round((w*(S.obs-S.pred_oracle).abs()).sum(),3),'face value',round((w*(S.obs-1).abs()).sum(),3))
        if var=='boot' and rule=='modal': print(R.round(3).to_string())
